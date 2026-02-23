"""Centralized LLM request scheduler with queueing, rate limiting, and quota tracking."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import TypeVar

from sentrysloth.config import LLMConfig, QuotaExhaustedMode
from sentrysloth.models import LLMResponse
from sentrysloth.providers.base import (
    LLMProvider,
    LLMQuotaExceededError,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SchedulerStats:
    enqueued: int = 0
    dequeued: int = 0
    dropped: int = 0
    quota_short_circuited: int = 0
    queue_wait_seconds_total: float = 0.0


@dataclass
class _Request:
    prompt: str
    response_model: type
    model: str | None
    temperature: float | None
    max_output_tokens: int
    future: asyncio.Future
    enqueued_at: float


class LlmRequestScheduler(LLMProvider):
    """Queue-based request scheduler wrapping an LLM provider.

    Provides FIFO ordering, per-minute rate limiting, and quota error propagation.
    Satisfies the LLMProvider interface so it can be used as a drop-in replacement.
    """

    def __init__(self, provider: LLMProvider, config: LLMConfig) -> None:
        super().__init__(config)
        self._provider = provider
        self._config = config
        self._queue: asyncio.Queue[_Request | None] = asyncio.Queue(maxsize=config.queue_max_size)
        self._workers: list[asyncio.Task] = []
        self._stats = SchedulerStats()
        self.quota_error: LLMQuotaExceededError | None = None
        self._quota_event = asyncio.Event()
        self._rpm_interval = 60.0 / config.max_requests_per_minute
        self._last_request_time: float = 0.0
        self._rpm_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start worker tasks that process the request queue."""
        for _ in range(self._config.scheduler_workers):
            task = asyncio.create_task(self._worker_loop())
            self._workers.append(task)

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ) -> LLMResponse[T]:
        # Short-circuit immediately if quota is already exhausted in fail_fast mode
        if (
            self.quota_error is not None
            and self._config.quota_exhausted_mode == QuotaExhaustedMode.FAIL_FAST
        ):
            self._stats.quota_short_circuited += 1
            self._stats.dropped += 1
            raise self.quota_error

        loop = asyncio.get_running_loop()
        future: asyncio.Future[LLMResponse[T]] = loop.create_future()
        request = _Request(
            prompt=prompt,
            response_model=response_model,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            future=future,
            enqueued_at=time.monotonic(),
        )

        put_task: asyncio.Task[None] = asyncio.create_task(self._queue.put(request))
        quota_task: asyncio.Task[bool] = asyncio.create_task(self._quota_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {put_task, quota_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if (
                quota_task in done
                and self.quota_error is not None
                and self._config.quota_exhausted_mode == QuotaExhaustedMode.FAIL_FAST
            ):
                # Quota was exhausted while we were waiting for queue space.
                self._stats.quota_short_circuited += 1
                self._stats.dropped += 1
                put_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await put_task
                # The request may have been enqueued already (race). Cancel its Future so
                # worker/drain logic won't set an exception that nobody awaits.
                if future.done() and not future.cancelled():
                    _ = future.exception()
                else:
                    future.cancel()
                raise self.quota_error

            # Otherwise, enqueue must complete (backpressure).
            await put_task
            self._stats.enqueued += 1
        except asyncio.CancelledError:
            # If caller cancels while we're waiting to enqueue, cancel the underlying
            # Future so worker/drain logic won't set exceptions that nobody awaits.
            if future.done() and not future.cancelled():
                _ = future.exception()
            else:
                future.cancel()
            put_task.cancel()
            quota_task.cancel()
            raise
        finally:
            quota_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await quota_task

        try:
            return await future
        except asyncio.CancelledError:
            # If caller cancels (e.g. asyncio.gather on first exception), cancel the
            # underlying Future so we don't later set an exception that no one awaits.
            if future.done() and not future.cancelled():
                _ = future.exception()
            else:
                future.cancel()
            raise

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ) -> ToolCallResponse:
        """Proxy tool-use calls with RPM throttling (no queue).

        Tool calls are multi-turn dialogues with low concurrency,
        so RPM limiting is sufficient — no need for FIFO queueing.
        """
        if (
            self.quota_error is not None
            and self._config.quota_exhausted_mode == QuotaExhaustedMode.FAIL_FAST
        ):
            self._stats.quota_short_circuited += 1
            self._stats.dropped += 1
            raise self.quota_error
        await self._wait_rpm()
        return await self._provider.generate_with_tools(
            messages,
            tools,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    async def _worker_loop(self) -> None:
        while True:
            request = await self._queue.get()
            if request is None:
                self._queue.task_done()
                break

            if request.future.cancelled():
                self._stats.dropped += 1
                self._queue.task_done()
                continue

            wait_time = time.monotonic() - request.enqueued_at
            self._stats.queue_wait_seconds_total += wait_time
            self._stats.dequeued += 1

            # Short-circuit if quota was exhausted while waiting in queue
            if (
                self.quota_error is not None
                and self._config.quota_exhausted_mode == QuotaExhaustedMode.FAIL_FAST
            ):
                self._stats.quota_short_circuited += 1
                self._stats.dropped += 1
                if not request.future.done():
                    request.future.set_exception(self.quota_error)
                self._queue.task_done()
                continue

            await self._wait_rpm()

            try:
                result = await self._provider.generate_structured(
                    prompt=request.prompt,
                    response_model=request.response_model,
                    model=request.model,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                )
                if not request.future.done():
                    request.future.set_result(result)
            except LLMQuotaExceededError as exc:
                self.quota_error = exc
                self._quota_event.set()
                if not request.future.done():
                    request.future.set_exception(exc)
                if self._config.quota_exhausted_mode == QuotaExhaustedMode.FAIL_FAST:
                    self._drain_queue()
            except Exception as exc:
                if not request.future.done():
                    request.future.set_exception(exc)

            self._queue.task_done()

    async def _wait_rpm(self) -> None:
        """Enforce requests-per-minute rate limit."""
        async with self._rpm_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._rpm_interval:
                await asyncio.sleep(self._rpm_interval - elapsed)
            self._last_request_time = time.monotonic()

    def _drain_queue(self) -> None:
        """Drop all pending requests from the queue after quota exhaustion."""
        while not self._queue.empty():
            try:
                request = self._queue.get_nowait()
                if request is None:
                    self._queue.task_done()
                    continue
                self._stats.dropped += 1
                self._stats.quota_short_circuited += 1
                if not request.future.done():
                    request.future.set_exception(self.quota_error)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    def _handle_retryable_error(
        self, exc: Exception, attempt: int, model_name: str, operation: str
    ) -> None:
        # Scheduler delegates to underlying provider; retries are not used here.
        raise exc

    async def get_stats(self) -> SchedulerStats:
        return self._stats

    async def count_tokens(self, text: str) -> int:
        return await self._provider.count_tokens(text)

    async def close(self) -> None:
        """Stop workers by sending sentinel values."""
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        await self._provider.close()
