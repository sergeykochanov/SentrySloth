"""Abstract base for LLM providers."""

from __future__ import annotations

import abc
import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from sentrysloth.config import LLMConfig
from sentrysloth.models import LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMProviderError(Exception):
    pass


class LLMQuotaExceededError(LLMProviderError):
    """Raised when upstream LLM quota is exhausted (429 RESOURCE_EXHAUSTED)."""

    def __init__(
        self,
        message: str,
        *,
        retry_delay_seconds: float | None = None,
        quota_metric: str = "",
        is_daily_quota: bool = False,
    ) -> None:
        super().__init__(message)
        self.retry_delay_seconds = retry_delay_seconds
        self.quota_metric = quota_metric
        self.is_daily_quota = is_daily_quota


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict


class ToolCallResponse(BaseModel):
    """Response from generate_with_tools — may contain content, tool_calls, or both."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(abc.ABC):
    """Abstract LLM provider with structured output, retries, and rate limiting."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._rate_limit_until: float = 0.0
        self._rate_limit_lock = asyncio.Lock()

    async def _wait_for_rate_limit(self) -> None:
        """Serialize access during rate-limit cooldown to prevent thundering herd."""
        async with self._rate_limit_lock:
            now = time.monotonic()
            if self._rate_limit_until > now:
                delay = self._rate_limit_until - now
                logger.info("Rate limited, waiting %.1fs before next request", delay)
                await asyncio.sleep(delay)
                # Stagger: push next wakeup to avoid simultaneous burst
                self._rate_limit_until = time.monotonic() + random.uniform(1.0, 3.0)  # noqa: S311

    async def _call_with_retries(
        self,
        call: Callable[[], Awaitable[Any]],
        model_name: str,
        *,
        operation: str = "Request",
    ) -> tuple[Any, float]:
        """Execute *call* with retries and exponential backoff.

        Returns ``(result, elapsed_ms)``.  On each failure
        ``_handle_retryable_error`` is invoked — it should **return** to retry
        or **raise** to propagate immediately.
        """
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            await self._wait_for_rate_limit()
            start = time.monotonic()
            try:
                result = await call()
                elapsed_ms = (time.monotonic() - start) * 1000
                return result, elapsed_ms
            except Exception as exc:
                self._handle_retryable_error(exc, attempt, model_name, operation)
                last_exc = exc

            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                await asyncio.sleep(delay)

        raise LLMProviderError(
            f"{operation} failed after {self.config.max_retries} attempts: {last_exc}"
        ) from last_exc

    @abc.abstractmethod
    def _handle_retryable_error(
        self, exc: Exception, attempt: int, model_name: str, operation: str
    ) -> None:
        """Classify a provider-specific exception.

        *Return* to signal that the call should be retried.
        *Raise* to propagate immediately (quota exhaustion, client errors, etc.).
        """
        ...

    @abc.abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ) -> LLMResponse[T]:
        """Generate a structured response matching the Pydantic model."""
        ...

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ) -> ToolCallResponse:
        """Generate a response that may include tool calls.

        Not abstract — default raises NotImplementedError so existing providers
        continue to work. Only providers that support tool use override this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support tool use")

    @abc.abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Estimate token count for the given text."""
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...
