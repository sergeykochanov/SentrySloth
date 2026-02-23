"""Tests for centralized LLM request scheduler."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from sentrysloth.config import LLMConfig, QuotaExhaustedMode
from sentrysloth.models import LLMResponse
from sentrysloth.providers.base import LLMProvider, LLMQuotaExceededError, ToolCallResponse
from sentrysloth.providers.scheduler import LlmRequestScheduler


class SimpleModel(BaseModel):
    value: str


class DummyProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(LLMConfig())
        self.calls: list[str] = []
        self.raise_quota = False
        self.tool_calls: list[list[dict]] = []
        self.tool_response: ToolCallResponse | None = None

    def _handle_retryable_error(
        self, exc: Exception, attempt: int, model_name: str, operation: str
    ) -> None:
        raise

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[SimpleModel],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ) -> LLMResponse[SimpleModel]:
        self.calls.append(prompt)
        if self.raise_quota:
            raise LLMQuotaExceededError("quota exhausted", is_daily_quota=True)
        return LLMResponse(
            data=response_model(value=prompt),
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            model=model or "dummy",
        )

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ) -> ToolCallResponse:
        self.tool_calls.append(tools)
        if self.tool_response is not None:
            return self.tool_response
        return ToolCallResponse(content="done", tool_calls=[], input_tokens=10, output_tokens=5)

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_scheduler_preserves_fifo_order():
    provider = DummyProvider()
    config = LLMConfig(
        scheduler_workers=1,
        queue_max_size=100,
        max_requests_per_minute=1000,
        max_tokens_per_minute=0,
    )
    scheduler = LlmRequestScheduler(provider, config)
    await scheduler.start()

    prompts = ["one", "two", "three"]
    results = await asyncio.gather(
        *[scheduler.generate_structured(prompt=p, response_model=SimpleModel) for p in prompts]
    )

    assert provider.calls == prompts
    assert [r.data.value for r in results] == prompts
    await scheduler.close()


@pytest.mark.asyncio
async def test_scheduler_fail_fast_short_circuits_queue():
    provider = DummyProvider()
    provider.raise_quota = True
    config = LLMConfig(
        scheduler_workers=1,
        queue_max_size=100,
        max_requests_per_minute=1000,
        max_tokens_per_minute=0,
        quota_exhausted_mode=QuotaExhaustedMode.FAIL_FAST,
    )
    scheduler = LlmRequestScheduler(provider, config)
    await scheduler.start()

    with pytest.raises(LLMQuotaExceededError):
        await asyncio.gather(
            scheduler.generate_structured(prompt="a", response_model=SimpleModel),
            scheduler.generate_structured(prompt="b", response_model=SimpleModel),
            scheduler.generate_structured(prompt="c", response_model=SimpleModel),
        )

    stats = await scheduler.get_stats()
    assert stats.quota_short_circuited >= 1
    assert stats.dropped >= 1
    await scheduler.close()


@pytest.mark.asyncio
async def test_scheduler_proxies_generate_with_tools():
    """Scheduler delegates generate_with_tools to underlying provider with RPM throttling."""
    provider = DummyProvider()
    config = LLMConfig(
        scheduler_workers=1,
        queue_max_size=100,
        max_requests_per_minute=1000,
        max_tokens_per_minute=0,
    )
    scheduler = LlmRequestScheduler(provider, config)
    await scheduler.start()

    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    result = await scheduler.generate_with_tools(messages, tools, model="test-model")

    assert result.content == "done"
    assert result.input_tokens == 10
    assert len(provider.tool_calls) == 1
    assert provider.tool_calls[0] == tools
    await scheduler.close()


@pytest.mark.asyncio
async def test_scheduler_generate_with_tools_propagates_not_implemented():
    """If underlying provider doesn't support tool use, NotImplementedError propagates."""

    class NoToolsProvider(DummyProvider):
        async def generate_with_tools(self, *args, **kwargs):
            raise NotImplementedError("NoToolsProvider does not support tool use")

    provider = NoToolsProvider()
    config = LLMConfig(
        scheduler_workers=1,
        queue_max_size=100,
        max_requests_per_minute=1000,
        max_tokens_per_minute=0,
    )
    scheduler = LlmRequestScheduler(provider, config)
    await scheduler.start()

    with pytest.raises(NotImplementedError, match="does not support tool use"):
        await scheduler.generate_with_tools(
            [{"role": "user", "content": "hi"}],
            [{"type": "function", "function": {"name": "t"}}],
        )
    await scheduler.close()
