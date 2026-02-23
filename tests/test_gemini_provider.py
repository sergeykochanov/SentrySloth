"""Tests for GeminiProvider retry logic, finish_reason handling, and error classification."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel

from sentrysloth.config import LLMConfig
from sentrysloth.providers.base import LLMProviderError, LLMQuotaExceededError
from sentrysloth.providers.gemini import GeminiProvider


class SimpleModel(BaseModel):
    value: str


def _make_provider(max_retries: int = 5) -> GeminiProvider:
    """Create a GeminiProvider with mocked client (no real API key needed)."""
    config = LLMConfig(
        analysis_model="test-model",
        analysis_temperature=0.0,
        max_retries=max_retries,
        retry_base_delay=0.0,  # no delay in tests
        total_timeout=5.0,
    )
    with patch("sentrysloth.providers.gemini.genai.Client"):
        provider = GeminiProvider(api_key="fake-key", config=config)
    return provider


def _make_success_response(text: str = '{"value": "ok"}') -> MagicMock:
    """Create a mock GenerateContentResponse with STOP finish_reason."""
    resp = MagicMock(spec=types.GenerateContentResponse)
    resp.text = text

    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.STOP
    resp.candidates = [candidate]

    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 5
    resp.usage_metadata = usage
    return resp


def _make_truncated_response() -> MagicMock:
    """Create a mock response with MAX_TOKENS finish_reason."""
    resp = MagicMock(spec=types.GenerateContentResponse)
    resp.text = '{"value": "is_'

    candidate = MagicMock()
    candidate.finish_reason = types.FinishReason.MAX_TOKENS
    resp.candidates = [candidate]

    resp.usage_metadata = None
    return resp


@pytest.mark.asyncio
async def test_retry_on_429():
    """429 errors should be retried, and succeed if a later attempt works."""
    provider = _make_provider(max_retries=5)
    success_resp = _make_success_response()

    call_count = 0

    async def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ClientError(429, {"error": "rate limited"})
        return success_resp

    provider._client.aio.models.generate_content = mock_generate

    result = await provider.generate_structured(
        prompt="test",
        response_model=SimpleModel,
    )
    assert result.data.value == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_no_retry_on_400():
    """Non-429 client errors (e.g. 400) should raise immediately without retry."""
    provider = _make_provider(max_retries=5)

    call_count = 0

    async def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ClientError(400, {"error": "bad request"})

    provider._client.aio.models.generate_content = mock_generate

    with pytest.raises(LLMProviderError, match="client error 400"):
        await provider.generate_structured(
            prompt="test",
            response_model=SimpleModel,
        )
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_on_server_error():
    """5xx ServerError should be retried."""
    provider = _make_provider(max_retries=5)
    success_resp = _make_success_response()

    call_count = 0

    async def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise ServerError(500, {"error": "internal"})
        return success_resp

    provider._client.aio.models.generate_content = mock_generate

    result = await provider.generate_structured(
        prompt="test",
        response_model=SimpleModel,
    )
    assert result.data.value == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_truncated_response_raises():
    """Response with finish_reason=MAX_TOKENS should raise LLMProviderError."""
    provider = _make_provider(max_retries=1)
    truncated_resp = _make_truncated_response()

    async def mock_generate(*args, **kwargs):
        return truncated_resp

    provider._client.aio.models.generate_content = mock_generate

    with pytest.raises(LLMProviderError, match="truncated"):
        await provider.generate_structured(
            prompt="test",
            response_model=SimpleModel,
        )


@pytest.mark.asyncio
async def test_max_retries_exhausted():
    """All attempts returning 429 should raise LLMProviderError."""
    provider = _make_provider(max_retries=5)

    call_count = 0

    async def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ClientError(429, {"error": "rate limited"})

    provider._client.aio.models.generate_content = mock_generate

    with pytest.raises(LLMProviderError, match="failed after 5 attempts"):
        await provider.generate_structured(
            prompt="test",
            response_model=SimpleModel,
        )
    assert call_count == 5


@pytest.mark.asyncio
async def test_quota_exhausted_raises_specialized_error():
    """RESOURCE_EXHAUSTED 429 should raise LLMQuotaExceededError without retries."""
    provider = _make_provider(max_retries=5)

    call_count = 0

    async def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ClientError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel",
                }
            },
        )

    provider._client.aio.models.generate_content = mock_generate

    with pytest.raises(LLMQuotaExceededError):
        await provider.generate_structured(
            prompt="test",
            response_model=SimpleModel,
        )
    assert call_count == 1


@pytest.mark.asyncio
async def test_shared_cooldown_on_429():
    """After a 429, _rate_limit_until is set so subsequent requests see cooldown."""
    provider = _make_provider(max_retries=5)
    success_resp = _make_success_response()

    call_count = 0

    async def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ClientError(429, {"error": "rate limited"})
        return success_resp

    provider._client.aio.models.generate_content = mock_generate

    # Before any request, no cooldown set.
    assert provider._rate_limit_until == 0.0

    result = await provider.generate_structured(
        prompt="test",
        response_model=SimpleModel,
    )
    assert result.data.value == "ok"
    assert call_count == 2

    # After a 429, shared cooldown timestamp was set (may have expired by now,
    # but it must have been set to a value > 0).
    assert provider._rate_limit_until > 0.0


@pytest.mark.asyncio
async def test_staggered_wakeup_after_429():
    """After cooldown, concurrent requests should be staggered (not burst)."""
    config = LLMConfig(
        analysis_model="test-model",
        analysis_temperature=0.0,
        max_retries=5,
        retry_base_delay=0.01,
        total_timeout=30.0,
    )
    with patch("sentrysloth.providers.gemini.genai.Client"):
        provider = GeminiProvider(api_key="fake-key", config=config)

    success_resp = _make_success_response()

    # Set a short cooldown so all 3 coroutines see it
    provider._rate_limit_until = time.monotonic() + 0.5

    call_times: list[float] = []

    async def mock_generate(*args, **kwargs):
        call_times.append(time.monotonic())
        return success_resp

    provider._client.aio.models.generate_content = mock_generate

    tasks = [
        provider.generate_structured(prompt="test", response_model=SimpleModel) for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)

    assert all(r.data.value == "ok" for r in results)
    assert len(call_times) == 3

    call_times.sort()
    # Between the first and last API call there should be at least 0.5s of stagger
    # (lock serializes wakeups; each pushes _rate_limit_until forward by 1-3s)
    total_spread = call_times[-1] - call_times[0]
    assert total_spread >= 0.5, f"Calls not staggered enough: spread={total_spread:.2f}s"


@pytest.mark.asyncio
async def test_no_overhead_without_rate_limit():
    """Without active cooldown, requests complete without artificial delay."""
    provider = _make_provider(max_retries=1)
    success_resp = _make_success_response()

    async def mock_generate(*args, **kwargs):
        return success_resp

    provider._client.aio.models.generate_content = mock_generate

    start = time.monotonic()
    result = await provider.generate_structured(
        prompt="test",
        response_model=SimpleModel,
    )
    elapsed = time.monotonic() - start

    assert result.data.value == "ok"
    assert elapsed < 1.0, f"Request took too long without rate limit: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_blocked_response_null_content():
    """Gemini blocked response (candidate.content is None) should raise LLMProviderError."""
    provider = _make_provider(max_retries=1)

    resp = MagicMock(spec=types.GenerateContentResponse)
    candidate = MagicMock()
    candidate.content = None
    resp.candidates = [candidate]
    resp.usage_metadata = None

    async def mock_generate(*args, **kwargs):
        return resp

    provider._client.aio.models.generate_content = mock_generate

    with pytest.raises(LLMProviderError, match="empty/blocked"):
        await provider.generate_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )


@pytest.mark.asyncio
async def test_tool_call_id_uniqueness():
    """Tool call IDs should be unique across invocations (no collision by index)."""
    provider = _make_provider(max_retries=1)

    fc1 = MagicMock()
    fc1.name = "read_file"
    fc1.args = {"file_path": "a.py"}

    fc2 = MagicMock()
    fc2.name = "read_file"
    fc2.args = {"file_path": "b.py"}

    part1 = MagicMock()
    part1.function_call = fc1
    part1.text = None

    part2 = MagicMock()
    part2.function_call = fc2
    part2.text = None

    candidate = MagicMock()
    candidate.content = MagicMock()
    candidate.content.parts = [part1, part2]

    resp = MagicMock(spec=types.GenerateContentResponse)
    resp.candidates = [candidate]
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 5
    resp.usage_metadata = usage

    async def mock_generate(*args, **kwargs):
        return resp

    provider._client.aio.models.generate_content = mock_generate

    result = await provider.generate_with_tools(
        messages=[{"role": "user", "content": "test"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].id != result.tool_calls[1].id
