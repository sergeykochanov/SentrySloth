"""Tests for OpenAICompatProvider retry logic, error handling, and structured output."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from sentrysloth.config import LLMConfig
from sentrysloth.providers.base import (
    LLMProviderError,
    LLMQuotaExceededError,
    ToolCallResponse,
)
from sentrysloth.providers.openai_compat import OpenAICompatProvider


class SimpleModel(BaseModel):
    value: str


def _make_provider(max_retries: int = 5) -> OpenAICompatProvider:
    """Create an OpenAICompatProvider with mocked client."""
    config = LLMConfig(
        analysis_model="test-model",
        analysis_temperature=0.0,
        max_retries=max_retries,
        retry_base_delay=0.0,
        total_timeout=5.0,
    )
    with patch("sentrysloth.providers.openai_compat.openai.AsyncOpenAI"):
        provider = OpenAICompatProvider(
            api_key="fake-key",
            config=config,
            base_url="https://api.x.ai/v1",
        )
    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()
    provider._client.close = AsyncMock()
    return provider


def _make_success_response(text: str = '{"value": "ok"}') -> MagicMock:
    """Create a mock ChatCompletion response."""
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = text

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_truncated_response() -> MagicMock:
    """Create a mock response with finish_reason=length."""
    choice = MagicMock()
    choice.finish_reason = "length"
    choice.message.content = '{"value": "trunc'

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def _make_rate_limit_error(message: str = "Rate limit exceeded") -> openai.RateLimitError:
    """Create a RateLimitError with the given message."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}
    mock_resp.json.return_value = {"error": {"message": message}}
    return openai.RateLimitError(
        message=message,
        response=mock_resp,
        body={"error": {"message": message}},
    )


def _make_api_status_error(status_code: int, message: str = "error") -> openai.APIStatusError:
    """Create an APIStatusError with the given status code."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = {}
    mock_resp.json.return_value = {"error": {"message": message}}
    return openai.APIStatusError(
        message=message,
        response=mock_resp,
        body={"error": {"message": message}},
    )


@pytest.mark.asyncio
async def test_retry_on_429():
    """429 rate-limit errors should be retried, succeeding on later attempt."""
    provider = _make_provider(max_retries=5)
    success_resp = _make_success_response()

    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _make_rate_limit_error()
        return success_resp

    provider._client.chat.completions.create = mock_create

    result = await provider.generate_structured(
        prompt="test",
        response_model=SimpleModel,
    )
    assert result.data.value == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_no_retry_on_400():
    """Non-retryable client errors (400) should raise immediately."""
    provider = _make_provider(max_retries=5)

    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_api_status_error(400, "bad request")

    provider._client.chat.completions.create = mock_create

    with pytest.raises(LLMProviderError, match="API error 400"):
        await provider.generate_structured(
            prompt="test",
            response_model=SimpleModel,
        )
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_on_server_error():
    """5xx errors should be retried."""
    provider = _make_provider(max_retries=5)
    success_resp = _make_success_response()

    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise _make_api_status_error(500, "internal server error")
        return success_resp

    provider._client.chat.completions.create = mock_create

    result = await provider.generate_structured(
        prompt="test",
        response_model=SimpleModel,
    )
    assert result.data.value == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_truncated_response_raises():
    """Response with finish_reason=length should raise LLMProviderError."""
    provider = _make_provider(max_retries=1)
    truncated_resp = _make_truncated_response()

    async def mock_create(*args, **kwargs):
        return truncated_resp

    provider._client.chat.completions.create = mock_create

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

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error()

    provider._client.chat.completions.create = mock_create

    with pytest.raises(LLMProviderError, match="failed after 5 attempts"):
        await provider.generate_structured(
            prompt="test",
            response_model=SimpleModel,
        )
    assert call_count == 5


@pytest.mark.asyncio
async def test_quota_exhausted_raises():
    """Quota-related 429 should raise LLMQuotaExceededError immediately."""
    provider = _make_provider(max_retries=5)

    call_count = 0

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error("You exceeded your current quota")

    provider._client.chat.completions.create = mock_create

    with pytest.raises(LLMQuotaExceededError):
        await provider.generate_structured(
            prompt="test",
            response_model=SimpleModel,
        )
    assert call_count == 1


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
    with patch("sentrysloth.providers.openai_compat.openai.AsyncOpenAI"):
        provider = OpenAICompatProvider(
            api_key="fake-key",
            config=config,
            base_url="https://api.x.ai/v1",
        )
    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()
    provider._client.close = AsyncMock()

    success_resp = _make_success_response()

    # Set a short cooldown so all 3 coroutines see it
    provider._rate_limit_until = time.monotonic() + 0.5

    call_times: list[float] = []

    async def mock_create(*args, **kwargs):
        call_times.append(time.monotonic())
        return success_resp

    provider._client.chat.completions.create = mock_create

    tasks = [
        provider.generate_structured(prompt="test", response_model=SimpleModel) for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)

    assert all(r.data.value == "ok" for r in results)
    assert len(call_times) == 3

    call_times.sort()
    total_spread = call_times[-1] - call_times[0]
    assert total_spread >= 0.5, f"Calls not staggered enough: spread={total_spread:.2f}s"


@pytest.mark.asyncio
async def test_no_overhead_without_rate_limit():
    """Without active cooldown, requests complete without artificial delay."""
    provider = _make_provider(max_retries=1)
    success_resp = _make_success_response()

    async def mock_create(*args, **kwargs):
        return success_resp

    provider._client.chat.completions.create = mock_create

    start = time.monotonic()
    result = await provider.generate_structured(
        prompt="test",
        response_model=SimpleModel,
    )
    elapsed = time.monotonic() - start

    assert result.data.value == "ok"
    assert elapsed < 1.0, f"Request took too long without rate limit: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_successful_structured_output():
    """Happy path: valid JSON is parsed into the response model."""
    provider = _make_provider(max_retries=1)
    resp = _make_success_response('{"value": "hello world"}')

    async def mock_create(*args, **kwargs):
        return resp

    provider._client.chat.completions.create = mock_create

    result = await provider.generate_structured(
        prompt="test",
        response_model=SimpleModel,
    )
    assert result.data.value == "hello world"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "test-model"


@pytest.mark.asyncio
async def test_invalid_json_response():
    """Invalid JSON in response should raise LLMProviderError."""
    provider = _make_provider(max_retries=1)
    resp = _make_success_response("not valid json {{{")

    async def mock_create(*args, **kwargs):
        return resp

    provider._client.chat.completions.create = mock_create

    with pytest.raises(LLMProviderError, match="Failed to parse response"):
        await provider.generate_structured(
            prompt="test",
            response_model=SimpleModel,
        )


def _make_tool_call_response() -> MagicMock:
    """Create a mock response with tool calls."""
    tc = MagicMock()
    tc.id = "call_123"
    tc.function.name = "read_file"
    tc.function.arguments = '{"file_path": "src/auth.py"}'

    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [tc]

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 20

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_content_only_tool_response(text: str = "No issues found.") -> MagicMock:
    """Create a mock response with content only (no tool calls)."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = None

    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.completion_tokens = 10

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


@pytest.mark.asyncio
async def test_tool_call_response_parsed():
    """generate_with_tools should parse tool_calls from API response."""
    provider = _make_provider(max_retries=1)
    resp = _make_tool_call_response()

    async def mock_create(*args, **kwargs):
        return resp

    provider._client.chat.completions.create = mock_create

    result = await provider.generate_with_tools(
        messages=[{"role": "user", "content": "test"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    assert isinstance(result, ToolCallResponse)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_123"
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"file_path": "src/auth.py"}
    assert result.input_tokens == 100
    assert result.output_tokens == 20


@pytest.mark.asyncio
async def test_malformed_json_in_tool_arguments():
    """Malformed JSON in tool call arguments should be skipped, not crash."""
    provider = _make_provider(max_retries=1)

    tc = MagicMock()
    tc.id = "call_bad"
    tc.function.name = "read_file"
    tc.function.arguments = "not valid json {{{"

    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [tc]

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 20

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage

    async def mock_create(*args, **kwargs):
        return resp

    provider._client.chat.completions.create = mock_create

    result = await provider.generate_with_tools(
        messages=[{"role": "user", "content": "test"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    assert isinstance(result, ToolCallResponse)
    assert len(result.tool_calls) == 0
    assert result.content == ""


@pytest.mark.asyncio
async def test_no_tool_calls_returns_content():
    """generate_with_tools returns content when no tool calls are made."""
    provider = _make_provider(max_retries=1)
    resp = _make_content_only_tool_response("Analysis complete.")

    async def mock_create(*args, **kwargs):
        return resp

    provider._client.chat.completions.create = mock_create

    result = await provider.generate_with_tools(
        messages=[{"role": "user", "content": "test"}],
        tools=[],
    )

    assert isinstance(result, ToolCallResponse)
    assert len(result.tool_calls) == 0
    assert result.content == "Analysis complete."
    assert result.input_tokens == 50
    assert result.output_tokens == 10
