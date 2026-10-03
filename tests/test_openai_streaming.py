"""Exercise the public provider API through the real SDK and a local SSE transport."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import openai
import pytest
from pydantic import BaseModel

from sentrysloth.config import LLMConfig
from sentrysloth.providers.base import LLMProviderError
from sentrysloth.providers.openai_compat import OpenAICompatProvider


class Result(BaseModel):
    value: str


def packet(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "response-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "grok-4.7",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def usage_packet() -> dict:
    return {
        **packet({}),
        "choices": [],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 170,
            "completion_tokens_details": {"reasoning_tokens": 50},
            "cost_in_usd_ticks": 1234,
        },
    }


def sse(packets: list[dict]) -> bytes:
    return (
        "".join(f"data: {json.dumps(item)}\n\n" for item in packets) + "data: [DONE]\n\n"
    ).encode()


def provider_for(handler, *, max_retries: int = 1) -> OpenAICompatProvider:
    config = LLMConfig(max_retries=max_retries, retry_base_delay=0.0)
    client = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5),
    )
    with patch("sentrysloth.providers.openai_compat.openai.AsyncOpenAI", return_value=client):
        return OpenAICompatProvider(api_key="test", config=config, base_url="https://api.x.ai/v1")


@pytest.mark.asyncio
async def test_streamed_structured_text_and_reasoning_usage():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(
                [
                    packet({"content": '{"val'}),
                    packet({"content": 'ue":"ok"}'}, "stop"),
                    usage_packet(),
                ]
            ),
        )

    provider = provider_for(handler)
    try:
        result = await provider.generate_structured("review", Result)
        assert result.data.value == "ok"
        assert result.input_tokens == 100
        assert result.output_tokens == 70
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_streamed_interleaved_tool_calls():
    packets = [
        packet(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "two",
                        "function": {"name": "search_code", "arguments": "{"},
                    },
                    {"index": 0, "id": "one", "function": {"name": "read_file", "arguments": "{"}},
                ]
            }
        ),
        packet(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '"path":"app.py"}'}},
                    {"index": 1, "function": {"arguments": '"query":"guard"}'}},
                ]
            },
            "tool_calls",
        ),
        usage_packet(),
    ]
    provider = provider_for(lambda request: httpx.Response(200, content=sse(packets)))
    try:
        result = await provider.generate_with_tools(messages=[], tools=[])
        assert [(call.id, call.name, call.arguments) for call in result.tool_calls] == [
            ("one", "read_file", {"path": "app.py"}),
            ("two", "search_code", {"query": "guard"}),
        ]
        assert result.output_tokens == 70
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("packets", "error"),
    [
        ([packet({"content": "partial"})], "without a final choice"),
        ([packet({"content": "partial"}, "stop")], "has no usage"),
        ([packet({"content": "partial"}, "length"), usage_packet()], "truncated"),
    ],
)
async def test_stream_failures_are_explicit(packets, error):
    provider = provider_for(lambda request: httpx.Response(200, content=sse(packets)))
    try:
        with pytest.raises(LLMProviderError, match=error):
            await provider.generate_with_tools(messages=[], tools=[])
    finally:
        await provider.close()


class InterruptedStream(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self):
        yield sse([packet({"content": '{"value":"discard'})]).replace(b"data: [DONE]\n\n", b"")
        raise httpx.ReadError("stream disconnected")

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_midstream_disconnect_retries_from_fresh_response():
    interrupted = InterruptedStream()
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=interrupted)
        return httpx.Response(
            200, content=sse([packet({"content": '{"value":"ok"}'}, "stop"), usage_packet()])
        )

    provider = provider_for(handler, max_retries=2)
    try:
        result = await provider.generate_structured("review", Result)
        assert result.data.value == "ok"
        assert len(requests) == 2
        assert interrupted.closed
        assert result.output_tokens == 70
    finally:
        await provider.close()
