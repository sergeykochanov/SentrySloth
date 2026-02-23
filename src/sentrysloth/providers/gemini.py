"""Gemini LLM provider via google-genai SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from typing import TypeVar

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import ValidationError

from sentrysloth.analyzers.diff_extractor import estimate_tokens
from sentrysloth.config import LLMConfig
from sentrysloth.models import LLMResponse
from sentrysloth.providers.base import (
    LLMProvider,
    LLMProviderError,
    LLMQuotaExceededError,
    ToolCall,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)

# Suppress noisy AFC / HTTP request logging from google-genai SDK.
logging.getLogger("google.genai").setLevel(logging.WARNING)

T = TypeVar("T")


class GeminiProvider(LLMProvider):
    """Google Gemini provider with rate limiting, retries, and structured output."""

    def __init__(self, api_key: str, config: LLMConfig) -> None:
        super().__init__(config)
        # Configure SDK-level request timeout (milliseconds).
        timeout_ms = max(1, int(config.total_timeout * 1000))
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    def _handle_retryable_error(
        self, exc: Exception, attempt: int, model_name: str, operation: str
    ) -> None:
        if isinstance(exc, TimeoutError):
            logger.warning(
                "%s attempt %d/%d timed out for model %s",
                operation,
                attempt + 1,
                self.config.max_retries,
                model_name,
            )
            return
        if isinstance(exc, ConnectionError):
            logger.warning(
                "%s attempt %d/%d connection error for model %s: %s",
                operation,
                attempt + 1,
                self.config.max_retries,
                model_name,
                exc,
            )
            return
        if isinstance(exc, ClientError):
            if exc.code == 429:
                quota_info = _classify_quota_429(exc)
                if quota_info.is_quota_exhausted:
                    raise LLMQuotaExceededError(
                        f"Quota exhausted for model {model_name}: {exc}",
                        retry_delay_seconds=quota_info.retry_delay_seconds,
                        quota_metric=quota_info.quota_metric,
                        is_daily_quota=quota_info.is_daily_quota,
                    ) from exc
                delay = self.config.retry_base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                self._rate_limit_until = time.monotonic() + delay
                logger.warning(
                    "%s attempt %d/%d rate limited (429) for model %s, cooldown %.1fs",
                    operation,
                    attempt + 1,
                    self.config.max_retries,
                    model_name,
                    delay,
                )
                return
            raise LLMProviderError(f"Gemini client error {exc.code}: {exc}") from exc
        if isinstance(exc, ServerError):
            logger.warning(
                "%s attempt %d/%d server error (%d) for model %s",
                operation,
                attempt + 1,
                self.config.max_retries,
                exc.code,
                model_name,
            )
            return
        raise

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ) -> LLMResponse[T]:
        model_name = model or self.config.analysis_model
        temp = temperature if temperature is not None else self.config.analysis_temperature

        schema = response_model.model_json_schema()

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )

        response, elapsed_ms = await self._call_with_retries(
            lambda: asyncio.wait_for(
                self._call_api(model_name, prompt, config),
                timeout=self.config.total_timeout,
            ),
            model_name,
        )

        # Check for truncated response before attempting to parse.
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason == types.FinishReason.MAX_TOKENS:
                logger.warning(
                    "Response truncated (MAX_TOKENS) for model %s",
                    model_name,
                )
                raise LLMProviderError(
                    f"Response truncated: model {model_name} hit max_output_tokens limit"
                )

        raw_text = response.text
        if not raw_text:
            raise LLMProviderError("Empty response from Gemini")

        try:
            parsed = json.loads(raw_text)
            data = response_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse Gemini response: %s\nRaw: %s", exc, raw_text[:500])
            raise LLMProviderError(f"Failed to parse response: {exc}") from exc

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            data=data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            model=model_name,
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
        model_name = model or self.config.analysis_model
        temp = temperature if temperature is not None else self.config.analysis_temperature

        # Convert OpenAI-format tools to Gemini function declarations
        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            function_declarations.append(
                types.FunctionDeclaration(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters=func.get("parameters", {}),
                )
            )
        gemini_tools = [types.Tool(function_declarations=function_declarations)]

        # Convert OpenAI-format messages to Gemini contents
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                # Gemini uses system_instruction, but for tool use we prepend to first user message
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )
            elif role == "assistant":
                parts = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        parts.append(
                            types.Part.from_function_call(
                                name=tc["function"]["name"],
                                args=json.loads(tc["function"]["arguments"])
                                if isinstance(tc["function"]["arguments"], str)
                                else tc["function"]["arguments"],
                            )
                        )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.get("name", ""),
                                response={"result": msg.get("content", "")},
                            )
                        ],
                    )
                )
            else:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.get("content", ""))],
                    )
                )

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_output_tokens,
            tools=gemini_tools,
        )

        response, _ = await self._call_with_retries(
            lambda: asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                ),
                timeout=self.config.total_timeout,
            ),
            model_name,
            operation="Tool call",
        )

        # Parse response
        tool_calls: list[ToolCall] = []
        content_text = ""

        if response.candidates:
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise LLMProviderError("Gemini returned empty/blocked response (no content parts)")
            for part in candidate.content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )
                elif part.text:
                    content_text += part.text

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return ToolCallResponse(
            content=content_text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _call_api(
        self,
        model_name: str,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        """Call Gemini API using native async client."""
        return await self._client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

    async def count_tokens(self, text: str) -> int:
        """Estimate token count — delegates to shared heuristic for consistency."""
        return estimate_tokens(text)

    async def close(self) -> None:
        await self._client.aio.aclose()


class _Quota429Info:
    def __init__(
        self,
        *,
        is_quota_exhausted: bool,
        is_daily_quota: bool = False,
        retry_delay_seconds: float | None = None,
        quota_metric: str = "",
    ) -> None:
        self.is_quota_exhausted = is_quota_exhausted
        self.is_daily_quota = is_daily_quota
        self.retry_delay_seconds = retry_delay_seconds
        self.quota_metric = quota_metric


def _classify_quota_429(exc: ClientError) -> _Quota429Info:
    payload = str(exc)
    lowered = payload.lower()

    is_quota = any(
        marker in lowered
        for marker in (
            "resource_exhausted",
            "quota exceeded",
            "quotafailure",
            "current quota",
        )
    )
    is_daily = any(
        marker in lowered
        for marker in (
            "generaterequestsperday",
            "perday",
            "requests per day",
        )
    )

    metric_match = re.search(r"quotaMetric['\"]?\s*:\s*['\"]([^'\"]+)['\"]", payload, flags=re.I)
    metric = metric_match.group(1) if metric_match else ""

    retry_delay_match = re.search(
        r"retry(?:\s*in|Delay)['\"]?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)s",
        payload,
    )
    retry_delay = float(retry_delay_match.group(1)) if retry_delay_match else None

    return _Quota429Info(
        is_quota_exhausted=is_quota,
        is_daily_quota=is_daily,
        retry_delay_seconds=retry_delay,
        quota_metric=metric,
    )
