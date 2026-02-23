"""OpenAI-compatible LLM provider (xAI Grok, etc.) via openai SDK."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import TypeVar

import httpx
import openai
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

T = TypeVar("T")


class OpenAICompatProvider(LLMProvider):
    """OpenAI-compatible provider with rate limiting, retries, and structured output."""

    def __init__(self, api_key: str, config: LLMConfig, base_url: str) -> None:
        super().__init__(config)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=config.connect_timeout,
                read=config.read_timeout,
                write=config.read_timeout,
                pool=config.connect_timeout,
            ),
            max_retries=config.max_retries,
        )

    def _handle_retryable_error(
        self, exc: Exception, attempt: int, model_name: str, operation: str
    ) -> None:
        if isinstance(exc, openai.APITimeoutError):
            logger.warning(
                "%s attempt %d/%d timed out for model %s",
                operation,
                attempt + 1,
                self.config.max_retries,
                model_name,
            )
            return
        if isinstance(exc, openai.APIConnectionError):
            logger.warning(
                "%s attempt %d/%d connection error for model %s: %s",
                operation,
                attempt + 1,
                self.config.max_retries,
                model_name,
                exc,
            )
            return
        if isinstance(exc, openai.RateLimitError):
            if _is_quota_exhausted(exc):
                raise LLMQuotaExceededError(
                    f"Quota exhausted for model {model_name}: {exc}",
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
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code >= 500:
                logger.warning(
                    "%s attempt %d/%d server error (%d) for model %s",
                    operation,
                    attempt + 1,
                    self.config.max_retries,
                    exc.status_code,
                    model_name,
                )
                return
            raise LLMProviderError(f"API error {exc.status_code}: {exc}") from exc
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

        response, elapsed_ms = await self._call_with_retries(
            lambda: self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                },
            ),
            model_name,
        )

        # Check for truncated response
        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning(
                "Response truncated (length) for model %s",
                model_name,
            )
            raise LLMProviderError(f"Response truncated: model {model_name} hit max_tokens limit")

        raw_text = choice.message.content
        if not raw_text:
            raise LLMProviderError("Empty response from API")

        try:
            parsed = json.loads(raw_text)
            data = response_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Failed to parse API response: %s\nRaw: %s", exc, raw_text[:500])
            raise LLMProviderError(f"Failed to parse response: {exc}") from exc

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0

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

        response, _ = await self._call_with_retries(
            lambda: self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                temperature=temp,
                max_tokens=max_output_tokens,
            ),
            model_name,
            operation="Tool call",
        )

        choice = response.choices[0]
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping tool call %s: malformed arguments: %s",
                        tc.function.name,
                        exc,
                    )
                    continue
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0

        return ToolCallResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def count_tokens(self, text: str) -> int:
        """Estimate token count — delegates to shared heuristic for consistency."""
        return estimate_tokens(text)

    async def close(self) -> None:
        await self._client.close()


def _is_quota_exhausted(exc: openai.RateLimitError) -> bool:
    """Check if a 429 error indicates quota exhaustion vs transient rate limit."""
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "quota",
            "billing",
            "insufficient_quota",
            "budget",
        )
    )
