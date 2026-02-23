"""LLM provider factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentrysloth.config import Settings
    from sentrysloth.providers.base import LLMProvider


def create_provider(settings: Settings) -> LLMProvider:
    """Create an LLM provider based on settings.

    Imports are lazy so that unused SDK dependencies are not required at import time.
    """
    provider = settings.llm.provider

    if provider == "gemini":
        if not settings.gemini_api_key.get_secret_value():
            raise ValueError("SENTRYSLOTH_GEMINI_API_KEY is required for provider=gemini")
        from sentrysloth.providers.gemini import GeminiProvider

        return GeminiProvider(settings.gemini_api_key.get_secret_value(), settings.llm)

    if provider == "grok":
        if not settings.grok_api_key.get_secret_value():
            raise ValueError("SENTRYSLOTH_GROK_API_KEY is required for provider=grok")
        from sentrysloth.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            api_key=settings.grok_api_key.get_secret_value(),
            config=settings.llm,
            base_url="https://api.x.ai/v1",
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
