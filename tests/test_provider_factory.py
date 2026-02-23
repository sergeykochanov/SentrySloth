"""Tests for the provider factory function."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sentrysloth.config import LLMConfig, Settings
from sentrysloth.providers import create_provider


def test_create_gemini_provider():
    settings = Settings(gemini_api_key="fake-key", llm=LLMConfig(provider="gemini"))
    with patch("sentrysloth.providers.gemini.genai.Client"):
        provider = create_provider(settings)
    from sentrysloth.providers.gemini import GeminiProvider

    assert isinstance(provider, GeminiProvider)


def test_create_grok_provider():
    settings = Settings(grok_api_key="fake-key", llm=LLMConfig(provider="grok"))
    with patch("sentrysloth.providers.openai_compat.openai.AsyncOpenAI"):
        provider = create_provider(settings)
    from sentrysloth.providers.openai_compat import OpenAICompatProvider

    assert isinstance(provider, OpenAICompatProvider)


def test_missing_gemini_api_key_raises():
    settings = Settings(gemini_api_key="", llm=LLMConfig(provider="gemini"))
    with pytest.raises(ValueError, match="SENTRYSLOTH_GEMINI_API_KEY"):
        create_provider(settings)


def test_missing_grok_api_key_raises():
    settings = Settings(grok_api_key="", llm=LLMConfig(provider="grok"))
    with pytest.raises(ValueError, match="SENTRYSLOTH_GROK_API_KEY"):
        create_provider(settings)
