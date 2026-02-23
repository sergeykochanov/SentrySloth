"""Configuration via Pydantic Settings with SENTRYSLOTH_* env vars."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentrysloth.models import Confidence, Severity


class QuotaExhaustedMode(StrEnum):
    FAIL_FAST = "fail_fast"
    HEURISTIC_FALLBACK = "heuristic_fallback"
    LEGACY_FAIL_OPEN = "legacy_fail_open"


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTRYSLOTH_LLM_")

    provider: Literal["gemini", "grok"] = "grok"
    model_prefix: str = ""

    triage_model: str = "grok-4-1-fast-non-reasoning"
    analysis_model: str = "grok-4-1-fast-reasoning"

    triage_temperature: float = 0.1
    analysis_temperature: float = 0.2

    max_retries: int = 5
    retry_base_delay: float = 2.0
    scheduler_workers: int = Field(default=4, ge=1, le=16)
    queue_max_size: int = Field(default=1000, ge=1)
    max_requests_per_minute: int = Field(default=120, ge=1)
    max_tokens_per_minute: int = Field(default=1_000_000, ge=0)
    quota_exhausted_mode: QuotaExhaustedMode = QuotaExhaustedMode.FAIL_FAST
    quota_fallback_security_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    connect_timeout: float = 10.0
    read_timeout: float = 120.0
    total_timeout: float = 300.0

    # Token budgets
    triage_max_input_tokens: int = 16000
    analysis_max_input_tokens: int = 64000


class CacheConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTRYSLOTH_CACHE_")

    enabled: bool = True
    db_path: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "sentrysloth" / "cache.db"
    )
    max_age_days: int = 30
    repo_profile_enabled: bool = True
    repo_profile_history_enabled: bool = False
    repo_profile_max_chars: int = Field(default=6000, ge=500)
    repo_profile_max_items: int = Field(default=24, ge=1)
    repo_profile_bootstrap_max_files: int = Field(default=20, ge=1)
    repo_profile_bootstrap_max_tree_paths: int = Field(default=500, ge=50)
    repo_profile_bootstrap_max_file_chars: int = Field(default=4000, ge=200)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SENTRYSLOTH_",
        env_nested_delimiter="__",
    )

    gemini_api_key: SecretStr = SecretStr("")
    grok_api_key: SecretStr = SecretStr("")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    clone_base_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "sentrysloth" / "repos"
    )

    # Diff extraction
    max_file_size_kb: int = Field(default=500, ge=1)
    chunk_token_budget: int = Field(default=4000, ge=1)

    # Analysis
    min_confidence: Confidence = Confidence.LOW
    fail_on_severity: Severity | None = None

    verbose: bool = False
    prompt_version: str = "v1"


def get_settings(**overrides: object) -> Settings:
    return Settings(**overrides)
