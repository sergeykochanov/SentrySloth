"""Tests for accumulated RepoProfile bootstrap and update helpers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from sentrysloth.cache.repo_profile import (
    load_or_bootstrap_repo_profile,
    serialize_repo_profile_for_prompt,
    update_repo_profile_after_scan,
)
from sentrysloth.cache.storage import CacheStorage
from sentrysloth.config import get_settings
from sentrysloth.models import (
    DiffChunk,
    DiffHunk,
    LLMResponse,
    RepoProfile,
    TriageResult,
    TriageStats,
)
from sentrysloth.providers.base import LLMProvider, LLMProviderError
from sentrysloth.sources.git import GitSource


def _mock_git_source() -> GitSource:
    gs = MagicMock(spec=GitSource)
    gs.get_file_content = AsyncMock(
        side_effect=lambda _ref, file_path: (
            "# readme"
            if file_path == "README.md"
            else "requests==2.0"
            if file_path == "requirements.txt"
            else None
        )
    )
    gs.list_files = AsyncMock(return_value=["src/app.py", "src/auth.py", "README.md"])
    return gs


def _chunk() -> DiffChunk:
    return DiffChunk(
        file_path="src/auth.py",
        hunks=[
            DiffHunk(
                source_start=1,
                source_length=1,
                target_start=1,
                target_length=1,
                content="+ return token == expected",
            )
        ],
        raw_diff="+ return token == expected",
        token_estimate=10,
    )


@pytest.mark.asyncio
async def test_bootstrap_creates_profile_and_persists(tmp_path):
    db = CacheStorage(tmp_path / "cache.db")
    await db.initialize()
    provider = AsyncMock(spec=LLMProvider)

    async def fake_generate(prompt, response_model, **kwargs):
        data = response_model(
            overview=["one", "two", "three"],
            tech_stack=["python"],
            modules=[
                {"path_prefix": "src", "purpose": "app code"},
                {"path_prefix": "tests", "purpose": "tests"},
                {"path_prefix": "docs", "purpose": "docs"},
            ],
        )
        return LLMResponse(data=data, input_tokens=1, output_tokens=1, latency_ms=1.0, model="fake")

    provider.generate_structured = fake_generate
    settings = get_settings(
        cache={
            "repo_profile_max_items": 2,
            "repo_profile_max_chars": 500,
        }
    )

    profile = await load_or_bootstrap_repo_profile(
        db,
        provider,
        settings,
        _mock_git_source(),
        "repo-url",
        "v1.1",
    )
    assert profile is not None
    assert len(profile.overview) == 2
    assert len(profile.modules) == 2
    cached = await db.get_repo_profile("repo-url")
    assert cached is not None
    await db.close()


@pytest.mark.asyncio
async def test_cache_hit_skips_bootstrap_call(tmp_path):
    db = CacheStorage(tmp_path / "cache.db")
    await db.initialize()
    profile = RepoProfile(repo="repo-url", last_ref="v1.0", overview=["cached"])
    await db.set_repo_profile("repo-url", profile.model_dump_json(), "v1.0")

    provider = AsyncMock(spec=LLMProvider)
    settings = get_settings()

    loaded = await load_or_bootstrap_repo_profile(
        db,
        provider,
        settings,
        _mock_git_source(),
        "repo-url",
        "v1.1",
    )
    assert loaded is not None
    assert loaded.overview == ["cached"]
    provider.generate_structured.assert_not_called()
    await db.close()


@pytest.mark.asyncio
async def test_update_profile_after_scan_persists_last_ref(tmp_path):
    db = CacheStorage(tmp_path / "cache.db")
    await db.initialize()
    provider = AsyncMock(spec=LLMProvider)

    async def fake_generate(prompt, response_model, **kwargs):
        data = response_model(
            overview=["updated profile"],
            hotspots=[{"path": "src/auth.py", "reason": "auth logic changed"}],
        )
        return LLMResponse(data=data, input_tokens=1, output_tokens=1, latency_ms=1.0, model="fake")

    provider.generate_structured = fake_generate
    settings = get_settings(cache={"repo_profile_history_enabled": True})
    current = RepoProfile(repo="repo-url", last_ref="v1.0", overview=["before"])

    updated = await update_repo_profile_after_scan(
        db,
        provider,
        settings,
        repo="repo-url",
        from_ref="v1.0",
        to_ref="v1.1",
        current_profile=current,
        triage_stats=TriageStats(total_chunks=1, security_relevant=1, filtered_out=0),
        relevant_pairs=[
            (_chunk(), TriageResult(chunk_file_path="src/auth.py", is_security_relevant=True))
        ],
        findings=[],
    )
    assert updated is not None
    assert updated.last_ref == "v1.1"
    assert updated.hotspots

    stats = await db.get_cache_stats()
    assert stats["repo_profiles"] == 1
    assert stats["repo_profile_history"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_update_profile_error_keeps_previous(tmp_path):
    db = CacheStorage(tmp_path / "cache.db")
    await db.initialize()
    provider = AsyncMock(spec=LLMProvider)
    provider.generate_structured = AsyncMock(side_effect=LLMProviderError("boom"))
    settings = get_settings()
    current = RepoProfile(repo="repo-url", last_ref="v1.0", overview=["before"])

    updated = await update_repo_profile_after_scan(
        db,
        provider,
        settings,
        repo="repo-url",
        from_ref="v1.0",
        to_ref="v1.1",
        current_profile=current,
        triage_stats=None,
        relevant_pairs=[],
        findings=[],
    )
    assert updated is not None
    assert updated.overview == ["before"]
    await db.close()


@pytest.mark.asyncio
async def test_bootstrap_fallback_when_provider_fails(tmp_path):
    db = CacheStorage(tmp_path / "cache.db")
    await db.initialize()
    provider = AsyncMock(spec=LLMProvider)
    provider.generate_structured = AsyncMock(side_effect=LLMProviderError("llm unavailable"))
    settings = get_settings()

    profile = await load_or_bootstrap_repo_profile(
        db,
        provider,
        settings,
        _mock_git_source(),
        "repo-url",
        "v1.2",
    )
    assert profile is not None
    assert profile.modules
    cached = await db.get_repo_profile("repo-url")
    assert cached is not None
    await db.close()


@pytest.mark.asyncio
async def test_bootstrap_disabled_by_cache_flags(tmp_path):
    db = CacheStorage(tmp_path / "cache.db")
    await db.initialize()
    provider = AsyncMock(spec=LLMProvider)

    disabled_cache = get_settings(cache={"enabled": False})
    profile = await load_or_bootstrap_repo_profile(
        db,
        provider,
        disabled_cache,
        _mock_git_source(),
        "repo-url",
        "v1.2",
    )
    assert profile is None

    disabled_profile = get_settings(cache={"repo_profile_enabled": False})
    profile = await load_or_bootstrap_repo_profile(
        db,
        provider,
        disabled_profile,
        _mock_git_source(),
        "repo-url",
        "v1.2",
    )
    assert profile is None
    provider.generate_structured.assert_not_called()
    await db.close()


@pytest.mark.asyncio
async def test_invalid_cached_profile_triggers_rebuild(tmp_path):
    db = CacheStorage(tmp_path / "cache.db")
    await db.initialize()
    # Missing required RepoProfile fields -> should force rebuild.
    await db.set_repo_profile("repo-url", json.dumps({"broken": True}), "v0")

    provider = AsyncMock(spec=LLMProvider)

    async def fake_generate(prompt, response_model, **kwargs):
        data = response_model(overview=["rebuilt"])
        return LLMResponse(data=data, input_tokens=1, output_tokens=1, latency_ms=1.0, model="fake")

    provider.generate_structured = fake_generate
    settings = get_settings()

    profile = await load_or_bootstrap_repo_profile(
        db,
        provider,
        settings,
        _mock_git_source(),
        "repo-url",
        "v2.0",
    )
    assert profile is not None
    assert profile.overview == ["rebuilt"]
    await db.close()


def test_serialize_repo_profile_is_capped():
    profile = RepoProfile(
        repo="repo-url",
        last_ref="v1.0",
        overview=["x" * 200, "y" * 200, "z" * 200],
    )
    text = serialize_repo_profile_for_prompt(profile, max_chars=120)
    assert len(text) <= 120


def test_serialize_repo_profile_always_valid_json():
    profile = RepoProfile(
        repo="repo-url",
        last_ref="v1.0",
        overview=["x" * 200],
    )
    text = serialize_repo_profile_for_prompt(profile, max_chars=3)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)


def test_serialize_repo_profile_fallback_keeps_truncation_hint():
    profile = RepoProfile(
        repo="very-long-repo-name",
        last_ref="v1234567890",
        overview=["security profile"],
    )
    text = serialize_repo_profile_for_prompt(profile, max_chars=25)
    parsed = json.loads(text)
    assert parsed.get("truncated") is True
