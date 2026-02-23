"""Tests for CacheStorage SQLite backend."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from sentrysloth.cache.storage import SCHEMA_VERSION, CacheStorage


@pytest.fixture
async def storage(tmp_path: Path):
    db_path = tmp_path / "test_cache.db"
    store = CacheStorage(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_db_init_creates_tables(storage: CacheStorage):
    """Schema tables exist after initialize()."""
    cursor = await storage.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    rows = await cursor.fetchall()
    table_names = {r[0] for r in rows}
    assert "repo_profiles" in table_names
    assert "repo_profile_history" in table_names
    assert "scan_history" in table_names
    assert "schema_version" in table_names


@pytest.mark.asyncio
async def test_store_and_retrieve_repo_profile(storage: CacheStorage):
    """Round-trip for repo profile JSON."""
    payload = {"overview": ["A web framework"], "hotspots": []}
    await storage.set_repo_profile("repo1", json.dumps(payload), "v1.0")
    result = await storage.get_repo_profile("repo1")
    assert result == payload


@pytest.mark.asyncio
async def test_cache_miss_returns_none(storage: CacheStorage):
    """Non-existent keys return None."""
    assert await storage.get_repo_profile("no") is None
    assert await storage.get_scan("nonexistent-id") is None


@pytest.mark.asyncio
async def test_save_and_get_scan(storage: CacheStorage):
    """Round-trip for scan history."""
    payload = {"findings": [], "status": "ok"}
    await storage.save_scan("scan-001", "repo1", "v1.0", "v1.1", json.dumps(payload))
    result = await storage.get_scan("scan-001")
    assert result == payload


@pytest.mark.asyncio
async def test_list_scans_filtered_by_repo(storage: CacheStorage):
    """list_scans filters by repo when provided."""
    await storage.save_scan("s1", "repo1", "a", "b", "{}")
    await storage.save_scan("s2", "repo2", "c", "d", "{}")
    await storage.save_scan("s3", "repo1", "e", "f", "{}")

    all_scans = await storage.list_scans()
    assert len(all_scans) == 3

    repo1_scans = await storage.list_scans(repo="repo1")
    assert len(repo1_scans) == 2
    assert all(s["repo"] == "repo1" for s in repo1_scans)


@pytest.mark.asyncio
async def test_get_cache_stats(storage: CacheStorage):
    """Stats reflect stored data."""
    await storage.save_scan("id1", "r", "a", "b", "{}")
    await storage.set_repo_profile("r", json.dumps({"overview": ["x"]}), "v1")
    stats = await storage.get_cache_stats()
    assert stats["scan_history"] == 1
    assert stats["repo_profiles"] == 1
    assert "file_summaries" not in stats


@pytest.mark.asyncio
async def test_concurrent_writes(storage: CacheStorage):
    """Parallel stores don't raise or lose data."""

    async def write(i: int) -> None:
        await storage.save_scan(f"scan-{i}", "repo", "a", f"b{i}", "{}")

    await asyncio.gather(*(write(i) for i in range(20)))
    stats = await storage.get_cache_stats()
    assert stats["scan_history"] == 20


@pytest.mark.asyncio
async def test_close_and_reopen(tmp_path: Path):
    """Data persists across close/reopen."""
    db_path = tmp_path / "persist.db"
    store = CacheStorage(db_path)
    await store.initialize()
    await store.save_scan("scan-persist", "repo", "v1.0", "v1.1", '{"status":"ok"}')
    await store.close()

    store2 = CacheStorage(db_path)
    await store2.initialize()
    result = await store2.get_scan("scan-persist")
    await store2.close()
    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_corrupted_scan_json_returns_none(storage: CacheStorage):
    """get_scan returns None for corrupted JSON."""
    await storage.save_scan("bad", "r", "a", "b", "not-json{{{")
    result = await storage.get_scan("bad")
    assert result is None


@pytest.mark.asyncio
async def test_initialize_resets_old_schema(tmp_path: Path):
    """Schema mismatch triggers database reset."""
    db_path = tmp_path / "mismatch.db"
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO schema_version (version) VALUES (1)")
        con.execute("CREATE TABLE project_summaries (repo TEXT PRIMARY KEY, summary TEXT)")
        con.commit()
    finally:
        con.close()

    store = CacheStorage(db_path)
    await store.initialize()
    try:
        cursor = await store.db.execute("SELECT version FROM schema_version")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == SCHEMA_VERSION

        cursor = await store.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cursor.fetchall()
        names = {r[0] for r in rows}
        assert "repo_profiles" in names
        assert "project_summaries" not in names
    finally:
        await store.close()
