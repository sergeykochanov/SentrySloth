"""SQLite storage backend for scan history and accumulated repo profiles."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS scan_history (
    scan_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    from_ref TEXT NOT NULL,
    to_ref TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repo_profiles (
    repo TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    last_ref TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repo_profile_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo TEXT NOT NULL,
    from_ref TEXT NOT NULL,
    to_ref TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scan_history_repo ON scan_history(repo);
CREATE INDEX IF NOT EXISTS idx_repo_profile_history_repo ON repo_profile_history(repo);
"""

# SQL query constants (extracted for line-length compliance)
_INSERT_SCAN = (
    "INSERT OR REPLACE INTO scan_history"
    " (scan_id, repo, from_ref, to_ref, result_json, created_at)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)
_INSERT_REPO_PROFILE = (
    "INSERT OR REPLACE INTO repo_profiles (repo, profile_json, last_ref, updated_at)"
    " VALUES (?, ?, ?, ?)"
)
_INSERT_REPO_PROFILE_HISTORY = (
    "INSERT INTO repo_profile_history (repo, from_ref, to_ref, profile_json, created_at)"
    " VALUES (?, ?, ?, ?, ?)"
)
_SELECT_SCANS = "SELECT scan_id, repo, from_ref, to_ref, created_at FROM scan_history"


class CacheStorage:
    """Async SQLite storage for sentrysloth cache."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._reset_if_schema_mismatch()
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA busy_timeout=5000;")
        await self._db.executescript(SCHEMA_SQL)
        await self._db.execute("DELETE FROM schema_version")
        await self._db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        await self._db.commit()

    async def _reset_if_schema_mismatch(self) -> None:
        if not self.db_path.exists():
            return

        schema_ok = False
        try:
            probe = await aiosqlite.connect(str(self.db_path))
            try:
                cursor = await probe.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
                )
                has_schema = await cursor.fetchone()
                if has_schema:
                    cursor = await probe.execute("SELECT version FROM schema_version LIMIT 1")
                    row = await cursor.fetchone()
                    schema_ok = bool(row and int(row[0]) == SCHEMA_VERSION)
            finally:
                await probe.close()
        except (aiosqlite.Error, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to probe cache schema at %s: %s", self.db_path, exc)
            schema_ok = False

        if schema_ok:
            return

        logger.warning("Resetting cache DB due to schema mismatch: %s", self.db_path)
        self.db_path.unlink(missing_ok=True)

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("CacheStorage not initialized. Call initialize() first.")
        return self._db

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> CacheStorage:
        await self.initialize()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # --- Scan history ---

    async def save_scan(
        self,
        scan_id: str,
        repo: str,
        from_ref: str,
        to_ref: str,
        result_json: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await self.db.execute(
            _INSERT_SCAN,
            (scan_id, repo, from_ref, to_ref, result_json, now),
        )
        await self.db.commit()

    async def get_scan(self, scan_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT result_json FROM scan_history WHERE scan_id=?",
            (scan_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            logger.error("Corrupted scan data for scan_id=%s", scan_id)
            return None

    async def list_scans(self, repo: str | None = None, limit: int = 20) -> list[dict]:
        if repo:
            sql = _SELECT_SCANS + " WHERE repo=? ORDER BY created_at DESC LIMIT ?"
            cursor = await self.db.execute(sql, (repo, limit))
        else:
            sql = _SELECT_SCANS + " ORDER BY created_at DESC LIMIT ?"
            cursor = await self.db.execute(sql, (limit,))
        rows = await cursor.fetchall()
        return [
            {
                "scan_id": r[0],
                "repo": r[1],
                "from_ref": r[2],
                "to_ref": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    # --- Repo profile ---

    async def get_repo_profile(self, repo: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT profile_json FROM repo_profiles WHERE repo=?",
            (repo,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            logger.error("Corrupted repo profile JSON for repo=%s", repo)
            return None

    async def set_repo_profile(self, repo: str, profile_json: str, last_ref: str) -> None:
        now = datetime.now(UTC).isoformat()
        await self.db.execute(_INSERT_REPO_PROFILE, (repo, profile_json, last_ref, now))
        await self.db.commit()

    async def append_repo_profile_history(
        self,
        repo: str,
        from_ref: str,
        to_ref: str,
        profile_json: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await self.db.execute(
            _INSERT_REPO_PROFILE_HISTORY,
            (repo, from_ref, to_ref, profile_json, now),
        )
        await self.db.commit()

    # --- Cache info ---

    _COUNT_QUERIES: ClassVar[dict[str, str]] = {
        "scan_history": "SELECT COUNT(*) FROM scan_history",
        "repo_profiles": "SELECT COUNT(*) FROM repo_profiles",
        "repo_profile_history": "SELECT COUNT(*) FROM repo_profile_history",
    }

    async def get_cache_stats(self) -> dict:
        stats: dict[str, int] = {}
        for table, query in self._COUNT_QUERIES.items():
            cursor = await self.db.execute(query)
            row = await cursor.fetchone()
            stats[table] = row[0] if row else 0
        return stats
