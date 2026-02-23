"""Tests for batch scanning module."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from git import Repo
from typer.testing import CliRunner

from sentrysloth.batch import (
    DEFAULT_CHRONOLOGICAL_TAG_FETCH_LIMIT,
    PER_MAJOR_TAG_FETCH_LIMIT,
    BatchError,
    TagPair,
    build_tag_pairs,
    load_repo_list,
    normalize_since_for_tag_dates,
    repo_name_from_url,
    resolve_tag_fetch_limit,
    run_batch_scan,
)
from sentrysloth.cli import app
from sentrysloth.config import get_settings
from sentrysloth.sources.git import GitSource
from tests.conftest import strip_ansi

# --- load_repo_list ---


class TestLoadRepoList:
    def test_loads_urls(self, tmp_path: Path):
        f = tmp_path / "repos.txt"
        f.write_text(
            "https://github.com/psf/requests\nhttps://github.com/pallets/flask\n",
            encoding="utf-8",
        )
        result = load_repo_list(f)
        assert result == [
            "https://github.com/psf/requests",
            "https://github.com/pallets/flask",
        ]

    def test_ignores_comments_and_blanks(self, tmp_path: Path):
        f = tmp_path / "repos.txt"
        f.write_text(
            "# header comment\n"
            "\n"
            "https://github.com/psf/requests\n"
            "  # indented comment\n"
            "\n"
            "https://github.com/pallets/flask\n",
            encoding="utf-8",
        )
        result = load_repo_list(f)
        assert len(result) == 2

    def test_strips_whitespace(self, tmp_path: Path):
        f = tmp_path / "repos.txt"
        f.write_text("  https://github.com/psf/requests  \n", encoding="utf-8")
        result = load_repo_list(f)
        assert result == ["https://github.com/psf/requests"]

    def test_empty_file_raises(self, tmp_path: Path):
        f = tmp_path / "repos.txt"
        f.write_text("# only comments\n\n", encoding="utf-8")
        with pytest.raises(BatchError, match="No repos found"):
            load_repo_list(f)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(OSError):
            load_repo_list(tmp_path / "nonexistent.txt")


# --- repo_name_from_url ---


class TestRepoNameFromUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://github.com/pallets/flask", "flask"),
            ("https://github.com/pallets/flask/", "flask"),
            ("https://github.com/pallets/flask.git", "flask"),
            ("git@github.com:psf/requests.git", "requests"),
            ("/local/path/to/myrepo", "myrepo"),
        ],
    )
    def test_extracts_name(self, url: str, expected: str):
        assert repo_name_from_url(url) == expected


class TestBatchHelpers:
    def test_resolve_tag_fetch_limit_per_major(self):
        assert resolve_tag_fetch_limit("per_major", 2) == PER_MAJOR_TAG_FETCH_LIMIT

    def test_resolve_tag_fetch_limit_last_releases(self):
        assert resolve_tag_fetch_limit("chronological", 3) == PER_MAJOR_TAG_FETCH_LIMIT

    def test_resolve_tag_fetch_limit_default(self):
        assert (
            resolve_tag_fetch_limit("chronological", None) == DEFAULT_CHRONOLOGICAL_TAG_FETCH_LIMIT
        )

    def test_normalize_since_for_tag_dates_adds_utc(self):
        since = datetime(2026, 1, 1)  # naive
        tag_dates = [datetime(2026, 1, 2, tzinfo=UTC)]
        normalized = normalize_since_for_tag_dates(since, tag_dates)
        assert normalized is not None
        assert normalized.tzinfo is UTC

    def test_normalize_since_for_tag_dates_keeps_existing_tz(self):
        since = datetime(2026, 1, 1, tzinfo=UTC)
        tag_dates = [datetime(2026, 1, 2, tzinfo=UTC)]
        assert normalize_since_for_tag_dates(since, tag_dates) == since


# --- build_tag_pairs ---


class TestBuildTagPairs:
    def test_last_releases_basic(self):
        tags = ["v3.2.0", "v3.1.0", "v3.0.0", "v2.2.0", "v2.1.0", "v2.0.0", "v1.0.0"]
        pairs = build_tag_pairs(tags, last_releases=3)
        assert pairs == [
            TagPair("v3.0.0", "v3.1.0"),
            TagPair("v3.1.0", "v3.2.0"),
            TagPair("v2.1.0", "v2.2.0"),
        ]

    def test_last_releases_exceeds_tags(self):
        tags = ["v2.1.0", "v2.0.0", "v1.1.0", "v1.0.0"]
        pairs = build_tag_pairs(tags, last_releases=10)
        assert pairs == [
            TagPair("v2.0.0", "v2.1.0"),
            TagPair("v1.0.0", "v1.1.0"),
        ]

    def test_last_releases_one(self):
        tags = ["v3.2.0", "v3.1.0", "v3.0.0", "v2.1.0", "v2.0.0"]
        pairs = build_tag_pairs(tags, last_releases=1)
        assert pairs == [TagPair("v3.1.0", "v3.2.0")]

    def test_last_releases_modes_match(self):
        tags = ["v3.2.0", "v3.1.0", "v3.0.0", "v2.2.0", "v2.1.0", "v2.0.0"]
        chronological = build_tag_pairs(tags, last_releases=3, pairing_mode="chronological")
        per_major = build_tag_pairs(tags, last_releases=3, pairing_mode="per_major")
        assert chronological == per_major

    def test_last_releases_falls_back_for_unparseable_tags(self):
        tags = ["stable", "beta", "alpha"]
        pairs = build_tag_pairs(tags, last_releases=2)
        assert pairs == [TagPair("alpha", "beta"), TagPair("beta", "stable")]

    def test_since_filters_by_date(self):
        now = datetime(2026, 2, 1, tzinfo=UTC)
        tags = ["v5", "v4", "v3", "v2", "v1"]
        dates = [
            now,
            now - timedelta(days=10),
            now - timedelta(days=20),
            now - timedelta(days=40),
            now - timedelta(days=60),
        ]
        cutoff = now - timedelta(days=25)

        pairs = build_tag_pairs(tags, since=cutoff, tag_dates=dates)
        # v3 (day -20), v4 (day -10), v5 (now) pass the filter
        assert pairs == [TagPair("v3", "v4"), TagPair("v4", "v5")]

    def test_since_no_dates_raises(self):
        with pytest.raises(BatchError, match="tag_dates required"):
            build_tag_pairs(["v2", "v1"], since=datetime(2026, 1, 1, tzinfo=UTC))

    def test_since_mismatched_lengths_raises(self):
        with pytest.raises(BatchError, match="length must match"):
            build_tag_pairs(
                ["v2", "v1"],
                since=datetime(2026, 1, 1, tzinfo=UTC),
                tag_dates=[datetime(2026, 1, 1, tzinfo=UTC)],
            )

    def test_since_too_few_matching_returns_empty(self):
        now = datetime(2026, 2, 1, tzinfo=UTC)
        tags = ["v2", "v1"]
        dates = [now - timedelta(days=10), now - timedelta(days=60)]
        pairs = build_tag_pairs(tags, since=now, tag_dates=dates)
        assert pairs == []

    def test_empty_tags_returns_empty(self):
        assert build_tag_pairs([]) == []

    def test_single_tag_returns_empty(self):
        assert build_tag_pairs(["v1"]) == []

    def test_no_filter_builds_all_pairs(self):
        tags = ["v3", "v2", "v1"]
        pairs = build_tag_pairs(tags)
        assert pairs == [TagPair("v1", "v2"), TagPair("v2", "v3")]

    def test_per_major_no_cross_major_pairs(self):
        tags = ["v2.0.0", "v1.26.20", "v1.26.19", "v2.1.0"]
        pairs = build_tag_pairs(tags, pairing_mode="per_major")
        assert pairs == [
            TagPair("v1.26.19", "v1.26.20"),
            TagPair("v2.0.0", "v2.1.0"),
        ]

    def test_per_major_prerelease_stays_in_major_stream(self):
        tags = ["5.2.9", "6.0", "6.0rc2", "6.0rc1"]
        pairs = build_tag_pairs(tags, pairing_mode="per_major")
        assert pairs == [
            TagPair("6.0rc1", "6.0rc2"),
            TagPair("6.0rc2", "6.0"),
        ]

    def test_per_major_last_releases_is_per_stream(self):
        tags = ["v2.1.0", "v1.26.20", "v1.26.19", "v2.0.0"]
        pairs = build_tag_pairs(tags, pairing_mode="per_major", last_releases=1)
        assert pairs == [TagPair("v2.0.0", "v2.1.0")]

    def test_per_major_parses_underscore_versions_like_sqlalchemy(self):
        tags = ["rel_1_1_0", "rel_1_0_0", "rel_0_9_0", "rel_0_8_0"]
        pairs = build_tag_pairs(tags, pairing_mode="per_major")
        assert pairs == [
            TagPair("rel_0_8_0", "rel_0_9_0"),
            TagPair("rel_1_0_0", "rel_1_1_0"),
        ]


# --- list_tags_with_dates ---


def _create_tagged_repo(path: Path, tag_names: list[str]) -> Repo:
    """Create a git repo with commits and tags for testing."""
    repo = Repo.init(path)
    for i, tag_name in enumerate(tag_names):
        f = path / f"file_{i}.txt"
        f.write_text(f"content {i}\n", encoding="utf-8")
        repo.index.add([f"file_{i}.txt"])
        repo.index.commit(f"commit {i}")
        repo.create_tag(tag_name)
    return repo


class TestListTagsWithDates:
    @pytest.mark.asyncio
    async def test_returns_tags_with_dates(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_tagged_repo(repo_dir, ["v1.0", "v1.1", "v1.2"])

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        result = await source.list_tags_with_dates(limit=10)
        assert len(result) == 3

        # All expected tags present
        names = {t[0] for t in result}
        assert names == {"v1.0", "v1.1", "v1.2"}

        # Each entry is (str, datetime), dates in descending (or equal) order
        for name, dt in result:
            assert isinstance(name, str)
            assert isinstance(dt, datetime)
        dates = [t[1] for t in result]
        assert dates == sorted(dates, reverse=True)

    @pytest.mark.asyncio
    async def test_limit_respected(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_tagged_repo(repo_dir, ["v1", "v2", "v3", "v4", "v5"])

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        result = await source.list_tags_with_dates(limit=2)
        assert len(result) == 2


# --- CLI smoke tests ---

runner = CliRunner()


class TestBatchScanCLI:
    def test_help_output(self):
        result = runner.invoke(app, ["batch-scan", "--help"])
        assert result.exit_code == 0
        out = strip_ansi(result.output)
        assert "--last-releases" in out
        assert "--since" in out
        assert "--dry-run" in out
        assert "--concurrency" in out
        assert "-j" in out

    def test_batch_scan_in_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert "batch-scan" in strip_ansi(result.output)

    def test_missing_filter_gives_error(self, tmp_path: Path):
        f = tmp_path / "repos.txt"
        f.write_text("https://github.com/psf/requests\n", encoding="utf-8")
        result = runner.invoke(app, ["batch-scan", str(f)])
        assert result.exit_code != 0

    def test_both_filters_gives_error(self, tmp_path: Path):
        f = tmp_path / "repos.txt"
        f.write_text("https://github.com/psf/requests\n", encoding="utf-8")
        result = runner.invoke(
            app, ["batch-scan", str(f), "--last-releases", "3", "--since", "2026-01-01"]
        )
        assert result.exit_code != 0

    def test_invalid_date_gives_error(self, tmp_path: Path):
        f = tmp_path / "repos.txt"
        f.write_text("https://github.com/psf/requests\n", encoding="utf-8")
        result = runner.invoke(app, ["batch-scan", str(f), "--since", "not-a-date"])
        assert result.exit_code != 0

    def test_missing_repos_file_gives_error(self, tmp_path: Path):
        result = runner.invoke(
            app, ["batch-scan", str(tmp_path / "nonexistent.txt"), "--last-releases", "3"]
        )
        assert result.exit_code != 0

    def test_empty_repos_file_gives_error(self, tmp_path: Path):
        f = tmp_path / "repos.txt"
        f.write_text("# only comments\n", encoding="utf-8")
        result = runner.invoke(app, ["batch-scan", str(f), "--last-releases", "3"])
        assert result.exit_code != 0

    def test_dry_run_with_local_repo(self, tmp_path: Path):
        """Dry run with a local repo should show planned scans without LLM calls."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_tagged_repo(repo_dir, ["v1.0", "v1.1", "v1.2"])

        f = tmp_path / "repos.txt"
        f.write_text(f"{repo_dir}\n", encoding="utf-8")

        result = runner.invoke(app, ["batch-scan", str(f), "--last-releases", "2", "--dry-run"])
        assert result.exit_code == 0
        out = strip_ansi(result.output)
        assert "v1.0" in out or "v1.1" in out

    def test_dry_run_not_affected_by_concurrency(self, tmp_path: Path):
        """Dry run with -j 2 should work identically to -j 1 (no progress bars)."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_tagged_repo(repo_dir, ["v1.0", "v1.1", "v1.2"])

        f = tmp_path / "repos.txt"
        f.write_text(f"{repo_dir}\n", encoding="utf-8")

        result = runner.invoke(
            app, ["batch-scan", str(f), "--last-releases", "2", "--dry-run", "-j", "2"]
        )
        assert result.exit_code == 0
        out = strip_ansi(result.output)
        assert "v1.0" in out or "v1.1" in out


# --- run_batch_scan concurrency ---


class TestRunBatchScanConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_limits_parallel_scans(self, tmp_path: Path):
        """Single repo pairs run sequentially even when concurrency > 1."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_tagged_repo(repo_dir, ["v1.0", "v1.1", "v1.2", "v1.3"])

        settings = get_settings(clone_base_dir=tmp_path / "clones")

        # Track concurrent execution
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_scan_fn(
            repo: str,
            from_ref: str,
            to_ref: str,
            s: object,
            fmt: str,
            baseline: str | None,
            output_file: str | None,
        ) -> int:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)  # simulate work
            async with lock:
                current_concurrent -= 1
            return 0

        output_dir = tmp_path / "reports"

        result = await run_batch_scan(
            [str(repo_dir)],
            settings,
            output_dir,
            "json",
            tracking_scan_fn,
            last_releases=3,
            concurrency=2,
        )

        # Per-repo sequencing: one repo cannot run multiple pairs concurrently.
        assert len(result.outcomes) == 3
        assert max_concurrent == 1
        assert all(o.exit_code == 0 for o in result.outcomes)

    @pytest.mark.asyncio
    async def test_concurrency_one_is_sequential(self, tmp_path: Path):
        """With concurrency=1, scans run one at a time (same as old behaviour)."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_tagged_repo(repo_dir, ["v1.0", "v1.1", "v1.2"])

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_scan_fn(
            repo: str,
            from_ref: str,
            to_ref: str,
            s: object,
            fmt: str,
            baseline: str | None,
            output_file: str | None,
        ) -> int:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return 0

        output_dir = tmp_path / "reports"

        result = await run_batch_scan(
            [str(repo_dir)],
            settings,
            output_dir,
            "json",
            tracking_scan_fn,
            last_releases=2,
            concurrency=1,
        )

        assert len(result.outcomes) == 2
        assert max_concurrent == 1

    @pytest.mark.asyncio
    async def test_scan_errors_captured_in_outcomes(self, tmp_path: Path):
        """Exceptions in scan_fn should be captured, not crash the batch."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_tagged_repo(repo_dir, ["v1.0", "v1.1", "v1.2"])

        settings = get_settings(clone_base_dir=tmp_path / "clones")

        async def failing_scan_fn(
            repo: str,
            from_ref: str,
            to_ref: str,
            s: object,
            fmt: str,
            baseline: str | None,
            output_file: str | None,
        ) -> int:
            raise RuntimeError("boom")

        output_dir = tmp_path / "reports"

        result = await run_batch_scan(
            [str(repo_dir)],
            settings,
            output_dir,
            "json",
            failing_scan_fn,
            last_releases=2,
            concurrency=4,
        )

        assert len(result.outcomes) == 2
        assert all(o.exit_code == 2 for o in result.outcomes)
        assert all("boom" in o.error for o in result.outcomes)

    @pytest.mark.asyncio
    async def test_two_repos_are_parallel_but_each_repo_order_is_sequential(self, tmp_path: Path):
        repo1 = tmp_path / "repo1"
        repo2 = tmp_path / "repo2"
        repo1.mkdir()
        repo2.mkdir()
        _create_tagged_repo(repo1, ["v1.0", "v1.1", "v1.2"])
        _create_tagged_repo(repo2, ["v2.0", "v2.1", "v2.2"])

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        call_order: dict[str, list[tuple[str, str]]] = {str(repo1): [], str(repo2): []}
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_scan_fn(
            repo: str,
            from_ref: str,
            to_ref: str,
            s: object,
            fmt: str,
            baseline: str | None,
            output_file: str | None,
        ) -> int:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                call_order[repo].append((from_ref, to_ref))
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return 0

        result = await run_batch_scan(
            [str(repo1), str(repo2)],
            settings,
            tmp_path / "reports",
            "json",
            tracking_scan_fn,
            last_releases=2,
            concurrency=2,
        )

        source1 = GitSource(str(repo1), settings)
        source2 = GitSource(str(repo2), settings)
        await source1.ensure_cloned()
        await source2.ensure_cloned()
        expected1 = [
            (p.from_ref, p.to_ref)
            for p in build_tag_pairs(
                [t[0] for t in await source1.list_tags_with_dates(limit=20)],
                last_releases=2,
            )
        ]
        expected2 = [
            (p.from_ref, p.to_ref)
            for p in build_tag_pairs(
                [t[0] for t in await source2.list_tags_with_dates(limit=20)],
                last_releases=2,
            )
        ]

        assert len(result.outcomes) == 4
        assert max_concurrent == 2
        assert call_order[str(repo1)] == expected1
        assert call_order[str(repo2)] == expected2
