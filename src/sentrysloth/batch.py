"""Batch scanning: load repo lists, build tag pairs, orchestrate scans."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sentrysloth.config import Settings
from sentrysloth.sources.git import GitSource, GitSourceError, extract_repo_name

logger = logging.getLogger(__name__)

# Type alias for the scan function injected by the CLI layer.
# Signature: (repo, from_ref, to_ref, settings, output_format, baseline, output_file) -> exit_code
ScanFn = Callable[[str, str, str, Settings, str, str | None, str | None], Awaitable[int]]


class BatchError(Exception):
    pass


_TAG_VERSION_RE = re.compile(
    r"(?:^|[-_/])v?(?P<major>0|[1-9]\d*)"
    r"(?:(?:\.|_)(?P<minor>\d+))?"
    r"(?:(?:\.|_)(?P<patch>\d+))?"
    r"(?:(?P<pre>a|b|rc)(?P<pre_num>\d+))?$",
    re.IGNORECASE,
)

PER_MAJOR_TAG_FETCH_LIMIT = 2000
DEFAULT_CHRONOLOGICAL_TAG_FETCH_LIMIT = 100


def parse_version(tag: str) -> tuple[int, tuple[int, ...], bool] | None:
    """Parse a version-like tag into a sortable key.

    Supported forms (optionally with common prefixes like ``release-``/``v``):
      - ``v1.2.3``, ``1.2.3``, ``1.26.19``
      - ``6.0rc1``, ``2.0b2``, ``1.0a1``

    Returns:
      (major, version_key, is_prerelease) or None if not parseable.

    Notes:
      - Missing minor/patch are treated as 0.
      - Sorting is semver-ish: prerelease < final for the same X.Y.Z
        (a < b < rc < final).
    """
    m = _TAG_VERSION_RE.search(tag.strip())
    if not m:
        return None

    major = int(m.group("major"))
    minor = int(m.group("minor") or 0)
    patch = int(m.group("patch") or 0)

    pre = m.group("pre")
    if pre:
        is_prerelease = True
        pre_kind = pre.lower()
        pre_rank = {"a": 0, "b": 1, "rc": 2}[pre_kind]
        pre_num = int(m.group("pre_num") or 0)
    else:
        is_prerelease = False
        pre_rank = 3
        pre_num = 0

    # Include major in the key for deterministic ordering.
    version_key: tuple[int, ...] = (major, minor, patch, pre_rank, pre_num)
    return major, version_key, is_prerelease


@dataclass(frozen=True)
class TagPair:
    from_ref: str
    to_ref: str


@dataclass
class ScanOutcome:
    repo: str
    repo_name: str
    pair: TagPair
    exit_code: int
    output_file: str
    error: str = ""
    duration_seconds: float = 0.0


@dataclass
class BatchResult:
    started_at: datetime
    completed_at: datetime | None = None
    outcomes: list[ScanOutcome] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (repo, reason)


def load_repo_list(path: Path) -> list[str]:
    """Load repo URLs from text file. One URL per line, # comments, blank lines ignored."""
    lines = path.read_text(encoding="utf-8").splitlines()
    repos: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        repos.append(stripped)
    if not repos:
        raise BatchError(f"No repos found in {path}")
    return repos


def repo_name_from_url(url: str) -> str:
    """Extract 'flask' from 'https://github.com/pallets/flask'."""
    return extract_repo_name(url)


def resolve_tag_fetch_limit(
    pairing_mode: Literal["chronological", "per_major"],
    last_releases: int | None,
) -> int:
    """Select tag fetch limit based on pairing strategy and filters."""
    if last_releases:
        # latest-major-first selection may need tags from multiple major streams.
        return PER_MAJOR_TAG_FETCH_LIMIT
    if pairing_mode == "per_major":
        return PER_MAJOR_TAG_FETCH_LIMIT
    return DEFAULT_CHRONOLOGICAL_TAG_FETCH_LIMIT


def normalize_since_for_tag_dates(
    since: datetime | None,
    tag_dates: list[datetime],
) -> datetime | None:
    """Make `since` timezone-aware when tag dates include timezone info."""
    if since is not None and since.tzinfo is None and tag_dates and tag_dates[0].tzinfo is not None:
        return since.replace(tzinfo=UTC)
    return since


def build_tag_pairs(
    tags_newest_first: list[str],
    *,
    last_releases: int | None = None,
    since: datetime | None = None,
    tag_dates: list[datetime] | None = None,
    pairing_mode: Literal["chronological", "per_major"] = "chronological",
) -> list[TagPair]:
    """Build consecutive tag pairs from tag list.

    Args:
        tags_newest_first: Tags sorted newest-first (from list_tags).
        last_releases: Take N transitions by scanning the latest major stream
            first, then backfilling older major streams as needed.
        since: Only include transitions where to_tag date >= since.
        tag_dates: Parallel list of commit datetimes for each tag
                   (needed only when using --since).
        pairing_mode: Pairing strategy:
          - chronological: consecutive pairs in commit-date order
          - per_major: group tags by major version and build pairs within each group

    Returns list of TagPair ordered old->new according to pairing_mode.
    """
    if not tags_newest_first or len(tags_newest_first) < 2:
        return []

    if last_releases is not None:
        if last_releases < 1:
            return []

        groups: dict[int, list[tuple[tuple[int, ...], str]]] = defaultdict(list)
        for tag in tags_newest_first:
            parsed = parse_version(tag)
            if parsed is None:
                continue
            major, version_key, _is_prerelease = parsed
            groups[major].append((version_key, tag))

        if not groups:
            # Fallback for non-semver tags: keep legacy chronological behavior.
            selected = tags_newest_first[: last_releases + 1]
            selected.reverse()
            return [TagPair(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]

        remaining = last_releases
        pairs: list[TagPair] = []
        for major in sorted(groups, reverse=True):
            # Sort by semantic-ish key (ascending).
            ordered = [tag for _key, tag in sorted(groups[major], key=lambda x: x[0])]
            if len(ordered) < 2:
                continue

            major_pairs = [TagPair(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
            take = min(remaining, len(major_pairs))
            if take <= 0:
                break
            pairs.extend(major_pairs[-take:])
            remaining -= take
            if remaining == 0:
                break

        return pairs

    if pairing_mode == "per_major":
        if since is not None:
            if tag_dates is None:
                raise BatchError("tag_dates required when using --since")
            if len(tag_dates) != len(tags_newest_first):
                raise BatchError("tag_dates length must match tags_newest_first length")
            tags_for_pairing = [
                tag for tag, dt in zip(tags_newest_first, tag_dates, strict=True) if dt >= since
            ]
        else:
            tags_for_pairing = tags_newest_first

        groups: dict[int, list[tuple[tuple[int, ...], str]]] = defaultdict(list)
        for tag in tags_for_pairing:
            parsed = parse_version(tag)
            if parsed is None:
                continue
            major, version_key, _is_prerelease = parsed
            groups[major].append((version_key, tag))

        pairs: list[TagPair] = []
        for major in sorted(groups):
            ordered = [tag for _key, tag in sorted(groups[major], key=lambda x: x[0])]
            if len(ordered) < 2:
                continue
            pairs.extend(TagPair(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1))
        return pairs

    if pairing_mode != "chronological":
        raise BatchError(f"Unknown pairing_mode: {pairing_mode}")

    if since is not None:
        if tag_dates is None:
            raise BatchError("tag_dates required when using --since")
        if len(tag_dates) != len(tags_newest_first):
            raise BatchError("tag_dates length must match tags_newest_first length")

        # Filter tags whose date >= since, keep newest-first order
        filtered = [
            (tag, dt) for tag, dt in zip(tags_newest_first, tag_dates, strict=True) if dt >= since
        ]
        if len(filtered) < 2:
            return []

        # Reverse to chronological order
        filtered.reverse()
        return [TagPair(filtered[i][0], filtered[i + 1][0]) for i in range(len(filtered) - 1)]

    # No filter: build pairs from all tags
    all_tags = list(reversed(tags_newest_first))
    return [TagPair(all_tags[i], all_tags[i + 1]) for i in range(len(all_tags) - 1)]


def _sanitize_ref_for_filename(ref: str) -> str:
    """Replace characters unsafe for filenames."""
    return ref.replace("/", "_").replace("\\", "_")


async def run_batch_scan(
    repos: list[str],
    settings: Settings,
    output_dir: Path,
    output_format: str,
    scan_fn: ScanFn,
    *,
    last_releases: int | None = None,
    since: datetime | None = None,
    pairing_mode: Literal["chronological", "per_major"] = "chronological",
    concurrency: int = 1,
) -> BatchResult:
    """Run scans for all repos with repo-level parallelism.

    Two-phase approach:
      1. **Resolve** (sequential) — clone repos, list tags, build work items.
      2. **Scan** (parallel per repo, sequential pairs within repo).

    This guarantees deterministic old->new processing for each repository, so
    accumulated repository context from earlier pairs can be reused by later ones.

    Args:
        repos: List of repo URLs to scan.
        settings: Global settings.
        output_dir: Base directory for reports.
        output_format: Output format string (json, markdown, sarif).
        scan_fn: Async scan function injected by CLI (avoids circular import).
        last_releases: Number of recent release pairs to scan (latest major
            stream first, then backfill from older majors).
        since: Only scan releases after this date.
        concurrency: Maximum number of repositories scanned in parallel.
    """
    result = BatchResult(started_at=datetime.now(UTC))

    # --- Phase 1: Resolve (sequential) ---
    work_items_by_repo: dict[str, tuple[str, list[tuple[TagPair, str]]]] = {}

    for repo_url in repos:
        name = repo_name_from_url(repo_url)
        git_source = GitSource(repo_url, settings)

        try:
            await git_source.ensure_cloned()
        except GitSourceError as exc:
            result.skipped.append((repo_url, str(exc)))
            logger.error("Skipping %s: %s", repo_url, exc)
            continue

        # Get tags with dates.
        # latest-major-first selection may need many tags to backfill across streams.
        limit = resolve_tag_fetch_limit(pairing_mode, last_releases)
        tags_with_dates = await git_source.list_tags_with_dates(limit=limit)

        if len(tags_with_dates) < 2:
            result.skipped.append((repo_url, "fewer than 2 tags"))
            logger.warning("Skipping %s: fewer than 2 tags", repo_url)
            continue

        tags = [t[0] for t in tags_with_dates]
        tag_dates = [t[1] for t in tags_with_dates]

        # Make since timezone-aware if tag dates are timezone-aware
        effective_since = normalize_since_for_tag_dates(since, tag_dates)

        pairs = build_tag_pairs(
            tags,
            last_releases=last_releases,
            since=effective_since,
            tag_dates=tag_dates if since is not None else None,
            pairing_mode=pairing_mode,
        )

        if not pairs:
            result.skipped.append((repo_url, "no tag pairs matched filters"))
            continue

        # Build work items for this repo
        repo_dir = output_dir / name
        repo_dir.mkdir(parents=True, exist_ok=True)

        repo_items: list[tuple[TagPair, str]] = []
        for pair in pairs:
            from_safe = _sanitize_ref_for_filename(pair.from_ref)
            to_safe = _sanitize_ref_for_filename(pair.to_ref)
            ext = "md" if output_format == "markdown" else output_format
            output_file = str(repo_dir / f"{from_safe}_to_{to_safe}.{ext}")
            repo_items.append((pair, output_file))
        work_items_by_repo[repo_url] = (name, repo_items)

    # --- Phase 2: Scan (parallel across repos, sequential pairs per repo) ---
    sem = asyncio.Semaphore(concurrency)

    async def _scan_repo(
        repo_url: str,
        name: str,
        repo_items: list[tuple[TagPair, str]],
    ) -> list[ScanOutcome]:
        async with sem:
            outcomes: list[ScanOutcome] = []
            for pair, output_file in repo_items:
                t0 = time.monotonic()
                try:
                    exit_code = await scan_fn(
                        repo_url,
                        pair.from_ref,
                        pair.to_ref,
                        settings,
                        output_format,
                        None,  # baseline
                        output_file,
                    )
                    duration = time.monotonic() - t0
                    outcomes.append(
                        ScanOutcome(
                            repo=repo_url,
                            repo_name=name,
                            pair=pair,
                            exit_code=exit_code,
                            output_file=output_file,
                            duration_seconds=duration,
                        )
                    )
                except Exception as exc:
                    duration = time.monotonic() - t0
                    logger.error(
                        "Scan failed for %s %s->%s: %s", name, pair.from_ref, pair.to_ref, exc
                    )
                    outcomes.append(
                        ScanOutcome(
                            repo=repo_url,
                            repo_name=name,
                            pair=pair,
                            exit_code=2,
                            output_file=output_file,
                            error=str(exc),
                            duration_seconds=duration,
                        )
                    )
            return outcomes

    if work_items_by_repo:
        grouped_outcomes = await asyncio.gather(
            *[
                _scan_repo(repo_url, name, repo_items)
                for repo_url, (name, repo_items) in work_items_by_repo.items()
            ]
        )
        result.outcomes = [outcome for group in grouped_outcomes for outcome in group]

    result.completed_at = datetime.now(UTC)
    return result
