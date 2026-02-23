"""Scan orchestration: extract diffs, triage, analyse, post-process."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from sentrysloth.analyzers.context_builder import ContextBuilder
from sentrysloth.analyzers.deep_analysis import run_deep_analysis
from sentrysloth.analyzers.diff_extractor import extract_chunks
from sentrysloth.analyzers.triage import run_triage
from sentrysloth.cache.repo_profile import (
    load_or_bootstrap_repo_profile,
    serialize_repo_profile_for_prompt,
    update_repo_profile_after_scan,
)
from sentrysloth.cache.storage import CacheStorage
from sentrysloth.config import QuotaExhaustedMode, Settings
from sentrysloth.models import (
    CONFIDENCE_ORDER,
    SEVERITY_ORDER,
    Baseline,
    DiffChunk,
    Finding,
    LLMMetrics,
    ReleaseInfo,
    RepoProfile,
    ScanResult,
    TriageResult,
    TriageStats,
)
from sentrysloth.providers import create_provider
from sentrysloth.providers.base import LLMQuotaExceededError
from sentrysloth.providers.scheduler import LlmRequestScheduler
from sentrysloth.sources.git import GitSource, GitSourceError

logger = logging.getLogger(__name__)

# Exit codes — shared source of truth for CLI and batch
EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2
EXIT_INCOMPLETE = 3

# Callback type for informational / progress messages emitted during a scan.
# ``phase`` updates are short status labels; ``info`` are verbose lines;
# ``error`` messages are always visible.
PhaseCallback = Callable[[str], None]
InfoCallback = Callable[[str], None]
ErrorCallback = Callable[[str], None]


class BaselineLoadError(Exception):
    pass


def _load_baseline(path: str) -> Baseline:
    """Load and validate baseline file. Raises BaselineLoadError on failure."""

    def _read() -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise BaselineLoadError(f"Baseline file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise BaselineLoadError(f"Invalid JSON in baseline file {path}: {exc}") from exc

    return Baseline.model_validate(_read())


async def _load_baseline_async(path: str) -> Baseline:
    return await asyncio.to_thread(_load_baseline, path)


# ---------------------------------------------------------------------------
# Scan phases
# ---------------------------------------------------------------------------


async def _extract_and_enrich(
    git_source: GitSource,
    from_ref: str,
    to_ref: str,
    settings: Settings,
    *,
    on_phase: PhaseCallback,
    on_info: InfoCallback,
    on_error: ErrorCallback,
) -> tuple[int, ReleaseInfo, list[DiffChunk]] | None:
    """Clone repo, get diff, extract and enrich chunks.

    Returns ``(exit_code, release, enriched_chunks)`` on success,
    or ``None`` when an early exit occurs (exit code already emitted).
    If the returned exit code is not ``EXIT_OK``, the caller should
    return it immediately.
    """
    try:
        await git_source.ensure_cloned()
    except GitSourceError as exc:
        on_error(f"[red]Error:[/red] {exc}")
        return EXIT_ERROR, ReleaseInfo(repo_url="", from_ref="", to_ref=""), []

    try:
        stat = await git_source.get_diff_stat(from_ref, to_ref)
    except GitSourceError as exc:
        on_error(f"[red]Error:[/red] {exc}")
        return EXIT_ERROR, ReleaseInfo(repo_url="", from_ref="", to_ref=""), []

    release = ReleaseInfo(
        repo_url=git_source.repo_url,
        from_ref=from_ref,
        to_ref=to_ref,
        total_files_changed=stat["files_changed"],
        total_additions=stat["additions"],
        total_deletions=stat["deletions"],
    )

    on_phase(f"{stat['files_changed']} files, +{stat['additions']}/-{stat['deletions']}")
    on_info(
        f"[bold]Scanning[/bold] {git_source.repo_url}\n"
        f"  {from_ref} -> {to_ref}\n"
        f"  {stat['files_changed']} files, "
        f"+{stat['additions']}/-{stat['deletions']}"
    )

    try:
        diff_text = await git_source.get_diff(from_ref, to_ref)
    except GitSourceError as exc:
        on_error(f"[red]Error:[/red] {exc}")
        return EXIT_ERROR, release, []

    if not diff_text.strip():
        on_info("[yellow]No changes found between refs.[/yellow]")
        return EXIT_OK, release, []

    chunks = extract_chunks(diff_text, settings)
    if not chunks:
        on_info("[yellow]No analyzable code changes found.[/yellow]")
        return EXIT_OK, release, []

    on_phase(f"{len(chunks)} chunks")
    on_info(f"  Extracted {len(chunks)} diff chunks")

    context_builder = ContextBuilder(git_source, to_ref)
    enriched = [await context_builder.enrich_chunk(chunk) for chunk in chunks]
    return None, release, enriched


async def _run_triage_phase(
    enriched: list[DiffChunk],
    scheduler: LlmRequestScheduler,
    settings: Settings,
    cache_store: CacheStorage | None,
    git_source: GitSource,
    repo: str,
    to_ref: str,
    *,
    on_phase: PhaseCallback,
    on_info: InfoCallback,
) -> tuple[
    list[tuple[DiffChunk, TriageResult]],
    TriageStats,
    LLMMetrics,
    RepoProfile | None,
    str,
]:
    """Run triage and optionally load/bootstrap repo profile.

    Returns (relevant_pairs, triage_stats, triage_metrics, repo_profile, repo_profile_text).
    """
    on_phase(f"Triage ({len(enriched)} chunks)...")
    on_info("[bold]Running triage...[/bold]")
    paired, triage_stats, triage_metrics = await run_triage(enriched, scheduler, settings)

    relevant = [(chunk, result) for chunk, result in paired if result.is_security_relevant]

    rel = triage_stats.security_relevant
    total = triage_stats.total_chunks
    on_phase(f"{rel}/{total} relevant")
    on_info(f"  {rel}/{total} chunks security-relevant")

    repo_profile: RepoProfile | None = None
    repo_profile_text = ""
    if settings.cache.enabled and settings.cache.repo_profile_enabled and cache_store is not None:
        repo_profile = await load_or_bootstrap_repo_profile(
            cache_store,
            scheduler,
            settings,
            git_source,
            repo,
            to_ref,
        )
        repo_profile_text = serialize_repo_profile_for_prompt(
            repo_profile,
            settings.cache.repo_profile_max_chars,
        )

    return relevant, triage_stats, triage_metrics, repo_profile, repo_profile_text


async def _run_analysis_phase(
    relevant: list[tuple[DiffChunk, TriageResult]],
    scheduler: LlmRequestScheduler,
    settings: Settings,
    repo: str,
    repo_profile_text: str,
    git_source: GitSource,
    from_ref: str,
    to_ref: str,
    *,
    on_phase: PhaseCallback,
    on_info: InfoCallback,
) -> tuple[list[Finding], LLMMetrics, bool]:
    """Run deep analysis. Returns (findings, analysis_metrics, scan_incomplete)."""
    if (
        settings.llm.quota_exhausted_mode == QuotaExhaustedMode.HEURISTIC_FALLBACK
        and scheduler.quota_error is not None
    ):
        on_info(
            "[yellow]Quota exhausted during triage; "
            "skipping deep analysis in heuristic mode.[/yellow]"
        )
        return [], LLMMetrics(), True

    on_phase(f"Analysis ({len(relevant)} chunks)...")
    on_info(f"[bold]Running deep analysis on {len(relevant)} chunks...[/bold]")
    findings, analysis_metrics = await run_deep_analysis(
        relevant,
        scheduler,
        settings,
        repo,
        project_summary=repo_profile_text,
        git_source=git_source,
        from_ref=from_ref,
        to_ref=to_ref,
    )
    return findings, analysis_metrics, False


async def _apply_post_processing(
    findings: list[Finding],
    settings: Settings,
    baseline_path: str | None,
    *,
    on_info: InfoCallback,
    on_error: ErrorCallback,
) -> tuple[list[Finding], int | None]:
    """Apply baseline suppression and confidence filter.

    Returns ``(filtered_findings, error_exit_code)``.
    ``error_exit_code`` is non-None only when baseline loading fails.
    """
    if baseline_path:
        try:
            baseline = await _load_baseline_async(baseline_path)
        except BaselineLoadError as exc:
            on_error(f"[red]Error:[/red] {exc}")
            return findings, EXIT_ERROR
        before = len(findings)
        findings = [f for f in findings if not baseline.is_suppressed(f.finding_id)]
        suppressed = before - len(findings)
        if suppressed:
            on_info(f"  Suppressed {suppressed} baseline findings")

    min_val = CONFIDENCE_ORDER[settings.min_confidence]
    findings = [f for f in findings if CONFIDENCE_ORDER.get(f.confidence, 0) >= min_val]
    return findings, None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_scan(
    repo: str,
    from_ref: str,
    to_ref: str,
    settings: Settings,
    baseline_path: str | None,
    *,
    scheduler: LlmRequestScheduler | None = None,
    cache_store: CacheStorage | None = None,
    on_phase: PhaseCallback = lambda _msg: None,
    on_info: InfoCallback = lambda _msg: None,
    on_error: ErrorCallback = lambda _msg: None,
    on_scheduler_info: InfoCallback = lambda _msg: None,
) -> tuple[int, ScanResult | None]:
    """Run a full security scan and return ``(exit_code, result)``.

    This is the core orchestration function.  It does **not** import from
    ``cli`` — all output is delivered via callbacks.
    """
    scan_id = uuid.uuid4().hex[:12]
    started_at = datetime.now(UTC)
    git_source = GitSource(repo, settings)

    # --- Extract & enrich ---
    early_exit, release, enriched = await _extract_and_enrich(
        git_source,
        from_ref,
        to_ref,
        settings,
        on_phase=on_phase,
        on_info=on_info,
        on_error=on_error,
    )
    if early_exit is not None:
        if early_exit != EXIT_OK:
            return early_exit, None
        result = ScanResult(
            scan_id=scan_id,
            release=release,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            prompt_version=settings.prompt_version,
        )
        return EXIT_OK, result

    # --- Scheduler ---
    owns_scheduler = scheduler is None
    if owns_scheduler:
        try:
            provider = create_provider(settings)
        except ValueError as exc:
            on_error(f"[red]Error:[/red] {exc}")
            return EXIT_ERROR, None
        scheduler = LlmRequestScheduler(provider, settings.llm)
        await scheduler.start()

    if owns_scheduler:
        on_scheduler_info(
            "  LLM scheduler: "
            f"workers={settings.llm.scheduler_workers}, "
            f"queue_max={settings.llm.queue_max_size}, "
            f"rpm={settings.llm.max_requests_per_minute}, "
            f"tpm={settings.llm.max_tokens_per_minute or 'off'}, "
            f"quota_mode={settings.llm.quota_exhausted_mode}"
        )

    scan_incomplete = False

    try:
        # --- Triage ---
        (
            relevant,
            triage_stats,
            triage_metrics,
            repo_profile,
            repo_profile_text,
        ) = await _run_triage_phase(
            enriched,
            scheduler,
            settings,
            cache_store,
            git_source,
            repo,
            to_ref,
            on_phase=on_phase,
            on_info=on_info,
        )

        if not relevant:
            on_info("[green]No security-relevant changes detected.[/green]")
            result = ScanResult(
                scan_id=scan_id,
                release=release,
                triage_stats=triage_stats,
                llm_metrics=triage_metrics,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                prompt_version=settings.prompt_version,
            )
            if settings.cache.enabled and cache_store is not None:
                await cache_store.save_scan(
                    scan_id,
                    repo,
                    from_ref,
                    to_ref,
                    result.model_dump_json(),
                )
                if repo_profile is not None:
                    repo_profile.last_ref = to_ref
                    await cache_store.set_repo_profile(
                        repo,
                        repo_profile.model_dump_json(),
                        to_ref,
                    )
            return EXIT_OK, result

        # --- Deep analysis ---
        findings, analysis_metrics, incomplete = await _run_analysis_phase(
            relevant,
            scheduler,
            settings,
            repo,
            repo_profile_text,
            git_source,
            from_ref,
            to_ref,
            on_phase=on_phase,
            on_info=on_info,
        )
        if incomplete:
            scan_incomplete = True

        merged_metrics = triage_metrics.merge(analysis_metrics)

        # --- Post-processing ---
        findings, baseline_err = await _apply_post_processing(
            findings,
            settings,
            baseline_path,
            on_info=on_info,
            on_error=on_error,
        )
        if baseline_err is not None:
            return baseline_err, None

        result = ScanResult(
            scan_id=scan_id,
            release=release,
            findings=findings,
            triage_stats=triage_stats,
            llm_metrics=merged_metrics,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            prompt_version=settings.prompt_version,
        )

        # --- Cache ---
        if settings.cache.enabled and cache_store is not None:
            await cache_store.save_scan(
                scan_id,
                repo,
                from_ref,
                to_ref,
                result.model_dump_json(),
            )
            if repo_profile is not None and settings.cache.repo_profile_enabled:
                updated_profile = await update_repo_profile_after_scan(
                    cache_store,
                    scheduler,
                    settings,
                    repo=repo,
                    from_ref=from_ref,
                    to_ref=to_ref,
                    current_profile=repo_profile,
                    triage_stats=triage_stats,
                    relevant_pairs=relevant,
                    findings=findings,
                )
                if updated_profile is not None:
                    repo_profile = updated_profile

        if scheduler.quota_error is not None:
            scan_incomplete = True

        # --- Exit code ---
        if scan_incomplete:
            on_info(
                "[yellow]Scan completed with degraded coverage "
                "due to LLM quota limitations.[/yellow]"
            )
            return EXIT_INCOMPLETE, result

        if settings.fail_on_severity is not None:
            threshold_val = SEVERITY_ORDER.get(settings.fail_on_severity, 0)
            has_severe = any(SEVERITY_ORDER.get(f.severity, 0) >= threshold_val for f in findings)
            if has_severe:
                return EXIT_FINDINGS, result

        return EXIT_OK, result

    except LLMQuotaExceededError as exc:
        on_error("[red]Quota exhausted:[/red] unable to complete scan.")
        if exc.quota_metric:
            on_error(f"  metric: {exc.quota_metric}")
        if exc.is_daily_quota:
            on_error("  quota_type: daily")
        if exc.retry_delay_seconds is not None:
            on_error(f"  suggested retry delay: {exc.retry_delay_seconds:.1f}s")
        on_error(
            "  recommendation: reduce scan scope or concurrency, or upgrade your LLM API tier."
        )
        return EXIT_INCOMPLETE, None

    finally:
        if owns_scheduler:
            stats = await scheduler.get_stats()
            if settings.verbose:
                avg_wait = stats.queue_wait_seconds_total / max(1, stats.dequeued)
                on_scheduler_info(
                    "Scheduler stats: "
                    f"enqueued={stats.enqueued}, "
                    f"dequeued={stats.dequeued}, "
                    f"dropped={stats.dropped}, "
                    f"quota_short_circuited={stats.quota_short_circuited}, "
                    f"avg_queue_wait={avg_wait:.2f}s"
                )
            await scheduler.close()
