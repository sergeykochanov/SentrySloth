"""Typer CLI: scan, list-versions, report, cache-info commands."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from sentrysloth import __version__
from sentrysloth.batch import (
    BatchError,
    BatchResult,
    build_tag_pairs,
    load_repo_list,
    normalize_since_for_tag_dates,
    repo_name_from_url,
    resolve_tag_fetch_limit,
    run_batch_scan,
)
from sentrysloth.cache.storage import CacheStorage
from sentrysloth.config import Settings, get_settings
from sentrysloth.models import (
    RepoProfile,
    ScanResult,
    Severity,
)
from sentrysloth.providers import create_provider
from sentrysloth.providers.scheduler import LlmRequestScheduler
from sentrysloth.reporters.json_reporter import generate_json_report
from sentrysloth.reporters.markdown_reporter import generate_markdown_report
from sentrysloth.reporters.sarif_reporter import generate_sarif_report
from sentrysloth.scanner import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, run_scan
from sentrysloth.sources.git import GitSource, GitSourceError

app = typer.Typer(
    name="sentrysloth",
    help="Change-risk / security-review assistant for open-source releases.",
    no_args_is_help=True,
)
console = Console(stderr=True)
output_console = Console()


class OutputFormat(StrEnum):
    JSON = "json"
    MARKDOWN = "markdown"
    SARIF = "sarif"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class ScanOutput:
    """Unified output for scan — verbose console or progress bar line."""

    def __init__(
        self,
        con: Console,
        label: str = "",
        *,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ):
        self._console = con
        self._label = label
        self._progress = progress
        self._task_id = task_id

    @property
    def quiet(self) -> bool:
        return self._progress is not None

    def phase(self, desc: str) -> None:
        """Update progress bar description (noop in verbose mode)."""
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=f"{self._label}: {desc}")

    def info(self, msg: str) -> None:
        """Print informational message (suppressed in progress mode)."""
        if not self.quiet:
            self._console.print(msg)

    def error(self, msg: str) -> None:
        """Print error (always visible, even above progress bars)."""
        self._console.print(msg)


class _CompletedProgressWindow:
    """Keep only the latest completed progress tasks visible."""

    def __init__(self, progress: Progress, *, max_completed: int = 5) -> None:
        self._progress = progress
        self._max_completed = max(1, max_completed)
        self._completed: deque[TaskID] = deque()
        self._seen: set[TaskID] = set()

    def mark_completed(self, task_id: TaskID) -> None:
        """Register a task as completed and evict oldest completed tasks."""
        if task_id in self._seen:
            return

        self._seen.add(task_id)
        self._completed.append(task_id)

        while len(self._completed) > self._max_completed:
            oldest = self._completed.popleft()
            self._seen.discard(oldest)
            self._progress.remove_task(oldest)


def _mark_batch_task_finished(
    progress: Progress,
    task_id: TaskID | None,
    *,
    scan_label: str,
    status_markup: str,
    completed_window: _CompletedProgressWindow | None,
) -> None:
    """Finalize task status and apply rolling visibility window."""
    if task_id is None:
        return

    progress.update(task_id, description=f"{scan_label}: {status_markup}", completed=1)
    if completed_window is not None:
        completed_window.mark_completed(task_id)


def _log_scheduler_stats(stats, con: Console, prefix: str = "") -> None:
    """Format and print scheduler stats."""
    avg_wait = stats.queue_wait_seconds_total / max(1, stats.dequeued)
    con.print(
        f"{prefix}Scheduler stats: "
        f"enqueued={stats.enqueued}, "
        f"dequeued={stats.dequeued}, "
        f"dropped={stats.dropped}, "
        f"quota_short_circuited={stats.quota_short_circuited}, "
        f"avg_queue_wait={avg_wait:.2f}s"
    )


# ---------------------------------------------------------------------------
# Core scan wrapper — delegates to scanner.run_scan()
# ---------------------------------------------------------------------------


async def _run_scan(
    repo: str,
    from_ref: str,
    to_ref: str,
    settings: Settings,
    output_format: OutputFormat,
    baseline_path: str | None,
    output_file: str | None,
    *,
    scheduler: LlmRequestScheduler | None = None,
    output: ScanOutput | None = None,
    cache: CacheStorage | None = None,
) -> int:
    """Run a scan, outputting results via ``output`` and writing reports."""
    if output is None:
        output = ScanOutput(console)

    owns_cache = False
    cache_store = cache
    if settings.cache.enabled and cache_store is None:
        cache_store = CacheStorage(settings.cache.db_path)
        await cache_store.initialize()
        owns_cache = True

    try:
        exit_code, result = await run_scan(
            repo,
            from_ref,
            to_ref,
            settings,
            baseline_path,
            scheduler=scheduler,
            cache_store=cache_store,
            on_phase=output.phase,
            on_info=output.info,
            on_error=output.error,
            on_scheduler_info=lambda msg: console.print(msg),
        )
        if result is not None:
            _output_result(result, output_format, output_file, quiet=output.quiet)
            owns_scheduler = scheduler is None
            _print_summary(result, show_empty=owns_scheduler)
        return exit_code
    finally:
        if owns_cache and cache_store is not None:
            await cache_store.close()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _output_result(
    result: ScanResult, fmt: OutputFormat, output_file: str | None, *, quiet: bool = False
) -> None:
    if fmt == OutputFormat.JSON:
        report = generate_json_report(result)
    elif fmt == OutputFormat.MARKDOWN:
        report = generate_markdown_report(result)
    elif fmt == OutputFormat.SARIF:
        report = generate_sarif_report(result)
    else:
        report = generate_json_report(result)

    if output_file:
        try:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report, encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Error writing report to {output_file}:[/red] {exc}")
            return
        if not quiet:
            console.print(f"Report saved to {output_file}")
    else:
        output_console.print(report)


def _print_summary(result: ScanResult, *, show_empty: bool = True) -> None:
    if not result.findings:
        if not show_empty:
            return
        repo = result.release.repo_url
        from_ref = result.release.from_ref
        to_ref = result.release.to_ref
        ctx = ""
        if repo and from_ref and to_ref:
            ctx = f" {repo} {from_ref}→{to_ref} (scan {result.scan_id})"
        elif repo:
            ctx = f" {repo} (scan {result.scan_id})"
        else:
            ctx = f" (scan {result.scan_id})"
        console.print(f"[green]No findings.[/green]{ctx}")
        return

    summary_title = "Findings Summary"
    repo = result.release.repo_url
    from_ref = result.release.from_ref
    to_ref = result.release.to_ref
    if repo and from_ref and to_ref:
        summary_title = f"Findings Summary: {repo_name_from_url(repo)} {from_ref}→{to_ref}"
    elif repo:
        summary_title = f"Findings Summary: {repo_name_from_url(repo)}"

    table = Table(title=summary_title)
    table.add_column("ID", style="dim")
    table.add_column("Severity")
    table.add_column("Confidence")
    table.add_column("Type")
    table.add_column("Title")
    table.add_column("File")

    severity_colors = {
        Severity.CRITICAL: "red bold",
        Severity.HIGH: "red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "blue",
        Severity.INFO: "dim",
    }

    for f in result.findings:
        color = severity_colors.get(f.severity, "white")
        table.add_row(
            f.finding_id,
            f"[{color}]{f.severity.value}[/{color}]",
            f.confidence.value,
            f.finding_type.value,
            f.title[:60],
            f.file_path,
        )

    console.print(table)


def _print_repo_profile(profile: RepoProfile) -> None:
    meta = Table(title=f"Repo Profile: {profile.repo}")
    meta.add_column("Field")
    meta.add_column("Value")
    meta.add_row("Last ref", profile.last_ref or "-")
    meta.add_row("Updated at", profile.updated_at.isoformat())
    meta.add_row("Overview items", str(len(profile.overview)))
    meta.add_row("Tech stack items", str(len(profile.tech_stack)))
    meta.add_row("Known risks", str(len(profile.known_risks)))
    console.print(meta)

    if profile.tech_stack:
        stack = Table(title="Tech Stack")
        stack.add_column("#", style="dim")
        stack.add_column("Library / Platform")
        for idx, item in enumerate(profile.tech_stack, 1):
            stack.add_row(str(idx), item)
        console.print(stack)
    else:
        console.print("[yellow]No tech stack data in cached repo profile yet.[/yellow]")

    if profile.overview:
        console.print("[bold]Overview[/bold]")
        for item in profile.overview[:5]:
            console.print(f"- {item}")
    if profile.known_risks:
        console.print("[bold]Known Risks[/bold]")
        for item in profile.known_risks[:5]:
            console.print(f"- {item}")


# Typer defaults use function calls in signatures (B008) — this is
# idiomatic for Typer and intentional.


@app.command()
def scan(
    repo: str = typer.Argument(help="Repository URL or local path"),
    from_ref: str = typer.Option(..., "--from", help="Start ref (tag, branch, commit)"),
    to_ref: str = typer.Option(..., "--to", help="End ref (tag, branch, commit)"),
    output: OutputFormat = typer.Option(  # noqa: B008
        OutputFormat.JSON, "--output", "-o", help="Output format"
    ),
    output_file: str | None = typer.Option(None, "--output-file", "-f", help="Save report to file"),
    fail_on: str | None = typer.Option(None, "--fail-on", help="Exit 1 if findings >= severity"),
    min_confidence: str = typer.Option("low", "--min-confidence", help="Minimum confidence filter"),
    baseline: str | None = typer.Option(None, "--baseline", help="Baseline file for suppression"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Scan a repository for security-relevant changes between two refs."""
    _setup_logging(verbose)

    overrides: dict = {
        "verbose": verbose,
        "min_confidence": min_confidence,
    }
    if fail_on:
        overrides["fail_on_severity"] = fail_on

    settings = get_settings(**overrides)

    exit_code = asyncio.run(
        _run_scan(
            repo,
            from_ref,
            to_ref,
            settings,
            output,
            baseline,
            output_file,
        )
    )
    raise typer.Exit(code=exit_code)


@app.command("list-versions")
def list_versions(
    repo: str = typer.Argument(help="Repository URL or local path"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of tags to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List recent tags/versions for a repository."""
    _setup_logging(verbose)
    settings = get_settings()

    async def _run() -> int:
        git_source = GitSource(repo, settings)
        try:
            await git_source.ensure_cloned()
        except GitSourceError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return EXIT_ERROR

        tags = await git_source.list_tags(limit=limit)
        if not tags:
            console.print("[yellow]No tags found.[/yellow]")
            return EXIT_OK

        table = Table(title=f"Tags for {repo}")
        table.add_column("#", style="dim")
        table.add_column("Tag")

        for i, tag in enumerate(tags, 1):
            table.add_row(str(i), tag)

        console.print(table)
        return EXIT_OK

    exit_code = asyncio.run(_run())
    raise typer.Exit(code=exit_code)


@app.command()
def report(
    scan_id: str = typer.Argument(help="Scan ID to retrieve"),
    output: OutputFormat = typer.Option(  # noqa: B008
        OutputFormat.JSON, "--output", "-o"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """View a previous scan result."""
    _setup_logging(verbose)
    settings = get_settings()

    async def _run() -> int:
        cache = CacheStorage(settings.cache.db_path)
        try:
            await cache.initialize()
            data = await cache.get_scan(scan_id)
            if not data:
                console.print(f"[red]Scan {scan_id} not found.[/red]")
                return EXIT_ERROR

            try:
                result = ScanResult.model_validate(data)
            except ValidationError:
                console.print(
                    f"[red]Scan {scan_id} has incompatible cached schema "
                    "and cannot be rendered.[/red]"
                )
                return EXIT_ERROR
            _output_result(result, output, None)
            return EXIT_OK
        finally:
            await cache.close()

    exit_code = asyncio.run(_run())
    raise typer.Exit(code=exit_code)


@app.command("cache-info")
def cache_info(
    repo: str | None = typer.Argument(None, help="Filter by repository URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show cache statistics."""
    _setup_logging(verbose)
    settings = get_settings()

    async def _run() -> int:
        cache = CacheStorage(settings.cache.db_path)
        try:
            await cache.initialize()

            stats = await cache.get_cache_stats()
            table = Table(title="Cache Statistics")
            table.add_column("Table")
            table.add_column("Entries", justify="right")

            for name, count in stats.items():
                table.add_row(name, str(count))
            console.print(table)

            scans = await cache.list_scans(repo=repo, limit=10)
            if scans:
                scan_table = Table(title="Recent Scans")
                scan_table.add_column("Scan ID")
                scan_table.add_column("Repository")
                scan_table.add_column("Refs")
                scan_table.add_column("Date")

                for s in scans:
                    scan_table.add_row(
                        s["scan_id"],
                        s["repo"],
                        f"{s['from_ref']} -> {s['to_ref']}",
                        s["created_at"],
                    )
                console.print(scan_table)

            return EXIT_OK
        finally:
            await cache.close()

    exit_code = asyncio.run(_run())
    raise typer.Exit(code=exit_code)


@app.command("repo-profile")
def repo_profile(
    repo: str = typer.Argument(help="Repository URL or local path as stored in cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """View cached RepoProfile knowledge for a repository."""
    _setup_logging(verbose)
    settings = get_settings()

    async def _run() -> int:
        cache = CacheStorage(settings.cache.db_path)
        try:
            await cache.initialize()
            data = await cache.get_repo_profile(repo)
            if not data:
                console.print(f"[red]Repo profile for {repo} not found.[/red]")
                return EXIT_ERROR

            try:
                profile = RepoProfile.model_validate(data)
            except ValidationError:
                console.print(
                    f"[red]Repo profile for {repo} has incompatible cached schema "
                    "and cannot be rendered.[/red]"
                )
                return EXIT_ERROR

            _print_repo_profile(profile)
            return EXIT_OK
        finally:
            await cache.close()

    exit_code = asyncio.run(_run())
    raise typer.Exit(code=exit_code)


def _print_batch_summary(result: BatchResult) -> None:
    """Print Rich table summarizing batch scan results."""
    table = Table(title="Batch Scan Summary")
    table.add_column("Project")
    table.add_column("Refs")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Report")

    for outcome in result.outcomes:
        refs = f"{outcome.pair.from_ref} -> {outcome.pair.to_ref}"
        duration = f"{outcome.duration_seconds:.0f}s"

        if outcome.error:
            status = "[red]error[/red]"
            report_col = outcome.error[:60]
        elif outcome.exit_code == EXIT_OK:
            status = "[green]clean[/green]"
            report_col = outcome.output_file
        elif outcome.exit_code == EXIT_FINDINGS:
            status = "[yellow]findings[/yellow]"
            report_col = outcome.output_file
        else:
            status = f"[red]exit {outcome.exit_code}[/red]"
            report_col = outcome.output_file

        table.add_row(outcome.repo_name, refs, status, duration, report_col)

    for repo_url, reason in result.skipped:
        table.add_row(repo_name_from_url(repo_url), "-", "[dim]skipped[/dim]", "-", reason)

    console.print(table)


@app.command("batch-scan")
def batch_scan(
    repos_file: Path = typer.Argument(help="Text file with repo URLs (one per line)"),  # noqa: B008
    last_releases: int | None = typer.Option(
        None,
        "--last-releases",
        "-n",
        help="Number of recent releases (latest major first, then backfill older majors)",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Scan releases after this date (YYYY-MM-DD)",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("batch-reports"),
        "--output-dir",
        "-d",
        help="Base directory for reports",
    ),
    output: OutputFormat = typer.Option(  # noqa: B008
        OutputFormat.JSON,
        "--output",
        "-o",
        help="Output format",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-j",
        min=1,
        help="Max parallel repositories (pairs in a repo run old->new sequentially)",
    ),
    pairing_mode: str = typer.Option(
        "per-major",
        "--pairing-mode",
        help=(
            "How to build tag pairs: chronological | per-major "
            "(for --last-releases both use latest-major-first selection)"
        ),
        show_default=True,
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be scanned"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Batch-scan repos from a file for recent releases."""
    _setup_logging(verbose)

    if last_releases is None and since is None:
        console.print("[red]Error:[/red] specify --last-releases or --since")
        raise typer.Exit(code=EXIT_ERROR)
    if last_releases is not None and since is not None:
        console.print("[red]Error:[/red] use only one of --last-releases or --since")
        raise typer.Exit(code=EXIT_ERROR)

    since_dt: datetime | None = None
    if since is not None:
        try:
            since_dt = datetime.strptime(since, "%Y-%m-%d")
        except ValueError:
            console.print(f"[red]Error:[/red] invalid date format: {since} (expected YYYY-MM-DD)")
            raise typer.Exit(code=EXIT_ERROR) from None

    try:
        repos = load_repo_list(repos_file)
    except (BatchError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=EXIT_ERROR) from exc

    console.print(f"Loaded {len(repos)} repos from {repos_file}")

    settings = get_settings(verbose=verbose)

    pairing_mode_norm = pairing_mode.strip().lower().replace("-", "_")
    if pairing_mode_norm not in {"chronological", "per_major"}:
        console.print(
            "[red]Error:[/red] invalid --pairing-mode "
            f"{pairing_mode!r} (expected chronological|per-major)"
        )
        raise typer.Exit(code=EXIT_ERROR)

    pairing_mode_value: Literal["chronological", "per_major"] = (
        "chronological" if pairing_mode_norm == "chronological" else "per_major"
    )

    if dry_run:
        exit_code = asyncio.run(
            _dry_run_batch(
                repos,
                settings,
                last_releases=last_releases,
                since=since_dt,
                pairing_mode=pairing_mode_value,
            )
        )
        raise typer.Exit(code=exit_code)

    async def _run_batch_with_scheduler() -> BatchResult:
        provider = create_provider(settings)
        shared_scheduler = LlmRequestScheduler(provider, settings.llm)
        await shared_scheduler.start()
        shared_cache: CacheStorage | None = None
        if settings.cache.enabled:
            shared_cache = CacheStorage(settings.cache.db_path)
            await shared_cache.initialize()

        use_progress = concurrency > 1

        try:
            progress_ctx = (
                Progress(
                    SpinnerColumn(finished_text="[green]\u2713[/green]"),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                )
                if use_progress
                else contextlib.nullcontext()
            )

            sentrysloth_logger = logging.getLogger("sentrysloth")
            original_level = sentrysloth_logger.level
            if use_progress:
                sentrysloth_logger.setLevel(logging.WARNING)

            try:
                with progress_ctx as progress_obj:
                    completed_window = (
                        _CompletedProgressWindow(progress_obj, max_completed=5)
                        if use_progress
                        else None
                    )

                    async def scan_fn(
                        repo: str,
                        from_ref: str,
                        to_ref: str,
                        s: Settings,
                        fmt: str,
                        baseline: str | None,
                        output_file: str | None,
                    ) -> int:
                        name = repo_name_from_url(repo)
                        scan_label = f"{name} {from_ref}\u2192{to_ref}"
                        tid = progress_obj.add_task(scan_label, total=1) if use_progress else None

                        scan_output = ScanOutput(
                            console,
                            scan_label,
                            progress=progress_obj if use_progress else None,
                            task_id=tid,
                        )

                        try:
                            result = await _run_scan(
                                repo,
                                from_ref,
                                to_ref,
                                s,
                                OutputFormat(fmt),
                                baseline,
                                output_file,
                                scheduler=shared_scheduler,
                                output=scan_output,
                                cache=shared_cache,
                            )
                            if tid is not None:
                                _mark_batch_task_finished(
                                    progress_obj,
                                    tid,
                                    scan_label=scan_label,
                                    status_markup="[green]done[/green]",
                                    completed_window=completed_window,
                                )
                            return result
                        except Exception:
                            if tid is not None:
                                _mark_batch_task_finished(
                                    progress_obj,
                                    tid,
                                    scan_label=scan_label,
                                    status_markup="[red]error[/red]",
                                    completed_window=completed_window,
                                )
                            raise

                    return await run_batch_scan(
                        repos,
                        settings,
                        output_dir,
                        output.value,
                        scan_fn,
                        last_releases=last_releases,
                        since=since_dt,
                        pairing_mode=pairing_mode_value,
                        concurrency=concurrency,
                    )
            finally:
                if use_progress:
                    sentrysloth_logger.setLevel(original_level)
        finally:
            stats = await shared_scheduler.get_stats()
            if settings.verbose:
                _log_scheduler_stats(stats, console, prefix="Batch ")
            await shared_scheduler.close()
            if shared_cache is not None:
                await shared_cache.close()

    batch_result = asyncio.run(_run_batch_with_scheduler())

    _print_batch_summary(batch_result)

    codes = [o.exit_code for o in batch_result.outcomes]
    worst = max(codes) if codes else EXIT_OK
    raise typer.Exit(code=worst)


async def _dry_run_batch(
    repos: list[str],
    settings: Settings,
    *,
    last_releases: int | None = None,
    since: datetime | None = None,
    pairing_mode: Literal["chronological", "per_major"] = "chronological",
) -> int:
    """Clone/fetch repos, list tag pairs, print what would be scanned."""
    table = Table(title="Dry Run: Planned Scans")
    table.add_column("Project")
    table.add_column("From")
    table.add_column("To")

    has_any = False
    for repo_url in repos:
        name = repo_name_from_url(repo_url)
        git_source = GitSource(repo_url, settings)

        try:
            await git_source.ensure_cloned()
        except GitSourceError as exc:
            console.print(f"[red]{name}:[/red] {exc}")
            continue

        limit = resolve_tag_fetch_limit(pairing_mode, last_releases)
        tags_with_dates = await git_source.list_tags_with_dates(limit=limit)

        if len(tags_with_dates) < 2:
            console.print(f"[yellow]{name}:[/yellow] fewer than 2 tags, skipping")
            continue

        tags = [t[0] for t in tags_with_dates]
        tag_dates = [t[1] for t in tags_with_dates]

        effective_since = normalize_since_for_tag_dates(since, tag_dates)

        pairs = build_tag_pairs(
            tags,
            last_releases=last_releases,
            since=effective_since,
            tag_dates=tag_dates if since is not None else None,
            pairing_mode=pairing_mode,
        )

        if not pairs:
            console.print(f"[yellow]{name}:[/yellow] no tag pairs matched filters")
            continue

        for pair in pairs:
            table.add_row(name, pair.from_ref, pair.to_ref)
            has_any = True

    if has_any:
        console.print(table)
    else:
        console.print("[yellow]No scans to run.[/yellow]")

    return EXIT_OK


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version"),
) -> None:
    """SentrySloth: Change-risk / security-review assistant."""
    if version:
        output_console.print(f"sentrysloth {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        raise typer.Exit()
