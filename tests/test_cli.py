"""Smoke tests for the CLI entry point."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from sentrysloth import __version__
from sentrysloth.cli import (
    OutputFormat,
    ScanOutput,
    _CompletedProgressWindow,
    _mark_batch_task_finished,
    _output_result,
    _print_summary,
    _run_scan,
    app,
    console,
)
from sentrysloth.models import (
    Confidence,
    FindingType,
    ReleaseInfo,
    RepoProfile,
    ScanResult,
    Severity,
)
from tests.conftest import strip_ansi

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in strip_ansi(result.output)


def test_help_output():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.output)
    assert "scan" in out
    assert "list-versions" in out


def test_scan_help():
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    out = strip_ansi(result.output)
    assert "--from" in out
    assert "--to" in out


def test_scan_missing_required_args():
    """scan without --from/--to should fail."""
    result = runner.invoke(app, ["scan", "https://example.com/repo"])
    assert result.exit_code != 0


def test_report_handles_incompatible_cached_schema():
    with patch("sentrysloth.cli.CacheStorage") as cache_cls:
        cache = cache_cls.return_value
        cache.initialize = AsyncMock()
        cache.get_scan = AsyncMock(return_value={"unexpected": "shape"})
        cache.close = AsyncMock()

        result = runner.invoke(app, ["report", "scan-123"])
        assert result.exit_code != 0
        assert "incompatible cached schema" in strip_ansi(result.output)


def test_repo_profile_prints_tech_stack():
    with patch("sentrysloth.cli.CacheStorage") as cache_cls:
        cache = cache_cls.return_value
        cache.initialize = AsyncMock()
        cache.get_repo_profile = AsyncMock(
            return_value=RepoProfile(
                repo="https://github.com/org/repo",
                last_ref="v1.1.0",
                tech_stack=["python", "fastapi", "pydantic"],
                overview=["API service"],
                known_risks=["Public endpoint auth drift"],
            ).model_dump(mode="json")
        )
        cache.close = AsyncMock()

        result = runner.invoke(app, ["repo-profile", "https://github.com/org/repo"])
        assert result.exit_code == 0
        out = strip_ansi(result.output)
        assert "Tech Stack" in out
        assert "fastapi" in out
        assert "Overview" in out


def test_repo_profile_not_found():
    with patch("sentrysloth.cli.CacheStorage") as cache_cls:
        cache = cache_cls.return_value
        cache.initialize = AsyncMock()
        cache.get_repo_profile = AsyncMock(return_value=None)
        cache.close = AsyncMock()

        result = runner.invoke(app, ["repo-profile", "https://github.com/org/missing"])
        assert result.exit_code != 0
        assert "not found" in strip_ansi(result.output).lower()


def test_repo_profile_handles_incompatible_cached_schema():
    with patch("sentrysloth.cli.CacheStorage") as cache_cls:
        cache = cache_cls.return_value
        cache.initialize = AsyncMock()
        cache.get_repo_profile = AsyncMock(return_value={"unexpected": "shape"})
        cache.close = AsyncMock()

        result = runner.invoke(app, ["repo-profile", "https://github.com/org/repo"])
        assert result.exit_code != 0
        assert "incompatible cached schema" in strip_ansi(result.output)


def test_scan_nonexistent_repo(tmp_path):
    """scan with a non-existent local path should give a graceful error."""
    fake = str(tmp_path / "nonexistent")
    result = runner.invoke(app, ["scan", fake, "--from", "v1", "--to", "v2"])
    # Should exit with error code (2) rather than traceback
    assert result.exit_code != 0


class TestOutputResultQuiet:
    def test_quiet_suppresses_report_saved_message(self, tmp_path):
        """When quiet=True, 'Report saved to ...' should not be printed."""
        result = ScanResult.model_construct(
            scan_id="test123",
            release=None,
            findings=[],
            triage_stats=None,
            llm_metrics=None,
            started_at=None,
            completed_at=None,
            prompt_version="v1",
        )
        out_file = str(tmp_path / "report.json")

        with patch.object(console, "print") as mock_print:
            _output_result(result, OutputFormat.JSON, out_file, quiet=True)

        # File should be written
        assert (tmp_path / "report.json").exists()
        # But "Report saved" should NOT be printed
        for call in mock_print.call_args_list:
            assert "Report saved" not in str(call)

    def test_not_quiet_prints_report_saved(self, tmp_path):
        """Default (quiet=False) should print 'Report saved to ...'."""
        result = ScanResult.model_construct(
            scan_id="test123",
            release=None,
            findings=[],
            triage_stats=None,
            llm_metrics=None,
            started_at=None,
            completed_at=None,
            prompt_version="v1",
        )
        out_file = str(tmp_path / "report.json")

        with patch.object(console, "print") as mock_print:
            _output_result(result, OutputFormat.JSON, out_file, quiet=False)

        mock_print.assert_called_once()
        assert "Report saved" in str(mock_print.call_args)


class TestPrintSummary:
    def test_suppresses_empty_summary_when_show_empty_false(self):
        result = ScanResult(
            scan_id="scan123",
            release=ReleaseInfo(
                repo_url="https://github.com/org/repo",
                from_ref="v1.0",
                to_ref="v1.1",
                total_files_changed=0,
                total_additions=0,
                total_deletions=0,
            ),
            findings=[],
        )
        with patch.object(console, "print") as mock_print:
            _print_summary(result, show_empty=False)
        mock_print.assert_not_called()

    def test_empty_summary_includes_context_when_enabled(self):
        result = ScanResult(
            scan_id="scan123",
            release=ReleaseInfo(
                repo_url="https://github.com/org/repo",
                from_ref="v1.0",
                to_ref="v1.1",
                total_files_changed=0,
                total_additions=0,
                total_deletions=0,
            ),
            findings=[],
        )
        with patch.object(console, "print") as mock_print:
            _print_summary(result, show_empty=True)
        assert mock_print.call_count == 1
        printed = str(mock_print.call_args)
        assert "No findings" in printed
        assert "https://github.com/org/repo" in printed
        assert "v1.0" in printed and "v1.1" in printed

    def test_non_empty_summary_title_includes_repo_and_refs(self):
        finding = MagicMock()
        finding.finding_id = "abcd1234"
        finding.severity = Severity.HIGH
        finding.confidence = Confidence.MEDIUM
        finding.finding_type = FindingType.AUTH_BYPASS
        finding.title = "Token verification removed"
        finding.file_path = "src/auth.py"

        result = ScanResult.model_construct(
            scan_id="scan123",
            release=ReleaseInfo(
                repo_url="https://github.com/org/repo",
                from_ref="v1.0",
                to_ref="v1.1",
                total_files_changed=1,
                total_additions=1,
                total_deletions=0,
            ),
            findings=[finding],
            triage_stats=None,
            llm_metrics=None,
            started_at=None,
            completed_at=None,
            prompt_version="v1",
        )
        with patch.object(console, "print") as mock_print:
            _print_summary(result, show_empty=True)

        assert mock_print.call_count == 1
        table = mock_print.call_args.args[0]
        assert getattr(table, "title", "") == "Findings Summary: repo v1.0→v1.1"


class TestRunScanOutputSignature:
    @pytest.mark.asyncio
    async def test_run_scan_accepts_output_kwarg(self, tmp_path):
        """_run_scan should accept output=ScanOutput kwarg without error."""
        fake = str(tmp_path / "nonexistent")
        # Will fail on GitSource.ensure_cloned() but should not fail on signature
        from sentrysloth.config import get_settings

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        scan_output = ScanOutput(
            console,
            "test v1→v2",
            progress=MagicMock(),
            task_id=MagicMock(),
        )
        exit_code = await _run_scan(
            fake,
            "v1",
            "v2",
            settings,
            OutputFormat.JSON,
            None,
            None,
            output=scan_output,
        )
        # Should return EXIT_ERROR (clone fails), not crash on output kwarg
        assert exit_code == 2


class TestBatchProgressWindow:
    def test_keeps_only_last_five_completed_tasks(self):
        progress = MagicMock()
        window = _CompletedProgressWindow(progress, max_completed=5)

        for task_id in range(1, 8):
            window.mark_completed(task_id)

        assert progress.remove_task.call_count == 2
        assert progress.remove_task.call_args_list[0].args == (1,)
        assert progress.remove_task.call_args_list[1].args == (2,)

    def test_done_and_error_both_use_completion_window(self):
        progress = MagicMock()
        window = _CompletedProgressWindow(progress, max_completed=5)

        _mark_batch_task_finished(
            progress,
            101,
            scan_label="repo v1→v2",
            status_markup="[green]done[/green]",
            completed_window=window,
        )
        _mark_batch_task_finished(
            progress,
            102,
            scan_label="repo v2→v3",
            status_markup="[red]error[/red]",
            completed_window=window,
        )

        assert progress.update.call_count == 2
        done_desc = progress.update.call_args_list[0].kwargs["description"]
        error_desc = progress.update.call_args_list[1].kwargs["description"]
        assert done_desc.endswith("[green]done[/green]")
        assert error_desc.endswith("[red]error[/red]")
        progress.remove_task.assert_not_called()
