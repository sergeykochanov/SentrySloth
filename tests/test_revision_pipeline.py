"""Integration tests use real local Git history, never execute scanned source."""

import difflib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from git import Repo

from sentrysloth.analyzers.analysis_shared import AnalysisResponse
from sentrysloth.analyzers.diff_extractor import _is_noise_only_change, extract_chunks
from sentrysloth.analyzers.repository_tools import execute_tool_call, source_range
from sentrysloth.batch import build_tag_pairs, resolve_tag_pairs
from sentrysloth.config import get_settings
from sentrysloth.models import LLMResponse, TriageResult
from sentrysloth.providers.base import LLMProviderError, ToolCall, ToolCallResponse
from sentrysloth.reporters.json_reporter import generate_json_report
from sentrysloth.reporters.markdown_reporter import generate_markdown_report
from sentrysloth.reporters.sarif_reporter import generate_sarif_report
from sentrysloth.scanner import EXIT_INCOMPLETE, EXIT_OK, run_scan
from sentrysloth.sources.git import GitSource
from tests.test_verification import candidate, decision


def commit(repo, content, tag):
    path = repo.working_tree_dir
    (Path(path) / "app.py").write_text(content)
    repo.index.add(["app.py"])
    revision = repo.index.commit(tag)
    repo.create_tag(tag)
    return revision.hexsha


@pytest.fixture
def history(tmp_path):
    repo = Repo.init(tmp_path / "repo")
    first = commit(repo, "def handler():\n    check_auth()\n    return private_data\n", "v1.0.0")
    second = commit(repo, "def handler():\n    return private_data\n", "v1.0.1")
    return repo, first, second


@pytest.mark.parametrize(
    "text",
    [
        "-check()\n-do_work()\n+do_work()\n+check()",
        "-    check()\n+        check()",
        "-check()\n+# check()",
        '-path = "a b"\n+path = "ab"',
    ],
)
def test_semantic_changes_are_never_noise(text):
    assert not _is_noise_only_change(text)


def test_large_hunk_preserves_every_changed_line():
    old = [f"old_{i} = 0\n" for i in range(800)]
    new = [f"new_{i} = 1\n" for i in range(800)]
    diff = "".join(difflib.unified_diff(old, new, fromfile="a/app.py", tofile="b/app.py"))
    chunks = extract_chunks(diff, get_settings(chunk_token_budget=100))
    changed = [
        line for c in chunks for line in c.raw_diff.splitlines() if line.startswith(("+", "-"))
    ]
    assert changed == ["-" + line.rstrip("\n") for line in old] + [
        "+" + line.rstrip("\n") for line in new
    ]
    assert not any(c.truncated for c in chunks)
    assert sum(h.source_length for c in chunks for h in c.hunks) == 800
    assert sum(h.target_length for c in chunks for h in c.hunks) == 800


def test_numbered_ranges_reach_tail():
    content = "\n".join(f"line{i}" for i in range(600))
    first = source_range(content)
    assert first["next_line"] == 201
    tail = source_range(content, 590, 600)
    assert "600: line599" in tail["content"]
    assert tail["next_line"] is None


@pytest.mark.asyncio
async def test_real_git_tools_and_ancestry(history):
    repo, first, second = history
    repo.git.checkout(first)
    branch = commit(repo, "def other():\n    validate()\n", "v1.1.0")
    source = GitSource(repo.working_tree_dir, get_settings())
    await source.ensure_cloned()
    assert await source.resolve_ref("v1.0.0") == first
    assert await source.is_ancestor(first, second)
    assert not await source.is_ancestor(second, branch)
    pairs = await resolve_tag_pairs(source, ["v1.1.0", "v1.0.1", "v1.0.0"])
    assert {(p.from_ref, p.to_ref) for p in pairs} == {("v1.0.0", "v1.0.1"), ("v1.0.0", "v1.1.0")}
    changed = json.loads(await execute_tool_call("list_changes", {}, source, first, second, {}))
    assert changed["files"] == ["app.py"]
    before = json.loads(
        await execute_tool_call(
            "read_file_before", {"file_path": "app.py"}, source, first, second, {}
        )
    )
    assert "check_auth()" in before["content"] and before["ref"] == first
    symbol = json.loads(
        await execute_tool_call(
            "read_symbol", {"file_path": "app.py", "symbol": "handler"}, source, first, second, {}
        )
    )
    assert symbol["symbols"][0]["end_line"] == 2
    diff = json.loads(
        await execute_tool_call("read_diff", {"file_path": "app.py"}, source, first, second, {})
    )
    assert "-    check_auth()" in diff["content"]


@pytest.mark.asyncio
async def test_real_search_pagination(history):
    repo, first, _ = history
    last = commit(repo, "\n".join(f"match_{i} = 1" for i in range(60)), "v1.0.2")
    source = GitSource(repo.working_tree_dir, get_settings())
    await source.ensure_cloned()
    page = json.loads(
        await execute_tool_call(
            "search_code", {"pattern": "match_", "cursor": 40}, source, first, last, {}
        )
    )
    assert page["matches"][0]["line"] == 41
    assert len(page["matches"]) == 20 and page["next_cursor"] is None


@pytest.mark.parametrize("mode", ["chronological", "per_major"])
def test_since_retains_predecessor(mode):
    now = datetime.now(UTC)
    pairs = build_tag_pairs(
        ["v1.0.1", "v1.0.0"],
        since=now - timedelta(days=1),
        tag_dates=[now, now - timedelta(days=2)],
        pairing_mode=mode,
    )
    assert [(p.from_ref, p.to_ref) for p in pairs] == [("v1.0.0", "v1.0.1")]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_scan_records_provider_outcome_with_pinned_refs(history, failed):
    repo, first, second = history
    scheduler = AsyncMock()
    scheduler.quota_error = None
    scheduler.generate_structured.return_value = LLMResponse(
        data=TriageResult(
            chunk_file_path="app.py", is_security_relevant=True, reason="auth guard removed"
        )
    )
    if failed:
        scheduler.generate_with_tools.side_effect = LLMProviderError("provider offline")
        # Triage succeeds; fallback analysis fails.
        scheduler.generate_structured.side_effect = [
            scheduler.generate_structured.return_value,
            LLMProviderError("provider offline"),
        ]
    else:
        scheduler.generate_with_tools.return_value = ToolCallResponse(
            content='{"findings": [], "summary": "reviewed"}', tool_calls=[]
        )
    code, report = await run_scan(
        repo.working_tree_dir,
        "v1.0.0",
        "v1.0.1",
        get_settings(cache={"enabled": False}),
        None,
        scheduler=scheduler,
    )
    assert code == (EXIT_INCOMPLETE if failed else EXIT_OK)
    assert report.complete == (not failed)
    assert report.release.from_sha == first and report.release.to_sha == second
    assert report.release.relationship == "ancestor"
    assert bool(report.coverage_issues) == failed


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_fails", [False, True])
@pytest.mark.parametrize("agentic_failure", ["parse", "provider"])
async def test_unparsed_candidate_survives_fallback_and_report_roundtrip(
    history, fallback_fails, agentic_failure
):
    repo, _, _ = history
    scheduler = AsyncMock()
    scheduler.quota_error = None
    original = '{"findings": [{"title": "Candidate needing review"'
    repair = "invalid repair"
    scheduler.generate_with_tools.side_effect = [
        ToolCallResponse(content=original, tool_calls=[], input_tokens=30, output_tokens=10),
        ToolCallResponse(content=repair, tool_calls=[], input_tokens=20, output_tokens=5),
    ]
    if agentic_failure == "provider":
        scheduler.generate_with_tools.side_effect = LLMProviderError("agentic offline")
    scheduler.generate_structured.side_effect = [
        LLMResponse(data=TriageResult(chunk_file_path="app.py", is_security_relevant=True)),
        LLMProviderError("fallback offline")
        if fallback_fails
        else LLMResponse(
            data=AnalysisResponse(findings=[], summary="no findings"),
            input_tokens=7,
        ),
    ]
    code, report = await run_scan(
        repo.working_tree_dir,
        "v1.0.0",
        "v1.0.1",
        get_settings(cache={"enabled": False}),
        None,
        scheduler=scheduler,
    )
    assert code == EXIT_INCOMPLETE
    assert not report.complete and report.coverage_issues
    assert report.findings == [] and report.candidates == []
    assert report.llm_metrics.analysis_completed == 0
    failed_input = 50 if agentic_failure == "parse" else 0
    assert report.llm_metrics.analysis_input_tokens == failed_input + (0 if fallback_fails else 7)
    assert report.llm_metrics.analysis_output_tokens == (15 if agentic_failure == "parse" else 0)
    serialized = json.loads(generate_json_report(report))
    if agentic_failure == "parse":
        failure = serialized["llm_metrics"]["analysis_failures"][0]
        assert failure["raw_response"] == original
        assert failure["repair_response"] == repair
        assert failure["from_ref"] == report.release.from_sha
        assert failure["to_ref"] == report.release.to_sha
        assert "Unparsed analysis requiring review" in generate_markdown_report(report)
    else:
        assert serialized["llm_metrics"]["analysis_failures"] == []
    sarif = json.loads(generate_sarif_report(report))
    assert not sarif["runs"][0]["invocations"][0]["executionSuccessful"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict,verify",
    [
        ("confirmed_static", True),
        ("rejected", True),
        ("needs_review", True),
        ("confirmed_static", False),
    ],
)
async def test_scan_routes_verdicts_without_promoting_candidates(history, verdict, verify):
    repo, _, _ = history
    scheduler = AsyncMock()
    scheduler.quota_error = None
    scheduler.generate_structured.return_value = LLMResponse(
        data=TriageResult(
            chunk_file_path="app.py", is_security_relevant=True, reason="guard removed"
        )
    )
    finding = candidate().model_dump(mode="json")
    finding["evidence"] = [
        {
            "description": "guard",
            "file_path": "app.py",
            "start_line": 2,
            "end_line": 2,
            "snippet": "    check_auth()",
            "revision": "before",
            "reasoning": "removed",
        }
    ]
    review = decision(verdict=verdict)
    review.evidence[0].start_line = review.evidence[0].end_line = 2
    review.evidence[0].snippet = "    check_auth()"
    review.evidence[1].start_line = review.evidence[1].end_line = 2
    review.evidence[1].snippet = "    return private_data"
    scheduler.generate_with_tools.side_effect = [
        ToolCallResponse(content=json.dumps({"findings": [finding]})),
        ToolCallResponse(
            tool_calls=[ToolCall(id="inspect", name="read_diff", arguments={"file_path": "app.py"})]
        ),
        ToolCallResponse(content=review.model_dump_json()),
    ]
    code, result = await run_scan(
        repo.working_tree_dir,
        "v1.0.0",
        "v1.0.1",
        get_settings(cache={"enabled": False}, verify_findings=verify),
        None,
        scheduler=scheduler,
    )
    assert code == EXIT_OK
    confirmed = verdict == "confirmed_static" and verify
    assert len(result.findings) == int(confirmed)
    assert len(result.candidates) == int(not confirmed)
    if confirmed:
        assert all(e.code.commit_sha for e in result.findings[0].evidence)
    if not verify:
        assert result.candidates[0].status.value == "needs_review"
        assert scheduler.generate_with_tools.call_count == 1
