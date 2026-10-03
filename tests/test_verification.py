"""Evidence gates must fail conservatively even when the model is confident."""

from unittest.mock import AsyncMock

import pytest

from sentrysloth.analyzers.verification import (
    VerificationDecision,
    apply_decision,
    check_citation,
    verify_candidate,
)
from sentrysloth.config import get_settings
from sentrysloth.models import (
    AffectedCode,
    Confidence,
    Evidence,
    Finding,
    FindingType,
    ReviewStatus,
    Severity,
)
from sentrysloth.providers.base import LLMProviderError, ToolCall, ToolCallResponse


def candidate():
    return Finding(
        repo="repo",
        file_path="app.py",
        title="Guard removed",
        description="Unauthorized read",
        finding_type=FindingType.SECURITY_REGRESSION,
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        hunk_signature="1",
        evidence=[
            Evidence(
                description="guard",
                reasoning="removed",
                code=AffectedCode(
                    file_path="app.py",
                    start_line=1,
                    end_line=1,
                    snippet="check_auth()",
                    is_added=False,
                    revision="before",
                ),
            )
        ],
    )


def decision(**overrides):
    data = dict(
        verdict="confirmed_static",
        reason="Public handler loses its guard",
        attacker_control="HTTP request",
        entry_point="handler",
        trust_boundary="auth",
        impact="private data read",
        before_after="guard removed",
        mitigations_checked=["router has no equivalent guard"],
        evidence=[
            dict(
                description="old",
                file_path="app.py",
                start_line=1,
                end_line=1,
                snippet="check_auth()",
                revision="before",
                reasoning="required before",
            ),
            dict(
                description="new",
                file_path="app.py",
                start_line=1,
                end_line=1,
                snippet="return private_data",
                revision="after",
                reasoning="no guard",
            ),
        ],
    )
    return VerificationDecision(**(data | overrides))


@pytest.mark.parametrize(
    "snippet,start,end,valid",
    [
        ("    check()", 2, 2, True),
        ("check()", 2, 2, False),
        ("invented()", 2, 2, False),
        ("", 1, 1, False),
        ("code", 0, 1, False),
        ("code", 1, 99, False),
    ],
)
def test_exact_citation(snippet, start, end, valid):
    code = AffectedCode(
        file_path="a.py", start_line=start, end_line=end, snippet=snippet, is_added=True
    )
    assert (check_citation(code, "code\n    check()\n") is None) == valid


@pytest.mark.parametrize(
    "overrides,relationship",
    [
        ({"entry_point": ""}, "ancestor"),
        ({"mitigations_checked": []}, "ancestor"),
        ({"evidence": []}, "ancestor"),
        ({}, "divergent"),
        ({"missing_evidence": ["Unknown dependency behavior"]}, "ancestor"),
    ],
)
def test_missing_proof_never_promotes(overrides, relationship):
    result = apply_decision(candidate(), decision(**overrides), [], relationship)
    assert result.status == ReviewStatus.NEEDS_REVIEW
    assert result.missing_evidence


@pytest.mark.asyncio
async def test_verifier_reads_then_pins_both_revisions():
    provider = AsyncMock()
    provider.generate_with_tools.side_effect = [
        ToolCallResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="1",
                    name="read_diff",
                    arguments={"file_path": "app.py", "revision": "before"},
                )
            ],
        ),
        ToolCallResponse(content=decision().model_dump_json(), tool_calls=[]),
    ]
    git = AsyncMock()
    git.get_file_diff.return_value = "-check_auth()\n+return private_data"
    git.get_file_content.side_effect = lambda ref, path: (
        "check_auth()" if ref == "a" * 40 else "return private_data"
    )
    result, metrics = await verify_candidate(
        candidate(), provider, git, get_settings(), "a" * 40, "b" * 40, "ancestor"
    )
    assert result.status == ReviewStatus.CONFIRMED_STATIC
    assert {e.code.commit_sha for e in result.evidence} == {"a" * 40, "b" * 40}
    assert not metrics.errors
    assert provider.generate_with_tools.call_count == 2


@pytest.mark.asyncio
async def test_invented_citation_cannot_be_confirmed():
    provider = AsyncMock()
    provider.generate_with_tools.return_value = ToolCallResponse(
        content=decision().model_dump_json(), tool_calls=[]
    )
    git = AsyncMock()
    git.get_file_content.return_value = "safe()"
    result, _ = await verify_candidate(
        candidate(), provider, git, get_settings(), "a" * 40, "b" * 40, "ancestor"
    )
    assert result.status == ReviewStatus.NEEDS_REVIEW
    assert any("match" in error for error in result.missing_evidence)


@pytest.mark.asyncio
async def test_provider_failure_keeps_candidate_and_records_incomplete():
    provider = AsyncMock()
    provider.generate_with_tools.side_effect = LLMProviderError("offline")
    git = AsyncMock()
    git.get_file_content.return_value = "check_auth()"
    result, metrics = await verify_candidate(
        candidate(), provider, git, get_settings(), "a" * 40, "b" * 40, "ancestor"
    )
    assert result.status == ReviewStatus.NEEDS_REVIEW
    assert metrics.errors and "offline" in metrics.errors[0]
