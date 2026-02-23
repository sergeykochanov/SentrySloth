"""Tests for data models."""

from __future__ import annotations

from sentrysloth.models import (
    AffectedCode,
    Baseline,
    BaselineEntry,
    Confidence,
    DiffChunk,
    DiffHunk,
    Evidence,
    Finding,
    FindingType,
    LLMResponse,
    Severity,
    TriageResult,
)


class TestFinding:
    def test_stable_finding_id(self):
        """Finding ID should be stable for same inputs."""
        f1 = Finding(
            repo="https://github.com/test/repo",
            file_path="src/auth.py",
            hunk_signature="10-10-1",
            finding_type=FindingType.VULNERABILITY,
            title="Test",
            description="Test finding",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            evidence=[
                Evidence(
                    description="test",
                    code=AffectedCode(
                        file_path="src/auth.py",
                        start_line=10,
                        end_line=15,
                        snippet="code",
                        is_added=True,
                    ),
                    reasoning="test",
                )
            ],
        )
        f2 = Finding(
            repo="https://github.com/test/repo",
            file_path="src/auth.py",
            hunk_signature="10-10-1",
            finding_type=FindingType.VULNERABILITY,
            title="Different title",
            description="Different description",
            severity=Severity.MEDIUM,
            confidence=Confidence.LOW,
            evidence=[
                Evidence(
                    description="other",
                    code=AffectedCode(
                        file_path="src/auth.py",
                        start_line=10,
                        end_line=15,
                        snippet="other code",
                        is_added=True,
                    ),
                    reasoning="other",
                )
            ],
        )
        # Same repo + file + hunk_signature + finding_type => same ID
        assert f1.finding_id == f2.finding_id

    def test_different_finding_id_for_different_type(self):
        base = dict(
            repo="repo",
            file_path="file.py",
            hunk_signature="1-1-1",
            title="t",
            description="d",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            evidence=[
                Evidence(
                    description="e",
                    code=AffectedCode(
                        file_path="file.py", start_line=1, end_line=2, snippet="x", is_added=True
                    ),
                    reasoning="r",
                )
            ],
        )
        f1 = Finding(finding_type=FindingType.VULNERABILITY, **base)
        f2 = Finding(finding_type=FindingType.AUTH_BYPASS, **base)
        assert f1.finding_id != f2.finding_id


class TestTriageResult:
    def test_default_values(self):
        t = TriageResult(
            chunk_file_path="test.py",
            is_security_relevant=False,
        )
        assert t.confidence == Confidence.MEDIUM
        assert t.suggested_severity == Severity.INFO


class TestBaseline:
    def test_suppression(self):
        baseline = Baseline(
            entries=[
                BaselineEntry(finding_id="abc123", reason="accepted risk"),
            ]
        )
        assert baseline.is_suppressed("abc123")
        assert not baseline.is_suppressed("xyz789")


class TestDiffChunk:
    def test_creation(self):
        chunk = DiffChunk(
            file_path="src/main.py",
            hunks=[
                DiffHunk(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=7,
                    content="test content",
                )
            ],
            raw_diff="test diff",
            token_estimate=100,
            security_score=0.8,
            language="python",
        )
        assert chunk.file_path == "src/main.py"
        assert len(chunk.hunks) == 1
        assert chunk.security_score == 0.8


class TestLLMResponse:
    def test_generic_type(self):
        resp = LLMResponse[TriageResult](
            data=TriageResult(
                chunk_file_path="test.py",
                is_security_relevant=True,
                reason="contains auth changes",
            ),
            input_tokens=100,
            output_tokens=50,
            latency_ms=500.0,
            model="gemini-2.5-flash",
        )
        assert resp.data.is_security_relevant
        assert resp.input_tokens == 100
