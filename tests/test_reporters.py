"""Smoke tests for JSON/Markdown/SARIF reporters."""

from __future__ import annotations

import json

from sentrysloth.models import (
    AffectedCode,
    Confidence,
    Evidence,
    Finding,
    FindingType,
    ReleaseInfo,
    ScanResult,
    Severity,
)
from sentrysloth.reporters.json_reporter import generate_json_report
from sentrysloth.reporters.markdown_reporter import generate_markdown_report
from sentrysloth.reporters.sarif_reporter import generate_sarif_report


def _base_result() -> ScanResult:
    return ScanResult(
        scan_id="scan123",
        release=ReleaseInfo(
            repo_url="https://github.com/org/repo",
            from_ref="v1.0.0",
            to_ref="v1.1.0",
            total_files_changed=1,
            total_additions=10,
            total_deletions=2,
        ),
    )


def _sample_finding() -> Finding:
    return Finding(
        repo="https://github.com/org/repo",
        file_path="src/auth.py",
        hunk_signature="1-1-1",
        finding_type=FindingType.AUTH_BYPASS,
        title="Token verification disabled",
        description="Signature verification was bypassed.",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        evidence=[
            Evidence(
                description="JWT decode disables signature checks",
                code=AffectedCode(
                    file_path="src/auth.py",
                    start_line=10,
                    end_line=12,
                    snippet="jwt.decode(token, options={'verify_signature': False})",
                    is_added=True,
                ),
                reasoning="Allows forged tokens to pass validation.",
            )
        ],
    )


def test_json_and_markdown_handle_empty_findings():
    result = _base_result()

    json_payload = json.loads(generate_json_report(result))
    markdown_payload = generate_markdown_report(result)

    assert json_payload["findings"] == []
    assert "No security-relevant findings detected." in markdown_payload


def test_sarif_smoke_with_single_finding():
    result = _base_result()
    result.findings = [_sample_finding()]

    sarif = json.loads(generate_sarif_report(result))

    assert sarif["version"] == "2.1.0"
    assert sarif["runs"]
    assert sarif["runs"][0]["results"]
