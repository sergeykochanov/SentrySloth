"""Offline evaluation against labeled, revision-pinned cases; never calls an LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sentrysloth.models import CONFIDENCE_ORDER, SEVERITY_ORDER, ScanResult


class ExpectedIssue(BaseModel):
    file_path: str
    anchor: str


class EvaluationCase(BaseModel):
    case_id: str
    split: Literal["development", "holdout"]
    label: Literal["positive", "negative", "unresolved"]
    repo: str
    from_sha: str
    to_sha: str
    reason: str
    expected: list[ExpectedIssue] = Field(default_factory=list)


class EvaluationDataset(BaseModel):
    cases: list[EvaluationCase]


def evaluate_reports(
    dataset: EvaluationDataset,
    reports: list[ScanResult],
    *,
    split: str = "holdout",
    k: int = 10,
    review_minutes: float | None = None,
) -> dict:
    """Match exact revisions and code anchors, keeping missing scans in recall's denominator."""
    if k < 1:
        raise ValueError("k must be positive")
    cases = [c for c in dataset.cases if c.split == split]
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    ranked = []
    matched_cases = set()
    complete = missing = unresolved = tokens = candidates = 0
    issues = []
    usage_complete = True
    for case in cases:
        matches = [
            r
            for r in reports
            if (r.release.repo_url, r.release.from_sha, r.release.to_sha)
            == (case.repo, case.from_sha, case.to_sha)
        ]
        if len(matches) > 1:
            raise ValueError(f"Multiple reports for {case.case_id}; compare runs separately")
        if not matches:
            missing += 1
            continue
        report = matches[0]
        complete += int(report.complete)
        if not report.complete:
            issues.append(case.case_id)
        candidates += len(report.candidates)
        if report.llm_metrics:
            m = report.llm_metrics
            usage_complete = usage_complete and m.token_usage_complete
            tokens += (
                m.triage_input_tokens
                + m.triage_output_tokens
                + m.analysis_input_tokens
                + m.analysis_output_tokens
                + m.verification_input_tokens
                + m.verification_output_tokens
            )
        if case.label == "unresolved":
            unresolved += len(report.findings)
            continue
        used_anchors = set()
        for finding in report.findings:
            hit = next(
                (
                    i
                    for i, expected in enumerate(case.expected)
                    if i not in used_anchors
                    and any(
                        e.code.file_path == expected.file_path
                        and expected.anchor in e.code.snippet
                        and e.code.commit_sha in {case.from_sha, case.to_sha}
                        for e in finding.evidence
                    )
                ),
                None,
            )
            correct = case.label == "positive" and hit is not None
            if correct:
                used_anchors.add(hit)
                matched_cases.add(case.case_id)
            ranked.append(
                (
                    CONFIDENCE_ORDER[finding.confidence],
                    SEVERITY_ORDER[finding.severity],
                    finding.finding_id,
                    correct,
                )
            )
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    true_positives = sum(row[3] for row in ranked)
    positives = sum(c.label == "positive" for c in cases)
    top = ranked[:k]
    return {
        "split": split,
        "cases": len(cases),
        "complete_cases": complete,
        "missing_cases": missing,
        "incomplete_cases": issues,
        "coverage": complete / len(cases) if cases else None,
        "true_positive_findings": true_positives,
        "false_positive_findings": len(ranked) - true_positives,
        "precision": true_positives / len(ranked) if ranked else None,
        "precision_at_k": sum(row[3] for row in top) / len(top) if top else None,
        "k": k,
        "ranked_findings": len(ranked),
        "case_recall": len(matched_cases) / positives if positives else None,
        "unresolved_findings_excluded": unresolved,
        "candidates_to_review": candidates,
        "total_tokens": tokens,
        "token_usage_complete": usage_complete,
        "manual_review_minutes": review_minutes,
        "matching": "Exact repo and SHA pair; literal evidence anchors. Human labels required.",
    }


def load_reports(directory: Path) -> list[ScanResult]:
    reports = []
    for path in sorted(directory.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "scan_id" in data:
            reports.append(ScanResult.model_validate(data))
    return reports
