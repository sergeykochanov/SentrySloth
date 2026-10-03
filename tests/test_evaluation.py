"""Evaluation must penalize missing coverage and exclude unresolved labels."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentrysloth.cli import app
from sentrysloth.evaluation import EvaluationCase, EvaluationDataset, evaluate_reports
from sentrysloth.models import ReleaseInfo, ScanResult
from tests.test_verification import candidate


def dataset():
    return EvaluationDataset(
        cases=[
            EvaluationCase(
                case_id=label,
                split="holdout",
                label=label,
                repo=label,
                from_sha="a" * 40,
                to_sha="b" * 40,
                reason="Human reviewed fixture",
                expected=[{"file_path": "app.py", "anchor": "check_auth()"}]
                if label == "positive"
                else [],
            )
            for label in ("positive", "negative", "unresolved")
        ]
    )


def reports():
    finding = candidate()
    finding.evidence[0].code.commit_sha = "a" * 40
    return [
        ScanResult(
            scan_id=c.case_id,
            release=ReleaseInfo(
                repo_url=c.repo, from_ref="old", to_ref="new", from_sha=c.from_sha, to_sha=c.to_sha
            ),
            findings=[finding],
        )
        for c in dataset().cases
    ]


def test_metrics_include_false_positive_and_exclude_unresolved():
    result = evaluate_reports(dataset(), reports(), k=2, review_minutes=12)
    assert result["precision"] == 0.5
    assert result["case_recall"] == 1
    assert result["coverage"] == 1
    assert result["unresolved_findings_excluded"] == 1
    assert result["manual_review_minutes"] == 12


def test_missing_and_incomplete_reports_are_visible():
    scanned = reports()[1:]
    scanned[0].complete = False
    result = evaluate_reports(dataset(), scanned)
    assert result["case_recall"] == 0
    assert result["missing_cases"] == 1
    assert result["incomplete_cases"] == ["negative"]
    assert result["coverage"] == pytest.approx(1 / 3)


def test_different_sha_is_not_the_same_experiment():
    scanned = reports()
    scanned[0].release.to_sha = "c" * 40
    assert evaluate_reports(dataset(), scanned)["case_recall"] == 0


def test_duplicate_runs_rejected():
    with pytest.raises(ValueError, match="Multiple reports"):
        evaluate_reports(dataset(), reports() + reports())


def test_evaluate_cli(tmp_path: Path):
    source = tmp_path / "dataset.json"
    source.write_text(dataset().model_dump_json())
    directory = tmp_path / "reports"
    directory.mkdir()
    for report in reports():
        (directory / f"{report.scan_id}.json").write_text(report.model_dump_json())
    result = CliRunner().invoke(app, ["evaluate", str(source), str(directory)])
    assert result.exit_code == 0, result.output
    assert '"precision": 0.5' in result.output
