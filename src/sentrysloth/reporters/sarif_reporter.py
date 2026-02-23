"""SARIF reporter — GitHub Code Scanning compatible output."""

from __future__ import annotations

import json

from sentrysloth import __version__
from sentrysloth.models import ScanResult, Severity

# SARIF severity mapping
SARIF_LEVEL = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}


def generate_sarif_report(result: ScanResult) -> str:
    """Generate a SARIF 2.1.0 report."""
    rules: list[dict] = []
    results: list[dict] = []
    rule_index: dict[str, int] = {}

    for finding in result.findings:
        rule_id = f"sentrysloth/{finding.finding_type.value}"
        if rule_id not in rule_index:
            rule_index[rule_id] = len(rules)
            rules.append(
                {
                    "id": rule_id,
                    "shortDescription": {
                        "text": finding.finding_type.value.replace("_", " ").title()
                    },
                    "defaultConfiguration": {
                        "level": SARIF_LEVEL.get(finding.severity, "note"),
                    },
                    "properties": {
                        "tags": ["security"],
                    },
                }
            )

        locations: list[dict] = []
        for ev in finding.evidence:
            locations.append(
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": ev.code.file_path},
                        "region": {
                            "startLine": ev.code.start_line,
                            "endLine": ev.code.end_line,
                        },
                    },
                }
            )

        results.append(
            {
                "ruleId": rule_id,
                "ruleIndex": rule_index[rule_id],
                "level": SARIF_LEVEL.get(finding.severity, "note"),
                "message": {
                    "text": f"{finding.title}\n\n{finding.description}",
                },
                "locations": locations
                or [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": finding.file_path},
                            "region": {"startLine": 1},
                        },
                    }
                ],
                "fingerprints": {
                    "sentrysloth/finding_id": finding.finding_id,
                },
                "properties": {
                    "confidence": finding.confidence.value,
                    "severity": finding.severity.value,
                },
            }
        )

    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "sentrysloth",
                        "version": __version__,
                        "informationUri": "https://github.com/sergeykochanov/SentrySloth",
                        "rules": rules,
                    },
                },
                "results": results,
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "properties": {
                            "scanId": result.scan_id,
                            "fromRef": result.release.from_ref,
                            "toRef": result.release.to_ref,
                        },
                    }
                ],
            }
        ],
    }

    return json.dumps(sarif, indent=2)
