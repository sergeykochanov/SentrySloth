"""JSON reporter — outputs scan results as JSON."""

from __future__ import annotations

from sentrysloth.models import ScanResult


def generate_json_report(result: ScanResult, indent: int = 2) -> str:
    """Generate a JSON report from scan results."""
    return result.model_dump_json(indent=indent)
