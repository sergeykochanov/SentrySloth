"""Markdown reporter — human-readable security report."""

from __future__ import annotations

from sentrysloth.models import Finding, ScanResult, Severity

SEVERITY_EMOJI = {
    Severity.CRITICAL: "!!",
    Severity.HIGH: "!",
    Severity.MEDIUM: "*",
    Severity.LOW: "-",
    Severity.INFO: ".",
}


def generate_markdown_report(result: ScanResult) -> str:
    """Generate a Markdown report from scan results."""
    lines: list[str] = []

    lines.append("# Security Scan Report")
    lines.append("")
    lines.append(f"**Repository**: {result.release.repo_url}")
    lines.append(f"**Changes**: {result.release.from_ref} -> {result.release.to_ref}")
    lines.append(f"**Scan ID**: {result.scan_id}")
    lines.append(f"**Date**: {result.started_at.isoformat()}")
    lines.append("")

    # Summary stats
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Files changed: {result.release.total_files_changed}")
    lines.append(f"- Additions: +{result.release.total_additions}")
    lines.append(f"- Deletions: -{result.release.total_deletions}")
    lines.append(f"- **Findings: {len(result.findings)}**")

    if result.triage_stats:
        lines.append(f"- Chunks analyzed: {result.triage_stats.total_chunks}")
        lines.append(f"- Security-relevant: {result.triage_stats.security_relevant}")
        lines.append(f"- Filtered out: {result.triage_stats.filtered_out}")
    lines.append("")

    if not result.findings:
        lines.append("No security-relevant findings detected.")
        return "\n".join(lines)

    # Findings by severity
    by_severity: dict[Severity, list[Finding]] = {}
    for f in result.findings:
        by_severity.setdefault(f.severity, []).append(f)

    lines.append("## Findings")
    lines.append("")

    for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
        findings = by_severity.get(sev, [])
        if not findings:
            continue

        lines.append(f"### {sev.value.upper()} ({len(findings)})")
        lines.append("")

        for f in findings:
            lines.append(f"#### [{SEVERITY_EMOJI[sev]}] {f.title}")
            lines.append("")
            lines.append(
                f"**Type**: {f.finding_type.value} | "
                f"**Confidence**: {f.confidence.value} | "
                f"**ID**: `{f.finding_id}`"
            )
            lines.append("")
            lines.append(f"**File**: `{f.file_path}`")
            lines.append("")
            lines.append(f"{f.description}")
            lines.append("")

            if f.cwe:
                cwe_str = ", ".join(f"{c.id}" for c in f.cwe)
                lines.append(f"**CWE**: {cwe_str}")
                lines.append("")

            for i, ev in enumerate(f.evidence, 1):
                lines.append(f"**Evidence {i}**: {ev.description}")
                if ev.code.snippet:
                    lines.append("```")
                    lines.append(f"// {ev.code.file_path}:{ev.code.start_line}-{ev.code.end_line}")
                    lines.append(ev.code.snippet)
                    lines.append("```")
                lines.append(f"*Reasoning*: {ev.reasoning}")
                lines.append("")

            if f.recommendation:
                lines.append(f"**Recommendation**: {f.recommendation}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # LLM metrics
    if result.llm_metrics:
        lines.append("## Metrics")
        lines.append("")
        lines.append(f"- Triage tokens: in={result.llm_metrics.triage_input_tokens}")
        lines.append(f"- Triage tokens: out={result.llm_metrics.triage_output_tokens}")
        lines.append(f"- Analysis tokens: in={result.llm_metrics.analysis_input_tokens}")
        lines.append(f"- Analysis tokens: out={result.llm_metrics.analysis_output_tokens}")
        lines.append(f"- Triage latency: {result.llm_metrics.triage_latency_ms:.0f}ms")
        lines.append(f"- Analysis latency: {result.llm_metrics.analysis_latency_ms:.0f}ms")
        lines.append("")

    return "\n".join(lines)
