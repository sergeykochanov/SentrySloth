"""Deep security analysis with pro model — only for triage-positive chunks."""

from __future__ import annotations

import logging
import time

from sentrysloth.analyzers.agentic_analysis import AgenticParseError, analyze_chunk_agentic
from sentrysloth.analyzers.analysis_shared import (
    AnalysisResponse,
    build_analysis_prompt_core,
    filter_and_convert_findings,
)
from sentrysloth.analyzers.worker_pool import run_bounded_pool
from sentrysloth.config import QuotaExhaustedMode, Settings
from sentrysloth.models import DiffChunk, Finding, LLMMetrics, TriageResult
from sentrysloth.providers.base import LLMProvider, LLMProviderError, LLMQuotaExceededError
from sentrysloth.sources.git import GitSource

logger = logging.getLogger(__name__)


def build_analysis_prompt(
    chunk: DiffChunk,
    triage: TriageResult,
    project_summary: str = "",
) -> str:
    """Build the deep analysis prompt."""
    parts = build_analysis_prompt_core(
        chunk,
        triage,
        project_summary=project_summary,
    )

    parts.append(
        "\nAnalyze this diff for security issues. If there are no real security "
        "concerns, return an empty findings list. Do not fabricate issues."
    )

    return "\n".join(parts)


async def analyze_chunk(
    chunk: DiffChunk,
    triage: TriageResult,
    provider: LLMProvider,
    settings: Settings,
    repo: str,
    project_summary: str = "",
) -> list[Finding]:
    """Run deep analysis on a single chunk."""
    findings, _in_tokens, _out_tokens, _latency_ms = await _analyze_chunk_with_metrics(
        chunk=chunk,
        triage=triage,
        provider=provider,
        settings=settings,
        repo=repo,
        project_summary=project_summary,
    )
    return findings


async def _analyze_chunk_with_metrics(
    chunk: DiffChunk,
    triage: TriageResult,
    provider: LLMProvider,
    settings: Settings,
    repo: str,
    project_summary: str = "",
) -> tuple[list[Finding], int, int, float]:
    prompt = build_analysis_prompt(chunk, triage, project_summary)

    try:
        response = await provider.generate_structured(
            prompt=prompt,
            response_model=AnalysisResponse,
            model=settings.llm.analysis_model,
            temperature=settings.llm.analysis_temperature,
            max_output_tokens=4096,
        )
    except LLMQuotaExceededError as exc:
        if settings.llm.quota_exhausted_mode == QuotaExhaustedMode.FAIL_FAST:
            raise
        logger.warning("Deep analysis quota fallback for %s: %s", chunk.file_path, exc)
        return [], 0, 0, 0.0
    except LLMProviderError as exc:
        logger.error("Deep analysis failed for %s: %s", chunk.file_path, exc)
        return [], 0, 0, 0.0

    findings = filter_and_convert_findings(response.data.findings, repo, chunk)
    return findings, response.input_tokens, response.output_tokens, response.latency_ms


async def run_deep_analysis(
    relevant_chunks: list[tuple[DiffChunk, TriageResult]],
    provider: LLMProvider,
    settings: Settings,
    repo: str,
    project_summary: str = "",
    git_source: GitSource | None = None,
    from_ref: str = "",
    to_ref: str = "",
) -> tuple[list[Finding], LLMMetrics]:
    """Run deep analysis on all triage-positive chunks.

    If git_source and refs are provided AND the provider supports tool use,
    uses the agentic multi-turn flow. Otherwise falls back to single-turn.
    """
    start = time.monotonic()

    # Determine if we can use agentic flow
    use_agentic = git_source is not None and from_ref and to_ref
    max_in_flight = max(1, int(settings.llm.scheduler_workers))
    if use_agentic:
        max_in_flight = min(max_in_flight, 2)

    async def _worker(
        idx: int,
        pair: tuple[DiffChunk, TriageResult],
    ) -> tuple[int, list[Finding], int, int, float]:
        chunk, triage = pair
        if use_agentic and git_source is not None:
            try:
                findings, in_tok, out_tok, elapsed = await analyze_chunk_agentic(
                    chunk=chunk,
                    triage=triage,
                    provider=provider,
                    git_source=git_source,
                    settings=settings,
                    repo=repo,
                    from_ref=from_ref,
                    to_ref=to_ref,
                    project_summary=project_summary,
                )
                return idx, findings, in_tok, out_tok, elapsed
            except NotImplementedError:
                logger.info("Provider does not support tool use, falling back to single-turn")
            except AgenticParseError as exc:
                logger.warning(
                    "agentic_fallback_singleturn reason=parse_error_after_repair file=%s: %s",
                    chunk.file_path,
                    exc,
                )
            except LLMProviderError as exc:
                logger.warning(
                    "Agentic analysis failed for %s: %s, falling back", chunk.file_path, exc
                )

        findings, in_tok, out_tok, elapsed = await _analyze_chunk_with_metrics(
            chunk=chunk,
            triage=triage,
            provider=provider,
            settings=settings,
            repo=repo,
            project_summary=project_summary,
        )
        return idx, findings, in_tok, out_tok, elapsed

    rows = await run_bounded_pool(relevant_chunks, _worker, max_in_flight)

    elapsed_ms = (time.monotonic() - start) * 1000

    all_findings: list[Finding] = []
    analysis_input_tokens = 0
    analysis_output_tokens = 0
    for _idx, chunk_findings, in_tokens, out_tokens, _latency in rows:
        all_findings.extend(chunk_findings)
        analysis_input_tokens += in_tokens
        analysis_output_tokens += out_tokens

    # Deduplicate by finding_id
    seen: set[str] = set()
    unique: list[Finding] = []
    for f in all_findings:
        if f.finding_id not in seen:
            seen.add(f.finding_id)
            unique.append(f)

    metrics = LLMMetrics(
        analysis_input_tokens=analysis_input_tokens,
        analysis_output_tokens=analysis_output_tokens,
        analysis_latency_ms=elapsed_ms,
    )

    logger.info(
        "Deep analysis complete: %d findings from %d chunks (%.0fms)",
        len(unique),
        len(relevant_chunks),
        elapsed_ms,
    )

    return unique, metrics
