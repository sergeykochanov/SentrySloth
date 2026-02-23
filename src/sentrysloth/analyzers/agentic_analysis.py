"""Agentic deep analysis: LLM uses tools to explore the codebase before concluding."""

from __future__ import annotations

import json
import logging
import time

from pydantic import ValidationError

from sentrysloth.analyzers.analysis_shared import (
    AnalysisResponse,
    build_analysis_prompt_core,
    filter_and_convert_findings,
)
from sentrysloth.analyzers.diff_extractor import sanitize_diff_content
from sentrysloth.config import Settings
from sentrysloth.models import DiffChunk, Finding, TriageResult
from sentrysloth.providers.base import LLMProvider, LLMProviderError, ToolCallResponse
from sentrysloth.sources.git import GitSource, GitSourceError

logger = logging.getLogger(__name__)

# Tool definitions in OpenAI function-calling format.
# Gemini provider maps these automatically.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the current version of a file in the repository. "
                "Use this to understand how the changed code is used in its wider context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file relative to the repository root.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_before",
            "description": (
                "Read the file as it was BEFORE the change (old version). "
                "Useful to compare what existed before the diff was applied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file relative to the repository root.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search the codebase for a text pattern using fixed-string git grep (-F). "
                "Use this to find callers, usages, or related code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Literal text to search for (not regex). Use plain snippets "
                            "such as function names, call fragments, or config keys."
                        ),
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Optional glob to filter files, e.g. '*.py'.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]

MAX_FILE_CONTENT_CHARS = 5000
MAX_SEARCH_RESULTS = 20
MAX_REPAIR_SOURCE_CHARS = 12000


class AgenticParseError(Exception):
    """Raised when final agentic response cannot be recovered to valid JSON."""


def _build_initial_messages(
    chunk: DiffChunk,
    triage: TriageResult,
    project_summary: str = "",
) -> list[dict]:
    """Build the initial message list for the agentic conversation."""
    parts = build_analysis_prompt_core(chunk, triage, project_summary=project_summary)

    parts.append(
        "\nAnalyze this diff for security issues. You have tools available to explore "
        "the codebase: read_file, read_file_before, and search_code. Use them to "
        "understand the full context before making your assessment.\n\n"
        "When you are done investigating, respond with a JSON object matching this schema:\n"
        '{"findings": [...], "summary": "..."}\n\n'
        "Each finding must have: title, description, finding_type, "
        "severity (medium/high/critical only), "
        "confidence, evidence (ARRAY of objects, each with: description, file_path, "
        "start_line, end_line, snippet, reasoning), "
        "cwe_ids, recommendation.\n\n"
        'If there are no real security concerns, return {"findings": [], "summary": "..."}.'
    )

    return [{"role": "user", "content": "\n".join(parts)}]


async def _execute_tool_call(
    name: str,
    arguments: dict,
    git_source: GitSource,
    from_ref: str,
    to_ref: str,
    cache: dict,
) -> str:
    """Execute a single tool call and return the result as a string."""
    cache_key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        if name == "read_file":
            file_path = arguments.get("file_path", "")
            content = await git_source.get_file_content(to_ref, file_path)
            if content is None:
                result = f"File not found: {file_path}"
            else:
                result = content[:MAX_FILE_CONTENT_CHARS]
                if len(content) > MAX_FILE_CONTENT_CHARS:
                    result += f"\n... (truncated, {len(content)} total chars)"

        elif name == "read_file_before":
            file_path = arguments.get("file_path", "")
            content = await git_source.get_file_content(from_ref, file_path)
            if content is None:
                result = f"File not found at ref {from_ref}: {file_path}"
            else:
                result = content[:MAX_FILE_CONTENT_CHARS]
                if len(content) > MAX_FILE_CONTENT_CHARS:
                    result += f"\n... (truncated, {len(content)} total chars)"

        elif name == "search_code":
            pattern = arguments.get("pattern", "")
            file_glob = arguments.get("file_glob", "")
            matches = await git_source.search_code(
                pattern, to_ref, file_glob=file_glob, max_results=MAX_SEARCH_RESULTS
            )
            if not matches:
                result = f"No matches found for pattern: {pattern}"
            else:
                lines = [f"{m['file']}:{m['line']}: {m['content']}" for m in matches]
                result = "\n".join(lines)

        else:
            result = f"Unknown tool: {name}"

    except (GitSourceError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Tool call failed (%s): %s", name, exc)
        result = f"Error executing {name}: {exc}"

    result = sanitize_diff_content(result)
    cache[cache_key] = result
    return result


def _try_parse_final_response(
    response: ToolCallResponse,
    repo: str,
    chunk: DiffChunk,
) -> tuple[list[Finding], bool]:
    """Parse final LLM response and return (findings, parse_ok)."""
    text = response.content.strip()
    if not text:
        return [], True

    # Try to extract JSON from the response (may be wrapped in markdown code blocks)
    if "```" in text:
        # Extract content between code fences
        parts = text.split("```")
        for part in parts[1::2]:  # odd indices are inside fences
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                text = cleaned
                break

    try:
        parsed = json.loads(text)
        analysis = AnalysisResponse.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Failed to parse agentic analysis response: %s", exc)
        return [], False

    return filter_and_convert_findings(analysis.findings, repo, chunk), True


def _parse_final_response(
    response: ToolCallResponse,
    repo: str,
    chunk: DiffChunk,
) -> list[Finding]:
    """Backward-compatible parser helper used by existing tests/callers."""
    findings, _ok = _try_parse_final_response(response, repo, chunk)
    return findings


async def _repair_final_response(
    provider: LLMProvider,
    settings: Settings,
    messages: list[dict],
    invalid_response: ToolCallResponse,
) -> ToolCallResponse:
    """Request one JSON-repair pass for an invalid final response."""
    repair_prompt = (
        "Your previous final answer was invalid JSON and could not be parsed.\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        '{"findings": [...], "summary": "..."}\n'
        "Do not include markdown fences or extra text.\n\n"
        "Previous invalid response:\n"
        f"{sanitize_diff_content(invalid_response.content)[:MAX_REPAIR_SOURCE_CHARS]}"
    )
    repair_messages = list(messages)
    repair_messages.append({"role": "user", "content": repair_prompt})
    return await provider.generate_with_tools(
        messages=repair_messages,
        tools=[],
        model=settings.llm.analysis_model,
        temperature=settings.llm.analysis_temperature,
        max_output_tokens=4096,
    )


async def analyze_chunk_agentic(
    chunk: DiffChunk,
    triage: TriageResult,
    provider: LLMProvider,
    git_source: GitSource,
    settings: Settings,
    repo: str,
    from_ref: str,
    to_ref: str,
    project_summary: str = "",
    *,
    max_turns: int = 10,
) -> tuple[list[Finding], int, int, float]:
    """Run agentic analysis on a single chunk with multi-turn tool use.

    Returns (findings, total_input_tokens, total_output_tokens, elapsed_ms).
    """
    start = time.monotonic()
    messages = _build_initial_messages(chunk, triage, project_summary=project_summary)
    tool_cache: dict[str, str] = {}

    total_input_tokens = 0
    total_output_tokens = 0
    token_budget = (
        settings.llm.analysis_max_input_tokens * 2
        if settings.llm.analysis_max_input_tokens
        else 50000
    )

    async def _finalize_with_repair(
        final_response: ToolCallResponse,
        *,
        phase: str,
    ) -> tuple[list[Finding], int, int]:
        findings, parse_ok = _try_parse_final_response(final_response, repo, chunk)
        if parse_ok:
            return findings, 0, 0

        logger.warning("agentic_parse_errors chunk=%s phase=%s", chunk.file_path, phase)
        logger.warning("agentic_repair_attempts chunk=%s phase=%s", chunk.file_path, phase)
        try:
            repaired_response = await _repair_final_response(
                provider,
                settings,
                messages,
                final_response,
            )
        except (LLMProviderError, NotImplementedError) as exc:
            raise AgenticParseError(
                f"JSON repair request failed for {chunk.file_path}: {exc}"
            ) from exc

        repaired_findings, repaired_ok = _try_parse_final_response(repaired_response, repo, chunk)
        if repaired_ok:
            logger.info("agentic_repair_success chunk=%s phase=%s", chunk.file_path, phase)
            return (
                repaired_findings,
                repaired_response.input_tokens,
                repaired_response.output_tokens,
            )

        raise AgenticParseError(
            f"Final response remained invalid after JSON repair for {chunk.file_path}"
        )

    for turn in range(max_turns):
        try:
            response = await provider.generate_with_tools(
                messages=messages,
                tools=TOOLS,
                model=settings.llm.analysis_model,
                temperature=settings.llm.analysis_temperature,
                max_output_tokens=4096,
            )
        except (LLMProviderError, NotImplementedError) as exc:
            logger.error(
                "Agentic analysis failed on turn %d for %s: %s", turn, chunk.file_path, exc
            )
            if turn == 0 and isinstance(exc, NotImplementedError):
                raise  # Let caller handle fallback to single-turn
            break

        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens

        if not response.tool_calls:
            # Final response — parse findings
            findings, repair_in_tok, repair_out_tok = await _finalize_with_repair(
                response,
                phase="final_response",
            )
            total_input_tokens += repair_in_tok
            total_output_tokens += repair_out_tok
            elapsed_ms = (time.monotonic() - start) * 1000
            return findings, total_input_tokens, total_output_tokens, elapsed_ms

        # Execute tool calls and build messages for next turn
        # Add assistant message with tool calls
        assistant_msg: dict = {"role": "assistant", "content": response.content or None}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in response.tool_calls
        ]
        messages.append(assistant_msg)

        # Execute each tool call and add results
        for tc in response.tool_calls:
            result = await _execute_tool_call(
                tc.name, tc.arguments, git_source, from_ref, to_ref, tool_cache
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result,
                }
            )

        # Check token budget
        if total_input_tokens + total_output_tokens > token_budget:
            logger.warning(
                "Token budget exceeded for %s (%d tokens), requesting final answer",
                chunk.file_path,
                total_input_tokens + total_output_tokens,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Token budget reached. Please provide your final analysis now as JSON. "
                        "Do not make further tool calls."
                    ),
                }
            )
            try:
                response = await provider.generate_with_tools(
                    messages=messages,
                    tools=[],  # No tools — force text response
                    model=settings.llm.analysis_model,
                    temperature=settings.llm.analysis_temperature,
                    max_output_tokens=4096,
                )
                total_input_tokens += response.input_tokens
                total_output_tokens += response.output_tokens
            except LLMProviderError as exc:
                logger.error("Final answer request failed for %s: %s", chunk.file_path, exc)
                break
            findings, repair_in_tok, repair_out_tok = await _finalize_with_repair(
                response,
                phase="token_budget",
            )
            total_input_tokens += repair_in_tok
            total_output_tokens += repair_out_tok
            elapsed_ms = (time.monotonic() - start) * 1000
            return findings, total_input_tokens, total_output_tokens, elapsed_ms
    else:
        # All turns genuinely exhausted — force a final answer
        logger.warning(
            "Max turns (%d) exhausted for %s, requesting final answer",
            max_turns,
            chunk.file_path,
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have used all available tool turns. Please provide your final "
                    "analysis now as JSON. Do not make further tool calls."
                ),
            }
        )
        try:
            response = await provider.generate_with_tools(
                messages=messages,
                tools=[],  # No tools — force text response
                model=settings.llm.analysis_model,
                temperature=settings.llm.analysis_temperature,
                max_output_tokens=4096,
            )
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            findings, repair_in_tok, repair_out_tok = await _finalize_with_repair(
                response,
                phase="max_turns",
            )
            total_input_tokens += repair_in_tok
            total_output_tokens += repair_out_tok
            elapsed_ms = (time.monotonic() - start) * 1000
            return findings, total_input_tokens, total_output_tokens, elapsed_ms
        except (LLMProviderError, NotImplementedError) as exc:
            logger.error("Final answer after max turns failed for %s: %s", chunk.file_path, exc)

    elapsed_ms = (time.monotonic() - start) * 1000
    return [], total_input_tokens, total_output_tokens, elapsed_ms
