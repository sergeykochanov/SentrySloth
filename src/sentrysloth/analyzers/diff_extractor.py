"""Diff parsing, chunking by token budget, and security-relevance scoring."""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath

from unidiff import PatchedFile, PatchSet, UnidiffParseError

from sentrysloth.config import Settings
from sentrysloth.models import DiffChunk, DiffHunk

logger = logging.getLogger(__name__)

# Files to skip entirely
SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\.md$", re.IGNORECASE),
    re.compile(r"\.rst$", re.IGNORECASE),
    re.compile(r"\.txt$", re.IGNORECASE),
    re.compile(r"(^|/)docs?/"),
    re.compile(r"(^|/)test[s]?/"),
    re.compile(r"(^|/)spec[s]?/"),
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"\.(lock|sum)$"),
    re.compile(r"(^|/)\.github/"),
    re.compile(r"(^|/)\.gitignore$"),
    re.compile(r"(^|/)(?:changelog|changelog\.[^/]+)$", re.IGNORECASE),
    re.compile(r"(^|/)(?:license|license\.[^/]+)$", re.IGNORECASE),
    re.compile(r"\.(png|jpg|jpeg|gif|svg|ico|woff|ttf|eot)$", re.IGNORECASE),
    re.compile(r"(^|/)vendor/"),
    re.compile(r"(^|/)node_modules/"),
    re.compile(r"package-lock\.json$"),
    re.compile(r"yarn\.lock$"),
    re.compile(r"go\.sum$"),
    re.compile(r"Cargo\.lock$"),
    re.compile(r"poetry\.lock$"),
    # Auto-generated
    re.compile(r"_generated\.", re.IGNORECASE),
    re.compile(r"\.min\.(js|css)$"),
    re.compile(r"\.pb\.go$"),
]

# Security-relevant patterns with weights
SECURITY_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(auth|authn|authz|login|logout|session|token|jwt|oauth)\b", re.I), 0.8),
    (re.compile(r"\b(password|passwd|secret|credential|api[_-]?key)\b", re.I), 0.9),
    (re.compile(r"\b(crypto|encrypt|decrypt|hash|hmac|sign|verify|cert)\b", re.I), 0.8),
    (re.compile(r"\b(sql|query|execute|cursor|prepared)\b", re.I), 0.6),
    (re.compile(r"\b(exec|eval|system|popen|subprocess|spawn|shell)\b", re.I), 0.9),
    (re.compile(r"\b(deserializ|unpickle|yaml\.load|json\.loads|marshal)\b", re.I), 0.7),
    (re.compile(r"\b(parse|sanitiz|escap|encod|decode|strip|filter)\b", re.I), 0.5),
    (re.compile(r"\b(input|request|header|cookie|param|query_string|form)\b", re.I), 0.5),
    (re.compile(r"\b(path|file|open|read|write|mkdir|rmdir|unlink|chmod)\b", re.I), 0.6),
    (re.compile(r"\b(template|render|inject|format|interpolat)\b", re.I), 0.5),
    (re.compile(r"\b(redirect|url|href|src|origin|cors|csp|referrer)\b", re.I), 0.5),
    (re.compile(r"\b(permission|role|admin|root|sudo|privilege|acl)\b", re.I), 0.7),
    (re.compile(r"\b(ssl|tls|https|certificate|verify)\b", re.I), 0.7),
    (re.compile(r"\b(timeout|rate[_-]?limit|throttl|backoff)\b", re.I), 0.4),
    (re.compile(r"\b(socket|bind|listen|connect|proxy)\b", re.I), 0.5),
    (re.compile(r"(verify\s*=\s*False|insecure|no[_-]?verify|skip[_-]?verif)", re.I), 0.95),
    (re.compile(r"(TODO|FIXME|HACK|XXX|SECURITY|VULNERABLE)", re.I), 0.6),
]

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
}


def should_skip_file(file_path: str) -> bool:
    """Check if a file should be skipped based on path patterns."""
    return any(p.search(file_path) for p in SKIP_PATTERNS)


def detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    return LANGUAGE_MAP.get(PurePosixPath(file_path).suffix, "")


def compute_security_score(content: str) -> float:
    """Compute a heuristic security relevance score for diff content."""
    if not content:
        return 0.0
    max_score = 0.0
    for pattern, weight in SECURITY_PATTERNS:
        if pattern.search(content):
            max_score = max(max_score, weight)
    return max_score


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def sanitize_diff_content(content: str) -> str:
    """Strip potential prompt-injection patterns from diff content."""
    # Remove markdown code fences that could confuse LLM
    content = re.sub(r"```[\s\S]*?```", "[CODE_BLOCK_REMOVED]", content)
    # Remove XML-like tags (opening and closing) that could be interpreted as instructions
    tag_pattern = r"</?(?:system|user|assistant|instruction)[^>]*>"
    content = re.sub(tag_pattern, "[TAG_REMOVED]", content, flags=re.I)
    # Remove bracket-style injection markers
    bracket_pattern = r"\[(?:SYSTEM|INST|/INST)\]"
    content = re.sub(bracket_pattern, "[TAG_REMOVED]", content, flags=re.I)
    # Remove role-play injection patterns
    role_pattern = r"^(?:Human|Assistant|System)\s*:"
    content = re.sub(role_pattern, "[TAG_REMOVED]:", content, flags=re.I | re.MULTILINE)
    return content


def extract_chunks(
    diff_text: str,
    settings: Settings,
) -> list[DiffChunk]:
    """Parse unified diff and produce token-budget-limited chunks.

    Groups hunks by file, respects token budget, filters irrelevant files.
    """
    if not diff_text.strip():
        return []

    try:
        patch_set = PatchSet(diff_text)
    except (ValueError, UnidiffParseError) as exc:
        logger.warning("Failed to parse diff (%s), falling back to raw chunking", exc)
        return _fallback_chunk(diff_text, settings)

    chunks: list[DiffChunk] = []
    skipped = 0

    for patched_file in patch_set:
        file_path = patched_file.path
        if should_skip_file(file_path):
            skipped += 1
            continue

        file_chunks = _chunk_patched_file(patched_file, settings)
        chunks.extend(file_chunks)

    logger.info(
        "Extracted %d chunks from %d files (%d skipped)",
        len(chunks),
        len(patch_set) - skipped,
        skipped,
    )

    # Sort by security score descending
    chunks.sort(key=lambda c: c.security_score, reverse=True)
    return chunks


def _chunk_patched_file(
    patched_file: PatchedFile,
    settings: Settings,
) -> list[DiffChunk]:
    """Split a PatchedFile into token-budget chunks."""
    file_path = patched_file.path
    language = detect_language(file_path)
    budget = settings.chunk_token_budget

    chunks: list[DiffChunk] = []
    current_hunks: list[DiffHunk] = []
    current_raw = ""
    current_tokens = 0

    for hunk in patched_file:
        hunk_text = str(hunk)
        hunk_tokens = estimate_tokens(hunk_text)

        # If single hunk exceeds budget, it gets its own chunk (possibly truncated)
        if hunk_tokens > budget:
            # Flush current accumulator
            if current_hunks:
                chunks.append(_make_chunk(file_path, current_hunks, current_raw, language))
                current_hunks = []
                current_raw = ""
                current_tokens = 0

            truncated_text = _truncate_to_budget(hunk_text, budget)
            diff_hunk = DiffHunk(
                source_start=hunk.source_start,
                source_length=hunk.source_length,
                target_start=hunk.target_start,
                target_length=hunk.target_length,
                content=truncated_text,
                header=str(hunk).split("\n", 1)[0] if str(hunk) else "",
            )
            chunks.append(
                _make_chunk(file_path, [diff_hunk], truncated_text, language, truncated=True)
            )
            continue

        # If adding this hunk would exceed budget, flush
        if current_tokens + hunk_tokens > budget and current_hunks:
            chunks.append(_make_chunk(file_path, current_hunks, current_raw, language))
            current_hunks = []
            current_raw = ""
            current_tokens = 0

        diff_hunk = DiffHunk(
            source_start=hunk.source_start,
            source_length=hunk.source_length,
            target_start=hunk.target_start,
            target_length=hunk.target_length,
            content=hunk_text,
            header=str(hunk).split("\n", 1)[0] if str(hunk) else "",
        )
        current_hunks.append(diff_hunk)
        current_raw += hunk_text + "\n"
        current_tokens += hunk_tokens

    # Flush remaining
    if current_hunks:
        chunks.append(_make_chunk(file_path, current_hunks, current_raw, language))

    return chunks


def _make_chunk(
    file_path: str,
    hunks: list[DiffHunk],
    raw_diff: str,
    language: str,
    truncated: bool = False,
) -> DiffChunk:
    sanitized = sanitize_diff_content(raw_diff)
    return DiffChunk(
        file_path=file_path,
        hunks=hunks,
        raw_diff=sanitized,
        token_estimate=estimate_tokens(sanitized),
        security_score=compute_security_score(sanitized),
        language=language,
        truncated=truncated,
    )


def _truncate_to_budget(text: str, token_budget: int) -> str:
    """Truncate text to fit within token budget.

    This is a fallback that may break diff structure (e.g. cutting mid-hunk).
    Consumers should check the `truncated` flag on the resulting DiffChunk.
    """
    char_budget = token_budget * 4
    if len(text) <= char_budget:
        return text
    return text[:char_budget] + "\n... [truncated]"


def _fallback_chunk(diff_text: str, settings: Settings) -> list[DiffChunk]:
    """Fallback: treat entire diff as a single chunk."""
    truncated = _truncate_to_budget(diff_text, settings.chunk_token_budget)
    sanitized = sanitize_diff_content(truncated)
    return [
        DiffChunk(
            file_path="<unknown>",
            hunks=[],
            raw_diff=sanitized,
            token_estimate=estimate_tokens(sanitized),
            security_score=compute_security_score(sanitized),
        )
    ]
