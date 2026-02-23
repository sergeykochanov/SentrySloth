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
    # .github/ is NOT skipped -- CI workflows are security-relevant
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

# Path-based score boost: files whose path signals security relevance get an
# extra weight injected into the accumulative scoring formula.
PATH_BOOST_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"(^|/)\.github/workflows/"), 0.4),
    (re.compile(r"(^|/)\.gitlab-ci"), 0.4),
    (re.compile(r"(^|/)Dockerfile", re.I), 0.3),
    (re.compile(r"(^|/)docker-compose", re.I), 0.3),
    (re.compile(r"(^|/)(?:k8s|helm|kustomize)/"), 0.3),
    (re.compile(r"(^|/)(?:nginx|traefik|haproxy|caddy)", re.I), 0.3),
    (re.compile(r"(^|/)(?:requirements.*\.txt|setup\.py|setup\.cfg|pyproject\.toml)$", re.I), 0.3),
    (re.compile(r"(^|/)(?:pom\.xml|build\.gradle|Gemfile|Cargo\.toml)$"), 0.3),
    (re.compile(r"(^|/)package\.json$"), 0.2),
]

# ---------------------------------------------------------------------------
# Security-relevant patterns with weights.
#
# Weights feed an accumulative formula: score = 1 - prod(1 - w_i).
# Keep weights calibrated so that:
#   - A single "noisy" keyword (parse/url/format) alone stays below 0.3.
#   - Two moderate signals compound above 0.5.
#   - Any critical pattern alone pushes above 0.85.
# ---------------------------------------------------------------------------

# Critical patterns (weight >= 0.9) -- a single match is high-signal
CRITICAL_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"(verify\s*=\s*False|insecure|no[_-]?verify|skip[_-]?verif)", re.I), 0.95),
    (re.compile(r"InsecureSkipVerify", re.I), 0.95),
    (re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY", re.I), 0.95),
    (re.compile(r"AKIA[0-9A-Z]{16}"), 0.95),
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), 0.95),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), 0.95),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"), 0.95),
    (re.compile(r"sk_live_[A-Za-z0-9]{10,}"), 0.95),
    (re.compile(r"rk_live_[A-Za-z0-9]{10,}"), 0.95),
    (
        re.compile(
            r"(?:skip|disable|bypass|ignore|trust|remove).{0,40}"
            r"(?:auth|verif|csrf|tls|cert|sanitiz|permission|valid)",
            re.I,
        ),
        0.9,
    ),
    (re.compile(r"169\.254\.169\.254|metadata\.google\.internal"), 0.9),
    (re.compile(r"\b(file|gopher|dict)://", re.I), 0.9),
    (re.compile(r"\b(exec|eval|system|popen|subprocess|spawn|shell)\b", re.I), 0.9),
    (re.compile(r"\b(password|passwd|secret|credential|api[_-]?key)\b", re.I), 0.9),
]

# Standard patterns (weight 0.3 - 0.85)
SECURITY_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    *CRITICAL_PATTERNS,
    # Auth / identity
    (re.compile(r"\b(auth|authn|authz|login|logout|session|token|jwt|oauth)\b", re.I), 0.7),
    # Crypto
    (re.compile(r"\b(crypto|encrypt|decrypt|hmac|sign|cert)\b", re.I), 0.7),
    # SQL / DB
    (re.compile(r"\b(sql|query|execute|cursor|prepared)\b", re.I), 0.6),
    # Deserialization
    (re.compile(r"\b(deserializ|unpickle|yaml\.load|marshal|ObjectInputStream)\b", re.I), 0.7),
    # Permissions / privilege
    (re.compile(r"\b(permission|role|admin|root|sudo|privilege|acl)\b", re.I), 0.7),
    # TLS / certificates
    (re.compile(r"\b(ssl|tls|https|certificate)\b", re.I), 0.6),
    # File system
    (re.compile(r"\b(path|file|open|read|write|mkdir|rmdir|unlink|chmod)\b", re.I), 0.5),
    # DOM XSS sinks
    (re.compile(r"\b(innerHTML|outerHTML|dangerouslySetInnerHTML|document\.write|v-html)\b"), 0.85),
    # Web vuln keywords
    (re.compile(r"\b(csrf|xsrf|ssrf|xxe|idor|xss)\b", re.I), 0.7),
    # Debug / open-access flags
    (re.compile(r"(debug\s*=\s*True|DEBUG\s*=\s*true|allowAll|permitAll)", re.I), 0.8),
    # Network
    (re.compile(r"\b(socket|bind|listen|connect|proxy)\b", re.I), 0.4),
    # Input sources
    (re.compile(r"\b(input|request|header|cookie|param|query_string|form)\b", re.I), 0.3),
    # Inject (standalone -- high-value keyword)
    (re.compile(r"\b(inject)\b", re.I), 0.5),
    # Templating / rendering (lower -- noisy without other signals)
    (re.compile(r"\b(template|render|format|interpolat)\b", re.I), 0.2),
    # Redirect / URL (lower -- noisy alone)
    (re.compile(r"\b(redirect|origin|cors|csp|referrer)\b", re.I), 0.3),
    (re.compile(r"\b(url|href|src)\b", re.I), 0.2),
    # Sanitisation / encoding helpers (very low alone, but compound well)
    (re.compile(r"\b(parse|sanitiz|escap|encod|decode|strip|filter)\b", re.I), 0.2),
    # Weak crypto (low alone; compounds with password/hmac/sign)
    (re.compile(r"\b(md5|sha1|rc4|des|3des)\b", re.I), 0.3),
    (re.compile(r"\bECB\b"), 0.4),
    # Rate limiting / DoS
    (re.compile(r"\b(timeout|rate[_-]?limit|throttl|backoff)\b", re.I), 0.3),
    # Markers
    (re.compile(r"(TODO|FIXME|HACK|XXX|SECURITY|VULNERABLE)", re.I), 0.4),
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
    ".cc": "cpp",
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


def _extract_changed_lines(raw_diff: str) -> str:
    """Extract only added/removed lines and hunk headers from a unified diff.

    Ignores context lines (leading space) and ---/+++ file headers so that
    security scoring focuses on what actually changed.
    """
    lines: list[str] = []
    for line in raw_diff.splitlines():
        if line.startswith("@@") or (
            line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ):
            lines.append(line)
    return "\n".join(lines)


_COMMENT_PREFIX_RE = re.compile(r"^\s*(?://|#|/\*|\*/?\s*|\*)\s*")


def _is_noise_only_change(raw_diff: str) -> bool:
    """Detect whitespace-only or comment-only changes.

    Compares added vs removed lines after normalising whitespace (and
    optionally stripping comment prefixes).  When the normalised sets are
    equal the diff is cosmetic.
    """
    added: list[str] = []
    removed: list[str] = []
    for line in raw_diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])

    if not added and not removed:
        return True

    def _normalize_ws(lines: list[str]) -> list[str]:
        return sorted("".join(s.split()) for s in lines)

    if _normalize_ws(added) == _normalize_ws(removed):
        return True

    def _strip_comments(lines: list[str]) -> list[str]:
        return sorted("".join(_COMMENT_PREFIX_RE.sub("", s).split()) for s in lines)

    if _strip_comments(added) == _strip_comments(removed):
        return True

    def _is_comment_line(line: str) -> bool:
        stripped = line.lstrip()
        return stripped.startswith(("#", "//", "/*", "*/", "*"))

    return all(_is_comment_line(s) for s in added) and all(_is_comment_line(s) for s in removed)


def _path_boost_weight(file_path: str) -> float:
    """Return the highest path-based boost weight for a file, or 0."""
    best = 0.0
    for pat, w in PATH_BOOST_PATTERNS:
        if pat.search(file_path):
            best = max(best, w)
    return best


def compute_security_score(
    content: str,
    *,
    changed_lines_only: str = "",
    file_path: str = "",
) -> float:
    """Compute a heuristic security relevance score for diff content.

    Uses an accumulative formula: ``1 - prod(1 - w_i)`` over all triggered
    pattern weights so that multiple moderate signals compound into a higher
    score (e.g. input 0.3 + sql 0.6 -> 0.72).

    When *changed_lines_only* is provided the patterns are matched against
    that text (only ``+``/``-`` lines), reducing noise from diff context.

    When *file_path* is provided, path-based boost weights from
    ``PATH_BOOST_PATTERNS`` are injected into the formula.
    """
    text = changed_lines_only or content
    if not text:
        return 0.0
    triggered = [w for pat, w in SECURITY_PATTERNS if pat.search(text)]
    if file_path:
        path_w = _path_boost_weight(file_path)
        if path_w > 0:
            triggered.append(path_w)
    if not triggered:
        return 0.0
    score = 1.0
    for w in triggered:
        score *= 1.0 - w
    return round(1.0 - score, 4)


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
    except (ValueError, UnidiffParseError, UnboundLocalError) as exc:
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
    changed = _extract_changed_lines(sanitized)
    if _is_noise_only_change(sanitized):
        sec_score = 0.0
    else:
        sec_score = compute_security_score(
            sanitized, changed_lines_only=changed, file_path=file_path
        )
    return DiffChunk(
        file_path=file_path,
        hunks=hunks,
        raw_diff=sanitized,
        token_estimate=estimate_tokens(sanitized),
        security_score=sec_score,
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
    changed = _extract_changed_lines(sanitized)
    return [
        DiffChunk(
            file_path="<unknown>",
            hunks=[],
            raw_diff=sanitized,
            token_estimate=estimate_tokens(sanitized),
            security_score=compute_security_score(
                sanitized, changed_lines_only=changed, file_path="<unknown>"
            ),
        )
    ]
