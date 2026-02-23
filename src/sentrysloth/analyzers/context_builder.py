"""Context builder: assemble surrounding code, function signatures, callers."""

from __future__ import annotations

import asyncio
import logging
import re

from sentrysloth.models import DiffChunk
from sentrysloth.sources.git import GitSource

logger = logging.getLogger(__name__)

# Patterns to extract function/class definitions across common languages
FUNC_PATTERNS: list[re.Pattern[str]] = [
    # Python: def/async def/class
    re.compile(r"^((?:async\s+)?def\s+\w+\s*\([^)]*\)(?:\s*->.*?)?):?", re.MULTILINE),
    re.compile(r"^(class\s+\w+(?:\([^)]*\))?):?", re.MULTILINE),
    # JS/TS: function, arrow, class
    re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+\s*\([^)]*\)", re.MULTILINE),
    re.compile(r"^(?:export\s+)?class\s+\w+", re.MULTILINE),
    # Go: func
    re.compile(r"^func\s+(?:\([^)]*\)\s*)?\w+\s*\([^)]*\)", re.MULTILINE),
    # Rust: fn, impl
    re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+\w+", re.MULTILINE),
    re.compile(r"^impl\b.*\{", re.MULTILINE),
    # Java/C#: method/class signatures
    re.compile(
        r"^(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\([^)]*\)", re.MULTILINE
    ),
]

# Pattern to extract function/method name from a definition line
FUNC_NAME_RE = re.compile(r"(?:def|function|func|fn|class)\s+(\w+)")


class ContextBuilder:
    """Build surrounding context for diff chunks."""

    def __init__(self, git_source: GitSource, to_ref: str) -> None:
        self.git_source = git_source
        self.to_ref = to_ref
        self._file_cache: dict[str, str | None] = {}
        self._file_locks: dict[str, asyncio.Lock] = {}

    async def enrich_chunk(self, chunk: DiffChunk) -> DiffChunk:
        """Add function signatures and surrounding context to a chunk."""
        file_content = await self._get_file(chunk.file_path)
        if file_content is None:
            return chunk

        signatures = extract_function_signatures(file_content)
        relevant_sigs = self._find_relevant_signatures(file_content, chunk, signatures)

        surrounding = self._extract_surrounding_context(file_content, chunk)

        chunk.function_signatures = relevant_sigs
        chunk.context = surrounding
        return chunk

    async def _get_file(self, file_path: str) -> str | None:
        lock = self._file_locks.setdefault(file_path, asyncio.Lock())
        async with lock:
            if file_path not in self._file_cache:
                self._file_cache[file_path] = await self.git_source.get_file_content(
                    self.to_ref, file_path
                )
            return self._file_cache[file_path]

    def _find_relevant_signatures(
        self,
        file_content: str,
        chunk: DiffChunk,
        signatures: list[tuple[int, str]],
    ) -> list[str]:
        """Find function signatures that contain or are near the changed lines."""
        if not chunk.hunks:
            return [sig for _, sig in signatures[:5]]

        changed_lines: set[int] = set()
        for hunk in chunk.hunks:
            for i in range(hunk.target_start, hunk.target_start + hunk.target_length):
                changed_lines.add(i)

        relevant: list[str] = []
        for line_no, sig in signatures:
            # Check if this signature is within or near any changed region
            if any(abs(line_no - cl) < 30 for cl in changed_lines):
                relevant.append(sig)

        return relevant[:10]

    def _extract_surrounding_context(
        self,
        file_content: str,
        chunk: DiffChunk,
        context_lines: int = 20,
    ) -> str:
        """Extract lines surrounding the changed hunks."""
        if not chunk.hunks:
            return ""

        lines = file_content.splitlines()
        collected: list[str] = []

        for hunk in chunk.hunks:
            start = max(0, hunk.target_start - context_lines - 1)
            end = min(len(lines), hunk.target_start + hunk.target_length + context_lines)

            context_block = "\n".join(
                f"{i + 1}: {line}" for i, line in enumerate(lines[start:end], start=start)
            )
            collected.append(context_block)

        return "\n---\n".join(collected)


def extract_function_signatures(content: str) -> list[tuple[int, str]]:
    """Extract (line_number, signature) pairs from source code."""
    results: list[tuple[int, str]] = []
    for pattern in FUNC_PATTERNS:
        for match in pattern.finditer(content):
            line_no = content[: match.start()].count("\n") + 1
            sig = match.group(0).strip()
            results.append((line_no, sig))

    # Deduplicate and sort by line number
    seen: set[str] = set()
    unique: list[tuple[int, str]] = []
    for ln, sig in sorted(results, key=lambda x: x[0]):
        if sig not in seen:
            seen.add(sig)
            unique.append((ln, sig))
    return unique


def extract_function_name(signature: str) -> str | None:
    """Extract function/method name from a signature line."""
    m = FUNC_NAME_RE.search(signature)
    return m.group(1) if m else None
