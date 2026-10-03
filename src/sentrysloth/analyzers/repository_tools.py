"""Read-only, revision-specific source access for discovery and verification."""

from __future__ import annotations

import ast
import json

from sentrysloth.sources.git import GitSource, GitSourceError

_RANGE = {
    "file_path": {"type": "string"},
    "revision": {"type": "string", "enum": ["before", "after"]},
    "start_line": {"type": "integer", "minimum": 1},
    "end_line": {"type": "integer", "minimum": 1},
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _tool(
        "read_file",
        "Read a numbered source range at before/after revision. Follow next_line to continue.",
        _RANGE,
        ["file_path"],
    ),
    _tool(
        "read_file_before",
        "Read the old revision of a file, with optional line range.",
        _RANGE,
        ["file_path"],
    ),
    _tool(
        "read_symbol",
        "Locate Python function/class bodies, including nested methods. "
        "Read returned ranges with read_file; for other languages use search_code.",
        {
            "file_path": {"type": "string"},
            "symbol": {"type": "string"},
            "revision": {"type": "string", "enum": ["before", "after"]},
        },
        ["file_path", "symbol"],
    ),
    _tool(
        "search_code",
        "Literal source search at before/after revision. Includes adjacent lines. "
        "Use next_cursor for more matches.",
        {
            "pattern": {"type": "string"},
            "file_glob": {"type": "string"},
            "revision": {"type": "string", "enum": ["before", "after"]},
            "cursor": {"type": "integer", "minimum": 0},
        },
        ["pattern"],
    ),
    _tool(
        "list_changes",
        "List all changed paths, including tests, docs and dependencies; paginate with cursor.",
        {
            "cursor": {"type": "integer", "minimum": 0},
        },
        [],
    ),
    _tool(
        "read_diff",
        "Read the change to one file, including tests/docs/dependency manifests. "
        "Paginate with start_line.",
        {
            "file_path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
        },
        ["file_path"],
    ),
]


def source_range(content: str, start: int = 1, end: int | None = None) -> dict:
    """Return a bounded range without hiding whether more lines remain."""
    if start < 1 or (end is not None and end < start):
        raise ValueError("Invalid line range")
    lines = content.splitlines()
    stop = min(len(lines), start + 199, end if end is not None else start + 199)
    selected = lines[start - 1 : stop]
    # Preserve full lines: a long literal must not silently lose its suffix.
    return {
        "start_line": start,
        "end_line": stop,
        "total_lines": len(lines),
        "next_line": stop + 1 if stop < len(lines) else None,
        "content": "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start)),
    }


def python_symbols(content: str, symbol: str) -> list[dict]:
    tree = ast.parse(content)
    return [
        {"name": n.name, "start_line": n.lineno, "end_line": n.end_lineno}
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.name == symbol.rsplit(".", 1)[-1]
    ]


async def execute_tool_call(
    name: str, arguments: dict, git_source: GitSource, from_ref: str, to_ref: str, cache: dict
) -> str:
    cache_key = f"{from_ref}:{to_ref}:{name}:{json.dumps(arguments, sort_keys=True)}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        revision = arguments.get("revision", "after")
        if revision not in ("before", "after"):
            raise ValueError("revision must be before or after")
        ref = from_ref if name == "read_file_before" or revision == "before" else to_ref
        path = arguments.get("file_path", "")
        if name in ("read_file", "read_file_before", "read_symbol"):
            content = await git_source.get_file_content(ref, path)
            if content is None:
                result = {"error": f"File not found: {path}", "ref": ref}
            elif name == "read_symbol":
                result = {
                    "ref": ref,
                    "file_path": path,
                    "symbols": python_symbols(content, arguments.get("symbol", "")),
                }
            else:
                result = {
                    "ref": ref,
                    "file_path": path,
                    **source_range(
                        content, int(arguments.get("start_line", 1)), arguments.get("end_line")
                    ),
                }
        elif name == "search_code":
            cursor = int(arguments.get("cursor", 0))
            if cursor < 0:
                raise ValueError("cursor must be nonnegative")
            matches = await git_source.search_code(
                arguments.get("pattern", ""),
                ref,
                file_glob=arguments.get("file_glob", ""),
                max_results=21,
                offset=cursor,
            )
            page = matches[:20]
            for match in page:
                content = await git_source.get_file_content(ref, match["file"])
                if content is not None:
                    match["context"] = source_range(
                        content, max(1, match["line"] - 3), match["line"] + 3
                    )
            result = {
                "ref": ref,
                "matches": page,
                "next_cursor": cursor + 20 if len(matches) > 20 else None,
            }
        elif name == "list_changes":
            cursor = int(arguments.get("cursor", 0))
            if cursor < 0:
                raise ValueError("cursor must be nonnegative")
            paths = await git_source.changed_files(from_ref, to_ref)
            result = {
                "files": paths[cursor : cursor + 50],
                "total": len(paths),
                "next_cursor": cursor + 50 if cursor + 50 < len(paths) else None,
            }
        elif name == "read_diff":
            content = await git_source.get_file_diff(from_ref, to_ref, path)
            result = {
                "from_ref": from_ref,
                "to_ref": to_ref,
                "file_path": path,
                **source_range(content, int(arguments.get("start_line", 1))),
            }
        else:
            result = {"error": f"Unknown tool: {name}"}
    except (GitSourceError, RuntimeError, TypeError, ValueError, SyntaxError) as exc:
        result = {"error": f"Error executing {name}: {exc}"}
    serialized = json.dumps(result, ensure_ascii=True)
    cache[cache_key] = serialized
    return serialized
