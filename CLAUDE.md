# SentrySloth — Project Guide

## What is this?

LLM-powered security review tool for open-source release diffs. Two-stage pipeline: fast triage model → deep analysis model. Outputs: JSON, Markdown, SARIF.

## Build & Test

```bash
pip install -e ".[dev]"              # install dev mode
.venv/bin/python -m pytest tests/    # run all tests (asyncio_mode=auto)
.venv/bin/ruff check src/ tests/     # lint
.venv/bin/ruff format src/ tests/    # format
```

## Project Layout

```
src/sentrysloth/
├── cli.py                  # Typer CLI: scan, batch-scan, list-versions, report, cache-info; ScanOutput
├── batch.py                # Batch scanning: resolve repos/pairs, run sequential per-repo scans with repo-level concurrency
├── config.py               # Pydantic Settings (SENTRYSLOTH_* env vars)
├── models.py               # Domain types: Finding, DiffChunk, ScanResult, RepoProfile, enums
├── analyzers/
│   ├── diff_extractor.py   # Parse unified diffs → DiffChunk (with security pre-score)
│   ├── triage.py           # Fast LLM pass: is chunk security-relevant?
│   ├── deep_analysis.py    # Deep LLM pass: structured findings (single-turn)
│   ├── agentic_analysis.py # Deep LLM pass: multi-turn with tools (read_file, search_code)
│   └── context_builder.py  # Enrich chunks with function signatures & surrounding code
├── cache/
│   ├── storage.py          # Async SQLite: scan history, repo profiles, profile history
│   └── repo_profile.py     # RepoProfile bootstrap/update + prompt serialization helpers
├── providers/
│   ├── base.py             # Abstract LLMProvider, ToolCallResponse, error types
│   ├── gemini.py           # Google Gemini implementation
│   ├── openai_compat.py    # OpenAI-compatible (Grok, etc.)
│   ├── scheduler.py        # Async queue + RPM rate limiting wrapper
│   └── __init__.py         # create_provider() factory
├── sources/
│   └── git.py              # Git operations: clone, diff, file content, search, list_tags_with_dates, list_files
└── reporters/
    ├── json_reporter.py
    ├── markdown_reporter.py
    └── sarif_reporter.py

tests/
├── conftest.py             # Shared fixtures, DummyProvider
├── fixtures/               # Test diff files
└── test_*.py               # 15 test modules, one per source module
```

## Architecture

```
Single scan:
  Git Source → Diff Extractor (chunks + security pre-score)
    → RepoProfile load/bootstrap (SQLite cache + triage-model bootstrap for new repos)
    → Context Builder (function signatures, surrounding code)
    → LLM Scheduler (FIFO queue, RPM limiting, quota tracking)
    → Triage (fast model) — filter noise
    → Deep Analysis (analysis model) — extract findings
      ├── Single-turn: generate_structured (constrained JSON via response_model)
      └── Agentic: generate_with_tools (multi-turn, tools: read_file, read_file_before, search_code)
    → RepoProfile incremental update (triage model; bounded JSON profile)
    → Baseline suppression → Confidence filter → Reporter

Batch scan:
  repos.txt → load_repo_list() → for each repo:
    → clone/fetch → list_tags_with_dates()
    → build_tag_pairs(--last-releases | --since)
      (for --last-releases: latest major stream first, then backfill older majors)
    → run repos in parallel (repo-level --concurrency)
      with sequential _run_scan() per pair inside each repo (old→new)
    → Rich summary table
```

## Key Patterns

- **Provider abstraction**: `LLMProvider` ABC → `GeminiProvider`, `OpenAICompatProvider`, `LlmRequestScheduler` (decorator)
- **Structured LLM output**: Pydantic models as `response_model` → constrained decoding (single-turn) or free-form JSON + defensive validators (agentic)
- **Defensive LLM parsing**: `field_validator(mode="before")` to handle non-deterministic output (dict→list coercion, missing optional fields)
- **Async-first**: all I/O async (httpx, aiosqlite, `asyncio.to_thread` for git)
- **Security pre-scoring**: 43 regex patterns with weights in `diff_extractor.py` to prioritize chunks before LLM
- **Accumulated context**: bounded `RepoProfile` is bootstrapped for new repos and updated after each scan; deep/agentic analysis injects profile context into prompts
- **Batch orchestration**: `batch.py` accepts `scan_fn: ScanFn` callback to avoid circular imports with `cli.py`; sequential pairs per repo with repo-level concurrency
- **ScanOutput abstraction**: `ScanOutput` class in `cli.py` unifies verbose console and progress bar output for `_run_scan`. Methods: `phase()` (update progress bar description), `info()` (print, suppressed in progress mode), `error()` (always visible). Property `quiet` signals downstream functions (e.g. `_output_result`) to suppress informational messages. Batch mode creates `ScanOutput` with a `Progress`/`TaskID`; single `scan` command gets a plain console-only instance.

## Config

Env vars with `SENTRYSLOTH_` prefix, nested `__` delimiter.
Default provider is `grok` with:
- `SENTRYSLOTH_LLM_TRIAGE_MODEL=grok-4-1-fast-non-reasoning`
- `SENTRYSLOTH_LLM_ANALYSIS_MODEL=grok-4-1-fast-reasoning`
- `SENTRYSLOTH_LLM_SCHEDULER_WORKERS=4`
- `SENTRYSLOTH_LLM_MAX_REQUESTS_PER_MINUTE=120`
- `SENTRYSLOTH_LLM_MAX_TOKENS_PER_MINUTE=1000000`
API keys:
- `SENTRYSLOTH_GROK_API_KEY` when provider is `grok`
- `SENTRYSLOTH_GEMINI_API_KEY` when provider is `gemini`
See `config.py` `Settings` class and `.env.example`.

## Code Conventions

- Python 3.11+, async/await for all I/O
- Pydantic v2 for data models and settings
- Ruff lint+format (line-length=100)
- pytest + pytest-asyncio (asyncio_mode=auto), respx for HTTP mocking
- All imports at module top — no local imports
- Explicit timeouts on all external calls
- Errors wrapped with context at system boundaries, never swallowed
