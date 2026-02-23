# SentrySloth

Security-focused code review assistant for open-source releases.

## What is it?

SentrySloth is an LLM-powered tool that analyzes diffs between software releases to find security-relevant changes. It uses a two-stage pipeline — fast triage followed by deep analysis — and outputs results in SARIF (for GitHub Code Scanning), Markdown, or JSON formats.

## Features

- **Two-stage pipeline**: fast triage filters noise, deep analysis examines only security-relevant chunks
- **Multiple output formats**: JSON (default), Markdown reports, SARIF for GitHub Code Scanning integration
- **SQLite caching**: scan history and accumulated repository profile across releases
- **Accumulating RepoProfile memory**: bootstrap context for a new repository, then incrementally update it after each scan
- **Baseline suppression**: mark known findings to exclude from future reports
- **Configurable severity/confidence thresholds**: fail CI builds on findings above a chosen severity
- **Typed Python package** with PEP 561 support

## Quick Start

```bash
pip install -e ".[dev]"
export SENTRYSLOTH_GROK_API_KEY=your-api-key
sentrysloth scan https://github.com/org/repo --from v1.0 --to v1.1
```

## Installation

### From PyPI

Published after tagged releases. If no release is available yet, use the source install below.

```bash
pip install sentrysloth
```

### From source (development)

```bash
git clone https://github.com/sergeykochanov/SentrySloth.git
cd SentrySloth
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

All settings are controlled via environment variables with the `SENTRYSLOTH_` prefix.

| Variable | Description | Default |
|---|---|---|
| `SENTRYSLOTH_LLM_PROVIDER` | LLM provider (`grok` or `gemini`) | `grok` |
| `SENTRYSLOTH_GROK_API_KEY` | **Required when provider=`grok`.** xAI API key | — |
| `SENTRYSLOTH_GEMINI_API_KEY` | **Required when provider=`gemini`.** Gemini API key | — |
| `SENTRYSLOTH_LLM_TRIAGE_MODEL` | Model for triage stage | `grok-4-1-fast-non-reasoning` |
| `SENTRYSLOTH_LLM_ANALYSIS_MODEL` | Model for deep analysis | `grok-4-1-fast-reasoning` |
| `SENTRYSLOTH_LLM_SCHEDULER_WORKERS` | Scheduler worker count | `4` |
| `SENTRYSLOTH_LLM_QUEUE_MAX_SIZE` | Max pending LLM requests in local queue | `1000` |
| `SENTRYSLOTH_LLM_MAX_REQUESTS_PER_MINUTE` | Local requests-per-minute limiter | `120` |
| `SENTRYSLOTH_LLM_MAX_TOKENS_PER_MINUTE` | Local token-per-minute limiter (`0` disables) | `1000000` |
| `SENTRYSLOTH_LLM_TRIAGE_MAX_INPUT_TOKENS` | Prompt budget for triage stage | `16000` |
| `SENTRYSLOTH_LLM_ANALYSIS_MAX_INPUT_TOKENS` | Prompt budget for analysis stage | `64000` |
| `SENTRYSLOTH_LLM_MAX_RETRIES` | Maximum retries for transient provider failures | `5` |
| `SENTRYSLOTH_LLM_QUOTA_EXHAUSTED_MODE` | Quota behavior: `fail_fast`, `heuristic_fallback`, `legacy_fail_open` | `fail_fast` |
| `SENTRYSLOTH_LLM_QUOTA_FALLBACK_SECURITY_THRESHOLD` | Security score threshold for heuristic fallback mode | `0.8` |
| `SENTRYSLOTH_LLM_CONNECT_TIMEOUT` | Connection timeout (seconds) | `10.0` |
| `SENTRYSLOTH_LLM_READ_TIMEOUT` | Read timeout (seconds) | `120.0` |
| `SENTRYSLOTH_LLM_TOTAL_TIMEOUT` | Total request timeout (seconds) | `300.0` |
| `SENTRYSLOTH_CACHE_ENABLED` | Enable SQLite cache | `true` |
| `SENTRYSLOTH_CACHE_DB_PATH` | Cache database path | `~/.cache/sentrysloth/cache.db` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_ENABLED` | Enable accumulated RepoProfile context | `true` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_HISTORY_ENABLED` | Store per-scan RepoProfile snapshots | `false` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_MAX_CHARS` | Max chars injected from RepoProfile into prompts | `6000` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_MAX_ITEMS` | Max items per list field in RepoProfile | `24` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_BOOTSTRAP_MAX_FILES` | Max metadata files read during initial bootstrap | `20` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_BOOTSTRAP_MAX_TREE_PATHS` | Max file-tree paths included in bootstrap payload | `500` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_BOOTSTRAP_MAX_FILE_CHARS` | Max chars read per bootstrap file | `4000` |
| `SENTRYSLOTH_CLONE_BASE_DIR` | Local directory for git clones | `~/.cache/sentrysloth/repos` |

See [`.env.example`](.env.example) for a template.

## Usage

### Scan a repository

```bash
# Compare two tags
sentrysloth scan https://github.com/org/repo --from v1.0 --to v1.1

# Output as Markdown
sentrysloth scan https://github.com/org/repo --from v1.0 --to v1.1 -o markdown

# Save SARIF report to file
sentrysloth scan https://github.com/org/repo --from v1.0 --to v1.1 -o sarif -f report.sarif

# Fail if HIGH or CRITICAL findings exist (useful in CI)
sentrysloth scan https://github.com/org/repo --from v1.0 --to v1.1 --fail-on high

# Apply baseline suppression
sentrysloth scan https://github.com/org/repo --from v1.0 --to v1.1 --baseline baseline.json
```

### List available versions

```bash
sentrysloth list-versions https://github.com/org/repo
sentrysloth list-versions https://github.com/org/repo -n 50
```

### Batch-scan multiple repos

```bash
# Last 3 release transitions per repo:
# latest major first, then backfill from older majors
sentrysloth batch-scan repos.txt --last-releases 3

# Process 4 repositories in parallel
# (pairs within each repo are scanned sequentially old->new)
sentrysloth batch-scan repos.txt --last-releases 3 -j 4
```

### View a cached scan result

```bash
sentrysloth report <scan-id>
sentrysloth report <scan-id> -o markdown
```

### Cache statistics

```bash
sentrysloth cache-info
sentrysloth cache-info https://github.com/org/repo
```

### View cached RepoProfile knowledge

```bash
sentrysloth repo-profile https://github.com/org/repo
```

## Output Formats

- **JSON** (default): structured scan result with all findings and metadata
- **Markdown**: human-readable report with severity-grouped findings
- **SARIF**: standard format for static analysis results, compatible with GitHub Code Scanning

## Architecture

```
Git Source
  |
  +--> RepoProfile Bootstrap (new repo only, metadata + file tree)
  |        |
  |        v
  |     SQLite Cache (repo_profiles)
  |
  v
Diff Extractor --> Chunks
  |
  v
Triage (fast model) --> Filter security-relevant chunks
  |
  v
Deep / Agentic Analysis (analysis model) + RepoProfile context --> Findings
  |
  v
RepoProfile Incremental Update (triage model) --> SQLite Cache
  |
  v
Reports (JSON / Markdown / SARIF)
```

## Docker

```bash
docker build -t sentrysloth .
docker run -e SENTRYSLOTH_GROK_API_KEY=your-key sentrysloth scan https://github.com/org/repo --from v1.0 --to v1.1
```

## Troubleshooting

### Quota exhausted / rate limited

- If you see a quota error, reduce scan scope (smaller tag diffs), reduce parallelism (`SENTRYSLOTH_LLM_SCHEDULER_WORKERS`, batch `--concurrency`), or switch to `SENTRYSLOTH_LLM_QUOTA_EXHAUSTED_MODE=heuristic_fallback` (triage will fall back to heuristic scoring when LLM quota is exhausted).
- If you hit transient 429 rate limits, lower `SENTRYSLOTH_LLM_MAX_REQUESTS_PER_MINUTE`.

### Where data is stored

- **Repo clones**: `SENTRYSLOTH_CLONE_BASE_DIR` (default `~/.cache/sentrysloth/repos`)
- **Cache DB**: `SENTRYSLOTH_CACHE_DB_PATH` (default `~/.cache/sentrysloth/cache.db`)
- On cache schema version mismatch, SentrySloth recreates the cache DB automatically.

## License

[MIT](LICENSE)
