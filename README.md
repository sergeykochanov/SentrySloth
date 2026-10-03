# SentrySloth

Security-focused code review assistant for open-source releases.

## What is it?

SentrySloth is an LLM-powered tool that analyzes diffs between software releases to find security-relevant changes. It uses a three-stage pipeline — fast triage, deep analysis, then independent candidate verification — and outputs results in SARIF (for GitHub Code Scanning), Markdown, or JSON formats.

## Features

- **Evidence-based review**: triage, discovery and independent verification with exact source citations
- **Multiple output formats**: JSON (default), Markdown reports, SARIF for GitHub Code Scanning integration
- **SQLite caching**: scan history and revision-scoped repository profiles
- **Regression and patch review**: investigate newly introduced bugs and incomplete fixes; compare actual Git ancestry
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
| `SENTRYSLOTH_LLM_TRIAGE_MODEL` | Model for triage stage | `grok-4.3` |
| `SENTRYSLOTH_LLM_ANALYSIS_MODEL` | Model for deep analysis | `grok-4.7` |
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
| `SENTRYSLOTH_CACHE_REPO_PROFILE_ENABLED` | Enable revision-scoped RepoProfile context | `true` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_HISTORY_ENABLED` | Store per-scan RepoProfile snapshots | `false` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_MAX_CHARS` | Max chars injected from RepoProfile into prompts | `6000` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_MAX_ITEMS` | Max items per list field in RepoProfile | `24` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_BOOTSTRAP_MAX_FILES` | Max metadata files read during initial bootstrap | `20` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_BOOTSTRAP_MAX_TREE_PATHS` | Max file-tree paths included in bootstrap payload | `500` |
| `SENTRYSLOTH_CACHE_REPO_PROFILE_BOOTSTRAP_MAX_FILE_CHARS` | Max chars read per bootstrap file | `4000` |
| `SENTRYSLOTH_CLONE_BASE_DIR` | Local directory for git clones | `~/.cache/sentrysloth/repos` |

See [`.env.example`](.env.example) for a template.

The default xAI setup uses **Grok 4.3 / none** for triage and **Grok 4.7 / high**
for deep analysis and independent verification. The flagship is reserved for the stages
that need source investigation and counterevidence; triage keeps the initial pass cheaper.
These defaults are based on the [Grok 4.7 capabilities](https://docs.x.ai/developers/models/grok-4.7)
and [Grok 4.3 capabilities](https://docs.x.ai/developers/models/grok-4.3), not a claimed benchmark gain.

Set `SENTRYSLOTH_LLM_TRIAGE_REASONING_EFFORT` or
`SENTRYSLOTH_LLM_ANALYSIS_REASONING_EFFORT` to override reasoning depth.
Supported 4.7 levels are `low`, `medium`, `high`, `xhigh`; reasoning cannot be disabled.
These options are sent only for supported Grok model families, not to Gemini.

Old `grok-4-1-fast-*` names are compatibility aliases redirected by xAI to Grok 4.3;
see the [retirement notice](https://docs.x.ai/developers/migration/may-15-retirement).
Use current [provider pricing](https://docs.x.ai/developers/pricing) when budgeting.

OpenAI-compatible providers stream responses by default to avoid waiting for a complete
long-running response before receiving data. Text and tool calls are assembled before
analysis continues; final usage includes reasoning tokens. An unfinished or truncated
stream is an error. Set `SENTRYSLOTH_LLM_STREAM_RESPONSES=false` for an endpoint that
does not support streaming or `stream_options.include_usage`.

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

## Review modes and evidence

```bash
sentrysloth scan ./project --from v1.0 --to v1.1 --mode both
sentrysloth scan ./project --from v1.0 --to v1.1 --mode patch
sentrysloth scan ./project --from v1.0 --to v1.1 --no-verify
```

`both` is the default. Patch review checks the intended security invariant against
sibling code paths, drivers, normalization, streaming and concurrency. Tools can
read changed tests and dependency manifests even when those files are excluded
from initial diff analysis.

The verifier uses a fresh conversation with the analysis model and seeks counterevidence.
`confirmed_static` means source inspection with matching citations, not a reproduced
exploit or maintainer acceptance. Missing reachability, dependency behavior or citations
keeps a hypothesis in `candidates` as `needs_review`. Refuted candidates are retained as
`rejected`. Only confirmed findings enter SARIF and `--fail-on` decisions.
`--no-verify` skips verification and produces candidates only.

Every scan pins the two refs to SHAs. Batch mode defaults to `--pairing-mode ancestry`:
each semantic-version release is compared with an older ancestor, preferring its own
major/minor stream. Legacy `chronological` and `per-major` modes remain available.
Non-version tags require an explicit comparison or a legacy pairing mode.
Divergent refs can be inspected directly, but cannot confirm a `security_regression`.

Exit codes: **0** completed, **1** confirmed finding meets `--fail-on`, **2** setup error,
**3** incomplete analysis. Inspect `complete`, `coverage_issues`, `release.excluded_paths`,
triage statistics and `candidates`; an empty findings list does not establish security.
Completion describes the selected scope, not exhaustive coverage of the repository.
Verification adds model calls. Set `SENTRYSLOTH_VERIFICATION_MAX_TURNS` to bound its turns.
Token counts with `token_usage_complete=false` are a lower bound after failed requests.

The heuristic prefilter is disabled by default (`SENTRYSLOTH_PREFILTER_MIN_SECURITY_SCORE=0`).
Raising it saves model calls but can miss changes; dropped counts are recorded.

### Migrating from 0.1

Finding IDs now include root cause and source anchors. Regenerate baselines from reviewed
0.2 reports; old hunk-based IDs could hide multiple unrelated findings and are not matched
automatically. Old cached reports remain readable and are not retroactively verified.
Source strings are preserved inside untrusted data instead of being destructively sanitized.

### Evaluate saved scans

See [evaluation/README.md](evaluation/README.md) for the synthetic suite and procedure.

```bash
python scripts/prepare_evaluation.py /tmp/sentrysloth-eval
sentrysloth evaluate /tmp/sentrysloth-eval/dataset.json ./reports --split holdout --top 10
```

No live-model precision or recall gain is claimed by this release. Compare old and new
runs on identical SHAs and a separate holdout set before tuning prompts or thresholds.

## Output Formats

- **JSON** (default): structured scan result with all findings and metadata
- **Markdown**: human-readable report with severity-grouped findings
- **SARIF**: standard format for static analysis results, compatible with GitHub Code Scanning

## Architecture

```
Git Source
  |
  +--> RepoProfile Bootstrap (exact revision, metadata + file tree)
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
Deep / Agentic Analysis + revision context --> Candidates
  |
  v
Independent Verification + exact citations --> Confirmed / Needs review / Rejected
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
