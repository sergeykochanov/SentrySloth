# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.2.1] - 2026-10-03

### Fixed
- OpenAI-compatible providers stream responses by default, avoiding the observed disconnects while waiting for long Grok responses.
- Streaming preserves fragmented text, parallel tool calls, and final usage with reasoning tokens.
- Unfinished streams and truncated tool-call responses fail explicitly instead of being accepted as complete.

### Added
- `SENTRYSLOTH_LLM_STREAM_RESPONSES=false` compatibility option for endpoints without streaming support.

## [0.2.0] - 2026-10-03

### Added
- Independent candidate verification with exact source citations pinned to commit SHAs.
- Candidate states: `needs_review`, `confirmed_static`, `rejected`; runtime reproduction is never inferred.
- `--mode regression|patch|both`, defaulting to both, and `--verify/--no-verify`.
- Source tools for numbered ranges, Python symbols, paginated search, changed files and per-file diffs.
- Ancestry-aware batch release pairing, now the default; divergent comparisons are explicitly labeled.
- Offline `evaluate` command with exact-revision matching, precision, case recall, coverage and token metrics.
- Eight synthetic evaluation cases and a deterministic Git-fixture preparation script.

### Fixed
- Statement reordering, indentation changes, and commented-out guards no longer count as noise.
- Large diff hunks are split on whole lines; their tail is not discarded.
- Analysis failures no longer appear as successful empty scans. JSON and SARIF expose incomplete coverage.
- `--since` preserves the predecessor needed to review the first qualifying release.
- Independent findings in the same hunk have distinct evidence-based IDs.
- xAI token accounting includes reasoning tokens reported separately from visible output.
- Repository profiles are reused only at the same revision and discard accumulated risk hypotheses.
- Source containing Markdown fences or role-like strings is preserved for analysis.

### Changed
- Default xAI models: Grok 4.3 without reasoning for triage, Grok 4.7 with high reasoning for analysis and verification. Reasoning effort is configurable.
- Heuristic prefilter defaults to zero to avoid silently dropping low-scoring changes.
- Low-severity candidates are retained; confirmation requires evidence, not a severity threshold.
- Final findings and the candidate review queue are separate. `--no-verify` produces candidates only.
- Baseline IDs changed; regenerate reviewed baselines from 0.2 reports. Old hunk IDs are not automatically matched.
- These changes have offline regression coverage; real-world precision and recall improvements are not yet measured.

## [0.1.0]


### Added
- Release workflow for building artifacts, publishing to PyPI, and creating GitHub releases.
- Coverage threshold enforcement in CI.

### Changed
- Deep and agentic analyzers now share common prompt/finding conversion logic.
- Error handling around git/tool/cache operations is stricter and better logged.
