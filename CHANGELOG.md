# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Release workflow for building artifacts, publishing to PyPI, and creating GitHub releases.
- Coverage threshold enforcement in CI.

### Changed
- Deep and agentic analyzers now share common prompt/finding conversion logic.
- Error handling around git/tool/cache operations is stricter and better logged.
