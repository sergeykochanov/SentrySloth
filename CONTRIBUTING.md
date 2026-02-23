# Contributing

Thanks for contributing to SentrySloth.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality Checks

Run these before opening a PR:

```bash
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format src/ tests/
.venv/bin/python -m pytest tests/
```

## Pull Requests

- Keep changes focused and include tests for new behavior.
- Update docs (`README.md`, `.env.example`, changelog) when behavior or configuration changes.
- Prefer small, reviewable commits with clear messages.
