"""Shared test fixtures."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sentrysloth.config import get_settings

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so CLI output assertions are reliable."""
    return _ANSI_RE.sub("", text)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def auth_diff() -> str:
    return (FIXTURES_DIR / "simple_auth.diff").read_text()


@pytest.fixture
def docs_diff() -> str:
    return (FIXTURES_DIR / "docs_only.diff").read_text()


@pytest.fixture
def mixed_diff() -> str:
    return (FIXTURES_DIR / "mixed_changes.diff").read_text()


@pytest.fixture
def default_settings():
    return get_settings()
