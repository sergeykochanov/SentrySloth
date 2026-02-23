"""Tests for context_builder: function signature extraction, name extraction."""

from __future__ import annotations

import pytest

from sentrysloth.analyzers.context_builder import (
    extract_function_name,
    extract_function_signatures,
)


class TestExtractFunctionSignatures:
    def test_python_def(self):
        code = "def hello(name: str) -> str:\n    return name\n"
        sigs = extract_function_signatures(code)
        assert len(sigs) >= 1
        assert any("hello" in sig for _, sig in sigs)

    def test_python_async_def(self):
        code = "async def fetch_data(url: str) -> dict:\n    pass\n"
        sigs = extract_function_signatures(code)
        assert len(sigs) >= 1
        assert any("fetch_data" in sig for _, sig in sigs)

    def test_python_class(self):
        code = "class MyService(BaseService):\n    pass\n"
        sigs = extract_function_signatures(code)
        assert len(sigs) >= 1
        assert any("MyService" in sig for _, sig in sigs)

    def test_go_func(self):
        code = "func (s *Server) HandleRequest(ctx context.Context) error {\n}\n"
        sigs = extract_function_signatures(code)
        assert len(sigs) >= 1
        assert any("HandleRequest" in sig for _, sig in sigs)

    def test_js_function(self):
        code = "export async function loadUser(id) {\n}\n"
        sigs = extract_function_signatures(code)
        assert len(sigs) >= 1
        assert any("loadUser" in sig for _, sig in sigs)

    def test_rust_fn(self):
        code = "pub async fn process(data: Vec<u8>) {\n}\n"
        sigs = extract_function_signatures(code)
        assert len(sigs) >= 1
        assert any("process" in sig for _, sig in sigs)

    def test_empty_content(self):
        assert extract_function_signatures("") == []

    def test_no_functions(self):
        code = "# just a comment\nx = 42\nprint(x)\n"
        assert extract_function_signatures(code) == []

    def test_line_numbers_monotonically_increase(self):
        code = "x = 1\n\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
        sigs = extract_function_signatures(code)
        # Signatures are sorted by line number
        line_numbers = [ln for ln, _ in sigs]
        assert line_numbers == sorted(line_numbers)
        # Both foo and bar are found
        names = {sig for _, sig in sigs}
        assert any("foo" in s for s in names)
        assert any("bar" in s for s in names)

    def test_deduplication(self):
        # Same signature appearing conceptually once should not be duplicated
        code = "def foo():\n    pass\n"
        sigs = extract_function_signatures(code)
        sig_texts = [s for _, s in sigs]
        assert len(sig_texts) == len(set(sig_texts))


class TestExtractFunctionName:
    @pytest.mark.parametrize(
        "sig, expected",
        [
            ("def hello()", "hello"),
            ("async def fetch_data(url)", "fetch_data"),
            ("function loadUser(id)", "loadUser"),
            ("func HandleRequest(ctx)", "HandleRequest"),
            ("pub fn process(data)", "process"),
            ("class MyService(Base)", "MyService"),
        ],
    )
    def test_extracts_name(self, sig: str, expected: str):
        assert extract_function_name(sig) == expected

    def test_returns_none_for_garbage(self):
        assert extract_function_name("x = 42") is None
        assert extract_function_name("") is None
