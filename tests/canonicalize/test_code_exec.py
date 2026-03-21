"""Tests for the code execution canonicalizer."""

from __future__ import annotations

import pytest

from theaios.trustgate.canonicalize.code_exec import CodeExecCanonicalizer, _extract_code


@pytest.fixture()
def canon() -> CodeExecCanonicalizer:
    return CodeExecCanonicalizer(timeout=5.0)


class TestCodeExtraction:
    def test_extracts_python_block(self) -> None:
        text = "Here's the code:\n```python\nprint('hello')\n```"
        assert _extract_code(text) == "print('hello')"

    def test_extracts_plain_block(self) -> None:
        text = "```\nprint('hello')\n```"
        assert _extract_code(text) == "print('hello')"

    def test_raw_code(self) -> None:
        text = "print('hello')"
        assert _extract_code(text) == "print('hello')"


class TestCodeExecCanonicalizer:
    def test_passing_code(self, canon: CodeExecCanonicalizer) -> None:
        code = "```python\nprint('hello')\n```"
        assert canon.canonicalize("", code) == "pass"

    def test_failing_code(self, canon: CodeExecCanonicalizer) -> None:
        code = "```python\nassert 1 == 2\n```"
        assert canon.canonicalize("", code) == "fail"

    def test_syntax_error(self, canon: CodeExecCanonicalizer) -> None:
        code = "```python\ndef foo(\n```"
        assert canon.canonicalize("", code) == "fail"

    def test_infinite_loop_timeout(self) -> None:
        canon = CodeExecCanonicalizer(timeout=1.0)
        code = "```python\nwhile True: pass\n```"
        assert canon.canonicalize("", code) == "fail"

    def test_runtime_error(self, canon: CodeExecCanonicalizer) -> None:
        code = "```python\nraise ValueError('bad')\n```"
        assert canon.canonicalize("", code) == "fail"

    def test_empty_answer(self, canon: CodeExecCanonicalizer) -> None:
        assert canon.canonicalize("", "") == "fail"

    def test_exit_zero(self, canon: CodeExecCanonicalizer) -> None:
        code = "```python\nimport sys\nsys.exit(0)\n```"
        assert canon.canonicalize("", code) == "pass"

    def test_exit_nonzero(self, canon: CodeExecCanonicalizer) -> None:
        code = "```python\nimport sys\nsys.exit(1)\n```"
        assert canon.canonicalize("", code) == "fail"
