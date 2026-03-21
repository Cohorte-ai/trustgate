"""Tests for canonicalize base class, registry, and shared preprocessing."""

from __future__ import annotations

import pytest

from theaios.trustgate.canonicalize import (
    Canonicalizer,
    _REGISTRY,
    get_canonicalizer,
    list_canonicalizers,
    register_canonicalizer,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class _DummyCanon(Canonicalizer):
    """Concrete canonicalizer for testing preprocessing."""

    def canonicalize(self, question: str, answer: str) -> str:
        return self.preprocess(answer)


class TestPreprocess:
    def setup_method(self) -> None:
        self.c = _DummyCanon()

    def test_strips_whitespace(self) -> None:
        assert self.c.preprocess("  hello  ") == "hello"

    def test_removes_code_fences(self) -> None:
        text = "```python\nprint('hi')\n```"
        assert self.c.preprocess(text) == "print('hi')"

    def test_removes_code_fences_no_lang(self) -> None:
        text = "```\ncode here\n```"
        assert self.c.preprocess(text) == "code here"

    def test_removes_preamble_sure(self) -> None:
        assert self.c.preprocess("Sure, the answer is 42.") == "the answer is 42."

    def test_removes_preamble_certainly(self) -> None:
        assert self.c.preprocess("Certainly! The answer is 42.") == "The answer is 42."

    def test_removes_preamble_i_think(self) -> None:
        assert self.c.preprocess("I think the answer is B.") == "the answer is B."

    def test_removes_preamble_here_is(self) -> None:
        assert self.c.preprocess("Here is the answer: 42") == "42"

    def test_handles_empty_string(self) -> None:
        assert self.c.preprocess("") == ""

    def test_handles_none_like_empty(self) -> None:
        # preprocess expects str; empty string is the safe path
        assert self.c.preprocess("") == ""

    def test_unicode_normalization(self) -> None:
        # e + combining acute vs precomposed e-acute
        nfc = "\u00e9"  # precomposed
        nfd = "e\u0301"  # decomposed
        assert self.c.preprocess(nfd) == nfc


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builtin_canonicalizers_registered(self) -> None:
        # Import modules to trigger registration
        import theaios.trustgate.canonicalize.numeric  # noqa: F401
        import theaios.trustgate.canonicalize.mcq  # noqa: F401
        import theaios.trustgate.canonicalize.code_exec  # noqa: F401
        import theaios.trustgate.canonicalize.llm_judge  # noqa: F401

        names = list_canonicalizers()
        assert "numeric" in names
        assert "mcq" in names
        assert "code_exec" in names
        assert "llm_judge" in names

    def test_get_canonicalizer_numeric(self) -> None:
        import theaios.trustgate.canonicalize.numeric  # noqa: F401

        c = get_canonicalizer("numeric")
        assert isinstance(c, Canonicalizer)

    def test_get_canonicalizer_unknown(self) -> None:
        with pytest.raises(KeyError, match="Unknown canonicalizer"):
            get_canonicalizer("nonexistent_canon")

    def test_register_custom(self) -> None:
        @register_canonicalizer("test_dummy")
        class TestDummyCanon(Canonicalizer):
            def canonicalize(self, question: str, answer: str) -> str:
                return "dummy"

        c = get_canonicalizer("test_dummy")
        assert c.canonicalize("q", "a") == "dummy"

        # Clean up
        del _REGISTRY["test_dummy"]

    def test_list_returns_sorted(self) -> None:
        names = list_canonicalizers()
        assert names == sorted(names)
