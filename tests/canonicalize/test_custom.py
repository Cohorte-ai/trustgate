"""Tests for the custom plugin canonicalizer loader."""

from __future__ import annotations

import pytest

from theaios.trustgate.canonicalize.custom import load_custom_canonicalizer


class TestLoadCustomCanonicalizer:
    def test_loads_numeric_by_path(self) -> None:
        # Load the built-in NumericCanonicalizer via dotted path
        canon = load_custom_canonicalizer(
            "theaios.trustgate.canonicalize.numeric.NumericCanonicalizer"
        )
        assert canon.canonicalize("", "The answer is 42") == "42"

    def test_loads_mcq_by_path(self) -> None:
        canon = load_custom_canonicalizer(
            "theaios.trustgate.canonicalize.mcq.MCQCanonicalizer"
        )
        assert canon.canonicalize("", "The answer is B") == "B"

    def test_raises_on_invalid_path(self) -> None:
        with pytest.raises(ImportError, match="Invalid dotted path"):
            load_custom_canonicalizer("NoDotsHere")

    def test_raises_on_missing_module(self) -> None:
        with pytest.raises(ImportError):
            load_custom_canonicalizer("nonexistent.module.SomeClass")

    def test_raises_on_missing_class(self) -> None:
        with pytest.raises(AttributeError):
            load_custom_canonicalizer(
                "theaios.trustgate.canonicalize.numeric.NonexistentClass"
            )

    def test_raises_on_non_canonicalizer(self) -> None:
        # DiskCache is not a Canonicalizer
        with pytest.raises(TypeError, match="not a Canonicalizer subclass"):
            load_custom_canonicalizer("theaios.trustgate.cache.DiskCache")
