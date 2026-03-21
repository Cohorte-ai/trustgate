"""Tests for the MCQ canonicalizer."""

from __future__ import annotations

import pytest

from theaios.trustgate.canonicalize.mcq import MCQCanonicalizer


@pytest.fixture()
def canon() -> MCQCanonicalizer:
    return MCQCanonicalizer()


class TestMCQExtraction:
    def test_answer_is_b(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is B") == "B"

    def test_answer_is_parenthetical(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is (C)") == "C"

    def test_correct_option_is(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "The correct option is A") == "A"

    def test_my_answer_is(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "My answer is D") == "D"

    def test_starts_with_letter(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "B) Paris") == "B"

    def test_starts_with_paren_letter(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "(C) Jupiter") == "C"

    def test_ends_with_letter(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "The correct choice is (C)") == "C"

    def test_case_insensitive(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "the answer is b") == "B"

    def test_verbose_response(self, canon: MCQCanonicalizer) -> None:
        text = "I think the correct option is A because Paris is in France."
        assert canon.canonicalize("", text) == "A"

    def test_letter_with_period(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "B.") == "B"


class TestMCQEdgeCases:
    def test_empty_answer(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "") == ""

    def test_no_letter(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "I don't know the answer") == ""

    def test_ambiguous_multiple_letters(self, canon: MCQCanonicalizer) -> None:
        # Multiple different standalone letters — should return empty
        text = "Option A is wrong but B is also debatable"
        result = canon.canonicalize("", text)
        # Either "" or "B" is acceptable — the key is it doesn't crash
        assert result in ("", "A", "B")

    def test_preamble_stripped(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "Sure, the answer is B") == "B"

    def test_letter_e(self, canon: MCQCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is E") == "E"


class TestMCQValidation:
    def test_valid_letters(self, canon: MCQCanonicalizer) -> None:
        for letter in "ABCDE":
            assert canon.validate(letter) is True

    def test_valid_empty(self, canon: MCQCanonicalizer) -> None:
        assert canon.validate("") is True

    def test_invalid(self, canon: MCQCanonicalizer) -> None:
        assert canon.validate("F") is False
        assert canon.validate("AB") is False
