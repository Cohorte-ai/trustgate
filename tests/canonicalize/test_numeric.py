"""Tests for the numeric canonicalizer."""

from __future__ import annotations

import pytest

from theaios.trustgate.canonicalize.numeric import NumericCanonicalizer


@pytest.fixture()
def canon() -> NumericCanonicalizer:
    return NumericCanonicalizer()


class TestNumericExtraction:
    def test_boxed_answer(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The result is \\boxed{42}") == "42"

    def test_boxed_decimal(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "\\boxed{3.14}") == "3.14"

    def test_answer_is_pattern(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is 3.14.") == "3.14"

    def test_answer_is_integer(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is 42") == "42"

    def test_final_answer_is(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The final answer is 7") == "7"

    def test_gsm8k_delimiter(self, canon: NumericCanonicalizer) -> None:
        text = "First I add 2+2=4. Then I multiply.\n#### 42"
        assert canon.canonicalize("", text) == "42"

    def test_last_number_fallback(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "I calculated and got 7") == "7"

    def test_multiple_numbers_takes_last(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "Step 1: 3+4=7. Step 2: 7*6=42") == "42"


class TestNumericNormalization:
    def test_commas(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is 1,234,567") == "1234567"

    def test_dollar_sign(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is $42.50") == "42.5"

    def test_trailing_zero(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is 42.0") == "42"

    def test_fraction(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is 3/4") == "0.75"

    def test_negative_number(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is -5") == "-5"

    def test_negative_decimal(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is -3.14") == "-3.14"

    def test_percentage(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is 42%") == "0.42"

    def test_zero(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "The answer is 0") == "0"


class TestNumericEdgeCases:
    def test_empty_answer(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "") == ""

    def test_no_numbers(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "I don't know the answer") == ""

    def test_only_text(self, canon: NumericCanonicalizer) -> None:
        assert canon.canonicalize("", "Paris is the capital of France") == ""

    def test_code_fence_with_number(self, canon: NumericCanonicalizer) -> None:
        text = "```python\nresult = 42\n```\nThe answer is 42"
        assert canon.canonicalize("", text) == "42"


class TestNumericValidation:
    def test_valid_number(self, canon: NumericCanonicalizer) -> None:
        assert canon.validate("42") is True

    def test_valid_decimal(self, canon: NumericCanonicalizer) -> None:
        assert canon.validate("3.14") is True

    def test_valid_empty(self, canon: NumericCanonicalizer) -> None:
        assert canon.validate("") is True

    def test_invalid(self, canon: NumericCanonicalizer) -> None:
        assert canon.validate("abc") is False
