"""Numeric extraction and normalization."""

from __future__ import annotations

import re

from theaios.trustgate.canonicalize import Canonicalizer, register_canonicalizer

# Priority-ordered extraction patterns
_BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)[\.\s]*$",
    re.IGNORECASE | re.MULTILINE,
)
_GSM8K_RE = re.compile(r"####\s*(.+?)$", re.MULTILINE)
_FRACTION_RE = re.compile(r"^(-?\d+)\s*/\s*(\d+)$")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _clean_number(raw: str) -> str:
    """Strip currency symbols, commas, whitespace; normalize to a clean numeric string."""
    s = raw.strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    # Handle percentages
    if s.endswith("%"):
        s = s[:-1]
        try:
            return _normalize_float(float(s) / 100)
        except ValueError:
            return ""
    # Handle fractions like 3/4
    m = _FRACTION_RE.match(s)
    if m:
        numer, denom = int(m.group(1)), int(m.group(2))
        if denom == 0:
            return ""
        return _normalize_float(numer / denom)
    # Plain number
    try:
        return _normalize_float(float(s))
    except ValueError:
        return ""


def _normalize_float(value: float) -> str:
    """Convert a float to a canonical string (no trailing .0)."""
    if value == int(value):
        return str(int(value))
    return str(value)


@register_canonicalizer("numeric")
class NumericCanonicalizer(Canonicalizer):
    """Extract and normalize the final numeric answer from an LLM response."""

    def canonicalize(self, question: str, answer: str) -> str:
        text = self.preprocess(answer)
        if not text:
            return ""

        # 1. Boxed answer: \boxed{42}
        m = _BOXED_RE.search(text)
        if m:
            result = _clean_number(m.group(1))
            if result:
                return result

        # 2. "The answer is X"
        m = _ANSWER_IS_RE.search(text)
        if m:
            result = _clean_number(m.group(1))
            if result:
                return result

        # 3. GSM8K delimiter: #### 42
        m = _GSM8K_RE.search(text)
        if m:
            result = _clean_number(m.group(1))
            if result:
                return result

        # 4. Fallback: last number in the text
        numbers = _NUMBER_RE.findall(text)
        if numbers:
            result = _clean_number(numbers[-1])
            if result:
                return result

        return ""

    def validate(self, canonical: str) -> bool:
        if not canonical:
            return True  # empty is valid (means "no number found")
        try:
            float(canonical)
            return True
        except ValueError:
            return False
