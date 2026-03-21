"""Multiple-choice option matching."""

from __future__ import annotations

import re

from theaios.trustgate.canonicalize import Canonicalizer, register_canonicalizer

# Priority-ordered extraction patterns
_ANSWER_IS_RE = re.compile(
    r"(?:(?:the|my|correct|best)\s+)?(?:answer|option|choice)\s+is\s*[:\s]*\(?([A-Ea-e])\)?",
    re.IGNORECASE,
)
_STARTS_WITH_RE = re.compile(r"^\s*\(?([A-Ea-e])\)?[\.\):\s]")
_ENDS_WITH_RE = re.compile(r"(?:^|\s)\(?([A-Ea-e])\)?\s*\.?\s*$")
_STANDALONE_RE = re.compile(r"(?:^|\s)\(?([A-Ea-e])\)?(?:\s|$)")


@register_canonicalizer("mcq")
class MCQCanonicalizer(Canonicalizer):
    """Extract the chosen option letter from an MCQ response."""

    def canonicalize(self, question: str, answer: str) -> str:
        text = self.preprocess(answer)
        if not text:
            return ""

        # 1. Explicit "the answer is (B)" pattern
        m = _ANSWER_IS_RE.search(text)
        if m:
            return m.group(1).upper()

        # 2. Starts with a letter: "B) Paris"
        m = _STARTS_WITH_RE.search(text)
        if m:
            return m.group(1).upper()

        # 3. Ends with a letter: "... (C)"
        m = _ENDS_WITH_RE.search(text)
        if m:
            return m.group(1).upper()

        # 4. Standalone letter anywhere (only if exactly one unique match)
        matches = _STANDALONE_RE.findall(text)
        unique = {ch.upper() for ch in matches}
        if len(unique) == 1:
            return str(unique.pop())

        return ""

    def validate(self, canonical: str) -> bool:
        return canonical == "" or (len(canonical) == 1 and canonical in "ABCDE")
