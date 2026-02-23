"""GSM8K dataset loader."""

from __future__ import annotations

import json
import re

from trustgate.datasets import _download_jsonl, subsample
from trustgate.types import Question

_GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
)

_ANSWER_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)


def _extract_answer(solution: str) -> str:
    """Extract the numeric answer from a GSM8K solution (after ####)."""
    m = _ANSWER_RE.search(solution)
    if m:
        return m.group(1).strip().replace(",", "")
    return ""


def load_gsm8k(
    n: int | None = None,
    seed: int = 42,
) -> list[Question]:
    """Load GSM8K math problems.

    Downloads from GitHub (cached after first download).
    Each question has text + acceptable_answers (the numeric answer).
    Recommended canonicalization: ``"numeric"``

    Args:
        n: Subsample to *n* questions (for quick testing). None = all.
        seed: Random seed for subsampling.
    """
    path = _download_jsonl(_GSM8K_URL, "gsm8k_test.jsonl")

    questions: list[Question] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        data = json.loads(line)
        answer = _extract_answer(data.get("answer", ""))
        questions.append(
            Question(
                id=f"gsm8k_{i}",
                text=data["question"],
                acceptable_answers=[answer] if answer else None,
                metadata={"source": "gsm8k"},
            )
        )

    if n is not None:
        questions = subsample(questions, n, seed=seed)

    return questions
