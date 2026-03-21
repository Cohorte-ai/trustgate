"""MMLU dataset loader."""

from __future__ import annotations

from theaios.trustgate.datasets import subsample
from theaios.trustgate.types import Question

# MMLU test set hosted as a CSV on HuggingFace
_MMLU_BASE_URL = (
    "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"
)

_CHOICES = ["A", "B", "C", "D"]


def _format_question(
    question: str,
    choices: list[str],
    subject: str,
) -> str:
    """Format an MMLU question with lettered options."""
    opts = "\n".join(f"  {letter}) {choice}" for letter, choice in zip(_CHOICES, choices))
    return f"[{subject}] {question}\n{opts}"


def load_mmlu(
    subjects: list[str] | None = None,
    n: int | None = None,
    seed: int = 42,
    data: list[dict[str, object]] | None = None,
) -> list[Question]:
    """Load MMLU multiple-choice questions.

    If *data* is provided, use it directly (useful for testing without download).
    Otherwise attempts to load from the HuggingFace ``datasets`` library,
    falling back to an empty list if unavailable.

    Each question has text with options + acceptable_answers (the letter).
    Recommended canonicalization: ``"mcq"``

    Args:
        subjects: Filter to specific subjects. None = all.
        n: Subsample to *n* questions. None = all.
        seed: Random seed for subsampling.
        data: Pre-loaded data rows (for testing).
    """
    if data is None:
        data = _load_mmlu_data()

    questions: list[Question] = []
    for i, row in enumerate(data):
        subject = str(row.get("subject", "unknown"))
        if subjects is not None and subject not in subjects:
            continue

        question_text = str(row.get("question", ""))
        raw_choices = row.get("choices", ["", "", "", ""])
        if isinstance(raw_choices, list):
            choices = [str(raw_choices[j]) if j < len(raw_choices) else "" for j in range(4)]
        else:
            choices = [str(row.get(f"choices_{j}", "")) for j in range(4)]
        answer_idx = int(str(row.get("answer", 0)))
        correct_letter = _CHOICES[answer_idx] if 0 <= answer_idx < 4 else "A"

        formatted = _format_question(question_text, choices, subject)
        questions.append(
            Question(
                id=f"mmlu_{i}",
                text=formatted,
                acceptable_answers=[correct_letter],
                metadata={"source": "mmlu", "subject": subject},
            )
        )

    if n is not None:
        questions = subsample(questions, n, seed=seed)

    return questions


def _load_mmlu_data() -> list[dict[str, object]]:
    """Try to load MMLU via the datasets library. Returns empty list on failure."""
    try:
        import datasets  # type: ignore[import-not-found]

        ds = datasets.load_dataset("cais/mmlu", "all", split="test")
        return [dict(row) for row in ds]
    except Exception:
        return []
