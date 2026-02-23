"""TruthfulQA dataset loader."""

from __future__ import annotations

from trustgate.datasets import subsample
from trustgate.types import Question


def load_truthfulqa(
    n: int | None = None,
    seed: int = 42,
    data: list[dict[str, object]] | None = None,
) -> list[Question]:
    """Load TruthfulQA questions.

    If *data* is provided, use it directly (useful for testing without download).
    Otherwise attempts to load from the HuggingFace ``datasets`` library,
    falling back to an empty list if unavailable.

    Open-ended questions about common misconceptions.
    Recommended canonicalization: ``"llm_judge"`` or ``"embedding"``

    Args:
        n: Subsample to *n* questions. None = all.
        seed: Random seed for subsampling.
        data: Pre-loaded data rows (for testing).
    """
    if data is None:
        data = _load_truthfulqa_data()

    questions: list[Question] = []
    for i, row in enumerate(data):
        question_text = str(row.get("question", ""))
        best_answer = str(row.get("best_answer", ""))

        questions.append(
            Question(
                id=f"truthfulqa_{i}",
                text=question_text,
                acceptable_answers=[best_answer] if best_answer else None,
                metadata={"source": "truthfulqa"},
            )
        )

    if n is not None:
        questions = subsample(questions, n, seed=seed)

    return questions


def _load_truthfulqa_data() -> list[dict[str, object]]:
    """Try to load TruthfulQA via the datasets library. Returns empty list on failure."""
    try:
        import datasets  # type: ignore[import-not-found]

        ds = datasets.load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        return [dict(row) for row in ds]
    except Exception:
        return []
