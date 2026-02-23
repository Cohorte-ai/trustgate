"""Tests for the MMLU dataset loader."""

from __future__ import annotations

from trustgate.datasets.mmlu import load_mmlu


def _mock_data() -> list[dict[str, object]]:
    return [
        {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "math",
        },
        {
            "question": "Capital of France?",
            "choices": ["London", "Berlin", "Paris", "Rome"],
            "answer": 2,
            "subject": "geography",
        },
        {
            "question": "Who wrote Hamlet?",
            "choices": ["Dickens", "Shakespeare", "Austen", "Twain"],
            "answer": 1,
            "subject": "literature",
        },
    ]


class TestLoadMmlu:
    def test_returns_questions(self) -> None:
        questions = load_mmlu(data=_mock_data())
        assert len(questions) == 3

    def test_question_text_includes_options(self) -> None:
        questions = load_mmlu(data=_mock_data())
        assert "A)" in questions[0].text
        assert "B)" in questions[0].text
        assert "C)" in questions[0].text
        assert "D)" in questions[0].text

    def test_correct_answer_letter(self) -> None:
        questions = load_mmlu(data=_mock_data())
        # answer=1 → B, answer=2 → C
        assert questions[0].acceptable_answers == ["B"]
        assert questions[1].acceptable_answers == ["C"]

    def test_subject_filter(self) -> None:
        questions = load_mmlu(data=_mock_data(), subjects=["math"])
        assert len(questions) == 1
        assert questions[0].metadata["subject"] == "math"

    def test_subsample(self) -> None:
        questions = load_mmlu(data=_mock_data(), n=2)
        assert len(questions) == 2

    def test_metadata_source(self) -> None:
        questions = load_mmlu(data=_mock_data())
        assert all(q.metadata.get("source") == "mmlu" for q in questions)

    def test_question_text_includes_subject(self) -> None:
        questions = load_mmlu(data=_mock_data())
        assert "[math]" in questions[0].text
