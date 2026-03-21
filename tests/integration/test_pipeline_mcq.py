"""Integration test: full MCQ certification pipeline with mock endpoint."""

from __future__ import annotations

import random

import httpx
import pytest
import respx

from theaios.trustgate.certification import certify
from theaios.trustgate.types import (
    CalibrationConfig,
    CanonConfig,
    EndpointConfig,
    Question,
    SamplingConfig,
    TrustGateConfig,
)

MOCK_URL = "https://mock-llm.example.com/v1/chat/completions"

# 30 MCQ questions with known correct answers
_MCQ_DATA = [
    ("q01", "What is the capital of France?", "B"),
    ("q02", "Which planet is closest to the Sun?", "A"),
    ("q03", "What is 2 + 2?", "C"),
    ("q04", "Which element has symbol O?", "B"),
    ("q05", "What color is the sky?", "A"),
    ("q06", "Who wrote Hamlet?", "D"),
    ("q07", "What is the boiling point of water?", "C"),
    ("q08", "Which continent is Brazil in?", "B"),
    ("q09", "What is the largest mammal?", "A"),
    ("q10", "How many days in a week?", "D"),
    ("q11", "What is the speed of light?", "A"),
    ("q12", "Which language is most spoken?", "C"),
    ("q13", "What year did WW2 end?", "B"),
    ("q14", "Who painted the Mona Lisa?", "A"),
    ("q15", "What is the chemical formula for water?", "D"),
    ("q16", "What is the largest ocean?", "C"),
    ("q17", "How many continents are there?", "B"),
    ("q18", "What is the tallest mountain?", "A"),
    ("q19", "Which gas do plants absorb?", "D"),
    ("q20", "What is the freezing point of water?", "C"),
    ("q21", "Who discovered penicillin?", "A"),
    ("q22", "What is the smallest planet?", "B"),
    ("q23", "Which metal is liquid at room temp?", "D"),
    ("q24", "What is the longest river?", "C"),
    ("q25", "Who invented the telephone?", "A"),
    ("q26", "What is the human body temperature?", "B"),
    ("q27", "Which vitamin comes from sunlight?", "D"),
    ("q28", "What is the hardest natural substance?", "A"),
    ("q29", "How many bones in the human body?", "C"),
    ("q30", "What is the closest star to Earth?", "B"),
]


def _questions() -> list[Question]:
    return [
        Question(id=qid, text=text, acceptable_answers=[answer])
        for qid, text, answer in _MCQ_DATA
    ]


def _labels() -> dict[str, str]:
    return {qid: answer for qid, _text, answer in _MCQ_DATA}


def _config() -> TrustGateConfig:
    return TrustGateConfig(
        endpoint=EndpointConfig(
            url=MOCK_URL,
            model="mock-model",
            api_key_env="MOCK_KEY",
            provider="generic",
        ),
        sampling=SamplingConfig(
            k_fixed=10,
            k_max=20,
            sequential_stopping=False,
            max_concurrent=10,
            retries=0,
        ),
        canonicalization=CanonConfig(type="mcq"),
        calibration=CalibrationConfig(
            n_cal=15,
            n_test=15,
            alpha_values=[0.05, 0.10, 0.20],
        ),
    )


def _mock_mcq_response(correct_answer: str, accuracy: float = 0.8) -> str:
    """Generate a mock MCQ response. Returns correct answer with given probability."""
    rng = random.Random()  # varies per call
    if rng.random() < accuracy:
        return f"The answer is ({correct_answer})"
    else:
        wrong = rng.choice([c for c in "ABCDE" if c != correct_answer])
        return f"The answer is ({wrong})"


@pytest.fixture(autouse=True)
def _set_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_KEY", "sk-mock-key")


# Track call count to simulate varied responses
_call_counter: dict[str, int] = {}


@respx.mock
def test_full_mcq_pipeline() -> None:
    """End-to-end MCQ pipeline with mock endpoint."""
    random.seed(42)

    def _responder(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        import json

        data = json.loads(body)
        prompt = data["messages"][0]["content"]

        # Find which question this is
        correct = "B"  # default
        for qid, text, answer in _MCQ_DATA:
            if text in prompt:
                correct = answer
                break

        response_text = _mock_mcq_response(correct, accuracy=0.85)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": response_text}}],
            },
        )

    respx.post(MOCK_URL).mock(side_effect=_responder)

    result = certify(
        config=_config(),
        questions=_questions(),
        labels=_labels(),
    )

    # Verify result structure
    assert 0.0 <= result.reliability_level <= 1.0
    assert result.m_star >= 1
    assert 0.0 <= result.coverage <= 1.0
    assert 0.0 <= result.conditional_coverage <= 1.0
    assert 0.0 <= result.capability_gap <= 1.0
    assert result.n_cal > 0
    assert result.n_test > 0
    assert result.k_used > 0
    assert len(result.alpha_coverage) == 3


@respx.mock
def test_mcq_pipeline_uses_acceptable_answers() -> None:
    """Pipeline can derive labels from questions' acceptable_answers."""
    random.seed(123)

    def _responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "The answer is (B)"}}]},
        )

    respx.post(MOCK_URL).mock(side_effect=_responder)

    result = certify(
        config=_config(),
        questions=_questions(),
        # No explicit labels — should derive from acceptable_answers
    )

    assert result.reliability_level >= 0.0
    assert result.n_cal > 0
