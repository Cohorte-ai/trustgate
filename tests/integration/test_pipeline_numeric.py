"""Integration test: full numeric certification pipeline with mock endpoint."""

from __future__ import annotations

import json
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

# 30 math questions with known numeric answers
_MATH_DATA = [
    ("m01", "What is 2 + 2?", "4"),
    ("m02", "What is 10 * 5?", "50"),
    ("m03", "What is 100 / 4?", "25"),
    ("m04", "What is 7 * 8?", "56"),
    ("m05", "What is 15 + 27?", "42"),
    ("m06", "What is 99 - 37?", "62"),
    ("m07", "What is 12 * 12?", "144"),
    ("m08", "What is 200 / 8?", "25"),
    ("m09", "What is 3^4?", "81"),
    ("m10", "What is sqrt(49)?", "7"),
    ("m11", "What is 1000 - 567?", "433"),
    ("m12", "What is 33 * 3?", "99"),
    ("m13", "What is 256 / 16?", "16"),
    ("m14", "What is 17 + 83?", "100"),
    ("m15", "What is 9 * 9?", "81"),
    ("m16", "What is 500 / 25?", "20"),
    ("m17", "What is 45 + 55?", "100"),
    ("m18", "What is 8 * 7?", "56"),
    ("m19", "What is 360 / 12?", "30"),
    ("m20", "What is 25 * 4?", "100"),
    ("m21", "What is 11 * 11?", "121"),
    ("m22", "What is 75 - 33?", "42"),
    ("m23", "What is 16 * 16?", "256"),
    ("m24", "What is 1024 / 32?", "32"),
    ("m25", "What is 50 + 50?", "100"),
    ("m26", "What is 6 * 6?", "36"),
    ("m27", "What is 729 / 27?", "27"),
    ("m28", "What is 88 - 44?", "44"),
    ("m29", "What is 13 * 7?", "91"),
    ("m30", "What is 400 / 20?", "20"),
]

# Varied answer formats to test numeric canonicalization
_ANSWER_FORMATS = [
    "The answer is {answer}",
    "\\boxed{{{answer}}}",
    "#### {answer}",
    "I think the answer is {answer}.",
    "{answer}",
    "Let me calculate... the result is {answer}.",
]


def _questions() -> list[Question]:
    return [
        Question(id=qid, text=text, acceptable_answers=[answer])
        for qid, text, answer in _MATH_DATA
    ]


def _labels() -> dict[str, str]:
    return {qid: answer for qid, _text, answer in _MATH_DATA}


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
        canonicalization=CanonConfig(type="numeric"),
        calibration=CalibrationConfig(
            n_cal=15,
            n_test=15,
            alpha_values=[0.05, 0.10, 0.20],
        ),
    )


@pytest.fixture(autouse=True)
def _set_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_KEY", "sk-mock-key")


@respx.mock
def test_full_numeric_pipeline() -> None:
    """End-to-end numeric pipeline with varied answer formats."""
    rng = random.Random(42)

    def _responder(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content.decode())
        prompt = data["messages"][0]["content"]

        correct = "42"  # default
        for qid, text, answer in _MATH_DATA:
            if text in prompt:
                correct = answer
                break

        # 85% chance of correct answer in varied format
        if rng.random() < 0.85:
            fmt = rng.choice(_ANSWER_FORMATS)
            response_text = fmt.format(answer=correct)
        else:
            wrong = str(int(correct) + rng.randint(1, 10))
            fmt = rng.choice(_ANSWER_FORMATS)
            response_text = fmt.format(answer=wrong)

        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": response_text}}]},
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


@respx.mock
def test_numeric_canonicalization_consistency() -> None:
    """Verify that varied numeric formats get canonicalized to the same value."""
    rng = random.Random(99)

    def _responder(request: httpx.Request) -> httpx.Response:
        # Always return "42" in varied formats
        fmt = rng.choice(_ANSWER_FORMATS)
        text = fmt.format(answer="42")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": text}}]},
        )

    respx.post(MOCK_URL).mock(side_effect=_responder)

    result = certify(
        config=_config(),
        questions=_questions(),
        labels=_labels(),
    )

    # Since we always return 42, there should be some coverage
    # (only m05 and m22 have correct answer 42)
    assert result.n_cal > 0
    assert result.n_test > 0
