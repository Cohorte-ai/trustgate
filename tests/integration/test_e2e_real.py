"""End-to-end tests with a real API endpoint.

Requires OPENAI_API_KEY environment variable.
Skipped in CI unless TRUSTGATE_E2E=1 is set.
"""

from __future__ import annotations

import os

import pytest

from theaios import trustgate

pytestmark = pytest.mark.skipif(
    os.getenv("TRUSTGATE_E2E") != "1",
    reason="E2E tests require TRUSTGATE_E2E=1 and API keys",
)


def test_e2e_certify_mmlu() -> None:
    """Certify GPT-4.1-mini on 20 MMLU questions."""
    from theaios.trustgate.datasets import load_mmlu

    questions = load_mmlu(subjects=["abstract_algebra"], n=20)
    labels = {q.id: q.acceptable_answers[0] for q in questions if q.acceptable_answers}

    result = trustgate.certify(
        config=trustgate.TrustGateConfig(
            endpoint=trustgate.EndpointConfig(
                url="https://api.openai.com/v1/chat/completions",
                model="gpt-4.1-mini",
                api_key_env="OPENAI_API_KEY",
            ),
            sampling=trustgate.SamplingConfig(k_fixed=5),
            canonicalization=trustgate.CanonConfig(type="mcq"),
            calibration=trustgate.CalibrationConfig(n_cal=10, n_test=10),
        ),
        questions=questions,
        labels=labels,
    )

    assert 0 <= result.reliability_level <= 1
    assert result.m_star >= 1
    assert 0 <= result.coverage <= 1
    assert 0 <= result.capability_gap <= 1
    assert result.n_cal == 10
    assert result.n_test == 10


def test_e2e_certify_gsm8k() -> None:
    """Certify GPT-4.1-mini on 20 GSM8K questions with sequential stopping."""
    from theaios.trustgate.datasets import load_gsm8k

    questions = load_gsm8k(n=20)
    labels = {q.id: q.acceptable_answers[0] for q in questions if q.acceptable_answers}

    result = trustgate.certify(
        config=trustgate.TrustGateConfig(
            endpoint=trustgate.EndpointConfig(
                url="https://api.openai.com/v1/chat/completions",
                model="gpt-4.1-mini",
                api_key_env="OPENAI_API_KEY",
                temperature=0.7,
            ),
            sampling=trustgate.SamplingConfig(
                k_max=10,
                sequential_stopping=True,
            ),
            canonicalization=trustgate.CanonConfig(type="numeric"),
            calibration=trustgate.CalibrationConfig(n_cal=10, n_test=10),
        ),
        questions=questions,
        labels=labels,
    )

    assert 0 <= result.reliability_level <= 1
    assert result.k_used <= 10
