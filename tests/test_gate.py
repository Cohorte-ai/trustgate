"""Tests for the TrustGate runtime trust layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from theaios.trustgate.gate import GateResponse, TrustGate
from theaios.trustgate.types import (
    CanonConfig,
    CertificationResult,
    EndpointConfig,
    SamplingConfig,
    SampleResponse,
    TrustGateConfig,
)


def _config() -> TrustGateConfig:
    return TrustGateConfig(
        endpoint=EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1-mini",
            api_key_env="TEST_KEY",
        ),
        sampling=SamplingConfig(k_fixed=5, sequential_stopping=False),
        canonicalization=CanonConfig(type="mcq"),
    )


def _certification(m_star: int = 1, reliability: float = 0.90) -> CertificationResult:
    return CertificationResult(
        reliability_level=reliability,
        m_star=m_star,
        coverage=0.95,
        conditional_coverage=0.98,
        capability_gap=0.02,
        n_cal=50,
        n_test=50,
        k_used=5,
        api_cost_estimate=1.0,
    )


def _mock_responses(answers: list[str], qid: str = "_gate") -> list[SampleResponse]:
    return [
        SampleResponse(question_id=qid, sample_index=i, raw_response=a)
        for i, a in enumerate(answers)
    ]


class TestTrustGate:
    def test_query_returns_gate_response(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification())

        responses = _mock_responses(["B", "B", "B", "B", "A"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("What is 2+2? (A) 3 (B) 4")

        assert isinstance(result, GateResponse)
        assert result.answer == "B"
        assert result.consensus == 0.8
        assert result.m_star == 1
        assert result.prediction_set == ["B"]
        assert result.is_singleton is True
        assert result.n_samples == 5

    def test_prediction_set_with_m_star_2(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(m_star=2))

        responses = _mock_responses(["B", "B", "A", "A", "C"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Hard question? (A) X (B) Y (C) Z")

        assert result.m_star == 2
        assert len(result.prediction_set) == 2
        assert result.answer == result.prediction_set[0]
        assert result.is_singleton is False

    def test_consensus_and_margin(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification())

        # Strong consensus: 4 out of 5 say B
        responses = _mock_responses(["B", "B", "B", "B", "A"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) X (B) Y")

        assert result.consensus == pytest.approx(0.8)
        assert result.margin == pytest.approx(0.6)  # 0.8 - 0.2

    def test_unanimous_consensus(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification())

        responses = _mock_responses(["B", "B", "B", "B", "B"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) X (B) Y")

        assert result.consensus == 1.0
        assert result.margin == 1.0  # only one class, margin = frequency
        assert result.prediction_set == ["B"]

    def test_raw_responses_preserved(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification())

        raw = ["The answer is B", "B) 4", "I think B", "B", "A"]
        responses = _mock_responses(raw)
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) 3 (B) 4")

        assert result.raw_responses == raw

    def test_full_profile_returned(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification())

        responses = _mock_responses(["B", "B", "A", "C", "B"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) X (B) Y (C) Z")

        assert len(result.profile) == 3
        assert result.profile[0][0] == "B"
        assert result.profile[0][1] == pytest.approx(0.6)

    def test_reliability_level_exposed(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(reliability=0.946))
        assert gate.reliability_level == pytest.approx(0.946)

    def test_raises_on_invalid_config(self) -> None:
        with pytest.raises(Exception, match="Invalid configuration"):
            TrustGate(
                config=TrustGateConfig(endpoint=EndpointConfig(url="not-a-url")),
                certification=_certification(),
            )

    def test_custom_question_id(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification())

        responses = _mock_responses(["B", "B", "B", "B", "B"], qid="custom_q1")
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q?", question_id="custom_q1")

        assert result.answer == "B"


class TestGateResponse:
    def test_is_singleton_true(self) -> None:
        r = GateResponse(
            answer="B", prediction_set=["B"], consensus=1.0, margin=1.0,
            profile=[("B", 1.0)], m_star=1,
        )
        assert r.is_singleton is True

    def test_is_singleton_false(self) -> None:
        r = GateResponse(
            answer="B", prediction_set=["B", "A"], consensus=0.6, margin=0.2,
            profile=[("B", 0.6), ("A", 0.4)], m_star=2,
        )
        assert r.is_singleton is False

    def test_n_samples(self) -> None:
        r = GateResponse(
            answer="B", prediction_set=["B"], consensus=1.0, margin=1.0,
            profile=[("B", 1.0)], m_star=1, raw_responses=["B"] * 7,
        )
        assert r.n_samples == 7
