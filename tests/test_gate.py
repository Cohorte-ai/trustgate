"""Tests for the TrustGate runtime trust layer (passthrough + sampled modes)."""

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


# ===========================================================================
# Passthrough mode (default)
# ===========================================================================


class TestPassthroughMode:
    def test_single_api_call(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification())
        assert gate.mode == "passthrough"

        with patch.object(gate._adapter, "send", new=AsyncMock(return_value="B) 4")):
            result = gate.query("What is 2+2?")

        assert result.mode == "passthrough"
        assert result.answer == "B) 4"
        assert result.reliability_level == 0.90
        assert result.m_star == 1
        assert result.n_samples == 1  # single call
        assert result.prediction_set == []  # not computed in passthrough
        assert result.consensus == 0.0  # not computed in passthrough

    def test_reliability_metadata_attached(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(reliability=0.946))

        with patch.object(gate._adapter, "send", new=AsyncMock(return_value="Paris")):
            result = gate.query("Capital of France?")

        assert result.reliability_level == pytest.approx(0.946)
        assert result.m_star == 1


# ===========================================================================
# Sampled mode
# ===========================================================================


class TestSampledMode:
    def test_k_samples_with_profile(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(), mode="sampled")
        assert gate.mode == "sampled"

        responses = _mock_responses(["B", "B", "B", "B", "A"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("What is 2+2? (A) 3 (B) 4")

        assert result.mode == "sampled"
        assert result.answer == "B"
        assert result.consensus == 0.8
        assert result.prediction_set == ["B"]
        assert result.is_singleton is True
        assert result.n_samples == 5

    def test_prediction_set_with_m_star_2(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(m_star=2), mode="sampled")

        responses = _mock_responses(["B", "B", "A", "A", "C"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) X (B) Y (C) Z")

        assert len(result.prediction_set) == 2
        assert result.is_singleton is False

    def test_consensus_and_margin(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(), mode="sampled")

        responses = _mock_responses(["B", "B", "B", "B", "A"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) X (B) Y")

        assert result.consensus == pytest.approx(0.8)
        assert result.margin == pytest.approx(0.6)

    def test_unanimous(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(), mode="sampled")

        responses = _mock_responses(["B", "B", "B", "B", "B"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) X (B) Y")

        assert result.consensus == 1.0
        assert result.margin == 1.0

    def test_raw_responses_preserved(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(), mode="sampled")

        raw = ["The answer is B", "B) 4", "I think B", "B", "A"]
        responses = _mock_responses(raw)
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) 3 (B) 4")

        assert result.raw_responses == raw

    def test_full_profile(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(), mode="sampled")

        responses = _mock_responses(["B", "B", "A", "C", "B"])
        with patch.object(gate._sampler, "sample_question", new=AsyncMock(return_value=responses)):
            result = gate.query("Q? (A) X (B) Y (C) Z")

        assert len(result.profile) == 3
        assert result.profile[0][0] == "B"


# ===========================================================================
# Common
# ===========================================================================


class TestTrustGateCommon:
    def test_reliability_level_property(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification(reliability=0.946))
        assert gate.reliability_level == pytest.approx(0.946)

    def test_raises_on_invalid_config(self) -> None:
        with pytest.raises(Exception, match="Invalid configuration"):
            TrustGate(
                config=TrustGateConfig(endpoint=EndpointConfig(url="not-a-url")),
                certification=_certification(),
            )

    def test_raises_on_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            TrustGate(config=_config(), certification=_certification(), mode="bad")

    def test_default_mode_is_passthrough(self) -> None:
        gate = TrustGate(config=_config(), certification=_certification())
        assert gate.mode == "passthrough"


class TestGateResponse:
    def test_is_singleton_true(self) -> None:
        r = GateResponse(
            answer="B", prediction_set=["B"], consensus=1.0, margin=1.0,
            profile=[("B", 1.0)], m_star=1, reliability_level=0.9, mode="sampled",
        )
        assert r.is_singleton is True

    def test_is_singleton_false(self) -> None:
        r = GateResponse(
            answer="B", prediction_set=["B", "A"], consensus=0.6, margin=0.2,
            profile=[("B", 0.6), ("A", 0.4)], m_star=2, reliability_level=0.9,
            mode="sampled",
        )
        assert r.is_singleton is False

    def test_n_samples_passthrough(self) -> None:
        r = GateResponse(
            answer="B", m_star=1, reliability_level=0.9, mode="passthrough",
            raw_responses=["B"],
        )
        assert r.n_samples == 1

    def test_n_samples_sampled(self) -> None:
        r = GateResponse(
            answer="B", prediction_set=["B"], consensus=1.0, margin=1.0,
            profile=[("B", 1.0)], m_star=1, reliability_level=0.9,
            mode="sampled", raw_responses=["B"] * 7,
        )
        assert r.n_samples == 7
