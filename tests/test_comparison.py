"""Tests for the multi-model comparison module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from theaios.trustgate.comparison import compare, compute_comparison_summary
from theaios.trustgate.types import (
    CertificationResult,
    EndpointConfig,
    Question,
    SamplingConfig,
    TrustGateConfig,
)


def _make_result(
    reliability: float = 0.90,
    m_star: int = 1,
    coverage: float = 0.95,
    gap: float = 0.02,
) -> CertificationResult:
    return CertificationResult(
        reliability_level=reliability,
        m_star=m_star,
        coverage=coverage,
        conditional_coverage=0.98,
        capability_gap=gap,
        n_cal=50,
        n_test=50,
        k_used=10,
        api_cost_estimate=5.0,
    )


def _make_config() -> TrustGateConfig:
    return TrustGateConfig(
        endpoint=EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1",
            api_key_env="TEST_KEY",
        ),
        sampling=SamplingConfig(k_fixed=5, retries=0),
    )


def _make_questions(n: int = 10) -> list[Question]:
    return [Question(id=f"q{i}", text=f"Q{i}") for i in range(n)]


class TestCompare:
    def test_runs_for_each_model(self) -> None:
        results = [
            _make_result(reliability=0.95),
            _make_result(reliability=0.90),
        ]
        call_count = 0

        async def mock_certify_async(**kwargs: object) -> CertificationResult:
            nonlocal call_count
            r = results[call_count]
            call_count += 1
            return r

        with patch("theaios.trustgate.comparison.certify_async", side_effect=mock_certify_async):
            output = compare(
                models=["model-a", "model-b"],
                config=_make_config(),
                questions=_make_questions(),
                labels={"q0": "A"},
            )

        assert len(output) == 2
        assert call_count == 2

    def test_sorted_by_reliability_descending(self) -> None:
        results = [
            _make_result(reliability=0.80),
            _make_result(reliability=0.95),
            _make_result(reliability=0.90),
        ]

        async def mock_certify_async(**kwargs: object) -> CertificationResult:
            return results.pop(0)

        with patch("theaios.trustgate.comparison.certify_async", side_effect=mock_certify_async):
            output = compare(
                models=["low", "high", "mid"],
                config=_make_config(),
                questions=_make_questions(),
                labels={"q0": "A"},
            )

        reliabilities = [r.reliability_level for _, r in output]
        assert reliabilities == sorted(reliabilities, reverse=True)

    def test_single_model(self) -> None:
        async def mock_certify_async(**kwargs: object) -> CertificationResult:
            return _make_result(reliability=0.92)

        with patch("theaios.trustgate.comparison.certify_async", side_effect=mock_certify_async):
            output = compare(
                models=["only-model"],
                config=_make_config(),
                questions=_make_questions(),
                labels={"q0": "A"},
            )

        assert len(output) == 1
        assert output[0][0] == "only-model"


class TestComputeComparisonSummary:
    def test_correct_ranking(self) -> None:
        results = [
            ("model-a", _make_result(reliability=0.95)),
            ("model-b", _make_result(reliability=0.85)),
            ("model-c", _make_result(reliability=0.90)),
        ]
        summary = compute_comparison_summary(results)

        assert summary["best_model"] == "model-a"
        assert summary["best_reliability"] == 0.95

        models = summary["models"]
        assert isinstance(models, list)
        assert len(models) == 3
        assert models[0]["rank"] == 1
        assert models[0]["name"] == "model-a"
        assert models[2]["rank"] == 3

    def test_pairwise_deltas(self) -> None:
        results = [
            ("model-a", _make_result(reliability=0.95, coverage=0.96)),
            ("model-b", _make_result(reliability=0.85, coverage=0.90)),
        ]
        summary = compute_comparison_summary(results)

        deltas = summary["deltas"]
        assert isinstance(deltas, dict)
        assert len(deltas) == 1  # one pair
        key = ("model-a", "model-b")
        assert key in deltas
        assert deltas[key]["reliability_delta"] == pytest.approx(0.10)

    def test_empty_results(self) -> None:
        summary = compute_comparison_summary([])
        assert summary["best_model"] == ""
        assert summary["models"] == []
