"""Tests for the Hoeffding-based sequential stopping."""

from __future__ import annotations

import asyncio
import math
from unittest.mock import patch

import pytest

from trustgate.sequential import SequentialSampler, hoeffding_bound, should_stop
from trustgate.types import (
    EndpointConfig,
    Question,
    SampleResponse,
    SamplingConfig,
    TrustGateConfig,
)


# ---------------------------------------------------------------------------
# hoeffding_bound
# ---------------------------------------------------------------------------


class TestHoeffdingBound:
    def test_known_value(self) -> None:
        # epsilon = sqrt(log(2/0.05) / (2*100)) = sqrt(log(40) / 200)
        expected = math.sqrt(math.log(40) / 200)
        assert hoeffding_bound(100, 0.05) == pytest.approx(expected)

    def test_decreases_with_k(self) -> None:
        eps_10 = hoeffding_bound(10, 0.05)
        eps_100 = hoeffding_bound(100, 0.05)
        eps_1000 = hoeffding_bound(1000, 0.05)
        assert eps_10 > eps_100 > eps_1000

    def test_increases_with_lower_delta(self) -> None:
        eps_high_conf = hoeffding_bound(100, 0.01)  # tighter
        eps_low_conf = hoeffding_bound(100, 0.10)  # looser
        assert eps_high_conf > eps_low_conf

    def test_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            hoeffding_bound(0, 0.05)

    def test_delta_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="delta must be in"):
            hoeffding_bound(100, 0.0)

    def test_delta_one_raises(self) -> None:
        with pytest.raises(ValueError, match="delta must be in"):
            hoeffding_bound(100, 1.0)


# ---------------------------------------------------------------------------
# should_stop
# ---------------------------------------------------------------------------


class TestShouldStop:
    def test_false_with_k1(self) -> None:
        assert should_stop(["A"], k=1, delta=0.05) is False

    def test_true_when_dominant(self) -> None:
        # 20 "A"s out of 20 → p_hat=1.0, epsilon small → should stop
        answers = ["A"] * 20
        assert should_stop(answers, k=20, delta=0.05) is True

    def test_false_when_flat(self) -> None:
        # 10 "A" and 10 "B" → p_hat = 0.5, so p_hat - eps < 0.5
        answers = ["A"] * 10 + ["B"] * 10
        assert should_stop(answers, k=20, delta=0.05) is False

    def test_false_with_few_samples(self) -> None:
        # Even with all same, k=2 should have too large epsilon
        answers = ["A", "A"]
        # epsilon at k=2 is large, p_hat=1.0 but 1.0 - eps might still be > 0.5
        # Actually at k=2, eps = sqrt(log(40)/4) ≈ 0.96, so 1.0 - 0.96 = 0.04 < 0.5
        assert should_stop(answers, k=2, delta=0.05) is False

    def test_dominant_mode_enough_samples(self) -> None:
        # Need enough samples that p_hat - eps > 0.5
        # p_hat=0.9, eps = sqrt(log(40)/(2*50)) ≈ 0.27 → 0.9-0.27=0.63 > 0.5
        answers = ["A"] * 45 + ["B"] * 5
        assert should_stop(answers, k=50, delta=0.05) is True

    def test_empty_answers(self) -> None:
        assert should_stop([], k=0, delta=0.05) is False


# ---------------------------------------------------------------------------
# SequentialSampler
# ---------------------------------------------------------------------------


def _make_config() -> TrustGateConfig:
    return TrustGateConfig(
        endpoint=EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4",
            api_key_env="TEST_KEY",
        ),
        sampling=SamplingConfig(
            k_max=20,
            max_concurrent=5,
            timeout=30.0,
            retries=0,
        ),
    )


def _make_question(qid: str = "q1") -> Question:
    return Question(id=qid, text="What is 2+2?")


class TestSequentialSampler:
    def test_stops_early_when_dominant(self) -> None:
        """If every response is the same, should stop well before k_max."""
        config = _make_config()

        with patch("trustgate.sampler.resolve_api_key", return_value="sk-test"):
            from trustgate.sampler import Sampler

            sampler = Sampler(config)

        # Mock _sample_one to always return the same answer
        call_count = 0

        async def mock_sample_one(
            client: object,
            question: Question,
            index: int,
            semaphore: object = None,
        ) -> SampleResponse:
            nonlocal call_count
            call_count += 1
            return SampleResponse(
                question_id=question.id,
                sample_index=index,
                raw_response="4",
            )

        sampler._sample_one = mock_sample_one  # type: ignore[assignment]

        seq = SequentialSampler(sampler, delta=0.05)
        responses = asyncio.run(seq.sample_question(_make_question(), k_max=50))

        # Should have stopped early (well before 50)
        assert len(responses) < 50
        assert call_count < 50
        # All answers should be "4"
        assert all(r.raw_response == "4" for r in responses)

    def test_uses_all_k_when_uncertain(self) -> None:
        """If answers alternate, should use all k_max."""
        config = _make_config()

        with patch("trustgate.sampler.resolve_api_key", return_value="sk-test"):
            from trustgate.sampler import Sampler

            sampler = Sampler(config)

        async def mock_sample_one(
            client: object,
            question: Question,
            index: int,
            semaphore: object = None,
        ) -> SampleResponse:
            # Alternate between A and B
            answer = "A" if index % 2 == 0 else "B"
            return SampleResponse(
                question_id=question.id,
                sample_index=index,
                raw_response=answer,
            )

        sampler._sample_one = mock_sample_one  # type: ignore[assignment]

        seq = SequentialSampler(sampler, delta=0.05)
        responses = asyncio.run(seq.sample_question(_make_question(), k_max=20))

        # Should use all 20 (50/50 split never triggers stopping)
        assert len(responses) == 20

    def test_sample_all(self) -> None:
        config = _make_config()

        with patch("trustgate.sampler.resolve_api_key", return_value="sk-test"):
            from trustgate.sampler import Sampler

            sampler = Sampler(config)

        async def mock_sample_one(
            client: object,
            question: Question,
            index: int,
            semaphore: object = None,
        ) -> SampleResponse:
            return SampleResponse(
                question_id=question.id,
                sample_index=index,
                raw_response="4",
            )

        sampler._sample_one = mock_sample_one  # type: ignore[assignment]

        seq = SequentialSampler(sampler, delta=0.05)
        questions = [
            Question(id="q1", text="Q1"),
            Question(id="q2", text="Q2"),
        ]
        results = asyncio.run(seq.sample_all(questions, k_max=50))

        assert "q1" in results
        assert "q2" in results
        # Both should have stopped early
        assert len(results["q1"]) < 50
        assert len(results["q2"]) < 50


# ---------------------------------------------------------------------------
# compute_savings
# ---------------------------------------------------------------------------


class TestComputeSavings:
    def test_basic(self) -> None:
        actual_k = {"q1": 5, "q2": 10, "q3": 20}
        savings = SequentialSampler.compute_savings(actual_k, k_max=20)
        assert savings["total_possible"] == 60
        assert savings["total_actual"] == 35
        assert savings["saved"] == 25
        assert savings["savings_pct"] == pytest.approx(25 / 60)

    def test_no_savings(self) -> None:
        actual_k = {"q1": 20, "q2": 20}
        savings = SequentialSampler.compute_savings(actual_k, k_max=20)
        assert savings["saved"] == 0
        assert savings["savings_pct"] == 0.0

    def test_all_early_stop(self) -> None:
        actual_k = {"q1": 5, "q2": 3}
        savings = SequentialSampler.compute_savings(actual_k, k_max=20)
        assert savings["total_possible"] == 40
        assert savings["total_actual"] == 8
        assert savings["saved"] == 32
        assert savings["savings_pct"] == pytest.approx(0.8)

    def test_empty(self) -> None:
        savings = SequentialSampler.compute_savings({}, k_max=20)
        assert savings["total_possible"] == 0
        assert savings["total_actual"] == 0
        assert savings["savings_pct"] == 0.0
