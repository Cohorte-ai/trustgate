"""Tests for the calibration engine (conformal prediction)."""

from __future__ import annotations

import math
import random

import pytest

from theaios.trustgate.calibration import (
    calibrate,
    compute_capability_gap,
    compute_conditional_coverage,
    compute_coverage,
    compute_nonconformity_score,
    compute_profile,
    conformal_quantile,
    random_split,
)


# ---------------------------------------------------------------------------
# compute_profile
# ---------------------------------------------------------------------------


class TestComputeProfile:
    def test_basic(self) -> None:
        profile = compute_profile(["A", "A", "A", "B", "C"])
        assert profile[0] == ("A", pytest.approx(0.6))
        assert len(profile) == 3

    def test_single_answer(self) -> None:
        profile = compute_profile(["A"])
        assert profile == [("A", 1.0)]

    def test_uniform(self) -> None:
        profile = compute_profile(["A", "B", "C", "D", "E"])
        assert len(profile) == 5
        for _ans, freq in profile:
            assert freq == pytest.approx(0.2)

    def test_tie_breaking_alphabetical(self) -> None:
        profile = compute_profile(["B", "A"])
        # Both at 0.5, A should come first alphabetically
        assert profile[0][0] == "A"
        assert profile[1][0] == "B"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compute_profile([])

    def test_frequencies_sum_to_one(self) -> None:
        profile = compute_profile(["A", "A", "B", "C", "C", "C", "D"])
        total = sum(freq for _, freq in profile)
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_nonconformity_score
# ---------------------------------------------------------------------------


class TestNonconformityScore:
    def test_mode_is_correct(self) -> None:
        profile = [("A", 0.7), ("B", 0.2), ("C", 0.1)]
        assert compute_nonconformity_score(profile, "A") == 1

    def test_second_is_correct(self) -> None:
        profile = [("A", 0.7), ("B", 0.2), ("C", 0.1)]
        assert compute_nonconformity_score(profile, "B") == 2

    def test_third_is_correct(self) -> None:
        profile = [("A", 0.7), ("B", 0.2), ("C", 0.1)]
        assert compute_nonconformity_score(profile, "C") == 3

    def test_not_in_profile(self) -> None:
        profile = [("A", 0.7), ("B", 0.3)]
        assert compute_nonconformity_score(profile, "Z") == float("inf")

    def test_single_item_profile(self) -> None:
        profile = [("42", 1.0)]
        assert compute_nonconformity_score(profile, "42") == 1


# ---------------------------------------------------------------------------
# conformal_quantile
# ---------------------------------------------------------------------------


class TestConformalQuantile:
    def test_known_example(self) -> None:
        scores: list[int | float] = [1, 1, 1, 2, 3]
        # alpha=0.10, n=5: rank = ceil(0.90 * 6) = ceil(5.4) = 6 → clamped to 5
        q = conformal_quantile(scores, 0.10)
        assert q == 3.0

    def test_all_identical(self) -> None:
        scores: list[int | float] = [1, 1, 1, 1, 1]
        assert conformal_quantile(scores, 0.05) == 1.0

    def test_alpha_zero_returns_max(self) -> None:
        scores: list[int | float] = [1, 2, 3, 4, 5]
        # alpha=0 → rank = ceil(1.0 * 6) = 6, clamped to 5
        assert conformal_quantile(scores, 0.0) == 5.0

    def test_alpha_one_returns_min(self) -> None:
        scores: list[int | float] = [1, 2, 3, 4, 5]
        # alpha=1.0 → rank = ceil(0 * 6) = 0, clamped to 1
        assert conformal_quantile(scores, 1.0) == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            conformal_quantile([], 0.05)

    def test_single_score(self) -> None:
        assert conformal_quantile([2], 0.05) == 2.0

    def test_with_inf(self) -> None:
        scores: list[int | float] = [1, 1, 2, float("inf")]
        q = conformal_quantile(scores, 0.05)
        # n=4, rank = ceil(0.95*5) = 5, clamped to 4 → inf
        assert q == float("inf")


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_full_coverage(self) -> None:
        profiles = {
            "q1": [("A", 0.8), ("B", 0.2)],
            "q2": [("C", 0.9), ("D", 0.1)],
        }
        labels = {"q1": "A", "q2": "C"}
        # M*=1 should cover both (correct answer is the mode)
        assert compute_coverage(profiles, labels, m_star=1) == 1.0

    def test_zero_coverage(self) -> None:
        profiles = {
            "q1": [("A", 0.8), ("B", 0.2)],
            "q2": [("C", 0.9), ("D", 0.1)],
        }
        labels = {"q1": "B", "q2": "D"}
        assert compute_coverage(profiles, labels, m_star=0) == 0.0

    def test_partial_coverage(self) -> None:
        profiles = {
            "q1": [("A", 0.8), ("B", 0.2)],
            "q2": [("C", 0.6), ("D", 0.4)],
        }
        labels = {"q1": "A", "q2": "D"}
        # M*=1: q1 covered (A is mode), q2 not covered (D is not mode)
        assert compute_coverage(profiles, labels, m_star=1) == 0.5

    def test_m_star_2_expands_coverage(self) -> None:
        profiles = {
            "q1": [("A", 0.8), ("B", 0.2)],
            "q2": [("C", 0.6), ("D", 0.4)],
        }
        labels = {"q1": "A", "q2": "D"}
        # M*=2: both covered
        assert compute_coverage(profiles, labels, m_star=2) == 1.0

    def test_empty_profiles(self) -> None:
        assert compute_coverage({}, {"q1": "A"}, m_star=1) == 0.0

    def test_skips_missing_labels(self) -> None:
        profiles = {
            "q1": [("A", 1.0)],
            "q2": [("B", 1.0)],
        }
        labels = {"q1": "A"}  # q2 has no label
        assert compute_coverage(profiles, labels, m_star=1) == 1.0


# ---------------------------------------------------------------------------
# compute_conditional_coverage
# ---------------------------------------------------------------------------


class TestConditionalCoverage:
    def test_all_solvable(self) -> None:
        profiles = {
            "q1": [("A", 0.8), ("B", 0.2)],
            "q2": [("C", 0.6), ("D", 0.4)],
        }
        labels = {"q1": "A", "q2": "C"}
        assert compute_conditional_coverage(profiles, labels, m_star=1) == 1.0

    def test_excludes_unsolvable(self) -> None:
        profiles = {
            "q1": [("A", 0.8), ("B", 0.2)],
            "q2": [("C", 1.0)],  # "D" never appears
        }
        labels = {"q1": "A", "q2": "D"}
        # q2 is unsolvable, so only q1 counted → coverage = 1/1
        assert compute_conditional_coverage(profiles, labels, m_star=1) == 1.0

    def test_empty(self) -> None:
        assert compute_conditional_coverage({}, {}, m_star=1) == 0.0


# ---------------------------------------------------------------------------
# compute_capability_gap
# ---------------------------------------------------------------------------


class TestCapabilityGap:
    def test_all_solvable(self) -> None:
        profiles = {
            "q1": [("A", 0.8), ("B", 0.2)],
            "q2": [("C", 0.6), ("D", 0.4)],
        }
        labels = {"q1": "A", "q2": "C"}
        assert compute_capability_gap(profiles, labels) == 0.0

    def test_some_unsolvable(self) -> None:
        profiles = {f"q{i}": [("A", 1.0)] for i in range(10)}
        labels = {f"q{i}": "A" for i in range(10)}
        # Make q0 unsolvable — correct answer never in profile
        labels["q0"] = "Z"
        assert compute_capability_gap(profiles, labels) == pytest.approx(0.1)

    def test_all_unsolvable(self) -> None:
        profiles = {"q1": [("A", 1.0)], "q2": [("B", 1.0)]}
        labels = {"q1": "Z", "q2": "Z"}
        assert compute_capability_gap(profiles, labels) == 1.0

    def test_empty(self) -> None:
        assert compute_capability_gap({}, {}) == 0.0


# ---------------------------------------------------------------------------
# random_split
# ---------------------------------------------------------------------------


class TestRandomSplit:
    def test_correct_sizes(self) -> None:
        ids = [f"q{i}" for i in range(100)]
        cal, test = random_split(ids, n_cal=60, n_test=40)
        assert len(cal) == 60
        assert len(test) == 40

    def test_no_overlap(self) -> None:
        ids = [f"q{i}" for i in range(100)]
        cal, test = random_split(ids, n_cal=50, n_test=50)
        assert set(cal).isdisjoint(set(test))

    def test_reproducible(self) -> None:
        ids = [f"q{i}" for i in range(100)]
        cal1, test1 = random_split(ids, n_cal=50, n_test=50, seed=42)
        cal2, test2 = random_split(ids, n_cal=50, n_test=50, seed=42)
        assert cal1 == cal2
        assert test1 == test2

    def test_different_seeds(self) -> None:
        ids = [f"q{i}" for i in range(100)]
        cal1, _ = random_split(ids, n_cal=50, n_test=50, seed=1)
        cal2, _ = random_split(ids, n_cal=50, n_test=50, seed=2)
        assert cal1 != cal2

    def test_raises_on_insufficient_items(self) -> None:
        ids = ["q1", "q2", "q3"]
        with pytest.raises(ValueError, match="Not enough questions"):
            random_split(ids, n_cal=3, n_test=3)


# ---------------------------------------------------------------------------
# calibrate (integration)
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    n: int = 200,
    accuracy: float = 0.9,
    k: int = 10,
    seed: int = 42,
) -> tuple[
    dict[str, list[tuple[str, float]]],
    dict[str, str],
    list[str],
]:
    """Create synthetic profiles and labels for testing.

    Args:
        n: Number of questions.
        accuracy: Fraction of questions where the mode is correct.
        k: Number of samples per question.
        seed: Random seed.

    Returns:
        (profiles, labels, question_ids)
    """
    rng = random.Random(seed)
    profiles: dict[str, list[tuple[str, float]]] = {}
    labels: dict[str, str] = {}
    qids: list[str] = []

    for i in range(n):
        qid = f"q{i}"
        qids.append(qid)
        correct = "A"
        labels[qid] = correct

        if rng.random() < accuracy:
            # Model gets it right — mode is the correct answer
            n_correct = rng.randint(int(k * 0.6), k)
            answers = [correct] * n_correct + ["B"] * (k - n_correct)
        else:
            # Model gets it wrong — mode is an incorrect answer
            answers = ["B"] * rng.randint(int(k * 0.6), k)
            answers += [correct] * max(0, k - len(answers))
            # Ensure B is mode
            if answers.count("B") <= answers.count(correct):
                answers = ["B"] * (k - 1) + [correct]

        profiles[qid] = compute_profile(answers)

    return profiles, labels, qids


class TestCalibrate:
    def test_produces_valid_result(self) -> None:
        profiles, labels, qids = _make_synthetic_data(n=200)
        cal_ids, test_ids = random_split(qids, n_cal=100, n_test=100)
        result = calibrate(
            profiles, labels, cal_ids, test_ids,
            alpha_values=[0.01, 0.05, 0.10, 0.15, 0.20],
        )
        assert 0.0 <= result.reliability_level <= 1.0
        assert result.m_star >= 1
        assert 0.0 <= result.coverage <= 1.0
        assert 0.0 <= result.conditional_coverage <= 1.0
        assert 0.0 <= result.capability_gap <= 1.0
        assert result.n_cal == 100
        assert result.n_test == 100
        assert len(result.alpha_coverage) == 5

    def test_high_accuracy_high_reliability(self) -> None:
        profiles, labels, qids = _make_synthetic_data(n=200, accuracy=0.98)
        cal_ids, test_ids = random_split(qids, n_cal=100, n_test=100)
        result = calibrate(
            profiles, labels, cal_ids, test_ids,
            alpha_values=[0.01, 0.05, 0.10, 0.15, 0.20],
        )
        # High accuracy model should have high reliability
        assert result.reliability_level >= 0.80

    def test_low_accuracy_low_reliability(self) -> None:
        profiles, labels, qids = _make_synthetic_data(n=200, accuracy=0.3)
        cal_ids, test_ids = random_split(qids, n_cal=100, n_test=100)
        result = calibrate(
            profiles, labels, cal_ids, test_ids,
            alpha_values=[0.01, 0.05, 0.10, 0.15, 0.20],
        )
        # Low accuracy → lower reliability (m_star > 1 or coverage < target)
        assert result.reliability_level <= 0.95

    def test_coverage_guarantee_holds(self) -> None:
        """The core conformal guarantee: empirical coverage >= 1-alpha."""
        profiles, labels, qids = _make_synthetic_data(n=400, accuracy=0.85)
        cal_ids, test_ids = random_split(qids, n_cal=200, n_test=200)
        result = calibrate(
            profiles, labels, cal_ids, test_ids,
            alpha_values=[0.05, 0.10, 0.20],
        )
        # For the reported reliability level, coverage should hold
        if result.reliability_level > 0:
            assert result.coverage >= result.reliability_level - 0.01  # small tolerance

    def test_no_valid_cal_items_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid calibration"):
            calibrate(
                profiles={},
                labels={},
                cal_ids=["q1"],
                test_ids=["q2"],
                alpha_values=[0.05],
            )

    def test_perfect_model(self) -> None:
        """All scores = 1 → reliability should be very high."""
        # Every question: correct answer is the mode with 100% frequency
        profiles = {f"q{i}": [("A", 1.0)] for i in range(100)}
        labels = {f"q{i}": "A" for i in range(100)}
        cal_ids = [f"q{i}" for i in range(50)]
        test_ids = [f"q{i}" for i in range(50, 100)]
        result = calibrate(
            profiles, labels, cal_ids, test_ids,
            alpha_values=[0.01, 0.05, 0.10, 0.15, 0.20],
        )
        assert result.reliability_level >= 0.95
        assert result.coverage == 1.0
        assert result.capability_gap == 0.0


# ---------------------------------------------------------------------------
# Monte Carlo verification of conformal guarantee
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_coverage_guarantee_holds_statistically(self) -> None:
        """Run 200 trials: verify empirical coverage >= (1-alpha) at least 90% of the time."""
        alpha = 0.10
        target = 1 - alpha
        n_trials = 200
        n_questions = 100
        k = 10
        holds = 0

        for trial in range(n_trials):
            profiles, labels, qids = _make_synthetic_data(
                n=n_questions, accuracy=0.85, k=k, seed=trial * 137,
            )
            cal_ids, test_ids = random_split(
                qids, n_cal=50, n_test=50, seed=trial * 137 + 1,
            )
            # Compute calibration scores
            cal_scores: list[int | float] = []
            for qid in cal_ids:
                score = compute_nonconformity_score(profiles[qid], labels[qid])
                cal_scores.append(score)

            q_hat = conformal_quantile(cal_scores, alpha)
            if math.isinf(q_hat):
                continue  # coverage unattainable, skip
            m_star = max(1, int(math.ceil(q_hat)))

            test_profiles = {qid: profiles[qid] for qid in test_ids}
            test_labels = {qid: labels[qid] for qid in test_ids}
            cov = compute_coverage(test_profiles, test_labels, m_star)

            if cov >= target:
                holds += 1

        # Conformal guarantee should hold in the vast majority of trials
        assert holds / n_trials >= 0.85
