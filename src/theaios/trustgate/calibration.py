"""Conformal calibration: nonconformity scores, M*, coverage guarantees."""

from __future__ import annotations

import math
import random
from collections import Counter

from theaios.trustgate.types import CertificationResult


def compute_profile(canonical_answers: list[str]) -> list[tuple[str, float]]:
    """Compute the self-consistency profile from K canonical answers.

    Returns [(answer, frequency), ...] sorted by frequency descending,
    then alphabetically for tie-breaking.
    """
    if not canonical_answers:
        raise ValueError("canonical_answers must not be empty")

    counts = Counter(canonical_answers)
    total = len(canonical_answers)
    items = [(answer, count / total) for answer, count in counts.items()]
    # Sort by frequency descending, then alphabetically ascending for ties
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


def compute_nonconformity_score(
    profile: list[tuple[str, float]],
    correct_answer: str,
) -> int | float:
    """Find the smallest M such that the top-M answers include the correct answer.

    Returns:
    - int M (1-indexed) if the correct answer is in the profile
    - float('inf') if the correct answer is not in the profile at all
    """
    for i, (answer, _freq) in enumerate(profile, start=1):
        if answer == correct_answer:
            return i
    return float("inf")


def conformal_quantile(scores: list[int | float], alpha: float) -> float:
    """Compute the conformal quantile threshold.

    Uses the standard split conformal formula:
        q_hat = the ceil((1-alpha)*(n+1))-th smallest score
    """
    n = len(scores)
    if n == 0:
        raise ValueError("scores must not be empty")

    sorted_scores = sorted(scores)
    # Conformal quantile index (1-indexed)
    rank = math.ceil((1 - alpha) * (n + 1))
    # Clamp to valid range
    rank = max(1, min(rank, n))
    return float(sorted_scores[rank - 1])


def compute_coverage(
    profiles: dict[str, list[tuple[str, float]]],
    labels: dict[str, str],
    m_star: int,
) -> float:
    """Empirical coverage: fraction of items where top-M* answers contain the correct answer."""
    if not profiles:
        return 0.0

    covered = 0
    total = 0
    for qid, profile in profiles.items():
        if qid not in labels:
            continue
        total += 1
        correct = labels[qid]
        top_m_answers = {ans for ans, _freq in profile[:m_star]}
        if correct in top_m_answers:
            covered += 1

    return covered / total if total > 0 else 0.0


def compute_conditional_coverage(
    profiles: dict[str, list[tuple[str, float]]],
    labels: dict[str, str],
    m_star: int,
) -> float:
    """Coverage on solvable items only (where correct answer appears somewhere in profile)."""
    if not profiles:
        return 0.0

    covered = 0
    solvable = 0
    for qid, profile in profiles.items():
        if qid not in labels:
            continue
        correct = labels[qid]
        all_answers = {ans for ans, _freq in profile}
        if correct not in all_answers:
            continue  # unsolvable — skip
        solvable += 1
        top_m_answers = {ans for ans, _freq in profile[:m_star]}
        if correct in top_m_answers:
            covered += 1

    return covered / solvable if solvable > 0 else 0.0


def compute_capability_gap(
    profiles: dict[str, list[tuple[str, float]]],
    labels: dict[str, str],
) -> float:
    """Fraction of items where the correct answer never appears in any of the K samples."""
    if not profiles:
        return 0.0

    unsolvable = 0
    total = 0
    for qid, profile in profiles.items():
        if qid not in labels:
            continue
        total += 1
        correct = labels[qid]
        all_answers = {ans for ans, _freq in profile}
        if correct not in all_answers:
            unsolvable += 1

    return unsolvable / total if total > 0 else 0.0


def random_split(
    question_ids: list[str],
    n_cal: int,
    n_test: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split question IDs into calibration and test sets.

    Uses a fixed seed for reproducibility.
    """
    if len(question_ids) < n_cal + n_test:
        raise ValueError(
            f"Not enough questions ({len(question_ids)}) for "
            f"n_cal={n_cal} + n_test={n_test} = {n_cal + n_test}"
        )

    rng = random.Random(seed)
    shuffled = list(question_ids)
    rng.shuffle(shuffled)
    return shuffled[:n_cal], shuffled[n_cal : n_cal + n_test]


def calibrate(
    profiles: dict[str, list[tuple[str, float]]],
    labels: dict[str, str],
    cal_ids: list[str],
    test_ids: list[str],
    alpha_values: list[float],
) -> CertificationResult:
    """Full calibration pipeline.

    1. Compute nonconformity scores on cal_ids.
    2. For each alpha, find M* (conformal quantile threshold).
    3. Evaluate coverage on test_ids.
    4. Find the reliability level (largest 1-alpha with coverage >= 1-alpha).
    5. Compute conditional coverage and capability gap on test set.
    """
    # --- 1. Nonconformity scores on calibration set ---
    cal_scores: list[int | float] = []
    for qid in cal_ids:
        if qid not in profiles or qid not in labels:
            continue
        score = compute_nonconformity_score(profiles[qid], labels[qid])
        cal_scores.append(score)

    if not cal_scores:
        raise ValueError("No valid calibration items (no matching profiles/labels)")

    # --- 2 & 3. For each alpha, compute M* and test coverage ---
    sorted_alphas = sorted(alpha_values)
    alpha_coverage: dict[float, float] = {}
    best_reliability = 0.0
    best_m_star = 1

    test_profiles = {qid: profiles[qid] for qid in test_ids if qid in profiles}
    test_labels = {qid: labels[qid] for qid in test_ids if qid in labels}

    for alpha in sorted_alphas:
        q_hat = conformal_quantile(cal_scores, alpha)
        if math.isinf(q_hat):
            # Infinite quantile means coverage is unattainable at this alpha
            alpha_coverage[alpha] = 0.0
            continue
        m_star = max(1, int(math.ceil(q_hat)))
        cov = compute_coverage(test_profiles, test_labels, m_star)
        alpha_coverage[alpha] = cov

        target = 1 - alpha
        if cov >= target and target > best_reliability:
            best_reliability = target
            best_m_star = m_star

    # --- 4 & 5. Compute additional metrics on test set ---
    coverage = compute_coverage(test_profiles, test_labels, best_m_star)
    cond_cov = compute_conditional_coverage(test_profiles, test_labels, best_m_star)
    cap_gap = compute_capability_gap(test_profiles, test_labels)

    return CertificationResult(
        reliability_level=best_reliability,
        m_star=best_m_star,
        coverage=coverage,
        conditional_coverage=cond_cov,
        capability_gap=cap_gap,
        n_cal=len(cal_scores),
        n_test=len(test_profiles),
        k_used=0,  # filled by the pipeline
        api_cost_estimate=0.0,  # filled by the pipeline
        alpha_coverage=alpha_coverage,
    )
