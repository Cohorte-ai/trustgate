"""Statistics for agent qualification. The unit of evidence is the scenario, not the run."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def binom_tail_ge(successes: int, n: int, p: float) -> float:
    """P(X >= successes) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(successes, n + 1))


def clopper_pearson_lower(successes: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided lower confidence bound on a binomial proportion."""
    if n <= 0 or successes <= 0:
        return 0.0
    if successes == n:
        return float(alpha ** (1 / n))
    lo, hi = 0.0, successes / n
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if binom_tail_ge(successes, n, mid) < alpha else (lo, mid)
    return lo


def zero_failure_upper(n: int, alpha: float = 0.05) -> float:
    """With 0 failures in n independent units, the one-sided upper bound on the failure rate."""
    return 1.0 if n <= 0 else float(1 - alpha ** (1 / n))


def min_units_to_approve(threshold: float, alpha: float = 0.05) -> int:
    """Fewest units such that a perfect record can push the lower bound to the threshold."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    return math.ceil(math.log(alpha) / math.log(threshold))


def safety_scenarios_needed(max_failure_share: float, alpha: float = 0.05) -> int:
    """Fewest failure-free units for the zero-failure upper bound to reach max_failure_share."""
    if not 0 < max_failure_share < 1:
        raise ValueError("max_failure_share must be in (0, 1)")
    return math.ceil(math.log(alpha) / math.log(1 - max_failure_share))


def scenarios_needed(
    failing: int, threshold: float, alpha: float = 0.05, limit: int = 2000
) -> int | None:
    """Fewest units n such that (n - failing) successes out of n reach the threshold."""
    for n in range(failing + 1, limit + 1):
        if clopper_pearson_lower(n - failing, n, alpha) >= threshold:
            return n
    return None


def bootstrap_mean_interval(
    values: Sequence[float], alpha: float = 0.05, draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval of a mean, resampling units (here: scenarios)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(draws))
    lo = means[int(math.floor(alpha / 2 * (draws - 1)))]
    hi = means[int(math.ceil((1 - alpha / 2) * (draws - 1)))]
    return (lo, hi)
