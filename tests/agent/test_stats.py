from __future__ import annotations

import pytest

from theaios.trustgate.agent.stats import (
    bootstrap_mean_interval,
    clopper_pearson_lower,
    min_units_to_approve,
    scenarios_needed,
    zero_failure_upper,
)


def test_clopper_pearson_known_values() -> None:
    assert clopper_pearson_lower(8, 10) == pytest.approx(0.4931, abs=1e-4)
    assert clopper_pearson_lower(10, 10) == pytest.approx(0.05**0.1)
    assert clopper_pearson_lower(0, 10) == 0.0
    assert clopper_pearson_lower(3, 3) == pytest.approx(0.368, abs=1e-3)


def test_lower_bound_is_monotone_in_successes_and_n() -> None:
    values = [clopper_pearson_lower(x, 20) for x in range(21)]
    assert values == sorted(values)
    assert clopper_pearson_lower(20, 20) < clopper_pearson_lower(40, 40)


def test_zero_failure_bound_and_rule_of_three() -> None:
    assert zero_failure_upper(300) == pytest.approx(0.00994, abs=1e-4)
    assert zero_failure_upper(100) == pytest.approx(3 / 100, rel=0.02)  # rule of three
    assert zero_failure_upper(0) == 1.0


def test_minimum_scenarios_to_approve() -> None:
    assert min_units_to_approve(0.8) == 14
    assert min_units_to_approve(0.9) == 29
    assert clopper_pearson_lower(14, 14) >= 0.8 > clopper_pearson_lower(13, 13)


def test_bootstrap_interval_contains_mean_and_is_deterministic() -> None:
    vals = [1.0, 0.8, 1.0, 0.6, 1.0, 1.0, 0.4, 1.0]
    lo, hi = bootstrap_mean_interval(vals)
    assert lo <= sum(vals) / len(vals) <= hi
    assert bootstrap_mean_interval(vals) == (lo, hi)


def test_scenarios_needed_with_failures() -> None:
    n = scenarios_needed(1, 0.8)
    assert n is not None
    assert clopper_pearson_lower(n - 1, n) >= 0.8 > clopper_pearson_lower(n - 2, n - 1)
    assert scenarios_needed(0, 0.8) == 14


def test_safety_scenarios_needed() -> None:
    from theaios.trustgate.agent.stats import safety_scenarios_needed

    assert safety_scenarios_needed(0.1) == 29
    assert zero_failure_upper(29) <= 0.1 < zero_failure_upper(28)
    with pytest.raises(ValueError):
        safety_scenarios_needed(1.5)
