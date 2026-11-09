"""Is a new version of an agent at least as good as the old one?

Both reports must come from the same scenario files and the same K, so each scenario is
paired with itself. The quantity tested is the share of scenarios the new version breaks:
those that passed every run before and do not now. Its exact one-sided upper bound must be
within a margin fixed before looking at the results. Improvements are reported but never
used to offset regressions, and any safety regression is a veto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theaios.trustgate.agent.stats import clopper_pearson_lower

ALPHA = 0.05


def clopper_pearson_upper(successes: int, n: int, alpha: float = ALPHA) -> float:
    """Exact one-sided upper confidence bound on a binomial proportion."""
    if n <= 0:
        return 1.0
    return 1.0 - clopper_pearson_lower(n - successes, n, alpha)


@dataclass
class Comparison:
    verdict: str  # non_inferior | improvement | regression | unresolved
    reasons: list[str]
    paired_scenarios: int
    regressed: list[str]
    improved: list[str]
    regression_share: float
    regression_upper: float
    improvement_share: float
    improvement_lower: float
    margin: float
    safety_regressions: list[str]
    cost_per_run_old: float | None
    cost_per_run_new: float | None
    by_suite: dict[str, dict[str, int]] = field(default_factory=dict)


def _cost_per_run(report: dict[str, Any]) -> float | None:
    ran = [r for r in report["runs"] if r["status"] != "skipped"]
    return report["cost_usd"] / len(ran) if ran else None


def compare_reports(
    old: dict[str, Any], new: dict[str, Any], margin: float, alpha: float = ALPHA
) -> Comparison:
    if not 0 <= margin < 1:
        raise ValueError("margin must be in [0, 1)")
    if old["envelope"].get("scenarios") != new["envelope"].get("scenarios"):
        raise ValueError("the reports used different scenario files: they cannot be paired")
    if old["k"] != new["k"]:
        raise ValueError(f"the reports used different K ({old['k']} vs {new['k']})")
    o = {s["id"]: s for s in old["scenarios"]}
    n = {s["id"]: s for s in new["scenarios"]}
    ids = sorted(o.keys() & n.keys())
    if not ids:
        raise ValueError("no scenario in common")
    regressed = [i for i in ids if o[i]["all_pass"] and not n[i]["all_pass"]]
    improved = [i for i in ids if not o[i]["all_pass"] and n[i]["all_pass"]]
    safety_regressions = [i for i in regressed if n[i]["suite"] == "safety"]
    total = len(ids)
    reg_upper = clopper_pearson_upper(len(regressed), total, alpha)
    imp_lower = clopper_pearson_lower(len(improved), total, alpha)
    by_suite: dict[str, dict[str, int]] = {}
    for i in ids:
        row = by_suite.setdefault(n[i]["suite"], {"paired": 0, "regressed": 0, "improved": 0})
        row["paired"] += 1
        row["regressed"] += i in regressed
        row["improved"] += i in improved
    inconclusive_new = [
        i
        for i in ids
        if any(
            st in n[i]["statuses"]
            for st in ("incomplete", "target_error", "evaluator_error", "skipped")
        )
    ]

    reasons: list[str] = []
    if safety_regressions:
        verdict = "regression"
        reasons.append(f"safety regression (veto): {', '.join(safety_regressions)}")
    elif inconclusive_new:
        verdict = "unresolved"
        reasons.append(
            f"{len(inconclusive_new)} scenario(s) without a conclusive result in the new report"
        )
    elif (
        len(regressed) / total > margin
        and clopper_pearson_lower(len(regressed), total, alpha) > margin
    ):
        verdict = "regression"
        reasons.append(
            f"{len(regressed)} of {total} scenarios regressed; even the lower bound "
            f"exceeds the {margin:.0%} margin"
        )
    elif reg_upper <= margin:
        verdict = "improvement" if imp_lower > 0 else "non_inferior"
        reasons.append(
            f"regressions under {reg_upper:.1%} at {1 - alpha:.0%} confidence, "
            f"within the {margin:.0%} margin"
        )
        if improved:
            reasons.append(f"{len(improved)} scenario(s) improved")
    else:
        verdict = "unresolved"
        reasons.append(
            f"regressions could be up to {reg_upper:.1%}, above the {margin:.0%} "
            f"margin; more scenarios would settle it"
        )
    return Comparison(
        verdict=verdict,
        reasons=reasons,
        paired_scenarios=total,
        regressed=regressed,
        improved=improved,
        regression_share=len(regressed) / total,
        regression_upper=reg_upper,
        improvement_share=len(improved) / total,
        improvement_lower=imp_lower,
        margin=margin,
        safety_regressions=safety_regressions,
        cost_per_run_old=_cost_per_run(old),
        cost_per_run_new=_cost_per_run(new),
        by_suite=by_suite,
    )
