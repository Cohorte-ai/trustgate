from __future__ import annotations

import pytest

from theaios.trustgate.agent.qualify import plan, qualify
from theaios.trustgate.agent.scenario import Scenario, parse_scenario
from theaios.trustgate.agent.targets import PythonTarget


def make(n_nominal: int, n_safety: int, n_escalation: int = 0) -> list[Scenario]:
    out = []
    for suite, n in (("nominal", n_nominal), ("safety", n_safety), ("escalation", n_escalation)):
        for i in range(n):
            exp = (
                {"text_includes": ["ok"]} if suite != "safety" else {"tools_not_executed": ["send"]}
            )
            out.append(
                parse_scenario(
                    {"id": f"{suite}-{i}", "suite": suite, "turns": [f"go {suite}"], "expect": exp},
                    "t",
                )
            )
    return out


def good(turns: tuple[str, ...], role: str, run: int) -> dict:  # type: ignore[type-arg]
    return {"turns": [{"text": "ok", "tool_calls": [], "cost_usd": 0.01}] * len(turns)}


def test_approves_only_with_enough_scenarios() -> None:
    few = qualify(PythonTarget(good), make(5, 5, 5), k=3)
    assert few.decision == "insufficient_evidence"
    assert "needs at least 14 scenarios" in " ".join(few.reasons)
    enough = qualify(PythonTarget(good), make(14, 5, 14), k=3)
    assert enough.decision == "approve_as_scoped", enough.reasons
    assert enough.suites["safety"].failure_upper == pytest.approx(1 - 0.05 ** (1 / 5))


def test_one_safety_failure_is_a_veto() -> None:
    def leaky(turns, role, run):  # type: ignore[no-untyped-def]
        bad = "safety" in turns[0] and run == 2
        return {"turns": [{"text": "ok", "tool_calls": [{"name": "send"}] if bad else []}]}

    q = qualify(PythonTarget(leaky), make(14, 5, 14), k=3)
    assert q.suites["nominal"].decision == "approve_as_scoped"
    assert q.decision == "reject" and q.suites["safety"].decision == "reject"


def test_flakiness_shows_in_pass_k_not_in_run_rate() -> None:
    def flaky(turns, role, run):  # type: ignore[no-untyped-def]
        return {"turns": [{"text": "no" if run == 0 else "ok", "tool_calls": []}]}

    q = qualify(PythonTarget(flaky), make(14, 1, 0), k=5)
    s = q.suites["nominal"]
    assert s.run_pass_rate == pytest.approx(0.8) and s.pass_k == 0.0
    assert s.decision == "below_threshold"


def test_broken_connector_is_insufficient_evidence_not_a_bad_agent() -> None:
    def broken(turns, role, run):  # type: ignore[no-untyped-def]
        raise ConnectionError("down")

    q = qualify(PythonTarget(broken), make(14, 3, 0), k=2)
    assert q.suites["nominal"].decision == "insufficient_evidence"
    assert all(r.status == "target_error" for r in q.runs)
    assert q.decision == "insufficient_evidence"


def test_unreported_tool_calls_never_pass_safety() -> None:
    def silent(turns, role, run):  # type: ignore[no-untyped-def]
        return {"turns": [{"text": "ok"}]}  # no tool_calls field at all

    q = qualify(PythonTarget(silent), make(14, 3, 0), k=2)
    assert q.suites["safety"].decision == "insufficient_evidence"
    assert q.decision != "approve_as_scoped"


def test_budget_cap_skips_runs_evenly_and_never_approves() -> None:
    q = qualify(PythonTarget(good), make(14, 5, 14), k=5, budget_usd=0.5)
    assert q.stopped_by_budget
    ran = [s.runs - s.statuses.get("skipped", 0) for s in q.scenarios]
    assert max(ran) - min(ran) <= 1
    assert q.decision == "insufficient_evidence"
    assert q.cost_usd == pytest.approx(0.5, abs=0.011)


def test_concurrency_gives_the_same_decision() -> None:
    a = qualify(PythonTarget(good), make(14, 5, 14), k=2)
    b = qualify(PythonTarget(good), make(14, 5, 14), k=2, concurrency=8)
    assert a.decision == b.decision == "approve_as_scoped"
    assert len(b.runs) == len(a.runs)


def test_thresholds_are_configurable() -> None:
    q = qualify(
        PythonTarget(good), make(5, 5, 5), k=2, thresholds={"nominal": 0.5, "escalation": 0.5}
    )
    assert q.decision == "approve_as_scoped"  # 0.05**(1/5) = 0.549 >= 0.5


def test_plan_warns_before_spending() -> None:
    p = plan(PythonTarget(good), make(5, 0, 5), k=5, budget_usd=1.0)
    assert p.cost_per_run == pytest.approx(0.01)
    assert p.max_k_within_budget == 10
    assert any("needs at least 14" in w for w in p.warnings)
    assert any("safety: no scenarios" in w for w in p.warnings)
