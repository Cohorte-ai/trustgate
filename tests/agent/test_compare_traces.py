from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from theaios.trustgate.agent.compare import clopper_pearson_upper, compare_reports
from theaios.trustgate.agent.scenario import ScenarioError, load_scenarios
from theaios.trustgate.agent.traces import candidates_from_traces
from theaios.trustgate.cli import main


def report(results: dict[str, tuple[str, bool]], k: int = 3, scen_hash: str = "h1") -> dict:  # type: ignore[type-arg]
    return {
        "schema": "trustgate.agent-qualification/1",
        "k": k,
        "cost_usd": 0.03,
        "envelope": {"scenarios": {"s.yaml": scen_hash}},
        "scenarios": [
            {
                "id": i,
                "suite": suite,
                "all_pass": ok,
                "statuses": {"passed": k} if ok else {"failed": 1, "passed": k - 1},
            }
            for i, (suite, ok) in results.items()
        ],
        "runs": [{"status": "passed"}] * (len(results) * k),
    }


def many(n: int, suite: str = "nominal", ok: bool = True) -> dict[str, tuple[str, bool]]:
    return {f"{suite}-{i}": (suite, ok) for i in range(n)}


def test_upper_bound_matches_rule_of_three() -> None:
    assert clopper_pearson_upper(0, 100) == pytest.approx(0.0295, abs=1e-3)


def test_identical_large_suites_are_non_inferior() -> None:
    a = report(many(60))
    c = compare_reports(a, a, margin=0.05)
    assert c.verdict == "non_inferior" and c.regression_upper < 0.05


def test_small_suites_cannot_prove_non_inferiority() -> None:
    a = report(many(10))
    c = compare_reports(a, a, margin=0.05)
    assert c.verdict == "unresolved"  # 0/10 regressions still allows up to 26%


def test_safety_regression_is_a_veto() -> None:
    old = report({**many(60), "safe-1": ("safety", True)})
    new = report({**many(60), "safe-1": ("safety", False)})
    c = compare_reports(old, new, margin=0.5)
    assert c.verdict == "regression" and c.safety_regressions == ["safe-1"]


def test_many_regressions_are_a_regression_and_gains_do_not_offset() -> None:
    old = report({**many(40), **{f"x{i}": ("nominal", False) for i in range(40)}})
    new_results = {
        **{f"nominal-{i}": ("nominal", i >= 20) for i in range(40)},
        **{f"x{i}": ("nominal", True) for i in range(40)},
    }
    c = compare_reports(old, report(new_results), margin=0.05)
    assert c.verdict == "regression" and len(c.improved) == 40


def test_unpaired_reports_are_refused() -> None:
    with pytest.raises(ValueError, match="different scenario files"):
        compare_reports(report(many(5)), report(many(5), scen_hash="h2"), 0.05)
    with pytest.raises(ValueError, match="different K"):
        compare_reports(report(many(5)), report(many(5), k=5), 0.05)


def test_traces_become_candidates_that_do_not_load(tmp_path: Path) -> None:
    lines = [
        json.dumps(
            {
                "id": "conv-1",
                "messages": [
                    {"role": "system", "content": "x"},
                    {"role": "user", "content": "Refund A1"},
                    {"role": "assistant", "tool_calls": [{"function": {"name": "refund"}}]},
                    {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
                ],
            }
        ),
        json.dumps(
            {
                "id": "conv-2",
                "messages": [
                    {"role": "user", "content": "Refund A1"},
                    {"role": "user", "content": "thanks"},
                ],
            }
        ),
        json.dumps(
            {
                "scenario": "s",
                "turns": ["hello"],
                "transcript": {"turns": [{"text": "hi", "tool_calls": [{"name": "greet"}]}]},
            }
        ),
    ]
    f = tmp_path / "t.jsonl"
    f.write_text("\n".join(lines))
    items = candidates_from_traces(f)
    assert len(items) == 2  # conv-2 duplicates conv-1's user turns
    assert items[0]["turns"] == ["Refund A1", "thanks"] and "refund" in items[0]["todo"]
    out = tmp_path / "c.yaml"
    res = CliRunner().invoke(main, ["cases-from-traces", str(f), "-o", str(out)])
    assert res.exit_code == 0 and yaml.safe_load(out.read_text())[1]["turns"] == ["hello"]
    with pytest.raises(ScenarioError):
        load_scenarios([out])  # todo + empty expect: refused until a person completes them


def test_diff_command(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(report(many(60))))
    b.write_text(json.dumps(report(many(60))))
    res = CliRunner().invoke(main, ["diff", str(a), str(b), "--margin", "0.05"])
    assert res.exit_code == 0 and "non_inferior" in res.output
    res = CliRunner().invoke(main, ["diff", str(a), str(b), "--margin", "0.01"])
    assert res.exit_code == 1 and "unresolved" in res.output
