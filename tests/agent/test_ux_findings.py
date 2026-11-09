"""Regression tests for the findings of the 2026-11-09 usability test (three personas)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from theaios.trustgate.agent.compare import compare_reports
from theaios.trustgate.agent.qualify import _probe_sample, plan, qualify
from theaios.trustgate.agent.report import render_card
from theaios.trustgate.agent.scenario import ScenarioError, check, parse_scenario
from theaios.trustgate.agent.targets import HttpTarget, PythonTarget
from theaios.trustgate.agent.transcript import ToolCallRecord, Transcript, TurnObservation
from theaios.trustgate.cli import main


def sc(expect: dict, suite: str = "safety"):  # type: ignore[no-untyped-def,type-arg]
    return parse_scenario({"id": "s", "suite": suite, "turns": ["hi"], "expect": expect}, "t")


NO_TOOLS = Transcript(turns=(TurnObservation(text="Done, renewed."),))
FORBIDDEN = Transcript(
    turns=(
        TurnObservation(
            text="Done.", tool_calls=(ToolCallRecord("renew", arguments={"id": "B4"}),)
        ),
    )
)


@pytest.mark.parametrize(
    "expect",
    [
        {"never_executed_with": [{"tool": "renew", "arguments": {"id": "B4"}}]},
        {"tools_not_executed": ["renew"]},
        {"tools_not_called": ["renew"]},
    ],
)
def test_safety_checks_never_pass_without_tool_evidence(expect: dict) -> None:  # type: ignore[type-arg]
    v = check(sc(expect), NO_TOOLS)
    assert not v.passed and v.evidence == "incomplete" and v.failures == ()
    assert not check(sc(expect), FORBIDDEN).passed  # and they still catch the violation


def test_missing_evidence_is_not_an_agent_failure() -> None:
    for expect in ({"tools_executed": ["lookup"]}, {"cites": ["doc-1"]}):
        v = check(sc(expect, "nominal"), NO_TOOLS)
        assert v.outcome == "pass" and v.evidence == "incomplete" and not v.passed


def test_a_violation_in_partial_evidence_is_still_proven() -> None:
    partial = Transcript(
        turns=(
            TurnObservation(text="a", tool_calls=(ToolCallRecord("renew"),)),
            TurnObservation(text="b", tool_calls=None),
        )
    )
    s = parse_scenario(
        {
            "id": "s",
            "suite": "safety",
            "turns": ["x", "y"],
            "expect": {"tools_not_executed": ["renew"]},
        },
        "t",
    )
    v = check(s, partial)
    assert v.outcome == "fail" and "'renew' was executed" in v.failures


def test_text_failure_shows_the_answer() -> None:
    v = check(
        sc({"text_includes": ["2"]}, "nominal"),
        Transcript(turns=(TurnObservation(text="It is due on Friday."),)),
    )
    assert 'answer: "It is due on Friday."' in v.failures[0]


def test_http_absent_field_is_unknown_and_bad_status_is_a_failed_run(respx_mock) -> None:  # type: ignore[no-untyped-def]
    import httpx

    respx_mock.post("https://a.test/chat").mock(
        return_value=httpx.Response(200, json={"answer": "ok"})
    )
    t = HttpTarget(
        url="https://a.test/chat",
        request={},
        response={"text": "answer", "tool_calls": "actions", "citations": "sources"},
    ).run(("x",), "u", 0)
    assert t.error is None and t.turns[0].tool_calls is None and t.turns[0].citations is None
    respx_mock.post("https://b.test/chat").mock(
        return_value=httpx.Response(
            200, json={"answer": "ok", "actions": [{"name": "reset", "status": "pending_approval"}]}
        )
    )
    bad = HttpTarget(
        url="https://b.test/chat", request={}, response={"text": "answer", "tool_calls": "actions"}
    ).run(("x",), "u", 0)
    assert (
        bad.error is not None and "invalid reply" in bad.error and "pending_approval" in bad.error
    )


def _report(statuses: dict[str, dict[str, int]], all_pass: dict[str, bool]) -> dict:  # type: ignore[type-arg]
    return {
        "schema": "trustgate.agent-qualification/1",
        "k": 3,
        "cost_usd": 0.0,
        "envelope": {"scenarios": {"s.yaml": "h"}},
        "scenarios": [
            {
                "id": i,
                "suite": "safety" if i.startswith("safe") else "nominal",
                "all_pass": all_pass[i],
                "statuses": st,
            }
            for i, st in statuses.items()
        ],
        "runs": [{"status": "passed"}],
    }


def test_diff_never_calls_a_broken_connector_a_regression() -> None:
    ids = [f"n{i}" for i in range(40)] + ["safe-1", "safe-2"]
    old = _report({i: {"passed": 3} for i in ids}, dict.fromkeys(ids, True))
    new = _report({i: {"target_error": 3} for i in ids}, dict.fromkeys(ids, False))
    c = compare_reports(old, new, margin=0.05)
    assert c.verdict == "unresolved" and not c.regressed and not c.safety_regressions


def test_probe_sample_spreads_across_suites() -> None:
    scen = [
        parse_scenario(
            {"id": f"{s}{i}", "suite": s, "turns": ["x"], "expect": {"outcome": ["a"]}}, "t"
        )
        for s, n in (("nominal", 14), ("escalation", 14), ("safety", 8))
        for i in range(n)
    ]
    picked = _probe_sample(scen, 6)
    assert len({x.id for x in picked}) == 6 and {x.suite for x in picked} == {
        "nominal",
        "escalation",
        "safety",
    }


def test_plan_does_not_blame_cost_reporting_when_probes_failed() -> None:
    def down(turns, role, run):  # type: ignore[no-untyped-def]
        raise ConnectionError("down")

    p = plan(PythonTarget(down), [sc({"outcome": ["a"]}, "nominal")], k=1)
    assert not any("does not report cost" in w for w in p.warnings)
    assert any("probe run(s) failed" in w for w in p.warnings)


def test_cost_not_reported_is_said_not_zeroed() -> None:
    def free(turns, role, run):  # type: ignore[no-untyped-def]
        return {"turns": [{"text": "ok", "outcome": "a"}]}

    q = qualify(PythonTarget(free), [sc({"outcome": ["a"]}, "nominal")], k=1)
    assert q.cost_reported is False
    from theaios.trustgate.agent.report import build_report

    card = render_card(build_report(q, {"target": {"type": "python"}}, "fp", {}))
    assert "not reported by the target" in card and "How to read this card" in card


def test_unknown_keys_suggest_and_todo_explains() -> None:
    with pytest.raises(ScenarioError, match="did you mean 'tools_executed'"):
        sc({"tools_excuted": ["x"]})
    with pytest.raises(ScenarioError, match="unfinished candidate"):
        parse_scenario({"id": "a", "turns": ["x"], "expect": {}, "todo": "x"}, "t")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "s.yaml").write_text("- {id: a, turns: [hi], expect: {outcome: [ok]}}\n")
    (tmp_path / "agent.py").write_text(
        "def f(turns, role, run):\n    return {'turns': [{'text': 'x', 'outcome': 'ok'}]}\n"
    )
    (tmp_path / "q.yaml").write_text(
        'target: {type: python, function: "agent:f"}\nscenarios: ["s.yaml"]\nk: 2\n'
    )
    return tmp_path


@pytest.mark.parametrize("k", ["0", "-1"])
def test_k_below_one_is_refused_on_the_command_line(project: Path, k: str) -> None:
    for cmd in ("qualify", "plan"):
        res = CliRunner().invoke(main, [cmd, "-c", str(project / "q.yaml"), "-k", k])
        assert res.exit_code == 2 and "Invalid value" in res.output


def test_unset_header_variable_is_refused_before_running(project: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TG_TEST_TOKEN_UNSET", raising=False)
    (project / "h.yaml").write_text(
        'target: {type: http, url: "http://127.0.0.1:9/x", '
        'headers: {Authorization: "Bearer ${TG_TEST_TOKEN_UNSET}"}}\nscenarios: ["s.yaml"]\n'
    )
    res = CliRunner().invoke(main, ["qualify", "-c", str(project / "h.yaml")])
    assert res.exit_code == 1 and "TG_TEST_TOKEN_UNSET" in res.output


def test_syntax_error_in_agent_module_is_one_clean_line(project: Path) -> None:
    (project / "broken_agent.py").write_text("def f(:\n")
    (project / "b.yaml").write_text(
        'target: {type: python, function: "broken_agent:f"}\nscenarios: ["s.yaml"]\n'
    )
    res = CliRunner().invoke(main, ["qualify", "-c", str(project / "b.yaml")])
    assert res.exit_code == 1 and "cannot import 'broken_agent': SyntaxError" in res.output
    assert "Traceback" not in res.output


def test_no_cost_warning_when_budget_is_set(project: Path) -> None:
    res = CliRunner().invoke(main, ["qualify", "-c", str(project / "q.yaml"), "--budget", "5"])
    assert "Cost: not reported by the target" in res.output
    assert "budget cap could not limit spending" in res.output
    report = json.loads((project / ".trustgate/qualification/report.json").read_text())
    assert report["cost_reported"] is False


def test_traces_report_merges_and_drop_non_text_parts(tmp_path: Path) -> None:
    from theaios.trustgate.agent.traces import candidates_from_traces

    lines = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                }
            ]
        },
        {"messages": [{"role": "user", "content": "What is this?"}]},
        {"messages": [{"role": "assistant", "content": "hello"}]},
    ]
    f = tmp_path / "t.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines))
    stats: dict[str, int] = {}
    items = candidates_from_traces(f, stats=stats)
    assert [i["turns"] for i in items] == [["What is this?"]]
    assert stats == {"lines": 3, "duplicates": 1, "no_user_turns": 1}
