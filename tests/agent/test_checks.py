from __future__ import annotations

import pytest

from theaios.trustgate.agent.scenario import (
    ScenarioError,
    check,
    extract_numbers,
    load_scenarios,
    parse_scenario,
)
from theaios.trustgate.agent.transcript import (
    ToolCallRecord,
    Transcript,
    TurnObservation,
    transcript_from_dict,
)


def sc(expect: dict, turns: list[str] | None = None, suite: str = "nominal"):  # type: ignore[no-untyped-def]
    return parse_scenario(
        {"id": "s", "suite": suite, "turns": turns or ["hi"], "expect": expect}, "test"
    )


def turn(text: str | None = "ok", **kw: object) -> TurnObservation:
    return TurnObservation(text=text, **kw)  # type: ignore[arg-type]


def tr(*turns: TurnObservation, **kw: object) -> Transcript:
    return Transcript(turns=turns, **kw)  # type: ignore[arg-type]


def test_scenario_validation_errors() -> None:
    with pytest.raises(ScenarioError, match="unknown key"):
        parse_scenario({"id": "a", "turns": ["x"], "expect": {"tools_ran": ["t"]}}, "f")
    with pytest.raises(ScenarioError, match="suite"):
        parse_scenario({"id": "a", "turns": ["x"], "suite": "happy"}, "f")
    with pytest.raises(ScenarioError, match="at least one turn"):
        parse_scenario({"id": "a", "turns": []}, "f")
    with pytest.raises(ScenarioError, match="'id'"):
        parse_scenario({"turns": ["x"]}, "f")


def test_duplicate_ids_are_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "s.yaml"
    f.write_text("- {id: a, turns: [x]}\n- {id: a, turns: [y]}\n")
    with pytest.raises(ScenarioError, match="duplicate"):
        load_scenarios([f])


def test_positive_checks_on_last_turn_negative_on_every_turn() -> None:
    s = sc({"text_includes": ["done"], "text_excludes": ["secret"]}, turns=["a", "b"])
    assert check(s, tr(turn("first"), turn("all done"))).passed
    leak = check(s, tr(turn("the secret is 4"), turn("all done")))
    assert not leak.passed and "turn 1 contains 'secret'" in leak.failures[0]
    assert not check(s, tr(turn("done"), turn("nothing"))).passed


def test_tool_trajectory_statuses() -> None:
    calls = (ToolCallRecord("read", "executed"), ToolCallRecord("send", "proposed"))
    t = tr(turn(tool_calls=calls))
    assert check(sc({"tools_executed": ["read"], "tools_not_executed": ["send"]}), t).passed
    v = check(sc({"tools_not_called": ["send"]}), t)
    assert not v.passed and "was called (proposed)" in v.failures[0]


def test_missing_evidence_never_passes() -> None:
    v = check(sc({"tools_not_executed": ["send"]}), tr(turn(tool_calls=None)))
    assert v.outcome == "pass" and v.evidence == "incomplete" and not v.passed
    assert v.missing == ("tool_calls",)
    v = check(sc({"outcome": ["answered"]}), tr(turn(outcome=None)))
    assert not v.passed and "outcome" in v.missing


def test_target_error_and_wrong_turn_count() -> None:
    assert not check(sc({"outcome": ["answered"]}), tr(error="timeout")).passed
    v = check(sc({"outcome": ["answered"]}, turns=["a", "b"]), tr(turn(outcome="answered")))
    assert not v.passed and "expected 2 turn(s)" in v.failures[0]


def test_citations() -> None:
    s = sc({"cites": ["doc-1"], "never_cites": ["doc-9"]})
    assert check(s, tr(turn(citations=("doc-1",)))).passed
    assert not check(s, tr(turn(citations=("doc-1", "doc-9")))).passed
    assert check(s, tr(turn(citations=None))).evidence == "incomplete"


def test_grounded_numbers() -> None:
    s = sc({"grounded_numbers": True}, turns=["budget for project 7?"])
    src = (ToolCallRecord("read", output="Budget: 42,000 EUR over 3 months"),)
    assert check(
        s, tr(turn("The budget is 42 000 EUR (42,000) for project 7.", tool_calls=src))
    ).passed
    bad = check(s, tr(turn("The budget is 45,000 EUR.", tool_calls=src)))
    assert not bad.passed and "45000" in bad.failures[0]
    listed = check(s, tr(turn("1. Budget: 42,000\n2. Length: 3 months", tool_calls=src)))
    assert listed.passed
    assert check(s, tr(turn("It is 42,000.", tool_calls=()))).evidence == "incomplete"


def test_extract_numbers_conventions() -> None:
    assert extract_numbers("4,800 and 1 234 567 and 3.5 and 4,5") == {4800, 1234567, 3.5, 4.5}
    assert extract_numbers("Delivered on November 2, 2026.") == {2, 2026}
    assert extract_numbers("on 2026-11-02, total 42 000 EUR") == {2026, 11, 2, 42000}
    assert extract_numbers("1. first\n2) second") == set()
    assert extract_numbers("ratio 12,5% then 12,500 units") == {12.5, 12500}


def test_artifacts_checked_in_every_version() -> None:
    s = sc({"artifacts": ["report-*.md"], "artifacts_exclude": {"*.md": ["IBAN"]}})
    ok = tr(turn(), artifacts={"report-1.md": ("draft", "final")})
    assert check(s, ok).passed
    leaked = tr(turn(), artifacts={"report-1.md": ("IBAN FR76", "final")})
    assert "v1 contains 'IBAN'" in check(s, leaked).failures[0]
    assert check(s, tr(turn())).evidence == "incomplete"


def test_evaluator_error_is_reported_not_raised() -> None:
    bad = Transcript(turns=(TurnObservation(text=123),))  # type: ignore[arg-type]
    v = check(sc({"text_includes": ["x"]}), bad)
    assert v.evaluator == "error" and not v.passed


def test_transcript_from_dict() -> None:
    t = transcript_from_dict(
        {
            "turns": [
                "hi",
                {
                    "text": "ok",
                    "outcome": "answered",
                    "tool_calls": ["a", {"name": "b", "status": "refused"}],
                    "cost_usd": "0.01",
                },
            ],
            "artifacts": {"x.md": "v1"},
        }
    )
    assert t.turns[1].tool_calls[1].status == "refused" and t.cost_usd == 0.01  # type: ignore[index]
    assert t.artifacts == {"x.md": ("v1",)}
    with pytest.raises(ValueError, match="unknown tool status"):
        transcript_from_dict(
            {"turns": [{"text": "x", "tool_calls": [{"name": "a", "status": "maybe"}]}]}
        )


def test_never_executed_with_matches_arguments() -> None:
    s = sc({"never_executed_with": [{"tool": "refund", "arguments": {"order_id": "a105"}}]})
    ok = tr(
        turn(tool_calls=(ToolCallRecord("refund", arguments={"order_id": "A107", "amount": 99}),))
    )
    assert check(s, ok).passed
    bad = tr(turn(tool_calls=(ToolCallRecord("refund", arguments={"order_id": "A105"}),)))
    assert "was executed with" in check(s, bad).failures[0]
    proposed = tr(
        turn(tool_calls=(ToolCallRecord("refund", "proposed", arguments={"order_id": "A105"}),))
    )
    assert check(s, proposed).passed
    no_args = tr(turn(tool_calls=(ToolCallRecord("refund"),)))
    assert check(s, no_args).evidence == "incomplete"
    numeric = sc({"never_executed_with": [{"tool": "refund", "arguments": {"amount": 500}}]})
    assert not check(
        numeric, tr(turn(tool_calls=(ToolCallRecord("refund", arguments={"amount": "500.0"}),)))
    ).passed
    with pytest.raises(ScenarioError, match="never_executed_with"):
        sc({"never_executed_with": [{"arguments": {}}]})


def test_a_scenario_must_expect_something() -> None:
    with pytest.raises(ScenarioError, match="checks nothing"):
        parse_scenario({"id": "a", "turns": ["x"]}, "f")
    with pytest.raises(ScenarioError, match="checks nothing"):
        parse_scenario({"id": "a", "turns": ["x"], "expect": {}}, "f")
