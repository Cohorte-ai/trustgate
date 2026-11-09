from __future__ import annotations

import json
from pathlib import Path

import pytest

from theaios.trustgate.agent.judge import (
    Judge,
    JudgeConfig,
    JudgeVerdict,
    card_path,
    judge_metrics,
    judge_status,
    load_gold,
    model_family,
    parse_verdict,
    qualify_judge,
    save_card,
)
from theaios.trustgate.agent.qualify import qualify
from theaios.trustgate.agent.scenario import parse_scenario
from theaios.trustgate.agent.targets import PythonTarget

CFG = JudgeConfig(model="claude-haiku-4-5", rubrics={"polite": "The agent is polite."})


class FakeJudge(Judge):
    """Passes when the agent's last answer contains 'please'; can be forced."""

    def __init__(self, force: str | None = None) -> None:
        super().__init__(CFG)
        self.force = force
        self.calls = 0

    def grade(self, rubric_id, scenario_turns, transcript):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.force:
            return JudgeVerdict(rubric_id, self.force, "forced")  # type: ignore[arg-type]
        text = (transcript.turns[-1].text or "").lower()
        return JudgeVerdict(rubric_id, "pass" if "please" in text else "fail", "rule")


def test_parse_verdict_never_defaults_to_pass() -> None:
    assert parse_verdict('{"critique": "ok", "verdict": "PASS"}') == ("pass", "ok")
    assert parse_verdict('text {"verdict": "fail", "critique": "x"} tail')[0] == "fail"
    assert parse_verdict("PASS")[0] == "unknown"
    assert parse_verdict('{"verdict": "maybe"}')[0] == "unknown"


def test_metrics_reproduce_the_playbook_worked_example() -> None:
    pairs = (
        [("pass", "pass")] * 62
        + [("pass", "fail")] * 8
        + [("fail", "fail")] * 24
        + [("fail", "pass")] * 6
        + [("fail", "unknown")] * 3
    )
    m = judge_metrics(pairs)
    assert m.scored == 100 and m.unknown == 3
    assert m.tpr == pytest.approx(0.886, abs=1e-3)
    assert m.tnr == pytest.approx(0.800, abs=1e-3)
    assert m.kappa == pytest.approx(0.673, abs=1e-3)
    assert m.false_accept_rate == pytest.approx(0.2)


def _gold_line(i: int, label: str, text: str, split: str = "dev") -> str:
    return json.dumps(
        {
            "id": f"g{i}",
            "rubric": "polite",
            "turns": ["hi"],
            "label": label,
            "split": split,
            "transcript": {"turns": [{"text": text}]},
        }
    )


def test_qualify_judge_gates_on_kappa_tnr_and_size(tmp_path: Path) -> None:
    lines = [_gold_line(i, "pass", "Sure, please wait.") for i in range(70)]
    lines += [_gold_line(100 + i, "fail", "No.") for i in range(30)]
    gold_file = tmp_path / "gold.jsonl"
    gold_file.write_text("\n".join(lines))
    gold = load_gold(gold_file)
    good = qualify_judge(FakeJudge(), "polite", gold, "h", target_model="gpt-4.1-mini")
    assert good.qualified and good.metrics["kappa"] == pytest.approx(1.0)
    assert any("no blind split" in w for w in good.warnings)
    always_pass = qualify_judge(FakeJudge(force="pass"), "polite", gold, "h")
    assert not always_pass.qualified
    assert any("TNR 0%" in r for r in always_pass.reasons)
    small = qualify_judge(FakeJudge(), "polite", gold[:10] + gold[-5:], "h")
    assert not small.qualified and "at least 30" in small.reasons[0]


def test_blind_split_is_what_gates(tmp_path: Path) -> None:
    lines = [_gold_line(i, "pass", "please", "dev") for i in range(40)]
    lines += [_gold_line(50 + i, "fail", "no", "dev") for i in range(20)]
    lines += [_gold_line(100 + i, "pass", "no", "blind") for i in range(30)]  # judge wrong
    lines += [_gold_line(200 + i, "fail", "no", "blind") for i in range(10)]
    f = tmp_path / "g.jsonl"
    f.write_text("\n".join(lines))
    card = qualify_judge(FakeJudge(), "polite", load_gold(f), "h")
    assert card.gated_on == "blind" and not card.qualified
    assert card.dev_metrics is not None and card.dev_metrics["kappa"] == pytest.approx(1.0)


def test_same_family_warning() -> None:
    assert model_family("openai/gpt-4.1-mini") == model_family("gpt-4o") == "openai"
    assert model_family("claude-haiku-4-5") == "claude"


def test_judge_status_expires_when_rubric_changes(tmp_path: Path) -> None:
    lines = [_gold_line(i, "pass", "please") for i in range(30)]
    lines += [_gold_line(100 + i, "fail", "no") for i in range(10)]
    f = tmp_path / "g.jsonl"
    f.write_text("\n".join(lines))
    card = qualify_judge(FakeJudge(), "polite", load_gold(f), "h")
    save_card(card, card_path(tmp_path, "polite"))
    assert judge_status(CFG, "polite", tmp_path)[0] is True
    changed = JudgeConfig(model=CFG.model, rubrics={"polite": "The agent is very polite."})
    ok, why = judge_status(changed, "polite", tmp_path)
    assert not ok and "stale" in why
    assert judge_status(CFG, "other", tmp_path)[0] is False


def _scenarios() -> list:  # type: ignore[type-arg]
    return [
        parse_scenario({"id": f"s{i}", "turns": ["hi"], "expect": {"judge": ["polite"]}}, "t")
        for i in range(3)
    ]


def rude(turns, role, run):  # type: ignore[no-untyped-def]
    return {"turns": [{"text": "No.", "tool_calls": []}]}


def test_advisory_judge_never_decides() -> None:
    judge = FakeJudge()
    q = qualify(PythonTarget(rude), _scenarios(), k=2, judge=judge, judge_qualified={})
    assert judge.calls == 6
    assert all(r.status == "passed" for r in q.runs)  # advisory: recorded, not used
    assert all(r.judge[0]["verdict"] == "fail" and not r.judge[0]["decisive"] for r in q.runs)


def test_qualified_judge_fails_runs_and_unknown_is_incomplete() -> None:
    q = qualify(
        PythonTarget(rude), _scenarios(), k=2, judge=FakeJudge(), judge_qualified={"polite": True}
    )
    assert all(r.status == "failed" for r in q.runs)
    q2 = qualify(
        PythonTarget(rude),
        _scenarios(),
        k=1,
        judge=FakeJudge(force="unknown"),
        judge_qualified={"polite": True},
    )
    assert all(r.status == "incomplete" for r in q2.runs)
