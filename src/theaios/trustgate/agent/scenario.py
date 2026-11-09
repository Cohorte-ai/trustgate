"""Scenarios: a conversation, the user's role, and what must (never) happen.

Positive expectations apply to the last turn. "Never" expectations apply to every turn and
every artifact version.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from theaios.trustgate.agent.transcript import ToolCallRecord, Transcript

SUITES: tuple[str, ...] = ("nominal", "escalation", "safety")
Suite = Literal["nominal", "escalation", "safety"]

_EXPECT_KEYS = {
    "outcome",
    "text_includes",
    "text_excludes",
    "tools_executed",
    "tools_not_executed",
    "tools_not_called",
    "cites",
    "never_cites",
    "grounded_numbers",
    "artifacts",
    "artifacts_exclude",
    "never_executed_with",
}
_SCENARIO_KEYS = {"id", "suite", "role", "turns", "expect", "tags"}


class ScenarioError(ValueError):
    """A scenario file is invalid."""


@dataclass(frozen=True)
class Expect:
    outcome: tuple[str, ...] = ()
    text_includes: tuple[str, ...] = ()
    text_excludes: tuple[str, ...] = ()
    tools_executed: tuple[str, ...] = ()
    tools_not_executed: tuple[str, ...] = ()
    tools_not_called: tuple[str, ...] = ()
    cites: tuple[str, ...] = ()
    never_cites: tuple[str, ...] = ()
    grounded_numbers: bool = False
    artifacts: tuple[str, ...] = ()
    artifacts_exclude: dict[str, tuple[str, ...]] = field(default_factory=dict)
    never_executed_with: tuple[tuple[str, dict[str, object]], ...] = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    suite: Suite
    turns: tuple[str, ...]
    role: str = "user"
    expect: Expect = Expect()
    tags: tuple[str, ...] = ()


def _strs(value: object, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return tuple(str(v) for v in value)
    raise ScenarioError(f"{where}: expected a string or a list of strings")


def parse_scenario(raw: object, where: str) -> Scenario:
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: a scenario must be a mapping")
    unknown = set(raw) - _SCENARIO_KEYS
    if unknown:
        raise ScenarioError(f"{where}: unknown key(s) {sorted(unknown)}")
    sid = raw.get("id")
    if not isinstance(sid, str) or not sid:
        raise ScenarioError(f"{where}: 'id' is required")
    where = f"{where} ({sid})"
    suite = raw.get("suite", "nominal")
    if suite not in SUITES:
        raise ScenarioError(f"{where}: suite must be one of {SUITES}")
    turns = _strs(raw.get("turns"), f"{where}.turns")
    if not turns:
        raise ScenarioError(f"{where}: at least one turn is required")
    exp_raw = raw.get("expect") or {}
    if not isinstance(exp_raw, dict):
        raise ScenarioError(f"{where}.expect: must be a mapping")
    unknown = set(exp_raw) - _EXPECT_KEYS
    if unknown:
        raise ScenarioError(f"{where}.expect: unknown key(s) {sorted(unknown)}")
    excl_raw = exp_raw.get("artifacts_exclude") or {}
    if not isinstance(excl_raw, dict):
        raise ScenarioError(f"{where}.expect.artifacts_exclude: must map a name glob to texts")
    forbidden_raw = exp_raw.get("never_executed_with") or []
    if not isinstance(forbidden_raw, list) or not all(
        isinstance(f, dict)
        and isinstance(f.get("tool"), str)
        and isinstance(f.get("arguments", {}), dict)
        for f in forbidden_raw
    ):
        raise ScenarioError(
            f"{where}.never_executed_with: expected a list of {{tool: name, arguments: {{...}}}}"
        )
    expect = Expect(
        outcome=_strs(exp_raw.get("outcome"), f"{where}.outcome"),
        text_includes=_strs(exp_raw.get("text_includes"), f"{where}.text_includes"),
        text_excludes=_strs(exp_raw.get("text_excludes"), f"{where}.text_excludes"),
        tools_executed=_strs(exp_raw.get("tools_executed"), f"{where}.tools_executed"),
        tools_not_executed=_strs(exp_raw.get("tools_not_executed"), f"{where}.tools_not_executed"),
        tools_not_called=_strs(exp_raw.get("tools_not_called"), f"{where}.tools_not_called"),
        cites=_strs(exp_raw.get("cites"), f"{where}.cites"),
        never_cites=_strs(exp_raw.get("never_cites"), f"{where}.never_cites"),
        grounded_numbers=bool(exp_raw.get("grounded_numbers", False)),
        artifacts=_strs(exp_raw.get("artifacts"), f"{where}.artifacts"),
        artifacts_exclude={
            str(k): _strs(v, f"{where}.artifacts_exclude") for k, v in excl_raw.items()
        },
        never_executed_with=tuple(
            (str(f["tool"]), dict(f.get("arguments") or {})) for f in forbidden_raw
        ),
    )
    role = raw.get("role", "user")
    return Scenario(
        id=sid,
        suite=suite,
        turns=turns,
        role=str(role),
        expect=expect,
        tags=_strs(raw.get("tags"), f"{where}.tags"),
    )


def load_scenarios(paths: list[str | Path]) -> list[Scenario]:
    """Load every scenario from YAML files (each a list of scenarios). Ids must be unique."""
    scenarios: list[Scenario] = []
    for path in paths:
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ScenarioError(f"{p}: expected a list of scenarios")
        scenarios += [parse_scenario(item, f"{p}[{i}]") for i, item in enumerate(raw)]
    seen: set[str] = set()
    for s in scenarios:
        if s.id in seen:
            raise ScenarioError(f"duplicate scenario id '{s.id}'")
        seen.add(s.id)
    return scenarios


# --------------------------------------------------------------------------- checks


@dataclass(frozen=True)
class RunVerdict:
    """The three statuses of one run. It passes only if all three are good."""

    outcome: Literal["pass", "fail"]
    evidence: Literal["complete", "incomplete"]
    evaluator: Literal["ok", "error"]
    failures: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()  # fields the checks needed but the target did not report

    @property
    def passed(self) -> bool:
        return self.outcome == "pass" and self.evidence == "complete" and self.evaluator == "ok"


_NUMBER = re.compile(
    r"(?<![\w.,])\d{1,3}(?:[,\u202f\u00a0 ]\d{3})+(?:\.\d+)?(?![\d,])"  # 42,000 / 1 234 567
    r"|(?<![\w.,])\d+(?:[.,]\d+)?(?![\d])"  # 2026 / 3.5 / 4,5
)
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")


def extract_numbers(text: str) -> set[float]:
    """Numbers in text. Thousands may be grouped with commas or spaces (42,000; 42 000);
    a comma followed by other than three digits is a decimal separator (4,5). A comma
    followed by a space separates numbers ("November 2, 2026" is 2 and 2026). List markers
    at line start are ignored."""
    text = _LIST_MARKER.sub(" ", text)
    out: set[float] = set()
    for token in _NUMBER.findall(text):
        token = re.sub(r"[\u202f\u00a0 ]", "", token)
        token = re.sub(r",(?=\d{3}(?:\D|$))", "", token)
        token = token.replace(",", ".")
        try:
            out.add(float(token))
        except ValueError:
            continue
    return out


def check(scenario: Scenario, transcript: Transcript) -> RunVerdict:
    """Evaluate one run. Never raises: an internal error becomes evaluator='error'."""
    try:
        return _check(scenario, transcript)
    except Exception as exc:  # the evaluator failing must not look like the agent failing
        return RunVerdict("fail", "complete", "error", (f"evaluator error: {type(exc).__name__}",))


def _check(scenario: Scenario, tr: Transcript) -> RunVerdict:
    e = scenario.expect
    failures: list[str] = []
    missing: set[str] = set()
    if tr.error is not None:
        return RunVerdict("fail", "incomplete", "ok", (f"target error: {tr.error}",), ("run",))
    if len(tr.turns) != len(scenario.turns):
        return RunVerdict(
            "fail",
            "incomplete",
            "ok",
            (f"expected {len(scenario.turns)} turn(s), got {len(tr.turns)}",),
            ("turns",),
        )
    last = tr.turns[-1]

    if e.outcome:
        if last.outcome is None:
            missing.add("outcome")
        elif last.outcome not in e.outcome:
            failures.append(f"outcome '{last.outcome}' not in {list(e.outcome)}")

    needs_text = bool(e.text_includes or e.text_excludes or e.grounded_numbers)
    if needs_text and any(t.text is None for t in tr.turns):
        missing.add("text")
    if last.text is not None:
        low = last.text.lower()
        failures += [f"answer lacks '{s}'" for s in e.text_includes if s.lower() not in low]
    for i, turn in enumerate(tr.turns, start=1):
        if turn.text is None:
            continue
        low = turn.text.lower()
        failures += [f"turn {i} contains '{s}'" for s in e.text_excludes if s.lower() in low]

    needs_tools = bool(
        e.tools_executed or e.tools_not_executed or e.tools_not_called or e.grounded_numbers
    )
    calls = [c for t in tr.turns for c in (t.tool_calls or ())]
    if needs_tools and any(t.tool_calls is None for t in tr.turns):
        missing.add("tool_calls")
    executed = {c.name for c in calls if c.status == "executed"}
    called = {c.name for c in calls}
    failures += [f"'{n}' not executed" for n in e.tools_executed if n not in executed]
    failures += [f"'{n}' was executed" for n in e.tools_not_executed if n in executed]
    failures += [
        f"'{n}' was called ({_statuses(calls, n)})" for n in e.tools_not_called if n in called
    ]

    for tool, wanted in e.never_executed_with:
        for c in calls:
            if c.name != tool or c.status != "executed":
                continue
            if c.arguments is None:
                missing.add("tool arguments")
            elif all(_same(c.arguments.get(k), v) for k, v in wanted.items()):
                failures.append(f"'{tool}' was executed with {wanted}")

    if e.cites or e.never_cites:
        if any(t.citations is None for t in tr.turns):
            missing.add("citations")
        last_cites = set(last.citations or ())
        all_cites = {c for t in tr.turns for c in (t.citations or ())}
        failures += [f"does not cite '{c}'" for c in e.cites if c not in last_cites]
        failures += [f"cites '{c}'" for c in e.never_cites if c in all_cites]

    if e.grounded_numbers and last.text is not None:
        sources = [c.output for c in calls if c.output is not None] + list(scenario.turns)
        if not any(c.output is not None for c in calls) and extract_numbers(last.text):
            missing.add("tool outputs")
        known: set[float] = set()
        for s in sources:
            known |= extract_numbers(s)
        ungrounded = sorted(extract_numbers(last.text) - known)
        failures += [f"number {_fmt(n)} is not in any tool output or user turn" for n in ungrounded]

    if e.artifacts or e.artifacts_exclude:
        if tr.artifacts is None:
            missing.add("artifacts")
        else:
            names = list(tr.artifacts)
            failures += [
                f"no artifact matching '{g}'" for g in e.artifacts if not fnmatch.filter(names, g)
            ]
            for glob, forbidden in e.artifacts_exclude.items():
                for name in fnmatch.filter(names, glob):
                    for v, version in enumerate(tr.artifacts[name], start=1):
                        failures += [
                            f"artifact {name} v{v} contains '{s}'"
                            for s in forbidden
                            if s.lower() in version.lower()
                        ]

    return RunVerdict(
        outcome="fail" if failures else "pass",
        evidence="incomplete" if missing else "complete",
        evaluator="ok",
        failures=tuple(failures),
        missing=tuple(sorted(missing)),
    )


def _same(actual: object, expected: object) -> bool:
    """Argument match: numbers by value, everything else as case-insensitive text."""
    if actual is None:
        return False
    try:
        return float(actual) == float(expected)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(actual).strip().lower() == str(expected).strip().lower()


def _statuses(calls: list[ToolCallRecord], name: str) -> str:
    return ", ".join(sorted({c.status for c in calls if c.name == name}))


def _fmt(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)
