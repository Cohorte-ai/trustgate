"""What an agent did in one run of a scenario.

Fields a target does not report are ``None``, never empty: absence of evidence is not
evidence, and checks that need a missing field mark the run's evidence incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ToolStatus = Literal["executed", "proposed", "refused", "failed"]
TOOL_STATUSES: tuple[str, ...] = ("executed", "proposed", "refused", "failed")


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    status: ToolStatus = "executed"
    arguments: dict[str, object] | None = None
    output: str | None = None


@dataclass(frozen=True)
class TurnObservation:
    """The agent's response to one user turn."""

    text: str | None
    outcome: str | None = None  # e.g. "answered", "escalated", "fallback"
    tool_calls: tuple[ToolCallRecord, ...] | None = None
    citations: tuple[str, ...] | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class Transcript:
    turns: tuple[TurnObservation, ...]
    artifacts: dict[str, tuple[str, ...]] | None = None  # name -> every version
    trace_id: str | None = None
    error: str | None = None  # set when the target failed; the run never passes

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd or 0.0 for t in self.turns)

    @property
    def latency_ms(self) -> float:
        return sum(t.latency_ms or 0.0 for t in self.turns)


def transcript_from_dict(data: dict[str, object]) -> Transcript:
    """Build a Transcript from plain data (what a Python target may return)."""
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("transcript needs a 'turns' list")
    turns = tuple(_turn_from_dict(t) for t in raw_turns)
    artifacts_raw = data.get("artifacts")
    artifacts: dict[str, tuple[str, ...]] | None = None
    if isinstance(artifacts_raw, dict):
        artifacts = {}
        for name, versions in artifacts_raw.items():
            if isinstance(versions, str):
                artifacts[str(name)] = (versions,)
            elif isinstance(versions, list):
                artifacts[str(name)] = tuple(str(v) for v in versions)
    trace = data.get("trace_id")
    error = data.get("error")
    return Transcript(
        turns=turns,
        artifacts=artifacts,
        trace_id=str(trace) if trace is not None else None,
        error=str(error) if error is not None else None,
    )


def _turn_from_dict(raw: object) -> TurnObservation:
    if isinstance(raw, str):
        return TurnObservation(text=raw)
    if not isinstance(raw, dict):
        raise ValueError("each turn must be a string or a mapping")
    calls_raw = raw.get("tool_calls")
    calls: tuple[ToolCallRecord, ...] | None = None
    if isinstance(calls_raw, list):
        calls = tuple(tool_call_from_dict(c) for c in calls_raw)
    cites_raw = raw.get("citations")
    cites = tuple(str(c) for c in cites_raw) if isinstance(cites_raw, list) else None
    text = raw.get("text")
    outcome = raw.get("outcome")
    return TurnObservation(
        text=str(text) if text is not None else None,
        outcome=str(outcome) if outcome is not None else None,
        tool_calls=calls,
        citations=cites,
        cost_usd=_float_or_none(raw.get("cost_usd")),
        latency_ms=_float_or_none(raw.get("latency_ms")),
    )


def tool_call_from_dict(raw: object) -> ToolCallRecord:
    if isinstance(raw, str):
        return ToolCallRecord(name=raw)
    if not isinstance(raw, dict) or "name" not in raw:
        raise ValueError("each tool call needs a 'name'")
    status = str(raw.get("status", "executed"))
    if status not in TOOL_STATUSES:
        raise ValueError(f"unknown tool status '{status}' (expected one of {TOOL_STATUSES})")
    args = raw.get("arguments")
    output = raw.get("output")
    return ToolCallRecord(
        name=str(raw["name"]),
        status=status,  # type: ignore[arg-type]
        arguments=args if isinstance(args, dict) else None,
        output=None if output is None else str(output),
    )


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
