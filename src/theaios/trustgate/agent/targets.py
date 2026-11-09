"""Targets: anything that can play a scenario's conversation and report what it did."""

from __future__ import annotations

import importlib
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from theaios.trustgate.agent.transcript import (
    Transcript,
    TurnObservation,
    tool_call_from_dict,
    transcript_from_dict,
)


class Target(Protocol):
    def run(self, turns: tuple[str, ...], role: str, run_index: int) -> Transcript: ...


AgentFunction = Callable[[tuple[str, ...], str, int], object]


@dataclass
class PythonTarget:
    """Calls ``fn(turns, role, run_index)``; it returns a Transcript or plain data."""

    fn: AgentFunction

    @classmethod
    def from_path(cls, path: str) -> PythonTarget:
        module_name, _, attr = path.partition(":")
        if not module_name or not attr:
            raise ValueError("python target must be 'module:function'")
        fn = getattr(importlib.import_module(module_name), attr)
        if not callable(fn):
            raise ValueError(f"{path} is not callable")
        return cls(fn)

    def run(self, turns: tuple[str, ...], role: str, run_index: int) -> Transcript:
        try:
            out = self.fn(turns, role, run_index)
        except Exception as exc:  # the agent crashing is a failed run, not a crash of ours
            return Transcript(turns=(), error=f"{type(exc).__name__}: {exc}"[:300])
        if isinstance(out, Transcript):
            return out
        if isinstance(out, dict):
            try:
                return transcript_from_dict(out)
            except ValueError as exc:
                return Transcript(turns=(), error=f"invalid transcript: {exc}")
        return Transcript(turns=(), error=f"unsupported return type {type(out).__name__}")


def resolve_path(data: object, path: str) -> object:
    """Dotted path with integer indexes: 'result.messages.0.text'. Missing -> None."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.lstrip("-").isdigit():
            idx = int(part)
            current = current[idx] if -len(current) <= idx < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _fill(template: object, values: dict[str, str]) -> object:
    if isinstance(template, str):
        for key, val in values.items():
            template = template.replace("{{" + key + "}}", val)
        return template
    if isinstance(template, dict):
        return {k: _fill(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [_fill(v, values) for v in template]
    return template


@dataclass
class HttpTarget:
    """One POST per user turn; a session id is kept across the turns of a run.

    ``request`` is a JSON template with ``{{message}}``, ``{{session}}``, ``{{role}}`` and
    ``{{turn}}`` placeholders. ``response`` maps transcript fields to dotted paths in the
    JSON reply: text, outcome, tool_calls, citations, cost_usd, trace_id.
    """

    url: str
    request: dict[str, Any]
    response: dict[str, str]
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0

    def run(self, turns: tuple[str, ...], role: str, run_index: int) -> Transcript:
        import httpx

        session = f"tg-{uuid.uuid4().hex[:12]}"
        headers = {k: os.path.expandvars(v) for k, v in self.headers.items()}
        observed: list[TurnObservation] = []
        trace_id: str | None = None
        with httpx.Client(timeout=self.timeout_s) as client:
            for i, message in enumerate(turns, start=1):
                body = _fill(
                    self.request,
                    {"message": message, "session": session, "role": role, "turn": str(i)},
                )
                start = time.perf_counter()
                try:
                    reply = client.post(self.url, json=body, headers=headers)
                    reply.raise_for_status()
                    data = reply.json()
                except Exception as exc:
                    return Transcript(
                        turns=tuple(observed), error=f"turn {i}: {type(exc).__name__}"
                    )
                elapsed = (time.perf_counter() - start) * 1000
                observed.append(self._observe(data, elapsed))
                tid = self._get(data, "trace_id")
                trace_id = str(tid) if tid is not None else trace_id
        return Transcript(turns=tuple(observed), trace_id=trace_id)

    def _get(self, data: object, key: str) -> object:
        path = self.response.get(key)
        return resolve_path(data, path) if path else None

    def _observe(self, data: object, latency_ms: float) -> TurnObservation:
        text = self._get(data, "text")
        outcome = self._get(data, "outcome")
        calls_raw = self._get(data, "tool_calls")
        cites_raw = self._get(data, "citations")
        cost = self._get(data, "cost_usd")
        calls = None
        if "tool_calls" in self.response:
            calls = (
                tuple(tool_call_from_dict(c) for c in calls_raw)
                if isinstance(calls_raw, list)
                else ()
            )
        cites = None
        if "citations" in self.response:
            cites = tuple(str(c) for c in cites_raw) if isinstance(cites_raw, list) else ()
        return TurnObservation(
            text=None if text is None else str(text),
            outcome=None if outcome is None else str(outcome),
            tool_calls=calls,
            citations=cites,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            latency_ms=latency_ms,
        )


def target_from_config(raw: dict[str, Any]) -> Target:
    kind = raw.get("type")
    if kind == "python":
        return PythonTarget.from_path(str(raw["function"]))
    if kind == "http":
        return HttpTarget(
            url=str(raw["url"]),
            request=dict(
                raw.get("request") or {"message": "{{message}}", "session_id": "{{session}}"}
            ),
            response={str(k): str(v) for k, v in (raw.get("response") or {"text": "text"}).items()},
            headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()},
            timeout_s=float(raw.get("timeout_s", 120.0)),
        )
    raise ValueError("target.type must be 'http' or 'python'")
