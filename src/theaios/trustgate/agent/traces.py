"""From production conversations to candidate scenarios.

Traces show what users actually ask. Turning them into scenarios is half the work; the other
half, writing what must (never) happen, needs a person, so candidates come out with an
empty ``expect`` and a ``todo`` note, and they do not load until someone completes them.

Accepted input, one JSON object per line:
- a TrustGate transcript line (``turns`` + ``transcript``), as written by ``qualify
  --save-transcripts``;
- an OpenAI-style conversation: ``{"messages": [{"role": "user" | "assistant" | "tool",
  ...}], "id": ...}``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _user_turns(record: dict[str, Any]) -> list[str]:
    if (
        isinstance(record.get("turns"), list)
        and record["turns"]
        and isinstance(record["turns"][0], str)
    ):
        return [str(t) for t in record["turns"]]
    messages = record.get("messages")
    if isinstance(messages, list):
        out = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list):  # content parts: keep text, skip images etc.
                    content = " ".join(
                        str(p.get("text", "")).strip()
                        for p in content
                        if isinstance(p, dict) and p.get("type", "text") == "text"
                    ).strip()
                if content:
                    out.append(str(content))
        return out
    return []


def _tools_used(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for turn in (record.get("transcript") or {}).get("turns", []) or []:
        for c in turn.get("tool_calls") or []:
            names.append(str(c.get("name")))
    for m in record.get("messages") or []:
        if isinstance(m, dict):
            for c in m.get("tool_calls") or []:
                fn = c.get("function", {}) if isinstance(c, dict) else {}
                if fn.get("name"):
                    names.append(str(fn["name"]))
    return sorted(set(names))


def candidates_from_traces(
    path: Path, max_turns: int = 12, stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Distinct conversations (by their user turns) as scenario skeletons. If given, ``stats``
    is filled with counts of lines read, duplicates merged and lines without user turns."""
    counts = {"lines": 0, "duplicates": 0, "no_user_turns": 0}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: not JSON ({exc.msg})") from None
        counts["lines"] += 1
        turns = _user_turns(record)[:max_turns]
        if not turns:
            counts["no_user_turns"] += 1
            continue
        key = hashlib.sha256(json.dumps(turns).encode()).hexdigest()[:10]
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        tools = _tools_used(record)
        source = record.get("id") or record.get("scenario") or f"line {n}"
        out.append(
            {
                "id": f"trace-{key}",
                "suite": "nominal",
                "turns": turns,
                "expect": {},
                "todo": (
                    f"from {source}; tools seen: {', '.join(tools) or 'none'}. Set the suite "
                    "and write what must and must never happen, then delete this line."
                ),
            }
        )
    if stats is not None:
        stats.update(counts)
    return out
