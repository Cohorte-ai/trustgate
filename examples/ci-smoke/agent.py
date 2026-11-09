"""A deterministic agent with no model, so the GitHub Action can be tested without keys."""

from __future__ import annotations


def agent(turns: tuple[str, ...], role: str, run_index: int) -> dict:  # type: ignore[type-arg]
    out = []
    for text in turns:
        if "refund" in text.lower():
            out.append({"text": "A person will review your refund.", "outcome": "escalated",
                        "tool_calls": [{"name": "escalate"}], "cost_usd": 0.0})
        else:
            out.append({"text": "Your order is on its way.", "outcome": "answered",
                        "tool_calls": [{"name": "get_order", "output": "in transit"}],
                        "cost_usd": 0.0})
    return {"turns": out}
