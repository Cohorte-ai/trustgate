"""The same support agent, with the refund policy enforced in code.

The model still decides what to do, but `issue_refund` checks eligibility itself and refuses
anything outside the policy: dates and amounts are computed, not guessed.
"""

from __future__ import annotations

import json
from datetime import date

from support_agent import ORDERS, TODAY, _tool, run_conversation

WINDOW_DAYS = 30
SELF_SERVICE_LIMIT = 100.0


def _days_since(order_date: str) -> int:
    return (date.fromisoformat(TODAY) - date.fromisoformat(order_date)).days


def _checked_tool(name: str, args: dict) -> tuple[str, str]:  # type: ignore[type-arg]
    if name == "get_order":
        order = ORDERS.get(str(args.get("order_id", "")).upper())
        if order is None:
            return "No such order.", "executed"
        public = {k: v for k, v in order.items() if k != "customer"}
        public["days_since_order"] = _days_since(order["date"])
        return json.dumps(public), "executed"
    if name == "issue_refund":
        order_id = str(args.get("order_id", "")).upper()
        order = ORDERS.get(order_id)
        try:
            amount = float(args.get("amount", 0))
        except (TypeError, ValueError):
            return "Refused: invalid amount.", "refused"
        if order is None:
            return "Refused: no such order.", "refused"
        if order["status"] != "delivered":
            return f"Refused: order {order_id} is {order['status']}.", "refused"
        if _days_since(order["date"]) > WINDOW_DAYS:
            return (f"Refused: order {order_id} is {_days_since(order['date'])} days old; "
                    f"the window is {WINDOW_DAYS} days."), "refused"
        # The limit applies to the order, not to the refund: a partial refund on a large
        # order still goes to a human.
        if order["amount"] > SELF_SERVICE_LIMIT:
            return (f"Refused: order {order_id} is above {SELF_SERVICE_LIMIT:.0f} EUR. "
                    "Escalate to a human instead."), "refused"
        if amount <= 0 or amount > order["amount"]:
            return f"Refused: the refund must be between 0 and {order['amount']} EUR.", "refused"
    return _tool(name, args), "executed"


def agent(turns: tuple[str, ...], role: str, run_index: int) -> dict:  # type: ignore[type-arg]
    """The model proposes; the refund tool enforces the policy."""
    return run_conversation(turns, _checked_tool)
