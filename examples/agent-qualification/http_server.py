"""Serve the policy-in-code support agent over HTTP, one request per user turn.

    python http_server.py            # http://127.0.0.1:8765/chat
    trustgate qualify -c qualify-http.yaml

Requests need the header ``Authorization: Bearer <AGENT_TOKEN>``.
"""

from __future__ import annotations

import os
import uuid

import httpx
from flask import Flask, jsonify, request

from support_agent import chat_turn, new_conversation
from support_agent_checked import _checked_tool

app = Flask(__name__)
SESSIONS: dict[str, list[dict]] = {}  # type: ignore[type-arg]
CLIENT = httpx.Client(timeout=60)


@app.post("/chat")
def chat():  # type: ignore[no-untyped-def]
    expected = os.environ.get("AGENT_TOKEN", "")
    if not expected or request.headers.get("Authorization") != f"Bearer {expected}":
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(force=True)
    session = str(body.get("session_id") or uuid.uuid4().hex)
    messages = SESSIONS.setdefault(session, new_conversation())
    turn = chat_turn(CLIENT, messages, str(body.get("message", "")), _checked_tool)
    return jsonify({
        "reply": turn["text"],
        "status": turn["outcome"],
        "steps": turn["tool_calls"],
        "usage": {"cost": turn["cost_usd"]},
        "trace_id": f"{session}:{len(messages)}",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8765")), threaded=True)
