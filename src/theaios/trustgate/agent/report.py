"""Report (JSON) and qualification card (Markdown), generated from measured results only."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from theaios.trustgate.agent.qualify import Qualification

SCHEMA = "trustgate.agent-qualification/1"

NON_CLAIMS = [
    "Results hold for these scenarios, this target and this fingerprint only. Any change "
    "listed in the envelope makes them stale.",
    "Bounds treat scenarios as the unit of evidence. They describe scenarios like these, "
    "not production traffic, unless the scenarios were sampled from production.",
    "Passing checks means the agent met the written expectations; it does not show that "
    "the expectations cover every way the agent can fail.",
    "Safety results are not adversarial guarantees: an attacker who adapts to the agent "
    "may find failures these scenarios do not contain.",
]


def build_report(
    q: Qualification, envelope: dict[str, Any], fingerprint: str, meta: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": q.decision,
        "reasons": q.reasons,
        "fingerprint": fingerprint,
        "envelope": envelope,
        "meta": meta,
        "k": q.k,
        "alpha": q.alpha,
        "cost_usd": q.cost_usd,
        "budget_usd": q.budget_usd,
        "stopped_by_budget": q.stopped_by_budget,
        "duration_s": q.duration_s,
        "suites": {name: asdict(s) for name, s in q.suites.items()},
        "scenarios": [asdict(s) for s in q.scenarios],
        "runs": [asdict(r) for r in q.runs],
        "non_claims": NON_CLAIMS,
    }


def save_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} report")
    return data


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_card(report: dict[str, Any]) -> str:
    env = report["envelope"]
    meta = report.get("meta", {})
    lines = [
        f"# Qualification card: {meta.get('name', 'agent')}",
        "",
        f"**Decision: `{report['decision']}`**",
        "",
        *[f"- {r}" for r in report["reasons"]],
        "",
        "## Scope",
        "",
        "| | |",
        "|---|---|",
        f"| Fingerprint | `{report['fingerprint']}` |",
        f"| Model | {env.get('model') or 'not declared'} |",
        f"| Target | {env.get('target', {}).get('type', '?')} |",
        f"| Files in envelope | {len(env.get('files', {}))} |",
        f"| Runs per scenario (K) | {report['k']} |",
        f"| Confidence | {1 - report['alpha']:.0%} |",
        f"| Measured | {report['created']} with TrustGate {env.get('trustgate')} |",
        f"| Cost | ${report['cost_usd']:.4f}"
        + (" (stopped by budget)" if report["stopped_by_budget"] else "")
        + " |",
        "",
        "## Results by suite",
        "",
        "| Suite | Scenarios | pass^K | Lower bound | Run pass rate | Threshold | Decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in report["suites"].items():
        lines.append(
            f"| {name} | {s['scenarios']} | {_pct(s['pass_k'])} | {_pct(s['pass_k_lower'])} | "
            f"{_pct(s['run_pass_rate'])} | {_pct(s['threshold'])} | `{s['decision']}` |"
        )
    safety = report["suites"].get("safety", {})
    if safety.get("failure_upper") is not None:
        lines += [
            "",
            f"Safety: no failure observed. At {1 - report['alpha']:.0%} confidence, "
            f"the share of scenarios like these that would fail at least once in "
            f"{report['k']} runs is under {safety['failure_upper']:.1%}.",
        ]
    failing = [s for s in report["scenarios"] if not s["all_pass"]]
    if failing:
        lines += ["", "## Scenarios that did not pass every run", ""]
        for s in failing[:30]:
            lines.append(f"- `{s['id']}` ({s['suite']}): {s['passes']}/{s['runs']} runs passed")
            lines += [f"  - {f}" for f in s["failures"][:3]]
            failed_run = next(
                (
                    r
                    for r in report["runs"]
                    if r["scenario"] == s["id"] and r["status"] != "passed" and r["calls"]
                ),
                None,
            )
            if failed_run is not None:
                calls = "; ".join(failed_run["calls"][:4])
                lines.append(f"  - tools in run {failed_run['index'] + 1}: {calls}")
    lines += ["", "## What this card does not claim", "", *[f"- {c}" for c in NON_CLAIMS], ""]
    return "\n".join(lines)
