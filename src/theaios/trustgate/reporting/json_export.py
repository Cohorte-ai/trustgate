"""Structured JSON results export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from theaios import trustgate
from theaios.trustgate.types import CertificationResult


def export_json(result: CertificationResult, path: str | None = None) -> str:
    """Export certification result as JSON.

    If *path* is provided, also writes to that file.
    Always returns the JSON string.
    """
    data = {
        "trustgate_version": trustgate.__version__,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "reliability_level": result.reliability_level,
        "m_star": result.m_star,
        "target_alpha": result.target_alpha,
        "coverage": result.coverage,
        "conditional_coverage": result.conditional_coverage,
        "capability_gap": result.capability_gap,
        "n_cal": result.n_cal,
        "n_test": result.n_test,
        "k_used": result.k_used,
        "api_cost_estimate": result.api_cost_estimate,
        "alpha_coverage": {
            str(k): v for k, v in result.alpha_coverage.items()
        },
        "per_item": result.per_item,
    }

    json_str = json.dumps(data, indent=2)

    if path is not None:
        Path(path).write_text(json_str, encoding="utf-8")

    return json_str
