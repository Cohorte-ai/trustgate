"""Flat CSV export for spreadsheets."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from trustgate.types import CertificationResult

_SUMMARY_HEADERS = [
    "reliability_level",
    "m_star",
    "coverage",
    "conditional_coverage",
    "capability_gap",
    "n_cal",
    "n_test",
    "k_used",
    "api_cost_estimate",
]


def export_csv(result: CertificationResult, path: str | None = None) -> str:
    """Export certification summary as CSV.

    If *path* is provided, also writes to that file.
    Always returns the CSV string.

    The first row contains the summary metrics.
    If ``per_item`` data is present, additional rows are appended
    with per-item diagnostics.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Summary row
    writer.writerow(_SUMMARY_HEADERS)
    writer.writerow([
        f"{result.reliability_level:.4f}",
        result.m_star,
        f"{result.coverage:.4f}",
        f"{result.conditional_coverage:.4f}",
        f"{result.capability_gap:.4f}",
        result.n_cal,
        result.n_test,
        result.k_used,
        f"{result.api_cost_estimate:.4f}",
    ])

    # Per-item rows (if present)
    if result.per_item:
        writer.writerow([])  # blank separator
        item_headers = sorted(result.per_item[0].keys())
        writer.writerow(item_headers)
        for item in result.per_item:
            writer.writerow([item.get(h, "") for h in item_headers])

    csv_str = buf.getvalue()

    if path is not None:
        Path(path).write_text(csv_str, encoding="utf-8")

    return csv_str
