"""Tests for the CSV reporter."""

from __future__ import annotations

import csv
import io

from theaios.trustgate.reporting.csv_export import export_csv
from theaios.trustgate.types import CertificationResult


def _sample_result() -> CertificationResult:
    return CertificationResult(
        reliability_level=0.946,
        m_star=1,
        coverage=0.956,
        conditional_coverage=0.980,
        capability_gap=0.024,
        n_cal=500,
        n_test=500,
        k_used=10,
        api_cost_estimate=12.40,
    )


def _result_with_items() -> CertificationResult:
    r = _sample_result()
    r.per_item = [
        {"question_id": "q1", "score": 1, "correct": True},
        {"question_id": "q2", "score": 2, "correct": False},
    ]
    return r


class TestExportCsv:
    def test_returns_valid_csv(self) -> None:
        text = export_csv(_sample_result())
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) >= 2  # header + data

    def test_correct_headers(self) -> None:
        text = export_csv(_sample_result())
        reader = csv.reader(io.StringIO(text))
        headers = next(reader)
        assert "reliability_level" in headers
        assert "m_star" in headers
        assert "coverage" in headers

    def test_data_row_values(self) -> None:
        text = export_csv(_sample_result())
        reader = csv.reader(io.StringIO(text))
        _headers = next(reader)
        row = next(reader)
        assert "0.9460" in row
        assert "1" in row

    def test_writes_to_file(self, tmp_path: object) -> None:
        import pathlib

        p = pathlib.Path(str(tmp_path)) / "result.csv"
        export_csv(_sample_result(), path=str(p))
        assert p.exists()
        content = p.read_text()
        assert "reliability_level" in content

    def test_per_item_rows(self) -> None:
        text = export_csv(_result_with_items())
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        # summary header, summary data, blank, item header, item1, item2
        assert len(rows) >= 6
        # Find item headers
        item_header_idx = None
        for i, row in enumerate(rows):
            if "question_id" in row:
                item_header_idx = i
                break
        assert item_header_idx is not None

    def test_loadable_by_csv_reader(self) -> None:
        text = export_csv(_sample_result())
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 1
        assert "reliability_level" in rows[0]
