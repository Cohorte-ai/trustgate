"""Tests for the JSON reporter."""

from __future__ import annotations

import json

from theaios.trustgate.reporting.json_export import export_json
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
        alpha_coverage={0.05: 0.934, 0.10: 0.956},
    )


class TestExportJson:
    def test_returns_valid_json(self) -> None:
        text = export_json(_sample_result())
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_contains_all_fields(self) -> None:
        text = export_json(_sample_result())
        data = json.loads(text)
        assert data["reliability_level"] == 0.946
        assert data["m_star"] == 1
        assert data["coverage"] == 0.956
        assert data["conditional_coverage"] == 0.980
        assert data["capability_gap"] == 0.024
        assert data["n_cal"] == 500
        assert data["n_test"] == 500
        assert data["k_used"] == 10
        assert data["api_cost_estimate"] == 12.40

    def test_contains_version(self) -> None:
        text = export_json(_sample_result())
        data = json.loads(text)
        assert "trustgate_version" in data

    def test_contains_timestamp(self) -> None:
        text = export_json(_sample_result())
        data = json.loads(text)
        assert "timestamp" in data

    def test_alpha_coverage_keys_are_strings(self) -> None:
        text = export_json(_sample_result())
        data = json.loads(text)
        assert "0.05" in data["alpha_coverage"]
        assert "0.1" in data["alpha_coverage"]

    def test_writes_to_file(self, tmp_path: object) -> None:
        import pathlib

        p = pathlib.Path(str(tmp_path)) / "result.json"
        export_json(_sample_result(), path=str(p))
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["reliability_level"] == 0.946

    def test_parseable_back(self) -> None:
        text = export_json(_sample_result())
        data = json.loads(text)
        assert isinstance(data["alpha_coverage"], dict)
        assert isinstance(data["per_item"], list)
