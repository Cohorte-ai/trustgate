"""Tests for the console (Rich) reporter."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from trustgate.reporting.console import print_certification_result, print_comparison_result
from trustgate.types import CertificationResult


def _sample_result() -> CertificationResult:
    return CertificationResult(
        reliability_level=0.90,
        m_star=1,
        coverage=0.956,
        conditional_coverage=0.980,
        capability_gap=0.024,
        n_cal=100,
        n_test=100,
        k_used=10,
        api_cost_estimate=12.40,
        alpha_coverage={0.05: 0.934, 0.10: 0.956, 0.20: 0.984},
    )


class TestPrintCertificationResult:
    def test_produces_output(self) -> None:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=100)
        print_certification_result(_sample_result(), console=console)
        output = buf.getvalue()
        assert "Reliability Level" in output
        assert "90.0%" in output

    def test_verbose_includes_alpha_table(self) -> None:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=100)
        print_certification_result(_sample_result(), verbose=True, console=console)
        output = buf.getvalue()
        assert "Coverage by Alpha" in output
        assert "0.05" in output
        assert "0.10" in output

    def test_pass_status(self) -> None:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=100)
        print_certification_result(_sample_result(), console=console)
        output = buf.getvalue()
        assert "PASS" in output

    def test_no_cost_if_zero(self) -> None:
        result = _sample_result()
        result.api_cost_estimate = 0.0
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=100)
        print_certification_result(result, console=console)
        output = buf.getvalue()
        assert "Est. API cost" not in output


class TestPrintComparisonResult:
    def test_renders_table(self) -> None:
        results = [
            ("gpt-4.1", _sample_result()),
            ("gpt-4.1-mini", CertificationResult(
                reliability_level=0.85,
                m_star=2,
                coverage=0.923,
                conditional_coverage=0.960,
                capability_gap=0.038,
                n_cal=100,
                n_test=100,
                k_used=10,
                api_cost_estimate=5.00,
            )),
        ]
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        print_comparison_result(results, console=console)
        output = buf.getvalue()
        assert "gpt-4.1" in output
        assert "gpt-4.1-mini" in output
        assert "Model Comparison" in output
