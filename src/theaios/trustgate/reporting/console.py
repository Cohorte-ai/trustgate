"""Rich terminal output for certification results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from theaios.trustgate.types import CertificationResult


def print_certification_result(
    result: CertificationResult,
    verbose: bool = False,
    console: Console | None = None,
) -> None:
    """Print a formatted certification result to the terminal using Rich."""
    if console is None:
        console = Console()

    # Status
    status = "PASS" if result.coverage >= result.reliability_level else "UNCERTAIN"
    status_style = "bold green" if status == "PASS" else "bold yellow"

    # Main results table
    table = Table(
        title="TrustGate Certification Result",
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Reliability Level", f"{result.reliability_level:.1%}")
    table.add_row("M* (prediction set)", str(result.m_star))
    table.add_row("Empirical Coverage", f"{result.coverage:.3f}")
    table.add_row("Conditional Coverage", f"{result.conditional_coverage:.3f}")
    table.add_row("Capability Gap", f"{result.capability_gap:.1%}")
    table.add_row("Calibration items", str(result.n_cal))
    table.add_row("Test items", str(result.n_test))
    table.add_row("Avg K used", str(result.k_used))
    if result.api_cost_estimate > 0:
        table.add_row("Est. API cost", f"${result.api_cost_estimate:.2f}")
    table.add_row("Status", f"[{status_style}]{status}[/{status_style}]")

    console.print(table)

    # Verbose: per-alpha coverage breakdown
    if verbose and result.alpha_coverage:
        alpha_table = Table(title="Coverage by Alpha")
        alpha_table.add_column("Alpha", justify="right")
        alpha_table.add_column("Target (1-alpha)", justify="right")
        alpha_table.add_column("Empirical Coverage", justify="right")
        alpha_table.add_column("Holds?", justify="center")

        for alpha in sorted(result.alpha_coverage.keys()):
            cov = result.alpha_coverage[alpha]
            target = 1 - alpha
            holds = cov >= target
            mark = "[green]YES[/green]" if holds else "[red]NO[/red]"
            alpha_table.add_row(
                f"{alpha:.2f}",
                f"{target:.2f}",
                f"{cov:.3f}",
                mark,
            )

        console.print()
        console.print(alpha_table)


def print_comparison_result(
    results: list[tuple[str, CertificationResult]],
    console: Console | None = None,
) -> None:
    """Print a side-by-side comparison table for multiple models."""
    if console is None:
        console = Console()

    table = Table(title="Model Comparison")
    table.add_column("Model", style="bold")
    table.add_column("Reliability", justify="right")
    table.add_column("M*", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Cond. Coverage", justify="right")
    table.add_column("Capability Gap", justify="right")
    table.add_column("K Used", justify="right")

    for model_name, result in results:
        table.add_row(
            model_name,
            f"{result.reliability_level:.1%}",
            str(result.m_star),
            f"{result.coverage:.3f}",
            f"{result.conditional_coverage:.3f}",
            f"{result.capability_gap:.1%}",
            str(result.k_used),
        )

    console.print(table)
