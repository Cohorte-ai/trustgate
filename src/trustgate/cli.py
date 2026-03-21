"""Click-based CLI: trustgate certify, compare, calibrate, sample, cache, version."""

from __future__ import annotations

import sys

import click

import trustgate
from trustgate.cache import DiskCache
from trustgate.certification import (
    ConfigError,
    LabelsRequired,
    certify,
    estimate_cost_reliability_arbitrage,
    estimate_preflight_cost,
    load_ground_truth,
)
from trustgate.config import load_config, load_questions
from trustgate.reporting import export_csv, export_json, print_certification_result
from trustgate.types import (
    CalibrationConfig,
    CanonConfig,
    EndpointConfig,
    TrustGateConfig,
)


@click.group()
def main() -> None:
    """TrustGate — Black-box AI reliability certification."""


# ---------------------------------------------------------------------------
# trustgate version
# ---------------------------------------------------------------------------


@main.command()
def version() -> None:
    """Show version."""
    click.echo(f"trustgate {trustgate.__version__}")


# ---------------------------------------------------------------------------
# trustgate certify
# ---------------------------------------------------------------------------


@main.command()
@click.option("--config", "-c", "config_path", default="trustgate.yaml", help="Config file path")
@click.option("--endpoint", help="AI endpoint URL (overrides config)")
@click.option("--api-key-env", help="Env var name for API key (overrides config)")
@click.option("--model", help="Model name (overrides config)")
@click.option(
    "--task-type",
    type=click.Choice(["numeric", "mcq", "code_exec", "llm_judge", "embedding", "custom"]),
    help="Canonicalization type (overrides config)",
)
@click.option("--questions", "-q", "questions_path", help="Questions file (CSV or JSON)")
@click.option("--ground-truth", "-g", "ground_truth_path", help="Ground truth labels file")
@click.option("--alpha", "-a", type=float, default=0.10, help="Significance level")
@click.option("--k", "k_fixed", type=int, help="Samples per question (overrides config)")
@click.option(
    "--output", "-o",
    type=click.Choice(["console", "json", "csv"]),
    default="console",
    help="Output format",
)
@click.option("--output-file", help="Write output to file instead of stdout")
@click.option("--no-cache", is_flag=True, help="Disable response cache")
@click.option("--verbose", "-v", is_flag=True, help="Detailed output")
@click.option(
    "--cost-per-request", type=float,
    help="Cost per API request in USD (for generic/agent endpoints)",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def certify_cmd(
    config_path: str,
    endpoint: str | None,
    api_key_env: str | None,
    model: str | None,
    task_type: str | None,
    questions_path: str | None,
    ground_truth_path: str | None,
    alpha: float,
    k_fixed: int | None,
    output: str,
    output_file: str | None,
    no_cache: bool,
    verbose: bool,
    cost_per_request: float | None,
    yes: bool,
) -> None:
    """Certify an AI endpoint's reliability."""
    try:
        config = _build_config(
            config_path, endpoint, api_key_env, model, task_type, k_fixed, alpha,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if cost_per_request is not None:
        config.endpoint.cost_per_request = cost_per_request

    questions = None
    if questions_path:
        try:
            questions = load_questions(questions_path)
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"Error loading questions: {exc}", err=True)
            sys.exit(1)

    # Try loading from config for pre-flight estimate (best-effort)
    if questions is None:
        try:
            questions = load_questions(config.questions)
        except Exception:
            pass  # pipeline will handle loading or raise later

    labels = None
    if ground_truth_path:
        try:
            labels = load_ground_truth(ground_truth_path)
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"Error loading ground truth: {exc}", err=True)
            sys.exit(1)

    # --- Pre-flight cost estimate ---
    if questions is not None and not yes:
        _show_preflight(config, len(questions))
        if not click.confirm("Proceed?", default=True):
            sys.exit(0)

    try:
        result = certify(
            config=config,
            questions=questions,
            labels=labels,
        )
    except ConfigError as exc:
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)
    except LabelsRequired as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    _output_result(result, output, output_file, verbose)


# ---------------------------------------------------------------------------
# trustgate compare (stub — full implementation in Phase 7)
# ---------------------------------------------------------------------------


@main.command()
@click.option("--models", required=True, help="Comma-separated model names")
@click.option("--config", "-c", "config_path", default="trustgate.yaml")
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(["numeric", "mcq", "code_exec", "llm_judge", "embedding", "custom"]),
)
@click.option("--questions", "-q", "questions_path", required=True)
@click.option("--ground-truth", "-g", "ground_truth_path")
@click.option("--alpha", "-a", type=float, default=0.10)
@click.option(
    "--output", "-o",
    type=click.Choice(["console", "json", "csv"]),
    default="console",
)
def compare(
    models: str,
    config_path: str,
    task_type: str,
    questions_path: str,
    ground_truth_path: str | None,
    alpha: float,
    output: str,
) -> None:
    """Compare reliability across multiple models."""
    from trustgate.reporting import print_comparison_result

    model_list = [m.strip() for m in models.split(",")]

    try:
        base_config = load_config(config_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    questions = load_questions(questions_path)
    labels = load_ground_truth(ground_truth_path) if ground_truth_path else None

    from trustgate.types import CertificationResult

    results: list[tuple[str, CertificationResult]] = []
    for model_name in model_list:
        config = TrustGateConfig(
            endpoint=EndpointConfig(
                url=base_config.endpoint.url,
                model=model_name,
                api_key_env=base_config.endpoint.api_key_env,
                provider=base_config.endpoint.provider,
            ),
            sampling=base_config.sampling,
            canonicalization=CanonConfig(type=task_type),
            calibration=CalibrationConfig(
                alpha_values=[alpha],
                n_cal=base_config.calibration.n_cal,
                n_test=base_config.calibration.n_test,
            ),
        )
        result = certify(config=config, questions=questions, labels=labels)
        results.append((model_name, result))

    if output == "console":
        print_comparison_result(results)
    elif output == "json":
        import json

        data = [
            {"model": name, "reliability_level": r.reliability_level, "m_star": r.m_star,
             "coverage": r.coverage, "capability_gap": r.capability_gap}
            for name, r in results
        ]
        click.echo(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# trustgate calibrate (stub — human calibration in Phase 7)
# ---------------------------------------------------------------------------


@main.command()
@click.option("--config", "-c", "config_path", default="trustgate.yaml")
@click.option("--questions", "-q", "questions_path", required=True)
@click.option("--serve", is_flag=True, help="Start local calibration web UI")
@click.option("--port", type=int, default=8080)
@click.option("--output", "-o", default="calibration_labels.json")
def calibrate(
    config_path: str,
    questions_path: str,
    serve: bool,
    port: int,
    output: str,
) -> None:
    """Collect human calibration labels."""
    if serve:
        click.echo(f"Starting calibration UI on http://localhost:{port}")
        click.echo("(Web UI will be available in a future release)")
    else:
        click.echo(
            "Run with --serve to start the calibration web UI, "
            "or provide labels via --ground-truth to the certify command."
        )


# ---------------------------------------------------------------------------
# trustgate sample
# ---------------------------------------------------------------------------


@main.command()
@click.option("--config", "-c", "config_path", default="trustgate.yaml")
@click.option("--questions", "-q", "questions_path", required=True)
@click.option("--k", "k_fixed", type=int, help="Samples per question")
@click.option("--verbose", "-v", is_flag=True)
def sample(
    config_path: str,
    questions_path: str,
    k_fixed: int | None,
    verbose: bool,
) -> None:
    """Sample responses only (no calibration)."""
    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if k_fixed:
        config.sampling.k_fixed = k_fixed

    questions = load_questions(questions_path)
    result = trustgate.sample(config, questions)

    total = sum(len(resps) for resps in result.values())
    cached = sum(1 for resps in result.values() for r in resps if r.cached)
    click.echo(f"Sampled {total} responses for {len(result)} questions")
    click.echo(f"  Cache hits: {cached}")
    click.echo(f"  New API calls: {total - cached}")


# ---------------------------------------------------------------------------
# trustgate cache
# ---------------------------------------------------------------------------


@main.group()
def cache() -> None:
    """Manage the response cache."""


@cache.command("stats")
@click.option("--cache-dir", default=".trustgate_cache", help="Cache directory")
def cache_stats(cache_dir: str) -> None:
    """Show cache statistics."""
    disk_cache = DiskCache(cache_dir)
    stats = disk_cache.stats()
    click.echo(f"Cache directory: {cache_dir}")
    click.echo(f"  Total entries: {stats['total_entries']}")
    click.echo(f"  Total size: {stats['total_size_bytes']:,} bytes")


@cache.command("clear")
@click.option("--cache-dir", default=".trustgate_cache", help="Cache directory")
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def cache_clear(cache_dir: str) -> None:
    """Clear the response cache."""
    disk_cache = DiskCache(cache_dir)
    disk_cache.clear()
    click.echo("Cache cleared.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_preflight(config: TrustGateConfig, n_questions: int) -> None:
    """Show pre-flight cost estimate and cost/reliability arbitrage."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    estimate = estimate_preflight_cost(config, n_questions)

    # --- Summary ---
    table = Table(title="Pre-flight Estimate", show_header=False, padding=(0, 1))
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    k = estimate["k"]
    table.add_row("Questions", str(n_questions))
    table.add_row("Samples per question (K)", str(k))
    table.add_row("Max requests", f"{estimate['total_requests']:,}")
    if estimate["sequential_stopping"]:
        table.add_row("Sequential stopping", "enabled (~50% savings)")
        table.add_row("Est. requests", f"~{estimate['est_requests']:,}")

    cost_per_req = estimate["cost_per_request"]
    if cost_per_req is not None:
        table.add_row("Cost per request", f"${cost_per_req:.4f}")
        table.add_row("Est. cost", f"${estimate['est_cost']:.2f}")
        table.add_row("Max cost", f"${estimate['max_cost']:.2f}")
    else:
        table.add_row(
            "Cost",
            "[dim]unknown (use --cost-per-request or set cost_per_request in config)[/dim]",
        )

    console.print(table)

    # --- Cost / Reliability arbitrage ---
    if cost_per_req is not None:
        arbitrage = estimate_cost_reliability_arbitrage(config, n_questions)
        arb_table = Table(title="Cost / Reliability Tradeoff")
        arb_table.add_column("K", justify="right")
        arb_table.add_column("Requests", justify="right")
        arb_table.add_column("Est. Cost", justify="right")
        arb_table.add_column("Max Cost", justify="right")
        arb_table.add_column("Resolution", justify="center")

        for row in arbitrage:
            rk = row["k"]
            # Higher K = finer resolution of self-consistency → tighter guarantees
            if rk <= 3:  # type: ignore[operator]
                resolution = "[red]coarse[/red]"
            elif rk <= 7:  # type: ignore[operator]
                resolution = "[yellow]moderate[/yellow]"
            else:
                resolution = "[green]fine[/green]"

            marker = " ←" if rk == k else ""  # type: ignore[operator]
            arb_table.add_row(
                f"{rk}{marker}",
                f"~{row['est_requests']:,}" if estimate["sequential_stopping"] else f"{row['total_requests']:,}",
                f"${row['est_cost']:.2f}" if row["est_cost"] is not None else "?",
                f"${row['max_cost']:.2f}" if row["max_cost"] is not None else "?",
                resolution,
            )

        console.print(arb_table)
        console.print(
            "[dim]Higher K → more samples per question → finer self-consistency signal → "
            "tighter reliability guarantee.[/dim]"
        )
    console.print()


def _build_config(
    config_path: str,
    endpoint: str | None,
    api_key_env: str | None,
    model: str | None,
    task_type: str | None,
    k_fixed: int | None,
    alpha: float,
) -> TrustGateConfig:
    """Build a TrustGateConfig from CLI options, falling back to YAML file."""
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        if endpoint is None:
            raise
        # Build a minimal config from CLI args only
        config = TrustGateConfig(
            endpoint=EndpointConfig(url=endpoint),
        )

    # Apply CLI overrides
    if endpoint:
        config.endpoint.url = endpoint
    if api_key_env:
        config.endpoint.api_key_env = api_key_env
    if model:
        config.endpoint.model = model
    if task_type:
        config.canonicalization.type = task_type
    if k_fixed:
        config.sampling.k_fixed = k_fixed
    config.calibration.alpha_values = [alpha]

    return config


def _output_result(
    result: object,
    output: str,
    output_file: str | None,
    verbose: bool,
) -> None:
    """Route the certification result to the correct reporter."""
    from trustgate.types import CertificationResult

    if not isinstance(result, CertificationResult):
        return

    if output == "json":
        text = export_json(result, path=output_file)
        if output_file is None:
            click.echo(text)
        else:
            click.echo(f"Results written to {output_file}")

    elif output == "csv":
        text = export_csv(result, path=output_file)
        if output_file is None:
            click.echo(text)
        else:
            click.echo(f"Results written to {output_file}")

    else:  # console
        print_certification_result(result, verbose=verbose)
        if output_file:
            export_json(result, path=output_file)
            click.echo(f"\nResults also written to {output_file}")
