"""Click-based CLI: trustgate certify, compare, calibrate, sample, cache, version."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from theaios import trustgate
from theaios.trustgate.cache import DiskCache
from theaios.trustgate.certification import (
    ConfigError,
    LabelsRequired,
    certify,
    estimate_cost_reliability_arbitrage,
    estimate_preflight_cost,
    load_ground_truth,
    sample_and_profile,
)
from theaios.trustgate.config import load_config, load_questions
from theaios.trustgate.reporting import export_csv, export_json, print_certification_result
from theaios.trustgate.types import (
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
    type=click.Choice(["numeric", "mcq", "code_exec", "llm_judge", "llm", "embedding", "custom"]),
    help="Canonicalization type (overrides config)",
)
@click.option("--auto-judge", is_flag=True, help="Use LLM-as-judge for automated calibration (no human needed)")
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
@click.option("--min-reliability", type=float, help="Minimum reliability level (0-100). Exit code 1 if below.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def certify_cmd(
    config_path: str,
    endpoint: str | None,
    api_key_env: str | None,
    model: str | None,
    task_type: str | None,
    auto_judge: bool,
    questions_path: str | None,
    ground_truth_path: str | None,
    alpha: float,
    k_fixed: int | None,
    output: str,
    output_file: str | None,
    no_cache: bool,
    verbose: bool,
    cost_per_request: float | None,
    min_reliability: float | None,
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
        choice = click.prompt(
            "Proceed? Enter Y to run, N to abort, or a number to change K",
            default="Y",
        )
        if choice.upper() == "N":
            sys.exit(0)
        if choice.upper() != "Y":
            try:
                new_k = int(choice)
                config.sampling.k_fixed = new_k
                click.echo(f"K set to {new_k}.")
            except ValueError:
                click.echo("Invalid input. Aborting.", err=True)
                sys.exit(1)

    # --- Auto-judge: generate labels via LLM if no ground truth ---
    if auto_judge and labels is None and questions is not None:
        from theaios.trustgate.auto_judge import auto_judge_labels

        if not config.canonicalization.judge_endpoint:
            click.echo(
                "Error: --auto-judge requires a judge_endpoint in config "
                "(canonicalization.judge_endpoint).",
                err=True,
            )
            sys.exit(1)

        from rich.console import Console as _Console
        from rich.status import Status as _Status

        _con = _Console()
        with _Status("[bold blue]Auto-judging with LLM...", console=_con):
            profiles = sample_and_profile(config, questions)
            q_texts = {q.id: q.text for q in questions}
            labels = auto_judge_labels(
                q_texts, profiles, config.canonicalization.judge_endpoint,
            )
        click.echo(f"Auto-judge labeled {len(labels)} questions.")

    from rich.console import Console
    from rich.status import Status

    console = Console()
    try:
        with Status(
            "[bold blue]Sampling and certifying... "
            "[dim](sequential sampling minimizes API costs — this takes a few minutes)[/dim]",
            console=console,
        ):
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

    # --- CI/CD gating: exit code 1 if below threshold ---
    if min_reliability is not None:
        threshold = min_reliability / 100 if min_reliability > 1 else min_reliability
        if result.reliability_level < threshold:
            click.echo(
                f"FAIL: reliability {result.reliability_level:.1%} "
                f"< threshold {threshold:.1%}",
                err=True,
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# trustgate compare (stub — full implementation in Phase 7)
# ---------------------------------------------------------------------------


@main.command()
@click.option("--models", required=True, help="Comma-separated model names")
@click.option("--config", "-c", "config_path", default="trustgate.yaml")
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(["numeric", "mcq", "code_exec", "llm_judge", "llm", "embedding", "custom"]),
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
    from theaios.trustgate.reporting import print_comparison_result

    model_list = [m.strip() for m in models.split(",")]

    try:
        base_config = load_config(config_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    questions = load_questions(questions_path)
    labels = load_ground_truth(ground_truth_path) if ground_truth_path else None

    from theaios.trustgate.types import CertificationResult

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
@click.option("--questions", "-q", "questions_path", help="Questions file (CSV or JSON)")
@click.option("--serve", is_flag=True, help="Start local calibration web UI")
@click.option("--export", "export_path", help="Export questionnaire as shareable HTML file")
@click.option("--port", type=int, default=8080, help="Port for calibration UI")
@click.option("--output", "-o", default="calibration_labels.json", help="Output labels file")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--cost-per-request", type=float,
    help="Cost per API request in USD (for generic/agent endpoints)",
)
def calibrate(
    config_path: str,
    questions_path: str | None,
    serve: bool,
    export_path: str | None,
    port: int,
    output: str,
    yes: bool,
    cost_per_request: float | None,
) -> None:
    """Sample responses and collect human calibration labels.

    This command:

    \b
    1. Samples K responses per question from your endpoint
    2. Canonicalizes and picks the top answer for each question
    3. Launches a web UI where a human reviewer marks each top answer
       as correct or incorrect
    4. Saves the labels to a JSON file

    Then use: trustgate certify --ground-truth calibration_labels.json
    """
    # Load config
    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if cost_per_request is not None:
        config.endpoint.cost_per_request = cost_per_request

    # Load questions
    questions = None
    if questions_path:
        try:
            questions = load_questions(questions_path)
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"Error loading questions: {exc}", err=True)
            sys.exit(1)
    if questions is None:
        try:
            questions = load_questions(config.questions)
        except Exception as exc:
            click.echo(f"Error loading questions: {exc}", err=True)
            sys.exit(1)

    # Pre-flight estimate
    if not yes:
        _show_preflight(config, len(questions))
        if not click.confirm("Proceed with sampling?", default=True):
            sys.exit(0)

    # Sample and build profiles
    from rich.console import Console
    from rich.status import Status

    console = Console()
    try:
        with Status(
            "[bold blue]Sampling and profiling... "
            "[dim](sequential sampling minimizes API costs — this takes a few minutes)[/dim]",
            console=console,
        ):
            profiles = sample_and_profile(config, questions)
    except ConfigError as exc:
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Profiled {len(profiles)} questions.")

    # Profile quality check
    from theaios.trustgate.calibration import diagnose_profiles

    diag = diagnose_profiles(profiles)
    if diag.warnings:
        for w in diag.warnings:
            click.echo(click.style(f"WARNING: {w}", fg="yellow"), err=True)
        click.echo()

    # --export: generate shareable HTML questionnaire
    if export_path:
        from theaios.trustgate.questionnaire import generate_questionnaire

        try:
            out = generate_questionnaire(questions, profiles, output_path=export_path)
        except OSError as exc:
            click.echo(f"Error writing questionnaire: {exc}", err=True)
            sys.exit(1)
        click.echo(f"Questionnaire exported to {out}")
        click.echo("Share this file with your reviewer (email, Slack, Drive).")
        click.echo("They open it in a browser, review answers, and download labels.json.")
        click.echo("Then run: trustgate certify --ground-truth labels.json")
        return

    if not serve:
        # Without --serve or --export, dump profiles for inspection
        import json as _json

        data = {}
        for q in questions:
            if q.id in profiles:
                data[q.id] = {
                    "question": q.text,
                    "ranked_answers": [
                        {"answer": ans, "frequency": round(freq, 4)}
                        for ans, freq in profiles[q.id]
                    ],
                }
        Path(output.replace(".json", "_profiles.json")).write_text(
            _json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        click.echo(
            f"Profiles saved. Run with --serve to launch the review UI, "
            f"or manually create {output} with labels."
        )
        return

    # Launch review UI
    try:
        from theaios.trustgate.serve import serve_calibration
    except ImportError:
        click.echo(
            "Flask is required for the calibration UI.\n"
            "Install with: pip install 'theaios-trustgate[serve]'",
            err=True,
        )
        sys.exit(1)

    click.echo(f"\nStarting calibration UI on http://localhost:{port}")
    click.echo(f"Admin panel at http://localhost:{port}/admin")
    click.echo(f"Labels will be saved to {output}")
    click.echo("Send the URL to your domain expert. Press Ctrl+C when done.\n")

    serve_calibration(
        questions=questions,
        profiles=profiles,
        port=port,
        output_file=output,
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
            rk: int = row["k"]  # type: ignore[assignment]
            # Higher K = finer resolution of self-consistency → tighter guarantees
            if rk <= 3:
                resolution = "[red]coarse[/red]"
            elif rk <= 7:
                resolution = "[yellow]moderate[/yellow]"
            else:
                resolution = "[green]fine[/green]"

            marker = " ←" if rk == k else ""
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
    from theaios.trustgate.types import CertificationResult

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
