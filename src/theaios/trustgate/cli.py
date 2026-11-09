"""Click-based CLI: trustgate certify, compare, calibrate, sample, cache, version."""

from __future__ import annotations

import hashlib
import sys
from typing import TYPE_CHECKING, Any
from pathlib import Path

import click

if TYPE_CHECKING:
    from theaios.trustgate.agent import QualifyConfig, Scenario, Target

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
    type=click.Choice(["numeric", "mcq", "llm_judge", "llm", "embedding", "custom"]),
    help="Canonicalization type (overrides config)",
)
@click.option(
    "--auto-judge",
    is_flag=True,
    help="Use LLM-as-judge for automated calibration (no human needed)",
)
@click.option("--questions", "-q", "questions_path", help="Questions file (CSV or JSON)")
@click.option("--ground-truth", "-g", "ground_truth_path", help="Ground truth labels file")
@click.option(
    "--alpha",
    "-a",
    type=float,
    default=0.05,
    help="Confidence level for M*: 0.05 = 95%, 0.10 = 90%, 0.01 = 99%",
)
@click.option("--k", "k_fixed", type=int, help="Samples per question (overrides config)")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json", "csv"]),
    default="console",
    help="Output format",
)
@click.option("--output-file", help="Write output to file instead of stdout")
@click.option("--no-cache", is_flag=True, help="Disable response cache")
@click.option("--verbose", "-v", is_flag=True, help="Detailed output")
@click.option(
    "--cost-per-request",
    type=float,
    help="Cost per API request in USD (for generic/agent endpoints)",
)
@click.option(
    "--min-reliability", type=float, help="Minimum reliability level (0-100). Exit code 1 if below."
)
@click.option(
    "--concurrency",
    type=int,
    help="Max concurrent API requests (default 30). Lower for rate-limited APIs.",
)
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
    concurrency: int | None,
    yes: bool,
) -> None:
    """Certify an AI endpoint's reliability."""
    try:
        config = _build_config(
            config_path,
            endpoint,
            api_key_env,
            model,
            task_type,
            k_fixed,
            alpha,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if cost_per_request is not None:
        config.endpoint.cost_per_request = cost_per_request

    if concurrency is not None:
        config.sampling.max_concurrent = concurrency

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

    # --- Measure API latency + Pre-flight estimate ---
    measured_latency: float | None = None
    if questions is not None and not yes:
        from rich.console import Console as _PrefConsole
        from rich.status import Status as _PrefStatus

        _pcon = _PrefConsole()
        with _PrefStatus("[bold blue]Measuring API latency...", console=_pcon):
            measured_latency = _measure_latency(config)

        if measured_latency is None:
            click.echo(
                click.style(
                    "WARNING: Could not reach the API endpoint. Check your connection and credentials.",
                    fg="yellow",
                ),
                err=True,
            )

        _show_preflight(config, len(questions), latency=measured_latency)
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
                if new_k > config.sampling.k_max:
                    config.sampling.k_max = new_k
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
            profiles, _ = sample_and_profile(config, questions)
            q_texts = {q.id: q.text for q in questions}
            labels = auto_judge_labels(
                q_texts,
                profiles,
                config.canonicalization.judge_endpoint,
            )
        click.echo(f"Auto-judge labeled {len(labels)} questions.")

    # Check if cache exists before the run
    from pathlib import Path as _Path

    _cache_dir = _Path(".trustgate_cache")
    had_cache = _cache_dir.exists() and any(_cache_dir.glob("*.json"))

    from rich.console import Console
    from rich.status import Status

    console = Console()

    n_q = len(questions) if questions else 0
    if measured_latency and n_q > 0:
        time_est = _estimate_time(config, n_q, latency=measured_latency)
        time_str = _format_time_estimate(time_est)
        spinner_msg = (
            f"[bold blue]Sampling and certifying — est. {time_str}[/bold blue]\n"
            f"  [dim](measured {measured_latency:.1f}s/call, sequential stopping saves ~50% API cost)[/dim]"
        )
    else:
        spinner_msg = (
            "[bold blue]Sampling and certifying...[/bold blue]\n"
            "  [dim](sequential stopping saves ~50% API cost)[/dim]"
        )

    try:
        with Status(spinner_msg, console=console):
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
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        exc_str = str(type(exc).__name__).lower()
        if "connect" in exc_str or "sampler" in exc_str or "timeout" in exc_str:
            click.echo(
                "Hint: if this is a connection or rate limit error, try reducing "
                "--concurrency (default 10) or K.",
                err=True,
            )
        sys.exit(1)

    _output_result(
        result,
        output,
        output_file,
        verbose,
        had_cache=had_cache,
        pass_level=config.thresholds.pass_level,
        weak_level=config.thresholds.weak_level,
    )

    # --- CI/CD gating: exit code 1 if below threshold ---
    if min_reliability is not None:
        threshold = min_reliability / 100 if min_reliability > 1 else min_reliability
        if result.reliability_level < threshold:
            click.echo(
                f"FAIL: reliability {result.reliability_level:.1%} < threshold {threshold:.1%}",
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
    type=click.Choice(["numeric", "mcq", "llm_judge", "llm", "embedding", "custom"]),
)
@click.option("--questions", "-q", "questions_path", required=True)
@click.option("--ground-truth", "-g", "ground_truth_path")
@click.option("--alpha", "-a", type=float, default=0.05)
@click.option(
    "--output",
    "-o",
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
            {
                "model": name,
                "reliability_level": r.reliability_level,
                "m_star": r.m_star,
                "coverage": r.coverage,
                "capability_gap": r.capability_gap,
            }
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
    "--cost-per-request",
    type=float,
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
            profiles, raw_by_canonical = sample_and_profile(config, questions)
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

    # For human calibration (--export or --serve), only show a random
    # subset of questions (n_cal from config). Per the paper (Section 6.3),
    # the calibration set must be randomly sampled and exchangeable with
    # the test set. LLM auto-judge can run on all questions.
    from theaios.trustgate.calibration import random_split

    n_cal = min(config.calibration.n_cal, len(profiles) // 2)
    n_test = min(config.calibration.n_test, len(profiles) - n_cal)
    cal_ids, _test_ids = random_split(list(profiles.keys()), n_cal, n_test)
    cal_set = set(cal_ids)
    cal_questions = [q for q in questions if q.id in cal_set]
    cal_profiles = {qid: profiles[qid] for qid in cal_ids if qid in profiles}
    cal_raw = {qid: raw_by_canonical[qid] for qid in cal_ids if qid in raw_by_canonical}

    click.echo(
        f"Selected {len(cal_questions)} questions for human calibration "
        f"(out of {len(profiles)} total)."
    )

    # --export: generate shareable HTML questionnaire
    if export_path:
        from theaios.trustgate.questionnaire import generate_questionnaire

        try:
            out = generate_questionnaire(
                cal_questions, cal_profiles, cal_raw, output_path=export_path
            )
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
                        {"answer": ans, "frequency": round(freq, 4)} for ans, freq in profiles[q.id]
                    ],
                }
        Path(output.replace(".json", "_profiles.json")).write_text(
            _json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        click.echo(
            f"Profiles saved. Run with --serve to launch the review UI, "
            f"or manually create {output} with labels."
        )
        return

    # Launch review UI
    from theaios.trustgate.serve import serve_calibration

    click.echo(f"\nStarting calibration UI on http://localhost:{port}")
    click.echo(f"Admin panel at http://localhost:{port}/admin")
    click.echo(f"Labels will be saved to {output}")
    click.echo("Send the URL to your domain expert. Press Ctrl+C when done.\n")

    serve_calibration(
        questions=cal_questions,
        profiles=cal_profiles,
        raw_by_canonical=cal_raw,
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

_DEFAULT_LATENCY = 2.0  # fallback if we can't measure


_CANON_CONCURRENCY = 20  # matches LLMSemanticCanonicalizer semaphore


def _measure_latency(
    config: TrustGateConfig,
) -> float | None:
    """Make a single probe call to measure real API latency."""
    import time

    from theaios.trustgate.sampler import EndpointAdapter

    adapter = EndpointAdapter.from_config(config.endpoint)
    prompt = "Reply with one word: hello"

    try:
        import asyncio

        import httpx as _httpx

        start = time.monotonic()

        async def _probe() -> str:
            async with _httpx.AsyncClient(timeout=30.0) as ac:
                return await adapter.send(ac, prompt, config.endpoint.temperature)

        asyncio.run(_probe())
        elapsed = time.monotonic() - start
        return max(elapsed, 0.1)  # floor at 0.1s

    except Exception:
        return None


def _estimate_time(
    config: TrustGateConfig,
    n_questions: int,
    latency: float | None = None,
) -> dict[str, object]:
    """Estimate wall time for the certification pipeline."""
    api_latency = latency or _DEFAULT_LATENCY
    k = config.sampling.k_fixed or config.sampling.k_max
    total_samples = n_questions * k

    # Estimate effective samples (sequential stopping saves ~50%)
    if config.sampling.sequential_stopping:
        effective_samples = int(total_samples * 0.5)
    else:
        effective_samples = total_samples

    # Sampling time
    max_concurrent = config.sampling.max_concurrent
    if config.sampling.sequential_stopping:
        sampling_s = (k * api_latency * n_questions) / max_concurrent
    else:
        sampling_s = (total_samples * api_latency) / max_concurrent

    # Canonicalization time (only for LLM-based, scales with effective samples)
    uses_llm_canon = config.canonicalization.type in ("llm", "llm_judge")
    if uses_llm_canon:
        canon_s = (effective_samples * api_latency) / _CANON_CONCURRENCY
    else:
        canon_s = 0.0

    # Calibration (auto-judge adds another round)
    judge_ep = config.canonicalization.judge_endpoint
    if judge_ep is not None:
        cal_s = (n_questions * api_latency) / _CANON_CONCURRENCY
    else:
        cal_s = 1.0

    total_s = sampling_s + canon_s + cal_s

    return {
        "sampling_s": round(sampling_s),
        "canon_s": round(canon_s),
        "cal_s": round(cal_s),
        "total_s": round(total_s),
        "total_min": round(total_s / 60, 1),
        "uses_llm_canon": uses_llm_canon,
    }


def _format_time_estimate(est: dict[str, object]) -> str:
    """Format time estimate as a human-readable string."""
    total_s = est["total_s"]
    if total_s < 60:  # type: ignore[operator]
        return f"~{total_s}s"
    return f"~{est['total_min']} min"


def _show_preflight(
    config: TrustGateConfig, n_questions: int, latency: float | None = None
) -> None:
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
    table.add_row("Requests", f"{estimate['est_requests']:,}")
    if estimate["sequential_stopping"]:
        table.add_row("Sequential stopping", "enabled (~50% fewer requests)")

    cost_per_req = estimate["cost_per_request"]
    if cost_per_req is not None:
        table.add_row("Cost per request", f"${cost_per_req:.4f}")
        table.add_row("Est. cost", f"${estimate['est_cost']:.2f}")
    else:
        table.add_row(
            "Cost",
            "[dim]unknown (use --cost-per-request or set cost_per_request in config)[/dim]",
        )

    if latency:
        time_est = _estimate_time(config, n_questions, latency=latency)
        time_str = _format_time_estimate(time_est)
        table.add_row("Measured latency", f"{latency:.1f}s per call")
        table.add_row("Est. time", time_str)
    else:
        table.add_row(
            "Est. time",
            "[dim]could not measure API latency[/dim]",
        )

    console.print(table)

    # --- Cost / Reliability tradeoff ---
    arb_table = Table(title="Cost / Reliability Tradeoff")
    arb_table.add_column("K", justify="right")
    arb_table.add_column("Requests", justify="right")
    if cost_per_req is not None:
        arb_table.add_column("Est. Cost", justify="right")
    if latency:
        arb_table.add_column("Est. Time", justify="right")
    arb_table.add_column("Resolution", justify="center")

    arbitrage = estimate_cost_reliability_arbitrage(config, n_questions)
    for row in arbitrage:
        rk: int = row["k"]  # type: ignore[assignment]
        if rk <= 3:
            resolution = "[red]coarse[/red]"
        elif rk <= 7:
            resolution = "[yellow]moderate[/yellow]"
        else:
            resolution = "[green]fine[/green]"

        marker = " ←" if rk == k else ""
        row_cells: list[str] = [
            f"{rk}{marker}",
            f"{row['est_requests']:,}",
        ]
        if cost_per_req is not None:
            row_cells.append(f"${row['est_cost']:.2f}" if row["est_cost"] is not None else "?")
        if latency:
            import copy

            cfg_k = copy.copy(config)
            cfg_k.sampling = copy.copy(config.sampling)
            cfg_k.sampling.k_fixed = rk
            row_time = _estimate_time(cfg_k, n_questions, latency=latency)
            row_cells.append(_format_time_estimate(row_time))
        row_cells.append(resolution)
        arb_table.add_row(*row_cells)

    console.print(arb_table)
    console.print(
        "[dim]Higher K → more samples per question → finer self-consistency signal → "
        "finer reliability estimate.[/dim]"
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
    had_cache: bool = False,
    pass_level: float = 0.80,
    weak_level: float = 0.50,
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
        print_certification_result(
            result,
            verbose=verbose,
            had_cache=had_cache,
            pass_level=pass_level,
            weak_level=weak_level,
        )
        if output_file:
            export_json(result, path=output_file)
            click.echo(f"\nResults also written to {output_file}")


# ---------------------------------------------------------------------------
# Agent qualification: qualify, plan, status
# ---------------------------------------------------------------------------


def _load_agent_setup(
    config_path: str,
) -> tuple[QualifyConfig, list[Scenario], Target, dict[str, Any]]:
    from theaios.trustgate.agent import (
        ScenarioError,
        envelope_components,
        load_qualify_config,
        load_scenarios,
        target_from_config,
    )

    try:
        cfg = load_qualify_config(config_path)
        files = cfg.scenario_files()
        if not files:
            raise ScenarioError(f"no scenario files match {cfg.scenario_globs}")
        scenarios = load_scenarios(list(files))
    except (ScenarioError, OSError) as exc:
        raise click.ClickException(str(exc)) from None
    if str(cfg.base_dir) not in sys.path:
        sys.path.insert(0, str(cfg.base_dir))  # python targets resolve next to the config
    try:
        target = target_from_config(cfg.target)
    except (ValueError, ImportError, AttributeError, KeyError) as exc:
        raise click.ClickException(f"target: {exc}") from None
    used = sorted({r for s in scenarios for r in s.expect.judge})
    if used and cfg.judge is None:
        raise click.ClickException(f"scenarios use judge rubrics {used} but no judge is configured")
    unknown = [r for r in used if cfg.judge is not None and r not in cfg.judge.rubrics]
    if unknown:
        raise click.ClickException(f"unknown judge rubric(s): {unknown}")
    envelope = envelope_components(
        cfg.base_dir,
        model=cfg.model,
        file_globs=cfg.envelope_files,
        extra=cfg.envelope_extra,
        scenario_files=files,
        target=cfg.target,
        judge=_judge_envelope(cfg),
    )
    return cfg, scenarios, target, envelope


def _judge_envelope(cfg: QualifyConfig) -> dict[str, Any] | None:
    """The judge's model, rubrics and current qualification are part of the grader."""
    from theaios.trustgate.agent.judge import card_path

    if cfg.judge is None:
        return None
    cards = {}
    for rubric in sorted(cfg.judge.rubrics):
        p = card_path(cfg.base_dir, rubric)
        cards[rubric] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None
    return {
        "model": cfg.judge.model,
        "rubrics": {r: cfg.judge.rubric_fingerprint(r) for r in sorted(cfg.judge.rubrics)},
        "cards": cards,
    }


@main.command()
@click.option("-c", "--config", "config_path", default="qualify.yaml", show_default=True)
@click.option("-k", type=int, default=None, help="Runs per scenario (overrides the config).")
@click.option("--budget", type=float, default=None, help="Hard spending cap in USD.")
@click.option(
    "--save-transcripts",
    is_flag=True,
    help="Also write every run's full transcript (transcripts.jsonl) for judge gold sets.",
)
def qualify(config_path: str, k: int | None, budget: float | None, save_transcripts: bool) -> None:
    """Qualify an agent: run every scenario K times, decide, write a report and a card."""
    import json
    import threading
    from dataclasses import asdict

    from theaios.trustgate.agent import build_report, fingerprint, render_card, save_report
    from theaios.trustgate.agent import qualify as run_qualification
    from theaios.trustgate.agent.judge import Judge, judge_status

    cfg, scenarios, target, envelope = _load_agent_setup(config_path)
    judge = Judge(cfg.judge) if cfg.judge is not None else None
    judge_meta: dict[str, Any] | None = None
    qualified: dict[str, bool] = {}
    if cfg.judge is not None:
        statuses = {r: judge_status(cfg.judge, r, cfg.base_dir) for r in cfg.judge.rubrics}
        qualified = {r: ok for r, (ok, _) in statuses.items()}
        judge_meta = {
            "model": cfg.judge.model,
            "rubrics": {r: why for r, (_, why) in statuses.items()},
        }
        for r, (_, why) in statuses.items():
            click.echo(f"Judge rubric '{r}': {why}")
    transcripts_path = cfg.report_path.parent / "transcripts.jsonl"
    sink_lock = threading.Lock()
    if save_transcripts:
        transcripts_path.parent.mkdir(parents=True, exist_ok=True)
        transcripts_path.write_text("", encoding="utf-8")

    def sink(scenario: Scenario, index: int, transcript: Any) -> None:
        line = json.dumps(
            {
                "scenario": scenario.id,
                "suite": scenario.suite,
                "run": index,
                "role": scenario.role,
                "turns": list(scenario.turns),
                "transcript": asdict(transcript),
            },
            default=str,
        )
        with sink_lock, open(transcripts_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    k = k or cfg.k
    budget = budget if budget is not None else cfg.budget_usd
    total = len(scenarios) * k
    done = 0

    def progress(_record: object) -> None:
        nonlocal done
        done += 1
        click.echo(f"\r  {done}/{total} runs", nl=False)

    click.echo(f"Qualifying {cfg.name}: {len(scenarios)} scenarios x {k} runs")
    q = run_qualification(
        target,
        scenarios,
        k=k,
        thresholds=cfg.thresholds,
        budget_usd=budget,
        concurrency=cfg.concurrency,
        on_run=progress,
        safety_max_failure_share=cfg.safety_max_failure_share,
        judge=judge,
        judge_qualified=qualified,
        on_transcript=sink if save_transcripts else None,
    )
    click.echo("")
    meta: dict[str, Any] = {"name": cfg.name}
    if judge_meta is not None:
        meta["judge"] = judge_meta
    report = build_report(q, envelope, fingerprint(envelope), meta)
    save_report(report, cfg.report_path)
    cfg.card_path.write_text(render_card(report), encoding="utf-8")
    for name, s in q.suites.items():
        click.echo(
            f"  {name:<11} {s.scenarios:>3} scenarios  pass^{k} {s.pass_k:>4.0%}  "
            f"lower bound {s.pass_k_lower:>4.0%}  -> {s.decision}"
        )
    click.echo(f"Decision: {q.decision}")
    for r in q.reasons:
        click.echo(f"  - {r}")
    click.echo(f"Cost ${q.cost_usd:.4f}" + ("  (stopped by budget)" if q.stopped_by_budget else ""))
    click.echo(f"Report: {cfg.report_path}\nCard:   {cfg.card_path}")
    if save_transcripts:
        click.echo(f"Transcripts: {transcripts_path}")
    if q.decision != "approve_as_scoped":
        sys.exit(1)  # usable as a CI gate: only an approval exits 0


@main.command()
@click.option("-c", "--config", "config_path", default="qualify.yaml", show_default=True)
@click.option("-k", type=int, default=None, help="Runs per scenario (overrides the config).")
@click.option("--budget", type=float, default=None, help="Budget in USD to size against.")
@click.option(
    "--probe",
    type=int,
    default=3,
    show_default=True,
    help="Scenarios to run once to measure cost and time.",
)
def plan(config_path: str, k: int | None, budget: float | None, probe: int) -> None:
    """Measure cost per run on a few scenarios, then size the full qualification."""
    from theaios.trustgate.agent import plan as make_plan

    cfg, scenarios, target, _ = _load_agent_setup(config_path)
    k = k or cfg.k
    budget = budget if budget is not None else cfg.budget_usd
    p = make_plan(
        target,
        scenarios,
        k,
        cfg.thresholds,
        budget,
        probe,
        cfg.concurrency,
        safety_max_failure_share=cfg.safety_max_failure_share,
    )
    click.echo(
        f"Probe: {p.probe_runs} run(s), {p.seconds_per_run:.1f}s per run, "
        + (f"${p.cost_per_run:.5f} per run" if p.cost_per_run is not None else "cost not reported")
    )
    click.echo(
        f"Full run: {p.total_runs} runs (K={k}), about {p.estimated_minutes:.1f} min"
        + (f", about ${p.estimated_cost:.2f}" if p.estimated_cost is not None else "")
    )
    if p.max_k_within_budget is not None:
        click.echo(f"Budget ${budget:.2f} allows K up to {p.max_k_within_budget}")
    for suite, n in p.scenarios_per_suite.items():
        need = p.min_scenarios_to_approve.get(suite)
        click.echo(
            f"  {suite:<11} {n:>3} scenarios" + (f" (approval needs >= {need})" if need else "")
        )
    for w in p.warnings:
        click.echo(f"Warning: {w}")


@main.command()
@click.option("-c", "--config", "config_path", default="qualify.yaml", show_default=True)
def status(config_path: str) -> None:
    """Is the last qualification still valid for the agent as it is now?"""
    from theaios.trustgate.agent import changed_components, fingerprint, load_report

    cfg, _, _, envelope = _load_agent_setup(config_path)
    if not cfg.report_path.exists():
        raise click.ClickException("never qualified: run `trustgate qualify`")
    try:
        report = load_report(cfg.report_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    if report["fingerprint"] == fingerprint(envelope):
        click.echo(f"Current: {report['decision']} (measured {report['created']})")
        return
    click.echo(f"Stale: the agent changed since it was qualified on {report['created']}.")
    for change in changed_components(report["envelope"], envelope):
        click.echo(f"  - {change}")
    sys.exit(1)


@main.command(name="judge-qualify")
@click.option("-c", "--config", "config_path", default="qualify.yaml", show_default=True)
@click.option("--rubric", required=True, help="Rubric id from the judge section of the config.")
@click.option(
    "--gold",
    "gold_path",
    required=True,
    type=click.Path(exists=True),
    help="JSONL gold set labeled by a person (see the docs).",
)
@click.option("--min-kappa", type=float, default=0.8, show_default=True)
@click.option(
    "--min-tnr",
    type=float,
    default=0.85,
    show_default=True,
    help="Minimum share of real failures the judge must catch.",
)
def judge_qualify(
    config_path: str, rubric: str, gold_path: str, min_kappa: float, min_tnr: float
) -> None:
    """Measure a judge against human labels; only a qualified judge can decide runs."""
    from theaios.trustgate.agent import ScenarioError, load_qualify_config
    from theaios.trustgate.agent.judge import (
        Judge,
        card_path,
        load_gold,
        qualify_judge,
        save_card,
    )

    try:
        cfg = load_qualify_config(config_path)
        gold = load_gold(Path(gold_path))
    except (ScenarioError, OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from None
    if cfg.judge is None or rubric not in cfg.judge.rubrics:
        raise click.ClickException(f"rubric '{rubric}' is not in the judge section of the config")
    gold_hash = hashlib.sha256(Path(gold_path).read_bytes()).hexdigest()[:16]
    try:
        card = qualify_judge(
            Judge(cfg.judge), rubric, gold, gold_hash, min_kappa, min_tnr, target_model=cfg.model
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    path = card_path(cfg.base_dir, rubric)
    save_card(card, path)
    m = card.metrics
    click.echo(f"Rubric '{rubric}' with {card.model} ({card.gated_on} split)")
    click.echo(
        f"  scored {m['scored']} (unknown {m['unknown']}), FAILs in gold {m['fails_in_gold']}"
    )
    click.echo(
        f"  kappa {m['kappa']:.2f}   TNR {m['tnr']:.0%} (lower bound {m['tnr_lower']:.0%})"
        f"   TPR {m['tpr']:.0%}"
    )
    click.echo(
        f"  passes {m['false_accept_rate']:.0%} of real failures; "
        f"fails {m['false_reject_rate']:.0%} of real passes"
    )
    for w in card.warnings:
        click.echo(f"Warning: {w}")
    click.echo(f"{'QUALIFIED' if card.qualified else 'NOT QUALIFIED (advisory only)'}")
    for r in card.reasons:
        click.echo(f"  - {r}")
    click.echo(f"{len(card.disagreements)} disagreement(s) listed in {path}")
    if not card.qualified:
        sys.exit(1)


@main.command(name="diff")
@click.argument("old_report", type=click.Path(exists=True))
@click.argument("new_report", type=click.Path(exists=True))
@click.option(
    "--margin",
    type=float,
    required=True,
    help="Largest acceptable share of scenarios broken by the new version, "
    "fixed before looking at results (e.g. 0.05).",
)
def diff_reports(old_report: str, new_report: str, margin: float) -> None:
    """Is the new version at least as good as the old one? Paired by scenario."""
    from theaios.trustgate.agent import load_report
    from theaios.trustgate.agent.compare import compare_reports

    try:
        c = compare_reports(load_report(Path(old_report)), load_report(Path(new_report)), margin)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(
        f"{c.paired_scenarios} paired scenarios: {len(c.regressed)} regressed, "
        f"{len(c.improved)} improved"
    )
    for suite, row in c.by_suite.items():
        click.echo(
            f"  {suite:<11} {row['paired']:>3} paired  -{row['regressed']}  +{row['improved']}"
        )
    if c.cost_per_run_old is not None and c.cost_per_run_new is not None:
        click.echo(f"Cost per run: ${c.cost_per_run_old:.5f} -> ${c.cost_per_run_new:.5f}")
    click.echo(f"Verdict: {c.verdict}")
    for r in c.reasons:
        click.echo(f"  - {r}")
    if c.regressed:
        click.echo(f"Regressed: {', '.join(c.regressed[:20])}")
    if c.verdict not in ("non_inferior", "improvement"):
        sys.exit(1)


@main.command(name="cases-from-traces")
@click.argument("traces", type=click.Path(exists=True))
@click.option("-o", "--output", default="candidates.yaml", show_default=True)
def cases_from_traces(traces: str, output: str) -> None:
    """Turn logged conversations into candidate scenarios for a person to complete."""
    import yaml

    from theaios.trustgate.agent.traces import candidates_from_traces

    try:
        items = candidates_from_traces(Path(traces))
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from None
    header = (
        "# Candidate scenarios from traces. Each needs a suite and expectations written\n"
        "# by a person; remove the 'todo' line when done. Files with 'todo' do not load.\n"
    )
    Path(output).write_text(
        header + yaml.safe_dump(items, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    click.echo(f"{len(items)} distinct conversation(s) written to {output}")
