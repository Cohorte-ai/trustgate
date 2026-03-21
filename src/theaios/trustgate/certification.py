"""Full pipeline orchestrator: sample -> canonicalize -> calibrate -> certify."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any

from theaios.trustgate.cache import DiskCache
from theaios.trustgate.calibration import calibrate, compute_profile, random_split
from theaios.trustgate.canonicalize import get_canonicalizer
from theaios.trustgate.canonicalize.custom import load_custom_canonicalizer
from theaios.trustgate.config import load_config, load_questions, validate_config
from theaios.trustgate.sampler import Sampler
from theaios.trustgate.sequential import SequentialSampler
from theaios.trustgate.types import (
    CanonConfig,
    CertificationResult,
    Question,
    SampleResponse,
    TrustGateConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Invalid configuration: {'; '.join(errors)}")


class LabelsRequired(Exception):
    """Raised when no ground-truth labels are available."""


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_ground_truth(file_path: str) -> dict[str, str]:
    """Load ground truth labels from CSV or JSON.

    CSV format:  id,label  (with header row)
    JSON format: {"q001": "correct", "q002": "incorrect", ...}
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Ground truth JSON must be a mapping of {id: label}")
        return {str(k): str(v) for k, v in data.items()}

    if suffix == ".csv":
        labels: dict[str, str] = {}
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "id" not in reader.fieldnames:
                raise ValueError("Ground truth CSV must have 'id' and 'label' columns")
            for row in reader:
                labels[row["id"]] = row["label"]
        return labels

    raise ValueError(f"Unsupported ground truth format: {suffix} (use .json or .csv)")


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Approximate pricing per token (USD).
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1": {"input": 2.00 / 1_000_000, "output": 8.00 / 1_000_000},
    "gpt-4.1-mini": {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000},
    "gpt-4.1-nano": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-3.5": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
}

# Rough chars-per-token estimate
_CHARS_PER_TOKEN = 4

# Rough per-request cost for known models (200 input + 500 output tokens)
_AVG_INPUT_TOKENS = 200
_AVG_OUTPUT_TOKENS = 500


def _per_request_cost(model: str) -> float | None:
    """Estimate per-request cost for a known model."""
    pricing = _PRICING.get(model)
    if pricing is None:
        return None
    return (
        _AVG_INPUT_TOKENS * pricing["input"]
        + _AVG_OUTPUT_TOKENS * pricing["output"]
    )


def estimate_preflight_cost(
    config: TrustGateConfig,
    n_questions: int,
) -> dict[str, object]:
    """Estimate cost *before* running the pipeline.

    Returns a dict with request counts, per-request cost, and totals.
    If cost cannot be estimated (unknown model, no cost_per_request), the
    cost fields are None.
    """
    k = config.sampling.k_fixed or config.sampling.k_max
    total_requests = n_questions * k

    if config.sampling.sequential_stopping:
        est_requests = int(total_requests * 0.5)
    else:
        est_requests = total_requests

    # Resolve per-request cost
    cost_per_req = config.endpoint.cost_per_request
    if cost_per_req is None:
        cost_per_req = _per_request_cost(config.endpoint.model)

    est_cost = est_requests * cost_per_req if cost_per_req is not None else None
    max_cost = total_requests * cost_per_req if cost_per_req is not None else None

    return {
        "n_questions": n_questions,
        "k": k,
        "total_requests": total_requests,
        "sequential_stopping": config.sampling.sequential_stopping,
        "est_requests": est_requests,
        "cost_per_request": cost_per_req,
        "est_cost": est_cost,
        "max_cost": max_cost,
    }


def estimate_cost_reliability_arbitrage(
    config: TrustGateConfig,
    n_questions: int,
    k_values: list[int] | None = None,
) -> list[dict[str, object]]:
    """Show the cost/reliability tradeoff across different K values.

    Higher K = more samples per question = better self-consistency signal
    = tighter reliability estimates, but higher cost.

    Returns a list of rows, one per K value.
    """
    if k_values is None:
        k_values = [3, 5, 10, 15, 20]

    cost_per_req = config.endpoint.cost_per_request
    if cost_per_req is None:
        cost_per_req = _per_request_cost(config.endpoint.model)

    rows: list[dict[str, object]] = []
    for k in k_values:
        total = n_questions * k
        if config.sampling.sequential_stopping:
            est = int(total * 0.5)
        else:
            est = total

        est_cost = est * cost_per_req if cost_per_req is not None else None
        max_cost = total * cost_per_req if cost_per_req is not None else None

        rows.append({
            "k": k,
            "total_requests": total,
            "est_requests": est,
            "est_cost": est_cost,
            "max_cost": max_cost,
        })
    return rows


def estimate_cost(
    responses: dict[str, list[SampleResponse]],
    config: TrustGateConfig,
) -> float:
    """Actual cost estimate after the pipeline has run."""
    n_calls = sum(1 for resps in responses.values() for r in resps if not r.cached)

    # Use user-provided cost_per_request if available
    if config.endpoint.cost_per_request is not None:
        return float(n_calls * config.endpoint.cost_per_request)

    # Fall back to known model pricing
    model = config.endpoint.model
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0

    total_output_chars = sum(
        len(r.raw_response)
        for resps in responses.values()
        for r in resps
        if not r.cached
    )
    total_output_tokens = total_output_chars / _CHARS_PER_TOKEN
    total_input_tokens = n_calls * _AVG_INPUT_TOKENS

    return (
        total_input_tokens * pricing["input"]
        + total_output_tokens * pricing["output"]
    )


# ---------------------------------------------------------------------------
# Canonicalizer construction helpers
# ---------------------------------------------------------------------------


def _canon_kwargs(canon_config: CanonConfig) -> dict[str, Any]:
    """Build kwargs for ``get_canonicalizer()`` from config."""
    kwargs: dict[str, Any] = {}
    if canon_config.type == "llm_judge" and canon_config.judge_endpoint is not None:
        kwargs["judge_config"] = canon_config.judge_endpoint
    return kwargs


def _build_canonicalizer(config: TrustGateConfig) -> Any:  # noqa: ANN401
    """Build a canonicalizer from config."""
    if config.canonicalization.type == "custom" and config.canonicalization.custom_class:
        return load_custom_canonicalizer(config.canonicalization.custom_class)
    return get_canonicalizer(
        config.canonicalization.type,
        **_canon_kwargs(config.canonicalization),
    )


# ---------------------------------------------------------------------------
# Sample + profile (reusable by calibrate and certify)
# ---------------------------------------------------------------------------


async def sample_and_profile_async(
    config: TrustGateConfig,
    questions: list[Question],
) -> dict[str, list[tuple[str, float]]]:
    """Sample K responses, canonicalize, and return ranked profiles.

    For each question, returns the self-consistency profile: a list of
    ``(canonical_answer, frequency)`` tuples sorted by frequency descending.

    This is the reusable core shared by ``calibrate`` (to show humans the
    ranked answers for labeling) and ``certify`` (to compute nonconformity
    scores).
    """
    errors = validate_config(config)
    if errors:
        raise ConfigError(errors)

    questions_by_id = {q.id: q for q in questions}
    cache = DiskCache()

    # Sample
    sampler = Sampler(config, cache=cache)
    k = sampler.k

    if config.sampling.sequential_stopping:
        seq_sampler = SequentialSampler(sampler, delta=config.sampling.delta)
        responses = await seq_sampler.sample_all(questions, k_max=k)
    else:
        responses = await sampler.sample_all(questions, k=k)

    # Canonicalize
    canonicalizer = _build_canonicalizer(config)
    canonical: dict[str, list[str]] = {}
    for qid, resps in responses.items():
        question_text = questions_by_id[qid].text
        canonical[qid] = [
            canonicalizer.canonicalize(question_text, r.raw_response)
            for r in resps
        ]

    # Build profiles
    profiles: dict[str, list[tuple[str, float]]] = {}
    for qid, answers in canonical.items():
        if answers:
            profiles[qid] = compute_profile(answers)

    return profiles


def sample_and_profile(
    config: TrustGateConfig,
    questions: list[Question],
) -> dict[str, list[tuple[str, float]]]:
    """Synchronous wrapper for :func:`sample_and_profile_async`."""
    return asyncio.run(sample_and_profile_async(config, questions))


def sample_and_rank(
    config: TrustGateConfig,
    questions: list[Question],
) -> dict[str, str]:
    """Sample and return just the top-ranked (mode) answer per question."""
    profiles = sample_and_profile(config, questions)
    return {qid: profile[0][0] for qid, profile in profiles.items() if profile}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def certify_async(
    config: TrustGateConfig | None = None,
    config_path: str = "trustgate.yaml",
    questions: list[Question] | None = None,
    labels: dict[str, str] | None = None,
    ground_truth_file: str | None = None,
) -> CertificationResult:
    """Async full certification pipeline.

    Steps:
    1. Load & validate config
    2. Load questions
    3. Initialize cache
    4. Sample K responses (with optional sequential stopping)
    5. Canonicalize all responses
    6. Compute self-consistency profiles
    7. Split into calibration and test sets
    8. Load or require labels
    9. Run conformal calibration
    10. Enrich result with metadata
    """
    # 1. Config
    if config is None:
        config = load_config(config_path)
    errors = validate_config(config)
    if errors:
        raise ConfigError(errors)

    # 2. Questions
    if questions is None:
        questions = load_questions(config.questions)

    questions_by_id = {q.id: q for q in questions}

    # 3. Cache
    cache = DiskCache()

    # 4. Sample
    sampler = Sampler(config, cache=cache)
    k = sampler.k

    if config.sampling.sequential_stopping:
        seq_sampler = SequentialSampler(sampler, delta=config.sampling.delta)
        responses = await seq_sampler.sample_all(questions, k_max=k)
    else:
        responses = await sampler.sample_all(questions, k=k)

    # 5. Canonicalize
    canonicalizer = _build_canonicalizer(config)

    canonical: dict[str, list[str]] = {}
    for qid, resps in responses.items():
        question_text = questions_by_id[qid].text
        canonical[qid] = [
            canonicalizer.canonicalize(question_text, r.raw_response)
            for r in resps
        ]

    # 6. Profiles
    profiles = {
        qid: compute_profile(answers)
        for qid, answers in canonical.items()
        if answers  # skip empty
    }

    # 7. Split
    all_ids = list(profiles.keys())
    cal_ids, test_ids = random_split(
        all_ids,
        n_cal=min(config.calibration.n_cal, len(all_ids) // 2),
        n_test=min(config.calibration.n_test, len(all_ids) - len(all_ids) // 2),
    )

    # 8. Labels
    if labels is None:
        if ground_truth_file:
            labels = load_ground_truth(ground_truth_file)
        else:
            # Try to derive labels from questions' acceptable_answers
            derived: dict[str, str] = {}
            for q in questions:
                if q.acceptable_answers:
                    derived[q.id] = q.acceptable_answers[0]
            if derived:
                labels = derived
            else:
                raise LabelsRequired(
                    "Provide labels via ground_truth_file, labels dict, or "
                    "questions with acceptable_answers."
                )

    # 9. Calibrate
    result = calibrate(
        profiles=profiles,
        labels=labels,
        cal_ids=cal_ids,
        test_ids=test_ids,
        alpha_values=config.calibration.alpha_values,
    )

    # 10. Enrich
    total_k = sum(len(resps) for resps in responses.values())
    n_questions = len(responses)
    result.k_used = total_k // n_questions if n_questions > 0 else 0
    result.api_cost_estimate = estimate_cost(responses, config)

    return result


def certify(
    config: TrustGateConfig | None = None,
    config_path: str = "trustgate.yaml",
    questions: list[Question] | None = None,
    labels: dict[str, str] | None = None,
    ground_truth_file: str | None = None,
) -> CertificationResult:
    """Synchronous full certification pipeline.

    See :func:`certify_async` for details.
    """
    return asyncio.run(
        certify_async(
            config=config,
            config_path=config_path,
            questions=questions,
            labels=labels,
            ground_truth_file=ground_truth_file,
        )
    )
