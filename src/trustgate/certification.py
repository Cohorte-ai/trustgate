"""Full pipeline orchestrator: sample -> canonicalize -> calibrate -> certify."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any

from trustgate.cache import DiskCache
from trustgate.calibration import calibrate, compute_profile, random_split
from trustgate.canonicalize import get_canonicalizer
from trustgate.canonicalize.custom import load_custom_canonicalizer
from trustgate.config import load_config, load_questions, validate_config
from trustgate.sampler import Sampler
from trustgate.sequential import SequentialSampler
from trustgate.types import (
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

# Approximate pricing per token (USD). Output tokens only for simplicity.
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


def estimate_cost(
    responses: dict[str, list[SampleResponse]],
    config: TrustGateConfig,
) -> float:
    """Rough cost estimate based on provider pricing and response length."""
    model = config.endpoint.model
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0  # unknown model — can't estimate

    total_output_chars = sum(
        len(r.raw_response)
        for resps in responses.values()
        for r in resps
        if not r.cached
    )
    total_output_tokens = total_output_chars / _CHARS_PER_TOKEN

    # Rough input estimate: each prompt ~200 tokens
    n_calls = sum(1 for resps in responses.values() for r in resps if not r.cached)
    total_input_tokens = n_calls * 200

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
    if config.canonicalization.type == "custom" and config.canonicalization.custom_class:
        canonicalizer = load_custom_canonicalizer(config.canonicalization.custom_class)
    else:
        canonicalizer = get_canonicalizer(
            config.canonicalization.type,
            **_canon_kwargs(config.canonicalization),
        )

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
