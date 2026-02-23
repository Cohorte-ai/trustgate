"""Multi-model side-by-side comparison."""

from __future__ import annotations

import asyncio
import copy
from itertools import combinations

from trustgate.certification import certify_async
from trustgate.types import CertificationResult, Question, TrustGateConfig


async def compare_async(
    models: list[str],
    config: TrustGateConfig,
    questions: list[Question],
    labels: dict[str, str],
) -> list[tuple[str, CertificationResult]]:
    """Compare multiple models on the same questions.

    Runs models sequentially (different models may have different rate limits)
    but samples within each model run in parallel.

    Returns results sorted by reliability_level descending.
    """
    results: list[tuple[str, CertificationResult]] = []

    for model_name in models:
        model_config = copy.deepcopy(config)
        model_config.endpoint.model = model_name

        result = await certify_async(
            config=model_config,
            questions=questions,
            labels=labels,
        )
        results.append((model_name, result))

    results.sort(key=lambda x: x[1].reliability_level, reverse=True)
    return results


def compare(
    models: list[str],
    config: TrustGateConfig,
    questions: list[Question],
    labels: dict[str, str],
) -> list[tuple[str, CertificationResult]]:
    """Synchronous wrapper for :func:`compare_async`."""
    return asyncio.run(compare_async(models, config, questions, labels))


def compute_comparison_summary(
    results: list[tuple[str, CertificationResult]],
) -> dict[str, object]:
    """Generate a comparison summary with rankings and pairwise deltas."""
    sorted_results = sorted(results, key=lambda x: x[1].reliability_level, reverse=True)

    models_info: list[dict[str, object]] = []
    for rank, (name, r) in enumerate(sorted_results, start=1):
        models_info.append({
            "name": name,
            "reliability_level": r.reliability_level,
            "m_star": r.m_star,
            "coverage": r.coverage,
            "capability_gap": r.capability_gap,
            "rank": rank,
        })

    deltas: dict[tuple[str, str], dict[str, float]] = {}
    for (name_a, res_a), (name_b, res_b) in combinations(sorted_results, 2):
        deltas[(name_a, name_b)] = {
            "reliability_delta": res_a.reliability_level - res_b.reliability_level,
            "coverage_delta": res_a.coverage - res_b.coverage,
        }

    best_name = sorted_results[0][0] if sorted_results else ""
    best_reliability = sorted_results[0][1].reliability_level if sorted_results else 0.0

    return {
        "best_model": best_name,
        "best_reliability": best_reliability,
        "models": models_info,
        "deltas": deltas,
    }
