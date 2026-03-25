"""Hoeffding-based sequential stopping to save API costs."""

from __future__ import annotations

import asyncio
import math
from collections import Counter

import httpx

from theaios.trustgate.sampler import Sampler
from theaios.trustgate.types import Question, SampleResponse


def hoeffding_bound(k: int, delta: float) -> float:
    """Compute the Hoeffding confidence half-width.

    epsilon = sqrt(log(2/delta) / (2*k))

    At sample k, if the mode frequency p_hat satisfies
    ``p_hat - epsilon > 0.5``, the mode is statistically dominant
    and we can safely stop sampling.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * k))


def should_stop(
    answers_so_far: list[str],
    k: int,
    delta: float = 0.05,
) -> bool:
    """Check if we can stop sampling early for this question.

    Returns True if the Hoeffding bound confirms the mode is stable:
    p_hat - epsilon > 0.5 where p_hat is the mode frequency.

    Requires at least 2 samples before stopping is considered.
    """
    if k < 2 or len(answers_so_far) < 2:
        return False

    counts = Counter(answers_so_far)
    mode_count = counts.most_common(1)[0][1]
    p_hat = mode_count / len(answers_so_far)
    eps = hoeffding_bound(k, delta)
    return (p_hat - eps) > 0.5


class SequentialSampler:
    """Wraps the base sampler with sequential stopping logic.

    Instead of always sampling K times, samples incrementally and
    stops early when the Hoeffding bound confirms the mode is dominant.
    """

    def __init__(self, sampler: Sampler, delta: float = 0.05) -> None:
        self.sampler = sampler
        self.delta = delta

    async def sample_question(
        self,
        question: Question,
        k_max: int,
        *,
        client: httpx.AsyncClient | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[SampleResponse]:
        """Sample with sequential stopping. Returns <= k_max responses."""
        responses: list[SampleResponse] = []

        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(
                timeout=self.sampler.sampling_config.timeout,
            )
        if semaphore is None:
            semaphore = asyncio.Semaphore(
                self.sampler.sampling_config.max_concurrent,
            )

        assert client is not None
        try:
            # Batch the first min_batch samples in parallel (can't stop before 2)
            min_batch = min(3, k_max)
            first_tasks = [
                self.sampler._sample_one(
                    client=client, question=question, index=i, semaphore=semaphore,
                )
                for i in range(min_batch)
            ]
            responses = list(await asyncio.gather(*first_tasks))

            # Check if we can already stop
            raw_answers = [r.raw_response for r in responses]
            if not should_stop(raw_answers, k=min_batch, delta=self.delta):
                # Continue one at a time with stopping checks
                for i in range(min_batch, k_max):
                    resp = await self.sampler._sample_one(
                        client=client, question=question, index=i,
                        semaphore=semaphore,
                    )
                    responses.append(resp)
                    raw_answers.append(resp.raw_response)
                    if should_stop(raw_answers, k=i + 1, delta=self.delta):
                        break
        finally:
            if own_client:
                await client.aclose()

        return responses

    async def sample_all(
        self,
        questions: list[Question],
        k_max: int,
    ) -> dict[str, list[SampleResponse]]:
        """Sample with sequential stopping for all questions.

        Returns ``{qid: [responses]}``.
        """
        semaphore = asyncio.Semaphore(
            self.sampler.sampling_config.max_concurrent,
        )
        results: dict[str, list[SampleResponse]] = {}

        async with httpx.AsyncClient(
            timeout=self.sampler.sampling_config.timeout,
        ) as client:
            # Run questions concurrently
            tasks = [
                self.sample_question(
                    q, k_max, client=client, semaphore=semaphore,
                )
                for q in questions
            ]
            all_results = await asyncio.gather(*tasks)

        for q, resps in zip(questions, all_results):
            results[q.id] = resps

        return results

    @staticmethod
    def compute_savings(actual_k: dict[str, int], k_max: int) -> dict[str, object]:
        """Report how many API calls were saved.

        Returns a dict with total_possible, total_actual, saved,
        savings_pct, and per_question breakdown.
        """
        n_questions = len(actual_k)
        total_possible = n_questions * k_max
        total_actual = sum(actual_k.values())
        saved = total_possible - total_actual
        savings_pct = saved / total_possible if total_possible > 0 else 0.0

        return {
            "total_possible": total_possible,
            "total_actual": total_actual,
            "saved": saved,
            "savings_pct": savings_pct,
            "per_question": dict(actual_k),
        }
