"""LLM-as-judge automated calibration.

After sampling and canonicalization produce ranked profiles, the judge
LLM identifies which canonical answer is correct — exactly like a human
reviewer would.  This gives the nonconformity score s_i = rank of the
correct canonical answer, which feeds into conformal calibration for M*.

Two modes:

**With ground truth:** The judge receives the question, the ground truth
label, and the ranked canonical answers.  It identifies which canonical
answer matches the ground truth semantically (not by string matching).

**Without ground truth:** The judge receives just the question and the
ranked canonical answers, and picks the correct one based on its own
knowledge.

Warning: LLM-as-judge has irreducible bias (Proposition 3.4 in the paper).
Human calibration is more rigorous when feasible.
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

from theaios.trustgate.sampler import EndpointAdapter, _backoff
from theaios.trustgate.types import EndpointConfig

logger = logging.getLogger(__name__)

_JUDGE_PROMPT_WITH_GT = """\
You are a calibration judge. Given a question, the known correct answer, \
and a list of candidate answers produced by an AI system, determine which \
candidate matches the correct answer semantically.

Question: {question}
Correct answer: {ground_truth}

Candidate answers (from the AI system):
{candidates}

Which candidate matches the correct answer? Reply with ONLY the number \
(e.g., "1" or "3"). If none of the candidates match, reply "none"."""

_JUDGE_PROMPT_NO_GT = """\
You are a calibration judge. Given a question and a list of candidate \
answers produced by an AI system, determine which candidate is correct.

Question: {question}

Candidate answers:
{candidates}

Which candidate is correct? Reply with ONLY the number (e.g., "1" or "3"). \
If none of the candidates are correct, reply "none"."""


async def auto_judge_labels_async(
    questions_text: dict[str, str],
    profiles: dict[str, list[tuple[str, float]]],
    judge_config: EndpointConfig,
    ground_truth: dict[str, str] | None = None,
    retries: int = 3,
    timeout: float = 60.0,
) -> dict[str, str]:
    """Use an LLM to label calibration items — like a human reviewer.

    All judge calls run in parallel for speed.
    """
    adapter = EndpointAdapter.from_config(judge_config)

    async with httpx.AsyncClient(timeout=timeout) as client:
        coros = []
        task_meta = []
        for qid, profile in profiles.items():
            question = questions_text.get(qid, "")
            if not profile or not question:
                continue

            candidates = "\n".join(
                f"{i + 1}. {ans}" for i, (ans, _freq) in enumerate(profile)
            )

            if ground_truth and qid in ground_truth:
                prompt = _JUDGE_PROMPT_WITH_GT.format(
                    question=question,
                    ground_truth=ground_truth[qid],
                    candidates=candidates,
                )
            else:
                prompt = _JUDGE_PROMPT_NO_GT.format(
                    question=question,
                    candidates=candidates,
                )

            coros.append(_call_judge(adapter, client, prompt, profile, retries))
            task_meta.append((qid, profile))

        # Run all judge calls in parallel
        results = await asyncio.gather(*coros)

    labels: dict[str, str] = {}
    for (qid, _profile), selected in zip(task_meta, results):
        if selected is not None:
            labels[qid] = selected

    return labels


async def _call_judge(
    adapter: EndpointAdapter,
    client: httpx.AsyncClient,
    prompt: str,
    profile: list[tuple[str, float]],
    retries: int,
) -> str | None:
    """Call the judge LLM and parse its response."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await adapter.send(client, prompt, 0.0)
            return _parse_judge_response(response, profile)
        except Exception as exc:
            last_error = exc
            delay = _backoff(attempt)
            logger.warning(
                "Judge call failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt + 1, retries, exc, delay,
            )
            await asyncio.sleep(delay)

    logger.error("Judge exhausted %d retries: %s", retries, last_error)
    return None


def _parse_judge_response(
    response: str,
    profile: list[tuple[str, float]],
) -> str | None:
    """Parse the judge's numeric selection into a canonical answer."""
    text = response.strip().lower()
    if "none" in text:
        return None

    match = re.search(r"\d+", text)
    if not match:
        return None

    idx = int(match.group()) - 1  # 1-indexed → 0-indexed
    if 0 <= idx < len(profile):
        return profile[idx][0]
    return None


def auto_judge_labels(
    questions_text: dict[str, str],
    profiles: dict[str, list[tuple[str, float]]],
    judge_config: EndpointConfig,
    ground_truth: dict[str, str] | None = None,
    retries: int = 3,
    timeout: float = 60.0,
) -> dict[str, str]:
    """Synchronous wrapper for :func:`auto_judge_labels_async`."""
    return asyncio.run(
        auto_judge_labels_async(
            questions_text, profiles, judge_config, ground_truth, retries, timeout,
        )
    )
