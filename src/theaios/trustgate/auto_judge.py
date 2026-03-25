"""LLM-as-judge automated calibration.

Replaces human review in the calibration step: given a question and its
ranked canonical answers, the judge LLM picks the acceptable one.
This produces the same labels format as human calibration — compatible
with ``trustgate certify --ground-truth``.

This is a CALIBRATION method, not a canonicalizer.  Canonicalization
(grouping semantically equivalent answers) happens first; the judge
decides which canonical group is correct.

Warning: LLM-as-judge has irreducible bias (Proposition 3.4 in the paper).
Human calibration is more rigorous.  Use this for rapid iteration or when
human reviewers are unavailable.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from theaios.trustgate.sampler import EndpointAdapter, _backoff
from theaios.trustgate.types import EndpointConfig

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an evaluation judge. Given a question and a list of candidate answers, \
determine which answer is correct.

Question: {question}

Candidate answers:
{candidates}

Reply with ONLY the number of the correct answer (e.g., "1" or "3"). \
If none of the answers are correct, reply "none"."""


async def auto_judge_labels_async(
    questions_text: dict[str, str],
    profiles: dict[str, list[tuple[str, float]]],
    judge_config: EndpointConfig,
    retries: int = 3,
    timeout: float = 60.0,
) -> dict[str, str]:
    """Use an LLM to automatically label calibration items.

    For each question, presents the ranked canonical answers to the judge
    and asks it to pick the correct one.

    Parameters
    ----------
    questions_text : dict[str, str]
        Mapping of question_id → question text.
    profiles : dict[str, list[tuple[str, float]]]
        Self-consistency profiles from ``sample_and_profile()``.
    judge_config : EndpointConfig
        Endpoint config for the judge LLM.
    retries : int
        Number of retries per judge call.
    timeout : float
        Timeout per request.

    Returns
    -------
    dict[str, str]
        Labels mapping ``{question_id: canonical_answer}``.
        Compatible with ``certify --ground-truth``.
    """
    adapter = EndpointAdapter.from_config(judge_config)
    labels: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        for qid, profile in profiles.items():
            question = questions_text.get(qid, "")
            if not profile or not question:
                continue

            # Build candidates list
            candidates = "\n".join(
                f"{i + 1}. {ans}" for i, (ans, _freq) in enumerate(profile)
            )
            prompt = _JUDGE_PROMPT.format(
                question=question, candidates=candidates,
            )

            # Call judge with retries
            selected = await _call_judge(
                adapter, client, prompt, profile, retries,
            )
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

    # Extract the number
    import re
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
    retries: int = 3,
    timeout: float = 60.0,
) -> dict[str, str]:
    """Synchronous wrapper for :func:`auto_judge_labels_async`."""
    return asyncio.run(
        auto_judge_labels_async(
            questions_text, profiles, judge_config, retries, timeout,
        )
    )
