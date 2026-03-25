"""LLM-based semantic canonicalization.

Uses a lightweight LLM to extract the core factual answer from a free-text
response, producing a short canonical form that groups semantically
equivalent answers into the same class.

This is a true canonicalizer (Definition 4.1 in the paper): it maps
semantically equivalent answers to the same canonical class without
judging correctness.  Correctness is determined later, during calibration
(by a human or by an LLM-as-judge).

Examples:
    "The capital of France is Paris"      → "paris"
    "I believe it's Paris"                → "paris"
    "Paris is the capital"                → "paris"
    "London"                              → "london"
    "The answer is approximately 42.5"    → "42.5"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

import httpx

from theaios.trustgate.canonicalize import Canonicalizer, register_canonicalizer
from theaios.trustgate.sampler import EndpointAdapter, _backoff
from theaios.trustgate.types import EndpointConfig

logger = logging.getLogger(__name__)

_CANONICALIZE_PROMPT = """\
You are a canonicalization function. Given a question and an answer, extract \
the core factual answer as a short normalized string (1-5 words, lowercase, \
no articles, no punctuation, no explanation).

If the answer contains a specific value, name, or entity, return just that.
If the answer is a number, return just the number.
If the answer is unclear or empty, return "unknown".

Examples:
  Q: "What is the capital of France?" A: "The capital of France is Paris." → paris
  Q: "What is 2+2?" A: "The answer is 4." → 4
  Q: "Who painted the Mona Lisa?" A: "It was painted by Leonardo da Vinci." → da vinci
  Q: "What is the pH of water?" A: "Pure water has a pH of approximately 7." → 7

Question: {question}
Answer: {answer}

Canonical form:"""

# Shared semaphore to limit concurrent LLM calls (avoid overwhelming the API)
_CANON_SEMAPHORE: asyncio.Semaphore | None = None
_MAX_CONCURRENT_CANON = 20


def _get_semaphore() -> asyncio.Semaphore:
    global _CANON_SEMAPHORE  # noqa: PLW0603
    if _CANON_SEMAPHORE is None:
        _CANON_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_CANON)
    return _CANON_SEMAPHORE


# Simple disk cache for canonicalization results
_CANON_CACHE_DIR = Path(".trustgate_cache") / "canon"


def _canon_cache_key(question: str, answer: str) -> str:
    blob = json.dumps({"q": question, "a": answer}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _canon_cache_get(key: str) -> str | None:
    path = _CANON_CACHE_DIR / f"{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _canon_cache_put(key: str, value: str) -> None:
    _CANON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CANON_CACHE_DIR / f"{key}.txt"
    path.write_text(value, encoding="utf-8")


@register_canonicalizer("llm")
class LLMSemanticCanonicalizer(Canonicalizer):
    """Use a lightweight LLM to extract the core answer for semantic grouping.

    Results are cached on disk — re-runs with the same question+answer
    skip the LLM call entirely. Concurrent calls are throttled to avoid
    overwhelming the API.
    """

    def __init__(
        self,
        judge_config: EndpointConfig | None = None,
        retries: int = 3,
        timeout: float = 60.0,
        **kwargs: object,
    ) -> None:
        if judge_config is None:
            raise ValueError(
                "LLMSemanticCanonicalizer requires a judge_config (EndpointConfig) "
                "pointing to the LLM used for semantic extraction."
            )
        self.adapter = EndpointAdapter.from_config(judge_config)
        self.retries = retries
        self.timeout = timeout

    def canonicalize(self, question: str, answer: str) -> str:
        """Synchronous canonicalization — only works outside an event loop."""
        text = self.preprocess(answer)
        if not text:
            return "unknown"
        return asyncio.run(self.canonicalize_async(question, text))

    async def canonicalize_async(self, question: str, answer: str) -> str:
        """Async canonicalization with caching and concurrency throttling."""
        text = self.preprocess(answer)
        if not text:
            return "unknown"

        # Check cache first
        cache_key = _canon_cache_key(question, text)
        cached = _canon_cache_get(cache_key)
        if cached is not None:
            return cached

        prompt = _CANONICALIZE_PROMPT.format(question=question, answer=text)

        sem = _get_semaphore()
        async with sem:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                last_error: Exception | None = None
                for attempt in range(self.retries + 1):
                    try:
                        response = await self.adapter.send(client, prompt, 0.0)
                        result = self._normalize(response)
                        _canon_cache_put(cache_key, result)
                        return result
                    except Exception as exc:
                        last_error = exc
                        delay = _backoff(attempt)
                        logger.warning(
                            "LLM canonicalization failed (attempt %d/%d): %s. "
                            "Retrying in %.1fs",
                            attempt + 1,
                            self.retries,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)

        logger.error(
            "LLM canonicalization exhausted %d retries, "
            "defaulting to 'unknown': %s",
            self.retries,
            last_error,
        )
        return "unknown"

    @staticmethod
    def _normalize(response: str) -> str:
        """Clean the LLM's canonical output."""
        text = response.strip().lower()
        text = text.strip("\"'`.,;:!?")
        for prefix in ("the ", "a ", "an "):
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text.strip() or "unknown"
