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
import logging

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


@register_canonicalizer("llm")
class LLMSemanticCanonicalizer(Canonicalizer):
    """Use a lightweight LLM to extract the core answer for semantic grouping.

    This is a true canonicalizer — it groups semantically equivalent answers
    into the same canonical class.  It does NOT judge correctness; that is
    determined during calibration (by a human or by an LLM-as-judge).

    Use a cheap, fast model (e.g., ``gpt-4.1-nano``) — it only needs to
    extract a short string, not reason.
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
        """Async canonicalization — safe to call from within an event loop."""
        text = self.preprocess(answer)
        if not text:
            return "unknown"

        prompt = _CANONICALIZE_PROMPT.format(question=question, answer=text)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            last_error: Exception | None = None
            for attempt in range(self.retries + 1):
                try:
                    response = await self.adapter.send(client, prompt, 0.0)
                    return self._normalize(response)
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
