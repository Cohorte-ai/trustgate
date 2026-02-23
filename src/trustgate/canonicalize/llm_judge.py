"""Binary LLM judge canonicalization."""

from __future__ import annotations

import asyncio
import logging

import httpx

from trustgate.canonicalize import Canonicalizer, register_canonicalizer
from trustgate.sampler import EndpointAdapter, _backoff
from trustgate.types import EndpointConfig

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an evaluation judge. Given a question and an answer, determine if the answer is correct.

Question: {question}
Answer: {answer}

Reply with exactly one word: "correct" or "incorrect"."""


@register_canonicalizer("llm_judge")
class LLMJudgeCanonicalizer(Canonicalizer):
    """Ask a judge LLM to classify an answer as correct or incorrect."""

    def __init__(
        self,
        judge_config: EndpointConfig | None = None,
        retries: int = 3,
        timeout: float = 60.0,
        **kwargs: object,
    ) -> None:
        if judge_config is None:
            raise ValueError(
                "LLMJudgeCanonicalizer requires a judge_config (EndpointConfig)"
            )
        self.judge_adapter = EndpointAdapter.from_config(judge_config)
        self.retries = retries
        self.timeout = timeout

    def canonicalize(self, question: str, answer: str) -> str:
        """Synchronous wrapper — runs the async judge call."""
        text = self.preprocess(answer)
        if not text:
            return "incorrect"
        try:
            return asyncio.run(self.canonicalize_async(question, text))
        except RuntimeError:
            # Already inside an event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.canonicalize_async(question, text))

    async def canonicalize_async(self, question: str, answer: str) -> str:
        """Ask the judge LLM and return 'correct' or 'incorrect'."""
        prompt = _JUDGE_PROMPT.format(question=question, answer=answer)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            last_error: Exception | None = None
            for attempt in range(self.retries + 1):
                try:
                    response = await self.judge_adapter.send(client, prompt, 0.0)
                    return self._parse_judgment(response)
                except Exception as exc:
                    last_error = exc
                    delay = _backoff(attempt)
                    logger.warning(
                        "Judge call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        self.retries,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

            logger.error(
                "Judge exhausted %d retries, defaulting to 'incorrect': %s",
                self.retries,
                last_error,
            )
            return "incorrect"

    @staticmethod
    def _parse_judgment(response: str) -> str:
        """Extract 'correct' or 'incorrect' from the judge response."""
        lower = response.strip().lower()
        if "incorrect" in lower:
            return "incorrect"
        if "correct" in lower:
            return "correct"
        return "incorrect"
