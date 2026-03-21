"""Runtime trust layer: per-query confidence via self-consistency sampling.

Use a pre-computed certification (M*, reliability level) to enrich
every query response with confidence metadata at inference time.

Usage::

    from theaios.trustgate import TrustGate

    gate = TrustGate(config=config, certification=result)
    response = gate.query("What is the treatment for X?")

    response.answer            # top-ranked canonical answer
    response.prediction_set    # top-M* answers (guaranteed to contain truth w.p. ≥ 1-α)
    response.consensus         # consensus strength (top-answer frequency)
    response.margin            # gap between #1 and #2 frequency
    response.profile           # full ranked profile
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from theaios.trustgate.cache import DiskCache
from theaios.trustgate.calibration import compute_profile
from theaios.trustgate.canonicalize import get_canonicalizer
from theaios.trustgate.canonicalize.custom import load_custom_canonicalizer
from theaios.trustgate.config import validate_config
from theaios.trustgate.sampler import Sampler
from theaios.trustgate.sequential import SequentialSampler
from theaios.trustgate.types import (
    CertificationResult,
    Question,
    TrustGateConfig,
)


@dataclass
class GateResponse:
    """Result of a trust-gated query."""

    answer: str
    """Top-ranked canonical answer (mode)."""

    prediction_set: list[str]
    """Top-M* canonical answers. The conformal guarantee says the acceptable
    answer is in this set with probability ≥ 1-α."""

    consensus: float
    """Consensus strength: frequency of the top answer (0–1).
    Higher = more agreement across K samples."""

    margin: float
    """Gap between #1 and #2 answer frequency. Large margin = strong confidence."""

    profile: list[tuple[str, float]]
    """Full ranked profile: [(answer, frequency), ...] sorted by frequency desc."""

    m_star: int
    """Prediction set size from the certification."""

    raw_responses: list[str] = field(default_factory=list)
    """The K raw responses from the endpoint (before canonicalization)."""

    @property
    def is_singleton(self) -> bool:
        """True if the prediction set is a single answer (maximum confidence)."""
        return len(self.prediction_set) == 1

    @property
    def n_samples(self) -> int:
        """Number of samples drawn (may be < K if sequential stopping fired)."""
        return len(self.raw_responses)


class TrustGate:
    """Runtime trust layer that wraps an AI endpoint.

    Combines a pre-computed certification (which determines M*) with
    per-query self-consistency sampling to provide confidence metadata
    on every response.

    Parameters
    ----------
    config : TrustGateConfig
        Endpoint and sampling configuration.
    certification : CertificationResult
        A previously computed certification (from ``trustgate.certify``).
        Provides M* (prediction set size) and the reliability level.
    cache : DiskCache | None
        Response cache. Uses default ``.trustgate_cache`` dir if not provided.
    """

    def __init__(
        self,
        config: TrustGateConfig,
        certification: CertificationResult,
        cache: DiskCache | None = None,
    ) -> None:
        errors = validate_config(config)
        if errors:
            from theaios.trustgate.certification import ConfigError

            raise ConfigError(errors)

        self.config = config
        self.certification = certification
        self.m_star = certification.m_star
        self._cache = cache or DiskCache()
        self._sampler = Sampler(config, cache=self._cache)

        # Build canonicalizer
        if config.canonicalization.type == "custom" and config.canonicalization.custom_class:
            self._canonicalizer = load_custom_canonicalizer(
                config.canonicalization.custom_class,
            )
        else:
            kwargs = {}
            if (
                config.canonicalization.type == "llm_judge"
                and config.canonicalization.judge_endpoint is not None
            ):
                kwargs["judge_config"] = config.canonicalization.judge_endpoint
            self._canonicalizer = get_canonicalizer(
                config.canonicalization.type, **kwargs,
            )

    @property
    def reliability_level(self) -> float:
        """The certified reliability level (1-α*)."""
        return self.certification.reliability_level

    async def query_async(self, question: str, question_id: str = "") -> GateResponse:
        """Query the endpoint with trust-gated confidence.

        Draws K self-consistency samples, canonicalizes, builds the profile,
        and returns the answer with prediction set and confidence metadata.
        """
        qid = question_id or f"_gate_{hash(question)}"
        q = Question(id=qid, text=question)
        k = self._sampler.k

        # Sample (with optional sequential stopping)
        if self.config.sampling.sequential_stopping:
            seq = SequentialSampler(self._sampler, delta=self.config.sampling.delta)
            responses = await seq.sample_question(q, k_max=k)
        else:
            responses = await self._sampler.sample_question(q, k=k)

        raw_texts = [r.raw_response for r in responses]

        # Canonicalize
        canonical = [
            self._canonicalizer.canonicalize(question, r.raw_response)
            for r in responses
        ]

        # Build profile
        profile = compute_profile(canonical)

        # Extract results
        top_answer = profile[0][0]
        consensus = profile[0][1]
        margin = (profile[0][1] - profile[1][1]) if len(profile) > 1 else profile[0][1]
        prediction_set = [ans for ans, _freq in profile[: self.m_star]]

        return GateResponse(
            answer=top_answer,
            prediction_set=prediction_set,
            consensus=consensus,
            margin=margin,
            profile=profile,
            m_star=self.m_star,
            raw_responses=raw_texts,
        )

    def query(self, question: str, question_id: str = "") -> GateResponse:
        """Synchronous wrapper for :meth:`query_async`."""
        return asyncio.run(self.query_async(question, question_id))
