"""Runtime trust layer for AI endpoints.

Two operating modes:

**Passthrough mode** (default, cheap): makes a single API call per query,
attaches pre-computed reliability metadata from the certification.  Use
this in production pipelines where per-query sampling is too expensive.

**Sampled mode** (``mode="sampled"``, expensive): draws K self-consistency
samples per query, builds a per-query profile, and returns the prediction
set with per-query confidence.  Use for high-stakes queries where you need
to know whether *this specific query* is in the reliable region.

Usage::

    from theaios.trustgate import TrustGate, certify

    result = certify(config=config, questions=questions, labels=labels)

    # Passthrough (1 API call per query)
    gate = TrustGate(config=config, certification=result)
    response = gate.query("What is the treatment for X?")
    response.answer              # raw endpoint response
    response.reliability_level   # 0.946 (from certification)

    # Sampled (K API calls per query — per-query confidence)
    gate = TrustGate(config=config, certification=result, mode="sampled")
    response = gate.query("What is the treatment for X?")
    response.prediction_set      # top-M* answers with conformal guarantee
    response.consensus           # per-query consensus strength
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from theaios.trustgate.cache import DiskCache
from theaios.trustgate.calibration import compute_profile
from theaios.trustgate.canonicalize import get_canonicalizer
from theaios.trustgate.canonicalize.custom import load_custom_canonicalizer
from theaios.trustgate.config import validate_config
from theaios.trustgate.sampler import EndpointAdapter, Sampler
from theaios.trustgate.sequential import SequentialSampler
from theaios.trustgate.types import (
    CertificationResult,
    Question,
    TrustGateConfig,
)

import httpx


@dataclass
class GateResponse:
    """Result of a trust-gated query."""

    answer: str
    """The endpoint's answer.  In passthrough mode this is the raw response;
    in sampled mode it is the top-ranked canonical answer (mode)."""

    reliability_level: float
    """Certified reliability level (from the offline certification)."""

    m_star: int
    """Prediction set size from the certification."""

    mode: str
    """``"passthrough"`` or ``"sampled"``."""

    # --- Sampled-mode fields (empty/defaults in passthrough mode) ---

    prediction_set: list[str] = field(default_factory=list)
    """Top-M* canonical answers (sampled mode only)."""

    consensus: float = 0.0
    """Consensus strength: top-answer frequency, 0–1 (sampled mode only)."""

    margin: float = 0.0
    """Gap between #1 and #2 answer frequency (sampled mode only)."""

    profile: list[tuple[str, float]] = field(default_factory=list)
    """Full ranked profile (sampled mode only)."""

    raw_responses: list[str] = field(default_factory=list)
    """K raw responses from the endpoint (sampled mode only)."""

    @property
    def is_singleton(self) -> bool:
        """True if prediction set has exactly one answer (maximum confidence)."""
        return len(self.prediction_set) == 1

    @property
    def n_samples(self) -> int:
        """Number of samples drawn (0 in passthrough mode)."""
        return len(self.raw_responses)


class TrustGate:
    """Runtime trust layer that wraps an AI endpoint.

    Parameters
    ----------
    config : TrustGateConfig
        Endpoint and sampling configuration.
    certification : CertificationResult
        A previously computed certification (from ``trustgate.certify``).
    mode : str
        ``"passthrough"`` (default) — single API call, attaches reliability
        metadata.  ``"sampled"`` — K samples per query, per-query confidence.
    cache : DiskCache | None
        Response cache (used in sampled mode).
    """

    def __init__(
        self,
        config: TrustGateConfig,
        certification: CertificationResult,
        mode: str = "passthrough",
        cache: DiskCache | None = None,
    ) -> None:
        if mode not in ("passthrough", "sampled"):
            raise ValueError(f"mode must be 'passthrough' or 'sampled', got '{mode}'")

        errors = validate_config(config)
        if errors:
            from theaios.trustgate.certification import ConfigError

            raise ConfigError(errors)

        self.config = config
        self.certification = certification
        self.mode = mode
        self.m_star = certification.m_star
        self._cache = cache or DiskCache()

        # Adapter for passthrough mode
        self._adapter = EndpointAdapter.from_config(config.endpoint)

        if mode == "sampled":
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
        """Query the endpoint with trust metadata.

        In passthrough mode: single API call, returns the raw answer enriched
        with pre-computed reliability metadata.

        In sampled mode: draws K samples, canonicalizes, returns prediction
        set with per-query confidence.
        """
        if self.mode == "passthrough":
            return await self._query_passthrough(question)
        return await self._query_sampled(question, question_id)

    def query(self, question: str, question_id: str = "") -> GateResponse:
        """Synchronous wrapper for :meth:`query_async`."""
        return asyncio.run(self.query_async(question, question_id))

    # ------------------------------------------------------------------
    # Passthrough mode
    # ------------------------------------------------------------------

    async def _query_passthrough(self, question: str) -> GateResponse:
        """Single API call — attach reliability metadata from certification."""
        async with httpx.AsyncClient(
            timeout=self.config.sampling.timeout,
        ) as client:
            answer = await self._adapter.send(
                client, question, self.config.endpoint.temperature,
            )

        return GateResponse(
            answer=answer,
            reliability_level=self.reliability_level,
            m_star=self.m_star,
            mode="passthrough",
            raw_responses=[answer],
        )

    # ------------------------------------------------------------------
    # Sampled mode
    # ------------------------------------------------------------------

    async def _query_sampled(self, question: str, question_id: str) -> GateResponse:
        """K samples — per-query confidence via self-consistency."""
        qid = question_id or f"_gate_{hash(question)}"
        q = Question(id=qid, text=question)
        k = self._sampler.k

        if self.config.sampling.sequential_stopping:
            seq = SequentialSampler(self._sampler, delta=self.config.sampling.delta)
            responses = await seq.sample_question(q, k_max=k)
        else:
            responses = await self._sampler.sample_question(q, k=k)

        raw_texts = [r.raw_response for r in responses]

        canonical = [
            await self._canonicalizer.canonicalize_async(question, r.raw_response)
            for r in responses
        ]

        profile = compute_profile(canonical)

        top_answer = profile[0][0]
        consensus = profile[0][1]
        margin = (profile[0][1] - profile[1][1]) if len(profile) > 1 else profile[0][1]
        prediction_set = [ans for ans, _freq in profile[: self.m_star]]

        return GateResponse(
            answer=top_answer,
            reliability_level=self.reliability_level,
            m_star=self.m_star,
            mode="sampled",
            prediction_set=prediction_set,
            consensus=consensus,
            margin=margin,
            profile=profile,
            raw_responses=raw_texts,
        )
