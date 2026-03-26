"""Query AI endpoints and collect K responses."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from theaios.trustgate.cache import DiskCache
from theaios.trustgate.config import resolve_api_key
from theaios.trustgate.types import (
    EndpointConfig,
    Question,
    SampleResponse,
    TrustGateConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template helpers (for generic endpoints)
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _substitute_template(template: Any, question: str) -> Any:
    """Recursively replace ``{{question}}`` in template values."""
    if isinstance(template, str):
        return template.replace("{{question}}", question)
    if isinstance(template, dict):
        return {k: _substitute_template(v, question) for k, v in template.items()}
    if isinstance(template, list):
        return [_substitute_template(item, question) for item in template]
    return template


def _extract_json_path(data: Any, path: str) -> str:
    """Extract a value from nested JSON using dot notation.

    Supports dict keys and integer list indices:
    ``"choices.0.message.content"`` → ``data["choices"][0]["message"]["content"]``
    """
    if not path:
        return str(data)
    current = data
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise ValueError(
                f"Cannot traverse path '{path}' at '{part}': got {type(current).__name__}"
            )
    return str(current)


def _expand_headers(headers: dict[str, str]) -> dict[str, str]:
    """Expand ``${VAR_NAME}`` in header values using environment variables."""
    return {
        k: _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), v)
        for k, v in headers.items()
    }


# ---------------------------------------------------------------------------
# Endpoint adapters
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, type[EndpointAdapter]] = {}  # filled after class defs


class EndpointAdapter(ABC):
    """Base class for AI endpoint adapters."""

    def __init__(self, config: EndpointConfig) -> None:
        self.config = config
        self._api_key: str | None = None

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = resolve_api_key(self.config)
        return self._api_key

    def _merge_headers(self, defaults: dict[str, str]) -> dict[str, str]:
        """Merge default headers with user-provided config headers.

        User headers take precedence — this allows overriding the
        Authorization header for providers that use custom auth
        (e.g., ``API-Key`` instead of ``Bearer``).
        """
        headers = dict(defaults)
        if self.config.headers:
            expanded = _expand_headers(self.config.headers)
            headers.update(expanded)
        return headers

    @abstractmethod
    async def send(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float | None,
    ) -> str:
        """Send a single prompt and return the raw text response."""

    @classmethod
    def from_config(cls, config: EndpointConfig) -> EndpointAdapter:
        """Factory: pick the right adapter based on config.provider."""
        provider = config.provider or _infer_provider(config)
        adapter_cls = _PROVIDER_MAP.get(provider, GenericOpenAIAdapter)
        return adapter_cls(config)


class OpenAIAdapter(EndpointAdapter):
    """OpenAI Chat Completions API."""

    async def send(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float | None,
    ) -> str:
        body: dict[str, object] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        defaults: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            defaults["Authorization"] = f"Bearer {self.api_key}"
        headers = self._merge_headers(defaults)
        resp = await client.post(self.config.url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])


class AnthropicAdapter(EndpointAdapter):
    """Anthropic Messages API."""

    async def send(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float | None,
    ) -> str:
        body: dict[str, object] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        defaults: dict[str, str] = {
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if self.config.api_key_env:
            defaults["x-api-key"] = self.api_key
        headers = self._merge_headers(defaults)
        resp = await client.post(self.config.url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return str(data["content"][0]["text"])


class GenericOpenAIAdapter(EndpointAdapter):
    """Any OpenAI-compatible endpoint (vLLM, Ollama, Together, LiteLLM, etc.)."""

    async def send(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float | None,
    ) -> str:
        body: dict[str, object] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        defaults: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            defaults["Authorization"] = f"Bearer {self.api_key}"
        headers = self._merge_headers(defaults)
        resp = await client.post(self.config.url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])


class GenericHTTPAdapter(EndpointAdapter):
    """Generic HTTP adapter for any endpoint (agents, RAG, custom APIs).

    Uses ``request_template`` to build the request body (with ``{{question}}``
    substitution) and ``response_path`` to extract the answer from the JSON
    response.  Supports ``${VAR}`` expansion in headers.
    """

    async def send(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float | None,
    ) -> str:
        template = self.config.request_template or {"input": prompt}
        body = _substitute_template(template, prompt)

        headers = _expand_headers(self.config.headers)
        headers.setdefault("Content-Type", "application/json")

        # Auto-add Bearer auth if api_key_env is set and no auth header present
        if self.config.api_key_env and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = await client.post(self.config.url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return _extract_json_path(data, self.config.response_path or "output")


# Provider map (filled now that classes are defined)
_PROVIDER_MAP.update(
    {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "together": GenericOpenAIAdapter,
        "generic": GenericOpenAIAdapter,
        "generic_http": GenericHTTPAdapter,
    }
)


def _infer_provider(config: EndpointConfig) -> str:
    """Infer the provider from the endpoint config."""
    if config.request_template is not None:
        return "generic_http"
    if "api.openai.com" in config.url:
        return "openai"
    if "api.anthropic.com" in config.url:
        return "anthropic"
    return "generic"


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class SamplerError(Exception):
    """Raised when sampling fails after all retries."""


class Sampler:
    """Sample K responses per question from an AI endpoint.

    Handles caching, concurrency limiting, and retries with exponential
    backoff.
    """

    def __init__(
        self,
        config: TrustGateConfig,
        cache: DiskCache | None = None,
    ) -> None:
        self.adapter = EndpointAdapter.from_config(config.endpoint)
        self.cache = cache or DiskCache()
        self.endpoint_config = config.endpoint
        self.sampling_config = config.sampling
        self._provider = config.endpoint.provider or _infer_provider(config.endpoint)

    @property
    def k(self) -> int:
        """Effective K (k_fixed if set, otherwise k_max)."""
        return int(self.sampling_config.k_fixed or self.sampling_config.k_max)

    async def sample_question(
        self,
        question: Question,
        k: int | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[SampleResponse]:
        """Sample *k* responses for a single question."""
        k = k or self.k
        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(timeout=self.sampling_config.timeout)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.sampling_config.max_concurrent)

        try:
            tasks = [
                self._sample_one(
                    client=client,  # type: ignore[arg-type]
                    question=question,
                    index=i,
                    semaphore=semaphore,
                )
                for i in range(k)
            ]
            return list(await asyncio.gather(*tasks))
        finally:
            if own_client:
                await client.aclose()  # type: ignore[union-attr]

    async def sample_all(
        self,
        questions: list[Question],
        k: int | None = None,
    ) -> dict[str, list[SampleResponse]]:
        """Sample K responses for every question.  Returns ``{qid: [responses]}``."""
        k = k or self.k
        semaphore = asyncio.Semaphore(self.sampling_config.max_concurrent)

        async with httpx.AsyncClient(timeout=self.sampling_config.timeout) as client:
            tasks = [
                self.sample_question(q, k, client=client, semaphore=semaphore)
                for q in questions
            ]
            results = await asyncio.gather(*tasks)

        return {q.id: resps for q, resps in zip(questions, results)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _sample_one(
        self,
        client: httpx.AsyncClient,
        question: Question,
        index: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> SampleResponse:
        """Fetch a single sample (with cache check and retry)."""
        cache_key = self.cache.key(
            self.endpoint_config.url,
            self._provider,
            self.endpoint_config.model,
            question.text,
            self.endpoint_config.temperature,
            index,
        )

        # Cache hit
        cached = self.cache.get(cache_key)
        if cached is not None:
            return SampleResponse(
                question_id=question.id,
                sample_index=index,
                raw_response=cached,
                cached=True,
            )

        # Cache miss → send request
        if semaphore is not None:
            async with semaphore:
                response = await self._send_with_retry(
                    client, question.text, self.endpoint_config.temperature
                )
        else:
            response = await self._send_with_retry(
                client, question.text, self.endpoint_config.temperature
            )

        # Store in cache
        self.cache.put(
            cache_key,
            response,
            provider=self._provider,
            model=self.endpoint_config.model,
            temperature=self.endpoint_config.temperature,
            index=index,
        )

        return SampleResponse(
            question_id=question.id,
            sample_index=index,
            raw_response=response,
            cached=False,
        )

    async def _send_with_retry(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float | None,
    ) -> str:
        """Send a request with exponential-backoff retries."""
        last_error: Exception | None = None

        for attempt in range(self.sampling_config.retries + 1):
            try:
                return await self.adapter.send(client, prompt, temperature)

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                last_error = exc

                # 4xx (non-retryable) except 429
                if 400 <= status < 500 and status != 429:
                    raise SamplerError(
                        f"Non-retryable HTTP {status}: {exc.response.text[:200]}"
                    ) from exc

                # 429 rate limit — respect Retry-After
                if status == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else _backoff(attempt)
                    logger.warning(
                        "Rate limited (429), retrying in %.1fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        self.sampling_config.retries,
                    )
                    await asyncio.sleep(delay)
                    continue

                # 5xx — retry with backoff
                delay = _backoff(attempt)
                logger.warning(
                    "Server error %d, retrying in %.1fs (attempt %d/%d)",
                    status,
                    delay,
                    attempt + 1,
                    self.sampling_config.retries,
                )
                await asyncio.sleep(delay)

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                delay = _backoff(attempt)
                logger.warning(
                    "Connection failed, retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self.sampling_config.retries,
                    type(exc).__name__,
                )
                await asyncio.sleep(delay)

        raise SamplerError(
            f"All {self.sampling_config.retries + 1} attempts failed"
        ) from last_error


def _backoff(attempt: int) -> float:
    """Exponential backoff: min(2^attempt, 120) seconds."""
    return min(2.0**attempt, 120.0)


# ---------------------------------------------------------------------------
# Public sync wrapper
# ---------------------------------------------------------------------------


def sample(
    config: TrustGateConfig,
    questions: list[Question],
    cache: DiskCache | None = None,
) -> dict[str, list[SampleResponse]]:
    """Synchronous convenience wrapper around :class:`Sampler`."""
    sampler = Sampler(config, cache=cache)
    return asyncio.run(sampler.sample_all(questions))
