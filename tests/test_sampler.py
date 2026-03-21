"""Tests for the sampler module (adapters, sampler, retry logic)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import respx

from trustgate.cache import DiskCache
from trustgate.sampler import (
    AnthropicAdapter,
    EndpointAdapter,
    GenericOpenAIAdapter,
    OpenAIAdapter,
    Sampler,
    SamplerError,
    _backoff,
    _infer_provider,
    sample,
)
from trustgate.types import (
    EndpointConfig,
    Question,
    SamplingConfig,
    TrustGateConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GENERIC_URL = "https://my-custom-llm.example.com/v1/chat/completions"


def _openai_response(text: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": text}}]}


def _anthropic_response(text: str) -> dict[str, object]:
    return {"content": [{"text": text}]}


def _endpoint(url: str = OPENAI_URL, provider: str = "openai") -> EndpointConfig:
    return EndpointConfig(
        url=url,
        model="gpt-4.1-mini",
        temperature=0.7,
        api_key_env="TEST_API_KEY",
        provider=provider,
    )


def _config(
    url: str = OPENAI_URL,
    provider: str = "openai",
    k_fixed: int = 3,
    retries: int = 2,
    max_concurrent: int = 10,
    timeout: float = 5.0,
) -> TrustGateConfig:
    return TrustGateConfig(
        endpoint=_endpoint(url, provider),
        sampling=SamplingConfig(
            k_fixed=k_fixed,
            k_max=20,
            retries=retries,
            max_concurrent=max_concurrent,
            timeout=timeout,
        ),
    )


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-key")


@pytest.fixture()
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache(cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------


class TestProviderInference:
    def test_openai(self) -> None:
        assert _infer_provider(_endpoint(OPENAI_URL, "")) == "openai"

    def test_anthropic(self) -> None:
        assert _infer_provider(_endpoint(ANTHROPIC_URL, "")) == "anthropic"

    def test_generic(self) -> None:
        assert _infer_provider(_endpoint(GENERIC_URL, "")) == "generic"

    def test_generic_http_from_request_template(self) -> None:
        cfg = EndpointConfig(
            url="https://my-agent.example.com/ask",
            request_template={"input": "{{question}}"},
        )
        assert _infer_provider(cfg) == "generic_http"


# ---------------------------------------------------------------------------
# EndpointAdapter.from_config
# ---------------------------------------------------------------------------


class TestEndpointAdapterFactory:
    def test_picks_openai(self) -> None:
        adapter = EndpointAdapter.from_config(_endpoint(OPENAI_URL, "openai"))
        assert isinstance(adapter, OpenAIAdapter)

    def test_picks_anthropic(self) -> None:
        adapter = EndpointAdapter.from_config(_endpoint(ANTHROPIC_URL, "anthropic"))
        assert isinstance(adapter, AnthropicAdapter)

    def test_picks_generic(self) -> None:
        adapter = EndpointAdapter.from_config(_endpoint(GENERIC_URL, "generic"))
        assert isinstance(adapter, GenericOpenAIAdapter)

    def test_infers_openai_from_url(self) -> None:
        adapter = EndpointAdapter.from_config(_endpoint(OPENAI_URL, ""))
        assert isinstance(adapter, OpenAIAdapter)

    def test_infers_anthropic_from_url(self) -> None:
        adapter = EndpointAdapter.from_config(_endpoint(ANTHROPIC_URL, ""))
        assert isinstance(adapter, AnthropicAdapter)

    def test_infers_generic_from_url(self) -> None:
        adapter = EndpointAdapter.from_config(_endpoint(GENERIC_URL, ""))
        assert isinstance(adapter, GenericOpenAIAdapter)


# ---------------------------------------------------------------------------
# Adapter.send()
# ---------------------------------------------------------------------------


class TestOpenAIAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_send(self) -> None:
        route = respx.post(OPENAI_URL).respond(
            json=_openai_response("The answer is B")
        )
        adapter = OpenAIAdapter(_endpoint(OPENAI_URL, "openai"))
        async with httpx.AsyncClient() as client:
            result = await adapter.send(client, "What is 2+2?", 0.7)
        assert result == "The answer is B"
        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_sends_correct_body(self) -> None:
        route = respx.post(OPENAI_URL).respond(
            json=_openai_response("B")
        )
        adapter = OpenAIAdapter(_endpoint(OPENAI_URL, "openai"))
        async with httpx.AsyncClient() as client:
            await adapter.send(client, "question?", 0.5)
        body = json.loads(route.calls[0].request.content)
        assert body["model"] == "gpt-4.1-mini"
        assert body["messages"] == [{"role": "user", "content": "question?"}]
        assert body["temperature"] == 0.5

    @respx.mock
    @pytest.mark.asyncio
    async def test_sends_auth_header(self) -> None:
        route = respx.post(OPENAI_URL).respond(
            json=_openai_response("B")
        )
        adapter = OpenAIAdapter(_endpoint(OPENAI_URL, "openai"))
        async with httpx.AsyncClient() as client:
            await adapter.send(client, "q", 0.7)
        auth = route.calls[0].request.headers["authorization"]
        assert auth == "Bearer sk-test-key"


class TestAnthropicAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_send(self) -> None:
        respx.post(ANTHROPIC_URL).respond(
            json=_anthropic_response("Paris")
        )
        adapter = AnthropicAdapter(_endpoint(ANTHROPIC_URL, "anthropic"))
        async with httpx.AsyncClient() as client:
            result = await adapter.send(client, "Capital of France?", 0.7)
        assert result == "Paris"

    @respx.mock
    @pytest.mark.asyncio
    async def test_sends_anthropic_headers(self) -> None:
        route = respx.post(ANTHROPIC_URL).respond(
            json=_anthropic_response("B")
        )
        adapter = AnthropicAdapter(_endpoint(ANTHROPIC_URL, "anthropic"))
        async with httpx.AsyncClient() as client:
            await adapter.send(client, "q", 0.7)
        headers = route.calls[0].request.headers
        assert headers["x-api-key"] == "sk-test-key"
        assert "anthropic-version" in headers


class TestGenericOpenAIAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_send(self) -> None:
        respx.post(GENERIC_URL).respond(
            json=_openai_response("42")
        )
        adapter = GenericOpenAIAdapter(_endpoint(GENERIC_URL, "generic"))
        async with httpx.AsyncClient() as client:
            result = await adapter.send(client, "What is 6*7?", 0.7)
        assert result == "42"


# ---------------------------------------------------------------------------
# Sampler: basic sampling
# ---------------------------------------------------------------------------


class TestSamplerBasic:
    @respx.mock
    @pytest.mark.asyncio
    async def test_sample_question_returns_k_responses(self, cache: DiskCache) -> None:
        respx.post(OPENAI_URL).respond(json=_openai_response("B"))
        sampler = Sampler(_config(k_fixed=5), cache=cache)
        q = Question(id="q1", text="What?")
        responses = await sampler.sample_question(q, k=5)
        assert len(responses) == 5
        assert all(r.question_id == "q1" for r in responses)
        assert all(r.raw_response == "B" for r in responses)
        assert [r.sample_index for r in responses] == [0, 1, 2, 3, 4]

    @respx.mock
    @pytest.mark.asyncio
    async def test_sample_all_multiple_questions(self, cache: DiskCache) -> None:
        respx.post(OPENAI_URL).respond(json=_openai_response("answer"))
        sampler = Sampler(_config(k_fixed=3), cache=cache)
        questions = [
            Question(id="q1", text="Q1?"),
            Question(id="q2", text="Q2?"),
            Question(id="q3", text="Q3?"),
        ]
        results = await sampler.sample_all(questions, k=3)
        assert len(results) == 3
        assert all(qid in results for qid in ["q1", "q2", "q3"])
        assert all(len(resps) == 3 for resps in results.values())


# ---------------------------------------------------------------------------
# Sampler: cache integration
# ---------------------------------------------------------------------------


class TestSamplerCache:
    @respx.mock
    @pytest.mark.asyncio
    async def test_caches_new_responses(self, cache: DiskCache) -> None:
        respx.post(OPENAI_URL).respond(json=_openai_response("B"))
        sampler = Sampler(_config(k_fixed=2), cache=cache)
        q = Question(id="q1", text="What?")
        responses = await sampler.sample_question(q, k=2)
        assert all(r.cached is False for r in responses)
        assert cache.stats()["total_entries"] == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_uses_cache_on_second_call(self, cache: DiskCache) -> None:
        route = respx.post(OPENAI_URL).respond(json=_openai_response("B"))
        sampler = Sampler(_config(k_fixed=2), cache=cache)
        q = Question(id="q1", text="What?")

        # First call — hits the API
        await sampler.sample_question(q, k=2)
        assert route.call_count == 2

        # Second call — all cached, no new API calls
        responses = await sampler.sample_question(q, k=2)
        assert route.call_count == 2  # no new calls
        assert all(r.cached is True for r in responses)

    @respx.mock
    @pytest.mark.asyncio
    async def test_rerun_zero_http_calls(self, cache: DiskCache) -> None:
        """Full integration: re-running sample_all with same questions uses only cache."""
        route = respx.post(OPENAI_URL).respond(json=_openai_response("X"))
        sampler = Sampler(_config(k_fixed=3), cache=cache)
        questions = [Question(id=f"q{i}", text=f"Q{i}?") for i in range(3)]

        await sampler.sample_all(questions, k=3)
        first_count = route.call_count
        assert first_count == 9  # 3 questions * 3 samples

        results = await sampler.sample_all(questions, k=3)
        assert route.call_count == 9  # unchanged — all from cache
        for resps in results.values():
            assert all(r.cached is True for r in resps)


# ---------------------------------------------------------------------------
# Sampler: concurrency
# ---------------------------------------------------------------------------


class TestSamplerConcurrency:
    @respx.mock
    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, cache: DiskCache) -> None:
        """Verify that the semaphore limits parallel in-flight requests."""
        max_concurrent = 2
        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _slow_handler(request: httpx.Request) -> httpx.Response:
            nonlocal peak_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                peak_concurrent = max(peak_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return httpx.Response(200, json=_openai_response("ok"))

        respx.post(OPENAI_URL).mock(side_effect=_slow_handler)
        sampler = Sampler(
            _config(k_fixed=6, max_concurrent=max_concurrent), cache=cache
        )
        q = Question(id="q1", text="What?")
        await sampler.sample_question(q, k=6)
        assert peak_concurrent <= max_concurrent


# ---------------------------------------------------------------------------
# Sampler: retry logic
# ---------------------------------------------------------------------------


class TestSamplerRetry:
    @respx.mock
    @pytest.mark.asyncio
    async def test_retries_on_500(self, cache: DiskCache) -> None:
        route = respx.post(OPENAI_URL).mock(
            side_effect=[
                httpx.Response(500, text="Internal Server Error"),
                httpx.Response(200, json=_openai_response("ok")),
            ]
        )
        sampler = Sampler(_config(k_fixed=1, retries=3), cache=cache)
        q = Question(id="q1", text="What?")
        responses = await sampler.sample_question(q, k=1)
        assert responses[0].raw_response == "ok"
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_retries_on_429(self, cache: DiskCache) -> None:
        route = respx.post(OPENAI_URL).mock(
            side_effect=[
                httpx.Response(429, text="Rate limited", headers={"Retry-After": "0"}),
                httpx.Response(200, json=_openai_response("ok")),
            ]
        )
        sampler = Sampler(_config(k_fixed=1, retries=3), cache=cache)
        q = Question(id="q1", text="What?")
        responses = await sampler.sample_question(q, k=1)
        assert responses[0].raw_response == "ok"
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_fails_on_400(self, cache: DiskCache) -> None:
        respx.post(OPENAI_URL).respond(400, text="Bad Request")
        sampler = Sampler(_config(k_fixed=1, retries=3), cache=cache)
        q = Question(id="q1", text="What?")
        with pytest.raises(SamplerError, match="Non-retryable HTTP 400"):
            await sampler.sample_question(q, k=1)

    @respx.mock
    @pytest.mark.asyncio
    async def test_fails_on_401(self, cache: DiskCache) -> None:
        respx.post(OPENAI_URL).respond(401, text="Unauthorized")
        sampler = Sampler(_config(k_fixed=1, retries=3), cache=cache)
        q = Question(id="q1", text="What?")
        with pytest.raises(SamplerError, match="Non-retryable HTTP 401"):
            await sampler.sample_question(q, k=1)

    @respx.mock
    @pytest.mark.asyncio
    async def test_exhausts_retries(self, cache: DiskCache) -> None:
        respx.post(OPENAI_URL).respond(500, text="Error")
        sampler = Sampler(_config(k_fixed=1, retries=2), cache=cache)
        q = Question(id="q1", text="What?")
        with pytest.raises(SamplerError, match="All 3 attempts failed"):
            await sampler.sample_question(q, k=1)


# ---------------------------------------------------------------------------
# Backoff helper
# ---------------------------------------------------------------------------


class TestBackoff:
    def test_exponential(self) -> None:
        assert _backoff(0) == 1.0
        assert _backoff(1) == 2.0
        assert _backoff(2) == 4.0
        assert _backoff(3) == 8.0

    def test_capped_at_120(self) -> None:
        assert _backoff(10) == 120.0
        assert _backoff(20) == 120.0


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


class TestSyncSample:
    @respx.mock
    def test_sample_sync(self, cache: DiskCache) -> None:
        respx.post(OPENAI_URL).respond(json=_openai_response("answer"))
        config = _config(k_fixed=2)
        questions = [Question(id="q1", text="What?")]
        results = sample(config, questions, cache=cache)
        assert "q1" in results
        assert len(results["q1"]) == 2
        assert all(r.raw_response == "answer" for r in results["q1"])


# ---------------------------------------------------------------------------
# Integration: full flow
# ---------------------------------------------------------------------------


class TestSamplerIntegration:
    @respx.mock
    @pytest.mark.asyncio
    async def test_full_flow(self, cache: DiskCache) -> None:
        """Load config → create sampler → sample 3 questions × K=5 → verify."""
        call_count = 0

        def _responder(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            prompt = body["messages"][0]["content"]
            # Return a response that includes the question for verification
            return httpx.Response(
                200, json=_openai_response(f"Answer to: {prompt[:20]}")
            )

        respx.post(OPENAI_URL).mock(side_effect=_responder)

        config = _config(k_fixed=5)
        sampler = Sampler(config, cache=cache)
        questions = [
            Question(id="q1", text="What is 2+2?"),
            Question(id="q2", text="Capital of France?"),
            Question(id="q3", text="Largest planet?"),
        ]

        results = await sampler.sample_all(questions, k=5)

        # 3 questions * 5 samples = 15 calls
        assert call_count == 15
        assert len(results) == 3
        for qid in ["q1", "q2", "q3"]:
            assert len(results[qid]) == 5

        # Re-run: all cached
        results2 = await sampler.sample_all(questions, k=5)
        assert call_count == 15  # no new calls
        for resps in results2.values():
            assert all(r.cached is True for r in resps)
