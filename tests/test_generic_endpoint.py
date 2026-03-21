"""Tests for generic endpoint support, pre-flight cost estimation, and related CLI changes."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx
from click.testing import CliRunner

from theaios.trustgate.cache import DiskCache
from theaios.trustgate.certification import (
    estimate_cost,
    estimate_cost_reliability_arbitrage,
    estimate_preflight_cost,
)
from theaios.trustgate.cli import main
from theaios.trustgate.config import load_config
from theaios.trustgate.sampler import (
    GenericHTTPAdapter,
    OpenAIAdapter,
    _expand_headers,
    _extract_json_path,
    _infer_provider,
    _substitute_template,
)
from theaios.trustgate.types import (
    CertificationResult,
    EndpointConfig,
    SamplingConfig,
    SampleResponse,
    TrustGateConfig,
)


# ===========================================================================
# _substitute_template
# ===========================================================================


class TestSubstituteTemplate:
    def test_simple_string(self) -> None:
        assert _substitute_template("{{question}}", "What is 2+2?") == "What is 2+2?"

    def test_string_without_placeholder(self) -> None:
        assert _substitute_template("no placeholder", "Q?") == "no placeholder"

    def test_dict_with_placeholder(self) -> None:
        template = {"query": "{{question}}", "context": "Be concise."}
        result = _substitute_template(template, "What is 2+2?")
        assert result == {"query": "What is 2+2?", "context": "Be concise."}

    def test_nested_dict(self) -> None:
        template = {"outer": {"inner": "{{question}}"}}
        result = _substitute_template(template, "Hello")
        assert result == {"outer": {"inner": "Hello"}}

    def test_list(self) -> None:
        template = ["{{question}}", "other"]
        result = _substitute_template(template, "Q?")
        assert result == ["Q?", "other"]

    def test_mixed_nested(self) -> None:
        template = {"messages": [{"role": "user", "content": "{{question}}"}]}
        result = _substitute_template(template, "What?")
        assert result == {"messages": [{"role": "user", "content": "What?"}]}

    def test_non_string_values_unchanged(self) -> None:
        template = {"query": "{{question}}", "max_tokens": 100, "stream": False}
        result = _substitute_template(template, "Q?")
        assert result == {"query": "Q?", "max_tokens": 100, "stream": False}

    def test_multiple_placeholders_in_one_string(self) -> None:
        template = "Q: {{question}} -- Repeat: {{question}}"
        result = _substitute_template(template, "Hi")
        assert result == "Q: Hi -- Repeat: Hi"

    def test_empty_question(self) -> None:
        assert _substitute_template("{{question}}", "") == ""

    def test_none_passthrough(self) -> None:
        assert _substitute_template(None, "Q?") is None

    def test_int_passthrough(self) -> None:
        assert _substitute_template(42, "Q?") == 42


# ===========================================================================
# _extract_json_path
# ===========================================================================


class TestExtractJsonPath:
    def test_simple_key(self) -> None:
        assert _extract_json_path({"output": "hello"}, "output") == "hello"

    def test_nested_key(self) -> None:
        data = {"data": {"text": "answer"}}
        assert _extract_json_path(data, "data.text") == "answer"

    def test_list_index(self) -> None:
        data = {"choices": [{"message": {"content": "Paris"}}]}
        assert _extract_json_path(data, "choices.0.message.content") == "Paris"

    def test_empty_path_returns_str(self) -> None:
        assert _extract_json_path({"a": 1}, "") == "{'a': 1}"

    def test_deep_nesting(self) -> None:
        data = {"a": {"b": {"c": {"d": "deep"}}}}
        assert _extract_json_path(data, "a.b.c.d") == "deep"

    def test_numeric_value(self) -> None:
        assert _extract_json_path({"result": 42}, "result") == "42"

    def test_boolean_value(self) -> None:
        assert _extract_json_path({"ok": True}, "ok") == "True"

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(KeyError):
            _extract_json_path({"a": 1}, "b")

    def test_invalid_index_raises(self) -> None:
        with pytest.raises(IndexError):
            _extract_json_path({"items": [1]}, "items.5")

    def test_traverse_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot traverse"):
            _extract_json_path({"a": "string"}, "a.b")


# ===========================================================================
# _expand_headers
# ===========================================================================


class TestExpandHeaders:
    def test_expands_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_SECRET", "s3cret")
        result = _expand_headers({"Authorization": "Bearer ${MY_SECRET}"})
        assert result == {"Authorization": "Bearer s3cret"}

    def test_missing_env_var_becomes_empty(self) -> None:
        result = _expand_headers({"Auth": "Bearer ${NONEXISTENT_VAR_XYZ_12345}"})
        assert result == {"Auth": "Bearer "}

    def test_no_expansion_needed(self) -> None:
        result = _expand_headers({"Content-Type": "application/json"})
        assert result == {"Content-Type": "application/json"}

    def test_multiple_vars_in_one_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("USER_A", "alice")
        monkeypatch.setenv("TOKEN_A", "tok123")
        result = _expand_headers({"X-Auth": "${USER_A}:${TOKEN_A}"})
        assert result == {"X-Auth": "alice:tok123"}

    def test_empty_dict(self) -> None:
        assert _expand_headers({}) == {}


# ===========================================================================
# _infer_provider (updated for generic_http)
# ===========================================================================


class TestInferProviderGeneric:
    def test_request_template_triggers_generic_http(self) -> None:
        cfg = EndpointConfig(
            url="https://my-agent.example.com/ask",
            request_template={"input": "{{question}}"},
        )
        assert _infer_provider(cfg) == "generic_http"

    def test_openai_url_without_template(self) -> None:
        cfg = EndpointConfig(url="https://api.openai.com/v1/chat/completions")
        assert _infer_provider(cfg) == "openai"

    def test_anthropic_url_without_template(self) -> None:
        cfg = EndpointConfig(url="https://api.anthropic.com/v1/messages")
        assert _infer_provider(cfg) == "anthropic"

    def test_unknown_url_without_template(self) -> None:
        cfg = EndpointConfig(url="https://my-llm.example.com/v1/chat")
        assert _infer_provider(cfg) == "generic"

    def test_template_overrides_url_detection(self) -> None:
        # Even if URL looks like OpenAI, template takes precedence
        cfg = EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            request_template={"input": "{{question}}"},
        )
        assert _infer_provider(cfg) == "generic_http"


# ===========================================================================
# GenericHTTPAdapter
# ===========================================================================

_AGENT_URL = "https://my-agent.example.com/api/ask"


class TestGenericHTTPAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_sends_request_with_template(self) -> None:
        route = respx.post(_AGENT_URL).respond(json={"answer": "Paris"})
        cfg = EndpointConfig(
            url=_AGENT_URL,
            request_template={"query": "{{question}}", "context": "Be brief."},
            response_path="answer",
        )
        adapter = GenericHTTPAdapter(cfg)
        async with httpx.AsyncClient() as client:
            result = await adapter.send(client, "Capital of France?", None)

        assert result == "Paris"
        assert route.called
        req_body = json.loads(route.calls[0].request.content)
        assert req_body == {"query": "Capital of France?", "context": "Be brief."}

    @respx.mock
    @pytest.mark.asyncio
    async def test_nested_response_path(self) -> None:
        respx.post(_AGENT_URL).respond(
            json={"data": {"response": {"text": "42"}}}
        )
        cfg = EndpointConfig(
            url=_AGENT_URL,
            request_template={"q": "{{question}}"},
            response_path="data.response.text",
        )
        adapter = GenericHTTPAdapter(cfg)
        async with httpx.AsyncClient() as client:
            result = await adapter.send(client, "What?", None)
        assert result == "42"

    @respx.mock
    @pytest.mark.asyncio
    async def test_default_template_and_path(self) -> None:
        respx.post(_AGENT_URL).respond(json={"output": "hello"})
        cfg = EndpointConfig(url=_AGENT_URL)
        adapter = GenericHTTPAdapter(cfg)
        async with httpx.AsyncClient() as client:
            result = await adapter.send(client, "Hi", None)
        assert result == "hello"

    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_KEY", "secret123")
        route = respx.post(_AGENT_URL).respond(json={"output": "ok"})
        cfg = EndpointConfig(
            url=_AGENT_URL,
            headers={"X-Custom": "value", "Authorization": "Bearer ${AGENT_KEY}"},
            request_template={"q": "{{question}}"},
        )
        adapter = GenericHTTPAdapter(cfg)
        async with httpx.AsyncClient() as client:
            await adapter.send(client, "Q?", None)

        req_headers = route.calls[0].request.headers
        assert req_headers["x-custom"] == "value"
        assert req_headers["authorization"] == "Bearer secret123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_auto_bearer_from_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "sk-test")
        route = respx.post(_AGENT_URL).respond(json={"output": "ok"})
        cfg = EndpointConfig(
            url=_AGENT_URL,
            api_key_env="MY_KEY",
            request_template={"q": "{{question}}"},
        )
        adapter = GenericHTTPAdapter(cfg)
        async with httpx.AsyncClient() as client:
            await adapter.send(client, "Q?", None)

        assert "Bearer sk-test" in route.calls[0].request.headers["authorization"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_temperature_ignored(self) -> None:
        route = respx.post(_AGENT_URL).respond(json={"output": "ok"})
        cfg = EndpointConfig(
            url=_AGENT_URL,
            temperature=None,
            request_template={"q": "{{question}}"},
        )
        adapter = GenericHTTPAdapter(cfg)
        async with httpx.AsyncClient() as client:
            # Temperature is passed but should be ignored by the adapter
            await adapter.send(client, "Q?", None)

        req_body = json.loads(route.calls[0].request.content)
        assert "temperature" not in req_body


# ===========================================================================
# Optional temperature in LLM adapters
# ===========================================================================


class TestOptionalTemperature:
    @respx.mock
    @pytest.mark.asyncio
    async def test_openai_skips_temperature_when_none(self) -> None:
        route = respx.post("https://api.openai.com/v1/chat/completions").respond(
            json={"choices": [{"message": {"content": "B"}}]}
        )
        cfg = EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1",
            temperature=None,
            api_key_env="TEST_KEY",
        )
        adapter = OpenAIAdapter(cfg)
        adapter._api_key = "sk-test"
        async with httpx.AsyncClient() as client:
            await adapter.send(client, "Q?", None)

        req_body = json.loads(route.calls[0].request.content)
        assert "temperature" not in req_body

    @respx.mock
    @pytest.mark.asyncio
    async def test_openai_sends_temperature_when_set(self) -> None:
        route = respx.post("https://api.openai.com/v1/chat/completions").respond(
            json={"choices": [{"message": {"content": "B"}}]}
        )
        cfg = EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1",
            temperature=0.7,
            api_key_env="TEST_KEY",
        )
        adapter = OpenAIAdapter(cfg)
        adapter._api_key = "sk-test"
        async with httpx.AsyncClient() as client:
            await adapter.send(client, "Q?", 0.7)

        req_body = json.loads(route.calls[0].request.content)
        assert req_body["temperature"] == 0.7


# ===========================================================================
# Cache key with URL and None temperature
# ===========================================================================


class TestCacheKeyGenericEndpoint:
    def test_none_temperature_is_deterministic(self) -> None:
        k1 = DiskCache.key("https://agent.example.com/ask", "generic_http", "", "Q?", None, 0)
        k2 = DiskCache.key("https://agent.example.com/ask", "generic_http", "", "Q?", None, 0)
        assert k1 == k2

    def test_none_vs_float_temperature_differ(self) -> None:
        k1 = DiskCache.key("https://x.com", "g", "", "Q?", None, 0)
        k2 = DiskCache.key("https://x.com", "g", "", "Q?", 0.7, 0)
        assert k1 != k2

    def test_different_urls_differ(self) -> None:
        k1 = DiskCache.key("https://a.com/ask", "g", "", "Q?", None, 0)
        k2 = DiskCache.key("https://b.com/ask", "g", "", "Q?", None, 0)
        assert k1 != k2


# ===========================================================================
# Config parsing for generic endpoint fields
# ===========================================================================


class TestConfigGenericFields:
    def test_temperature_null(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""\
            endpoint:
              url: "https://agent.example.com/ask"
              temperature: null
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg_text)
        config = load_config(str(p))
        assert config.endpoint.temperature is None

    def test_temperature_omitted_defaults_to_0_7(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""\
            endpoint:
              url: "https://api.openai.com/v1/chat/completions"
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg_text)
        config = load_config(str(p))
        assert config.endpoint.temperature == 0.7

    def test_headers_parsed(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""\
            endpoint:
              url: "https://agent.example.com/ask"
              headers:
                X-Custom: "value"
                Authorization: "Bearer ${KEY}"
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg_text)
        config = load_config(str(p))
        assert config.endpoint.headers == {"X-Custom": "value", "Authorization": "Bearer ${KEY}"}

    def test_request_template_parsed(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""\
            endpoint:
              url: "https://agent.example.com/ask"
              request_template:
                query: "{{question}}"
                options:
                  verbose: true
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg_text)
        config = load_config(str(p))
        assert config.endpoint.request_template == {
            "query": "{{question}}",
            "options": {"verbose": True},
        }

    def test_response_path_parsed(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""\
            endpoint:
              url: "https://agent.example.com/ask"
              response_path: "data.answer"
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg_text)
        config = load_config(str(p))
        assert config.endpoint.response_path == "data.answer"

    def test_cost_per_request_parsed(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""\
            endpoint:
              url: "https://agent.example.com/ask"
              cost_per_request: 0.03
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg_text)
        config = load_config(str(p))
        assert config.endpoint.cost_per_request == 0.03

    def test_cost_per_request_absent_is_none(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""\
            endpoint:
              url: "https://api.openai.com/v1/chat/completions"
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg_text)
        config = load_config(str(p))
        assert config.endpoint.cost_per_request is None

    def test_headers_absent_is_empty_dict(self, tmp_path: Path) -> None:
        cfg_text = textwrap.dedent("""\
            endpoint:
              url: "https://agent.example.com/ask"
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg_text)
        config = load_config(str(p))
        assert config.endpoint.headers == {}


# ===========================================================================
# Pre-flight cost estimation
# ===========================================================================


class TestPreflightCost:
    def test_known_model(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://api.openai.com/v1/chat/completions",
                model="gpt-4.1-mini",
            ),
            sampling=SamplingConfig(k_fixed=10, sequential_stopping=False),
        )
        est = estimate_preflight_cost(config, 100)
        assert est["n_questions"] == 100
        assert est["k"] == 10
        assert est["total_requests"] == 1000
        assert est["est_requests"] == 1000  # no sequential stopping
        assert est["cost_per_request"] is not None
        assert est["est_cost"] is not None
        assert est["est_cost"] > 0  # type: ignore[operator]

    def test_unknown_model_no_cost(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://custom-llm.example.com/v1/chat",
                model="my-custom-model",
            ),
        )
        est = estimate_preflight_cost(config, 100)
        assert est["cost_per_request"] is None
        assert est["est_cost"] is None
        assert est["max_cost"] is None

    def test_cost_per_request_override(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://agent.example.com/ask",
                cost_per_request=0.05,
            ),
            sampling=SamplingConfig(k_fixed=10, sequential_stopping=False),
        )
        est = estimate_preflight_cost(config, 100)
        assert est["cost_per_request"] == 0.05
        assert est["est_cost"] == 50.0
        assert est["max_cost"] == 50.0

    def test_sequential_stopping_halves_estimate(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://agent.example.com/ask",
                cost_per_request=0.10,
            ),
            sampling=SamplingConfig(k_fixed=10, sequential_stopping=True),
        )
        est = estimate_preflight_cost(config, 100)
        assert est["total_requests"] == 1000
        assert est["est_requests"] == 500
        assert est["est_cost"] == 50.0
        assert est["max_cost"] == 100.0

    def test_zero_questions(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://agent.example.com/ask",
                cost_per_request=0.10,
            ),
        )
        est = estimate_preflight_cost(config, 0)
        assert est["total_requests"] == 0
        assert est["est_cost"] == 0.0

    def test_uses_k_max_when_k_fixed_is_none(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://x.com"),
            sampling=SamplingConfig(k_fixed=None, k_max=20),
        )
        est = estimate_preflight_cost(config, 10)
        assert est["k"] == 20


# ===========================================================================
# Cost/reliability arbitrage
# ===========================================================================


class TestCostReliabilityArbitrage:
    def test_default_k_values(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://agent.example.com/ask",
                cost_per_request=0.01,
            ),
            sampling=SamplingConfig(sequential_stopping=False),
        )
        rows = estimate_cost_reliability_arbitrage(config, 100)
        assert len(rows) == 5
        k_values = [r["k"] for r in rows]
        assert k_values == [3, 5, 10, 15, 20]

    def test_custom_k_values(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://x.com", cost_per_request=0.01),
        )
        rows = estimate_cost_reliability_arbitrage(config, 100, k_values=[5, 10])
        assert len(rows) == 2

    def test_costs_scale_with_k(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://x.com", cost_per_request=0.01),
            sampling=SamplingConfig(sequential_stopping=False),
        )
        rows = estimate_cost_reliability_arbitrage(config, 100)
        costs = [r["max_cost"] for r in rows]
        assert all(costs[i] < costs[i + 1] for i in range(len(costs) - 1))  # type: ignore[operator]

    def test_unknown_cost_returns_none(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://x.com", model="unknown"),
        )
        rows = estimate_cost_reliability_arbitrage(config, 100)
        assert all(r["est_cost"] is None for r in rows)


# ===========================================================================
# estimate_cost (post-run) with cost_per_request
# ===========================================================================


class TestEstimateCostPostRun:
    def test_uses_cost_per_request_when_set(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://x.com", cost_per_request=0.05),
        )
        responses = {
            "q1": [
                SampleResponse(question_id="q1", sample_index=0, raw_response="A", cached=False),
                SampleResponse(question_id="q1", sample_index=1, raw_response="B", cached=False),
                SampleResponse(question_id="q1", sample_index=2, raw_response="C", cached=True),
            ],
        }
        cost = estimate_cost(responses, config)
        # 2 non-cached calls × $0.05
        assert cost == pytest.approx(0.10)

    def test_skips_cached_responses(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://x.com", cost_per_request=1.0),
        )
        responses = {
            "q1": [
                SampleResponse(question_id="q1", sample_index=0, raw_response="A", cached=True),
            ],
        }
        cost = estimate_cost(responses, config)
        assert cost == 0.0

    def test_unknown_model_no_cost_per_request_returns_zero(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://x.com", model="unknown"),
        )
        responses = {
            "q1": [
                SampleResponse(question_id="q1", sample_index=0, raw_response="A", cached=False),
            ],
        }
        cost = estimate_cost(responses, config)
        assert cost == 0.0


# ===========================================================================
# CLI: --cost-per-request and --yes flags
# ===========================================================================


class TestCLINewFlags:
    def test_certify_help_shows_new_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["certify", "--help"])
        assert result.exit_code == 0
        assert "--cost-per-request" in result.output
        assert "--yes" in result.output

    def test_yes_flag_skips_confirmation(self, tmp_path: Path) -> None:
        mock_result = CertificationResult(
            reliability_level=0.90, m_star=1, coverage=0.95,
            conditional_coverage=0.98, capability_gap=0.02,
            n_cal=50, n_test=50, k_used=10, api_cost_estimate=5.0,
            alpha_coverage={0.10: 0.95},
        )
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "endpoint:\n"
            "  url: https://api.openai.com/v1/chat/completions\n"
            "  model: gpt-4.1-mini\n"
            "  api_key_env: TEST_KEY\n"
        )
        runner = CliRunner(env={"TEST_KEY": "sk-test"})
        with patch("theaios.trustgate.cli.certify", return_value=mock_result):
            result = runner.invoke(main, [
                "certify", "--config", str(cfg), "--yes",
                "--output", "json",
            ])
        assert result.exit_code == 0

    def test_cost_per_request_flag_applied(self, tmp_path: Path) -> None:
        # Create a minimal config + questions so preflight runs
        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent("""\
            endpoint:
              url: https://agent.example.com/ask
            questions:
              file: q.csv
        """))
        q = tmp_path / "q.csv"
        q.write_text("id,question,acceptable_answers\nq1,\"What?\",\"A\"\n")

        mock_result = CertificationResult(
            reliability_level=0.90, m_star=1, coverage=0.95,
            conditional_coverage=0.98, capability_gap=0.02,
            n_cal=1, n_test=1, k_used=10, api_cost_estimate=0.5,
            alpha_coverage={0.10: 0.95},
        )
        runner = CliRunner()
        with patch("theaios.trustgate.cli.certify", return_value=mock_result):
            result = runner.invoke(main, [
                "certify", "--config", str(cfg),
                "--cost-per-request", "0.03",
                "--yes", "--output", "json",
            ])
        assert result.exit_code == 0
