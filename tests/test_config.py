"""Tests for config loading, validation, and question loading."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from theaios.trustgate.config import (
    ConfigError,
    load_config,
    load_questions,
    resolve_api_key,
    validate_config,
)
from theaios.trustgate.types import (
    CanonConfig,
    EndpointConfig,
    QuestionsConfig,
    SamplingConfig,
    TrustGateConfig,
)


# -----------------------------------------------------------------------
# load_config
# -----------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_yaml(self, sample_yaml: Path) -> None:
        config = load_config(str(sample_yaml))
        assert config.endpoint.url == "https://api.openai.com/v1/chat/completions"
        assert config.endpoint.model == "gpt-4.1-mini"
        assert config.endpoint.temperature == 0.7
        assert config.sampling.k_fixed == 10
        assert config.sampling.k_max == 20
        assert config.canonicalization.type == "mcq"
        assert config.calibration.n_cal == 250
        assert config.calibration.alpha_values == [0.01, 0.05, 0.10]

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/trustgate.yaml")

    def test_raises_on_missing_endpoint(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("sampling:\n  k_fixed: 5\n")
        with pytest.raises(ConfigError, match="endpoint"):
            load_config(str(p))

    def test_raises_on_missing_url(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            endpoint:
              model: "gpt-4.1-mini"
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "no_url.yaml"
        p.write_text(content)
        with pytest.raises(ConfigError, match="endpoint.url"):
            load_config(str(p))

    def test_applies_defaults(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            endpoint:
              url: "https://api.openai.com/v1/chat/completions"
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "minimal.yaml"
        p.write_text(content)
        config = load_config(str(p))
        assert config.sampling.k_max == 20
        assert config.sampling.sequential_stopping is True
        assert config.canonicalization.type == "mcq"
        assert config.calibration.n_cal == 500

    def test_generic_endpoint_config(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            endpoint:
              url: "https://my-agent.example.com/ask"
              temperature: null
              headers:
                X-Custom: "value"
              request_template:
                query: "{{question}}"
              response_path: "answer"
              cost_per_request: 0.05
            questions:
              file: "q.csv"
        """)
        p = tmp_path / "generic.yaml"
        p.write_text(content)
        config = load_config(str(p))
        assert config.endpoint.temperature is None
        assert config.endpoint.model == ""
        assert config.endpoint.headers == {"X-Custom": "value"}
        assert config.endpoint.request_template == {"query": "{{question}}"}
        assert config.endpoint.response_path == "answer"
        assert config.endpoint.cost_per_request == 0.05

    def test_applies_overrides(self, sample_yaml: Path) -> None:
        config = load_config(
            str(sample_yaml),
            overrides={"endpoint.model": "gpt-4.1", "sampling.k_fixed": 5},
        )
        assert config.endpoint.model == "gpt-4.1"
        assert config.sampling.k_fixed == 5

    def test_overrides_create_nested_keys(self, sample_yaml: Path) -> None:
        config = load_config(
            str(sample_yaml),
            overrides={"canonicalization.type": "numeric"},
        )
        assert config.canonicalization.type == "numeric"

    def test_raises_on_non_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "scalar.yaml"
        p.write_text("just a string\n")
        with pytest.raises(ConfigError, match="YAML mapping"):
            load_config(str(p))


# -----------------------------------------------------------------------
# validate_config
# -----------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config(self, sample_config: TrustGateConfig) -> None:
        assert validate_config(sample_config) == []

    def test_invalid_url(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="not-a-url"),
            questions=QuestionsConfig(file="q.csv"),
        )
        errors = validate_config(config)
        assert any("endpoint.url" in e for e in errors)

    def test_k_fixed_exceeds_k_max(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://api.openai.com/v1/chat/completions"),
            sampling=SamplingConfig(k_fixed=30, k_max=20),
        )
        errors = validate_config(config)
        assert any("k_fixed" in e for e in errors)

    def test_invalid_canon_type(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://api.openai.com/v1/chat/completions"),
            canonicalization=CanonConfig(type="invalid_type"),
        )
        errors = validate_config(config)
        assert any("canonicalization.type" in e for e in errors)

    def test_custom_without_class(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://api.openai.com/v1/chat/completions"),
            canonicalization=CanonConfig(type="custom"),
        )
        errors = validate_config(config)
        assert any("custom_class" in e for e in errors)

    def test_llm_judge_without_endpoint(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://api.openai.com/v1/chat/completions"),
            canonicalization=CanonConfig(type="llm_judge"),
        )
        errors = validate_config(config)
        assert any("judge_endpoint" in e for e in errors)

    def test_n_cal_zero(self) -> None:
        from theaios.trustgate.types import CalibrationConfig

        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://api.openai.com/v1/chat/completions"),
            calibration=CalibrationConfig(n_cal=0),
        )
        errors = validate_config(config)
        assert any("n_cal" in e for e in errors)

    def test_n_test_zero(self) -> None:
        from theaios.trustgate.types import CalibrationConfig

        config = TrustGateConfig(
            endpoint=EndpointConfig(url="https://api.openai.com/v1/chat/completions"),
            calibration=CalibrationConfig(n_test=0),
        )
        errors = validate_config(config)
        assert any("n_test" in e for e in errors)


# -----------------------------------------------------------------------
# resolve_api_key
# -----------------------------------------------------------------------


class TestResolveApiKey:
    def test_reads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "sk-test-123")
        key = resolve_api_key(EndpointConfig(url="https://x.com", api_key_env="MY_KEY"))
        assert key == "sk-test-123"

    def test_raises_on_missing_env_var(self) -> None:
        with pytest.raises(ConfigError, match="not set"):
            resolve_api_key(
                EndpointConfig(url="https://x.com", api_key_env="NONEXISTENT_VAR_12345")
            )

    def test_raises_on_empty_env_name(self) -> None:
        with pytest.raises(ConfigError, match="api_key_env"):
            resolve_api_key(EndpointConfig(url="https://x.com", api_key_env=""))


# -----------------------------------------------------------------------
# load_questions
# -----------------------------------------------------------------------


class TestLoadQuestions:
    def test_loads_csv(self, sample_questions_csv: Path) -> None:
        questions = load_questions(str(sample_questions_csv))
        assert len(questions) == 3
        assert questions[0].id == "q001"
        assert questions[0].text.startswith("What is 2+2?")
        assert questions[0].acceptable_answers == ["B"]

    def test_loads_json(self, sample_questions_json: Path) -> None:
        questions = load_questions(str(sample_questions_json))
        assert len(questions) == 3
        assert questions[0].id == "q001"
        assert questions[0].text == "What is 2+2?"
        assert questions[0].acceptable_answers == ["4"]

    def test_loads_from_questions_config(self, sample_questions_csv: Path) -> None:
        config = QuestionsConfig(file=str(sample_questions_csv))
        questions = load_questions(config)
        assert len(questions) == 3

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_questions("/nonexistent/questions.csv")

    def test_raises_on_unsupported_format(self, tmp_path: Path) -> None:
        p = tmp_path / "questions.xml"
        p.write_text("<questions/>")
        with pytest.raises(ConfigError, match="Unsupported"):
            load_questions(str(p))

    def test_raises_on_malformed_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"not": "a list"}')
        with pytest.raises(ConfigError, match="list"):
            load_questions(str(p))

    def test_raises_on_missing_id_in_json(self, tmp_path: Path) -> None:
        data = [{"question": "What?"}]
        p = tmp_path / "no_id.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ConfigError, match="'id'"):
            load_questions(str(p))

    def test_raises_on_csv_missing_columns(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("name,value\nfoo,bar\n")
        with pytest.raises(ConfigError, match="'id'"):
            load_questions(str(p))

    def test_raises_on_empty_questions_config(self) -> None:
        with pytest.raises(ConfigError, match="questions.file"):
            load_questions(QuestionsConfig())

    def test_csv_pipe_separated_answers(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            id,question,acceptable_answers
            q001,"What is 2+2?","4|four"
        """)
        p = tmp_path / "multi.csv"
        p.write_text(content)
        questions = load_questions(str(p))
        assert questions[0].acceptable_answers == ["4", "four"]

    def test_json_string_acceptable_answer(self, tmp_path: Path) -> None:
        data = [{"id": "q1", "question": "Q?", "acceptable_answers": "42"}]
        p = tmp_path / "str_answer.json"
        p.write_text(json.dumps(data))
        questions = load_questions(str(p))
        assert questions[0].acceptable_answers == ["42"]
