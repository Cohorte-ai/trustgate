"""Tests for the certification pipeline orchestrator."""

from __future__ import annotations

import json

import pytest

from theaios.trustgate.certification import (
    ConfigError,
    LabelsRequired,
    _canon_kwargs,
    estimate_cost,
    load_ground_truth,
)
from theaios.trustgate.sampler import Sampler
from theaios.trustgate.types import (
    CanonConfig,
    EndpointConfig,
    SampleResponse,
    SamplingConfig,
    TrustGateConfig,
)


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_loads_csv(self, tmp_path: object) -> None:
        import pathlib

        p = pathlib.Path(str(tmp_path)) / "labels.csv"
        p.write_text("id,label\nq1,correct\nq2,incorrect\n")
        labels = load_ground_truth(str(p))
        assert labels == {"q1": "correct", "q2": "incorrect"}

    def test_loads_json(self, tmp_path: object) -> None:
        import pathlib

        p = pathlib.Path(str(tmp_path)) / "labels.json"
        p.write_text(json.dumps({"q1": "42", "q2": "B"}))
        labels = load_ground_truth(str(p))
        assert labels == {"q1": "42", "q2": "B"}

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_ground_truth("/nonexistent/labels.csv")

    def test_raises_on_unsupported_format(self, tmp_path: object) -> None:
        import pathlib

        p = pathlib.Path(str(tmp_path)) / "labels.txt"
        p.write_text("something")
        with pytest.raises(ValueError, match="Unsupported ground truth format"):
            load_ground_truth(str(p))

    def test_raises_on_invalid_json(self, tmp_path: object) -> None:
        import pathlib

        p = pathlib.Path(str(tmp_path)) / "labels.json"
        p.write_text("[1, 2, 3]")  # list, not dict
        with pytest.raises(ValueError, match="must be a mapping"):
            load_ground_truth(str(p))

    def test_csv_missing_id_column(self, tmp_path: object) -> None:
        import pathlib

        p = pathlib.Path(str(tmp_path)) / "labels.csv"
        p.write_text("name,value\nq1,correct\n")
        with pytest.raises(ValueError, match="id"):
            load_ground_truth(str(p))


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def _config(self, model: str = "gpt-4.1-mini") -> TrustGateConfig:
        return TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://api.openai.com/v1/chat/completions",
                model=model,
            ),
        )

    def test_known_model(self) -> None:
        responses = {
            "q1": [
                SampleResponse(question_id="q1", sample_index=0, raw_response="A" * 400),
                SampleResponse(question_id="q1", sample_index=1, raw_response="B" * 400),
            ],
        }
        cost = estimate_cost(responses, self._config("gpt-4.1-mini"))
        assert cost > 0

    def test_unknown_model_returns_zero(self) -> None:
        responses = {
            "q1": [
                SampleResponse(question_id="q1", sample_index=0, raw_response="test"),
            ],
        }
        cost = estimate_cost(responses, self._config("unknown-model"))
        assert cost == 0.0

    def test_cached_responses_not_counted(self) -> None:
        responses = {
            "q1": [
                SampleResponse(
                    question_id="q1", sample_index=0, raw_response="A" * 400, cached=True,
                ),
            ],
        }
        cost = estimate_cost(responses, self._config("gpt-4.1-mini"))
        assert cost == 0.0

    def test_empty_responses(self) -> None:
        cost = estimate_cost({}, self._config())
        assert cost == 0.0


# ---------------------------------------------------------------------------
# _canon_kwargs
# ---------------------------------------------------------------------------


class TestCanonKwargs:
    def test_mcq_no_extra(self) -> None:
        assert _canon_kwargs(CanonConfig(type="mcq")) == {}

    def test_llm_judge_includes_config(self) -> None:
        ep = EndpointConfig(url="https://api.openai.com/v1/chat/completions")
        kwargs = _canon_kwargs(CanonConfig(type="llm_judge", judge_endpoint=ep))
        assert "judge_config" in kwargs
        assert kwargs["judge_config"] is ep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_all_exports_importable(self) -> None:
        from theaios import trustgate

        for name in trustgate.__all__:
            assert hasattr(trustgate, name), f"Missing export: {name}"

    def test_certify_exists(self) -> None:
        from theaios import trustgate

        assert callable(trustgate.certify)
        assert callable(trustgate.certify_async)

    def test_version(self) -> None:
        from theaios import trustgate

        assert trustgate.__version__  # just check it's set


# ---------------------------------------------------------------------------
# certify() error handling
# ---------------------------------------------------------------------------


class TestCertifyErrors:
    def test_raises_config_error_on_invalid_config(self) -> None:
        from theaios.trustgate.certification import certify

        bad_config = TrustGateConfig(
            endpoint=EndpointConfig(url="not-a-url"),
        )
        with pytest.raises(ConfigError, match="Invalid configuration"):
            certify(config=bad_config, questions=[], labels={})

    def test_raises_labels_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import AsyncMock, patch

        from theaios.trustgate.certification import certify
        from theaios.trustgate.types import Question

        monkeypatch.setenv("TEST_KEY", "sk-test")

        config = TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://api.openai.com/v1/chat/completions",
                model="gpt-4.1-mini",
                api_key_env="TEST_KEY",
            ),
            sampling=SamplingConfig(
                k_fixed=5,
                sequential_stopping=False,
            ),
        )
        # Questions with NO acceptable_answers → should raise LabelsRequired
        questions = [
            Question(id=f"q{i}", text=f"Question {i}")
            for i in range(20)
        ]

        mock_responses = {
            f"q{i}": [
                SampleResponse(
                    question_id=f"q{i}", sample_index=j, raw_response="B",
                )
                for j in range(5)
            ]
            for i in range(20)
        }

        with patch.object(
            Sampler, "sample_all", new_callable=AsyncMock, return_value=mock_responses,
        ):
            with pytest.raises(LabelsRequired):
                certify(config=config, questions=questions)
