"""Tests for the calibrate command and sample_and_rank pipeline."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from theaios.trustgate.certification import sample_and_rank
from theaios.trustgate.cli import main
from theaios.trustgate.types import (
    CanonConfig,
    EndpointConfig,
    Question,
    SamplingConfig,
    SampleResponse,
    TrustGateConfig,
)


# ===========================================================================
# sample_and_rank
# ===========================================================================


class TestSampleAndRank:
    def _make_config(self) -> TrustGateConfig:
        return TrustGateConfig(
            endpoint=EndpointConfig(
                url="https://api.openai.com/v1/chat/completions",
                model="gpt-4.1-mini",
                api_key_env="TEST_KEY",
            ),
            sampling=SamplingConfig(k_fixed=3, sequential_stopping=False),
            canonicalization=CanonConfig(type="mcq"),
        )

    def _make_questions(self) -> list[Question]:
        return [
            Question(id="q1", text="What is 2+2? (A) 3 (B) 4 (C) 5"),
            Question(id="q2", text="Capital of France? (A) London (B) Paris (C) Berlin"),
        ]

    def test_returns_top_answers(self) -> None:
        config = self._make_config()
        questions = self._make_questions()

        # Mock the sampler to return pre-canned responses
        mock_responses = {
            "q1": [
                SampleResponse(question_id="q1", sample_index=0, raw_response="The answer is B"),
                SampleResponse(question_id="q1", sample_index=1, raw_response="B) 4"),
                SampleResponse(question_id="q1", sample_index=2, raw_response="I think B"),
            ],
            "q2": [
                SampleResponse(question_id="q2", sample_index=0, raw_response="The answer is B"),
                SampleResponse(question_id="q2", sample_index=1, raw_response="B) Paris"),
                SampleResponse(question_id="q2", sample_index=2, raw_response="A) London"),
            ],
        }

        with patch("theaios.trustgate.certification.Sampler") as MockSampler:
            instance = MockSampler.return_value
            instance.k = 3
            instance.sample_all = AsyncMock(return_value=mock_responses)

            top = sample_and_rank(config, questions)

        assert top["q1"] == "B"  # all 3 said B
        assert top["q2"] == "B"  # 2 said B, 1 said A → mode is B

    def test_handles_unanimous_answers(self) -> None:
        config = self._make_config()
        questions = [Question(id="q1", text="2+2? (A) 3 (B) 4")]

        mock_responses = {
            "q1": [
                SampleResponse(question_id="q1", sample_index=i, raw_response="B")
                for i in range(3)
            ],
        }

        with patch("theaios.trustgate.certification.Sampler") as MockSampler:
            instance = MockSampler.return_value
            instance.k = 3
            instance.sample_all = AsyncMock(return_value=mock_responses)

            top = sample_and_rank(config, questions)

        assert top["q1"] == "B"

    def test_handles_split_answers(self) -> None:
        config = self._make_config()
        questions = [Question(id="q1", text="Hard question? (A) X (B) Y (C) Z")]

        # All different answers → mode is whichever is first alphabetically at same freq
        mock_responses = {
            "q1": [
                SampleResponse(question_id="q1", sample_index=0, raw_response="A"),
                SampleResponse(question_id="q1", sample_index=1, raw_response="B"),
                SampleResponse(question_id="q1", sample_index=2, raw_response="C"),
            ],
        }

        with patch("theaios.trustgate.certification.Sampler") as MockSampler:
            instance = MockSampler.return_value
            instance.k = 3
            instance.sample_all = AsyncMock(return_value=mock_responses)

            top = sample_and_rank(config, questions)

        # All equal frequency → alphabetical tiebreak → "A"
        assert top["q1"] == "A"

    def test_raises_on_invalid_config(self) -> None:
        config = TrustGateConfig(
            endpoint=EndpointConfig(url="not-a-url"),
        )
        with pytest.raises(Exception, match="Invalid configuration"):
            sample_and_rank(config, [])


# ===========================================================================
# CLI: trustgate calibrate
# ===========================================================================


class TestCalibrateCLI:
    def _make_config_and_questions(self, tmp_path: Path) -> tuple[Path, Path]:
        cfg = tmp_path / "trustgate.yaml"
        cfg.write_text(textwrap.dedent("""\
            endpoint:
              url: "https://api.openai.com/v1/chat/completions"
              model: "gpt-4.1-mini"
              api_key_env: "TEST_KEY"
            sampling:
              k_fixed: 3
              sequential_stopping: false
            canonicalization:
              type: "mcq"
            questions:
              file: "questions.csv"
        """))
        q = tmp_path / "questions.csv"
        q.write_text(textwrap.dedent("""\
            id,question,acceptable_answers
            q1,"2+2? (A) 3 (B) 4","B"
            q2,"Capital of France? (A) London (B) Paris","B"
        """))
        return cfg, q

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["calibrate", "--help"])
        assert result.exit_code == 0
        assert "--serve" in result.output
        assert "--port" in result.output
        assert "--cost-per-request" in result.output

    def test_without_serve_saves_top_answers(self, tmp_path: Path) -> None:
        cfg, q = self._make_config_and_questions(tmp_path)

        mock_top_answers = {"q1": "B", "q2": "B"}

        runner = CliRunner(env={"TEST_KEY": "sk-test"})
        with patch("theaios.trustgate.cli.sample_and_rank", return_value=mock_top_answers):
            result = runner.invoke(main, [
                "calibrate",
                "--config", str(cfg),
                "--questions", str(q),
                "--output", str(tmp_path / "labels.json"),
                "--yes",
            ])

        assert result.exit_code == 0
        assert "Sampled 2 questions" in result.output

        # Should save a *_answers.json file with top answers
        answers_file = tmp_path / "labels_answers.json"
        assert answers_file.exists()
        data = json.loads(answers_file.read_text())
        assert data["q1"]["top_answer"] == "B"
        assert data["q2"]["top_answer"] == "B"

    def test_questions_from_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg, _ = self._make_config_and_questions(tmp_path)
        # Config has relative path "questions.csv", so we need to be in tmp_path
        monkeypatch.chdir(tmp_path)

        mock_top_answers = {"q1": "B", "q2": "B"}

        runner = CliRunner(env={"TEST_KEY": "sk-test"})
        # Don't pass --questions; should load from config
        with patch("theaios.trustgate.cli.sample_and_rank", return_value=mock_top_answers):
            result = runner.invoke(main, [
                "calibrate",
                "--config", str(cfg),
                "--output", str(tmp_path / "labels.json"),
                "--yes",
            ])

        assert result.exit_code == 0
        assert "Sampled 2 questions" in result.output

    def test_missing_config_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, [
            "calibrate", "--config", "/nonexistent/config.yaml", "--yes",
        ])
        assert result.exit_code == 1

    def test_serve_flag_launches_ui(self, tmp_path: Path) -> None:
        cfg, q = self._make_config_and_questions(tmp_path)

        mock_top_answers = {"q1": "B", "q2": "B"}

        runner = CliRunner(env={"TEST_KEY": "sk-test"})
        with (
            patch("theaios.trustgate.cli.sample_and_rank", return_value=mock_top_answers),
            patch("theaios.trustgate.serve.serve_calibration") as mock_serve,
        ):
            result = runner.invoke(main, [
                "calibrate",
                "--config", str(cfg),
                "--questions", str(q),
                "--serve", "--port", "9999",
                "--output", str(tmp_path / "labels.json"),
                "--yes",
            ])

        assert result.exit_code == 0
        mock_serve.assert_called_once()
        call_kwargs = mock_serve.call_args
        assert call_kwargs.kwargs["port"] == 9999
        assert call_kwargs.kwargs["output_file"] == str(tmp_path / "labels.json")


# ===========================================================================
# End-to-end: calibrate then certify
# ===========================================================================


class TestCalibrateAndCertifyFlow:
    def test_labels_file_format_works_with_certify(self, tmp_path: Path) -> None:
        """Verify the labels format produced by the review UI is accepted by certify."""
        # Simulate what the review UI saves
        labels = {"q1": "correct", "q2": "incorrect", "q3": "correct"}
        labels_file = tmp_path / "calibration_labels.json"
        labels_file.write_text(json.dumps(labels))

        # Verify load_ground_truth can read it
        from theaios.trustgate.certification import load_ground_truth

        loaded = load_ground_truth(str(labels_file))
        assert loaded == {"q1": "correct", "q2": "incorrect", "q3": "correct"}

    def test_serve_app_produces_valid_labels(self) -> None:
        """Verify the Flask app saves labels in the format certify expects."""
        from theaios.trustgate.serve import create_app

        questions = [
            Question(id="q1", text="What is 2+2?"),
            Question(id="q2", text="Capital of France?"),
        ]
        top_answers = {"q1": "4", "q2": "Paris"}

        app = create_app(questions, top_answers, output_file="/dev/null")
        client = app.test_client()

        # Submit reviews
        client.post("/api/review", json={"question_id": "q1", "judgment": True})
        client.post("/api/review", json={"question_id": "q2", "judgment": False})

        # Check results
        resp = client.get("/api/results")
        data = resp.get_json()
        assert data == {"q1": "correct", "q2": "incorrect"}

        # Check progress
        resp = client.get("/api/progress")
        progress = resp.get_json()
        assert progress["completed"] == 2
        assert progress["total"] == 2
        assert progress["pct"] == 100.0
