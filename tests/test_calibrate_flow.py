"""Tests for the calibrate command and sample_and_profile pipeline."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from theaios.trustgate.certification import sample_and_profile, sample_and_rank
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
# sample_and_profile / sample_and_rank
# ===========================================================================


class TestSampleAndProfile:
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

    def _mock_responses(self) -> dict[str, list[SampleResponse]]:
        return {
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

    def test_returns_full_profiles(self) -> None:
        config = self._make_config()
        questions = self._make_questions()

        with patch("theaios.trustgate.certification.Sampler") as MockSampler:
            instance = MockSampler.return_value
            instance.k = 3
            instance.sample_all = AsyncMock(return_value=self._mock_responses())

            profiles = sample_and_profile(config, questions)

        # q1: all 3 said B → B at 100%
        assert profiles["q1"][0][0] == "B"
        assert profiles["q1"][0][1] == pytest.approx(1.0)

        # q2: 2 said B, 1 said A → B at ~67%, A at ~33%
        assert profiles["q2"][0][0] == "B"
        assert profiles["q2"][0][1] == pytest.approx(2 / 3)
        assert profiles["q2"][1][0] == "A"
        assert profiles["q2"][1][1] == pytest.approx(1 / 3)

    def test_sample_and_rank_returns_top_answers(self) -> None:
        config = self._make_config()
        questions = self._make_questions()

        with patch("theaios.trustgate.certification.Sampler") as MockSampler:
            instance = MockSampler.return_value
            instance.k = 3
            instance.sample_all = AsyncMock(return_value=self._mock_responses())

            top = sample_and_rank(config, questions)

        assert top["q1"] == "B"
        assert top["q2"] == "B"

    def test_split_answers_profile(self) -> None:
        config = self._make_config()
        questions = [Question(id="q1", text="Hard? (A) X (B) Y (C) Z")]
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

            profiles = sample_and_profile(config, questions)

        # All equal frequency — profile should have 3 entries
        assert len(profiles["q1"]) == 3
        freqs = [f for _, f in profiles["q1"]]
        assert all(f == pytest.approx(1 / 3) for f in freqs)

    def test_raises_on_invalid_config(self) -> None:
        config = TrustGateConfig(endpoint=EndpointConfig(url="not-a-url"))
        with pytest.raises(Exception, match="Invalid configuration"):
            sample_and_profile(config, [])


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

    def test_without_serve_saves_profiles(self, tmp_path: Path) -> None:
        cfg, q = self._make_config_and_questions(tmp_path)
        mock_profiles = {
            "q1": [("B", 1.0)],
            "q2": [("B", 0.67), ("A", 0.33)],
        }

        runner = CliRunner(env={"TEST_KEY": "sk-test"})
        with patch("theaios.trustgate.cli.sample_and_profile", return_value=mock_profiles):
            result = runner.invoke(main, [
                "calibrate",
                "--config", str(cfg),
                "--questions", str(q),
                "--output", str(tmp_path / "labels.json"),
                "--yes",
            ])

        assert result.exit_code == 0
        assert "Profiled 2 questions" in result.output

        # Should save a profiles JSON
        profiles_file = tmp_path / "labels_profiles.json"
        assert profiles_file.exists()
        data = json.loads(profiles_file.read_text())
        assert data["q1"]["ranked_answers"][0]["answer"] == "B"

    def test_questions_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg, _ = self._make_config_and_questions(tmp_path)
        monkeypatch.chdir(tmp_path)
        mock_profiles = {"q1": [("B", 1.0)], "q2": [("B", 0.67)]}

        runner = CliRunner(env={"TEST_KEY": "sk-test"})
        with patch("theaios.trustgate.cli.sample_and_profile", return_value=mock_profiles):
            result = runner.invoke(main, [
                "calibrate",
                "--config", str(cfg),
                "--output", str(tmp_path / "labels.json"),
                "--yes",
            ])

        assert result.exit_code == 0

    def test_serve_flag_launches_ui(self, tmp_path: Path) -> None:
        cfg, q = self._make_config_and_questions(tmp_path)
        mock_profiles = {"q1": [("B", 1.0)], "q2": [("B", 0.67)]}

        runner = CliRunner(env={"TEST_KEY": "sk-test"})
        with (
            patch("theaios.trustgate.cli.sample_and_profile", return_value=mock_profiles),
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
        assert call_kwargs.kwargs["profiles"] == mock_profiles


# ===========================================================================
# End-to-end: labels format compatibility
# ===========================================================================


class TestLabelsCompatibility:
    def test_labels_from_ui_work_with_certify(self, tmp_path: Path) -> None:
        """Labels saved by the UI ({qid: answer}) work with certify --ground-truth."""
        labels = {"q1": "B", "q2": "A", "q3": "C"}
        labels_file = tmp_path / "calibration_labels.json"
        labels_file.write_text(json.dumps(labels))

        from theaios.trustgate.certification import load_ground_truth

        loaded = load_ground_truth(str(labels_file))
        assert loaded == {"q1": "B", "q2": "A", "q3": "C"}

    def test_nonconformity_scores_from_labels(self) -> None:
        """Verify that labels produce correct nonconformity scores."""
        from theaios.trustgate.calibration import compute_nonconformity_score

        profile = [("B", 0.8), ("A", 0.1), ("C", 0.1)]

        # Human selected B (rank 1)
        assert compute_nonconformity_score(profile, "B") == 1

        # Human selected A (rank 2)
        assert compute_nonconformity_score(profile, "A") == 2

        # Human selected C (rank 3)
        assert compute_nonconformity_score(profile, "C") == 3

        # Human selected "none" → answer not in profile → ∞
        assert compute_nonconformity_score(profile, "D") == float("inf")

    def test_serve_app_export_compatible_with_certify(self) -> None:
        """The Flask app's export format is directly usable as ground truth."""
        from theaios.trustgate.serve import create_app

        questions = [
            Question(id="q1", text="What is 2+2?"),
            Question(id="q2", text="Capital of France?"),
        ]
        profiles = {
            "q1": [("4", 0.8), ("5", 0.1), ("3", 0.1)],
            "q2": [("Paris", 0.7), ("London", 0.2), ("Berlin", 0.1)],
        }

        app = create_app(questions, profiles, output_file="/dev/null")
        client = app.test_client()

        # Human picks rank-1 for q1, rank-2 for q2
        client.post("/api/review", json={"question_id": "q1", "selected_answer": "4"})
        client.post("/api/review", json={"question_id": "q2", "selected_answer": "London"})

        # Export
        resp = client.get("/api/export")
        data = resp.get_json()

        # Format: {qid: canonical_answer} — ready for certify --ground-truth
        assert data == {"q1": "4", "q2": "London"}

        # Verify these produce correct nonconformity scores
        from theaios.trustgate.calibration import compute_nonconformity_score

        assert compute_nonconformity_score(profiles["q1"], data["q1"]) == 1  # rank 1
        assert compute_nonconformity_score(profiles["q2"], data["q2"]) == 2  # rank 2
