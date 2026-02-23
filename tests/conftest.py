"""Shared fixtures for TrustGate tests."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from trustgate.types import (
    CalibrationConfig,
    CanonConfig,
    EndpointConfig,
    QuestionsConfig,
    SamplingConfig,
    TrustGateConfig,
)


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def sample_config() -> TrustGateConfig:
    """A valid TrustGateConfig for testing."""
    return TrustGateConfig(
        endpoint=EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_API_KEY",
        ),
        sampling=SamplingConfig(k_fixed=5, k_max=20),
        canonicalization=CanonConfig(type="mcq"),
        calibration=CalibrationConfig(n_cal=10, n_test=10),
        questions=QuestionsConfig(file="questions.csv"),
    )


@pytest.fixture()
def sample_yaml(tmp_path: Path) -> Path:
    """Write a valid trustgate.yaml and return its path."""
    content = textwrap.dedent("""\
        endpoint:
          url: "https://api.openai.com/v1/chat/completions"
          model: "gpt-4.1-mini"
          temperature: 0.7
          api_key_env: "OPENAI_API_KEY"

        sampling:
          k_max: 20
          k_fixed: 10
          sequential_stopping: true
          delta: 0.05

        canonicalization:
          type: "mcq"

        calibration:
          alpha_values: [0.01, 0.05, 0.10]
          n_cal: 250
          n_test: 250

        questions:
          file: "questions.csv"
    """)
    p = tmp_path / "trustgate.yaml"
    p.write_text(content)
    return p


@pytest.fixture()
def sample_questions_csv(tmp_path: Path) -> Path:
    """Write a sample questions CSV and return its path."""
    content = textwrap.dedent("""\
        id,question,acceptable_answers
        q001,"What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6","B"
        q002,"Capital of France? (A) London (B) Paris (C) Berlin (D) Madrid","B"
        q003,"Largest planet? (A) Earth (B) Mars (C) Jupiter (D) Venus","C"
    """)
    p = tmp_path / "questions.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def sample_questions_json(tmp_path: Path) -> Path:
    """Write a sample questions JSON and return its path."""
    data = [
        {"id": "q001", "question": "What is 2+2?", "acceptable_answers": ["4"]},
        {"id": "q002", "question": "Capital of France?", "acceptable_answers": ["Paris"]},
        {"id": "q003", "question": "Largest planet?", "acceptable_answers": ["Jupiter"]},
    ]
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(data, indent=2))
    return p
