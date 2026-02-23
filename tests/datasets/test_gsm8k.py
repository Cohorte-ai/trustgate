"""Tests for the GSM8K dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from trustgate.datasets.gsm8k import _extract_answer, load_gsm8k


class TestExtractAnswer:
    def test_standard_format(self) -> None:
        assert _extract_answer("Some steps\n#### 42") == "42"

    def test_with_comma(self) -> None:
        assert _extract_answer("#### 1,234") == "1234"

    def test_no_delimiter(self) -> None:
        assert _extract_answer("No answer here") == ""

    def test_decimal(self) -> None:
        assert _extract_answer("#### 3.14") == "3.14"


class TestLoadGsm8k:
    def _mock_data(self, tmp_path: Path) -> Path:
        """Create a mock JSONL file."""
        rows = [
            {"question": f"Q{i}", "answer": f"Step 1\n#### {i * 10}"}
            for i in range(20)
        ]
        path = tmp_path / "gsm8k_test.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        return path

    def test_returns_questions(self, tmp_path: Path) -> None:
        mock_path = self._mock_data(tmp_path)
        with patch("trustgate.datasets.gsm8k._download_jsonl", return_value=mock_path):
            questions = load_gsm8k()

        assert len(questions) == 20
        assert questions[0].id == "gsm8k_0"
        assert questions[0].text == "Q0"

    def test_acceptable_answers(self, tmp_path: Path) -> None:
        mock_path = self._mock_data(tmp_path)
        with patch("trustgate.datasets.gsm8k._download_jsonl", return_value=mock_path):
            questions = load_gsm8k()

        assert questions[0].acceptable_answers == ["0"]
        assert questions[1].acceptable_answers == ["10"]

    def test_subsample(self, tmp_path: Path) -> None:
        mock_path = self._mock_data(tmp_path)
        with patch("trustgate.datasets.gsm8k._download_jsonl", return_value=mock_path):
            questions = load_gsm8k(n=5)

        assert len(questions) == 5

    def test_metadata(self, tmp_path: Path) -> None:
        mock_path = self._mock_data(tmp_path)
        with patch("trustgate.datasets.gsm8k._download_jsonl", return_value=mock_path):
            questions = load_gsm8k()

        assert questions[0].metadata.get("source") == "gsm8k"
