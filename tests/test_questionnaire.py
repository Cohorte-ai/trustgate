"""Tests for the exportable HTML questionnaire."""

from __future__ import annotations

import json
from pathlib import Path

from theaios.trustgate.questionnaire import generate_questionnaire
from theaios.trustgate.types import Question


def _questions() -> list[Question]:
    return [
        Question(id="q1", text="What is 2+2?"),
        Question(id="q2", text="Capital of France?"),
    ]


def _profiles() -> dict[str, list[tuple[str, float]]]:
    return {
        "q1": [("4", 0.8), ("5", 0.1), ("3", 0.1)],
        "q2": [("Paris", 0.7), ("London", 0.2), ("Berlin", 0.1)],
    }


class TestGenerateQuestionnaire:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        out = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q.html"),
        )
        assert Path(out).exists()
        html = Path(out).read_text()
        assert "<!DOCTYPE html>" in html
        assert "TrustGate Calibration" in html

    def test_embeds_all_questions(self, tmp_path: Path) -> None:
        out = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q.html"),
        )
        html = Path(out).read_text()
        assert "What is 2+2?" in html
        assert "Capital of France?" in html

    def test_embeds_all_answers(self, tmp_path: Path) -> None:
        out = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q.html"),
        )
        html = Path(out).read_text()
        for ans in ["4", "5", "3", "Paris", "London", "Berlin"]:
            assert f'"answer": "{ans}"' in html or f'"answer":"{ans}"' in html

    def test_no_frequencies_in_output(self, tmp_path: Path) -> None:
        out = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q.html"),
        )
        html = Path(out).read_text()
        assert "frequency" not in html
        assert "0.8" not in html
        assert "0.7" not in html

    def test_answers_are_shuffled(self, tmp_path: Path) -> None:
        out = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q.html"),
        )
        html = Path(out).read_text()
        # Extract the embedded JSON
        start = html.index("const ITEMS = ") + len("const ITEMS = ")
        end = html.index(";\nconst labels")
        items = json.loads(html[start:end])
        # Check that answers are present (order may differ from original)
        q1_answers = {a["answer"] for a in items[0]["answers"]}
        assert q1_answers == {"4", "5", "3"}

    def test_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        out1 = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q1.html"), seed=42,
        )
        out2 = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q2.html"), seed=42,
        )
        assert Path(out1).read_text() == Path(out2).read_text()

    def test_different_seed_shuffles_differently(self, tmp_path: Path) -> None:
        out1 = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q1.html"), seed=1,
        )
        out2 = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q2.html"), seed=99,
        )
        # Different seeds should produce different order (with high probability)
        # We just check the files differ
        assert Path(out1).read_text() != Path(out2).read_text()

    def test_skips_questions_without_profile(self, tmp_path: Path) -> None:
        questions = _questions() + [Question(id="q3", text="No profile")]
        out = generate_questionnaire(
            questions, _profiles(),
            output_path=str(tmp_path / "q.html"),
        )
        html = Path(out).read_text()
        assert "No profile" not in html

    def test_download_produces_valid_labels_format(self, tmp_path: Path) -> None:
        """The JS download produces {qid: answer} — same format as certify --ground-truth."""
        out = generate_questionnaire(
            _questions(), _profiles(),
            output_path=str(tmp_path / "q.html"),
        )
        html = Path(out).read_text()
        # The JS stores labels as: labels[item.question_id] = answer
        # and downloads JSON.stringify(labels)
        # Verify the JS logic is correct by checking it exists
        assert "labels[item.question_id] = answer" in html
        assert "JSON.stringify(labels, null, 2)" in html
        assert 'a.download = \'labels.json\'' in html
