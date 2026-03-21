"""Tests for the local calibration UI server (ranked-answer selection)."""

from __future__ import annotations

import json

import pytest

from theaios.trustgate.serve import create_app
from theaios.trustgate.types import Question


def _questions() -> list[Question]:
    return [
        Question(id="q1", text="What is 2+2?"),
        Question(id="q2", text="Capital of France?"),
        Question(id="q3", text="Largest planet?"),
    ]


def _profiles() -> dict[str, list[tuple[str, float]]]:
    """Ranked canonical answer profiles for each question."""
    return {
        "q1": [("4", 0.8), ("5", 0.1), ("3", 0.1)],
        "q2": [("Paris", 0.7), ("London", 0.2), ("Berlin", 0.1)],
        "q3": [("Jupiter", 0.9), ("Saturn", 0.1)],
    }


@pytest.fixture()
def client() -> object:
    app = create_app(_questions(), _profiles(), output_file="/dev/null")
    app.config["TESTING"] = True  # type: ignore[union-attr]
    with app.test_client() as c:  # type: ignore[union-attr]
        yield c


class TestReviewerUI:
    def test_index_returns_html(self, client: object) -> None:
        resp = client.get("/")  # type: ignore[union-attr]
        assert resp.status_code == 200  # type: ignore[union-attr]
        data = resp.data.decode()  # type: ignore[union-attr]
        assert "TrustGate Calibration" in data
        assert "Which answer is acceptable?" in data


class TestAdminUI:
    def test_admin_returns_html(self, client: object) -> None:
        resp = client.get("/admin")  # type: ignore[union-attr]
        assert resp.status_code == 200  # type: ignore[union-attr]
        assert "Calibration Admin" in resp.data.decode()  # type: ignore[union-attr]


class TestAPINext:
    def test_returns_question_with_shuffled_answers(self, client: object) -> None:
        resp = client.get("/api/next")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["question_id"] == "q1"
        assert data["question"] == "What is 2+2?"
        assert data["done"] is False
        # Answers present but no frequency or rank info
        answers = data["answers"]
        assert len(answers) == 3
        answer_texts = {a["answer"] for a in answers}
        assert answer_texts == {"4", "5", "3"}
        # No frequency or rank keys exposed to reviewer
        for a in answers:
            assert "frequency" not in a
            assert "rank" not in a


class TestAPIReview:
    def test_select_answer(self, client: object) -> None:
        resp = client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "selected_answer": "4"}),
            content_type="application/json",
        )
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["ok"] is True

    def test_select_none(self, client: object) -> None:
        resp = client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "selected_answer": None}),
            content_type="application/json",
        )
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["ok"] is True

    def test_overwrite_selection(self, client: object) -> None:
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "selected_answer": "4"}),
            content_type="application/json",
        )
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "selected_answer": "5"}),
            content_type="application/json",
        )
        resp = client.get("/api/results")  # type: ignore[union-attr]
        results = json.loads(resp.data)  # type: ignore[union-attr]
        assert results["q1"]["answer"] == "5"
        assert results["q1"]["rank"] == 2


class TestAPIProgress:
    def test_initial_progress(self, client: object) -> None:
        resp = client.get("/api/progress")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["completed"] == 0
        assert data["total"] == 3
        assert data["pct"] == 0.0

    def test_progress_after_review(self, client: object) -> None:
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "selected_answer": "4"}),
            content_type="application/json",
        )
        resp = client.get("/api/progress")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["completed"] == 1
        assert data["pct"] == pytest.approx(100 / 3)


class TestAPIResults:
    def test_empty_results(self, client: object) -> None:
        resp = client.get("/api/results")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data == {}

    def test_results_with_ranks(self, client: object) -> None:
        # Select top answer (rank 1)
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "selected_answer": "4"}),
            content_type="application/json",
        )
        # Select second answer (rank 2)
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q2", "selected_answer": "London"}),
            content_type="application/json",
        )
        # Select none
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q3", "selected_answer": None}),
            content_type="application/json",
        )
        resp = client.get("/api/results")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["q1"] == {"answer": "4", "rank": 1}
        assert data["q2"] == {"answer": "London", "rank": 2}
        assert data["q3"] == {"answer": None, "rank": None}


class TestAPIExport:
    def test_export_returns_labels_for_certify(self, client: object) -> None:
        """Export format must be {qid: answer} — compatible with certify --ground-truth."""
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "selected_answer": "4"}),
            content_type="application/json",
        )
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q2", "selected_answer": None}),
            content_type="application/json",
        )
        resp = client.get("/api/export")  # type: ignore[union-attr]
        assert resp.status_code == 200  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        # Only non-null answers exported (compatible with load_ground_truth)
        assert data == {"q1": "4"}
        assert "q2" not in data  # null = unsolvable, excluded


class TestNextAfterReviews:
    def test_skips_reviewed(self, client: object) -> None:
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "selected_answer": "4"}),
            content_type="application/json",
        )
        resp = client.get("/api/next")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["question_id"] == "q2"

    def test_done_after_all_reviewed(self, client: object) -> None:
        for qid, ans in [("q1", "4"), ("q2", "Paris"), ("q3", "Jupiter")]:
            client.post(  # type: ignore[union-attr]
                "/api/review",
                data=json.dumps({"question_id": qid, "selected_answer": ans}),
                content_type="application/json",
            )
        resp = client.get("/api/next")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["done"] is True
