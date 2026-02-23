"""Tests for the local calibration UI server."""

from __future__ import annotations

import json

import pytest

from trustgate.serve import create_app
from trustgate.types import Question


def _questions() -> list[Question]:
    return [
        Question(id="q1", text="What is 2+2?"),
        Question(id="q2", text="Capital of France?"),
        Question(id="q3", text="Largest planet?"),
    ]


def _answers() -> dict[str, str]:
    return {"q1": "4", "q2": "Paris", "q3": "Jupiter"}


@pytest.fixture()
def client() -> object:
    app = create_app(_questions(), _answers(), output_file="/dev/null")
    app.config["TESTING"] = True  # type: ignore[union-attr]
    with app.test_client() as c:  # type: ignore[union-attr]
        yield c


class TestReviewerUI:
    def test_index_returns_html(self, client: object) -> None:
        resp = client.get("/")  # type: ignore[union-attr]
        assert resp.status_code == 200  # type: ignore[union-attr]
        data = resp.data.decode()  # type: ignore[union-attr]
        assert "TrustGate Calibration" in data
        assert "Correct" in data
        assert "Incorrect" in data


class TestAdminUI:
    def test_admin_returns_html(self, client: object) -> None:
        resp = client.get("/admin")  # type: ignore[union-attr]
        assert resp.status_code == 200  # type: ignore[union-attr]
        assert "Calibration Admin" in resp.data.decode()  # type: ignore[union-attr]


class TestAPINext:
    def test_returns_first_question(self, client: object) -> None:
        resp = client.get("/api/next")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["question_id"] == "q1"
        assert data["question"] == "What is 2+2?"
        assert data["answer"] == "4"
        assert data["done"] is False


class TestAPIReview:
    def test_submit_correct(self, client: object) -> None:
        resp = client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "judgment": True}),
            content_type="application/json",
        )
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["ok"] is True

    def test_overwrite_judgment(self, client: object) -> None:
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "judgment": True}),
            content_type="application/json",
        )
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "judgment": False}),
            content_type="application/json",
        )
        resp = client.get("/api/results")  # type: ignore[union-attr]
        results = json.loads(resp.data)  # type: ignore[union-attr]
        assert results["q1"] == "incorrect"


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
            data=json.dumps({"question_id": "q1", "judgment": True}),
            content_type="application/json",
        )
        resp = client.get("/api/progress")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["completed"] == 1
        assert data["pct"] == pytest.approx(100 / 3)

    def test_full_progress(self, client: object) -> None:
        for qid in ["q1", "q2", "q3"]:
            client.post(  # type: ignore[union-attr]
                "/api/review",
                data=json.dumps({"question_id": qid, "judgment": True}),
                content_type="application/json",
            )
        resp = client.get("/api/progress")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["completed"] == 3
        assert data["pct"] == 100.0


class TestAPIResults:
    def test_empty_results(self, client: object) -> None:
        resp = client.get("/api/results")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data == {}

    def test_results_after_reviews(self, client: object) -> None:
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "judgment": True}),
            content_type="application/json",
        )
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q2", "judgment": False}),
            content_type="application/json",
        )
        resp = client.get("/api/results")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["q1"] == "correct"
        assert data["q2"] == "incorrect"


class TestAPIExport:
    def test_export_returns_json(self, client: object) -> None:
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "judgment": True}),
            content_type="application/json",
        )
        resp = client.get("/api/export")  # type: ignore[union-attr]
        assert resp.status_code == 200  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["q1"] == "correct"


class TestNextAfterReviews:
    def test_skips_reviewed(self, client: object) -> None:
        client.post(  # type: ignore[union-attr]
            "/api/review",
            data=json.dumps({"question_id": "q1", "judgment": True}),
            content_type="application/json",
        )
        resp = client.get("/api/next")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["question_id"] == "q2"

    def test_done_after_all_reviewed(self, client: object) -> None:
        for qid in ["q1", "q2", "q3"]:
            client.post(  # type: ignore[union-attr]
                "/api/review",
                data=json.dumps({"question_id": qid, "judgment": True}),
                content_type="application/json",
            )
        resp = client.get("/api/next")  # type: ignore[union-attr]
        data = json.loads(resp.data)  # type: ignore[union-attr]
        assert data["done"] is True
