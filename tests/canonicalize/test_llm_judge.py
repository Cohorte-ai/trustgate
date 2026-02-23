"""Tests for the LLM judge canonicalizer."""

from __future__ import annotations

import httpx
import pytest
import respx

from trustgate.canonicalize.llm_judge import LLMJudgeCanonicalizer
from trustgate.types import EndpointConfig

JUDGE_URL = "https://api.openai.com/v1/chat/completions"


def _judge_config() -> EndpointConfig:
    return EndpointConfig(
        url=JUDGE_URL,
        model="gpt-4.1-mini",
        api_key_env="TEST_JUDGE_KEY",
        provider="openai",
    )


def _judge_response(text: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture(autouse=True)
def _set_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "sk-judge-key")


class TestLLMJudge:
    def test_requires_config(self) -> None:
        with pytest.raises(ValueError, match="judge_config"):
            LLMJudgeCanonicalizer()

    @respx.mock
    def test_correct_answer(self) -> None:
        respx.post(JUDGE_URL).respond(json=_judge_response("correct"))
        canon = LLMJudgeCanonicalizer(judge_config=_judge_config(), retries=0)
        assert canon.canonicalize("What is 2+2?", "4") == "correct"

    @respx.mock
    def test_incorrect_answer(self) -> None:
        respx.post(JUDGE_URL).respond(json=_judge_response("incorrect"))
        canon = LLMJudgeCanonicalizer(judge_config=_judge_config(), retries=0)
        assert canon.canonicalize("What is 2+2?", "5") == "incorrect"

    @respx.mock
    def test_verbose_correct(self) -> None:
        respx.post(JUDGE_URL).respond(
            json=_judge_response("The answer is correct.")
        )
        canon = LLMJudgeCanonicalizer(judge_config=_judge_config(), retries=0)
        assert canon.canonicalize("Q?", "A") == "correct"

    @respx.mock
    def test_verbose_incorrect(self) -> None:
        respx.post(JUDGE_URL).respond(
            json=_judge_response("This is incorrect because...")
        )
        canon = LLMJudgeCanonicalizer(judge_config=_judge_config(), retries=0)
        assert canon.canonicalize("Q?", "A") == "incorrect"

    @respx.mock
    def test_ambiguous_defaults_incorrect(self) -> None:
        respx.post(JUDGE_URL).respond(
            json=_judge_response("I'm not sure about this one")
        )
        canon = LLMJudgeCanonicalizer(judge_config=_judge_config(), retries=0)
        assert canon.canonicalize("Q?", "A") == "incorrect"

    @respx.mock
    def test_empty_answer_returns_incorrect(self) -> None:
        canon = LLMJudgeCanonicalizer(judge_config=_judge_config(), retries=0)
        assert canon.canonicalize("Q?", "") == "incorrect"

    @respx.mock
    def test_retries_on_failure(self) -> None:
        route = respx.post(JUDGE_URL).mock(
            side_effect=[
                httpx.Response(500, text="Error"),
                httpx.Response(200, json=_judge_response("correct")),
            ]
        )
        canon = LLMJudgeCanonicalizer(judge_config=_judge_config(), retries=1)
        assert canon.canonicalize("Q?", "A") == "correct"
        assert route.call_count == 2
