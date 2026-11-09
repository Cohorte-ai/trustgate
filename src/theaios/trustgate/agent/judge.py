"""LLM judges for criteria code cannot check, with a qualification lifecycle.

A judge is a classifier and is validated like one (Agent Eval Playbook, judge meta-eval):
one binary criterion per rubric, critique before verdict, UNKNOWN when the item cannot be
judged (never counted as a pass), and a gold set labeled by a person, independent of the
agent under test. Until a rubric's judge is qualified, its verdicts are advisory: reported,
never used to decide.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from theaios.trustgate.agent.stats import clopper_pearson_lower
from theaios.trustgate.agent.transcript import Transcript, transcript_from_dict

PROMPT_VERSION = "judge-v1"
Verdict = Literal["pass", "fail", "unknown"]

JUDGE_PROMPT = """You are grading one conversation between a user and an AI agent against \
ONE criterion. Grade only the criterion, nothing else.

CRITERION:
{rubric}

CONVERSATION (user turns, the agent's answers, and the tools it called):
{conversation}

Instructions:
1. Write a 1-3 sentence critique reasoning only against the criterion.
2. Then give a verdict: PASS, FAIL, or UNKNOWN. Use UNKNOWN only if the conversation cannot
   be judged against the criterion (missing context, malformed input).
Reply with JSON only: {{"critique": "...", "verdict": "PASS" | "FAIL" | "UNKNOWN"}}"""

# Defaults from the judge meta-eval how-to: kappa > 0.8 and TNR (recall on FAIL) >= 0.85,
# on at least 30 scored items (100 recommended), with a meaningful share of FAILs.
DEFAULT_MIN_KAPPA = 0.8
DEFAULT_MIN_TNR = 0.85
MIN_SCORED = 30
RECOMMENDED_SCORED = 100
FAMILIES = (
    "gpt",
    "o1",
    "o3",
    "o4",
    "claude",
    "gemini",
    "llama",
    "mistral",
    "qwen",
    "deepseek",
    "grok",
    "command",
)


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    rubrics: dict[str, str]
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "LLM_API_KEY"
    timeout_s: float = 60.0

    def rubric_fingerprint(self, rubric_id: str) -> str:
        """What a qualification is bound to: the model, the prompt version, the rubric."""
        blob = json.dumps([self.model, PROMPT_VERSION, JUDGE_PROMPT, self.rubrics[rubric_id]])
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class JudgeVerdict:
    rubric: str
    verdict: Verdict
    critique: str
    cost_usd: float = 0.0


def render_conversation(scenario_turns: Sequence[str], transcript: Transcript) -> str:
    lines: list[str] = []
    for i, user in enumerate(scenario_turns):
        lines.append(f"USER: {user}")
        if i < len(transcript.turns):
            turn = transcript.turns[i]
            for c in turn.tool_calls or ():
                args = "" if c.arguments is None else json.dumps(c.arguments, default=str)
                out = "" if c.output is None else f" -> {c.output[:500]}"
                lines.append(f"  [tool {c.name} ({c.status}) {args}{out}]")
            lines.append(f"AGENT: {turn.text if turn.text is not None else '(no text)'}")
    return "\n".join(lines)


def parse_verdict(text: str) -> tuple[Verdict, str]:
    """Read the judge's JSON reply. Anything unreadable is UNKNOWN, never PASS."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "")).strip().upper()
            critique = str(data.get("critique", "")).strip()
            if verdict in ("PASS", "FAIL", "UNKNOWN"):
                return verdict.lower(), critique  # type: ignore[return-value]
        except (json.JSONDecodeError, AttributeError):
            pass
    return "unknown", "unreadable judge reply"


class Judge:
    """Calls an OpenAI-compatible chat API at temperature 0."""

    PRICE_PER_TOKEN = {"input": 0.40 / 1e6, "output": 1.60 / 1e6}  # rough default for costing

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def grade(
        self, rubric_id: str, scenario_turns: Sequence[str], transcript: Transcript
    ) -> JudgeVerdict:
        import httpx

        prompt = JUDGE_PROMPT.format(
            rubric=self.config.rubrics[rubric_id],
            conversation=render_conversation(scenario_turns, transcript),
        )
        key = os.environ.get(self.config.api_key_env, "")
        try:
            r = httpx.post(
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": self.config.model,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.config.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            return JudgeVerdict(rubric_id, "unknown", f"judge error: {type(exc).__name__}")
        usage = data.get("usage", {})
        cost = (
            usage.get("prompt_tokens", 0) * self.PRICE_PER_TOKEN["input"]
            + usage.get("completion_tokens", 0) * self.PRICE_PER_TOKEN["output"]
        )
        verdict, critique = parse_verdict(data["choices"][0]["message"].get("content") or "")
        return JudgeVerdict(rubric_id, verdict, critique, cost)


# ---------------------------------------------------------------- qualification


@dataclass(frozen=True)
class GoldItem:
    rubric: str
    turns: tuple[str, ...]
    transcript: Transcript
    label: Literal["pass", "fail"]
    split: Literal["dev", "blind"] = "dev"
    id: str = ""


def load_gold(path: Path) -> list[GoldItem]:
    """JSONL, one item per line: {id, rubric, turns, transcript, label: pass|fail, split}."""
    items: list[GoldItem] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        label = str(raw.get("label", "")).lower()
        if label not in ("pass", "fail"):
            raise ValueError(f"{path}:{n}: label must be pass or fail")
        split = str(raw.get("split", "dev")).lower()
        if split not in ("dev", "blind"):
            raise ValueError(f"{path}:{n}: split must be dev or blind")
        items.append(
            GoldItem(
                rubric=str(raw["rubric"]),
                turns=tuple(str(t) for t in raw["turns"]),
                transcript=transcript_from_dict(raw["transcript"]),
                label=label,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
                id=str(raw.get("id", n)),
            )
        )
    return items


@dataclass
class JudgeMetrics:
    scored: int
    unknown: int
    fails_in_gold: int
    tpr: float  # recall on PASS
    tnr: float  # recall on FAIL: share of real failures the judge catches
    tnr_lower: float  # exact one-sided 95% lower bound on TNR
    false_accept_rate: float  # 1 - TNR: real failures the judge passes
    false_reject_rate: float  # 1 - TPR
    kappa: float


def judge_metrics(pairs: Sequence[tuple[str, Verdict]]) -> JudgeMetrics:
    """pairs: (human label, judge verdict). UNKNOWN verdicts are dropped and counted."""
    scored = [(h, j) for h, j in pairs if j != "unknown"]
    tp = sum(1 for h, j in scored if h == "pass" and j == "pass")
    fn = sum(1 for h, j in scored if h == "pass" and j == "fail")
    tn = sum(1 for h, j in scored if h == "fail" and j == "fail")
    fp = sum(1 for h, j in scored if h == "fail" and j == "pass")
    n = len(scored)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    if n:
        p_o = (tp + tn) / n
        p_e = ((tp + fn) / n) * ((tp + fp) / n) + ((tn + fp) / n) * ((tn + fn) / n)
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0
    else:
        kappa = 0.0
    return JudgeMetrics(
        scored=n,
        unknown=len(pairs) - n,
        fails_in_gold=tn + fp,
        tpr=tpr,
        tnr=tnr,
        tnr_lower=clopper_pearson_lower(tn, tn + fp),
        false_accept_rate=1 - tnr if tn + fp else 1.0,
        false_reject_rate=1 - tpr if tp + fn else 1.0,
        kappa=kappa,
    )


@dataclass
class JudgeCard:
    rubric: str
    rubric_text: str
    model: str
    prompt_version: str
    fingerprint: str
    gold_hash: str
    gated_on: str  # "blind" or "all"
    metrics: dict[str, Any]
    dev_metrics: dict[str, Any] | None
    qualified: bool
    reasons: list[str]
    warnings: list[str] = field(default_factory=list)
    disagreements: list[dict[str, str]] = field(default_factory=list)


def qualify_judge(
    judge: Judge,
    rubric_id: str,
    gold: list[GoldItem],
    gold_hash: str,
    min_kappa: float = DEFAULT_MIN_KAPPA,
    min_tnr: float = DEFAULT_MIN_TNR,
    target_model: str | None = None,
) -> JudgeCard:
    items = [g for g in gold if g.rubric == rubric_id]
    if not items:
        raise ValueError(f"no gold items for rubric '{rubric_id}'")
    results = [(g, judge.grade(rubric_id, g.turns, g.transcript)) for g in items]
    blind = [(g.label, v.verdict) for g, v in results if g.split == "blind"]
    dev = [(g.label, v.verdict) for g, v in results if g.split == "dev"]
    gated_pairs, gated_on = (blind, "blind") if blind else (dev, "all")
    m = judge_metrics(gated_pairs)
    reasons: list[str] = []
    warnings: list[str] = []
    if m.scored < MIN_SCORED:
        reasons.append(f"only {m.scored} scored gold items; at least {MIN_SCORED} needed")
    elif m.scored < RECOMMENDED_SCORED:
        warnings.append(f"{m.scored} scored gold items; about {RECOMMENDED_SCORED} recommended")
    if m.fails_in_gold == 0:
        reasons.append(
            "no FAIL items in the gold set: the judge's ability to catch failures is unmeasured"
        )
    elif m.fails_in_gold / max(1, m.scored) < 0.2:
        warnings.append(
            f"FAILs are {m.fails_in_gold / max(1, m.scored):.0%} of the gold set; "
            "25-40% recommended"
        )
    if m.kappa <= min_kappa:
        reasons.append(f"kappa {m.kappa:.2f} not above {min_kappa}")
    if m.tnr < min_tnr:
        reasons.append(
            f"TNR {m.tnr:.0%} under {min_tnr:.0%}: the judge passes "
            f"{m.false_accept_rate:.0%} of real failures"
        )
    if not blind:
        warnings.append("no blind split: metrics are on the items used to tune the rubric")
    if target_model and model_family(target_model) == model_family(judge.config.model):
        warnings.append(
            f"judge and agent share a model family ({model_family(target_model)}); "
            "self-preference can inflate PASS verdicts"
        )
    disagreements = [
        {"id": g.id, "human": g.label, "judge": v.verdict, "critique": v.critique[:300]}
        for g, v in results
        if v.verdict != "unknown" and v.verdict != g.label
    ]
    return JudgeCard(
        rubric=rubric_id,
        rubric_text=judge.config.rubrics[rubric_id],
        model=judge.config.model,
        prompt_version=PROMPT_VERSION,
        fingerprint=judge.config.rubric_fingerprint(rubric_id),
        gold_hash=gold_hash,
        gated_on=gated_on,
        metrics=asdict(m),
        dev_metrics=asdict(judge_metrics(dev)) if blind and dev else None,
        qualified=not reasons,
        reasons=reasons,
        warnings=warnings,
        disagreements=disagreements[:50],
    )


def model_family(model: str) -> str:
    name = model.lower().split("/")[-1]
    for fam in FAMILIES:
        if name.startswith(fam):
            return "openai" if fam in ("gpt", "o1", "o3", "o4") else fam
    return name.split("-")[0]


def card_path(base: Path, rubric_id: str) -> Path:
    return base / ".trustgate" / "judges" / f"{rubric_id}.json"


def save_card(card: JudgeCard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(card), indent=1), encoding="utf-8")


def load_card(path: Path) -> JudgeCard | None:
    if not path.is_file():
        return None
    return JudgeCard(**json.loads(path.read_text(encoding="utf-8")))


def judge_status(config: JudgeConfig, rubric_id: str, base: Path) -> tuple[bool, str]:
    """(qualified, reason). Qualified means: a card exists, it passed, and it was earned by
    this exact model, prompt version and rubric text."""
    card = load_card(card_path(base, rubric_id))
    if card is None:
        return False, "not qualified yet (advisory)"
    if card.fingerprint != config.rubric_fingerprint(rubric_id):
        return False, "qualification is stale: model, prompt or rubric changed (advisory)"
    if not card.qualified:
        return False, "failed qualification (advisory): " + "; ".join(card.reasons)
    return True, (
        f"qualified: kappa {card.metrics['kappa']:.2f}, "
        f"TNR {card.metrics['tnr']:.0%} on {card.metrics['scored']} items"
    )
