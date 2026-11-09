"""``qualify.yaml``: the target, the scenarios, and the rules of the qualification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from theaios.trustgate.agent.judge import JudgeConfig
from theaios.trustgate.agent.scenario import (
    SUITES,
    ScenarioError,
    _yaml_position,
    unknown_keys_message,
)

_KEYS = {
    "name",
    "target",
    "scenarios",
    "k",
    "thresholds",
    "budget_usd",
    "concurrency",
    "envelope",
    "output",
    "safety_max_failure_share",
    "judge",
}


@dataclass
class QualifyConfig:
    base_dir: Path
    name: str
    target: dict[str, Any]
    scenario_globs: list[str]
    k: int = 5
    thresholds: dict[str, float] = field(default_factory=dict)
    budget_usd: float | None = None
    concurrency: int = 1
    model: str | None = None
    envelope_files: list[str] = field(default_factory=list)
    envelope_extra: dict[str, Any] = field(default_factory=dict)
    output_dir: str = ".trustgate/qualification"
    safety_max_failure_share: float | None = None
    judge: JudgeConfig | None = None

    def scenario_files(self) -> list[Path]:
        files: list[Path] = []
        for pattern in self.scenario_globs:
            files += [p for p in sorted(self.base_dir.glob(pattern)) if p.is_file()]
        return files

    @property
    def report_path(self) -> Path:
        return self.base_dir / self.output_dir / "report.json"

    @property
    def card_path(self) -> Path:
        return self.base_dir / self.output_dir / "QUALIFICATION.md"


def load_qualify_config(path: str | Path) -> QualifyConfig:
    p = Path(path).resolve()
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{p}: invalid YAML ({_yaml_position(exc)})") from None
    if not isinstance(raw, dict):
        raise ScenarioError(f"{p}: expected a mapping")
    unknown = set(raw) - _KEYS
    if unknown:
        raise ScenarioError(f"{p}: {unknown_keys_message(unknown, _KEYS)}")
    target = raw.get("target")
    if not isinstance(target, dict) or target.get("type") not in ("http", "python"):
        raise ScenarioError(f"{p}: target.type must be 'http' or 'python'")
    scen = raw.get("scenarios")
    globs = [scen] if isinstance(scen, str) else scen
    if not isinstance(globs, list) or not globs:
        raise ScenarioError(f"{p}: 'scenarios' must list one or more file globs")
    thresholds = raw.get("thresholds") or {}
    if not isinstance(thresholds, dict) or not set(thresholds) <= set(SUITES) - {"safety"}:
        raise ScenarioError(
            f"{p}: thresholds apply to nominal and escalation only (safety is a veto)"
        )
    for suite, t in thresholds.items():
        if not isinstance(t, (int, float)) or not 0 < t < 1:
            raise ScenarioError(f"{p}: threshold for {suite} must be in (0, 1)")
    env = raw.get("envelope") or {}
    k = raw.get("k", 5)
    if not isinstance(k, int) or k < 1:
        raise ScenarioError(f"{p}: k must be a positive integer")
    budget = raw.get("budget_usd")
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget < 0
    ):
        raise ScenarioError(f"{p}: budget_usd must be a number >= 0 (USD, no currency sign)")
    judge_raw = raw.get("judge")
    judge = None
    if judge_raw is not None:
        rubrics = judge_raw.get("rubrics") if isinstance(judge_raw, dict) else None
        if (
            not isinstance(judge_raw, dict)
            or not judge_raw.get("model")
            or not isinstance(rubrics, dict)
            or not rubrics
        ):
            raise ScenarioError(f"{p}: judge needs a model and at least one rubric")
        judge = JudgeConfig(
            model=str(judge_raw["model"]),
            rubrics={str(k): str(v).strip() for k, v in rubrics.items()},
            base_url=str(judge_raw.get("base_url", "https://api.openai.com/v1")),
            api_key_env=str(judge_raw.get("api_key_env", "LLM_API_KEY")),
        )
    cap = raw.get("safety_max_failure_share")
    if cap is not None and (not isinstance(cap, (int, float)) or not 0 < cap < 1):
        raise ScenarioError(f"{p}: safety_max_failure_share must be in (0, 1)")
    return QualifyConfig(
        base_dir=p.parent,
        name=str(raw.get("name") or p.parent.name),
        target=target,
        scenario_globs=[str(g) for g in globs],
        k=k,
        thresholds={str(s): float(t) for s, t in thresholds.items()},
        budget_usd=float(budget) if budget is not None else None,
        concurrency=int(raw.get("concurrency", 1)),
        model=str(env["model"]) if env.get("model") is not None else None,
        envelope_files=[str(g) for g in env.get("files", [])],
        envelope_extra=dict(env.get("extra") or {}),
        output_dir=str(raw.get("output") or ".trustgate/qualification"),
        safety_max_failure_share=float(cap) if cap is not None else None,
        judge=judge,
    )
