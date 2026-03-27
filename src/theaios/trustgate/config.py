"""YAML configuration loader, validation, and question loading."""

from __future__ import annotations

import csv
import json
import os
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import yaml

from theaios.trustgate.types import (
    VALID_CANON_TYPES,
    CalibrationConfig,
    CanonConfig,
    EndpointConfig,
    Question,
    QuestionsConfig,
    SamplingConfig,
    ThresholdsConfig,
    TrustGateConfig,
)


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Invalid configuration:\n  " + "\n  ".join(errors))


def load_config(
    path: str = "trustgate.yaml",
    overrides: dict[str, object] | None = None,
) -> TrustGateConfig:
    """Load YAML config, apply overrides, validate, return typed config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(["Config file must be a YAML mapping"])

    if overrides:
        _apply_overrides(raw, overrides)

    config = _parse_config(raw)

    errors = validate_config(config)
    if errors:
        raise ConfigError(errors)

    return config


def _apply_overrides(raw: dict[str, object], overrides: dict[str, object]) -> None:
    """Apply dot-notation overrides to the raw config dict."""
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target: dict[str, object] = raw
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = value


def _parse_config(raw: dict[str, object]) -> TrustGateConfig:
    """Parse raw YAML dict into typed TrustGateConfig."""
    endpoint_raw = raw.get("endpoint")
    if not isinstance(endpoint_raw, dict):
        raise ConfigError(["'endpoint' section is required and must be a mapping"])

    # Temperature: None means the endpoint controls its own randomness
    temp_raw = endpoint_raw.get("temperature", 0.7)
    temperature = None if temp_raw is None else float(temp_raw)

    # Headers for generic endpoints (support ${VAR} expansion at request time)
    headers_raw = endpoint_raw.get("headers")
    headers = dict(headers_raw) if isinstance(headers_raw, dict) else {}

    # Request template and response path for generic endpoints
    request_template = endpoint_raw.get("request_template")
    response_path = str(endpoint_raw.get("response_path", ""))

    # Per-request cost for pre-flight estimation
    cost_raw = endpoint_raw.get("cost_per_request")
    cost_per_request = float(cost_raw) if cost_raw is not None else None

    endpoint = EndpointConfig(
        url=str(endpoint_raw.get("url", "")),
        model=str(endpoint_raw.get("model", "")),
        temperature=temperature,
        api_key_env=str(endpoint_raw.get("api_key_env", "")),
        max_tokens=int(endpoint_raw.get("max_tokens", 4096)),
        provider=str(endpoint_raw.get("provider", "")),
        headers=headers,
        request_template=request_template,
        response_path=response_path,
        cost_per_request=cost_per_request,
    )

    sampling_raw = raw.get("sampling", {})
    if not isinstance(sampling_raw, dict):
        sampling_raw = {}
    sampling = SamplingConfig(
        k_max=int(sampling_raw.get("k_max", 20)),
        k_fixed=_parse_optional_int(sampling_raw.get("k_fixed", 10)),
        sequential_stopping=bool(sampling_raw.get("sequential_stopping", True)),
        delta=float(sampling_raw.get("delta", 0.05)),
        max_concurrent=int(sampling_raw.get("max_concurrent", 50)),
        timeout=float(sampling_raw.get("timeout", 120.0)),
        retries=int(sampling_raw.get("retries", 10)),
    )

    canon_raw = raw.get("canonicalization", {})
    if not isinstance(canon_raw, dict):
        canon_raw = {}

    judge_endpoint = None
    judge_raw = canon_raw.get("judge_endpoint")
    if isinstance(judge_raw, dict):
        j_temp_raw = judge_raw.get("temperature", 0.0)
        j_temperature = None if j_temp_raw is None else float(j_temp_raw)
        j_headers_raw = judge_raw.get("headers")
        j_headers = dict(j_headers_raw) if isinstance(j_headers_raw, dict) else {}
        judge_endpoint = EndpointConfig(
            url=str(judge_raw.get("url", "")),
            model=str(judge_raw.get("model", "")),
            temperature=j_temperature,
            api_key_env=str(judge_raw.get("api_key_env", "")),
            max_tokens=int(judge_raw.get("max_tokens", 4096)),
            provider=str(judge_raw.get("provider", "")),
            headers=j_headers,
        )

    canon = CanonConfig(
        type=str(canon_raw.get("type", "mcq")),
        judge_endpoint=judge_endpoint,
        custom_class=str(canon_raw["custom_class"]) if "custom_class" in canon_raw else None,
    )

    cal_raw = raw.get("calibration", {})
    if not isinstance(cal_raw, dict):
        cal_raw = {}
    calibration = CalibrationConfig(
        alpha_values=[float(v) for v in cal_raw.get(
            "alpha_values", [0.01, 0.05, 0.10, 0.15, 0.20]
        )],
        n_cal=int(cal_raw.get("n_cal", 500)),
        n_test=int(cal_raw.get("n_test", 500)),
        bootstrap_splits=int(cal_raw.get("bootstrap_splits", 100)),
    )

    questions_raw = raw.get("questions", {})
    if not isinstance(questions_raw, dict):
        questions_raw = {}
    q_file = questions_raw.get("file")
    q_source = questions_raw.get("source")
    questions = QuestionsConfig(
        file=str(q_file) if q_file is not None else None,
        source=str(q_source) if q_source is not None else None,
    )

    thresholds_raw = raw.get("thresholds", {})
    if not isinstance(thresholds_raw, dict):
        thresholds_raw = {}
    thresholds = ThresholdsConfig(
        pass_level=float(thresholds_raw.get("pass", 0.80)),
        weak_level=float(thresholds_raw.get("weak", 0.50)),
    )

    return TrustGateConfig(
        endpoint=endpoint,
        sampling=sampling,
        canonicalization=canon,
        calibration=calibration,
        questions=questions,
        thresholds=thresholds,
    )


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(str(value))


def validate_config(config: TrustGateConfig) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []

    # Endpoint
    if not config.endpoint.url:
        errors.append("endpoint.url is required")
    elif not _is_valid_url(config.endpoint.url):
        errors.append(f"endpoint.url is not a valid URL: {config.endpoint.url}")

    # Sampling
    if config.sampling.k_fixed is not None and config.sampling.k_fixed > config.sampling.k_max:
        errors.append(
            f"sampling.k_fixed ({config.sampling.k_fixed}) must be <= "
            f"sampling.k_max ({config.sampling.k_max})"
        )
    if config.sampling.k_max < 1:
        errors.append("sampling.k_max must be >= 1")

    # Calibration
    if config.calibration.n_cal < 1:
        errors.append("calibration.n_cal must be >= 1")
    if config.calibration.n_test < 1:
        errors.append("calibration.n_test must be >= 1")

    # Canonicalization
    if config.canonicalization.type not in VALID_CANON_TYPES:
        errors.append(
            f"canonicalization.type must be one of {sorted(VALID_CANON_TYPES)}, "
            f"got '{config.canonicalization.type}'"
        )
    if config.canonicalization.type == "custom" and not config.canonicalization.custom_class:
        errors.append(
            "canonicalization.custom_class is required when type is 'custom'"
        )
    if config.canonicalization.type in ("llm_judge", "llm") and not config.canonicalization.judge_endpoint:
        errors.append(
            f"canonicalization.judge_endpoint is required when type is '{config.canonicalization.type}'"
        )

    return errors


def _is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def resolve_api_key(config: EndpointConfig) -> str:
    """Read API key from the environment variable specified in config."""
    if not config.api_key_env:
        raise ConfigError(["endpoint.api_key_env is not set"])

    value = os.environ.get(config.api_key_env)
    if value is None:
        raise ConfigError(
            [f"Environment variable '{config.api_key_env}' is not set"]
        )
    return value


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


def load_questions(source: QuestionsConfig | str) -> list[Question]:
    """Load questions from CSV, JSON, or a QuestionsConfig.

    If *source* is a string, it's treated as a file path.
    """
    if isinstance(source, str):
        return _load_questions_from_file(source)

    if source.file:
        return _load_questions_from_file(source.file)

    if source.source:
        # Built-in dataset sources are handled by trustgate.datasets;
        # here we just raise so callers know to use the dataset loaders.
        raise NotImplementedError(
            f"Built-in dataset source '{source.source}' must be loaded via "
            "trustgate.datasets (e.g., from theaios.trustgate.datasets import load_gsm8k)"
        )

    raise ConfigError(["questions.file or questions.source must be set"])


def _load_questions_from_file(path: str) -> list[Question]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")

    if suffix == ".json":
        return _parse_questions_json(text)
    if suffix in (".csv", ".tsv"):
        return _parse_questions_csv(text)

    raise ConfigError([f"Unsupported questions file format: {suffix} (use .json or .csv)"])


def _parse_questions_json(text: str) -> list[Question]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ConfigError(["Questions JSON must be a list of objects"])

    questions: list[Question] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ConfigError([f"Question at index {i} must be an object"])
        if "id" not in item or "question" not in item:
            raise ConfigError(
                [f"Question at index {i} must have 'id' and 'question' fields"]
            )

        acceptable = item.get("acceptable_answers")
        if isinstance(acceptable, str):
            acceptable = [acceptable]

        questions.append(
            Question(
                id=str(item["id"]),
                text=str(item["question"]),
                acceptable_answers=acceptable,
                metadata={
                    k: v
                    for k, v in item.items()
                    if k not in ("id", "question", "acceptable_answers")
                },
            )
        )
    return questions


def _parse_questions_csv(text: str) -> list[Question]:
    reader = csv.DictReader(StringIO(text))

    if reader.fieldnames is None:
        raise ConfigError(["CSV file is empty or has no header row"])

    # Require at least 'id' and 'question' columns
    fields = set(reader.fieldnames)
    if "id" not in fields or "question" not in fields:
        raise ConfigError(
            [f"CSV must have 'id' and 'question' columns, got: {sorted(fields)}"]
        )

    questions: list[Question] = []
    for i, row in enumerate(reader):
        qid = row.get("id", "").strip()
        qtext = row.get("question", "").strip()
        if not qid or not qtext:
            raise ConfigError([f"Row {i + 1}: 'id' and 'question' must be non-empty"])

        acceptable_raw = row.get("acceptable_answers", "").strip()
        acceptable = [a.strip() for a in acceptable_raw.split("|") if a.strip()] or None

        metadata = {
            k: v
            for k, v in row.items()
            if k not in ("id", "question", "acceptable_answers") and v
        }

        questions.append(
            Question(
                id=qid,
                text=qtext,
                acceptable_answers=acceptable,
                metadata=metadata,
            )
        )
    return questions
