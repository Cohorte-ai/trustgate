"""Shared data models for the TrustGate pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EndpointConfig:
    """Configuration for an AI endpoint (LLM, agent, or any AI system)."""

    url: str
    model: str = ""
    temperature: float | None = 0.7  # None = endpoint controls its own randomness
    api_key_env: str = ""
    max_tokens: int = 4096
    provider: str = ""  # openai, anthropic, together, generic, generic_http

    # Generic endpoint support (agents, RAG, custom APIs)
    headers: dict[str, str] = field(default_factory=dict)
    request_template: dict[str, object] | None = None  # {{question}} placeholder
    response_path: str = ""  # dot-notation path to extract answer from JSON response

    # Cost estimation
    cost_per_request: float | None = None  # USD per request (for pre-flight estimate)


@dataclass
class SamplingConfig:
    """Parameters for response sampling."""

    k_max: int = 20
    k_fixed: int | None = 10
    sequential_stopping: bool = True
    delta: float = 0.05
    max_concurrent: int = 50
    timeout: float = 120.0
    retries: int = 10


VALID_CANON_TYPES = {"numeric", "mcq", "code_exec", "llm_judge", "embedding", "custom"}


@dataclass
class CanonConfig:
    """Configuration for the canonicalization step."""

    type: str = "mcq"
    judge_endpoint: EndpointConfig | None = None
    custom_class: str | None = None


@dataclass
class CalibrationConfig:
    """Parameters for conformal calibration."""

    alpha_values: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.15, 0.20]
    )
    n_cal: int = 500
    n_test: int = 500
    bootstrap_splits: int = 100


@dataclass
class QuestionsConfig:
    """How to load evaluation questions."""

    file: str | None = None
    source: str | None = None


@dataclass
class TrustGateConfig:
    """Top-level configuration — maps 1:1 to trustgate.yaml."""

    endpoint: EndpointConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    canonicalization: CanonConfig = field(default_factory=CanonConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    questions: QuestionsConfig = field(default_factory=QuestionsConfig)


@dataclass
class Question:
    """A single evaluation question."""

    id: str
    text: str
    acceptable_answers: list[str] | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SampleResponse:
    """A single sampled response from an AI endpoint."""

    question_id: str
    sample_index: int
    raw_response: str
    cached: bool = False


@dataclass
class CertificationResult:
    """Output of the full certification pipeline."""

    reliability_level: float
    m_star: int
    coverage: float
    conditional_coverage: float
    capability_gap: float
    n_cal: int
    n_test: int
    k_used: int
    api_cost_estimate: float
    alpha_coverage: dict[float, float] = field(default_factory=dict)
    per_item: list[dict[str, object]] = field(default_factory=list)
