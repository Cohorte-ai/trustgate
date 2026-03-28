# API Reference

Complete Python API reference for TrustGate. All public symbols listed here are
importable directly from the `trustgate` package unless otherwise noted.

---

## Core Pipeline Functions

### `trustgate.certify`

```python
trustgate.certify(
    config: TrustGateConfig | None = None,
    config_path: str = "trustgate.yaml",
    questions: list[Question] | None = None,
    labels: dict[str, str] | None = None,
    ground_truth_file: str | None = None,
) -> CertificationResult
```

Run the full certification pipeline synchronously. This is the main entry point
for most users.

**Steps performed:**

1. Load and validate configuration
2. Load evaluation questions
3. Initialize the response cache
4. Sample K responses per question (with optional sequential stopping)
5. Canonicalize all responses
6. Compute self-consistency profiles
7. Split data into calibration and test sets
8. Load or derive ground-truth labels
9. Run conformal calibration
10. Enrich the result with cost estimates and metadata

**Parameters:**

| Parameter | Description |
|---|---|
| `config` | A pre-built `TrustGateConfig`. If `None`, loaded from `config_path`. |
| `config_path` | Path to a YAML config file. Defaults to `"trustgate.yaml"`. Ignored when `config` is provided. |
| `questions` | List of `Question` objects. If `None`, loaded from the config's `questions` section. |
| `labels` | Dict mapping question ID to the correct answer string. If `None`, derived from `ground_truth_file` or from each question's `acceptable_answers`. |
| `ground_truth_file` | Path to a CSV or JSON file containing ground-truth labels. |

**Returns:** `CertificationResult`

**Raises:**

- `ConfigError` -- if config validation fails.
- `LabelsRequired` -- if no labels can be found from any source.
- `FileNotFoundError` -- if `config_path` or `ground_truth_file` does not exist.

**Example:**

```python
from theaios import trustgate

result = trustgate.certify(config_path="trustgate.yaml")
print(f"Reliability: {result.reliability_level:.0%}")
```

---

### `trustgate.certify_async`

```python
async trustgate.certify_async(
    config: TrustGateConfig | None = None,
    config_path: str = "trustgate.yaml",
    questions: list[Question] | None = None,
    labels: dict[str, str] | None = None,
    ground_truth_file: str | None = None,
) -> CertificationResult
```

Async version of `certify()`. Identical parameters, return type, and behavior.
Use this when you are already inside an async context (e.g., an `asyncio` event
loop or a framework like FastAPI).

**Example:**

```python
import asyncio
from theaios import trustgate

result = asyncio.run(trustgate.certify_async(config_path="trustgate.yaml"))
```

---

### `trustgate.calibrate`

```python
trustgate.calibrate(
    profiles: dict[str, list[tuple[str, float]]],
    labels: dict[str, str],
    cal_ids: list[str],
    test_ids: list[str],
    alpha_values: list[float],
) -> CertificationResult
```

Low-level conformal calibration. Most users should call `certify()` instead;
this function is useful when you have already computed self-consistency profiles
and want direct control over the calibration/test split.

**Steps performed:**

1. Compute nonconformity scores on `cal_ids`.
2. For each alpha, find M\* (the conformal quantile threshold).
3. Evaluate empirical coverage on `test_ids`.
4. Determine the reliability level (largest 1-alpha where coverage holds).
5. Compute conditional coverage and capability gap on the test set.

**Parameters:**

| Parameter | Description |
|---|---|
| `profiles` | Dict mapping question ID to a self-consistency profile (list of `(answer, frequency)` tuples sorted by frequency descending). |
| `labels` | Dict mapping question ID to the correct canonical answer. |
| `cal_ids` | List of question IDs to use for calibration. |
| `test_ids` | List of question IDs to use for testing. |
| `alpha_values` | List of significance levels to evaluate (e.g., `[0.01, 0.05, 0.10]`). |

**Returns:** `CertificationResult` (with `k_used` and `api_cost_estimate` set to 0; the caller should fill these in if needed).

---

### `trustgate.compute_profile`

```python
trustgate.compute_profile(
    canonical_answers: list[str],
) -> list[tuple[str, float]]
```

Compute the self-consistency profile from K canonicalized responses to a single
question.

**Parameters:**

| Parameter | Description |
|---|---|
| `canonical_answers` | List of K canonical answer strings for a single question. Must not be empty. |

**Returns:** A list of `(answer, frequency)` tuples sorted by frequency
descending, then alphabetically for tie-breaking. Frequencies sum to 1.0.

**Raises:** `ValueError` if the input list is empty.

**Example:**

```python
from theaios import trustgate

profile = trustgate.compute_profile(["42", "42", "42", "43", "42"])
# [("42", 0.8), ("43", 0.2)]
```

---

### `trustgate.sample`

```python
trustgate.sample(
    config: TrustGateConfig,
    questions: list[Question],
    cache: DiskCache | None = None,
) -> dict[str, list[SampleResponse]]
```

Low-level synchronous sampling. Queries the configured AI endpoint K times per
question, returning raw responses. Uses caching, concurrency limiting, and
retries with exponential backoff internally.

**Parameters:**

| Parameter | Description |
|---|---|
| `config` | A `TrustGateConfig` with endpoint and sampling parameters. |
| `questions` | List of `Question` objects to sample. |
| `cache` | Optional `DiskCache` instance. A default disk cache is used if not provided. |

**Returns:** Dict mapping question ID to a list of `SampleResponse` objects.

---

### `trustgate.load_config`

```python
trustgate.load_config(
    path: str = "trustgate.yaml",
    overrides: dict[str, object] | None = None,
) -> TrustGateConfig
```

Load and validate a YAML configuration file.

**Parameters:**

| Parameter | Description |
|---|---|
| `path` | Path to the YAML config file. |
| `overrides` | Optional dict of dot-notation overrides (e.g., `{"sampling.k_fixed": 5}`). |

**Returns:** A validated `TrustGateConfig` instance.

**Raises:**

- `FileNotFoundError` -- if the config file does not exist.
- `ConfigError` -- if validation fails.

**Example:**

```python
from theaios import trustgate

config = trustgate.load_config("trustgate.yaml")
print(config.endpoint.model)
```

---

## Configuration Classes

### `trustgate.TrustGateConfig`

```python
@dataclass
class TrustGateConfig:
    endpoint: EndpointConfig
    sampling: SamplingConfig          # default: SamplingConfig()
    canonicalization: CanonConfig      # default: CanonConfig()
    calibration: CalibrationConfig    # default: CalibrationConfig()
    questions: QuestionsConfig        # default: QuestionsConfig()
    thresholds: ThresholdsConfig      # default: ThresholdsConfig()
```

Top-level configuration dataclass. Maps 1:1 to the sections in `trustgate.yaml`.

---

### `trustgate.EndpointConfig`

```python
@dataclass
class EndpointConfig:
    url: str
    model: str = ""
    temperature: float | None = 0.7
    api_key_env: str = ""
    max_tokens: int = 4096
    provider: str = ""  # "openai", "anthropic", "together", "generic_http", "generic" (auto-detected if empty)
    headers: dict[str, str] = {}
    request_template: dict | None = None
    response_path: str = ""
    cost_per_request: float | None = None
```

Configuration for an AI API endpoint. The `provider` field is auto-detected from
the URL if left empty. Supported providers:

- **openai** -- OpenAI Chat Completions API
- **anthropic** -- Anthropic Messages API
- **together** -- Together AI (OpenAI-compatible)
- **generic** -- Any OpenAI-compatible endpoint (vLLM, Ollama, LiteLLM, etc.)

The `api_key_env` field specifies the name of the environment variable holding
the API key (e.g., `"LLM_API_KEY"`).

---

### `trustgate.SamplingConfig`

```python
@dataclass
class SamplingConfig:
    k_max: int = 20
    k_fixed: int | None = 10
    sequential_stopping: bool = True
    delta: float = 0.05
    max_concurrent: int = 10
    timeout: float = 120.0
    retries: int = 10
```

| Field | Description |
|---|---|
| `k_max` | Maximum number of samples per question. |
| `k_fixed` | Fixed K to use. If set, overrides `k_max`. Set to `None` to use adaptive stopping up to `k_max`. |
| `sequential_stopping` | Enable early stopping when self-consistency stabilizes. |
| `delta` | Stability threshold for sequential stopping. |
| `max_concurrent` | Maximum concurrent API requests. |
| `timeout` | Per-request timeout in seconds. |
| `retries` | Number of retries on transient errors (429, 5xx, timeouts). |

---

### `trustgate.CanonConfig`

```python
@dataclass
class CanonConfig:
    type: str = "mcq"
    judge_endpoint: EndpointConfig | None = None
    custom_class: str | None = None
```

| Field | Description |
|---|---|
| `type` | Canonicalization strategy. One of: `"numeric"`, `"mcq"`, `"llm_judge"`, `"llm"`, `"embedding"`, `"custom"`. |
| `judge_endpoint` | Required when `type` is `"llm_judge"` or `"llm"`. Configures the LLM used as the judge. |
| `custom_class` | Required when `type` is `"custom"`. Dot-path to a `Canonicalizer` subclass (e.g., `"mypackage.MyCanonicalizer"`). |

---

### `trustgate.CalibrationConfig`

```python
@dataclass
class CalibrationConfig:
    alpha_values: list[float] = [0.01, 0.05, 0.10, 0.15, 0.20]
    n_cal: int = 500
    n_test: int = 500
    bootstrap_splits: int = 100
```

| Field | Description |
|---|---|
| `alpha_values` | Significance levels to evaluate. Smaller alpha = stricter reliability requirement. |
| `n_cal` | Number of questions for the calibration set. |
| `n_test` | Number of questions for the test set. |
| `bootstrap_splits` | Number of bootstrap re-splits for stability analysis. |

---

## Data Classes

### `trustgate.Question`

```python
@dataclass
class Question:
    id: str
    text: str
    acceptable_answers: list[str] | None = None
    metadata: dict[str, str] = {}
```

A single evaluation question.

| Field | Description |
|---|---|
| `id` | Unique identifier for the question (e.g., `"gsm8k_42"`). |
| `text` | The question text sent to the model. |
| `acceptable_answers` | Optional list of correct answers. Used to derive labels when no external ground truth is provided. |
| `metadata` | Arbitrary key-value metadata (e.g., `{"source": "gsm8k", "subject": "algebra"}`). |

---

### `trustgate.CertificationResult`

```python
@dataclass
class CertificationResult:
    reliability_level: float
    m_star: int
    coverage: float
    conditional_coverage: float
    capability_gap: float
    target_alpha: float = 0.05
    n_cal: int
    n_test: int
    k_used: int
    api_cost_estimate: float
    alpha_coverage: dict[float, float] = {}
    per_item: list[dict[str, object]] = []
```

Output of the certification pipeline.

| Field | Description |
|---|---|
| `reliability_level` | The largest `1 - alpha` for which conformal coverage holds on the test set. Higher is better. For example, `0.90` means 90% reliability. |
| `m_star` | The size of the conformal prediction set. `m_star = 1` means the model's top answer is reliably correct. |
| `coverage` | Empirical coverage on the test set: fraction of questions where the top-M\* answers contain the correct one. |
| `conditional_coverage` | Coverage on solvable items only (where the correct answer appears somewhere in the K samples). |
| `capability_gap` | Fraction of questions where the correct answer never appeared in any of the K samples. Indicates inherent model limitations. |
| `n_cal` | Number of calibration items used. |
| `n_test` | Number of test items used. |
| `k_used` | Average number of samples per question. |
| `api_cost_estimate` | Estimated API cost in USD (based on known model pricing). |
| `alpha_coverage` | Per-alpha coverage breakdown: `{alpha: empirical_coverage}`. |
| `per_item` | Optional per-question diagnostics. |

---

## Canonicalization

### `trustgate.Canonicalizer`

```python
class Canonicalizer(ABC):
    def canonicalize(self, question: str, answer: str) -> str: ...
    def validate(self, canonical: str) -> bool: ...
    def preprocess(self, answer: str) -> str: ...
```

Abstract base class for all canonicalizers. Subclass this to create custom
canonicalization logic.

| Method | Description |
|---|---|
| `canonicalize(question, answer)` | **Abstract.** Map a raw LLM answer to a canonical form. Two answers that mean the same thing should produce the same string. |
| `validate(canonical)` | Optional. Check that the canonical form is well-formed. Returns `True` by default. |
| `preprocess(answer)` | Shared preprocessing applied before canonicalization. Strips whitespace, removes markdown code fences, removes common LLM preambles, and applies Unicode NFC normalization. |

**Built-in canonicalizers:**

| Name | Use case |
|---|---|
| `"numeric"` | Math problems -- extracts and normalizes numeric answers. |
| `"mcq"` | Multiple-choice -- extracts letter answers (A, B, C, D). |
| `"llm"` | LLM Semantic -- extracts core factual answers via LLM prompt. Source: `theaios.trustgate.canonicalize.llm_semantic`. |
| `"llm_judge"` | Open-ended -- uses another LLM to judge equivalence. |
| `"embedding"` | Semantic similarity -- clusters answers by embedding distance. |

---

### `trustgate.register_canonicalizer`

```python
trustgate.register_canonicalizer(name: str) -> Callable
```

Class decorator that registers a `Canonicalizer` subclass under the given name.

**Example:**

```python
from theaios.trustgate import Canonicalizer, register_canonicalizer

@register_canonicalizer("my_custom")
class MyCanonicalizer(Canonicalizer):
    def canonicalize(self, question: str, answer: str) -> str:
        text = self.preprocess(answer)
        return text.lower().strip()
```

---

### `trustgate.get_canonicalizer`

```python
trustgate.get_canonicalizer(name: str, **kwargs) -> Canonicalizer
```

Instantiate a registered canonicalizer by name.

**Parameters:**

| Parameter | Description |
|---|---|
| `name` | The registered name (e.g., `"numeric"`, `"mcq"`, `"my_custom"`). |
| `**kwargs` | Passed to the canonicalizer constructor. For `"llm_judge"`, pass `judge_config=EndpointConfig(...)`. |

**Raises:** `KeyError` if the name is not registered.

**Example:**

```python
from theaios import trustgate

canon = trustgate.get_canonicalizer("numeric")
result = canon.canonicalize("What is 2+2?", "The answer is 4.")
# "4"
```

---

## Model Comparison

### `trustgate.comparison.compare`

```python
trustgate.comparison.compare(
    models: list[str],
    config: TrustGateConfig,
    questions: list[Question],
    labels: dict[str, str],
) -> list[tuple[str, CertificationResult]]
```

Compare multiple models on the same question set. Runs each model through the
full certification pipeline and returns results sorted by `reliability_level`
descending.

Models are evaluated sequentially (to respect per-model rate limits), but
sampling within each model run is parallelized.

**Parameters:**

| Parameter | Description |
|---|---|
| `models` | List of model name strings (e.g., `["gpt-4.1-mini", "gpt-4.1"]`). Each name is set as `config.endpoint.model` for that run. |
| `config` | Base configuration. The `endpoint.model` field is overridden for each model. |
| `questions` | Shared question set for all models. |
| `labels` | Shared ground-truth labels for all models. |

**Returns:** List of `(model_name, CertificationResult)` tuples, sorted by
reliability level descending (best model first).

There is also an async variant: `trustgate.comparison.compare_async(...)` with
the same signature.

**Example:**

```python
from theaios.trustgate.comparison import compare

results = compare(
    models=["gpt-4.1-mini", "gpt-4.1"],
    config=config,
    questions=questions,
    labels=labels,
)

for name, result in results:
    print(f"{name}: {result.reliability_level:.0%}")
```

---

## Dataset Loaders

Built-in loaders for popular evaluation benchmarks. Import from
`trustgate.datasets`.

### `trustgate.datasets.load_gsm8k`

```python
trustgate.datasets.load_gsm8k(
    n: int | None = None,
    seed: int = 42,
) -> list[Question]
```

Load GSM8K grade-school math problems. Downloads from GitHub on first use and
caches locally in `~/.trustgate/datasets/`.

Each returned `Question` has `acceptable_answers` set to the numeric answer
extracted from the solution. Recommended canonicalization: `"numeric"`.

**Parameters:**

| Parameter | Description |
|---|---|
| `n` | Subsample to `n` questions. `None` returns all (~1319 questions). |
| `seed` | Random seed for reproducible subsampling. |

---

### `trustgate.datasets.load_mmlu`

```python
trustgate.datasets.load_mmlu(
    subjects: list[str] | None = None,
    n: int | None = None,
    seed: int = 42,
    data: list[dict[str, object]] | None = None,
) -> list[Question]
```

Load MMLU multiple-choice questions. Uses the HuggingFace `datasets` library if
installed; falls back to an empty list otherwise.

Each question is formatted with lettered options (A, B, C, D) and
`acceptable_answers` set to the correct letter. Recommended canonicalization:
`"mcq"`.

**Parameters:**

| Parameter | Description |
|---|---|
| `subjects` | Filter to specific MMLU subjects. `None` returns all subjects. |
| `n` | Subsample to `n` questions. `None` returns all. |
| `seed` | Random seed for reproducible subsampling. |
| `data` | Pre-loaded data rows (list of dicts). Useful for testing without downloading. |

---

### `trustgate.datasets.load_truthfulqa`

```python
trustgate.datasets.load_truthfulqa(
    n: int | None = None,
    seed: int = 42,
    data: list[dict[str, object]] | None = None,
) -> list[Question]
```

Load TruthfulQA questions about common misconceptions. Uses the HuggingFace
`datasets` library if installed; falls back to an empty list otherwise.

Open-ended questions. Recommended canonicalization: `"llm_judge"` or
`"embedding"`.

**Parameters:**

| Parameter | Description |
|---|---|
| `n` | Subsample to `n` questions. `None` returns all. |
| `seed` | Random seed for reproducible subsampling. |
| `data` | Pre-loaded data rows (list of dicts). Useful for testing without downloading. |

---

## Reporting

Functions for outputting certification results. Import from
`trustgate.reporting`.

### `trustgate.reporting.console.print_certification_result`

```python
trustgate.reporting.console.print_certification_result(
    result: CertificationResult,
    verbose: bool = False,
    console: Console | None = None,
) -> None
```

Print a formatted certification result table to the terminal using
[Rich](https://rich.readthedocs.io/). Shows reliability level, M\*, coverage
metrics, capability gap, and a PASS/UNCERTAIN status indicator.

When `verbose=True`, also prints a per-alpha coverage breakdown table.

---

### `trustgate.reporting.console.print_comparison_result`

```python
trustgate.reporting.console.print_comparison_result(
    results: list[tuple[str, CertificationResult]],
    console: Console | None = None,
) -> None
```

Print a side-by-side comparison table for multiple models. Shows reliability,
M\*, coverage, conditional coverage, capability gap, and K used for each model.

---

### `trustgate.reporting.json_export.export_json`

```python
trustgate.reporting.json_export.export_json(
    result: CertificationResult,
    path: str | None = None,
) -> str
```

Export a certification result as a JSON string. If `path` is provided, also
writes the JSON to that file. Always returns the JSON string.

The JSON includes a `trustgate_version` field and a UTC `timestamp`.

---

### `trustgate.reporting.csv_export.export_csv`

```python
trustgate.reporting.csv_export.export_csv(
    result: CertificationResult,
    path: str | None = None,
) -> str
```

Export a certification result as a CSV string. If `path` is provided, also
writes the CSV to that file. Always returns the CSV string.

The first row contains summary metrics. If `per_item` data is present,
additional rows with per-question diagnostics are appended after a blank
separator row.
