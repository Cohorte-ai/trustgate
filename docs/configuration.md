# Configuration Reference

TrustGate is configured via a YAML file, typically named `trustgate.yaml` in your
project root. The CLI looks for this file by default, but you can point to any
path with `--config`.

The configuration maps 1:1 to the `TrustGateConfig` dataclass defined in
`src/trustgate/types.py`.

---

## Minimal Example

The only required field is `endpoint.url`. Everything else has sensible defaults:

```yaml
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  api_key_env: "OPENAI_API_KEY"

questions:
  file: "questions.csv"
```

---

## Full Example

```yaml
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4.1-mini"
  temperature: 0.7
  api_key_env: "OPENAI_API_KEY"
  max_tokens: 4096
  provider: ""                  # auto-detected from URL if left empty

sampling:
  k_max: 20
  k_fixed: 10
  sequential_stopping: true
  delta: 0.05
  max_concurrent: 50
  timeout: 120.0
  retries: 10

canonicalization:
  type: "mcq"
  # judge_endpoint:             # required when type is "llm_judge" or "llm"
  #   url: "https://api.openai.com/v1/chat/completions"
  #   model: "gpt-4.1"
  #   api_key_env: "OPENAI_API_KEY"
  # custom_class: "mymodule.MyCanonicalizer"  # required only when type is "custom"

calibration:
  alpha_values: [0.01, 0.05, 0.10, 0.15, 0.20]
  n_cal: 500
  n_test: 500
  bootstrap_splits: 100

questions:
  file: "examples/example_questions.csv"
  # source: "gsm8k"            # alternative: load a built-in benchmark
```

---

## Section Reference

### `endpoint`

Connection details for the AI model being certified.

| Field         | Type    | Default | Description |
|---------------|---------|---------|-------------|
| `url`         | string  | **(required)** | Base URL of the chat completions endpoint. Must be a valid `http` or `https` URL. |
| `model`       | string  | `""`    | Model identifier sent in API requests (e.g., `"gpt-4.1-mini"`, `"claude-sonnet-4-20250514"`). |
| `temperature` | float   | `0.7`   | Sampling temperature. Higher values increase response diversity, which is important for conformal prediction. |
| `api_key_env` | string  | `""`    | Name of the environment variable that holds the API key (e.g., `"OPENAI_API_KEY"`). TrustGate reads the key from the environment at runtime -- the key itself is never stored in the config file. |
| `max_tokens`  | integer | `4096`  | Maximum tokens per completion response. |
| `provider`    | string  | `""`    | Provider hint: `"openai"`, `"anthropic"`, `"together"`, or `"generic"`. If left empty, the provider is auto-detected from the URL. |

#### Example

```yaml
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4.1-mini"
  temperature: 0.7
  api_key_env: "OPENAI_API_KEY"
```

---

### `sampling`

Controls how many responses are drawn per question and how they are collected.

| Field                 | Type          | Default | Description |
|-----------------------|---------------|---------|-------------|
| `k_max`               | integer       | `20`    | Maximum number of samples per question. Acts as an upper bound when sequential stopping is enabled. Must be >= 1. |
| `k_fixed`             | integer/null  | `10`    | Fixed number of samples per question. Set to `null` (or omit) to use adaptive sequential stopping instead. Must be <= `k_max` when set. |
| `sequential_stopping` | boolean       | `true`  | When `true` and `k_fixed` is null, TrustGate draws samples adaptively and stops early once the conformal set has stabilized, up to `k_max`. |
| `delta`               | float         | `0.05`  | Confidence parameter for the sequential stopping criterion. Smaller values require more samples before stopping. |
| `max_concurrent`      | integer       | `50`    | Maximum number of concurrent API requests. Increase for throughput; decrease if you hit rate limits. |
| `timeout`             | float         | `120.0` | Per-request timeout in seconds. |
| `retries`             | integer       | `10`    | Number of retry attempts for failed API requests (uses exponential back-off). |

#### Example -- fixed sampling

```yaml
sampling:
  k_fixed: 10
  k_max: 20
```

#### Example -- adaptive sequential stopping

```yaml
sampling:
  k_fixed: null
  k_max: 30
  sequential_stopping: true
  delta: 0.03
```

---

### `canonicalization`

Determines how free-text model responses are mapped to canonical answer tokens
for comparison with ground truth.

| Field            | Type   | Default | Description |
|------------------|--------|---------|-------------|
| `type`           | string | `"mcq"` | Canonicalization strategy. Must be one of: `"numeric"`, `"mcq"`, `"code_exec"`, `"llm_judge"`, `"llm"`, `"embedding"`, `"custom"`. |
| `judge_endpoint` | object | `null`  | An `endpoint` block (same schema as the top-level `endpoint`) for the LLM. **Required** when `type` is `"llm_judge"` or `"llm"`. Also used by `--auto-judge` for automated calibration. |
| `custom_class`   | string | `null`  | Fully-qualified Python class path (e.g., `"mypackage.canon.MyCanon"`). **Required** when `type` is `"custom"`. |

#### Canonicalization types

| Type          | Use case | Notes |
|---------------|----------|-------|
| `mcq`         | Multiple-choice questions | Extracts A/B/C/D letter answers from free-text responses. |
| `numeric`     | Math and numerical answers | Parses numeric values, tolerates minor formatting differences. |
| `code_exec`   | Code generation tasks | Executes generated code in a sandbox and compares outputs. |
| `llm_judge`   | Open-ended / subjective tasks | Uses a separate LLM to judge equivalence. Requires `judge_endpoint`. |
| `embedding`   | Semantic similarity | Uses embedding distance to determine answer equivalence. |
| `custom`      | Anything else | Load your own canonicalizer class. Requires `custom_class`. |

#### Example -- LLM judge

```yaml
canonicalization:
  type: "llm_judge"
  judge_endpoint:
    url: "https://api.openai.com/v1/chat/completions"
    model: "gpt-4.1"
    temperature: 0.0
    api_key_env: "OPENAI_API_KEY"
```

#### Example -- custom canonicalizer

```yaml
canonicalization:
  type: "custom"
  custom_class: "my_project.canonicalizers.RegexCanon"
```

---

### `calibration`

Parameters for the conformal calibration procedure.

| Field              | Type         | Default                          | Description |
|--------------------|--------------|----------------------------------|-------------|
| `alpha_values`     | list[float]  | `[0.01, 0.05, 0.10, 0.15, 0.20]` | Significance levels to evaluate. Each alpha produces a coverage guarantee of `1 - alpha`. For example, `0.05` targets 95% coverage. |
| `n_cal`            | integer      | `500`                            | Number of calibration questions. Must be >= 1. |
| `n_test`           | integer      | `500`                            | Number of test questions held out for coverage evaluation. Must be >= 1. |
| `bootstrap_splits` | integer      | `100`                            | Number of bootstrap re-splits for computing confidence intervals on coverage estimates. |

#### Example

```yaml
calibration:
  alpha_values: [0.05, 0.10]
  n_cal: 1000
  n_test: 1000
  bootstrap_splits: 200
```

---

### `questions`

Tells TrustGate where to find the evaluation question set.

| Field    | Type   | Default | Description |
|----------|--------|---------|-------------|
| `file`   | string | `null`  | Path to a CSV or JSON file containing questions. Relative paths are resolved from the current working directory. |
| `source` | string | `null`  | Name of a built-in benchmark dataset (e.g., `"gsm8k"`, `"mmlu"`, `"truthfulqa"`). The dataset is downloaded and cached automatically in `~/.trustgate/datasets/`. |

You must set exactly one of `file` or `source`.

#### CSV format

The CSV must have at least `id` and `question` columns. An optional
`acceptable_answers` column can contain pipe-separated (`|`) values:

```csv
id,question,acceptable_answers
q001,"What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6","B"
q002,"Name two prime numbers less than 10","2|3|5|7"
```

#### JSON format

A JSON array of objects. Each object must have `id` and `question` fields.
`acceptable_answers` can be a string or a list of strings:

```json
[
  {
    "id": "q001",
    "question": "What is 2+2?",
    "acceptable_answers": ["4"]
  },
  {
    "id": "q002",
    "question": "Capital of France?",
    "acceptable_answers": "Paris"
  }
]
```

#### Example -- local file

```yaml
questions:
  file: "data/my_questions.csv"
```

#### Example -- built-in benchmark

```yaml
questions:
  source: "gsm8k"
```

---

## CLI Overrides

Most configuration fields can be overridden from the command line. CLI flags
take precedence over values in the YAML file. For example:

```bash
trustgate certify --config trustgate.yaml \
    --model gpt-4.1 \
    --k 15 \
    --alpha 0.05
```

This loads `trustgate.yaml` but overrides the model name, sample count, and
significance level.

---

## Programmatic Overrides

When loading a config via the Python API, you can pass dot-notation overrides:

```python
from theaios.trustgate import load_config

config = load_config(
    "trustgate.yaml",
    overrides={
        "endpoint.model": "gpt-4.1",
        "sampling.k_fixed": 5,
        "canonicalization.type": "numeric",
    },
)
```

---

## Environment Variables

TrustGate never stores API keys in configuration files. Instead, set
`endpoint.api_key_env` to the **name** of an environment variable:

```yaml
endpoint:
  api_key_env: "OPENAI_API_KEY"
```

Then export the key in your shell:

```bash
export OPENAI_API_KEY="sk-..."
```

TrustGate will read the key at runtime and raise a `ConfigError` if the
variable is missing or empty.

---

## Validation Rules

The config loader runs the following checks after parsing. If any fail, a
`ConfigError` is raised with a list of all violations:

- `endpoint.url` must be a valid `http://` or `https://` URL.
- `sampling.k_max` must be >= 1.
- `sampling.k_fixed` (when set) must be <= `sampling.k_max`.
- `calibration.n_cal` must be >= 1.
- `calibration.n_test` must be >= 1.
- `canonicalization.type` must be one of: `code_exec`, `custom`, `embedding`, `llm`, `llm_judge`, `mcq`, `numeric`.
- `canonicalization.custom_class` is required when type is `"custom"`.
- `canonicalization.judge_endpoint` is required when type is `"llm_judge"` or `"llm"`.
