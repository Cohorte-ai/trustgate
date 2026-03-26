# CLI Reference

TrustGate ships a command-line interface built with [Click](https://click.palletsprojects.com/).
All commands are accessed through the `trustgate` entry point.

```
trustgate [COMMAND] [OPTIONS]
```

If a `trustgate.yaml` file exists in the current directory, it is loaded
automatically. All flags are optional when the config file provides the
necessary values. CLI flags override values from the config file.

---

## Commands

- [`trustgate certify`](#trustgate-certify) -- Run the full certification pipeline
- [`trustgate compare`](#trustgate-compare) -- Compare multiple models side by side
- [`trustgate sample`](#trustgate-sample) -- Sample responses only (no certification)
- [`trustgate cache stats`](#trustgate-cache-stats) -- Show cache size and entry count
- [`trustgate cache clear`](#trustgate-cache-clear) -- Delete all cached responses
- [`trustgate version`](#trustgate-version) -- Print version

---

## `trustgate certify`

Run the full certification pipeline: sample responses from an AI endpoint,
canonicalize them, perform conformal calibration, and produce a reliability
certificate.

```
trustgate certify [OPTIONS]
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | PATH | `trustgate.yaml` | Path to the YAML config file. |
| `--endpoint` | | URL | *(from config)* | AI endpoint URL. Overrides `endpoint.url` in the config. |
| `--model` | | TEXT | *(from config)* | Model name. Overrides `endpoint.model` in the config. |
| `--api-key-env` | | TEXT | *(from config)* | Environment variable name holding the API key. Overrides `endpoint.api_key_env`. |
| `--task-type` | | CHOICE | *(from config)* | Canonicalization type. One of: `numeric`, `mcq`, `code_exec`, `llm_judge`, `llm`, `embedding`, `custom`. |
| `--questions` | `-q` | PATH | *(from config)* | Path to a questions file (CSV or JSON). |
| `--ground-truth` | `-g` | PATH | *(none)* | Path to a ground-truth labels file. Required for some task types. |
| `--k` | | INT | *(from config)* | Fixed number of samples per question. Overrides `sampling.k_fixed`. |
| `--alpha` | `-a` | FLOAT | `0.05` | Confidence level for M\*. Controls the prediction set size: `0.05` = 95% confidence, `0.10` = 90%, `0.01` = 99%. Lower α → stricter guarantee → potentially larger M\*. |
| `--output` | `-o` | CHOICE | `console` | Output format: `console`, `json`, or `csv`. |
| `--output-file` | | PATH | *(none)* | Write output to a file instead of stdout. When used with `--output console`, the result is printed to the terminal and also saved as JSON to the specified file. |
| `--no-cache` | | FLAG | `false` | Disable the disk-based response cache for this run. |
| `--verbose` | `-v` | FLAG | `false` | Show detailed per-question results in console output. |
| `--auto-judge` | | FLAG | `false` | Use LLM-as-judge for automated calibration instead of human review. Requires `judge_endpoint` in config. |
| `--min-reliability` | | FLOAT | *(none)* | Minimum reliability level (0–100). **Exit code 1** if below threshold. Use in CI/CD pipelines. |
| `--cost-per-request` | | FLOAT | *(none)* | Cost per API request in USD (for generic/agent endpoints). |
| `--concurrency` | | INT | `10` | Max concurrent API requests. Lower for rate-limited APIs (e.g., `5`), raise for fast APIs (e.g., `30`). |
| `--yes` | `-y` | FLAG | `false` | Skip the pre-flight confirmation prompt. |

### Examples

**Basic certification with a config file:**

```bash
trustgate certify
```

**Override model and significance level:**

```bash
trustgate certify --model gpt-4.1 --alpha 0.05
```

**Certify without a config file (all options via CLI):**

```bash
trustgate certify \
    --endpoint "https://api.openai.com/v1/chat/completions" \
    --model "gpt-4.1-mini" \
    --api-key-env "OPENAI_API_KEY" \
    --task-type mcq \
    --questions questions.csv \
    --ground-truth labels.json \
    --k 10 \
    --alpha 0.05
```

**Export results as JSON:**

```bash
trustgate certify --output json --output-file results.json
```

**Export results as CSV:**

```bash
trustgate certify --output csv --output-file results.csv
```

**Verbose console output with results saved to a file:**

```bash
trustgate certify -v --output-file results.json
```

---

## `trustgate compare`

Certify multiple models against the same question set and display their
reliability metrics side by side. Each model is certified independently using
the same sampling and calibration settings from the config file.

```
trustgate compare [OPTIONS]
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--models` | | TEXT | **(required)** | Comma-separated list of model names to compare. |
| `--config` | `-c` | PATH | `trustgate.yaml` | Path to the YAML config file. The `endpoint` section provides the base URL and API key; the model name is overridden for each entry in `--models`. |
| `--task-type` | | CHOICE | **(required)** | Canonicalization type. One of: `numeric`, `mcq`, `code_exec`, `llm_judge`, `llm`, `embedding`, `custom`. |
| `--questions` | `-q` | PATH | **(required)** | Path to the questions file (CSV or JSON). |
| `--ground-truth` | `-g` | PATH | *(none)* | Path to a ground-truth labels file. |
| `--alpha` | `-a` | FLOAT | `0.05` | Confidence level for M\* in compare mode. |
| `--output` | `-o` | CHOICE | `console` | Output format: `console` or `json`. |

### Examples

**Compare two models:**

```bash
trustgate compare \
    --models "gpt-4.1-mini,gpt-4.1" \
    --task-type mcq \
    --questions questions.csv
```

**Compare with ground truth and JSON output:**

```bash
trustgate compare \
    --models "gpt-4.1-mini,gpt-4.1,claude-sonnet-4-20250514" \
    --task-type mcq \
    --questions questions.csv \
    --ground-truth labels.json \
    --alpha 0.05 \
    --output json
```

---

## `trustgate sample`

Sample responses from the configured endpoint without running calibration or
certification. Useful for pre-populating the cache or inspecting raw model
outputs.

```
trustgate sample [OPTIONS]
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | PATH | `trustgate.yaml` | Path to the YAML config file. |
| `--questions` | `-q` | PATH | **(required)** | Path to the questions file (CSV or JSON). |
| `--k` | | INT | *(from config)* | Number of samples per question. Overrides `sampling.k_fixed`. |
| `--verbose` | `-v` | FLAG | `false` | Show detailed output. |

### Examples

**Sample responses using config defaults:**

```bash
trustgate sample --questions questions.csv
```

**Sample 20 responses per question:**

```bash
trustgate sample --questions questions.csv --k 20
```

**Pre-populate the cache before certification:**

```bash
trustgate sample --questions questions.csv --k 15
trustgate certify --questions questions.csv --k 15
```

The second command will be fast because all responses are already cached.

---

## `trustgate cache stats`

Display statistics about the disk-based response cache.

```
trustgate cache stats [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cache-dir` | PATH | `.trustgate_cache` | Path to the cache directory. |

### Example

```bash
trustgate cache stats
```

```
Cache directory: .trustgate_cache
  Total entries: 1,250
  Total size: 3,482,016 bytes
```

---

## `trustgate cache clear`

Delete all cached API responses. You will be prompted for confirmation before
any data is removed.

```
trustgate cache clear [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cache-dir` | PATH | `.trustgate_cache` | Path to the cache directory. |
| `--yes` | FLAG | `false` | Skip the confirmation prompt. |

### Examples

**Clear with confirmation prompt:**

```bash
trustgate cache clear
```

```
Are you sure you want to clear the cache? [y/N]: y
Cache cleared.
```

**Clear without prompt (for scripts):**

```bash
trustgate cache clear --yes
```

---

## `trustgate version`

Print the installed TrustGate version and exit.

```
trustgate version
```

### Example

```bash
trustgate version
```

```
trustgate 0.1.0
```

---

## Global Behavior

### Config file resolution

Every command that accepts `--config` defaults to `trustgate.yaml` in the
current working directory. If the file does not exist and sufficient options
are provided via CLI flags (e.g., `--endpoint`), the command proceeds without
a config file. If neither a config file nor sufficient CLI flags are provided,
the command exits with an error.

### Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `1`  | Configuration error, missing file, or runtime failure |

### Response caching

By default, all API responses are cached to disk in `.trustgate_cache/` so that
repeated runs with the same questions and parameters do not incur additional API
costs. Use `--no-cache` on the `certify` command to bypass the cache, or use
`trustgate cache clear` to reset it.
