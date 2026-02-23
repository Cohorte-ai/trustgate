# TrustGate

**Know if your AI is ready to ship — one number, one guarantee.**

TrustGate certifies the reliability of any LLM or AI endpoint using self-consistency sampling and conformal prediction. Point it at your API, run the pipeline, and get a single reliability level (e.g., 94.6%) backed by a formal statistical guarantee — not a vibe, not a leaderboard score, a mathematical proof.

Works with any provider (OpenAI, Anthropic, Together, self-hosted), any task type, any model. Black-box, no model internals required.

---

## Installation

```bash
pip install trustgate
```

Optional extras for specific canonicalization methods:

```bash
# LLM-as-judge canonicalization (needs openai SDK)
pip install "trustgate[judge]"

# Embedding-based canonicalization (needs sentence-transformers)
pip install "trustgate[embedding]"

# Local human calibration UI (needs Flask)
pip install "trustgate[serve]"

# Everything
pip install "trustgate[all]"
```

---

## Quickstart

### 1. Create a config file

```yaml
# trustgate.yaml
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4.1"
  temperature: 0.7
  api_key_env: "OPENAI_API_KEY"

sampling:
  k_fixed: 10
  sequential_stopping: true

canonicalization:
  type: "mcq"

calibration:
  alpha_values: [0.01, 0.05, 0.10, 0.15, 0.20]
  n_cal: 250
  n_test: 250

questions:
  file: "questions.csv"
```

### 2. Prepare your questions

```csv
id,question,acceptable_answers
q001,"What is the capital of France? (A) London (B) Paris (C) Berlin (D) Madrid","B"
q002,"Which planet is largest? (A) Earth (B) Mars (C) Jupiter (D) Venus","C"
```

### 3. Run certification

```bash
trustgate certify
```

```
TrustGate Certification Result
────────────────────────────────────────────
  Reliability Level:   94.6%  (CI: 93.2–95.8%)
  M* (prediction set): 1
  Empirical Coverage:   0.956  (target: 0.900)
  Conditional Coverage: 0.980
  Capability Gap:       2.4%
  Items:                250 cal / 250 test
  Sampling:             K=10, saved $11.20 (47%) via sequential stopping
  Status:               PASS
────────────────────────────────────────────
```

That's it. Your model is 94.6% reliable at the 90% confidence level.

---

## CLI Reference

### Certify a model

```bash
trustgate certify \
  --endpoint "https://api.openai.com/v1/chat/completions" \
  --api-key $OPENAI_API_KEY \
  --task-type mcq \
  --questions questions.csv \
  --ground-truth labels.csv \
  --alpha 0.10 \
  --k 10
```

All flags are optional if you have a `trustgate.yaml` in the current directory.

### Compare models

```bash
trustgate compare \
  --models gpt-4.1,gpt-4.1-mini,claude-sonnet-4-6 \
  --task-type mcq \
  --questions questions.csv \
  --ground-truth labels.csv
```

```
Model              | Reliability | M* | Coverage | Capability Gap
gpt-4.1            | 94.6%       | 1  | 0.956    | 2.4%
claude-sonnet-4-6  | 96.0%       | 1  | 0.968    | 1.8%
gpt-4.1-mini       | 91.2%       | 2  | 0.923    | 3.8%
```

### Human calibration

When you don't have ground truth labels, use human reviewers:

```bash
trustgate calibrate --serve --port 8080
```

This opens a local web UI where a reviewer sees each question + the model's top answer and marks it correct or incorrect. 50 items takes ~5 minutes. Send the link to a domain expert — they don't need to know anything about ML.

```
┌──────────────────────────────────────────┐
│ ██████████████░░░░░░░░░░  23/50  (46%)   │
│                                          │
│  Question:                               │
│  What is the standard treatment for      │
│  acute myocardial infarction?            │
│                                          │
│  AI's Answer:                            │
│  Aspirin, heparin, and percutaneous      │
│  coronary intervention (PCI)             │
│                                          │
│  ┌─────────────┐  ┌──────────────┐       │
│  │  Correct     │  │  Incorrect   │       │
│  └─────────────┘  └──────────────┘       │
└──────────────────────────────────────────┘
```

Then certify using the collected labels:

```bash
trustgate certify --ground-truth calibration_labels.json
```

### Export results

```bash
# JSON (for CI/CD pipelines)
trustgate certify --output json --output-file result.json

# CSV (for spreadsheets)
trustgate certify --output csv --output-file result.csv
```

### Cache management

Responses are cached locally so re-runs don't cost API calls:

```bash
trustgate cache stats    # show cache size and entry count
trustgate cache clear    # delete all cached responses
```

---

## Python API

```python
import trustgate

result = trustgate.certify(config_path="trustgate.yaml")

print(result.reliability_level)    # 0.946
print(result.m_star)               # 1
print(result.coverage)             # 0.956
print(result.capability_gap)       # 0.024
```

### Inline configuration (no YAML needed)

```python
import trustgate

result = trustgate.certify(
    config=trustgate.TrustGateConfig(
        endpoint=trustgate.EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_API_KEY",
        ),
        sampling=trustgate.SamplingConfig(k_fixed=10),
        canonicalization=trustgate.CanonConfig(type="mcq"),
    ),
    questions=trustgate.load_questions("questions.csv"),
    labels={"q001": "B", "q002": "C"},
)
```

### Built-in datasets

```python
from trustgate.datasets import load_gsm8k, load_mmlu, load_truthfulqa

# Grab 100 MMLU questions (auto-downloads, cached locally)
questions = load_mmlu(subjects=["abstract_algebra"], n=100)

# Grab 200 GSM8K math problems
questions = load_gsm8k(n=200)
```

### Custom canonicalization

Write your own canonicalizer for domain-specific tasks:

```python
from trustgate import Canonicalizer, register_canonicalizer

@register_canonicalizer("legal_citation")
class LegalCitationCanonicalizer(Canonicalizer):
    def canonicalize(self, question: str, answer: str) -> str:
        # Extract and normalize case citations
        import re
        citations = re.findall(r'\d+\s+\w+\.?\s*\d+', answer)
        return "|".join(sorted(citations)) if citations else "no_citation"

# Use it
result = trustgate.certify(
    config=trustgate.TrustGateConfig(
        endpoint=...,
        canonicalization=trustgate.CanonConfig(type="legal_citation"),
    ),
    questions=my_legal_questions,
    labels=my_labels,
)
```

### CI/CD gating

```python
result = trustgate.certify(config_path="trustgate.yaml")
if result.reliability_level < 0.90:
    sys.exit(1)  # block deployment
```

Or from the CLI:

```bash
trustgate certify --min-reliability 90
# Exit code 0 = PASS, 1 = FAIL
```

---

## Task Types & Canonicalization

TrustGate supports multiple task types out of the box. Each task type has a built-in canonicalizer that normalizes raw LLM responses into comparable forms:

| Task type | Canonicalizer | What it does | Example |
|-----------|--------------|-------------|---------|
| `mcq` | MCQ | Extracts chosen option letter | "I think B) Paris" -> `"B"` |
| `numeric` | Numeric | Extracts final number | "The answer is $42.50" -> `"42.5"` |
| `code_exec` | Code execution | Runs code in sandbox | Python code -> `"pass"` / `"fail"` |
| `llm_judge` | LLM-as-judge | Asks a judge LLM | (question, answer) -> `"correct"` / `"incorrect"` |
| `embedding` | Embedding clustering | Groups by semantic similarity | Free text -> `"cluster_0"` |
| `custom` | Your own plugin | Whatever you need | Your logic -> your canonical form |

---

## How It Works

TrustGate implements a four-step pipeline:

```
1. SAMPLE        Ask the AI the same question K times (with temperature > 0)
2. CANONICALIZE  Normalize raw responses into comparable forms
3. CALIBRATE     Use human labels or ground truth to compute conformal scores
4. CERTIFY       Produce a reliability level with a formal coverage guarantee
```

**Self-consistency sampling:** If you ask GPT-4 "What is 2+2?" ten times at temperature 0.7, it will say "4" every time. If you ask it a harder question, answers will vary. The pattern of agreement tells you how confident you should be.

**Conformal prediction:** A statistical framework that converts these agreement patterns into a guaranteed reliability level. If TrustGate says "94.6% reliable at alpha=0.10," that means the model's top answer is correct for at least 94.6% of questions — and this guarantee holds with 90% confidence, no matter the data distribution.

**Sequential stopping:** You don't always need all K samples. TrustGate uses Hoeffding bounds to detect when the answer pattern has stabilized and stops early, saving ~50% of API costs.

For the full theory, see our paper: *[paper title and link]*

---

## Configuration Reference

<details>
<summary>Full <code>trustgate.yaml</code> reference</summary>

```yaml
# Required: the AI endpoint to certify
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4.1"
  temperature: 0.7
  api_key_env: "OPENAI_API_KEY"     # env var name (not the key itself)
  provider: "openai"                 # openai, anthropic, together, generic (auto-detected from URL)
  max_tokens: 4096

# Sampling parameters
sampling:
  k_max: 20                          # maximum samples per question
  k_fixed: 10                        # fixed K (overrides k_max if set)
  sequential_stopping: true           # stop early when answer stabilizes
  delta: 0.05                         # confidence parameter for stopping
  max_concurrent: 50                  # parallel requests
  timeout: 120.0                      # per-request timeout (seconds)
  retries: 10                         # max retries on failure

# How to normalize raw responses
canonicalization:
  type: "mcq"                         # mcq, numeric, code_exec, llm_judge, embedding, custom
  # For llm_judge:
  # judge_endpoint:
  #   url: "https://api.openai.com/v1/chat/completions"
  #   model: "gpt-4.1"
  #   api_key_env: "OPENAI_API_KEY"
  # For custom:
  # custom_class: "my_package.my_module.MyCanonicalizer"

# Conformal calibration parameters
calibration:
  alpha_values: [0.01, 0.05, 0.10, 0.15, 0.20]
  n_cal: 500                          # calibration set size
  n_test: 500                         # test set size
  bootstrap_splits: 100

# Questions to evaluate
questions:
  file: "questions.csv"               # CSV or JSON file
  # source: "huggingface/gsm8k"      # or load from a built-in dataset
```

</details>

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/Cohorte-ai/trustgate.git
cd trustgate
pip install -e ".[dev,all]"
pytest
ruff check src/ tests/
```

---

## Citation

If you use TrustGate in your research, please cite:

```bibtex
@article{mouzouni2026trustgate,
  title={TrustGate: Black-Box AI Reliability Certification via Self-Consistency Sampling and Conformal Calibration},
  author={Mouzouni, Charafeddine},
  year={2026}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
