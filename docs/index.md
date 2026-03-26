# TrustGate

**Know if your AI is ready to ship — one number, one guarantee.**

TrustGate certifies the reliability of any AI endpoint — LLMs, agents, RAG pipelines, or any system you can ask a question to. It uses self-consistency sampling and conformal prediction to produce a single **reliability level** (e.g., 94.6%) backed by a formal statistical guarantee.

Black-box. No model internals required. Works with any provider.

---

## Installation

```bash
pip install theaios-trustgate
```

This installs everything you need — sampling, canonicalization, calibration, certification, CLI, and the web UI.

Optional extra for embedding-based canonicalization (large ML dependencies):

```bash
pip install "theaios-trustgate[embedding]"
```

Requires Python 3.10+.

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

```csv
id,question,acceptable_answers
q001,"Capital of France? (A) London (B) Paris (C) Berlin (D) Madrid","B"
q002,"Largest planet? (A) Earth (B) Mars (C) Jupiter (D) Venus","C"
```

### 2. Certify

```bash
trustgate certify
```

### 3. Read the result

```
     TrustGate Certification Result
┌──────────────────────┬──────────┐
│ Reliability Level    │ 94.6%    │
│ M* (prediction set)  │ 1        │
│ Empirical Coverage   │ 0.956    │
│ Conditional Coverage │ 0.980    │
│ Capability Gap       │ 2.4%     │
│ Status               │ PASS     │
└──────────────────────┴──────────┘
```

**Reliability level** (94.6%) is the number that matters: the model's top answer is correct for at least 94.6% of questions, backed by a conformal coverage guarantee.

---

## How It Works

```
1. SAMPLE        Ask the AI the same question K times
2. CANONICALIZE  Normalize raw responses into comparable forms
3. CALIBRATE     Human or ground truth labels → nonconformity scores
4. CERTIFY       Conformal prediction → reliability level with guarantee
```

For each calibration question, the system finds where the correct answer ranks in the AI's self-consistency profile. If the AI's top answer is correct → score = 1. If the correct answer is second → score = 2. These scores are sorted, and the conformal quantile gives M* — the prediction set size that guarantees coverage.

The human reviewer (or ground truth dataset) provides the correct answers. The math is the same either way.

→ **[Full explanation in Concepts](concepts.md)**

---

## Key Features

### Any endpoint
Works with LLMs, agents, RAG pipelines, or any HTTP API. No model or temperature control required.

→ **[Configuration guide](configuration.md)**

### Human calibration without ground truth
Generate a shareable HTML questionnaire for domain experts. No server needed — works offline in any browser. Answers are shown in randomized order to prevent bias. Each selection produces a nonconformity score that feeds into the conformal calibration.

→ **[Human Calibration guide](human-calibration.md)**

### Runtime trust layer
Wrap your endpoint with `TrustGate` to attach reliability metadata to every response. Passthrough mode (1 API call) or sampled mode (K calls with per-query confidence).

→ **[API Reference](api-reference.md)**

### Cost-aware
Pre-flight estimate shows the cost/reliability tradeoff before you spend money. Sequential stopping via Hoeffding bounds saves ~50% of API costs.

→ **[CLI Reference](cli.md)**

### Component-level certification
Don't certify the whole pipeline — certify each component independently to find where reliability breaks down. The retriever, the reasoner, the generator — each gets its own reliability level.

→ **[Canonicalization guide](canonicalization.md#certifying-pipeline-components)**

---

## Metrics Reference

| Metric | What it means |
|--------|--------------|
| **Reliability Level** | Highest confidence level where the coverage guarantee holds |
| **M\*** | How many top answers needed to guarantee the correct one is included |
| **Empirical Coverage** | Fraction of test questions where top-M* answers contain the correct one |
| **Conditional Coverage** | Same, but only on questions the model can solve |
| **Capability Gap** | Fraction of questions where the correct answer never appeared in K samples |

→ **[Detailed explanations in Concepts](concepts.md)**

---

## Python API

```python
from theaios import trustgate

# Certify
result = trustgate.certify(config_path="trustgate.yaml")

# Sample + profile (for custom pipelines)
profiles = trustgate.sample_and_profile(config, questions)

# Diagnose profile quality
diag = trustgate.diagnose_profiles(profiles)

# Generate shareable questionnaire
trustgate.generate_questionnaire(questions, profiles, "questionnaire.html")

# Runtime trust layer
gate = trustgate.TrustGate(config=config, certification=result)
response = gate.query("What is 2+2?")
```

→ **[Full API Reference](api-reference.md)**

---

## Further Reading

- **[Concepts](concepts.md)** — Self-consistency, conformal prediction, sequential stopping, how human feedback becomes a reliability guarantee
- **[Getting Your Questions](getting-questions.md)** — Generate with AI, extract from production logs, augment, or use built-in benchmarks
- **[Configuration](configuration.md)** — Full `trustgate.yaml` reference, generic endpoints
- **[CLI Reference](cli.md)** — All commands and flags
- **[Canonicalization](canonicalization.md)** — Built-in canonicalizers, custom plugins, decision-point guidance
- **[Human Calibration](human-calibration.md)** — HTML questionnaire, web UI, labels format
- **[API Reference](api-reference.md)** — Python API, TrustGate runtime class
- **[FAQ](faq.md)** — Common questions

---

## Paper

For the full theory: [*Black-Box Reliability Certification for AI Agents via Self-Consistency Sampling and Conformal Calibration*](https://arxiv.org/abs/2602.21368) (Mouzouni, 2026).
