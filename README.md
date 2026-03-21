# TrustGate

**Know if your AI is ready to ship — one number, one guarantee.**

TrustGate certifies the reliability of any AI endpoint — LLMs, agents, RAG pipelines, or any system you can ask a question to. It uses self-consistency sampling and conformal prediction to produce a single reliability level (e.g., 94.6%) backed by a formal statistical guarantee. Not a vibe, not a leaderboard score — a mathematical proof.

Black-box. No model internals required. Works with any provider, any task type, any endpoint.

```bash
pip install theaios-trustgate
```

---

## Quickstart

### 1. Create a config

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

### 2. Prepare questions

```csv
id,question,acceptable_answers
q001,"What is the capital of France? (A) London (B) Paris (C) Berlin (D) Madrid","B"
q002,"Which planet is largest? (A) Earth (B) Mars (C) Jupiter (D) Venus","C"
```

### 3. Certify

```bash
trustgate certify
```

You see a pre-flight cost estimate first:

```
     Pre-flight Estimate           Cost / Reliability Tradeoff
┌─────────────────────────┬──────┐ ┌────┬──────────┬───────────┬────────────┐
│ Questions               │ 500  │ │  K │ Est.Cost │ Max Cost  │ Resolution │
│ Samples per question    │ 10   │ │  3 │ $9.00    │ $18.00    │   coarse   │
│ Max requests            │ 5000 │ │  5 │ $15.00   │ $30.00    │  moderate  │
│ Sequential stopping     │ ~50% │ │ 10←│ $30.00   │ $60.00    │    fine    │
│ Est. cost               │ $30  │ │ 20 │ $60.00   │ $120.00   │    fine    │
└─────────────────────────┴──────┘ └────┴──────────┴───────────┴────────────┘
Proceed? [Y/n]:
```

Then the certification result:

```
     TrustGate Certification Result
┌──────────────────────┬──────────┐
│ Reliability Level    │ 94.6%    │
│ M* (prediction set)  │ 1        │
│ Empirical Coverage   │ 0.956    │
│ Conditional Coverage │ 0.980    │
│ Capability Gap       │ 2.4%     │
│ Calibration items    │ 250      │
│ Test items           │ 250      │
│ Status               │ PASS     │
└──────────────────────┴──────────┘
```

Your model is 94.6% reliable at the 90% confidence level.

---

## Generic Endpoints (Agents, RAG, Custom APIs)

TrustGate works with **any endpoint you can ask a question to** — not just LLMs. You don't need to control the model or temperature:

```yaml
endpoint:
  url: "https://my-agent.example.com/api/ask"
  temperature: null                          # endpoint controls its own randomness
  headers:
    Authorization: "Bearer ${AGENT_API_KEY}" # env var expansion
  request_template:
    query: "{{question}}"                    # {{question}} replaced with question text
  response_path: "answer"                    # dot-notation path in JSON response
  cost_per_request: 0.03                     # for pre-flight cost estimate
```

---

## Human Calibration (No Ground Truth)

When you don't have a gold-standard dataset, human reviewers provide the labels. TrustGate samples K responses, canonicalizes them, and shows the reviewer all candidate answers for each question. The reviewer picks the acceptable one — this gives the exact nonconformity score needed for conformal calibration.

Answers are shown in **randomized order with no frequency or rank information** to prevent anchoring bias.

### Option A: Shareable HTML questionnaire (recommended)

Generate a self-contained HTML file and share it with anyone — email, Slack, Google Drive. No server needed. The reviewer opens it in any browser (works offline, works on mobile), reviews answers, and downloads `labels.json`.

```bash
# Step 1: Sample + generate questionnaire
trustgate calibrate --export questionnaire.html

# Step 2: Share questionnaire.html with your domain expert
#         They open it in a browser, pick answers, download labels.json
#         They send labels.json back to you (email, Slack, etc.)

# Step 3: Certify using the labels
trustgate certify --ground-truth labels.json
```

### Option B: Local web UI

For reviewers on the same network:

```bash
trustgate calibrate --serve --port 8080
```

### The review interface

```
┌──────────────────────────────────────────────────┐
│ █████████████████░░░░░░░  12/50  (24%)           │
│                                                  │
│  Question:                                       │
│  What is the standard treatment for              │
│  acute myocardial infarction?                    │
│                                                  │
│  Which answer is acceptable?                     │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Beta-blockers and bed rest              │    │
│  └──────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────┐    │
│  │  Aspirin + heparin + PCI                 │    │
│  └──────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────┐    │
│  │  Thrombolysis with tPA                   │    │
│  └──────────────────────────────────────────┘    │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐    │
│  │     None of these are correct            │    │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘    │
│                                                  │
│  Keyboard: 1-9 to pick, 0 for none              │
└──────────────────────────────────────────────────┘
```

No rank numbers. No percentages. Randomized order. The reviewer judges purely on content. 50 questions takes ~10 minutes.

---

## Using TrustGate in Production

TrustGate supports three deployment patterns:

### Pattern 1: Deployment gate (certify once, deploy with confidence)

The simplest pattern. Certify offline before shipping. If reliability is above your threshold, deploy. No per-query overhead at runtime.

```python
from theaios.trustgate import certify

result = certify(config_path="trustgate.yaml")
if result.reliability_level < 0.90:
    raise SystemExit("Reliability too low to deploy")
# Deploy with confidence — 94.6% of queries will get the right answer
```

From the CLI:
```bash
trustgate certify --yes --output json --output-file result.json
# Exit code 0 = PASS, 1 = FAIL
```

### Pattern 2: Runtime trust layer (per-query confidence)

Wrap your endpoint with `TrustGate` to get reliability metadata on every response.

**Passthrough mode** (default, cheap — 1 API call per query): attaches pre-computed reliability metadata without re-sampling.

```python
from theaios.trustgate import TrustGate, certify

# Certify offline (once)
result = certify(config_path="trustgate.yaml")

# At runtime: single call per query, reliability metadata attached
gate = TrustGate(config=config, certification=result)  # mode="passthrough"
response = gate.query("What is the treatment for X?")

response.answer              # "Aspirin + PCI"
response.reliability_level   # 0.946 (from the certification)
response.m_star              # 1
```

**Sampled mode** (expensive — K API calls per query): draws K samples per query and builds a per-query self-consistency profile. Use for high-stakes decisions where you need to know if *this specific query* is in the reliable region.

```python
gate = TrustGate(config=config, certification=result, mode="sampled")
response = gate.query("What is the treatment for X?")

response.answer          # "Aspirin + PCI" (top canonical answer)
response.prediction_set  # ["Aspirin + PCI"] — top-M* answers
response.consensus       # 0.8 (80% of K samples agreed)
response.margin          # 0.6 (gap between #1 and #2)
response.is_singleton    # True (prediction set size = 1 → high confidence)
```

### Pattern 3: Periodic recalibration

AI systems drift over time — model updates, data shifts, prompt changes. Re-run certification periodically to detect reliability degradation.

```bash
# In a cron job or CI schedule (weekly, after model updates, etc.)
trustgate certify --yes --output json --output-file latest_result.json
```

```python
# In your application: load the latest certification
import json
from theaios.trustgate import TrustGate, CertificationResult

with open("latest_result.json") as f:
    data = json.load(f)
result = CertificationResult(**data)  # or deserialize as needed
gate = TrustGate(config=config, certification=result)
```

The library doesn't impose a recalibration schedule — that's infrastructure (cron, CI, Airflow). TrustGate makes recertification cheap: cached responses mean re-runs only cost new API calls for questions that changed.

---

## Certify at the Decision Point

TrustGate measures **self-consistency** — how often the AI gives the same answer. This only works when answers are short enough to compare meaningfully.

**If your AI produces long outputs** (reports, essays, multi-paragraph analyses), raw self-consistency won't work: every sample will be unique.

The fix: **certify at the decision point, not the final output.** Most AI systems have a short, structured decision buried inside the long output:

| Your system | Long output | Certify on (the decision point) |
|-------------|------------|--------------------------------|
| SQL agent | English report from query results | The SQL query itself |
| Medical triage | Full patient summary with reasoning | The triage category (1–5) |
| Legal review | Multi-page contract analysis | The conclusion (approve / reject / escalate) |
| RAG pipeline | Synthesized answer with citations | The cited document IDs or key claim |
| Code agent | Full implementation with comments | Test pass/fail or function signature |
| Support bot | Detailed customer response | The intent classification or action taken |

Write a custom canonicalizer that extracts the decision:

```python
from theaios.trustgate import Canonicalizer, register_canonicalizer

@register_canonicalizer("sql_decision")
class SQLDecisionCanonicalizer(Canonicalizer):
    def canonicalize(self, question: str, answer: str) -> str:
        import re
        match = re.search(r"```sql\n(.+?)```", answer, re.DOTALL)
        return match.group(1).strip() if match else ""
```

TrustGate automatically warns when canonicalization is failing (all-unique answers, no consensus).

---

## Task Types & Canonicalization

| Task type | Canonicalizer | What it does | Example |
|-----------|--------------|-------------|---------|
| `mcq` | MCQ | Extracts chosen option letter | "I think B) Paris" → `"B"` |
| `numeric` | Numeric | Extracts final number | "The answer is $42.50" → `"42.5"` |
| `code_exec` | Code execution | Runs code in sandbox | Python code → `"pass"` / `"fail"` |
| `llm_judge` | LLM-as-judge | Asks a judge LLM | (question, answer) → `"correct"` / `"incorrect"` |
| `embedding` | Embedding clustering | Groups by semantic similarity | Free text → `"cluster_0"` |
| `custom` | Your own plugin | Whatever you need | Your logic → your canonical form |

---

## How It Works

```
1. SAMPLE        Ask the AI the same question K times
2. CANONICALIZE  Normalize raw responses into comparable forms
3. CALIBRATE     Human or ground truth labels → nonconformity scores
4. CERTIFY       Conformal prediction → reliability level with guarantee
```

**Self-consistency sampling:** Ask the same question K times. Group identical canonical answers and rank by frequency. The pattern of agreement measures how confident you should be.

**Conformal prediction:** Converts agreement patterns into a guaranteed reliability level. If TrustGate says "94.6% reliable at α=0.10," the model's top answer is correct for at least 94.6% of questions — this guarantee holds with 90% confidence, regardless of data distribution.

**Sequential stopping:** Hoeffding bounds detect when the answer pattern has stabilized and stop early, saving ~50% of API costs.

For the full theory: *Black-Box Reliability Certification for AI Agents via Self-Consistency Sampling and Conformal Calibration* (Mouzouni, 2026).

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
if diag.status == "poor":
    print(diag.warnings)

# Generate shareable questionnaire
trustgate.generate_questionnaire(questions, profiles, "questionnaire.html")

# Runtime trust layer
gate = trustgate.TrustGate(config=config, certification=result)
response = gate.query("What is 2+2?")
```

---

## Configuration Reference

<details>
<summary>Full <code>trustgate.yaml</code> reference</summary>

```yaml
# The AI endpoint to certify
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4.1"                    # optional for generic endpoints
  temperature: 0.7                     # null = endpoint controls randomness
  api_key_env: "OPENAI_API_KEY"        # env var name (not the key itself)
  provider: "openai"                   # auto-detected from URL if omitted
  max_tokens: 4096

  # Generic endpoint support (agents, RAG, custom APIs)
  # headers:
  #   Authorization: "Bearer ${MY_KEY}"
  # request_template:
  #   query: "{{question}}"
  # response_path: "answer"
  # cost_per_request: 0.03

# Sampling parameters
sampling:
  k_max: 20                            # maximum samples per question
  k_fixed: 10                          # fixed K (overrides k_max if set)
  sequential_stopping: true             # stop early when answer stabilizes
  delta: 0.05                           # confidence parameter for stopping
  max_concurrent: 50                    # parallel requests
  timeout: 120.0                        # per-request timeout (seconds)
  retries: 10                           # max retries on failure

# How to normalize raw responses
canonicalization:
  type: "mcq"                           # mcq, numeric, code_exec, llm_judge, embedding, custom

# Conformal calibration parameters
calibration:
  alpha_values: [0.01, 0.05, 0.10, 0.15, 0.20]
  n_cal: 500
  n_test: 500

# Questions to evaluate
questions:
  file: "questions.csv"
```

</details>

---

## CLI Reference

```
trustgate certify     Certify an endpoint's reliability
trustgate calibrate   Sample + collect human labels (--serve, --export)
trustgate compare     Compare reliability across models
trustgate sample      Sample responses only (no calibration)
trustgate cache       Manage the response cache (stats, clear)
trustgate version     Show version
```

---

## Contributing

```bash
git clone https://github.com/Cohorte-ai/trustgate.git
cd trustgate
pip install -e ".[dev,serve,judge]"
pytest
ruff check src/ tests/
```

---

## Citation

```bibtex
@article{mouzouni2026trustgate,
  title={TrustGate: Black-Box AI Reliability Certification via
         Self-Consistency Sampling and Conformal Calibration},
  author={Mouzouni, Charafeddine},
  year={2026}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
