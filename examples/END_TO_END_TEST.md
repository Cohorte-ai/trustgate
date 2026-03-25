# TrustGate — End-to-End Test Guide

Test every feature of the library. Requires an OpenAI API key.

**Estimated cost:** ~$1–$2 (120 questions × multiple sampling rounds × gpt-4.1-mini)
**Estimated time:** ~10 minutes

---

## Setup

```bash
# Create a venv and install from source (for testing local changes)
cd /path/to/trustgate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,serve,judge]"

# Or install from PyPI
pip install "theaios-trustgate[serve,judge]"

# Set your API key
export OPENAI_API_KEY="sk-your-key-here"
```

---

# Part 1: CLI

## 1. Version & Help

```bash
trustgate version
trustgate --help
trustgate certify --help
trustgate calibrate --help
```

Expected: version shown, all commands listed (`certify`, `calibrate`, `compare`, `sample`, `cache`, `version`). Certify help shows `--auto-judge`, `--cost-per-request`, `--yes`, `--task-type` with `llm` option.

---

## 2. Pre-flight Cost Estimate (Interactive)

```bash
trustgate certify --config examples/test_config_e2e.yaml
```

Expected:
- Pre-flight Estimate table (questions, K, requests, cost)
- Cost / Reliability Tradeoff table (K=3,5,10,15,20)
- Prompt: `Proceed? Enter Y to run, N to abort, or a number to change K`
- Type `N` to abort (no API calls)
- Type `10` to change K to 10

---

## 3. Certify with Ground Truth

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --yes
```

Expected:
- Spinner while running
- TrustGate Certification Result table
- Docs link below the table
- Status = PASS

---

## 4. Output Formats

```bash
# JSON
trustgate certify --config examples/test_config_e2e.yaml --yes --output json

# CSV
trustgate certify --config examples/test_config_e2e.yaml --yes --output csv

# To file
trustgate certify --config examples/test_config_e2e.yaml --yes --output json --output-file result.json
cat result.json
```

---

## 5. Verbose (Per-Alpha Breakdown)

```bash
trustgate certify --config examples/test_config_e2e.yaml --yes --verbose
```

Expected: main result table + "Coverage by Alpha" table.

---

## 6. Cost Per Request Flag

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --cost-per-request 0.05
```

Expected: Pre-flight table shows `$0.0500` per request. Type `N` to abort.

---

## 7. Sample Only

```bash
trustgate sample \
  --config examples/test_config_e2e.yaml \
  --questions examples/test_questions_120.csv
```

Expected: "Sampled X responses for 120 questions" + cache hit stats.

---

## 8. Cache Management

```bash
trustgate cache stats
trustgate cache clear
```

---

## 9. Calibrate: Export HTML Questionnaire

```bash
trustgate calibrate \
  --config examples/test_config_e2e.yaml \
  --export questionnaire.html \
  --yes
```

Expected: "Questionnaire exported to questionnaire.html"

```bash
open questionnaire.html    # macOS
```

Expected in browser:
- Progress bar
- Question text (clean, no MCQ options)
- Answer buttons with enriched text (e.g., "paris" not just a letter)
- Randomized order, NO frequencies, NO ranks
- "None of these are correct" button
- Keyboard shortcuts (1-9, 0)
- "Download Labels" button at the end → saves `labels.json`

---

## 10. Calibrate: Local Web UI

```bash
trustgate calibrate \
  --config examples/test_config_e2e.yaml \
  --serve \
  --port 9999 \
  --yes
```

Expected:
- Browser opens to `http://localhost:9999`
- Same review interface as HTML export
- Admin panel at `http://localhost:9999/admin` (shows rank stats)
- Press Ctrl+C to stop

---

## 11. Calibrate: Profile Dump (no --serve or --export)

```bash
trustgate calibrate \
  --config examples/test_config_e2e.yaml \
  --yes
```

Expected: creates `calibration_labels_profiles.json` with ranked answers.

```bash
python3 -m json.tool calibration_labels_profiles.json | head -20
```

---

## 12. Certify with External Ground Truth

```bash
python3 -c "
import csv, json
labels = {}
with open('examples/test_questions_120.csv') as f:
    for row in csv.DictReader(f):
        labels[row['id']] = row['acceptable_answers']
json.dump(labels, open('ground_truth.json', 'w'), indent=2)
print(f'Wrote {len(labels)} labels')
"

trustgate certify \
  --config examples/test_config_e2e.yaml \
  --ground-truth ground_truth.json \
  --yes
```

---

## 13. Auto-Judge (LLM Calibration)

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --auto-judge \
  --yes
```

Expected:
- "Auto-judging with LLM..." spinner
- "Auto-judge labeled N questions."
- Certification result (may differ from ground-truth result due to judge bias)

---

## 14. Compare Models

```bash
trustgate compare \
  --config examples/test_config_e2e.yaml \
  --models gpt-4.1-mini,gpt-4.1-nano \
  --task-type llm \
  --questions examples/test_questions_120.csv \
  --ground-truth ground_truth.json
```

Expected: side-by-side table with reliability, M*, coverage for each model.

---

# Part 2: Python API

## 15. Certify

```python
from theaios import trustgate

result = trustgate.certify(config_path="examples/test_config_e2e.yaml")
print(f"Reliability: {result.reliability_level:.1%}")
print(f"M*: {result.m_star}")
print(f"Coverage: {result.coverage:.3f}")
print(f"Capability Gap: {result.capability_gap:.1%}")
```

---

## 16. Sample and Profile

```python
from theaios.trustgate import sample_and_profile
from theaios.trustgate.config import load_config, load_questions

config = load_config("examples/test_config_e2e.yaml")
questions = load_questions("examples/test_questions_120.csv")

profiles = sample_and_profile(config, questions)
for qid in list(profiles.keys())[:5]:
    print(f"{qid}: {profiles[qid]}")
```

Expected: ranked profiles per question.

---

## 17. Profile Diagnostic

```python
from theaios.trustgate import sample_and_profile, diagnose_profiles
from theaios.trustgate.config import load_config, load_questions

config = load_config("examples/test_config_e2e.yaml")
questions = load_questions("examples/test_questions_120.csv")

profiles = sample_and_profile(config, questions)
diag = diagnose_profiles(profiles)
print(f"Status: {diag.status}")
print(f"Mean consensus: {diag.mean_consensus:.2f}")
print(f"Mean classes: {diag.mean_n_classes:.1f}")
print(f"All-unique fraction: {diag.frac_all_unique:.1%}")
print(f"Warnings: {diag.warnings}")
```

Expected: status = "good" or "weak" depending on canonicalization.

---

## 18. Generate Questionnaire

```python
from theaios.trustgate import sample_and_profile, generate_questionnaire
from theaios.trustgate.config import load_config, load_questions

config = load_config("examples/test_config_e2e.yaml")
questions = load_questions("examples/test_questions_120.csv")
profiles = sample_and_profile(config, questions)

path = generate_questionnaire(questions, profiles, "test_questionnaire.html")
print(f"Generated: {path}")
```

Then open `test_questionnaire.html` in a browser.

---

## 19. Auto-Judge Labels (Python API)

```python
from theaios.trustgate import sample_and_profile, auto_judge_labels
from theaios.trustgate.config import load_config, load_questions

config = load_config("examples/test_config_e2e.yaml")
questions = load_questions("examples/test_questions_120.csv")

profiles = sample_and_profile(config, questions)
q_texts = {q.id: q.text for q in questions}

labels = auto_judge_labels(
    q_texts, profiles,
    config.canonicalization.judge_endpoint,
)
print(f"Labeled {len(labels)} questions")
for qid in list(labels.keys())[:5]:
    print(f"  {qid}: {labels[qid]}")
```

Expected: labels like `{"q001": "correct", "q002": "correct", ...}` (from llm_judge canonicalization) or `{"q001": "paris", ...}` (from llm canonicalization).

---

## 20. TrustGate Runtime — Passthrough

```python
from theaios.trustgate import TrustGate, certify
from theaios.trustgate.config import load_config

config = load_config("examples/test_config_e2e.yaml")
result = certify(config=config)

gate = TrustGate(config=config, certification=result)
print(f"Mode: {gate.mode}")  # passthrough
print(f"Reliability: {gate.reliability_level:.1%}")

response = gate.query("What is the capital of France?")
print(f"Answer: {response.answer}")
print(f"Reliability: {response.reliability_level:.1%}")
print(f"Samples: {response.n_samples}")  # 1 (single call)
```

---

## 21. TrustGate Runtime — Sampled

```python
from theaios.trustgate import TrustGate, certify
from theaios.trustgate.config import load_config

config = load_config("examples/test_config_e2e.yaml")
result = certify(config=config)

gate = TrustGate(config=config, certification=result, mode="sampled")
response = gate.query("What is the capital of France?")

print(f"Answer: {response.answer}")
print(f"Prediction set: {response.prediction_set}")
print(f"Consensus: {response.consensus:.0%}")
print(f"Margin: {response.margin:.2f}")
print(f"Singleton: {response.is_singleton}")
print(f"Samples: {response.n_samples}")
print(f"Profile: {response.profile}")
```

---

## 22. Reporting

```python
from theaios.trustgate import certify
from theaios.trustgate.reporting import export_json, export_csv, print_certification_result

result = certify(config_path="examples/test_config_e2e.yaml")

# Console output
print_certification_result(result, verbose=True)

# JSON
print(export_json(result)[:200] + "...")

# CSV
print(export_csv(result)[:200] + "...")
```

---

## 23. Custom Canonicalizer

```python
from theaios.trustgate import Canonicalizer, register_canonicalizer, get_canonicalizer

@register_canonicalizer("upper")
class UpperCanonicalizer(Canonicalizer):
    def canonicalize(self, question, answer):
        return self.preprocess(answer).upper()

c = get_canonicalizer("upper")
print(c.canonicalize("Q?", "The answer is hello"))
# Expected: THE ANSWER IS HELLO

# Verify all 6 built-in canonicalizers
from theaios.trustgate.canonicalize import list_canonicalizers
print(f"All canonicalizers: {list_canonicalizers()}")
# Expected: ['code_exec', 'embedding', 'llm', 'llm_judge', 'mcq', 'numeric', 'upper']
```

---

## 24. Built-in Datasets

```python
from theaios.trustgate.datasets import load_mmlu, load_gsm8k

# MMLU (requires network for first download)
questions = load_mmlu(subjects=["abstract_algebra"], n=10)
for q in questions[:3]:
    print(f"{q.id}: {q.text[:80]}...")
    print(f"  Answer: {q.acceptable_answers}")
```

---

## 25. CI/CD Gating: --min-reliability

```bash
# Should PASS (threshold low)
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --min-reliability 50 --yes
echo "Exit code: $?"
# Expected: exit code 0

# Should FAIL (threshold high)
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --min-reliability 99 --yes
echo "Exit code: $?"
# Expected: exit code 1 + "FAIL: reliability X% < threshold 99.0%"
```

---

## 26. CI/CD: JSON Output + Gating Combined

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --min-reliability 80 \
  --output json \
  --output-file result.json \
  --yes

# Parse result in CI
python3 -c "
import json, sys
r = json.load(open('result.json'))
print(f'Reliability: {r[\"reliability_level\"]:.1%}')
print(f'M*: {r[\"m_star\"]}')
if r['reliability_level'] < 0.80:
    print('WOULD BLOCK DEPLOYMENT')
    sys.exit(1)
print('DEPLOYMENT APPROVED')
"
```

---

## 27. Status Display (PASS/FAIL/UNCERTAIN)

```bash
# Run and check that Status shows:
# - PASS (green) when reliability > 0 and coverage meets target
# - FAIL (red) when reliability = 0%
# - UNCERTAIN (yellow) otherwise
trustgate certify --config examples/test_config_e2e.yaml --yes
```

---

# Summary Checklist

| # | Feature | Type | Status |
|---|---------|------|--------|
| 1 | Version/help | CLI | |
| 2 | Pre-flight estimate + K selection | CLI | |
| 3 | Certify with ground truth | CLI | |
| 4 | JSON/CSV/file output | CLI | |
| 5 | Verbose per-alpha breakdown | CLI | |
| 6 | Cost per request flag | CLI | |
| 7 | Sample only | CLI | |
| 8 | Cache stats/clear | CLI | |
| 9 | Export HTML questionnaire | CLI | |
| 10 | Local web UI + admin | CLI | |
| 11 | Profile dump | CLI | |
| 12 | External ground truth | CLI | |
| 13 | Auto-judge (LLM calibration) | CLI | |
| 14 | Model comparison | CLI | |
| 15 | certify() | Python | |
| 16 | sample_and_profile() | Python | |
| 17 | diagnose_profiles() | Python | |
| 18 | generate_questionnaire() | Python | |
| 19 | auto_judge_labels() | Python | |
| 20 | TrustGate passthrough | Python | |
| 21 | TrustGate sampled | Python | |
| 22 | Reporting (console/json/csv) | Python | |
| 23 | Custom canonicalizer | Python | |
| 24 | Built-in datasets | Python | |
| 25 | --min-reliability gating | CLI | |
| 26 | JSON + gating combined | CLI | |
| 27 | PASS/FAIL/UNCERTAIN status | CLI | |
