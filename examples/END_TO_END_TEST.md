# TrustGate — End-to-End Test Guide

Test every feature from a fresh install. No repo cloning needed.

**Requirements:** Python 3.10+, an OpenAI API key
**Estimated cost:** ~$0.50–$2.00 depending on K and number of questions

---

## Setup

### macOS / Linux

```bash
mkdir trustgate-test && cd trustgate-test
python3 -m venv .venv
source .venv/bin/activate
pip install theaios-trustgate
export LLM_API_KEY="sk-your-key-here"   # any OpenAI-compatible API key
```

### Windows (PowerShell)

```powershell
mkdir trustgate-test; cd trustgate-test
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install theaios-trustgate
$env:LLM_API_KEY="sk-your-key-here"   # any OpenAI-compatible API key
```


---

## Generate test data

Create a config file and test questions. On macOS/Linux use the commands below. On Windows, create these two files manually or use the Python script at the end of this section.

```bash
cat > trustgate.yaml << 'EOF'
# Works with any OpenAI-compatible API (OpenAI, Together, Ollama, LiteLLM, etc.)
# Auth option 1: env var (standard Bearer token)
#   api_key_env: "LLM_API_KEY"
# Auth option 2: custom headers (LiteLLM, Azure, etc.)
#   headers:
#     API-Key: "your-key-here"
endpoint:
  url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4.1-mini"
  api_key_env: "LLM_API_KEY"

canonicalization:
  type: "llm"
  judge_endpoint:
    url: "https://api.openai.com/v1/chat/completions"
    model: "gpt-4.1-nano"
    api_key_env: "LLM_API_KEY"

sampling:
  k_fixed: 5
  sequential_stopping: true

calibration:
  alpha_values: [0.01, 0.05, 0.10, 0.20]
  n_cal: 25
  n_test: 25

questions:
  file: "questions.csv"
EOF

cat > questions.csv << 'EOF'
id,question,acceptable_answers
q01,What is the capital of France?,Paris
q02,What is 2+2?,4
q03,Who painted the Mona Lisa?,Da Vinci
q04,What is the chemical symbol for gold?,Au
q05,How many continents are there?,7
q06,What is the largest ocean?,Pacific
q07,Who wrote Romeo and Juliet?,Shakespeare
q08,What is the boiling point of water in Celsius?,100
q09,What is the square root of 144?,12
q10,Which planet is the largest?,Jupiter
q11,What is the capital of Japan?,Tokyo
q12,Who developed the theory of relativity?,Einstein
q13,What is the chemical formula for water?,H2O
q14,What is the tallest mountain in the world?,Everest
q15,How many minutes are in an hour?,60
q16,What is the smallest prime number?,2
q17,What is the longest river in the world?,Nile
q18,Which element has atomic number 1?,Hydrogen
q19,What is the capital of Australia?,Canberra
q20,Who discovered penicillin?,Fleming
q21,What is the speed of light approximately?,300000 km/s
q22,What is the capital of Germany?,Berlin
q23,How many bones in the adult human body?,206
q24,What is the largest mammal?,Blue whale
q25,Who invented the telephone?,Bell
q26,What is the pH of pure water?,7
q27,What is the capital of Brazil?,Brasilia
q28,How many planets in our solar system?,8
q29,What is the hardest natural substance?,Diamond
q30,Who painted Starry Night?,Van Gogh
q31,What is the capital of Canada?,Ottawa
q32,What is the freezing point of water in Celsius?,0
q33,How many legs does a spider have?,8
q34,What is the chemical symbol for sodium?,Na
q35,Who was the first person to walk on the moon?,Armstrong
q36,What is the capital of Italy?,Rome
q37,What is the cube root of 27?,3
q38,Who wrote Don Quixote?,Cervantes
q39,What is the unit of electrical resistance?,Ohm
q40,What is the capital of India?,New Delhi
q41,How many strings does a standard guitar have?,6
q42,What is the largest desert in the world?,Antarctic
q43,What is the capital of Mexico?,Mexico City
q44,Who composed the Four Seasons?,Vivaldi
q45,What is the smallest continent?,Australia
q46,What is the capital of Egypt?,Cairo
q47,What is the atomic number of carbon?,6
q48,Which is the hottest planet?,Venus
q49,How many colors are in a rainbow?,7
q50,What is the capital of Spain?,Madrid
EOF
```

**Windows alternative** — run this Python script to create both files:

```python
python -c "
config = '''endpoint:
  url: https://api.openai.com/v1/chat/completions
  model: gpt-4.1-mini
  api_key_env: LLM_API_KEY
canonicalization:
  type: llm
  judge_endpoint:
    url: https://api.openai.com/v1/chat/completions
    model: gpt-4.1-nano
    api_key_env: LLM_API_KEY
sampling:
  k_fixed: 5
  sequential_stopping: true
calibration:
  alpha_values: [0.01, 0.05, 0.10, 0.20]
  n_cal: 25
  n_test: 25
questions:
  file: questions.csv
'''
open('trustgate.yaml','w').write(config)
questions = 'id,question,acceptable_answers\n'
qs = [('q01','What is the capital of France?','Paris'),('q02','What is 2+2?','4'),('q03','Who painted the Mona Lisa?','Da Vinci'),('q04','What is the chemical symbol for gold?','Au'),('q05','How many continents are there?','7'),('q06','What is the largest ocean?','Pacific'),('q07','Who wrote Romeo and Juliet?','Shakespeare'),('q08','What is the boiling point of water in Celsius?','100'),('q09','What is the square root of 144?','12'),('q10','Which planet is the largest?','Jupiter'),('q11','What is the capital of Japan?','Tokyo'),('q12','Who developed the theory of relativity?','Einstein'),('q13','What is the chemical formula for water?','H2O'),('q14','What is the tallest mountain in the world?','Everest'),('q15','How many minutes are in an hour?','60'),('q16','What is the smallest prime number?','2'),('q17','What is the longest river in the world?','Nile'),('q18','Which element has atomic number 1?','Hydrogen'),('q19','What is the capital of Australia?','Canberra'),('q20','Who discovered penicillin?','Fleming'),('q21','What is the speed of light approximately?','300000 km/s'),('q22','What is the capital of Germany?','Berlin'),('q23','How many bones in the adult human body?','206'),('q24','What is the largest mammal?','Blue whale'),('q25','Who invented the telephone?','Bell'),('q26','What is the pH of pure water?','7'),('q27','What is the capital of Brazil?','Brasilia'),('q28','How many planets in our solar system?','8'),('q29','What is the hardest natural substance?','Diamond'),('q30','Who painted Starry Night?','Van Gogh'),('q31','What is the capital of Canada?','Ottawa'),('q32','What is the freezing point of water in Celsius?','0'),('q33','How many legs does a spider have?','8'),('q34','What is the chemical symbol for sodium?','Na'),('q35','Who was the first person to walk on the moon?','Armstrong'),('q36','What is the capital of Italy?','Rome'),('q37','What is the cube root of 27?','3'),('q38','Who wrote Don Quixote?','Cervantes'),('q39','What is the unit of electrical resistance?','Ohm'),('q40','What is the capital of India?','New Delhi'),('q41','How many strings does a standard guitar have?','6'),('q42','What is the largest desert in the world?','Antarctic'),('q43','What is the capital of Mexico?','Mexico City'),('q44','Who composed the Four Seasons?','Vivaldi'),('q45','What is the smallest continent?','Australia'),('q46','What is the capital of Egypt?','Cairo'),('q47','What is the atomic number of carbon?','6'),('q48','Which is the hottest planet?','Venus'),('q49','How many colors are in a rainbow?','7'),('q50','What is the capital of Spain?','Madrid')]
for qid,q,a in qs: questions+=f'{qid},{q},{a}\n'
open('questions.csv','w').write(questions)
print('Created trustgate.yaml and questions.csv')
"
```

---

# Part 1: CLI

## 1. Version & Help

```bash
trustgate version
trustgate --help
trustgate certify --help
```

Expected: version number, all commands listed, all flags shown (including `--alpha`, `--concurrency`, `--auto-judge`, `--min-reliability`).

---

## 2. Pre-flight Estimate + K Selection

```bash
trustgate certify
```

Expected:
- "Measuring API latency..." spinner
- Pre-flight Estimate table with measured latency and estimated time
- Cost / Reliability Tradeoff table with Est. Time per K
- Prompt: `Proceed? Enter Y, N, or a number to change K`
- Type `N` to abort

---

## 3. Certify with Ground Truth

```bash
trustgate certify --yes
```

Expected:
- Spinner with estimated time
- TrustGate Certification Result table
- Plain-English explanation of Reliability Level and M*
- Docs link

---

## 4. Output Formats

```bash
# JSON
trustgate certify --yes --output json

# CSV
trustgate certify --yes --output csv

# To file
trustgate certify --yes --output json --output-file result.json
cat result.json
```

---

## 5. Verbose (Per-Alpha Breakdown)

```bash
trustgate certify --yes --verbose
```

Expected: result table + Coverage by Alpha table.

---

## 6. Change Alpha (M* Confidence Level)

```bash
# 99% confidence — stricter, M* may be larger
trustgate certify --yes --alpha 0.01

# 90% confidence — looser, M* may be smaller
trustgate certify --yes --alpha 0.10
```

---

## 7. Cost Per Request

```bash
trustgate certify --cost-per-request 0.05
```

Expected: pre-flight shows $0.0500 per request. Type `N` to abort.

---

## 8. Concurrency Control

```bash
# Conservative (safe for rate-limited APIs)
trustgate certify --yes --concurrency 5

# Aggressive (faster, may hit rate limits)
trustgate certify --yes --concurrency 30
```

---

## 9. CI/CD Gating

```bash
# Should PASS
trustgate certify --yes --min-reliability 50
echo "Exit code: $?"

# Should FAIL
trustgate certify --yes --min-reliability 99
echo "Exit code: $?"
```

Expected: exit code 0 for PASS, 1 for FAIL.

---

## 10. Sample Only

```bash
trustgate sample --questions questions.csv
```

---

## 11. Cache Management

```bash
trustgate cache stats
trustgate cache clear
```

---

## 12. Calibrate: Export HTML Questionnaire

```bash
trustgate calibrate --export questionnaire.html --yes
open questionnaire.html   # macOS (or xdg-open on Linux)
```

Expected:
- "Selected N questions for human calibration (out of 50 total)"
- HTML file opens in browser
- Questions with randomized answer buttons (no ranks, no frequencies)
- "Download Labels" at the end

---

## 13. Calibrate: Local Web UI

```bash
trustgate calibrate --serve --port 9999 --yes
```

Expected: browser opens, review UI, admin at `/admin`. Press Ctrl+C when done.

---

## 14. Certify with Human Labels

After reviewing in the questionnaire or web UI:

```bash
trustgate certify --ground-truth labels.json --yes
```

---

## 15. Auto-Judge (LLM Calibration)

```bash
trustgate certify --auto-judge --yes
```

Expected: "Auto-judging with LLM..." then certification result.

---

## 16. Compare Models

```bash
# Generate ground truth file
python3 -c "
import csv, json
labels = {}
with open('questions.csv') as f:
    for row in csv.DictReader(f):
        labels[row['id']] = row['acceptable_answers']
json.dump(labels, open('ground_truth.json', 'w'), indent=2)
"

trustgate compare \
  --models gpt-4.1-mini,gpt-4.1-nano \
  --task-type llm \
  --questions questions.csv \
  --ground-truth ground_truth.json
```

---

# Part 2: Python API

Run each block with `python3 -c "..."` or in a Python shell.

## 17. Certify

```python
from theaios import trustgate

result = trustgate.certify(config_path="trustgate.yaml")
print(f"Reliability: {result.reliability_level:.1%}")
print(f"M* (at {1-result.target_alpha:.0%} confidence): {result.m_star}")
print(f"Coverage: {result.coverage:.3f}")
print(f"Capability Gap: {result.capability_gap:.1%}")
```

---

## 18. Sample and Profile

```python
from theaios.trustgate import sample_and_profile
from theaios.trustgate.config import load_config, load_questions

config = load_config("trustgate.yaml")
questions = load_questions("questions.csv")

profiles = sample_and_profile(config, questions)
for qid in list(profiles.keys())[:5]:
    print(f"{qid}: {profiles[qid]}")
```

---

## 19. Profile Diagnostic

```python
from theaios.trustgate import diagnose_profiles, sample_and_profile
from theaios.trustgate.config import load_config, load_questions

config = load_config("trustgate.yaml")
questions = load_questions("questions.csv")
profiles = sample_and_profile(config, questions)

diag = diagnose_profiles(profiles)
print(f"Status: {diag.status}")
print(f"Mean consensus: {diag.mean_consensus:.2f}")
print(f"Warnings: {diag.warnings}")
```

---

## 20. Generate Questionnaire

```python
from theaios.trustgate import sample_and_profile, generate_questionnaire
from theaios.trustgate.config import load_config, load_questions

config = load_config("trustgate.yaml")
questions = load_questions("questions.csv")
profiles = sample_and_profile(config, questions)

path = generate_questionnaire(questions, profiles, "test_questionnaire.html")
print(f"Generated: {path}")
```

---

## 21. Auto-Judge Labels

```python
from theaios.trustgate import sample_and_profile, auto_judge_labels
from theaios.trustgate.config import load_config, load_questions

config = load_config("trustgate.yaml")
questions = load_questions("questions.csv")
profiles = sample_and_profile(config, questions)

q_texts = {q.id: q.text for q in questions}
labels = auto_judge_labels(q_texts, profiles, config.canonicalization.judge_endpoint)
print(f"Labeled {len(labels)} questions")
for qid in list(labels.keys())[:5]:
    print(f"  {qid}: {labels[qid]}")
```

---

## 22. TrustGate Runtime — Passthrough

```python
from theaios.trustgate import TrustGate, certify
from theaios.trustgate.config import load_config

config = load_config("trustgate.yaml")
result = certify(config=config)

gate = TrustGate(config=config, certification=result)
response = gate.query("What is the capital of France?")
print(f"Answer: {response.answer}")
print(f"Reliability: {response.reliability_level:.1%}")
print(f"Mode: {response.mode}")
```

---

## 23. TrustGate Runtime — Sampled

```python
from theaios.trustgate import TrustGate, certify
from theaios.trustgate.config import load_config

config = load_config("trustgate.yaml")
result = certify(config=config)

gate = TrustGate(config=config, certification=result, mode="sampled")
response = gate.query("What is the capital of France?")
print(f"Answer: {response.answer}")
print(f"Prediction set: {response.prediction_set}")
print(f"Consensus: {response.consensus:.0%}")
print(f"Singleton: {response.is_singleton}")
```

---

## 24. Reporting

```python
from theaios.trustgate import certify
from theaios.trustgate.reporting import export_json, export_csv, print_certification_result

result = certify(config_path="trustgate.yaml")
print_certification_result(result, verbose=True)
print(export_json(result)[:200] + "...")
```

---

## 25. Custom Canonicalizer

```python
from theaios.trustgate import Canonicalizer, register_canonicalizer, get_canonicalizer

@register_canonicalizer("upper")
class UpperCanonicalizer(Canonicalizer):
    def canonicalize(self, question, answer):
        return self.preprocess(answer).upper()

c = get_canonicalizer("upper")
print(c.canonicalize("Q?", "The answer is hello"))

from theaios.trustgate.canonicalize import list_canonicalizers
print(f"All canonicalizers: {list_canonicalizers()}")
```

---

## 26. Built-in Datasets

```python
from theaios.trustgate.datasets import load_mmlu

questions = load_mmlu(subjects=["abstract_algebra"], n=5)
for q in questions:
    print(f"{q.id}: {q.text[:60]}...")
```

---

# Summary Checklist

| # | Feature | Type | Status |
|---|---------|------|--------|
| 1 | Version/help | CLI | |
| 2 | Pre-flight + K selection | CLI | |
| 3 | Certify with ground truth | CLI | |
| 4 | JSON/CSV/file output | CLI | |
| 5 | Verbose per-alpha breakdown | CLI | |
| 6 | --alpha for M* confidence | CLI | |
| 7 | Cost per request | CLI | |
| 8 | Concurrency control | CLI | |
| 9 | CI/CD gating (--min-reliability) | CLI | |
| 10 | Sample only | CLI | |
| 11 | Cache stats/clear | CLI | |
| 12 | Export HTML questionnaire | CLI | |
| 13 | Local web UI | CLI | |
| 14 | Certify with human labels | CLI | |
| 15 | Auto-judge (--auto-judge) | CLI | |
| 16 | Model comparison | CLI | |
| 17 | certify() | Python | |
| 18 | sample_and_profile() | Python | |
| 19 | diagnose_profiles() | Python | |
| 20 | generate_questionnaire() | Python | |
| 21 | auto_judge_labels() | Python | |
| 22 | TrustGate passthrough | Python | |
| 23 | TrustGate sampled | Python | |
| 24 | Reporting | Python | |
| 25 | Custom canonicalizer | Python | |
| 26 | Built-in datasets | Python | |

---

## Cleanup

```bash
# macOS / Linux
cd .. && rm -rf trustgate-test

# Windows (PowerShell)
cd ..; Remove-Item -Recurse -Force trustgate-test
```
