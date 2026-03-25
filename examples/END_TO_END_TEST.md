# TrustGate — End-to-End Test Guide

Run every feature of the library from scratch. Requires an OpenAI API key.

**Estimated cost:** ~$0.50–$1.00 (120 questions × 5 samples × gpt-4.1-mini)
**Estimated time:** ~5 minutes

---

## Setup

```bash
# Install from PyPI
pip install theaios-trustgate

# Set your API key
export OPENAI_API_KEY="sk-your-key-here"

# Go to the repo root (where examples/ lives)
cd /path/to/trustgate
```

---

## 1. CLI: Version & Help

```bash
trustgate version
trustgate --help
trustgate certify --help
trustgate calibrate --help
```

Expected: version `0.1.0`, all commands listed, all flags shown.

---

## 2. Certify with Ground Truth (Full Pipeline)

This runs the complete pipeline: sample → canonicalize → calibrate → certify.

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --yes
```

Expected output:
- Pre-flight estimate (skipped with `--yes`)
- TrustGate Certification Result table
- Reliability Level should be **90%+** (these are easy questions for gpt-4.1-mini)
- M* = 1
- Status = PASS

---

## 3. Certify with JSON Output

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --yes \
  --output json
```

Expected: JSON with `reliability_level`, `m_star`, `coverage`, etc.

---

## 4. Certify with CSV Output

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --yes \
  --output csv
```

Expected: CSV row with all metrics.

---

## 5. Certify with Output File

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --yes \
  --output json \
  --output-file /tmp/trustgate_result.json

cat /tmp/trustgate_result.json
```

Expected: result written to file + readable JSON.

---

## 6. Certify with Verbose (Per-Alpha Breakdown)

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --yes \
  --verbose
```

Expected: main result table + "Coverage by Alpha" table showing each alpha value.

---

## 7. Pre-flight Cost Estimate (Interactive)

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml
```

Expected:
- Pre-flight Estimate table
- Cost / Reliability Tradeoff table (K=3,5,10,15,20)
- `Proceed? [Y/n]:` prompt
- Press `n` to abort (no API calls made)

---

## 8. Cost Per Request (Generic Endpoint Flag)

```bash
trustgate certify \
  --config examples/test_config_e2e.yaml \
  --cost-per-request 0.01
```

Expected: Pre-flight table shows `$0.0100` per request and calculated totals.

---

## 9. Sample Only (No Calibration)

```bash
trustgate sample \
  --config examples/test_config_e2e.yaml \
  --questions examples/test_questions_120.csv
```

Expected: "Sampled X responses for 120 questions" + cache hit stats.

---

## 10. Cache Stats & Clear

```bash
# Check cache
trustgate cache stats

# Clear (will ask confirmation)
trustgate cache clear
```

Expected: entry count > 0 after step 9, then 0 after clear.

---

## 11. Calibrate: Export HTML Questionnaire

```bash
trustgate calibrate \
  --config examples/test_config_e2e.yaml \
  --export /tmp/questionnaire.html \
  --yes
```

Expected:
- "Sampling responses and building self-consistency profiles..."
- "Profiled 120 questions."
- "Questionnaire exported to /tmp/questionnaire.html"

**Open it:**

```bash
open /tmp/questionnaire.html    # macOS
# or: xdg-open /tmp/questionnaire.html  (Linux)
```

Expected in browser:
- Progress bar
- Question text
- Multiple answer buttons (randomized order, NO frequencies, NO rank numbers)
- "None of these are correct" button
- Keyboard shortcuts (1-9, 0)
- Click a few answers → they advance automatically
- Click "Download Labels" at the end → saves `labels.json`

---

## 12. Calibrate: Local Web UI

```bash
pip install "theaios-trustgate[serve]"

trustgate calibrate \
  --config examples/test_config_e2e.yaml \
  --serve \
  --port 9999 \
  --yes
```

Expected:
- Browser opens to `http://localhost:9999`
- Same review interface as the HTML (but live server)
- Admin panel at `http://localhost:9999/admin`
- Press Ctrl+C to stop

---

## 13. Calibrate Without --serve or --export (Profile Dump)

```bash
trustgate calibrate \
  --config examples/test_config_e2e.yaml \
  --output /tmp/labels.json \
  --yes
```

Expected:
- "Profiled 120 questions."
- Creates `/tmp/labels_profiles.json` with ranked answers per question

```bash
cat /tmp/labels_profiles.json | python3 -m json.tool | head -20
```

---

## 14. Certify with External Ground Truth File

```bash
# Create a ground truth file manually
python3 -c "
import json
labels = {f'q{i:03d}': row.split(',')[-1].strip().strip('\"')
          for i, row in enumerate(open('examples/test_questions_120.csv').readlines()[1:], 1)}
json.dump(labels, open('/tmp/ground_truth.json', 'w'), indent=2)
print(f'Wrote {len(labels)} labels')
"

trustgate certify \
  --config examples/test_config_e2e.yaml \
  --ground-truth /tmp/ground_truth.json \
  --yes
```

Expected: same certification result (labels come from file instead of CSV column).

---

## 15. Compare Models

```bash
trustgate compare \
  --config examples/test_config_e2e.yaml \
  --models gpt-4.1-mini,gpt-4.1-nano \
  --task-type mcq \
  --questions examples/test_questions_120.csv \
  --ground-truth /tmp/ground_truth.json
```

Expected: side-by-side table with reliability, M*, coverage for each model.

---

## 16. Python API: Full Pipeline

```bash
python3 -c "
from theaios import trustgate

result = trustgate.certify(config_path='examples/test_config_e2e.yaml')
print(f'Reliability: {result.reliability_level:.1%}')
print(f'M*: {result.m_star}')
print(f'Coverage: {result.coverage:.3f}')
print(f'Capability Gap: {result.capability_gap:.1%}')
print(f'API Cost: \${result.api_cost_estimate:.2f}')
"
```

---

## 17. Python API: Sample and Profile

```bash
python3 -c "
from theaios.trustgate import sample_and_profile
from theaios.trustgate.config import load_config, load_questions

config = load_config('examples/test_config_e2e.yaml')
questions = load_questions('examples/test_questions_120.csv')

profiles = sample_and_profile(config, questions)
for qid in list(profiles.keys())[:5]:
    print(f'{qid}: {profiles[qid]}')
"
```

Expected: ranked profiles for each question, e.g., `[('B', 1.0)]` for easy questions.

---

## 18. Python API: Profile Diagnostic

```bash
python3 -c "
from theaios.trustgate import sample_and_profile, diagnose_profiles
from theaios.trustgate.config import load_config, load_questions

config = load_config('examples/test_config_e2e.yaml')
questions = load_questions('examples/test_questions_120.csv')

profiles = sample_and_profile(config, questions)
diag = diagnose_profiles(profiles)
print(f'Status: {diag.status}')
print(f'Mean consensus: {diag.mean_consensus:.2f}')
print(f'Mean classes: {diag.mean_n_classes:.1f}')
print(f'All-unique fraction: {diag.frac_all_unique:.1%}')
print(f'Warnings: {diag.warnings}')
"
```

Expected: status = "good", high consensus, low unique fraction, no warnings.

---

## 19. Python API: Generate Questionnaire

```bash
python3 -c "
from theaios.trustgate import sample_and_profile, generate_questionnaire
from theaios.trustgate.config import load_config, load_questions

config = load_config('examples/test_config_e2e.yaml')
questions = load_questions('examples/test_questions_120.csv')

profiles = sample_and_profile(config, questions)
path = generate_questionnaire(questions, profiles, '/tmp/test_questionnaire.html')
print(f'Generated: {path}')
"

open /tmp/test_questionnaire.html
```

---

## 20. Python API: TrustGate Runtime (Passthrough)

```bash
python3 -c "
from theaios.trustgate import TrustGate, certify
from theaios.trustgate.config import load_config

config = load_config('examples/test_config_e2e.yaml')
result = certify(config=config)

gate = TrustGate(config=config, certification=result)
print(f'Mode: {gate.mode}')
print(f'Reliability: {gate.reliability_level:.1%}')

response = gate.query('What is the capital of France? (A) London (B) Paris (C) Berlin (D) Madrid')
print(f'Answer: {response.answer}')
print(f'Reliability level: {response.reliability_level:.1%}')
print(f'Mode: {response.mode}')
print(f'Samples: {response.n_samples}')
"
```

Expected: passthrough mode, single API call, reliability metadata attached.

---

## 21. Python API: TrustGate Runtime (Sampled)

```bash
python3 -c "
from theaios.trustgate import TrustGate, certify
from theaios.trustgate.config import load_config

config = load_config('examples/test_config_e2e.yaml')
result = certify(config=config)

gate = TrustGate(config=config, certification=result, mode='sampled')

response = gate.query('What is the capital of France? (A) London (B) Paris (C) Berlin (D) Madrid')
print(f'Answer: {response.answer}')
print(f'Prediction set: {response.prediction_set}')
print(f'Consensus: {response.consensus:.0%}')
print(f'Margin: {response.margin:.2f}')
print(f'Singleton: {response.is_singleton}')
print(f'Samples: {response.n_samples}')
print(f'Profile: {response.profile}')
"
```

Expected: sampled mode, K samples, prediction set, high consensus.

---

## 22. Python API: Reporting

```bash
python3 -c "
from theaios.trustgate import certify
from theaios.trustgate.reporting import export_json, export_csv, print_certification_result

result = certify(config_path='examples/test_config_e2e.yaml')

# Console
print_certification_result(result, verbose=True)

# JSON
print(export_json(result)[:200] + '...')

# CSV
print(export_csv(result)[:200] + '...')
"
```

---

## 23. Python API: Custom Canonicalizer

```bash
python3 -c "
from theaios.trustgate import Canonicalizer, register_canonicalizer, get_canonicalizer

@register_canonicalizer('upper')
class UpperCanonicalizer(Canonicalizer):
    def canonicalize(self, question, answer):
        return self.preprocess(answer).upper()

c = get_canonicalizer('upper')
print(c.canonicalize('Q?', 'The answer is hello'))
# Expected: THE ANSWER IS HELLO
"
```

---

## 24. Built-in Datasets (Requires Network)

```bash
python3 -c "
from theaios.trustgate.datasets import load_mmlu
questions = load_mmlu(subjects=['abstract_algebra'], n=10)
for q in questions[:3]:
    print(f'{q.id}: {q.text[:80]}...')
    print(f'  Acceptable: {q.acceptable_answers}')
"
```

---

## Summary Checklist

| # | Feature | Command |
|---|---------|---------|
| 1 | CLI version/help | `trustgate version` |
| 2 | Full certification | `trustgate certify --yes` |
| 3 | JSON output | `--output json` |
| 4 | CSV output | `--output csv` |
| 5 | Output to file | `--output-file result.json` |
| 6 | Verbose (per-alpha) | `--verbose` |
| 7 | Pre-flight estimate | interactive prompt |
| 8 | Cost per request | `--cost-per-request 0.01` |
| 9 | Sample only | `trustgate sample` |
| 10 | Cache stats/clear | `trustgate cache stats/clear` |
| 11 | Export HTML questionnaire | `trustgate calibrate --export` |
| 12 | Local web UI | `trustgate calibrate --serve` |
| 13 | Profile dump | `trustgate calibrate` (no flags) |
| 14 | External ground truth | `--ground-truth file.json` |
| 15 | Model comparison | `trustgate compare` |
| 16 | Python: certify | `trustgate.certify()` |
| 17 | Python: sample_and_profile | `sample_and_profile()` |
| 18 | Python: diagnose | `diagnose_profiles()` |
| 19 | Python: questionnaire | `generate_questionnaire()` |
| 20 | Python: TrustGate passthrough | `TrustGate(mode="passthrough")` |
| 21 | Python: TrustGate sampled | `TrustGate(mode="sampled")` |
| 22 | Python: reporting | `export_json/csv`, `print_certification_result` |
| 23 | Python: custom canonicalizer | `@register_canonicalizer` |
| 24 | Built-in datasets | `load_mmlu()`, `load_gsm8k()` |
