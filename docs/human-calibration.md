# Human Calibration

TrustGate includes tools for collecting human calibration labels when you
don't have ground truth. A domain expert reviews the AI's candidate answers
and identifies the acceptable one — providing the exact nonconformity scores
needed for conformal calibration.

---

## When to Use Human Calibration

- **No ground truth labels.** Many real-world tasks (medical triage, legal
  analysis, customer support) lack pre-labeled datasets.
- **Domain expert evaluation.** The reviewer doesn't need ML knowledge —
  they see a question, see the AI's candidate answers, and pick the correct one.
- **Quick calibration.** Labeling ~50 items takes about 10 minutes.

---

## How It Works

1. TrustGate samples K responses for **all** your questions.
2. Responses are canonicalized and ranked by frequency (the self-consistency profile).
3. A **random subset** is selected for human review (default: `n_cal` from config, typically 50). You don't need to prepare exactly 50 questions — provide as many as you have, and TrustGate selects the calibration subset automatically.
4. The reviewer sees each question alongside **all candidate answers in randomized order** — no frequencies, no rank numbers, preventing anchoring bias.
5. The reviewer picks the acceptable answer (or "none of these").
6. The system internally resolves the rank of the selected answer → nonconformity score.
7. Labels are saved as `{question_id: canonical_answer}` — directly compatible with `trustgate certify --ground-truth`.

### What happens to the reviewer's selections

Each selection produces a **nonconformity score** — the rank of the selected
answer in the AI's original self-consistency profile:

- Reviewer picks an answer that was the AI's **most frequent** response → score = 1 (the AI agreed with the human)
- Reviewer picks an answer that was the AI's **second most frequent** → score = 2 (the AI's top pick was wrong, but the correct answer was close)
- Reviewer picks an answer that was **third or lower** → score = 3+ (the AI buried the correct answer)
- Reviewer picks **"none of these"** → score = ∞ (the AI never produced the correct answer — capability gap)

These scores feed into conformal calibration: they are sorted, and the
quantile at confidence level 1-α gives M\* (how many top answers you need to
include to guarantee coverage). If most scores are 1, then M\*=1 and the
reliability level is high. See [Concepts](concepts.md#worked-example-from-human-clicks-to-reliability-level) for a worked example.

The reviewer doesn't need to understand any of this — they just pick the
correct answer. The math happens behind the scenes.

---

## Option A: Shareable HTML Questionnaire (Recommended)

Generate a self-contained HTML file and share it with anyone — email, Slack,
Google Drive. No server needed. Works offline, works on mobile.

```bash
# 1. Sample + generate questionnaire
trustgate calibrate --export questionnaire.html

# 2. Share questionnaire.html with your reviewer
#    They open it in any browser, pick answers, click "Download Labels"
#    → downloads labels.json

# 3. Reviewer sends labels.json back to you

# 4. Certify
trustgate certify --ground-truth labels.json
```

The HTML file embeds all questions and shuffled answers as inline JSON.
Everything runs client-side in the browser. Zero infrastructure.

### From Python

```python
from theaios.trustgate import sample_and_profile, generate_questionnaire
from theaios.trustgate.config import load_config, load_questions

config = load_config("trustgate.yaml")
questions = load_questions("questions.csv")

# Sample and build profiles
profiles = sample_and_profile(config, questions)

# Generate the questionnaire
generate_questionnaire(questions, profiles, "questionnaire.html")
```

---

## Option B: Local Web UI

For reviewers on the same network. Requires Flask (`pip install "theaios-trustgate[serve]"`).

```bash
trustgate calibrate --serve --port 8080
```

| Flag              | Default                    | Description                       |
|-------------------|----------------------------|-----------------------------------|
| `--serve`         | (flag)                     | Start the web UI server           |
| `--export`        |                            | Export as shareable HTML file      |
| `--questions`     |                            | Questions file (CSV/JSON)         |
| `--port`          | `8080`                     | Port for the local server         |
| `--output`        | `calibration_labels.json`  | Where to save the labels          |
| `--config`        | `trustgate.yaml`           | Config file                       |
| `--cost-per-request` |                         | USD per request (generic endpoints)|
| `--yes`           |                            | Skip confirmation prompt          |

### From Python

```python
from theaios.trustgate.serve import serve_calibration
from theaios.trustgate import sample_and_profile
from theaios.trustgate.config import load_config, load_questions

config = load_config("trustgate.yaml")
questions = load_questions("questions.csv")
profiles = sample_and_profile(config, questions)

serve_calibration(
    questions=questions,
    profiles=profiles,
    port=8080,
    output_file="calibration_labels.json",
)
```

---

## The Review Interface

The reviewer sees each question with all candidate answers in **randomized
order** — no rank numbers, no frequency percentages. They judge purely on
content.

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

- **Keyboard shortcuts**: `1`-`9` to pick an answer, `0` for "none"
- **Auto-advance**: next question loads automatically after each pick
- **Auto-save**: labels saved to disk after every judgment (web UI only)
- **Mobile-friendly**: works on phones and tablets

---

## The Admin Panel

Navigate to `http://localhost:8080/admin` (web UI only):

- Progress bar and completion percentage
- Count of top-1 correct / lower rank / none acceptable
- Table of recent judgments with resolved ranks
- **Download JSON** button to export labels at any time

### API Endpoints

| Endpoint       | Method | Description                                      |
|----------------|--------|--------------------------------------------------|
| `/`            | GET    | Reviewer UI (HTML)                               |
| `/admin`       | GET    | Admin dashboard (HTML)                           |
| `/api/next`    | GET    | Next question + shuffled candidate answers       |
| `/api/review`  | POST   | Submit selection (`question_id`, `selected_answer`) |
| `/api/progress`| GET    | Progress (`completed`, `total`, `pct`)           |
| `/api/results` | GET    | All labels with resolved ranks                   |
| `/api/export`  | GET    | Download labels JSON                             |

---

## Labels Format

The labels file maps question IDs to the selected canonical answer:

```json
{
  "q001": "B",
  "q002": "Paris",
  "q003": "42"
}
```

Questions where the reviewer picked "none" are excluded (they represent
unsolvable items — the correct answer never appeared in K samples).

This format is directly compatible with `trustgate certify --ground-truth`.

---

## End-to-End Workflow

```bash
# 1. Install
pip install "theaios-trustgate[serve]"

# 2. Prepare questions
# questions.csv:
#   id,question
#   q001,"What is the capital of France? (A) London (B) Paris (C) Berlin"
#   q002,"What causes type 2 diabetes?"

# 3. Option A: Generate shareable questionnaire
trustgate calibrate --export questionnaire.html
# Share with reviewer → they send back labels.json

# 3. Option B: Start local web UI
trustgate calibrate --serve --port 8080
# Reviewer opens browser, reviews items, labels auto-saved

# 4. Certify using the labels
trustgate certify --ground-truth labels.json
```

---

## Practical Tips

- **50 items in 10 minutes.** Plan ~10 seconds per item with keyboard shortcuts.

- **Share the HTML questionnaire** for cross-organization reviews. The reviewer
  doesn't need Python, network access, or any setup — just a browser.

- **Quality over quantity.** 50 well-labeled items are more valuable than 500
  noisy ones. Choose a reviewer who understands the domain.

- **Answers are randomized** to prevent the reviewer from anchoring on the
  AI's confidence. The system resolves ranks internally.

- **Combine with ground truth.** If you have partial labels, run human
  calibration for the rest, merge the JSON files, then certify.
