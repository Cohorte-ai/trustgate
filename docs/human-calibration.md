# Human Calibration

TrustGate includes a local web UI for collecting human correctness judgments.
Use it when you do not have ground truth labels and need a domain expert to
evaluate whether the AI's answers are correct.

---

## When to Use Human Calibration

Use the calibration UI when:

- **You have no ground truth labels.** Many real-world tasks (medical triage,
  legal analysis, customer support quality) lack pre-labeled datasets. A human
  reviewer can provide the labels TrustGate needs for conformal calibration.

- **You want domain expert evaluation.** The reviewer does not need any ML
  knowledge. They see a question, see the AI's answer, and mark it correct or
  incorrect. Send the URL to a doctor, lawyer, or subject-matter expert and
  they can start immediately.

- **You need a quick calibration set.** Labeling ~50 items takes about 5 minutes.
  That is enough for a meaningful reliability estimate.

---

## How It Works

1. TrustGate samples responses from your AI endpoint and selects the top answer
   (most frequent canonical form) for each question.
2. The calibration server presents each (question, top answer) pair in a clean
   web interface.
3. The human reviewer marks each pair as **Correct** or **Incorrect**.
4. Labels are auto-saved to a JSON file after every judgment.
5. You feed the saved labels into `trustgate certify` to produce the reliability
   certificate.

---

## Starting the Calibration UI

### From the CLI

```bash
trustgate calibrate --serve --questions questions.csv --port 8080
```

| Flag            | Default                    | Description                      |
|-----------------|----------------------------|----------------------------------|
| `--serve`       | (required flag)            | Start the web UI server          |
| `--questions`   | (required)                 | Path to questions file (CSV/JSON)|
| `--port`        | `8080`                     | Port for the local server        |
| `--output`      | `calibration_labels.json`  | Where to save the labels         |
| `--config`      | `trustgate.yaml`           | Config file (for endpoint info)  |

The server starts on `http://localhost:8080` and opens your browser
automatically.

### From Python

```python
from trustgate.serve import serve_calibration
from trustgate.config import load_questions

questions = load_questions("questions.csv")

# top_answers maps question ID -> the AI's best answer (string)
top_answers = {
    "q001": "Paris",
    "q002": "Jupiter",
    # ...
}

serve_calibration(
    questions=questions,
    top_answers=top_answers,
    port=8080,
    output_file="calibration_labels.json",
)
```

You can also create the Flask app directly for integration into an existing
server:

```python
from trustgate.serve import create_app

app = create_app(
    questions=questions,
    top_answers=top_answers,
    output_file="calibration_labels.json",
)

# Use with any WSGI server
app.run(host="0.0.0.0", port=8080)
```

### Installation

The calibration UI requires Flask. Install it with the `serve` extra:

```bash
pip install "trustgate[serve]"
```

---

## The Reviewer Interface

When the reviewer opens `http://localhost:8080`, they see:

```
 +------------------------------------------+
 | ██████████████░░░░░░░░░░  23/50  (46%)   |
 |                                          |
 |  Question:                               |
 |  What is the standard treatment for      |
 |  acute myocardial infarction?            |
 |                                          |
 |  AI's Answer:                            |
 |  Aspirin, heparin, and percutaneous      |
 |  coronary intervention (PCI)             |
 |                                          |
 |  +-------------+  +--------------+       |
 |  |   Correct   |  |  Incorrect   |       |
 |  +-------------+  +--------------+       |
 +------------------------------------------+
```

### Features

- **Progress bar** --- Shows how many items have been reviewed out of the total,
  with a percentage. Updates after every judgment.

- **Keyboard shortcuts** --- Press **Y** to mark the answer as correct, **N** to
  mark it as incorrect. This makes rapid labeling possible without touching the
  mouse.

- **Auto-save** --- Labels are written to disk (as JSON) after every single
  judgment. If the browser crashes or the server is stopped, no work is lost.

- **Auto-advance** --- After marking an answer, the next unlabeled question
  loads automatically. No need to click "Next".

---

## The Admin Panel

Navigate to `http://localhost:8080/admin` for an overview dashboard:

- Overall progress bar and completion percentage
- Count of correct vs. incorrect judgments so far
- Table of the 20 most recent judgments (auto-refreshes every 3 seconds)
- **Download JSON** button to export the current labels file at any time

The admin panel is useful for monitoring progress when you have sent the
reviewer URL to someone else.

### API Endpoints

The calibration server exposes a simple REST API used by the frontend:

| Endpoint          | Method | Description                             |
|-------------------|--------|-----------------------------------------|
| `/`               | GET    | Reviewer UI (HTML)                      |
| `/admin`          | GET    | Admin dashboard (HTML)                  |
| `/api/next`       | GET    | Next unlabeled question + answer        |
| `/api/review`     | POST   | Submit a judgment (`question_id`, `judgment`) |
| `/api/progress`   | GET    | Current progress (`completed`, `total`, `pct`) |
| `/api/results`    | GET    | All labels collected so far             |
| `/api/export`     | GET    | Download labels as a JSON file          |

---

## After Calibration

The calibration UI saves labels to a JSON file (default:
`calibration_labels.json`). The file maps question IDs to judgments:

```json
{
  "q001": "correct",
  "q002": "incorrect",
  "q003": "correct",
  "q004": "correct",
  "q005": "incorrect"
}
```

Feed this file into the certification pipeline:

### From the CLI

```bash
trustgate certify --ground-truth calibration_labels.json
```

### From Python

```python
import json
import trustgate

with open("calibration_labels.json") as f:
    labels = json.load(f)

result = trustgate.certify(
    config=trustgate.TrustGateConfig(
        endpoint=trustgate.EndpointConfig(
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-4.1",
            api_key_env="OPENAI_API_KEY",
        ),
        canonicalization=trustgate.CanonConfig(type="mcq"),
    ),
    questions=trustgate.load_config("trustgate.yaml").questions,
    labels=labels,
)

print(f"Reliability: {result.reliability_level:.1%}")
```

---

## Practical Tips

- **50 items in 5 minutes.** That is the typical pace with keyboard shortcuts.
  Plan for about 6 seconds per item. For a calibration set of 250 items, budget
  roughly 25 minutes.

- **Send the URL to a domain expert.** The reviewer does not need Python, ML
  knowledge, or any special setup. They just need a web browser and the URL
  (e.g., `http://your-machine:8080`). If you are on a shared network, bind to
  `0.0.0.0` instead of `127.0.0.1` so others can access it.

- **Labels are auto-saved.** You can stop and resume at any time. Already-labeled
  questions are skipped on reload.

- **Use the admin panel to monitor.** If you delegate labeling to someone else,
  keep `/admin` open to track their progress in real time.

- **Quality over quantity.** For conformal calibration, 50 well-labeled items are
  more valuable than 500 noisy ones. Choose a reviewer who genuinely understands
  the domain.

- **Combine with ground truth.** If you have partial ground truth labels (e.g.,
  from a test set) and need more, run human calibration for the unlabeled
  portion, then merge the two JSON files before running `trustgate certify`.

---

## End-to-End Workflow

Here is the complete workflow from zero labels to a reliability certificate:

```bash
# 1. Install with the serve extra
pip install "trustgate[serve]"

# 2. Prepare your questions file (CSV with id, question columns)
# questions.csv:
#   id,question
#   q001,"What is the capital of France?"
#   q002,"What causes type 2 diabetes?"
#   ...

# 3. Sample responses and start the calibration UI
trustgate calibrate --serve --questions questions.csv --port 8080

# 4. Review items in the browser (Y/N keyboard shortcuts)
#    Labels are auto-saved to calibration_labels.json

# 5. Certify using the collected labels
trustgate certify --ground-truth calibration_labels.json --questions questions.csv

# Output:
#   Reliability Level:   92.3%
#   Status:              PASS
```
