"""Self-contained HTML questionnaire for calibration delegation.

Generates a single HTML file that embeds all questions and shuffled
canonical answers as inline JSON.  The reviewer opens it in any browser
(works offline, no server), picks the acceptable answer for each
question, and downloads a ``labels.json`` at the end.

The developer imports the labels with:
    trustgate certify --ground-truth labels.json
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from theaios.trustgate.types import Question

_MCQ_OPTION_RE = re.compile(
    r"\(([A-Ea-e])\)\s*([^(]+?)(?=\s*\([A-Ea-e]\)|$)",
)


def _enrich_mcq_answer(canonical: str, question: str) -> str:
    """If canonical is a single MCQ letter, extract the option text from the question.

    Returns e.g. ``"B — Paris"`` instead of just ``"B"``.
    Falls back to the raw canonical string if not matchable.
    """
    if len(canonical) != 1 or canonical.upper() not in "ABCDE":
        return canonical

    for match in _MCQ_OPTION_RE.finditer(question):
        letter = match.group(1).upper()
        text = match.group(2).strip().rstrip(".,;:?!")
        if letter == canonical.upper():
            return f"{canonical} — {text}"
    return canonical

_QUESTIONNAIRE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrustGate Calibration Questionnaire</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;max-width:660px;margin:0 auto;padding:20px;background:#f5f5f5}
h1{font-size:22px;margin-bottom:4px}
.subtitle{color:#666;font-size:14px;margin-bottom:24px}
.progress{background:#ddd;border-radius:8px;overflow:hidden;margin-bottom:24px;height:28px;position:relative}
.progress-bar{background:#4CAF50;height:100%;transition:width .3s}
.progress-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-weight:600;font-size:14px}
.card{background:white;border-radius:12px;padding:24px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.1)}
.label{font-size:13px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.question{font-size:18px;line-height:1.5}
h3{margin-bottom:12px;font-size:15px;color:#444}
.answer-btn{display:block;width:100%;padding:14px 18px;margin-bottom:8px;
  border:2px solid #e0e0e0;border-radius:10px;background:white;font-size:16px;
  cursor:pointer;transition:all .15s;text-align:left;word-break:break-word}
.answer-btn:hover{border-color:#4CAF50;background:#f0faf0}
.answer-btn:active{transform:scale(.98)}
.none-btn{display:block;width:100%;padding:14px;margin-top:4px;border:2px dashed #ccc;border-radius:10px;
  background:white;font-size:16px;color:#888;cursor:pointer;text-align:center;transition:all .15s}
.none-btn:hover{border-color:#f44336;color:#f44336;background:#fff5f5}
.hint{text-align:center;color:#999;font-size:13px;margin-top:12px}
.key{display:inline-block;background:#eee;border-radius:4px;padding:2px 6px;font-family:monospace;font-size:12px}
.done{text-align:center;padding:40px}
.done h2{font-size:24px;margin-bottom:16px;color:#4CAF50}
.done p{color:#666;margin-bottom:20px;font-size:16px}
.dl-btn{display:inline-block;padding:14px 32px;background:#4CAF50;color:white;border:none;border-radius:10px;
  font-size:18px;font-weight:600;cursor:pointer;text-decoration:none;transition:all .15s}
.dl-btn:hover{background:#43A047}
.dl-btn:active{transform:scale(.97)}
#review{display:block}
#finish{display:none}
</style>
</head>
<body>
<h1>TrustGate Calibration</h1>
<p class="subtitle">Pick the acceptable answer for each question, or mark "none."</p>
<div class="progress"><div class="progress-bar" id="bar"></div><div class="progress-text" id="ptext"></div></div>
<div id="review">
<div class="card"><div class="label">Question</div><div class="question" id="q"></div></div>
<div class="card">
  <h3>Which answer is acceptable?</h3>
  <div id="answers"></div>
  <button class="none-btn" onclick="pick(null)">None of these are correct</button>
</div>
<div class="hint">Keyboard: <span class="key">1</span>–<span class="key">9</span> to pick, <span class="key">0</span> for none</div>
</div>
<div id="finish" class="done">
  <h2>All done!</h2>
  <p>You reviewed <span id="n_reviewed"></span> questions.</p>
  <button class="dl-btn" onclick="download()">Download Labels</button>
  <p style="margin-top:16px;font-size:13px;color:#999">
    Send the downloaded <code>labels.json</code> back to the developer.
  </p>
</div>
<script>
const ITEMS = __ITEMS_JSON__;
const labels = {};
let idx = 0;
let answerValues = [];

function render() {
  const total = ITEMS.length;
  const pct = (idx / total * 100);
  document.getElementById('bar').style.width = pct + '%';
  document.getElementById('ptext').textContent = idx + '/' + total + ' (' + Math.round(pct) + '%)';
  if (idx >= total) {
    document.getElementById('review').style.display = 'none';
    document.getElementById('finish').style.display = 'block';
    document.getElementById('n_reviewed').textContent = total;
    return;
  }
  const item = ITEMS[idx];
  document.getElementById('q').textContent = item.question;
  const c = document.getElementById('answers');
  c.innerHTML = '';
  answerValues = item.answers.map(a => a.answer);
  item.answers.forEach((a) => {
    const b = document.createElement('button');
    b.className = 'answer-btn';
    b.textContent = a.display || a.answer;
    b.onclick = () => pick(a.answer);
    c.appendChild(b);
  });
}

function pick(answer) {
  if (idx >= ITEMS.length) return;
  const item = ITEMS[idx];
  if (answer !== null) {
    labels[item.question_id] = answer;
  }
  // null = none acceptable → skip (not included in labels)
  idx++;
  render();
}

function download() {
  const blob = new Blob([JSON.stringify(labels, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'labels.json';
  a.click();
  URL.revokeObjectURL(url);
}

document.addEventListener('keydown', e => {
  const k = parseInt(e.key);
  if (k === 0) pick(null);
  else if (k >= 1 && k <= answerValues.length) pick(answerValues[k - 1]);
});

render();
</script>
</body>
</html>
"""


def generate_questionnaire(
    questions: list[Question],
    profiles: dict[str, list[tuple[str, float]]],
    output_path: str = "questionnaire.html",
    seed: int = 42,
) -> str:
    """Generate a self-contained HTML calibration questionnaire.

    The file embeds all questions and shuffled canonical answers as inline
    JSON.  No server or network access required — the reviewer opens the
    file in any browser, reviews answers, and downloads ``labels.json``.

    Returns the output file path.
    """
    rng = random.Random(seed)

    items = []
    for q in questions:
        profile = profiles.get(q.id)
        if not profile:
            continue
        answers = [ans for ans, _freq in profile]
        # For MCQ single-letter answers, enrich with option text from question
        enriched = [_enrich_mcq_answer(ans, q.text) for ans in answers]
        shuffled = list(zip(answers, enriched))
        rng.shuffle(shuffled)
        items.append({
            "question_id": q.id,
            "question": q.text,
            "answers": [{"answer": ans, "display": disp} for ans, disp in shuffled],
        })

    items_json = json.dumps(items, ensure_ascii=False)
    html = _QUESTIONNAIRE_HTML.replace("__ITEMS_JSON__", items_json)

    path = Path(output_path)
    path.write_text(html, encoding="utf-8")
    return str(path)
