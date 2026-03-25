"""Local calibration UI server (Flask).

The reviewer sees each question alongside its canonical answer candidates
in **randomized order with no frequency or rank information** — preventing
anchoring bias.  The reviewer picks the acceptable answer (or "none"),
and the system internally resolves the rank for the nonconformity score
(Definition 6.2 in the paper).
"""

from __future__ import annotations

import json
import random
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from theaios.trustgate.questionnaire import _enrich_mcq_answer
from theaios.trustgate.types import Question

if TYPE_CHECKING:
    from flask import Flask

_REVIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrustGate Calibration</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;max-width:640px;margin:0 auto;padding:20px;background:#f5f5f5}
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
.done{text-align:center;font-size:20px;padding:40px}
.hint{text-align:center;color:#999;font-size:13px;margin-top:12px}
.key{display:inline-block;background:#eee;border-radius:4px;padding:2px 6px;font-family:monospace;font-size:12px}
</style>
</head>
<body>
<div class="progress"><div class="progress-bar" id="bar"></div><div class="progress-text" id="ptext"></div></div>
<div class="card"><div class="label">Question</div><div class="question" id="q"></div></div>
<div class="card">
  <h3>Which answer is acceptable?</h3>
  <div id="answers"></div>
  <button class="none-btn" onclick="pick(null)">None of these are correct</button>
</div>
<div class="hint">Keyboard: <span class="key">1</span>–<span class="key">9</span> to pick, <span class="key">0</span> for none</div>
<script>
let qid=null,answerValues=[];
async function load(){
 const r=await fetch('/api/next');const d=await r.json();
 if(d.done){document.body.innerHTML='<div class="done">All done! Labels saved.</div>';return}
 qid=d.question_id;
 document.getElementById('q').textContent=d.question;
 const c=document.getElementById('answers');c.innerHTML='';
 answerValues=d.answers.map(a=>a.answer);
 d.answers.forEach((a,i)=>{
  const b=document.createElement('button');b.className='answer-btn';
  b.textContent=a.display||a.answer;
  b.onclick=()=>pick(a.answer);c.appendChild(b);
 });
 const p=await(await fetch('/api/progress')).json();
 document.getElementById('bar').style.width=p.pct+'%';
 document.getElementById('ptext').textContent=p.completed+'/'+p.total+' ('+Math.round(p.pct)+'%)';
}
async function pick(answer){
 if(!qid)return;
 await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({question_id:qid,selected_answer:answer})});load();}
document.addEventListener('keydown',e=>{
 const k=parseInt(e.key);
 if(k===0)pick(null);
 else if(k>=1&&k<=answerValues.length)pick(answerValues[k-1]);
});
load();
</script>
</body></html>"""

_ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrustGate Calibration Admin</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;max-width:800px;margin:0 auto;padding:20px;background:#f5f5f5}
h1{margin-bottom:20px}
.card{background:white;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.1)}
.progress{background:#ddd;border-radius:8px;overflow:hidden;height:24px;margin:12px 0}
.progress-bar{background:#4CAF50;height:100%;transition:width .3s}
.stats{display:flex;gap:20px;margin:12px 0}
.stat{font-size:18px;font-weight:600}
.rank1{color:#4CAF50}.rankN{color:#FF9800}.none{color:#f44336}
table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #eee}
th{background:#f9f9f9}
a.btn{display:inline-block;padding:10px 20px;background:#2196F3;color:white;text-decoration:none;border-radius:6px;margin-right:8px}
</style>
</head>
<body>
<h1>Calibration Admin</h1>
<div class="card" id="info"></div>
<div class="card"><a class="btn" href="/api/export">Download Labels JSON</a></div>
<div class="card"><h3>Recent Judgments</h3><table id="tbl"><tr><th>ID</th><th>Selected Answer</th><th>Rank</th></tr></table></div>
<script>
async function refresh(){
 const p=await(await fetch('/api/progress')).json();
 const r=await(await fetch('/api/results')).json();
 let n1=0,nOther=0,nNone=0;
 for(const v of Object.values(r)){
  if(v.rank===1)n1++;else if(v.rank===null)nNone++;else nOther++;
 }
 document.getElementById('info').innerHTML=
  '<h3>Progress: '+p.completed+'/'+p.total+' ('+Math.round(p.pct)+'%)</h3>'+
  '<div class="progress"><div class="progress-bar" style="width:'+p.pct+'%"></div></div>'+
  '<div class="stats">'+
  '<span class="stat rank1">Top-1 correct: '+n1+'</span>'+
  '<span class="stat rankN">Lower rank: '+nOther+'</span>'+
  '<span class="stat none">None acceptable: '+nNone+'</span></div>';
 const tbl=document.getElementById('tbl');
 tbl.innerHTML='<tr><th>ID</th><th>Selected Answer</th><th>Rank</th></tr>';
 const entries=Object.entries(r).reverse().slice(0,20);
 for(const[k,v]of entries){
  const tr=document.createElement('tr');
  tr.innerHTML='<td>'+k+'</td><td>'+(v.answer||'(none)')+'</td><td>'+(v.rank||'∞')+'</td>';
  tbl.appendChild(tr);
 }
}
refresh();setInterval(refresh,3000);
</script>
</body></html>"""


def create_app(
    questions: list[Question],
    profiles: dict[str, list[tuple[str, float]]],
    output_file: str = "calibration_labels.json",
    seed: int = 42,
) -> Flask:
    """Create the Flask app for the calibration UI.

    Answers are shown to the reviewer in **randomized order** with no
    frequency or rank information, preventing anchoring bias.  The system
    resolves the rank internally after the reviewer selects an answer.

    Labels are saved as ``{qid: canonical_answer}`` — directly compatible
    with ``trustgate certify --ground-truth``.

    Requires Flask: ``pip install 'theaios-trustgate[serve]'``
    """
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:
        raise ImportError(
            "Flask is required for the calibration UI. "
            "Install with: pip install 'theaios-trustgate[serve]'"
        ) from exc

    app = Flask(__name__)

    # Pre-compute shuffled answer orders per question (fixed seed for reproducibility)
    rng = random.Random(seed)
    shuffled_answers: dict[str, list[str]] = {}
    for qid, profile in profiles.items():
        answers = [ans for ans, _freq in profile]
        shuffled = list(answers)
        rng.shuffle(shuffled)
        shuffled_answers[qid] = shuffled

    # State
    labels: dict[str, str | None] = {}  # qid → selected canonical answer (or None)
    question_map = {q.id: q for q in questions}
    pending = [q.id for q in questions if q.id in profiles]
    pending_idx = 0
    output_path = Path(output_file)

    def _rank_of(qid: str, answer: str | None) -> int | None:
        """Find the rank of the selected answer in the original profile."""
        if answer is None:
            return None
        profile = profiles.get(qid, [])
        for i, (ans, _freq) in enumerate(profile, start=1):
            if ans == answer:
                return i
        return None

    def _save() -> None:
        # Save in format compatible with certify --ground-truth:
        # {qid: canonical_answer} — skip null entries (unsolvable items)
        export = {qid: ans for qid, ans in labels.items() if ans is not None}
        output_path.write_text(json.dumps(export, indent=2), encoding="utf-8")

    @app.route("/")
    def index() -> str:
        return _REVIEWER_HTML

    @app.route("/admin")
    def admin() -> str:
        return _ADMIN_HTML

    @app.route("/api/next")
    def api_next() -> Response:
        nonlocal pending_idx
        while pending_idx < len(pending):
            qid = pending[pending_idx]
            if qid not in labels:
                # Serve answers in shuffled order, no frequencies, no ranks
                q_text = question_map[qid].text
                answers = [
                    {"answer": ans, "display": _enrich_mcq_answer(ans, q_text)}
                    for ans in shuffled_answers.get(qid, [])
                ]
                return jsonify({
                    "question_id": qid,
                    "question": question_map[qid].text,
                    "answers": answers,
                    "done": False,
                })
            pending_idx += 1
        return jsonify({"done": True})

    @app.route("/api/review", methods=["POST"])
    def api_review() -> Response:
        data = request.get_json(force=True)
        qid = data["question_id"]
        selected = data.get("selected_answer")  # str or None
        labels[qid] = selected
        _save()
        return jsonify({"ok": True})

    @app.route("/api/progress")
    def api_progress() -> Response:
        total = len(pending)
        completed = len(labels)
        pct = (completed / total * 100) if total > 0 else 0
        return jsonify({"completed": completed, "total": total, "pct": pct})

    @app.route("/api/results")
    def api_results() -> Response:
        results = {
            qid: {"answer": ans, "rank": _rank_of(qid, ans)}
            for qid, ans in labels.items()
        }
        return jsonify(results)

    @app.route("/api/export")
    def api_export() -> Response:
        export = {qid: ans for qid, ans in labels.items() if ans is not None}
        return app.response_class(
            json.dumps(export, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename={output_path.name}"},
        )

    return app


def serve_calibration(
    questions: list[Question],
    profiles: dict[str, list[tuple[str, float]]],
    port: int = 8080,
    output_file: str = "calibration_labels.json",
) -> None:
    """Start the calibration server.

    Opens browser automatically. Saves labels on each review.
    """
    import webbrowser

    app = create_app(questions, profiles, output_file)

    def _shutdown(signum: int, frame: object) -> None:
        print(f"\nLabels saved to {output_file}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    print(f"Starting calibration UI at http://localhost:{port}")
    print(f"Admin panel at http://localhost:{port}/admin")
    print(f"Labels will be saved to {output_file}")
    webbrowser.open(f"http://localhost:{port}")

    app.run(host="127.0.0.1", port=port, debug=False)
