"""Local calibration UI server (Flask)."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
body{font-family:system-ui,-apple-system,sans-serif;max-width:600px;margin:0 auto;padding:20px;background:#f5f5f5}
.progress{background:#ddd;border-radius:8px;overflow:hidden;margin-bottom:24px;height:28px;position:relative}
.progress-bar{background:#4CAF50;height:100%;transition:width .3s}
.progress-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-weight:600;font-size:14px}
.card{background:white;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.1)}
.label{font-size:13px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.question{font-size:18px;line-height:1.5;margin-bottom:4px}
.answer{font-size:18px;line-height:1.5;color:#333}
.buttons{display:flex;gap:12px;margin-top:20px}
.btn{flex:1;padding:16px;border:none;border-radius:10px;font-size:18px;font-weight:600;cursor:pointer;transition:transform .1s}
.btn:active{transform:scale(.95)}
.btn-correct{background:#4CAF50;color:white}
.btn-incorrect{background:#f44336;color:white}
.done{text-align:center;font-size:20px;padding:40px}
.hint{text-align:center;color:#999;font-size:13px;margin-top:12px}
</style>
</head>
<body>
<div class="progress"><div class="progress-bar" id="bar"></div><div class="progress-text" id="ptext"></div></div>
<div class="card"><div class="label">Question</div><div class="question" id="q"></div></div>
<div class="card"><div class="label">AI's Answer</div><div class="answer" id="a"></div></div>
<div class="buttons">
<button class="btn btn-correct" onclick="judge(true)">Correct</button>
<button class="btn btn-incorrect" onclick="judge(false)">Incorrect</button>
</div>
<div class="hint">Keyboard: Y = Correct, N = Incorrect</div>
<script>
let qid=null;
async function load(){
 const r=await fetch('/api/next');const d=await r.json();
 if(d.done){document.body.innerHTML='<div class="done">All done! Labels saved.</div>';return}
 qid=d.question_id;document.getElementById('q').textContent=d.question;
 document.getElementById('a').textContent=d.answer;
 const p=await(await fetch('/api/progress')).json();
 document.getElementById('bar').style.width=p.pct+'%';
 document.getElementById('ptext').textContent=p.completed+'/'+p.total+' ('+Math.round(p.pct)+'%)';
}
async function judge(correct){
 if(!qid)return;
 await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({question_id:qid,judgment:correct})});load();}
document.addEventListener('keydown',e=>{if(e.key==='y'||e.key==='Y')judge(true);if(e.key==='n'||e.key==='N')judge(false);});
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
.correct{color:#4CAF50}.incorrect{color:#f44336}
table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #eee}
th{background:#f9f9f9}
a.btn{display:inline-block;padding:10px 20px;background:#2196F3;color:white;text-decoration:none;border-radius:6px;margin-right:8px}
</style>
</head>
<body>
<h1>Calibration Admin</h1>
<div class="card" id="info"></div>
<div class="card"><a class="btn" href="/api/export">Download JSON</a></div>
<div class="card"><h3>Recent Judgments</h3><table id="tbl"><tr><th>ID</th><th>Judgment</th></tr></table></div>
<script>
async function refresh(){
 const p=await(await fetch('/api/progress')).json();
 const r=await(await fetch('/api/results')).json();
 let nc=0,ni=0;for(const v of Object.values(r)){if(v==='correct')nc++;else ni++;}
 document.getElementById('info').innerHTML=
  '<h3>Progress: '+p.completed+'/'+p.total+' ('+Math.round(p.pct)+'%)</h3>'+
  '<div class="progress"><div class="progress-bar" style="width:'+p.pct+'%"></div></div>'+
  '<div class="stats"><span class="stat correct">Correct: '+nc+'</span><span class="stat incorrect">Incorrect: '+ni+'</span></div>';
 const tbl=document.getElementById('tbl');
 tbl.innerHTML='<tr><th>ID</th><th>Judgment</th></tr>';
 for(const[k,v]of Object.entries(r).reverse().slice(0,20)){
  const tr=document.createElement('tr');tr.innerHTML='<td>'+k+'</td><td>'+v+'</td>';tbl.appendChild(tr);}
}
refresh();setInterval(refresh,3000);
</script>
</body></html>"""


def create_app(
    questions: list[Question],
    top_answers: dict[str, str],
    output_file: str = "calibration_labels.json",
) -> Flask:
    """Create the Flask app for the calibration UI.

    Requires Flask to be installed (``pip install 'trustgate[serve]'``).
    """
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:
        raise ImportError(
            "Flask is required for the calibration UI. "
            "Install with: pip install 'trustgate[serve]'"
        ) from exc

    app = Flask(__name__)

    # State
    labels: dict[str, str] = {}
    question_map = {q.id: q for q in questions}
    pending = [q.id for q in questions]
    pending_idx = 0
    output_path = Path(output_file)

    def _save() -> None:
        output_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    @app.route("/")
    def index() -> str:
        return _REVIEWER_HTML

    @app.route("/admin")
    def admin() -> str:
        return _ADMIN_HTML

    @app.route("/api/next")
    def api_next() -> Response:
        nonlocal pending_idx
        # Find next unreviewed question
        while pending_idx < len(pending):
            qid = pending[pending_idx]
            if qid not in labels:
                answer = top_answers.get(qid, "(no answer)")
                return jsonify({
                    "question_id": qid,
                    "question": question_map[qid].text,
                    "answer": answer,
                    "done": False,
                })
            pending_idx += 1
        return jsonify({"done": True})

    @app.route("/api/review", methods=["POST"])
    def api_review() -> Response:
        data = request.get_json(force=True)
        qid = data["question_id"]
        judgment = "correct" if data["judgment"] else "incorrect"
        labels[qid] = judgment
        _save()
        return jsonify({"ok": True})

    @app.route("/api/progress")
    def api_progress() -> Response:
        total = len(questions)
        completed = len(labels)
        pct = (completed / total * 100) if total > 0 else 0
        return jsonify({"completed": completed, "total": total, "pct": pct})

    @app.route("/api/results")
    def api_results() -> Response:
        return jsonify(labels)

    @app.route("/api/export")
    def api_export() -> Response:
        return app.response_class(
            json.dumps(labels, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename={output_path.name}"},
        )

    return app


def serve_calibration(
    questions: list[Question],
    top_answers: dict[str, str],
    port: int = 8080,
    output_file: str = "calibration_labels.json",
) -> None:
    """Start the calibration server.

    Opens browser automatically. Saves labels on Ctrl+C.
    """
    import webbrowser

    app = create_app(questions, top_answers, output_file)

    def _shutdown(signum: int, frame: object) -> None:
        print(f"\nLabels saved to {output_file}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    print(f"Starting calibration UI at http://localhost:{port}")
    print(f"Labels will be saved to {output_file}")
    webbrowser.open(f"http://localhost:{port}")

    app.run(host="127.0.0.1", port=port, debug=False)
