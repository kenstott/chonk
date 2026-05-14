#!/usr/bin/env python3
"""GPU run monitor — runs hourly via crontab on the GPU.

Sends status email via Gmail SMTP (App Password).
Requires in .env:
  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
  GMAIL_TO=kennethstott@gmail.com        (optional, defaults to GMAIL_FROM)
  GMAIL_FROM=kennethstott@gmail.com      (optional, defaults to gmail username from key)

Crontab (add with `crontab -e`):
  7 * * * * cd /root/chunkymonkey && /root/miniforge/envs/chonk/bin/python work/gpu_monitor.py >> work/logs/monitor.log 2>&1
"""

import json
import math
import os
import re
import smtplib
import subprocess
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent
WORK = REPO / "work"

# Load .env
env_path = REPO / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()

GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
GMAIL_FROM = os.environ.get("GMAIL_FROM", "kennethstott@gmail.com")
GMAIL_TO = os.environ.get("GMAIL_TO", GMAIL_FROM)

LOGS = {
    "parallel_runs": WORK / "logs" / "parallel_runs.log",
    "paper_runs": WORK / "logs" / "paper_runs.log",
    "fang2026": WORK / "logs" / "fang2026_paper.log",
}

GRB_RESULTS = WORK / "results"
FANG_RESULTS = WORK / "fang2026" / "results"


# ── Data collection ────────────────────────────────────────────────────────────
def gpu_stats() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"], text=True, timeout=10
        ).strip()
        return out
    except Exception as e:
        return f"unavailable ({e})"


def tail_log(path: Path, n: int = 5) -> str:
    if not path.exists():
        return "(no log)"
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:]) if lines else "(empty)"
    except Exception as e:
        return f"(error: {e})"


def fatal_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    ignore = re.compile(r"transformers|token indices|indexing error", re.I)
    fatal = re.compile(r"fatal|FAIL|KeyError|Traceback|Exception|Error", re.I)
    try:
        lines = path.read_text(errors="replace").splitlines()
        return [l for l in lines if fatal.search(l) and not ignore.search(l)]
    except Exception:
        return []


def parse_eval_file(path: Path) -> dict:
    try:
        raw = path.read_text(errors="replace")
        data = json.loads(raw)
        params = data.pop("_params", {})
        rows = {}
        for cat, metrics in data.items():
            if isinstance(metrics, dict):
                ac = metrics.get("answer_correctness")
                if ac is not None and not (isinstance(ac, float) and math.isnan(ac)):
                    rows[cat] = round(float(ac), 3)
        avg = round(sum(rows.values()) / len(rows), 3) if rows else None
        return {"avg": avg, "cats": rows, "n": params.get("n_evaluated", "?")}
    except Exception as e:
        return {"avg": None, "cats": {}, "n": "?", "error": str(e)}


def all_scores() -> list[tuple[str, str, dict]]:
    results = []
    for path in sorted(GRB_RESULTS.glob("bench_eval_*_rp.json")):
        name = path.name.replace("bench_eval_", "").replace("_rp.json", "")
        results.append(("GRB", name, parse_eval_file(path)))
    for path in sorted(FANG_RESULTS.glob("bench_eval_*_rp.json")):
        name = path.name.replace("bench_eval_", "").replace("_rp.json", "")
        results.append(("FANG", name, parse_eval_file(path)))
    return results


def check_complete() -> bool:
    for log in LOGS.values():
        text = tail_log(log, 30)
        if re.search(r"DONE:.*succeeded", text) or re.search(r"ALL.*COMPLETE", text, re.I):
            return True
    return False


# ── Email ──────────────────────────────────────────────────────────────────────
def send_email(subject: str, html_body: str, urgent: bool = False) -> None:
    if not GMAIL_APP_PASSWORD:
        print(f"[monitor] No GMAIL_APP_PASSWORD — cannot send email. Subject: {subject}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_FROM
    msg["To"] = GMAIL_TO
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
            s.login(GMAIL_FROM, GMAIL_APP_PASSWORD)
            s.sendmail(GMAIL_FROM, [GMAIL_TO], msg.as_string())
        print(f"[monitor] Email sent: {subject}")
    except Exception as e:
        print(f"[monitor] Email failed: {e}")


# ── Report builder ─────────────────────────────────────────────────────────────
def build_report() -> tuple[str, str, list[str]]:
    ts = time.strftime("%Y-%m-%d %H:%M")
    gpu = gpu_stats()
    scores = all_scores()
    fatals: list[str] = []
    for name, path in LOGS.items():
        for line in fatal_lines(path):
            fatals.append(f"[{name}] {line}")
    done = check_complete()

    subject = f"Chunkymonkey GPU Status — {ts}"
    if done:
        subject = f"Chunkymonkey GPU Status — ALL JOBS COMPLETE — {ts}"

    # scores table
    grb_rows = [(n, s) for suite, n, s in scores if suite == "GRB"]
    fang_rows = [(n, s) for suite, n, s in scores if suite == "FANG"]

    def score_table(rows: list, cats: list[str]) -> str:
        if not rows:
            return "<p>None yet.</p>"
        header = "<tr><th>Run</th><th>Avg</th>" + "".join(f"<th>{c}</th>" for c in cats) + "<th>n</th></tr>"
        body_rows = []
        for name, s in rows:
            avg = s.get("avg", "—")
            n = s.get("n", "?")
            if s.get("error"):
                body_rows.append(f"<tr><td>{name}</td><td colspan='{len(cats)+2}'>ERROR: {s['error']}</td></tr>")
            else:
                cat_cells = "".join(f"<td>{s['cats'].get(c, '—')}</td>" for c in cats)
                body_rows.append(f"<tr><td>{name}</td><td><b>{avg}</b></td>{cat_cells}<td>{n}</td></tr>")
        return f"<table border='1' cellpadding='4' cellspacing='0'>{header}{''.join(body_rows)}</table>"

    grb_cats = ["Fact Retrieval", "Contextual Summarize", "Complex Reasoning", "Creative Generation"]
    fang_cats = ["Cross-Domain Entity Resolution", "Absence/Negation", "Multi-Document Join", "Temporal Versioning", "Quantitative Synthesis"]

    parallel_tail = tail_log(LOGS["parallel_runs"], 5)
    paper_tail = tail_log(LOGS["paper_runs"], 3)
    fang_tail = tail_log(LOGS["fang2026"], 3)

    fatal_html = ("<pre style='color:red'>" + "\n".join(fatals[:30]) + "</pre>") if fatals else "<p>None.</p>"

    html = f"""<h2>{subject}</h2>
<h3>GPU</h3>
<pre>{gpu}</pre>
<h3>Completed Runs — GraphRAG-Bench ({len(grb_rows)} evals)</h3>
{score_table(grb_rows, grb_cats)}
<h3>Completed Runs — FANG-2026 ({len(fang_rows)} evals)</h3>
{score_table(fang_rows, fang_cats)}
<h3>In Progress — parallel_runs</h3>
<pre>{parallel_tail}</pre>
<h3>In Progress — paper_runs</h3>
<pre>{paper_tail}</pre>
<h3>In Progress — fang2026</h3>
<pre>{fang_tail}</pre>
<h3>Errors / Fatals</h3>
{fatal_html}
<hr/><p><em>GPU: 140.82.47.26 | {ts}</em></p>"""

    return subject, html, fatals


def main() -> None:
    print(f"[monitor] {time.strftime('%Y-%m-%d %H:%M:%S')} — collecting status")
    subject, html, fatals = build_report()

    if fatals:
        urgent_html = f"<h2>FATAL ERRORS</h2><pre style='color:red'>{'<br>'.join(fatals[:30])}</pre><hr/>" + html
        send_email(
            f"URGENT: Chunkymonkey GPU Fatal Error — {time.strftime('%Y-%m-%d %H:%M')}",
            urgent_html,
            urgent=True,
        )

    send_email(subject, html)


if __name__ == "__main__":
    main()
