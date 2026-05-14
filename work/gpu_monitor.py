#!/usr/bin/env python3
"""GPU run monitor — runs hourly via crontab on the GPU.

Publishes to ntfy.sh for zero-config push notifications.
Requires in .env:
  NTFY_TOPIC=chunkymonkey-ks-a7f3b2   (pick any unguessable name)

Crontab (add with `crontab -e`):
  7 * * * * cd /root/chunkymonkey && /root/miniforge/envs/chonk/bin/python work/gpu_monitor.py >> work/logs/monitor.log 2>&1

Subscribe (bot or browser):
  https://ntfy.sh/<NTFY_TOPIC>
  curl ntfy.sh/<NTFY_TOPIC>/json?poll=1&since=1h
"""

import json
import math
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent
WORK = REPO / "work"

env_path = REPO / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")

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
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"], text=True, timeout=10
        ).strip()
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
        return [ln for ln in lines if fatal.search(ln) and not ignore.search(ln)]
    except Exception:
        return []


def parse_eval_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(errors="replace"))
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


# ── ntfy publish ───────────────────────────────────────────────────────────────
def ntfy(title: str, body: str, priority: str = "default", tags: list[str] | None = None) -> None:
    if not NTFY_TOPIC:
        print(f"[monitor] No NTFY_TOPIC in .env — skipping publish. Title: {title}")
        return
    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"
    headers = {
        "Title": title,
        "Priority": priority,
        "Content-Type": "text/plain",
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    try:
        req = urllib.request.Request(url, data=body.encode(), headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=15)
        print(f"[monitor] ntfy published: {title}")
    except Exception as e:
        print(f"[monitor] ntfy failed: {e}")


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
    grb_rows = [(n, s) for suite, n, s in scores if suite == "GRB"]
    fang_rows = [(n, s) for suite, n, s in scores if suite == "FANG"]

    title = f"GPU Status {ts}"
    if done:
        title = f"ALL JOBS COMPLETE — {ts}"

    lines = [
        f"GPU: {gpu}",
        f"GRB evals done: {len(grb_rows)}  |  FANG evals done: {len(fang_rows)}",
        "",
    ]

    if grb_rows:
        lines.append("── GRB scores ──")
        for name, s in grb_rows:
            avg = s.get("avg", "n/a")
            lines.append(f"  {name}: {avg}")

    if fang_rows:
        lines.append("── FANG scores ──")
        for name, s in fang_rows:
            avg = s.get("avg", "n/a")
            lines.append(f"  {name}: {avg}")

    lines += [
        "",
        "── parallel_runs (last 3) ──",
        tail_log(LOGS["parallel_runs"], 3),
        "",
        "── fang2026 (last 3) ──",
        tail_log(LOGS["fang2026"], 3),
    ]

    if fatals:
        lines += ["", "── ERRORS ──"] + fatals[:10]

    body = "\n".join(lines)
    return title, body, fatals


def main() -> None:
    print(f"[monitor] {time.strftime('%Y-%m-%d %H:%M:%S')} — collecting status")
    title, body, fatals = build_report()

    if fatals:
        ntfy(
            f"URGENT: GPU Fatal Error {time.strftime('%H:%M')}",
            "\n".join(fatals[:20]),
            priority="urgent",
            tags=["warning", "rotating_light"],
        )

    priority = "high" if "COMPLETE" in title else "default"
    tags = ["white_check_mark"] if "COMPLETE" in title else ["chart_with_upwards_trend"]
    ntfy(title, body, priority=priority, tags=tags)


if __name__ == "__main__":
    main()
