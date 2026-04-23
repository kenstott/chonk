"""Chunk-size grid search for GraphRAG-Bench evaluation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_BENCH_SCRIPT = _SCRIPT_DIR / "graphrag_bench.py"
_PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

GRID_CONFIGS = [
    # (name, type, chunk_tokens, min_chunk, max_chunk, rerank)
    ("vanilla_128",          "vanilla",    128,  None, None,  False),
    ("vanilla_192",          "vanilla",    192,  None, None,  False),
    ("vanilla_256",          "vanilla",    256,  None, None,  False),
    ("vanilla_384",          "vanilla",    384,  None, None,  False),
    ("nobc_200_500",         "contextual", None,  200,  500,  False),
    ("nobc_300_600",         "contextual", None,  300,  600,  False),
    ("nobc_400_800",         "contextual", None,  400,  800,  False),
    ("nobc_400_1000",        "contextual", None,  400, 1000,  False),
    ("nobc_400_1200",        "contextual", None,  400, 1200,  False),
    ("nobc_900_1300",        "contextual", None,  900, 1300,  False),
    ("nobc_1100_2200",       "contextual", None, 1100, 2200,  False),
    # rerank variants for top-4
    ("nobc_400_1000_rerank", "contextual", None,  400, 1000,  True),
    ("vanilla_256_rerank",   "vanilla",    256,  None, None,  True),
    ("nobc_1100_2200_rerank","contextual", None, 1100, 2200,  True),
    ("vanilla_384_rerank",   "vanilla",    384,  None, None,  True),
    # NER + cluster expansion + rerank
    ("nobc_400_1000_ner_rerank",    "contextual", None,  400, 1000, True),
    ("nobc_1100_2200_ner_rerank",   "contextual", None, 1100, 2200, True),
    # NER + entity embedding ANN expansion + rerank
    ("nobc_400_1000_ner_x_rerank",  "contextual", None,  400, 1000, True),
    ("nobc_1100_2200_ner_x_rerank", "contextual", None, 1100, 2200, True),
]

VANILLA_BASELINE   = "vanilla_256"
CTX_BASELINE       = "nobc_400_1200"
VANILLA_CHUNK_DEFAULT = 256
CTX_MIN_DEFAULT       = 400
CTX_MAX_DEFAULT       = 1200


def _db_name(cfg_type: str, chunk_tokens: int | None, min_chunk: int | None, max_chunk: int | None) -> str:
    if cfg_type == "vanilla":
        if chunk_tokens == VANILLA_CHUNK_DEFAULT:
            return "vanilla_rag.duckdb"
        return f"vanilla_rag_{chunk_tokens}.duckdb"
    if min_chunk == CTX_MIN_DEFAULT and max_chunk == CTX_MAX_DEFAULT:
        return "chunkymonkey_nobc.duckdb"
    return f"chunkymonkey_nobc_{min_chunk}_{max_chunk}.duckdb"


def _run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "a") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def cmd_draw_sample(args: argparse.Namespace) -> None:
    from chunkymonkey import NOVEL_STRUCTURAL_LEVELS  # noqa: F401 — just verify import path works

    out_dir   = Path(args.out_dir)
    n_obs     = args.n_obs
    seed      = args.seed
    data_dir  = out_dir / "data"

    import random
    rng = random.Random(seed)

    questions_by_corpus: dict[str, list[dict]] = {}
    for subset in ("medical", "novel"):
        f = data_dir / f"{subset}_questions.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"Questions file not found: {f}. Run 'download' first.")
        qs = []
        with open(f) as fh:
            for line in fh:
                qs.append(json.loads(line))
        questions_by_corpus[subset] = qs

    n_per_corpus = n_obs // 2
    remainder    = n_obs - n_per_corpus * 2

    sample_ids: list[str] = []
    breakdown: dict[str, dict[str, int]] = {}

    for idx, (corpus, questions) in enumerate(questions_by_corpus.items()):
        n_corpus = n_per_corpus + (1 if idx < remainder else 0)
        by_type: dict[str, list[dict]] = defaultdict(list)
        for q in questions:
            by_type[q.get("question_type", "unknown")].append(q)

        total_q = len(questions)
        type_counts: dict[str, int] = {}
        allocated = 0
        types = sorted(by_type.keys())
        for i, qtype in enumerate(types):
            if i == len(types) - 1:
                type_counts[qtype] = n_corpus - allocated
            else:
                count = round(len(by_type[qtype]) / total_q * n_corpus)
                type_counts[qtype] = count
                allocated += count

        breakdown[corpus] = {}
        for qtype in types:
            n_sample = min(type_counts[qtype], len(by_type[qtype]))
            sampled = rng.sample(by_type[qtype], n_sample)
            for q in sampled:
                qid = q.get("id")
                if qid is None:
                    raise KeyError(f"Question missing 'id' field in {corpus}: {q}")
                sample_ids.append(qid)
            breakdown[corpus][qtype] = n_sample

    out_path = data_dir / f"grid_sample_{n_obs}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample_ids, indent=2), encoding="utf-8")

    print(f"\nSample breakdown (seed={seed}, n={len(sample_ids)}):")
    all_types = sorted({t for corpus_types in breakdown.values() for t in corpus_types})
    col_w = max(len(t) for t in all_types) + 2
    corpora = list(breakdown.keys())
    header = f"  {'type':<{col_w}}" + "".join(f"  {c:>10}" for c in corpora) + f"  {'total':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for qtype in all_types:
        row = f"  {qtype:<{col_w}}"
        total = 0
        for corpus in corpora:
            n = breakdown[corpus].get(qtype, 0)
            row += f"  {n:>10}"
            total += n
        row += f"  {total:>7}"
        print(row)
    totals_row = f"  {'TOTAL':<{col_w}}"
    grand = 0
    for corpus in corpora:
        s = sum(breakdown[corpus].values())
        totals_row += f"  {s:>10}"
        grand += s
    totals_row += f"  {grand:>7}"
    print("  " + "-" * (len(header) - 2))
    print(totals_row)
    print(f"\nSaved {len(sample_ids)} IDs → {out_path}")


def cmd_run_grid(args: argparse.Namespace) -> None:
    out_dir     = Path(args.out_dir)
    data_dir    = out_dir / "data"
    logs_dir    = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sample_file = args.sample_file or str(data_dir / "grid_sample_300.json")
    force       = args.force
    cfg_filter  = None if args.configs == "all" else set(args.configs.split(","))

    configs = GRID_CONFIGS if cfg_filter is None else [c for c in GRID_CONFIGS if c[0] in cfg_filter]

    for name, cfg_type, chunk_tokens, min_chunk, max_chunk, use_rerank in configs:
        print(f"\n{'='*60}")
        print(f"Config: {name}  type={cfg_type}  rerank={use_rerank}")
        print(f"{'='*60}")

        db_name = _db_name(cfg_type, chunk_tokens, min_chunk, max_chunk)
        db_path = data_dir / db_name
        log = logs_dir / f"grid_{name}.log"

        use_ner = "ner" in name and cfg_type != "vanilla"

        if not db_path.exists() or force:
            if cfg_type == "vanilla":
                index_cmd = [
                    sys.executable, str(_BENCH_SCRIPT),
                    "index-vanilla",
                    "--out-dir", str(out_dir),
                    "--chunk-tokens", str(chunk_tokens),
                ]
            else:
                index_cmd = [
                    sys.executable, str(_BENCH_SCRIPT),
                    "index",
                    "--out-dir", str(out_dir),
                    "--embed-content-only",
                    "--min-chunk", str(min_chunk),
                    "--max-chunk", str(max_chunk),
                ]
            if force:
                index_cmd.append("--force")
            if use_ner:
                index_cmd.append("--with-ner")
            print(f"Indexing: {' '.join(index_cmd)}")
            _run_cmd(index_cmd, log)
        else:
            print(f"Index exists: {db_path}")
            if use_ner:
                # Run NER on existing index if chunk_entities is empty
                use_ner_x = "ner_x" in name
                ner_cmd = [
                    sys.executable, str(_BENCH_SCRIPT),
                    "build-ner",
                    "--out-dir", str(out_dir),
                    "--db-name", db_name,
                ]
                if use_ner_x:
                    ner_cmd.append("--with-embeddings")
                print(f"NER build: {' '.join(ner_cmd)}")
                _run_cmd(ner_cmd, log)

        use_ner_x = "ner_x" in name
        run_cmd = [
            sys.executable, str(_BENCH_SCRIPT),
            "run",
            "--out-dir", str(out_dir),
            "--run-name", name,
            "--question-ids", str(sample_file),
            "--db-name", db_name,
            "--no-breadcrumb-embed",
        ]
        if cfg_type == "vanilla":
            run_cmd.append("--vanilla")
            run_cmd = [x for x in run_cmd if x not in ("--no-breadcrumb-embed",)]
        if use_rerank:
            run_cmd.append("--rerank")
        if "ner" in name:
            run_cmd.append("--enhanced")
        if use_ner_x:
            run_cmd.append("--ner-x")
        print(f"Running: {' '.join(run_cmd)}")
        _run_cmd(run_cmd, log)

        eval_cmd = [
            sys.executable, str(_BENCH_SCRIPT),
            "eval",
            "--out-dir", str(out_dir),
            "--run-name", name,
            "--judge", "gpt-4o-mini",
        ]
        print(f"Evaluating: {' '.join(eval_cmd)}")
        _run_cmd(eval_cmd, log)

        eval_f = out_dir / "results" / f"bench_eval_{name}.json"
        if eval_f.exists():
            data = json.loads(eval_f.read_text())
            ac_vals = [v.get("answer_correctness", float("nan")) for v in data.values()
                       if "answer_correctness" in v]
            avg_ac = sum(ac_vals) / len(ac_vals) if ac_vals else float("nan")
            print(f"  --> {name} avg AC = {avg_ac:.4f}")


def cmd_report(args: argparse.Namespace) -> None:
    from scipy import stats as _stats

    out_dir     = Path(args.out_dir)
    results_dir = out_dir / "results"

    config_names = [c[0] for c in GRID_CONFIGS]

    rows: list[dict] = []
    per_item: dict[str, list[dict]] = {}

    for name in config_names:
        ckpt_f = results_dir / f"bench_eval_ckpt_{name}.jsonl"
        if not ckpt_f.exists():
            continue
        items = []
        with open(ckpt_f) as f:
            for line in f:
                items.append(json.loads(line))
        per_item[name] = items

        run_f = results_dir / f"{name}.jsonl"
        id_to_source: dict[str, str] = {}
        if run_f.exists():
            with open(run_f) as f:
                for line in f:
                    r = json.loads(line)
                    id_to_source[r["id"]] = r.get("source", "?")

        all_ac, med_ac, nov_ac = [], [], []
        by_type: dict[str, list[float]] = defaultdict(list)
        for item in items:
            ac = item.get("answer_correctness")
            if ac is None or (isinstance(ac, float) and ac != ac):
                continue
            all_ac.append(ac)
            src = id_to_source.get(item["id"], "?")
            if "med" in src.lower() or src.lower() == "medical":
                med_ac.append(ac)
            elif "nov" in src.lower() or src.lower() == "novel":
                nov_ac.append(ac)
            by_type[item.get("question_type", "?")].append(ac)

        rows.append({
            "name":     name,
            "overall":  sum(all_ac) / len(all_ac) if all_ac else float("nan"),
            "medical":  sum(med_ac) / len(med_ac) if med_ac else float("nan"),
            "novel":    sum(nov_ac) / len(nov_ac) if nov_ac else float("nan"),
            "by_type":  {t: sum(v) / len(v) for t, v in by_type.items()},
            "n":        len(all_ac),
        })

    if not rows:
        print("No eval checkpoints found. Run 'run' first.")
        return

    rows.sort(key=lambda r: r["overall"] if r["overall"] == r["overall"] else -1, reverse=True)

    baselines = {
        "vanilla": VANILLA_BASELINE,
        "contextual": CTX_BASELINE,
    }

    def _paired_p(name_a: str, name_b: str) -> float | None:
        a = per_item.get(name_a)
        b = per_item.get(name_b)
        if not a or not b:
            return None
        id_to_ac_a = {i["id"]: i.get("answer_correctness") for i in a}
        id_to_ac_b = {i["id"]: i.get("answer_correctness") for i in b}
        shared = [k for k in id_to_ac_a if k in id_to_ac_b]
        if len(shared) < 5:
            return None
        va = [id_to_ac_a[k] for k in shared if id_to_ac_a[k] == id_to_ac_a[k] and id_to_ac_b[k] == id_to_ac_b[k]]
        vb = [id_to_ac_b[k] for k in shared if id_to_ac_a[k] == id_to_ac_a[k] and id_to_ac_b[k] == id_to_ac_b[k]]
        if len(va) < 5:
            return None
        _, p = _stats.ttest_rel(va, vb)
        return float(p)

    fmt = lambda v: f"{v:.4f}" if v == v else "  —  "  # noqa: E731

    def _print_ranked(title: str, sort_key: str, p_baseline: str) -> None:
        ranked = sorted(rows, key=lambda r: r[sort_key] if r[sort_key] == r[sort_key] else -1, reverse=True)
        print(f"\n── {title} ──\n")
        header = f"  {'name':<22}  {sort_key:>8}  {'n':>5}  {'p(vs base)':>11}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in ranked:
            name = row["name"]
            cfg  = next((c for c in GRID_CONFIGS if c[0] == name), None)
            if cfg:
                baseline_name = baselines["vanilla"] if cfg[1] == "vanilla" else baselines["contextual"]
            else:
                baseline_name = None
            p_val = _paired_p(name, baseline_name) if baseline_name and baseline_name != name else None
            sig   = "*" if p_val is not None and p_val < 0.05 else " "
            p_str = f"{p_val:.4f}{sig}" if p_val is not None else "    —    "
            is_baseline = name in baselines.values()
            marker = " [baseline]" if is_baseline else ""
            print(f"  {name:<22}  {fmt(row[sort_key]):>8}  {row['n']:>5}  {p_str:>11}{marker}")

    _print_ranked("Medical Leaderboard (answer_correctness)", "medical", VANILLA_BASELINE)
    _print_ranked("Novel Leaderboard (answer_correctness)", "novel", VANILLA_BASELINE)

    print("\n── Overall (answer_correctness) ──\n")
    header = f"  {'name':<22}  {'overall':>8}  {'medical':>8}  {'novel':>8}  {'n':>5}  {'p(vs base)':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in rows:
        name = row["name"]
        cfg  = next((c for c in GRID_CONFIGS if c[0] == name), None)
        if cfg:
            baseline_name = baselines["vanilla"] if cfg[1] == "vanilla" else baselines["contextual"]
        else:
            baseline_name = None

        p_val = _paired_p(name, baseline_name) if baseline_name and baseline_name != name else None
        sig   = "*" if p_val is not None and p_val < 0.05 else " "
        p_str = f"{p_val:.4f}{sig}" if p_val is not None else "    —    "

        is_baseline = name in baselines.values()
        marker = " [baseline]" if is_baseline else ""

        print(f"  {name:<22}  {fmt(row['overall']):>8}  {fmt(row['medical']):>8}  {fmt(row['novel']):>8}  {row['n']:>5}  {p_str:>11}{marker}")

    print()
    all_types = sorted({t for row in rows for t in row["by_type"]})
    if all_types:
        type_header = f"  {'name':<20}" + "".join(f"  {t[:12]:>12}" for t in all_types)
        print("── By question type ──")
        print(type_header)
        print("  " + "-" * (len(type_header) - 2))
        for row in rows:
            line = f"  {row['name']:<20}"
            for t in all_types:
                v = row["by_type"].get(t, float("nan"))
                line += f"  {fmt(v):>12}"
            print(line)


def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Chunk-size grid search for GraphRAG-Bench")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("draw-sample", help="Draw stratified random sample of question IDs")
    p.add_argument("--out-dir", required=True, metavar="DIR")
    p.add_argument("--n-obs",   type=int, default=300)
    p.add_argument("--seed",    type=int, default=42)
    p.set_defaults(func=cmd_draw_sample)

    p = sub.add_parser("run", help="Run full chunk-size grid search")
    p.add_argument("--out-dir",     required=True, metavar="DIR")
    p.add_argument("--sample-file", default=None, metavar="PATH",
                   help="JSON file with question IDs (default: {out_dir}/data/grid_sample_300.json)")
    p.add_argument("--configs",     default="all",
                   help="Comma-separated list of config names, or 'all' (default: all)")
    p.add_argument("--force",       action="store_true",
                   help="Re-run even if checkpoint exists")
    p.set_defaults(func=cmd_run_grid)

    p = sub.add_parser("report", help="Compare grid results with paired t-tests")
    p.add_argument("--out-dir", required=True, metavar="DIR")
    p.set_defaults(func=cmd_report)

    return ap


def main() -> None:
    ap   = _make_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
