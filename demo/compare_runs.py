# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 92373334-a139-4220-8393-03672db82bb4
"""
Compare two benchmark runs question-by-question.

Loads run checkpoints + eval scores, finds divergent questions (run_a >> run_b
or run_b >> run_a), shows chunk-level diff, answers, and optionally runs an
LLM judge to classify WHY one run outperformed.

Usage:
    python demo/compare_runs.py \\
        --run-a vanilla_rag_v2 \\
        --run-b contextual_plain_nobc \\
        --subset novel \\
        --top-n 50 \\
        --out-dir work \\
        --judge \\
        --output work/results/compare_vanilla_vs_nobc.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_run(results_dir: Path, run_name: str) -> dict[str, dict]:
    """Load run checkpoint keyed by question id."""
    path = results_dir / f"{run_name}_checkpoint.jsonl"
    rows: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def _load_eval(results_dir: Path, run_name: str) -> dict[str, float]:
    """Load eval checkpoint → {id: answer_correctness}. NaN → None."""
    path = results_dir / f"bench_eval_ckpt_{run_name}.jsonl"
    scores: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ac = r.get("answer_correctness")
            if ac is not None and not (isinstance(ac, float) and math.isnan(ac)):
                scores[r["id"]] = ac
    return scores


def _load_chunks(db_path: Path, chunk_ids: list[str]) -> dict[str, dict]:
    """Fetch chunk content + breadcrumb from DuckDB for a list of chunk_ids."""
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    placeholders = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"SELECT chunk_id, content, breadcrumb, section FROM embeddings "
        f"WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    con.close()
    return {r[0]: {"content": r[1], "breadcrumb": r[2], "section": r[3]} for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Judge
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an expert evaluator for question-answering systems.
Given a question, the gold answer, and two candidate answers (A and B), determine:
1. Which answer is more correct/complete (A, B, or tie).
2. The primary reason for the difference. Choose ONE category:
   - CHUNK_SELECTION: One system retrieved more relevant passages
   - ANSWER_SYNTHESIS: Similar passages but one system synthesized a better answer
   - CONTEXT_FRAGMENTATION: One system's chunks were poorly bounded, missing key info
   - HALLUCINATION: One system invented facts not in its context
   - TIE: Both answers are roughly equivalent
3. A one-sentence explanation.

Respond ONLY with valid JSON:
{"winner": "A"|"B"|"tie", "category": "<category>", "explanation": "<one sentence>"}
"""

_JUDGE_USER = """\
Question: {question}

Gold answer: {gold}

Answer A ({run_a}):
{answer_a}

Answer B ({run_b}):
{answer_b}
"""


def _judge(question: str, gold: str, answer_a: str, answer_b: str,
           run_a: str, run_b: str, client) -> dict:
    prompt = _JUDGE_USER.format(
        question=question, gold=gold,
        answer_a=answer_a, answer_b=answer_b,
        run_a=run_a, run_b=run_b,
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"winner": "error", "category": "error", "explanation": resp.choices[0].message.content}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_diff(ids_a: list[str], ids_b: list[str]) -> dict:
    set_a, set_b = set(ids_a), set(ids_b)
    shared = set_a & set_b
    return {
        "overlap_count": len(shared),
        "only_a": len(set_a - set_b),
        "only_b": len(set_b - set_a),
        "jaccard": len(shared) / len(set_a | set_b) if set_a | set_b else 0.0,
    }


def _format_chunks(chunk_ids: list[str], chunk_data: dict[str, dict], max_chars: int = 300) -> list[dict]:
    out = []
    for cid in chunk_ids:
        info = chunk_data.get(cid, {})
        content = info.get("content", "[not found]")
        out.append({
            "chunk_id": cid,
            "breadcrumb": info.get("breadcrumb") or info.get("section") or "",
            "content_preview": content[:max_chars] + ("…" if len(content) > max_chars else ""),
        })
    return out


def run_compare(args) -> None:
    import duckdb
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results_dir = Path(args.out_dir) / "results"
    data_dir    = Path(args.out_dir) / "data"

    print(f"Loading run checkpoints…")
    run_a_data = _load_run(results_dir, args.run_a)
    run_b_data = _load_run(results_dir, args.run_b)

    print(f"Loading eval scores…")
    scores_a = _load_eval(results_dir, args.run_a)
    scores_b = _load_eval(results_dir, args.run_b)

    # Intersect: questions scored in both runs
    common_ids = set(scores_a) & set(scores_b) & set(run_a_data) & set(run_b_data)
    print(f"Questions scored in both runs: {len(common_ids)}")

    # Filter by subset
    if args.subset != "all":
        prefix = args.subset.capitalize()
        common_ids = {qid for qid in common_ids if qid.startswith(prefix)}
    print(f"After subset filter ({args.subset}): {len(common_ids)}")

    # Compute deltas and sort
    deltas: list[tuple[float, str]] = []
    for qid in common_ids:
        delta = scores_a[qid] - scores_b[qid]
        deltas.append((delta, qid))

    # Sort by absolute delta descending — largest divergences first
    deltas.sort(key=lambda x: abs(x[0]), reverse=True)

    top = deltas[:args.top_n]
    print(f"Top {len(top)} most divergent questions selected.")

    # Gather all chunk IDs to look up
    all_chunk_ids_a: set[str] = set()
    all_chunk_ids_b: set[str] = set()
    for _, qid in top:
        all_chunk_ids_a.update(run_a_data[qid].get("retrieved_chunks", []))
        all_chunk_ids_b.update(run_b_data[qid].get("retrieved_chunks", []))

    db_a = data_dir / ("vanilla_rag.duckdb" if "vanilla" in args.run_a else "chonk.duckdb")
    db_b = data_dir / ("vanilla_rag.duckdb" if "vanilla" in args.run_b else "chonk.duckdb")

    print(f"Fetching {len(all_chunk_ids_a)} chunks from {db_a.name}…")
    chunk_data_a = _load_chunks(db_a, list(all_chunk_ids_a)) if all_chunk_ids_a else {}
    if db_a == db_b:
        chunk_data_b = chunk_data_a
    else:
        print(f"Fetching {len(all_chunk_ids_b)} chunks from {db_b.name}…")
        chunk_data_b = _load_chunks(db_b, list(all_chunk_ids_b)) if all_chunk_ids_b else {}

    # Build LLM client if judging
    client = None
    if args.judge:
        import openai
        client = openai.OpenAI(base_url=f"http://localhost:{os.environ.get('PROXY_PORT', '10011')}/v1",
                               api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"))
        print("LLM judge enabled (gpt-4o-mini).")

    # Assemble records
    records: list[dict[str, Any]] = []
    for delta, qid in top:
        row_a = run_a_data[qid]
        row_b = run_b_data[qid]
        chunks_a = row_a.get("retrieved_chunks", [])
        chunks_b = row_b.get("retrieved_chunks", [])

        record: dict[str, Any] = {
            "id": qid,
            "question": row_a["question"],
            "question_type": row_a.get("question_type", ""),
            "gold_answer": row_a.get("gold_answer", ""),
            "evidence": row_a.get("evidence", []),
            "delta": round(delta, 4),
            "score_a": round(scores_a[qid], 4),
            "score_b": round(scores_b[qid], 4),
            "run_a": args.run_a,
            "run_b": args.run_b,
            "answer_a": row_a.get("generated_answer", ""),
            "answer_b": row_b.get("generated_answer", ""),
            "chunk_diff": _chunk_diff(chunks_a, chunks_b),
            "chunks_a": _format_chunks(chunks_a, chunk_data_a),
            "chunks_b": _format_chunks(chunks_b, chunk_data_b),
        }

        if client:
            verdict = _judge(
                record["question"], record["gold_answer"],
                record["answer_a"], record["answer_b"],
                args.run_a, args.run_b, client,
            )
            record["judge"] = verdict

        records.append(record)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records → {out_path}")

    # Print summary
    _print_summary(records, args)


def _print_summary(records: list[dict], args) -> None:
    a_wins = [r for r in records if r["delta"] > 0.05]
    b_wins = [r for r in records if r["delta"] < -0.05]
    ties   = [r for r in records if abs(r["delta"]) <= 0.05]

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.run_a} vs {args.run_b}  |  subset={args.subset}")
    print(f"{'='*60}")
    print(f"  {args.run_a} wins (delta>0.05): {len(a_wins)}")
    print(f"  {args.run_b} wins (delta<-0.05): {len(b_wins)}")
    print(f"  Near-ties (|delta|<=0.05): {len(ties)}")

    if records and "judge" in records[0]:
        from collections import Counter
        cats = Counter(r["judge"].get("category", "?") for r in records)
        winners = Counter(r["judge"].get("winner", "?") for r in records)
        print(f"\nJudge winner breakdown: {dict(winners)}")
        print(f"Judge category breakdown: {dict(cats)}")

    print(f"\nTop 10 largest divergences:")
    print(f"  {'ID':<30} {'delta':>7}  {'score_a':>8}  {'score_b':>8}  {'jaccard':>7}")
    for r in records[:10]:
        print(f"  {r['id']:<30} {r['delta']:>7.3f}  {r['score_a']:>8.3f}  {r['score_b']:>8.3f}  {r['chunk_diff']['jaccard']:>7.3f}")

    if records and "judge" in records[0]:
        print(f"\nSample judge verdicts (top 5 {args.run_a} wins):")
        for r in sorted(records, key=lambda x: -x["delta"])[:5]:
            j = r.get("judge", {})
            print(f"  [{r['id']}] winner={j.get('winner')} cat={j.get('category')}")
            print(f"    {j.get('explanation', '')}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Compare two benchmark runs question-by-question")
    p.add_argument("--run-a", default="vanilla_rag_v2",
                   help="Name of run A (the 'better' run to investigate)")
    p.add_argument("--run-b", default="contextual_plain_nobc",
                   help="Name of run B")
    p.add_argument("--subset", choices=["medical", "novel", "all"], default="novel",
                   help="Question subset to analyse")
    p.add_argument("--top-n", type=int, default=50,
                   help="Number of most-divergent questions to analyse")
    p.add_argument("--out-dir", default="work",
                   help="Base work directory (same as bench script --out-dir)")
    p.add_argument("--judge", action="store_true",
                   help="Run LLM-as-judge (gpt-4o-mini) to classify divergences")
    p.add_argument("--output", default="work/results/compare_runs.jsonl",
                   help="Output JSONL path")
    args = p.parse_args()
    run_compare(args)


if __name__ == "__main__":
    main()
