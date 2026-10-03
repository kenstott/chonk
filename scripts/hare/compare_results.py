# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Compare reproduced HARE-Bench scores with the reported ones.

Reads bench_eval_<run>_rp.json from both directories and prints the overall
score and per-question-type scores side by side, with the difference.
Both score file layouts are read: the typed scorer's and the evaluator's.

With --retrieval DIR (a reference run's outputs, e.g. the published
diagnostics), it also compares each question's retrieved chunk IDs from
<run>.jsonl, which separates retrieval drift (hardware, library versions) from
generation drift (LLM sampling):

  identical   questions whose retrieved chunk list matches exactly, in order
  same set    questions with the same chunks in any order
  mean jacc.  mean Jaccard overlap of the retrieved chunk sets

Usage:
    python scripts/hare/compare_results.py --reported work/fang2026/results \\
        --reproduced work/reproduce/results [--retrieval work/reference]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TYPES = {
    "Cross-Domain Entity Resolution": "CE",
    "Multi-Document Join": "MDJ",
    "Temporal Versioning": "TVR",
    "Targeted Attribute Lookup": "TAL",
    "Descriptive Attribute Lookup": "DAL",
}


def _scores(path: Path) -> dict[str, float]:
    """Overall and per-type scores from either score file layout.

    work/score_typed.py writes {"overall", "by_type"}; `graphrag_bench.py run-all`
    writes one {"typed_score"} block per question type, and the overall score is
    the mean of the five per-type scores.
    """
    d = json.loads(path.read_text())
    if "by_type" in d:
        out = {short: float(d["by_type"][name]) for name, short in TYPES.items()}
        return {"overall": float(d["overall"]), **out}
    out = {short: float(d[name]["typed_score"]) for name, short in TYPES.items()}
    return {"overall": sum(out.values()) / len(out), **out}


def _retrieved(path: Path) -> dict[str, list[str]]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = list(r["retrieved_chunks"])
    return out


def compare_retrieval(reference: Path, reproduced: Path) -> tuple[int, int, int, float]:
    """(questions compared, identical order, same set, mean Jaccard) for one run."""
    ref, rep = _retrieved(reference), _retrieved(reproduced)
    shared = sorted(ref.keys() & rep.keys())
    if not shared:
        raise ValueError(f"no question IDs in common: {reference} vs {reproduced}")
    identical = sum(ref[q] == rep[q] for q in shared)
    same_set = sum(set(ref[q]) == set(rep[q]) for q in shared)
    jaccard = [
        len(set(ref[q]) & set(rep[q])) / len(set(ref[q]) | set(rep[q]))
        if set(ref[q]) | set(rep[q])
        else 1.0
        for q in shared
    ]
    return len(shared), identical, same_set, sum(jaccard) / len(jaccard)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare reproduced and reported scores.")
    parser.add_argument("--reported", type=Path, required=True)
    parser.add_argument("--reproduced", type=Path, required=True)
    parser.add_argument(
        "--retrieval", type=Path, help="reference run outputs to compare retrieval against"
    )
    args = parser.parse_args()

    runs = sorted(args.reproduced.glob("bench_eval_*_rp.json"))
    if not runs:
        print(f"No bench_eval_*_rp.json files in {args.reproduced}", file=sys.stderr)
        return 1

    cols = ["overall", *TYPES.values()]
    print(f"{'run':<70} " + " ".join(f"{c:>17}" for c in cols))
    missing = []
    for rep in runs:
        orig = args.reported / rep.name
        name = rep.name.removeprefix("bench_eval_").removesuffix("_rp.json")
        if not orig.exists():
            missing.append(name)
            continue
        a, b = _scores(orig), _scores(rep)
        cells = [f"{a[c]:.3f}>{b[c]:.3f} {b[c] - a[c]:+.3f}" for c in cols]
        print(f"{name:<70} " + " ".join(f"{x:>17}" for x in cells))
    print("\ncells: reported>reproduced difference")
    if missing:
        print(f"No reported score for: {', '.join(missing)}")

    if args.retrieval:
        print(f"\nRetrieval vs {args.retrieval}")
        print(f"{'run':<70} {'questions':>9} {'identical':>9} {'same set':>9} {'mean jacc.':>10}")
        for rep in runs:
            name = rep.name.removeprefix("bench_eval_").removesuffix("_rp.json")
            ref_trace = args.retrieval / f"{name}.jsonl"
            if not ref_trace.exists():
                print(f"{name:<70} no reference trace")
                continue
            n, ident, same, jac = compare_retrieval(ref_trace, args.reproduced / f"{name}.jsonl")
            print(f"{name:<70} {n:>9} {ident:>9} {same:>9} {jac:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
