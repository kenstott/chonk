# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Compare reproduced HARE-Bench scores with the reported ones.

Reads bench_eval_<run>_rp.json from both directories and prints the overall
score and per-question-type scores side by side, with the difference.

Usage:
    python scripts/hare/compare_results.py --reported work/fang2026/results \\
        --reproduced work/reproduce/results
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
    d = json.loads(path.read_text())
    out = {"overall": float(d["overall"])}
    for name, short in TYPES.items():
        out[short] = float(d["by_type"][name])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare reproduced and reported scores.")
    parser.add_argument("--reported", type=Path, required=True)
    parser.add_argument("--reproduced", type=Path, required=True)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
