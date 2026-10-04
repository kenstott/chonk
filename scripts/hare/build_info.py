# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Record the index build environment in work/fang2026/data/index_build_info.json.

Embeddings differ slightly across hardware (CPU, Apple MPS, CUDA) and library
versions, so this is published alongside the index stores.

Usage: python scripts/hare/build_info.py <where>     # e.g. local, vultr-gpu
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import sentence_transformers
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "fang2026" / "data" / "index_build_info.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=ROOT
    ).stdout.rstrip("\n")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    if torch.cuda.is_available():
        accel = f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        accel = "mps"
    else:
        accel = "cpu"
    info = {
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "where": sys.argv[1],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "accelerator_available": accel,
        "git_commit": _git("rev-parse", "HEAD"),
        # Uncommitted paths at build time; code paths here mean the build is not
        # reproducible from git_commit alone.
        "git_uncommitted": [line[3:] for line in _git("status", "--porcelain").splitlines()],
    }
    OUT.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
