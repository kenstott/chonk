# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Publish, download, and verify the HARE-Bench artifacts.

Artifacts are grouped in parts. Each part is a directory in the repository and a
prefix in the chonk R2 bucket:

  corpus  work/corpus/         benchmark/fang2026/corpus/   source documents
  index   work/fang2026/data/  benchmark/fang2026/index/    index stores + caches

The committed manifest (work/fang2026/artifact_manifest.json) records the
SHA-256 digest and size of every file. Downloads are checked against it; any
missing or altered file is an error.

Usage:
    # Anyone: download from the public mirror and verify
    python scripts/hare/artifacts.py download corpus
    python scripts/hare/artifacts.py download index --out-dir work/reproduce/data

    # Verify files already on disk
    python scripts/hare/artifacts.py verify corpus

    # Maintainers: hash local files, rewrite the manifest, upload with rclone
    python scripts/hare/artifacts.py publish corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "work" / "fang2026" / "artifact_manifest.json"
PUBLIC_URL = "https://chonk.simpleishard.io/benchmark/fang2026"
RCLONE_BASE = "r2:chonk/benchmark/fang2026"
ATTEMPTS = 3
USER_AGENT = "chonk-hare-artifacts/1.0"  # Cloudflare rejects urllib's default with 403

PARTS: dict[str, dict[str, object]] = {
    "corpus": {
        "local": ROOT / "work" / "corpus",
        # Everything under the directory, except macOS metadata.
        "include": ["**/*"],
    },
    "index": {
        "local": ROOT / "work" / "fang2026" / "data",
        "include": [
            "chonk_bc_1100_2200.duckdb",
            "chonk_nobc_1100_2200.duckdb",
            "chonk_nobc_1100_2200_gleif.duckdb",
            "vanilla_rag.duckdb",
            "question_embeddings.npy",
            "question_ids.json",
            "entity_vecs_*.npz",
            "entity_ents_*.json",
            "index_build_info.json",
        ],
    },
}

Files = dict[str, dict[str, str | int]]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict[str, Files]:
    return json.loads(MANIFEST.read_text())


def _local(part: str) -> Path:
    local = PARTS[part]["local"]
    assert isinstance(local, Path)
    return local


def _collect(part: str) -> list[Path]:
    base = _local(part)
    include = PARTS[part]["include"]
    assert isinstance(include, list)
    found: set[Path] = set()
    for pattern in include:
        matches = [p for p in base.glob(pattern) if p.is_file()]
        if not matches and "*" not in pattern:
            raise FileNotFoundError(f"{part}: required file missing: {base / pattern}")
        found.update(p for p in matches if not p.name.startswith(("._", ".DS_Store")))
    return sorted(found)


def verify(part: str, out_dir: Path) -> list[str]:
    """Return problems found; empty means every file is present and matches."""
    problems = []
    for rel, meta in sorted(_load_manifest()[part].items()):
        path = out_dir / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif _sha256(path) != meta["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
    return problems


def _report(part: str, out_dir: Path) -> int:
    problems = verify(part, out_dir)
    n = len(_load_manifest()[part])
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print(f"FAILED: {part}: {len(problems)} of {n} files", file=sys.stderr)
        return 1
    print(f"OK: {part}: {n} files verified in {out_dir}")
    return 0


def _fetch(url: str, dest: Path) -> None:
    """Download url to dest; retry transient network errors, then fail."""
    for attempt in range(1, ATTEMPTS + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:  # noqa: S310
                shutil.copyfileobj(resp, out)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError) or attempt == ATTEMPTS:
                raise
            print(f"    attempt {attempt} failed ({exc}); retrying", file=sys.stderr)
            time.sleep(5 * attempt)


def download(part: str, out_dir: Path, base_url: str) -> int:
    for rel, meta in sorted(_load_manifest()[part].items()):
        dest = out_dir / rel
        if dest.is_file() and _sha256(dest) == meta["sha256"]:
            continue
        print(f"  {part}/{rel} ({meta['bytes']:,} bytes)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        _fetch(f"{base_url}/{part}/{rel}", tmp)
        tmp.replace(dest)
    return _report(part, out_dir)


def publish(part: str) -> int:
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is not on PATH")
    base = _local(part)
    files = _collect(part)
    entries: Files = {
        str(p.relative_to(base)): {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in files
    }
    listing = base / f".{part}_publish_list.txt"
    listing.write_text("".join(f"{rel}\n" for rel in entries))
    try:
        subprocess.run(
            [
                "rclone",
                "copy",
                str(base),
                f"{RCLONE_BASE}/{part}",
                "--files-from",
                str(listing),
                "--progress",
            ],
            check=True,
        )
    finally:
        listing.unlink()
    manifest = _load_manifest() if MANIFEST.exists() else {}
    manifest[part] = entries
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Published {len(entries)} {part} files to {RCLONE_BASE}/{part}; updated {MANIFEST}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="HARE-Bench artifact tool.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "verify", "publish"):
        p = sub.add_parser(name)
        p.add_argument("part", choices=sorted(PARTS))
        if name != "publish":
            p.add_argument("--out-dir", type=Path, help="default: the part's repo directory")
        if name == "download":
            p.add_argument("--base-url", default=PUBLIC_URL)
    args = parser.parse_args()

    if args.command == "publish":
        return publish(args.part)
    out_dir: Path = args.out_dir or _local(args.part)
    if args.command == "download":
        return download(args.part, out_dir, args.base_url)
    return _report(args.part, out_dir)


if __name__ == "__main__":
    sys.exit(main())
