# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Publish, download, and verify the HARE-Bench artifacts through a bucket.

The bucket is the only link between the two activities that use it:
initialization publishes the corpus and index to it, and the benchmark
downloads them from it. Nothing is exchanged through git.

Layout under the bucket prefix:

  manifest.json      SHA-256 and size of every published file, per part
  corpus/...         source documents          (local: work/corpus/)
  index/...          index stores + caches     (local: work/fang2026/data/)
  diagnostics/...    reference run outputs     (local: work/reproduce/results/)

diagnostics is optional: reproduce.sh never needs it. It holds a reference
run's per-question retrieval trace (<run>.jsonl: retrieved chunk IDs, reranker
scores, context, answer), its scores, and its rerank cache, so a reproduction
can be compared question by question (compare_results.py --retrieval).

Configuration (environment, or .env at the repository root):

  HARE_BUCKET_URL          s3://<bucket>/<prefix>        publish; download via S3
  AWS_ACCESS_KEY_ID        S3 credentials                publish; download via S3
  AWS_SECRET_ACCESS_KEY
  AWS_ENDPOINT_OVERRIDE    S3-compatible endpoint (e.g. Cloudflare R2), optional
  AWS_REGION               optional ("auto" for R2)
  HARE_PUBLIC_URL          https://... serving the same prefix; download without
                           credentials

Downloads use HARE_PUBLIC_URL when it is set, otherwise the S3 API with
credentials. Every downloaded file is checked against manifest.json; any
missing or altered file is an error.

Usage:
    python scripts/hare/artifacts.py publish corpus
    python scripts/hare/artifacts.py publish index
    python scripts/hare/artifacts.py publish diagnostics [--src-dir DIR]
    python scripts/hare/artifacts.py download index [--out-dir DIR]
    python scripts/hare/artifacts.py verify corpus [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
ATTEMPTS = 3
USER_AGENT = "chonk-hare-artifacts/1.0"  # Cloudflare rejects urllib's default with 403
MANIFEST_KEY = "manifest.json"

PARTS: dict[str, tuple[Path, list[str]]] = {
    # part: (local directory, glob patterns; a pattern without "*" is required)
    "corpus": (ROOT / "work" / "corpus", ["**/*"]),
    "index": (
        ROOT / "work" / "fang2026" / "data",
        [
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
    ),
    "diagnostics": (
        ROOT / "work" / "reproduce" / "results",
        ["*.jsonl", "bench_eval_*_rp.json", "_rerank_ckpt_*.json", "*_flags.json"],
    ),
}
# Partial or resumable state, never published.
_EXCLUDE_MARKERS = ("checkpoint", "_ckpt_", ".part")

Files = dict[str, dict[str, str | int]]
Manifest = dict[str, Files]


def _load_dotenv() -> None:
    """Set variables from ROOT/.env that are not already in the environment."""
    env = ROOT / ".env"
    if not env.is_file():
        return
    for raw in env.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (environment or .env)")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ── Bucket access ───────────────────────────────────────────────────────────


class S3Bucket:
    """Read and write through the S3 API (HARE_BUCKET_URL + AWS_* credentials)."""

    def __init__(self) -> None:
        import boto3

        url = urlparse(_require("HARE_BUCKET_URL"))
        if url.scheme != "s3" or not url.netloc:
            raise SystemExit("HARE_BUCKET_URL must look like s3://<bucket>/<prefix>")
        self.bucket = url.netloc
        self.prefix = url.path.strip("/")
        _require("AWS_ACCESS_KEY_ID")
        _require("AWS_SECRET_ACCESS_KEY")
        kwargs: dict[str, str] = {}
        if os.environ.get("AWS_ENDPOINT_OVERRIDE"):
            kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_OVERRIDE"]
        if os.environ.get("AWS_REGION"):
            kwargs["region_name"] = os.environ["AWS_REGION"]
        self.client = boto3.client("s3", **kwargs)

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def get(self, rel: str, dest: Path) -> None:
        self.client.download_file(self.bucket, self._key(rel), str(dest))

    def get_manifest(self, *, missing_ok: bool) -> Manifest:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self._key(MANIFEST_KEY))
        except self.client.exceptions.NoSuchKey:
            if missing_ok:
                return {}
            raise
        return json.loads(body["Body"].read())

    def put(self, path: Path, rel: str) -> None:
        self.client.upload_file(str(path), self.bucket, self._key(rel))

    def put_manifest(self, manifest: Manifest) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(MANIFEST_KEY),
            Body=(json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            ContentType="application/json",
        )


class PublicBucket:
    """Read-only access over HTTPS (HARE_PUBLIC_URL)."""

    def __init__(self) -> None:
        self.base = _require("HARE_PUBLIC_URL").rstrip("/")

    def get(self, rel: str, dest: Path) -> None:
        url = f"{self.base}/{rel}"
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

    def get_manifest(self, *, missing_ok: bool) -> Manifest:
        if missing_ok:
            raise ValueError("public access is read-only; publishing needs HARE_BUCKET_URL")
        tmp = ROOT / "work" / f".{MANIFEST_KEY}.part"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        self.get(MANIFEST_KEY, tmp)
        manifest: Manifest = json.loads(tmp.read_text())
        tmp.unlink()
        return manifest


def _reader() -> S3Bucket | PublicBucket:
    return PublicBucket() if os.environ.get("HARE_PUBLIC_URL") else S3Bucket()


# ── Commands ────────────────────────────────────────────────────────────────


def _collect(part: str, base: Path) -> list[Path]:
    _, patterns = PARTS[part]
    found: set[Path] = set()
    for pattern in patterns:
        matches = [p for p in base.glob(pattern) if p.is_file()]
        if not matches and "*" not in pattern:
            raise FileNotFoundError(f"{part}: required file missing: {base / pattern}")
        found.update(
            p
            for p in matches
            if not p.name.startswith(("._", ".DS_Store"))
            and not (any(m in p.name for m in _EXCLUDE_MARKERS) and "_rerank_ckpt_" not in p.name)
        )
    return sorted(found)


def verify(files: Files, out_dir: Path) -> list[str]:
    """Return problems found; empty means every file is present and matches."""
    problems = []
    for rel, meta in sorted(files.items()):
        path = out_dir / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif _sha256(path) != meta["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
    return problems


def _report(part: str, files: Files, out_dir: Path) -> int:
    problems = verify(files, out_dir)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print(f"FAILED: {part}: {len(problems)} of {len(files)} files", file=sys.stderr)
        return 1
    print(f"OK: {part}: {len(files)} files verified in {out_dir}")
    return 0


def download(part: str, out_dir: Path) -> int:
    bucket = _reader()
    files = bucket.get_manifest(missing_ok=False)[part]
    for rel, meta in sorted(files.items()):
        dest = out_dir / rel
        if dest.is_file() and _sha256(dest) == meta["sha256"]:
            continue
        print(f"  {part}/{rel} ({meta['bytes']:,} bytes)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        bucket.get(f"{part}/{rel}", tmp)
        tmp.replace(dest)
    return _report(part, files, out_dir)


def publish(part: str, src_dir: Path | None) -> int:
    bucket = S3Bucket()
    base = src_dir or PARTS[part][0]
    entries: Files = {}
    for path in _collect(part, base):
        rel = str(path.relative_to(base))
        entries[rel] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        print(f"  {part}/{rel}")
        bucket.put(path, f"{part}/{rel}")
    manifest = bucket.get_manifest(missing_ok=True)
    manifest[part] = entries
    bucket.put_manifest(manifest)  # last, so the manifest never names a missing file
    print(f"Published {len(entries)} {part} files to s3://{bucket.bucket}/{bucket.prefix}")
    return 0


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="HARE-Bench artifact tool.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "verify", "publish"):
        p = sub.add_parser(name)
        p.add_argument("part", choices=sorted(PARTS))
        if name == "publish":
            p.add_argument("--src-dir", type=Path, help="default: the part's repo directory")
        else:
            p.add_argument("--out-dir", type=Path, help="default: the part's repo directory")
    args = parser.parse_args()

    if args.command == "publish":
        return publish(args.part, args.src_dir)
    out_dir: Path = args.out_dir or PARTS[args.part][0]
    if args.command == "download":
        return download(args.part, out_dir)
    files = _reader().get_manifest(missing_ok=False)[args.part]
    return _report(args.part, files, out_dir)


if __name__ == "__main__":
    sys.exit(main())
