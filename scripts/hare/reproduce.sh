#!/usr/bin/env bash
# Reproduce the HARE-Bench results from the published index.
#
# This is the reproduction method. It downloads the published index stores and
# embedding caches from the artifact bucket, verifies each file against the
# bucket's manifest.json, generates and scores answers for every run config, and
# compares the scores with the reported ones. It does no chunking or indexing: the index is a
# published artifact (see scripts/hare/02_publish_index.sh).
#
# Usage:
#   scripts/hare/reproduce.sh [--config-dir DIR] [--out-dir DIR]
#
#   --config-dir   run configs to execute (default: work/configs/fang, all runs)
#   --out-dir      working directory (default: work/reproduce)
#
# Needs, in the environment or .env: OPENAI_API_KEY (generator, ADF classifier)
# and HARE_PUBLIC_URL (public bucket URL; see .env.example), or HARE_BUCKET_URL
# with AWS_* credentials to download through the S3 API.
# Each 500-question run with gpt-4o-mini costs roughly US$5-10.
set -euo pipefail

CONFIG_DIR="work/configs/fang"
OUT="work/reproduce"
while [ $# -gt 0 ]; do
    case "$1" in
        --config-dir) CONFIG_DIR="$2"; shift ;;
        --out-dir)    OUT="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

cd "$(dirname "$0")/../.."
PY="${PY:-uv run python}"
mkdir -p "$OUT/data"
if [ "$(cd "$OUT" && pwd -P)" = "$(cd work/fang2026 && pwd -P)" ]; then
    echo "--out-dir must not be work/fang2026: its committed results would make every run skip" >&2
    exit 2
fi
if [ -z "${OPENAI_API_KEY:-}" ] && ! grep -q '^OPENAI_API_KEY=.' .env 2>/dev/null; then
    echo "OPENAI_API_KEY is not set (environment or .env)" >&2
    exit 1
fi

echo "=== Downloading and verifying the published index ==="
$PY scripts/hare/artifacts.py download index --out-dir "$OUT/data"
cp work/fang2026/data/fang2026_questions.jsonl work/fang2026/data/fang2026_gold_schemas.jsonl "$OUT/data/"

echo "=== Generating and scoring ==="
$PY demo/graphrag_bench.py run-all --config-dir "$CONFIG_DIR" --out-dir "$OUT"

echo "=== Reported vs reproduced ==="
$PY scripts/hare/compare_results.py --reported work/fang2026/results --reproduced "$OUT/results"
