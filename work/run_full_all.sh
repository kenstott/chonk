#!/usr/bin/env bash
# Full-corpus evaluation of all configs via run-all.
set -eo pipefail

if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

PY="${PY:-/root/miniforge/envs/chonk/bin/python}"

echo "=== Priming embedding caches ==="
$PY demo/graphrag_bench.py prime-cache --out-dir work

$PY demo/graphrag_bench.py run-all \
    --config-dir work/configs/runs \
    --out-dir work \
    --question-ids work/data/full_corpus_stratified_order.json

echo "=== ALL FULL RUNS COMPLETE ==="
