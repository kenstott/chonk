#!/usr/bin/env bash
# Build every HARE-Bench index store from the corpus in work/corpus/.
#
# Usage: scripts/hare/build_index.sh OUT_DIR
#   Writes stores and embedding caches to OUT_DIR/data/ (graphrag_bench.py layout).
#
# Per store: chonk build (ingest, embed, FTS) -> build-ner -> build-community.
# Then vanilla_rag.duckdb from the gleif store, then the question/entity caches.
set -euo pipefail

OUT="${1:?usage: build_index.sh OUT_DIR}"
DATA="$OUT/data"
PY="${PY:-uv run python}"

$PY scripts/hare/artifacts.py verify corpus

mkdir -p "$DATA"
SRC="work/fang2026/data"
if [ "$(cd "$DATA" && pwd -P)" != "$(cd "$SRC" && pwd -P)" ]; then
    cp "$SRC/fang2026_questions.jsonl" "$SRC/fang2026_gold_schemas.jsonl" "$DATA/"
fi

build_store() {
    local cfg=$1 db=$2
    echo "=== ${db}: ingest + embed + FTS ==="
    $PY -m chonk build "work/configs/ingest/${cfg}" --store "$DATA/$db"
    echo "=== ${db}: NER ==="
    $PY demo/graphrag_bench.py build-ner --out-dir "$OUT" --db-name "$db" --with-embeddings
    echo "=== ${db}: community ==="
    $PY demo/graphrag_bench.py build-community --out-dir "$OUT" --db-name "$db"
}

build_store fang2026_bc.yaml         chonk_bc_1100_2200.duckdb
build_store fang2026_nobc.yaml       chonk_nobc_1100_2200.duckdb
build_store fang2026_nobc_gleif.yaml chonk_nobc_1100_2200_gleif.duckdb

echo "=== vanilla_rag.duckdb (naive chunks from the gleif store) ==="
$PY demo/graphrag_bench.py index-vanilla --out-dir "$OUT" \
    --from-store "$DATA/chonk_nobc_1100_2200_gleif.duckdb"

echo "=== question and entity embedding caches ==="
$PY demo/graphrag_bench.py prime-cache --out-dir "$OUT"

echo "=== index ready in $DATA ==="
