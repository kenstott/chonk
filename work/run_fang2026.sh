#!/usr/bin/env bash
# FANG-2026 full evaluation matrix.
# Configs: vanilla+rerank, laned60+community, cluster+community, graph_first
# Features: plain vs SR
# Models:   gpt-4o-mini, claude-haiku-4-5-20251001
# Run build_fang2026.sh first.
set -eo pipefail

if [ -f ".env" ]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] && export "$line"
    done < .env
fi

PY="${PY:-/root/miniforge/envs/chonk/bin/python}"
OUT="work/fang2026"
DB="chonk_nobc_1100_2200.duckdb"
VANILLA_DB="vanilla_rag.duckdb"
COMMON="--out-dir $OUT --question-ids $OUT/data/fang2026_question_ids.json"
EVAL_FLAGS="--judge gpt-4o-mini --eval-rpm 8000 --eval-batch-size 20 --concurrency 50 --nan-limit 5"

_get_score() {
    $PY demo/graphrag_bench.py score --out-dir "$OUT" --run-name "$1" 2>/dev/null || echo "nan"
}

run_and_eval_rp() {
    local name=$1; shift

    if [ -f "$OUT/results/bench_eval_${name}_rp.json" ]; then
        echo "=== SKIP ${name}_rp (done, score=$(_get_score "${name}_rp")) ==="
        return 0
    fi

    if [ ! -f "$OUT/results/${name}.jsonl" ]; then
        echo "=== GEN $name ==="
        $PY -u demo/graphrag_bench.py run $COMMON --run-name "$name" "$@" \
            2>&1 | tee "$OUT/logs/run_${name}.log"
    else
        echo "=== SKIP GEN $name (output exists) ==="
    fi

    if [ ! -f "$OUT/results/${name}.jsonl" ]; then
        echo "=== SKIP EVAL ${name}_rp (no gen output) ==="
        return 0
    fi
    [ ! -f "$OUT/results/${name}_rp.jsonl" ] && cp "$OUT/results/${name}.jsonl" "$OUT/results/${name}_rp.jsonl"

    echo "=== EVAL ${name}_rp ==="
    local eval_exit=0
    $PY demo/graphrag_bench.py eval --out-dir "$OUT" --run-name "${name}_rp" $EVAL_FLAGS \
        2>&1 | tee -a "$OUT/logs/run_${name}.log" || eval_exit=$?
    [ "$eval_exit" -ne 0 ] && exit "$eval_exit"
    echo "=== DONE ${name}_rp: All=$(_get_score "${name}_rp") ==="
    $PY demo/graphrag_bench.py report --out-dir "$OUT" 2>/dev/null
}

mkdir -p "$OUT/results" "$OUT/logs"

echo "=== Priming embedding caches ==="
$PY demo/graphrag_bench.py prime-cache --out-dir "$OUT"

# ── Vanilla index (shared baseline) ───────────────────────────────────────────
$PY demo/graphrag_bench.py index-vanilla --out-dir "$OUT" \
    --from-store "$OUT/data/$DB" 2>&1 | tee -a "$OUT/logs/run_vanilla.log"

# ══════════════════════════════════════════════════════════════════════════════
# gpt-4o-mini
# ══════════════════════════════════════════════════════════════════════════════
MINI="--gen-provider openai --gen-model gpt-4o-mini"

# Baselines
run_and_eval_rp fang_vanilla_rerank_mini \
    --vanilla --rerank $MINI

run_and_eval_rp fang_vanilla_rerank_srr_mini \
    --vanilla --rerank --srr $MINI

# laned60 + community
run_and_eval_rp fang_ner_ref_laned60_community_k10_mini \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 --db-name "$DB" $MINI

run_and_eval_rp fang_ner_ref_laned60_community_k10_srr_mini \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 --db-name "$DB" --srr $MINI

# cluster + community
run_and_eval_rp fang_ner_ref_cluster_community_k10_mini \
    --rerank --enhanced --entity-ref-expansion \
    --cluster \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 --db-name "$DB" $MINI

run_and_eval_rp fang_ner_ref_cluster_community_k10_srr_mini \
    --rerank --enhanced --entity-ref-expansion \
    --cluster \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 --db-name "$DB" --srr $MINI

# graph_first
run_and_eval_rp fang_ner_ref_graph_first_k10_mini \
    --enhanced --entity-ref-expansion \
    --search-mode graph_first \
    --top-k 10 --db-name "$VANILLA_DB" $MINI

run_and_eval_rp fang_ner_ref_graph_first_k10_srr_mini \
    --enhanced --entity-ref-expansion \
    --search-mode graph_first \
    --top-k 10 --db-name "$VANILLA_DB" --srr $MINI

# ══════════════════════════════════════════════════════════════════════════════
# claude-haiku-4-5
# ══════════════════════════════════════════════════════════════════════════════
HAIKU="--gen-provider anthropic --gen-model claude-haiku-4-5-20251001"

# Baselines
run_and_eval_rp fang_vanilla_rerank_haiku \
    --vanilla --rerank $HAIKU

run_and_eval_rp fang_vanilla_rerank_srr_haiku \
    --vanilla --rerank --srr $HAIKU

# laned60 + community
run_and_eval_rp fang_ner_ref_laned60_community_k10_haiku \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 --db-name "$DB" $HAIKU

run_and_eval_rp fang_ner_ref_laned60_community_k10_srr_haiku \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 --db-name "$DB" --srr $HAIKU

# cluster + community
run_and_eval_rp fang_ner_ref_cluster_community_k10_haiku \
    --rerank --enhanced --entity-ref-expansion \
    --cluster \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 --db-name "$DB" $HAIKU

run_and_eval_rp fang_ner_ref_cluster_community_k10_srr_haiku \
    --rerank --enhanced --entity-ref-expansion \
    --cluster \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 --db-name "$DB" --srr $HAIKU

# graph_first
run_and_eval_rp fang_ner_ref_graph_first_k10_haiku \
    --enhanced --entity-ref-expansion \
    --search-mode graph_first \
    --top-k 10 --db-name "$VANILLA_DB" $HAIKU


# ══════════════════════════════════════════════════════════════════════════════
# claude-sonnet-4-6 — model capability ceiling
# ══════════════════════════════════════════════════════════════════════════════
SONNET="--gen-provider anthropic --gen-model claude-sonnet-4-6"

run_and_eval_rp fang_ner_ref_laned60_community_k30_rerank_srr_sonnet \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 30 --db-name "$DB" --srr $SONNET

# ══════════════════════════════════════════════════════════════════════════════
# openai/gpt-oss-120b (Together AI) — sovereign/open-weight candidate
# ══════════════════════════════════════════════════════════════════════════════
GPTOSS="--gen-provider together --gen-model openai/gpt-oss-120b"

run_and_eval_rp fang_ner_ref_laned60_community_k30_rerank_srr_gptoss120b \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 30 --db-name "$DB" --srr $GPTOSS

echo "=== ALL FANG-2026 RUNS COMPLETE ==="
$PY demo/graphrag_bench.py report --out-dir "$OUT" 2>/dev/null
