#!/usr/bin/env bash
# Full-corpus evaluation of all configs at or near the cutoff (grid All >= 0.650).
set -e

GEN_PHASE=1
EVAL_PHASE=1
for _arg in "$@"; do
    case "$_arg" in
        --gen-only)  EVAL_PHASE=0 ;;
        --eval-only) GEN_PHASE=0  ;;
    esac
done

COMMON="--out-dir work --question-ids work/data/full_corpus_stratified_order.json"
NOBC="--db-name chunkymonkey_nobc_1100_2200.duckdb"
JUDGE="gpt-4o-mini"
_get_score() {
    python demo/graphrag_bench.py score --out-dir work --run-name "$1" 2>/dev/null || echo "nan"
}

run_and_eval() {
    local name=$1; shift

    # Skip entirely if eval is already complete
    if [ -f "work/results/bench_eval_${name}.json" ]; then
        echo "=== SKIP $name (eval complete, score=$(_get_score "$name")) ==="
        return 0
    fi

    if [ "$GEN_PHASE" -eq 1 ]; then
        if [ -f "work/results/${name}.jsonl" ]; then
            echo "=== SKIP GEN $name (output exists) ==="
        else
            echo "=== START $name ==="
            python -u demo/graphrag_bench.py run $COMMON --run-name "$name" "$@" \
                2>&1 | tee work/logs/run_${name}.log
        fi
    fi

    if [ "$EVAL_PHASE" -eq 1 ]; then
        if [ ! -f "work/results/${name}.jsonl" ]; then
            echo "=== SKIP EVAL $name (no output jsonl) ==="
            return 0
        fi
        echo "=== EVAL $name (judge=$JUDGE) ==="
        local eval_exit=0
        python demo/graphrag_bench.py eval --out-dir work --run-name "$name" --judge "$JUDGE" \
            --eval-rpm 8000 --eval-batch-size 20 --concurrency 50 --nan-limit 136 \
            2>&1 | tee -a work/logs/run_${name}.log || eval_exit=$?

        if [ "$eval_exit" -ne 0 ]; then
            exit "$eval_exit"
        fi

        local score; score=$(_get_score "$name")
        echo "=== DONE $name: All=$score ==="
        python demo/graphrag_bench.py report --out-dir work 2>/dev/null
    fi
}

if [ "$GEN_PHASE" -eq 1 ]; then
    echo "=== Priming embedding caches ==="
    python demo/graphrag_bench.py prime-cache --out-dir work
fi

# ── Grid rank order (best → worst) ────────────────────────────────────────────
# Rank  Run                               Grid-All
#  1    laned_community_k10               0.685
#  2    cluster_community_k10             0.682
#  3    laned_pruned_k10                  0.676
#  4    laned55_community_k10             0.672
#  5    cluster_laned_community_pruned_k10 0.668
#  6    rerank_k10  (ablation)            0.664
#  7    laned_community_pruned_k10        0.661
#  8    laned_community_pruned_k20        0.661
#  9    laned_community                   0.661
# 10    laned_community_k15               0.659
# 11    laned_k10                         0.656
# 12    laned60_community_k10             0.653
# 13    laned_community_k7                0.650
# ─────────────────────────────────────────────────────────────────────────────

run_and_eval nobc_ner_ref_rerank_laned_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_cluster_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --cluster \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned_pruned_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --redundancy-threshold 0.92 \
    --top-k 10 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned55_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.55 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_cluster_laned_community_pruned_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --cluster \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --redundancy-threshold 0.92 \
    --top-k 10 \
    $NOBC

run_and_eval nobc_rerank_k10_full \
    --rerank \
    --top-k 10 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned_community_pruned_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --redundancy-threshold 0.92 \
    --top-k 10 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned_community_pruned_k20_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --redundancy-threshold 0.92 \
    --top-k 20 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned_community_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned_community_k15_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --top-k 15 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --top-k 10 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned60_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 \
    $NOBC

run_and_eval nobc_ner_ref_rerank_laned_community_k7_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --top-k 7 \
    $NOBC

run_and_eval vanilla_256_rerank_full \
    --rerank --vanilla

echo "=== ALL FULL RUNS COMPLETE ==="
