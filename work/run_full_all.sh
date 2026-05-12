#!/usr/bin/env bash
# Full-corpus evaluation of all configs. All evals use JR (judge reprompt) — _rp suffix is canonical.
set -eo pipefail

# Load API keys if not already in environment
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

GEN_PHASE=1
EVAL_PHASE=1
for _arg in "$@"; do
    case "$_arg" in
        --gen-only)  EVAL_PHASE=0 ;;
        --eval-only) GEN_PHASE=0  ;;
    esac
done

PY="${PY:-/root/miniforge/envs/chonk/bin/python}"
COMMON="--out-dir work --question-ids work/data/full_corpus_stratified_order.json"
NOBC="--db-name chunkymonkey_nobc_1100_2200.duckdb"
EVAL_FLAGS="--judge gpt-4o-mini --eval-rpm 8000 --eval-batch-size 20 --concurrency 50 --nan-limit 136"

_get_score() {
    $PY demo/graphrag_bench.py score --out-dir work --run-name "$1" 2>/dev/null || echo "nan"
}

# Generate answers (if needed), then evaluate with JR as ${name}_rp.
run_and_eval_rp() {
    local name=$1; shift

    if [ -f "work/results/bench_eval_${name}_rp.json" ]; then
        echo "=== SKIP ${name}_rp (done, score=$(_get_score "${name}_rp")) ==="
        return 0
    fi

    if [ "$GEN_PHASE" -eq 1 ]; then
        if [ -f "work/results/${name}.jsonl" ]; then
            echo "=== SKIP GEN $name (output exists) ==="
        else
            echo "=== GEN $name ==="
            $PY -u demo/graphrag_bench.py run $COMMON --run-name "$name" "$@" \
                2>&1 | tee work/logs/run_${name}.log
        fi
    fi

    if [ "$EVAL_PHASE" -eq 1 ]; then
        if [ ! -f "work/results/${name}.jsonl" ]; then
            echo "=== SKIP EVAL ${name}_rp (no gen output) ==="
            return 0
        fi
        if [ ! -f "work/results/${name}_rp.jsonl" ]; then
            cp "work/results/${name}.jsonl" "work/results/${name}_rp.jsonl"
        fi
        echo "=== EVAL ${name}_rp ==="
        local eval_exit=0
        $PY demo/graphrag_bench.py eval --out-dir work --run-name "${name}_rp" $EVAL_FLAGS \
            2>&1 | tee -a work/logs/run_${name}.log || eval_exit=$?
        if [ "$eval_exit" -ne 0 ]; then exit "$eval_exit"; fi
        echo "=== DONE ${name}_rp: All=$(_get_score "${name}_rp") ==="
        $PY demo/graphrag_bench.py report --out-dir work 2>/dev/null
    fi
}

if [ "$GEN_PHASE" -eq 1 ]; then
    echo "=== Priming embedding caches ==="
    $PY demo/graphrag_bench.py prime-cache --out-dir work
fi

# ── Original grid promotions (ranked by grid score) ───────────────────────────
run_and_eval_rp ner_ref_rerank_laned_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_cluster_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --cluster \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned_pruned_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --redundancy-threshold 0.92 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned55_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.55 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_cluster_laned_community_pruned_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --cluster \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --redundancy-threshold 0.92 \
    --top-k 10 $NOBC

run_and_eval_rp rerank_k10_full \
    --rerank --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned_community_pruned_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --redundancy-threshold 0.92 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned_community_pruned_k20_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --redundancy-threshold 0.92 \
    --top-k 20 $NOBC

run_and_eval_rp ner_ref_rerank_laned_community_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    $NOBC

run_and_eval_rp ner_ref_rerank_laned_community_k15_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --top-k 15 $NOBC

run_and_eval_rp ner_ref_rerank_laned_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned60_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned_community_k7_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --top-k 7 $NOBC

# ── Baselines ─────────────────────────────────────────────────────────────────
run_and_eval_rp vanilla_256_rerank_full \
    --rerank --vanilla

run_and_eval_rp vanilla_256_no_rerank_full \
    --vanilla

# ── Extended: sim threshold sweep ─────────────────────────────────────────────
run_and_eval_rp ner_ref_rerank_laned65_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.65 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned70_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.70 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned60_community_k15_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 15 $NOBC

run_and_eval_rp ner_ref_rerank_laned60_community60_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.6 \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_rerank_laned60_pruned_community_k10_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --redundancy-threshold 0.92 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 $NOBC

# ── GraphRAG native search modes ──────────────────────────────────────────────
run_and_eval_rp ner_ref_global_search_k10_full \
    --enhanced --entity-ref-expansion \
    --search-mode global \
    --top-k 10 $NOBC

run_and_eval_rp ner_ref_graph_first_k10_full \
    --enhanced --entity-ref-expansion \
    --search-mode graph_first \
    --top-k 10 $NOBC

# ── Generator model ablations (gpt-4o, no rerank, laned60 + community) ────────
run_and_eval_rp vanilla_256_gpt4o_full \
    --vanilla --gen-model gpt-4o

run_and_eval_rp vanilla_256_llama8b_full \
    --vanilla --rerank \
    --gen-provider together \
    --gen-model "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

run_and_eval_rp ner_ref_laned60_community_k30_gpt4o_full \
    --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 30 \
    --gen-model gpt-4o \
    $NOBC

run_and_eval_rp ner_ref_laned60_community_k30_gpt4o_srr_full \
    --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 30 \
    --gen-model gpt-4o \
    --srr \
    $NOBC

run_and_eval_rp ner_ref_rerank_laned60_community_k30_gpt4o_srr_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 30 \
    --gen-model gpt-4o \
    --srr \
    $NOBC

run_and_eval_rp ner_ref_laned60_community_k30_mini_srr_full \
    --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 30 \
    --srr \
    $NOBC

run_and_eval_rp ner_ref_rerank_laned60_community_k30_mini_srr_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 30 \
    --srr \
    $NOBC

run_and_eval_rp ner_ref_laned60_community_k30_llama8b_srr_full \
    --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 30 \
    --gen-provider together \
    --gen-model "meta-llama/Meta-Llama-3-8B-Instruct-Lite" \
    --srr \
    $NOBC

run_and_eval_rp ner_ref_rerank_laned60_community_k10_llama8b_srr_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.60 \
    --community-context --community-min-coherence 0.5 \
    --top-k 10 \
    --gen-provider together \
    --gen-model "meta-llama/Meta-Llama-3-8B-Instruct-Lite" \
    --srr \
    $NOBC

# ── Constrained generator: top graph config without SRR (§7.6) ────────────────
run_and_eval_rp ner_ref_rerank_laned_community_pruned_k20_llama8b_full \
    --rerank --enhanced --entity-ref-expansion \
    --lane-entity-min-sim 0.45 \
    --community-context --community-min-coherence 0.5 \
    --redundancy-threshold 0.92 \
    --top-k 20 \
    --gen-provider together \
    --gen-model "meta-llama/Meta-Llama-3-8B-Instruct-Lite" \
    $NOBC

echo "=== ALL FULL RUNS COMPLETE ==="
