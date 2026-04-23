#!/bin/bash
set -e
BENCH=/Volumes/MacAlt/Users/kennethstott/PycharmProjects/chunkymonkey/demo/graphrag_bench.py
OUT=/Volumes/MacAlt/Users/kennethstott/PycharmProjects/chunkymonkey/work

RUNS=(
  contextual_enhanced
  contextual_no_para
  contextual_plain
  contextual_plain_nobc
  contextual_plain_rerank
  contextual_rerank
  nobc_1100_2200
  nobc_1100_2200_ner_rerank
  nobc_1100_2200_rerank
  nobc_200_500
  nobc_300_600
  nobc_400_1000
  nobc_400_1000_ner_rerank
  nobc_400_1000_rerank
  nobc_400_1200
  nobc_400_800
  nobc_900_1300
  vanilla_128
  vanilla_192
  vanilla_256
  vanilla_256_rerank
  vanilla_384
  vanilla_384_rerank
  vanilla_rag_rerank
  vanilla_rag_v2
)

for name in "${RUNS[@]}"; do
  echo "=== eval: $name ==="
  python "$BENCH" eval --out-dir "$OUT" --run-name "$name" --judge gpt-4o-mini
done
echo "ALL DONE"
