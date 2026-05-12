# Requirements

## R1 — Model Selection

**Generation model:** `gpt-4o-mini`
**Judge model:** `gpt-4o-mini`

All benchmark runs must use `gpt-4o-mini` for both answer generation and evaluation judging. This matches the GraphRAG-Bench leaderboard standard (arXiv:2506.05690), where all published entries use `gpt-4o-mini` as both generator and judge, enabling direct comparison against leaderboard scores.

## R2 — Answer Correctness NaN Handling

- **REQ-1** (2026-04-24): NaN rate for answer_correctness scores must be ≤6% per eval run.
- **REQ-2** (2026-04-24): When factuality scoring fails (NaN) but semantic similarity succeeds, fall back to similarity-only score instead of propagating NaN.
- **REQ-3** (2026-04-24): Return NaN only when both factuality and semantic similarity scoring fail.

## R3 — Graph Asset Observability in Bench

- **REQ-4** (2026-05-12): Each answer record in bench JSONL must capture which graph assets contributed to context: entities matched from query, SVO triples retrieved, entity descriptions/aliases used, community summaries included, and entity-ref expansion stats for post-hoc analysis of graph asset influence on answer quality.
