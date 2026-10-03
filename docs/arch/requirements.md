# Requirements

## Public API & Extension

- **REQ-001** (2026-05-20): Redaction filter hook—pluggable filter on index, called on every search/ask response if configured, intercepts generated answers before return to caller, for sanitizing sensitive data in sovereign RAG deployments.

## Storage & Vector Search

- **REQ-002** (2026-05-27): PgVectorBackend must expose `_conn` property returning a DuckDB-compatible adapter (`_PsycopgAdapter`) that translates ? placeholders to %s, allowing Store catalog methods to work transparently with PostgreSQL.
- **REQ-003** (2026-05-27): PgVectorBackend default table name is `embeddings` (matches DuckDB schema); configurable via `table` constructor parameter for backward compatibility.
- **REQ-004** (2026-05-27): Store constructor accepts `dsn: str | None = None`; when set, creates PgVectorBackend instead of DuckDB. Methods `attach_global`, `detach_global`, `build_context_graph`, `get_context_graph` raise NotImplementedError for PG backend.
- **REQ-005** (2026-05-27): PG schema includes catalog tables (namespaces, domains, sources, community_cache, namespace_build_log, entities, chunk_entities, entity_aliases, svo_triples, ner_cache, chunk_clusters, context_graph_edges, context_graph_cache) plus `ingest_queue` and `control` tables for horizontal scale support.

## Infrastructure

- **REQ-006** (2026-05-27): ingest.py exposes `run_worker(queue_dsn, backend_dsn, ...)` and `run_coordinator(queue_dsn, backend_dsn, ...)` functions; CLI via `python -m chonk.ingest --worker/--coordinator --queue DSN --backend DSN`.
- **REQ-007** (2026-05-27): Worker state machine pulls pending job via SELECT FOR UPDATE SKIP LOCKED, checks `control.workers_paused` flag before each job, marks job done/failed on completion.
- **REQ-008** (2026-05-27): Coordinator state machine runs DISPATCHING → DRAINING → BUILDING → DISPATCHING cycle; graph builds run per-namespace using temp DuckDB store for community detection; stale leases (>10 min) are requeued.
- **REQ-009** (2026-05-27): `build()` function remains DuckDB-only; PG ingestion is exclusively via worker/coordinator CLI modes.
- **REQ-010** (2026-06-16): Type-safety gate for the Go migration. Pyright standard mode promotes `reportMissingTypeArgument` and `reportOptionalSubscript` to errors; ruff enables `ANN`+`RET`. No bare collection generics (`list`/`dict`/`tuple`) and no `Any`-erasing unions (`str | Any`) on migration-critical surfaces: `EmbedModel` protocol for embedders, typed `Callable` progress/error callbacks, `FetchOptions` dataclass replacing `Transport.fetch(**kwargs)`, typed `ChonkConfig` hierarchy replacing `dict[str, Any]` config, and typed `VectorBackend._conn`. Unguarded `.fetchone()[0]` must raise, not fall back. Enforced proactively via `.claude/hooks/no-silent-fallback.py` (no fallback values / silent error handling).
- **REQ-011** (2026-09-30): Every release of chonk-rag must add a CHANGELOG.md section in Keep a Changelog format (`## [X.Y.Z] - YYYY-MM-DD`) before the version tag is pushed; entries under [Unreleased] move into the new section. Documented in `.claude/skills/release/SKILL.md`.

## Testing & Quality

- **REQ-012** (2026-09-30): HARE-Bench reproducibility: all generated benchmark inputs (corpus, index stores, embedding caches) are published to the public chonk R2 bucket (https://chonk.simpleishard.io/benchmark/fang2026/) and pinned by SHA-256 in the bucket's `manifest.json`; the bucket and its credentials are configured through environment variables, and it is the only link between initialization and benchmarking. `scripts/hare/reproduce.sh` is the claimed reproduction method: it downloads the published index and only generates and scores results—it does no chunking or indexing.
- **REQ-013** (2026-09-30): The published benchmark index is built locally (not on a GPU) from a clean git commit; the build environment is recorded in `index_build_info.json` published with the stores.
- **REQ-014** (2026-09-30): HARE-Bench "reproducible" is defined as reproducing benchmark results from published, SHA-256-pinned corpus and index artifacts; corpus/index generation is scripted and documented but not part of the reproducibility claim.
- **REQ-015** (2026-10-03): HARE-Bench rerank checkpoint validity: a checkpointed reranking is reused only for the candidate set it was ranked from. The checkpoint key includes BM25 and Auto Domain Filter settings, and each checkpoint entry stores a digest of the question's candidate chunk IDs; when the digest differs the question is reranked again and the count is reported. Implemented in `demo/graphrag_bench.py`: `_rerank_candidates_digest`, `_restore_reranked`.
- **REQ-016** (2026-10-03): HARE-Bench reproduction on a GPU: `scripts/hare/reproduce.sh --gpu` runs generation and scoring on a Vultr GPU instance with `RERANKER_DEVICE=cuda`, copies results back to the local `--out-dir` every minute, and destroys the instance when the run ends, also on failure. Model API keys and `HARE_PUBLIC_URL` are passed in the remote process environment only and are never written to the instance's disk; the remote host receives no `.env`. Finished runs are skipped so an interrupted run can be repeated. Shared Vultr provisioning lives in `scripts/hare/vultr_gpu.sh`, used by both `reproduce.sh` and `02_publish_index.sh`.
- **REQ-017** (2026-10-03): HARE-Bench GPU instances must not outlive the job. On GCP (`scripts/hare/gcp_gpu.sh`), instances are created with provider-side deadline (`--max-run-duration` with `GCP_MAX_HOURS` default 4, DELETE termination) so Compute Engine deletes them even if the launching machine dies; boot disk auto-deletes with it; no other disk, address, or snapshot is created; resources are named `chonk-hare-*` and the script fails loudly if any remain after teardown; before creating anything the script prints worst-case cost and requires confirmation (`HARE_GPU_YES=1` to pre-confirm). Vultr has no provider-side deadline and relies on the local exit trap only.
