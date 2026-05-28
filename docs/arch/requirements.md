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
