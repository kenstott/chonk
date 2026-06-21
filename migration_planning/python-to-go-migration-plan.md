# Chonk: Python → Go Migration Plan

> **Source:** Python 3.11+ monolith (`chonk` package, ~25,600 LOC across 116 source files)
> **Target:** Go 1.22+ monolith (`chonk-go` module, single deployable binary)
> **Architecture:** Monolithic — all packages compiled into one executable, no microservices split
> **Prepared:** Based on full codebase analysis + existing future-state architecture documents

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Target State Definition](#3-target-state-definition)
4. [Dependency & Risk Mapping](#4-dependency--risk-mapping)
5. [Migration Strategy](#5-migration-strategy)
6. [Phased Plan & Timeline](#6-phased-plan--timeline)
7. [Rollback Procedures](#7-rollback-procedures)
8. [Testing Strategy](#8-testing-strategy)
9. [Tooling & Infrastructure Setup](#9-tooling--infrastructure-setup)
10. [Go Dependency Manifest](#10-go-dependency-manifest)
11. [Definition of Done](#11-definition-of-done)

---

## 1. Executive Summary

Chonk is a RAG (Retrieval-Augmented Generation) pipeline library currently implemented in Python.
The migration rewrites it in Go to achieve a single self-contained binary with no Python runtime,
no `pip install`, no model downloads, and a significantly smaller memory footprint.

**Key architectural decisions already finalised in prior design work:**

| Decision | Detail |
|---|---|
| ML inference | Replace all in-process ML (`sentence-transformers`, `torch`, `spacy`, `scikit-learn`, `igraph`, `leidenalg`) with Together.ai REST API calls |
| Extraction scope | Reduce from 27 Python extractors to 5 Go extractors (text, CSV, Markdown, HTML, database schema) |
| Package layout | Mirror Python sub-package names exactly; add one new package (`graphtypes`) to break a circular import |
| Config compatibility | Go YAML config schema is backwards-compatible with existing Python `.yaml` / `.toml` config files |
| Storage | Keep DuckDB (via `go-duckdb`) and pgvector (via `pgx/v5`) — same SQL schemas |

**Total estimated calendar time: 20 weeks** across 7 sequential dependency layers with internal
parallelism where the DAG permits.

---

## 2. Current State Assessment

### 2.1 Codebase Metrics

| Dimension | Value |
|---|---|
| Python source files | 116 `.py` files |
| Total Python LOC | ~25,600 lines |
| Go files already present | 13 stub/prototype files in `chonk/extractors/go/` |
| Unit test files | 45 test files |
| Integration test files | 8 test files |
| Test LOC | ~13,900 lines |
| Package groups | 13 functional sub-packages |

### 2.2 Package Inventory

| Package | Key files | LOC (approx) | Complexity |
|---|---|---|---|
| `storage` | `_store.py` (1055), `_vector.py` (775), `_pg.py` (922) | ~3,200 | High |
| `chunking` | `chunking.py` (935) | ~935 | Medium |
| `ingest` | `ingest.py` (899), `_ingest_worker.py` (531) | ~1,430 | High |
| `search` | `_enhanced.py` (892), `_enhanced_graph.py` (699) | ~1,800 | High |
| `loader` | `loader.py` (792) | ~792 | Medium |
| `community` | `_index.py` (731), `_builder.py`, `_build.py` | ~1,100 | High |
| `transports` | 20 transport files including SharePoint (533), web crawler (349), GitHub (322) | ~3,500 | High |
| `extractors` | 27 extractor files, largest: `_edgar.py` (485) | ~2,800 | Medium–High |
| `ner` | `_schema_vocab.py` (464), `_build.py` (315), `_pipeline.py` (279) | ~1,500 | High |
| `graph` | `_context_graph.py` (353), `_builder.py` (156) | ~800 | High |
| `cluster` | `_clusterer.py`, `_cooccurrence.py`, `_map.py` | ~400 | Medium |
| `generation` | `_answer.py`, `_context.py`, `_prompt_builder.py` | ~350 | Low |
| `models` / `_config` / `_types` | core dataclasses | ~250 | Low |

### 2.3 External Python Dependencies Being Replaced

| Python dependency | Version | Purpose | Go replacement |
|---|---|---|---|
| `sentence-transformers` | ≥5.5.1 | Text embedding | Together.ai Embeddings API |
| `torch` | ≥2.12 | Tensor ops (transitive) | Eliminated entirely |
| `spacy` | ≥3.8.14 | NER tagging | Together.ai Chat API (structured JSON) |
| `scikit-learn` | ≥1.9 | Agglomerative clustering, DBSCAN | Pure Go implementation |
| `igraph` | ≥1.0.0 | Graph construction for community detection | Pure Go (custom adjacency list) |
| `leidenalg` | ≥0.12.0 | Leiden/Louvain community partitioning | Pure Go Louvain (~120 lines) |
| `duckdb` | ≥1.5.3 | Vector + FTS storage | `github.com/marcboeker/go-duckdb` |
| `psycopg2-binary` + `pgvector` | ≥2.9.12 | Postgres + vector extension | `github.com/jackc/pgx/v5` + `pgvector-go` |
| `boto3` | ≥1.43.23 | S3 transport | `github.com/aws/aws-sdk-go-v2` |
| `paramiko` | ≥3.5.1 | SFTP transport | `golang.org/x/crypto/ssh` |
| `pypdf` | ≥6.12.2 | PDF extraction | **Dropped** (out of scope for Go port) |
| `python-docx` | ≥1.2 | DOCX extraction | **Dropped** |
| `openpyxl` | ≥3.1.5 | XLSX extraction | **Dropped** |
| `pyyaml` | ≥6.0.3 | YAML extraction + config | `gopkg.in/yaml.v3` (config only; YAML extractor dropped) |
| `requests` | ≥2.34.2 | HTTP transport | `net/http` (stdlib) |
| `pandas` | ≥2.3.3 | CSV extraction | `encoding/csv` (stdlib) |

### 2.4 Known Structural Challenges

1. **Circular import: `graph` → `cluster` → `ner` → `graph`**
   Python resolves this via lazy imports inside function bodies. Go's compiler rejects
   circular package imports at build time. Resolution: extract a new `graphtypes` package
   (Layer 2) containing only `ContextEdge`, `ContextGraph`, and `ContextGraphStats` structs.

2. **Dynamic duck-typing in protocols**
   Python uses `@runtime_checkable Protocol` for `Transport`, `Extractor`, and `VectorBackend`.
   In Go these become explicit interfaces — every implementation must declare conformance via
   method signatures. No structural typing surprises.

3. **`np.ndarray` embeddings**
   NumPy arrays appear throughout the storage and search layers. In Go these become `[]float32`
   slices (or `[][]float32` for batches). The DuckDB storage layer uses `FLOAT[1024]` column
   type unchanged.

4. **`Any`-typed fields**
   Several Python dataclasses use `Any` for optional/heterogeneous fields (e.g. `source_detail`,
   `svo_llm_client`). In Go these become `interface{}` or concrete typed unions where inference
   is possible.

5. **Dropped transports**
   GitHub, Gmail, SharePoint, Cassandra, Cosmos, Firestore, DynamoDB, IMAP, SOLR are out of
   scope for the Go port. These are stubbed with a `NotImplemented` error at registration time
   so the binary still compiles with config that references them.

---

## 3. Target State Definition

### 3.1 Go Module Layout

```
chonk-go/
├── go.mod                  ← module github.com/kenstott/chonk-go, go 1.22
├── go.sum
├── main.go                 ← CLI entry point (serve, index, search subcommands)
├── models/
│   └── models.go           ← DocumentChunk, Entity, ScoredChunk, LoadedDocument, …
├── schema/
│   └── schema.go           ← ColumnMeta, TableMeta, FieldMeta, EndpointMeta
├── graphtypes/             ← NEW: cycle-breaker package
│   └── graphtypes.go       ← ContextEdge, ContextGraph, ContextGraphStats
├── transports/
│   ├── protocol.go         ← Transport interface, FetchResult, FetchOptions
│   ├── local.go
│   ├── http.go
│   ├── s3.go
│   ├── sftp.go
│   ├── directory.go
│   ├── sql_query.go
│   └── web_crawler.go
├── extractors/
│   ├── protocol.go         ← Extractor interface, Registry
│   ├── text.go
│   ├── csv.go
│   ├── markdown.go
│   ├── db.go
│   └── html.go
├── chunking/
│   └── chunking.go         ← ChunkDocument(), section extraction, heading promotion
├── context/
│   └── context.go          ← EnrichChunk(), EnrichChunks(), breadcrumb injection
├── structinfer/
│   └── structinfer.go      ← InferCSV(), InferJSON(), InferJSONL(), InferParquet()
├── generation/
│   ├── answer.go           ← AnswerGenerator (Together.ai chat)
│   ├── context.go          ← AnswerContext builder
│   └── prompt.go           ← PromptBuilder (pure string logic)
├── together/
│   └── client.go           ← Together.ai REST client (Embed + Chat)
├── loader/
│   └── loader.go           ← DocumentLoader composing extractors + transports
├── graph/
│   ├── svo.go
│   ├── extractor.go
│   ├── index.go
│   ├── builder.go
│   ├── entity_pipeline.go
│   ├── context_graph.go
│   └── llm.go              ← LLM client interface + Together.ai impl
├── storage/
│   ├── protocol.go         ← VectorBackend interface
│   ├── store.go            ← Store façade
│   ├── vector.go           ← DuckDB vector backend
│   ├── pg.go               ← pgvector backend
│   ├── relational.go
│   ├── schema.go
│   └── pool.go
├── community/
│   ├── index.go
│   ├── builder.go
│   ├── build.go
│   └── summarizer.go       ← Together.ai community label generation
├── cluster/
│   ├── clusterer.go
│   ├── cooccurrence.go
│   └── map.go
├── ner/
│   ├── index.go
│   ├── pipeline.go         ← Together.ai NER inference
│   ├── build.go
│   ├── schema.go
│   ├── vocabulary.go
│   └── normalizer.go
├── indexer/
│   └── indexer.go          ← Background goroutine indexer, IndexHandle
├── lifecycle/
│   └── lifecycle.go        ← BuildNamespaceAsync
├── search/
│   ├── enhanced.go
│   ├── graph.go
│   ├── scoring.go
│   └── support.go
└── ingest/
    ├── ingest.go           ← Build(configPath) → *Index
    ├── index.go            ← Index façade (AddSource, Search, …)
    ├── worker.go
    └── phases.go
```

### 3.2 Runtime Properties (Target)

| Property | Python (current) | Go (target) |
|---|---|---|
| Deployment artifact | `pip install chonk` + venv | Single static binary |
| Cold start time | ~8–15 s (model load) | < 200 ms |
| Memory (idle) | ~2–4 GB (torch + model weights) | ~50–150 MB |
| Memory (indexing 10k docs) | ~6–12 GB | ~300–600 MB |
| Python runtime required | Yes | No |
| ML model download required | Yes (first run) | No (API calls only) |
| Embedding latency | Local GPU/CPU | Together.ai ~50–200 ms/batch |
| Binary size | N/A | ~30–60 MB (statically linked) |

### 3.3 Scope Boundaries

**In scope:**
- All 13 Python packages listed in §2.2
- DuckDB and pgvector storage backends
- Local, S3, SFTP, HTTP, directory, web-crawler, SQL-query transports
- Text, CSV, Markdown, HTML, database-schema extractors
- Together.ai as the sole ML inference provider

**Explicitly out of scope (initial Go port):**
- GitHub, Gmail, SharePoint, Cassandra, Cosmos, Firestore, DynamoDB, IMAP, SOLR transports
- PDF, DOCX, XLSX, PPTX, YAML, ODF, JSON, XML, Parquet, Python-AST, TypeScript, Java,
  FHIR, EDGAR, email, MIME, ClinicalTrials, FDA-label, ATT&CK, CVE, CWE, NIST extractors
- MCP server (`mcp_chonk_server.py`)
- Demo scripts and benchmark harness

---

## 4. Dependency & Risk Mapping

### 4.1 Package Dependency DAG

```
Layer 0 (pure leaves)
  models ─────────────────────────────────────────────────────┐
  schema ──────────────────┐                                  │
  transports ──────────────┤                                  │
                           │                                  │
Layer 1 (stateless logic)  │                                  │
  structinfer  ◄───────────┤                                  │
  chunking     ◄───────────┼──────────────────────────────────┤
  context      ◄───────────┼──────────────────────────────────┤
  extractors   ◄───────────┼──────────────────────────────────┤
  generation   ◄───────────┘                                  │
                                                              │
Layer 2 (cycle break)                                         │
  graphtypes   ◄────────────────────────────────────────────── (models only)

Layer 3 (graph + loader)
  graph        ◄─── graphtypes, models, together
  loader       ◄─── extractors, transports, chunking, context, structinfer

Layer 4 (storage + community)
  storage      ◄─── models, graphtypes
  community    ◄─── models, storage, together

Layer 5 (parallel group)
  cluster      ◄─── graphtypes, ner
  ner          ◄─── graphtypes, storage, together
  indexer      ◄─── models, storage, together
  lifecycle    ◄─── ner, community, storage

Layer 6 (search)
  search       ◄─── cluster, community, generation, graph, ner, storage

Layer 7 (orchestration)
  ingest       ◄─── ALL of the above
```

### 4.2 Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Together.ai API latency degrades real-time search | Medium | High | Add local embedding cache (DuckDB `embeddings` table); expose `CHONK_EMBED_CACHE=true` env flag |
| R2 | Together.ai rate limits block bulk ingestion | Medium | High | Implement token-bucket rate limiter in `together.Client`; expose configurable RPS cap |
| R3 | `go-duckdb` CGo requirement breaks cross-compilation | High | Medium | Document that the Go binary must be built on the target OS/arch; use Docker build image for Linux |
| R4 | DuckDB SQL schema drift between Python and Go versions | Low | High | Snapshot current DDL from `chonk/storage/_schema.py` and pin it as a migration baseline in `storage/schema.go` |
| R5 | Circular import cycle not fully resolved by `graphtypes` | Low | High | `go build ./...` is the gate; CI must pass before any Layer 3+ work starts |
| R6 | Behaviour divergence in chunking (heading detection, overlap) | Medium | High | Port all Python chunking unit tests verbatim as Go table-driven tests before writing any Go implementation |
| R7 | NER quality regression (spacy → LLM) | Medium | Medium | Baseline NER F1 on held-out test set before migration; re-run after Go NER is live; define acceptable regression threshold (≤5% F1 drop) |
| R8 | Together.ai structured JSON output unreliable for NER | Medium | Medium | Add JSON repair fallback (equivalent to Python `json-repair` dev dep); define retry-with-reprompt path |
| R9 | Key personnel unavailable during long-running layers | Low | Medium | Document all design decisions in `docs/` as each layer completes; no bus-factor concentration |
| R10 | Go DuckDB FTS index rebuild semantics differ from Python | Low | High | Write integration test that round-trips the same document through both Python and Go builds and asserts identical FTS results |

---

## 5. Migration Strategy

### 5.1 Approach: Parallel Build, Not Strangler Fig

Because `chonk` is a library (not a running service), the strangler-fig pattern does not
directly apply. Instead, the strategy is:

1. **Build the Go module in a sibling repository** (`chonk-go/`) developed in parallel
   with the Python source.
2. **Use the Python test suite as a specification**: every Python unit test is ported to
   Go before its corresponding package is implemented (test-first / red-green-refactor).
3. **Freeze Python after Phase 3**: once the Go storage and search layers are validated,
   no new Python features are accepted — only bug fixes on a maintenance branch.
4. **Cutover via config flag**: consuming applications switch by pointing to the Go binary
   instead of the Python library. No code changes required in call sites that use the
   YAML-based `ChonkConfig`.

### 5.2 Branching Model

```
main (Python — frozen after Phase 3)
  └── maintenance/python-v1  ← security/bug fixes only after cutover
chonk-go/main                ← new Go repository, independent git history
  ├── layer/0-models
  ├── layer/1-stateless
  ├── layer/2-graphtypes
  ├── layer/3-graph-loader
  ├── layer/4-storage-community
  ├── layer/5-cluster-ner-indexer
  ├── layer/6-search
  └── layer/7-ingest
```

Each layer branch is merged to `chonk-go/main` only after its completion criterion
(all tests green, no race detector warnings) is met.

### 5.3 Parity Validation Gates

At the end of each layer, a parity check is run:

- **Unit gate:** `go test -race ./[package]/...` — zero failures, zero races.
- **Diff gate (layers 1, 3, 4, 6, 7):** A golden-file comparator runs the Python
  and Go implementations on the same corpus and diffs outputs (chunk boundaries,
  entity lists, search result IDs). Divergences above threshold block merge.
- **Performance gate (layer 4+):** `go test -bench=. ./storage/...` — throughput must
  be ≥ Python baseline on the same hardware (DuckDB ingestion, vector search QPS).

---

## 6. Phased Plan & Timeline

> Estimates assume one senior Go engineer and one Python domain expert working full-time.
> Calendar weeks account for review, testing, and integration overhead.

### Phase 0 — Foundation (Weeks 1–2)

**Goal:** Repository, toolchain, and CI bootstrapped. Layer 0 packages complete.

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Create `chonk-go` repository, initialise `go.mod` | Go Eng | Day 1 | `go build ./...` on empty module |
| Set up CI (GitHub Actions): `go test -race ./...`, `go vet`, `staticcheck` | Go Eng | Day 1–2 | CI green on empty module |
| Port `models` package — all dataclasses to Go structs | Go Eng | Days 3–4 | `go test ./models/...` |
| Port `schema` package — config schema types | Go Eng | Day 4 | `go test ./schema/...` |
| Port `transports` — `Transport` interface + local, HTTP, S3, SFTP, directory, SQL | Go Eng | Days 5–10 | `go test ./transports/...` |
| Document dropped transport stubs (`NotImplemented` returns) | Go Eng | Day 10 | Stubs compile, return descriptive error |

**Deliverable:** `chonk-go` Layer 0 branch merged to main. All pure data types in Go.

**Risks this phase:** CGo toolchain setup for `go-duckdb` may require OS-level headers.
Mitigate by using the `go-duckdb` amalgamation build tag which bundles DuckDB source.

---

### Phase 1 — Stateless Logic (Weeks 3–5)

**Goal:** All pure-logic packages complete. No ML, no network, no storage.

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Port `chunking` — `ChunkDocument()`, heading promotion, overlap, sentence split | Go Eng | Days 1–4 | All 935 LOC ported; Python `test_chunking.py` cases replicated as Go table tests |
| Port `context` — `EnrichChunk()`, `EnrichChunks()`, breadcrumb injection | Go Eng | Day 5 | `go test ./context/...` |
| Port `extractors` — 5 formats only; `Extractor` interface + `Registry` | Go Eng | Days 6–10 | `go test ./extractors/...` — text, CSV, MD, HTML, DB schema |
| Port `generation` — `PromptBuilder`, `AnswerContext` (pure string); stub `AnswerGenerator` HTTP | Go Eng | Days 11–13 | `go test ./generation/...` — stubbed HTTP interface |
| Port `structinfer` — `InferCSV()`, `InferJSON()`, `InferJSONL()` | Go Eng | Day 14 | `go test ./structinfer/...` |
| Implement `together` client — `Embed()` + `Chat()` with retry/back-off | Go Eng | Days 13–15 | Unit test with `httptest` server mock |

**Deliverable:** `chonk-go` Layer 1 branch merged. Diff-gate: chunking outputs match Python
on 5 fixture documents (≤1 chunk boundary difference per document).

**Key translation notes for `chunking`:**
- All regexps must be compiled at `init()` (Go best practice, mirrors Python module-level compile).
- `NOVEL_STRUCTURAL_LEVELS` constant set ported as a `map[string]bool`.
- `promote_plain_text_headers` becomes `PromotePlainTextHeaders([]DocumentChunk) []DocumentChunk`.

---

### Phase 2 — Cycle Breaker (Week 6, partial)

**Goal:** `graphtypes` package created, unblocking parallel Layer 3 work.

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Extract `ContextEdge`, `ContextGraph`, `ContextGraphStats` from Python `graph/_context_graph.py` | Go Eng | 1 day | `go test ./graphtypes/...` |
| Verify `graph`, `cluster`, `ner` do NOT yet exist (CI check) | CI | Automated | `go build ./...` emits no cycle error |

**Deliverable:** `graphtypes` package merged. Layer 3 work can begin.

---

### Phase 3 — Graph & Loader (Weeks 6–9)

**Goal:** Document loading pipeline and graph extraction complete. Can be worked in parallel.

#### Stream A: `graph` package (Weeks 6–9)

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Port `SVOExtractor` and `SVOTriple` — calls Together.ai Chat | Go Eng | Days 1–3 | Unit test with mock LLM interface |
| Port `RelationshipIndex` and `RelationshipIndexBuilder` | Go Eng | Days 4–6 | `go test ./graph/...` — index construction |
| Port `ContextGraph` — uses `graphtypes` types | Go Eng | Days 7–9 | Graph traversal tests |
| Port `EntityGraphPipeline` — orchestration with injectable LLM | Go Eng | Days 10–14 | End-to-end pipeline test with stub LLM |

#### Stream B: `loader` package (Weeks 6–8)

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Port `DocumentLoader` struct and constructor options | Go Eng | Days 1–2 | Compiles |
| Wire `extractors.Registry` + `transports.Transport` + `chunking` + `context` | Go Eng | Days 3–6 | `go test ./loader/...` — load from local file, assert chunks |
| Port `LoadedDocument` format detection logic | Go Eng | Days 7–10 | MIME-type and extension routing tested |

**Deliverable:** `graph` and `loader` Layer 3 branch merged. Diff-gate: `loader` produces
≡ chunk count ± 2% vs Python on test corpus of 50 documents.

---

### Phase 4 — Storage & Community (Weeks 9–13)

**Goal:** Persistence layer and community detection complete. Highest complexity phase.

#### Stream A: `storage` package (Weeks 9–12)

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Snapshot Python DuckDB DDL from `_schema.py`; pin as migration baseline | Python Expert | Day 1 | `storage/schema.go` contains exact DDL |
| Port `VectorBackend` interface | Go Eng | Day 1 | Interface defined |
| Implement `DuckDBVectorBackend` — `AddChunks`, `RegisterDocument`, `Search`, `RebuildFTSIndex` | Go Eng | Days 2–10 | `go test ./storage/...` with `:memory:` DuckDB |
| Implement `PgVectorBackend` — same interface over pgvector | Go Eng | Days 11–16 | `go test ./storage/...` with Docker Postgres (integration) |
| Port `Store` façade — namespace/domain routing, `SyncDocument` | Go Eng | Days 17–20 | Round-trip: add → search → delete |
| Thread safety: `sync.RWMutex` on all public methods | Go Eng | Day 20 | `go test -race ./storage/...` — zero races |

#### Stream B: `community` package (Weeks 9–11)

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Implement pure-Go Louvain (~120 lines) replacing `leidenalg` | Go Eng | Days 1–4 | Unit test: partition stable synthetic graph |
| Port `CommunityIndex` and `CommunityIndexBuilder` | Go Eng | Days 5–8 | `go test ./community/...` |
| Port `CommunitySummarizer` — Together.ai Chat API | Go Eng | Days 9–12 | Mock HTTP test |
| Port `build_community()` orchestration | Go Eng | Days 13–15 | End-to-end test with synthetic embeddings |

**Deliverable:** `storage` and `community` Layer 4 branch merged. Performance gate:
DuckDB ingestion ≥ Python throughput; vector search P95 latency ≤ 2× Python baseline.

---

### Phase 5 — Cluster, NER, Indexer, Lifecycle (Weeks 13–17)

**Goal:** All four parallel packages complete. Highest concurrency risk (race conditions).

#### `cluster` package (Weeks 13–15)

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Implement agglomerative clustering in pure Go (replaces `scikit-learn`) | Go Eng | Days 1–5 | Deterministic output on fixed seed; `go test ./cluster/...` |
| Port `CooccurrenceMatrix`, `ClusterMap` | Go Eng | Days 6–8 | Matrix construction + lookup tests |

#### `ner` package (Weeks 13–16)

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Embed NER label taxonomy via `//go:embed` | Go Eng | Day 1 | Binary includes vocab JSON |
| Port `VocabularyMatcher` and `EntityIndex` (pure in-memory + DuckDB) | Go Eng | Days 2–6 | `go test ./ner/...` — index CRUD |
| Implement `NERPipeline` via Together.ai structured JSON output | Go Eng | Days 7–12 | Mock LLM test; JSON repair fallback tested |
| Baseline NER F1 evaluation: Python spacy vs Go Together.ai LLM | Python Expert + Go Eng | Days 13–15 | F1 regression ≤ 5% on held-out set |

#### `indexer` package (Weeks 14–16)

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Port `Indexer` — background goroutine, `IndexHandle`, abort via `context.CancelFunc` | Go Eng | Days 1–5 | `go test -race ./indexer/...` |
| Implement `embedBatch()` via Together.ai `Embed()` | Go Eng | Days 3–5 | Batch ordering test (sort by index) |

#### `lifecycle` package (Week 16–17)

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Port `BuildNamespaceAsync` — orchestrates NER + community rebuild | Go Eng | Days 1–4 | `go test ./lifecycle/...` with stub NER + community |

**Deliverable:** All four Layer 5 packages merged. Race detector clean across all.
NER F1 baseline documented in `docs/ner-quality-baseline.md`.

---

### Phase 6 — Search (Weeks 17–18)

**Goal:** 4-lane enhanced search complete and behaviorally equivalent to Python.

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Port `EnhancedSearch` struct + constructor | Go Eng | Day 1 | Compiles, fields match Python |
| Implement vector (seed) lane | Go Eng | Days 1–2 | Unit test: top-k retrieved by cosine |
| Implement structural (adjacency) lane | Go Eng | Days 3–4 | Unit test: adjacent-chunk retrieval |
| Implement entity lane | Go Eng | Days 5–6 | Unit test: entity-linked chunks |
| Implement cluster lane | Go Eng | Days 7–8 | Unit test: cluster-adjacent chunks |
| Implement BM25 hybrid via DuckDB FTS | Go Eng | Days 9–10 | Hybrid RRF scoring test |
| Port `RetrievalTrace` provenance tracking | Go Eng | Days 10 | Trace JSON output matches Python format |
| Diff-gate: compare Go and Python search result IDs on shared corpus | Both | Days 11–12 | Overlap ≥ 80% top-10 results |

**Deliverable:** `search` Layer 6 branch merged. Search QPS ≥ Python baseline.

---

### Phase 7 — Orchestration & Cutover (Weeks 18–20)

**Goal:** `ingest` package complete; end-to-end validation; Python frozen; cutover.

| Task | Owner | Duration | Exit Criteria |
|---|---|---|---|
| Port `Build(configPath)` — YAML config load, backwards-compatible with Python config | Go Eng | Days 1–3 | Loads existing `.yaml` config files unchanged |
| Port `Index` façade — `AddSource`, `RemoveSource`, `Search`, `AddNamespace`, `AddDomain` | Go Eng | Days 4–7 | Integration test: full pipeline on 5 file formats |
| Port background `IngestWorker` | Go Eng | Days 8–9 | Goroutine worker test |
| **Freeze Python `main` branch** | Lead | Day 7 | `maintenance/python-v1` branch created; PR policy updated |
| End-to-end integration test: load `aipa_test_mcp_server/index_config.yaml`, ingest fixture corpus, run 20 search queries, assert result quality | Both | Days 10–12 | All 20 queries return ≥ 1 relevant result |
| Performance benchmarks: ingestion throughput, search latency, memory footprint | Go Eng | Day 13 | Results recorded in `docs/performance-baseline.md` |
| Security audit: `go vet`, `gosec`, `staticcheck` on full module | Go Eng | Day 14 | Zero high-severity findings |
| **Tag `chonk-go` v0.1.0** | Lead | Day 14 | GitHub release created |
| Update consumer documentation to reference Go binary | Both | Day 15 | README updated |

**Deliverable:** `chonk-go` v0.1.0 released. Python library moved to maintenance-only.

---

### Timeline Summary

```
Week  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
      ├──┤
P0    [══]                                                       Foundation
P1         [═══════]                                            Stateless Logic
P2                  [═]                                         Cycle Breaker
P3    Stream A           [══════════]                           graph
      Stream B       [═══════]                                  loader
P4    Stream A (storage)          [══════════════]              Storage
      Stream B (community)        [═══════]                     Community
P5    cluster                                  [═══]            Cluster
      ner                                      [══════]         NER
      indexer                                     [═══]         Indexer
      lifecycle                                      [══]       Lifecycle
P6                                                    [═══]     Search
P7                                                        [═══] Ingest + Cutover
```

Total: **20 calendar weeks** (~5 months) for two engineers working full-time.

---

## 7. Rollback Procedures

Rollback is possible at any phase because:
1. The Go module lives in a separate repository — the Python library is always runnable.
2. Config files are backwards-compatible — switching back requires only changing which
   binary consumers invoke.
3. DuckDB store files are format-stable — a store written by the Go binary can be read
   by the Python library (same DDL).

### 7.1 Per-Phase Rollback Triggers

| Trigger | Action |
|---|---|
| Layer tests fail and cannot be fixed within 3 business days | Revert layer branch; re-evaluate design |
| NER F1 regression > 10% (Phase 5) | Pause Phase 5 NER work; evaluate alternative Together.ai model or hybrid spacy+LLM strategy |
| DuckDB schema divergence detected (Phase 4) | Freeze Go storage; run Python schema migration to align; re-run storage tests |
| Performance regression > 2× Python (Phase 6 search) | Profile with `pprof`; identify hot path; do NOT merge until resolved |
| Critical security finding in `gosec` scan (Phase 7) | Block v0.1.0 tag; fix before release |

### 7.2 Full Rollback Procedure

If a decision is made to abandon the Go migration after Phase 4:

1. Tag the current Go state as `chonk-go/abandoned-vX.Y` for reference.
2. Re-open the Python `main` branch for new features.
3. Discard the `maintenance/python-v1` branch.
4. Document lessons learned in `docs/migration-retrospective.md`.

No data is lost: DuckDB stores written by Go are readable by Python and vice versa.

### 7.3 Partial Rollback: Single Package

If a single Go package is non-functional (e.g. `ner`):

1. Identify all callers of the broken package.
2. Replace the Together.ai NER call with a no-op that returns an empty entity list.
3. Re-tag and release a `v0.1.x` patch with NER disabled (feature flag: `CHONK_NER=false`).
4. Continue debugging the NER implementation on a separate branch without blocking the release.

---

## 8. Testing Strategy

### 8.1 Test Tier Structure

```
tests/
├── unit/           ← Pure Go tests, no external dependencies, run in CI on every commit
│   ├── models/
│   ├── chunking/
│   ├── context/
│   ├── extractors/
│   ├── generation/
│   ├── graph/
│   ├── ner/
│   ├── cluster/
│   ├── search/
│   └── ingest/
├── integration/    ← Require Docker (DuckDB :memory: is fine; pgvector needs container)
│   ├── storage/
│   ├── loader/
│   └── ingest/
└── golden/         ← Diff outputs against Python-generated fixture files
    ├── chunking/
    ├── search/
    └── ner/
```

### 8.2 Python-to-Go Test Porting Guide

Each Python unit test is ported as a Go table-driven test before the corresponding
implementation is written. The mapping is:

| Python test file | Go test file | Notes |
|---|---|---|
| `tests/unit/test_chunking.py` | `chunking/chunking_test.go` | Each `@pytest.mark.parametrize` case → table row |
| `tests/unit/test_context.py` | `context/context_test.go` | Struct field assertions |
| `tests/unit/test_extractors.py` | `extractors/extractors_test.go` | 5 format fixtures |
| `tests/unit/test_generation.py` | `generation/generation_test.go` | Prompt string assertions |
| `tests/unit/test_context_graph.py` | `graphtypes/graphtypes_test.go` | Struct construction |
| `tests/unit/test_graph.py` | `graph/graph_test.go` | Mock LLM interface |
| `tests/unit/test_ner.py` | `ner/ner_test.go` | Mock Together.ai via `httptest` |
| `tests/unit/test_ner_pipeline.py` | `ner/pipeline_test.go` | NER pipeline orchestration |
| `tests/unit/test_cluster.py` | `cluster/cluster_test.go` | Deterministic seed clustering |
| `tests/unit/test_enhanced_search.py` | `search/enhanced_test.go` | All 4 lanes tested independently |
| `tests/unit/test_indexer.py` | `indexer/indexer_test.go` | Goroutine lifecycle, race-free |
| `tests/integration/test_storage.py` | `storage/storage_test.go` | DuckDB `:memory:` round-trip |
| `tests/integration/test_pg_backend.py` | `storage/pg_test.go` | Docker postgres integration |

### 8.3 Golden File Generation

Before porting each layer, run the Python implementation on the standard corpus and save
outputs as golden files:

```bash
# Generate chunking golden files
python scripts/generate_golden.py --module chunking --corpus tests/fixtures/ \
    --output chonk-go/golden/chunking/

# Verify Go outputs match
go test ./chunking/... -run TestGolden -update=false
```

Golden files are committed to the repository and act as regression guards.

### 8.4 Race Detector Policy

**Every package test must pass `go test -race`** before its layer branch is merged.
Race conditions in the indexer (goroutine embedder) and storage (concurrent reads/writes)
are the highest-risk areas. The CI pipeline runs `-race` on every push.

---

## 9. Tooling & Infrastructure Setup

### 9.1 Go Toolchain Requirements

```
Go 1.22+           # Minimum for range-over-func, improved slog
CGo enabled        # Required for go-duckdb (DuckDB amalgamation)
gcc / clang        # C compiler for CGo
```

### 9.2 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/go.yml (chonk-go repository)
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env: { POSTGRES_PASSWORD: test }
    steps:
      - uses: actions/setup-go@v5
        with: { go-version: '1.22' }
      - run: go test -race -count=1 ./...
      - run: go vet ./...
      - run: staticcheck ./...
      - run: gosec ./...
  bench:
    runs-on: ubuntu-latest
    steps:
      - run: go test -bench=. -benchmem ./storage/... ./search/... | tee bench.txt
      - uses: benchmark-action/github-action-benchmark@v1
```

### 9.3 Development Environment

```bash
# Install Go 1.22
brew install go@1.22   # macOS
# or use https://go.dev/dl/

# Install DuckDB CGo requirements
brew install duckdb    # provides headers

# Install static analysis tools
go install honnef.co/go/tools/cmd/staticcheck@latest
go install github.com/securego/gosec/v2/cmd/gosec@latest

# Clone and initialise
git clone https://github.com/kenstott/chonk-go
cd chonk-go
go mod download
go test ./...
```

### 9.4 Environment Variables (Runtime)

```bash
# Required
TOGETHER_API_KEY=<key>

# Optional (with defaults)
CHONK_EMBED_MODEL=togethercomputer/m2-bert-80M-8k-retrieval
CHONK_NER_MODEL=meta-llama/Llama-3-8b-chat-hf
CHONK_CHAT_MODEL=meta-llama/Llama-3-70b-chat-hf
CHONK_EMBED_BATCH_SIZE=256
CHONK_NER=true
CHONK_EMBED_CACHE=false
```

---

## 10. Go Dependency Manifest

```
module github.com/kenstott/chonk-go

go 1.22

require (
    // Storage
    github.com/marcboeker/go-duckdb     v1.x   // DuckDB via CGo
    github.com/jackc/pgx/v5             v5.x   // PostgreSQL driver
    github.com/pgvector/pgvector-go     v0.x   // pgvector helper types

    // Transports
    github.com/aws/aws-sdk-go-v2        v2.x   // S3 transport
    golang.org/x/crypto                 v0.x   // SFTP (ssh package)

    // Extractors
    golang.org/x/net                    v0.x   // HTML parser (x/net/html)

    // Config
    gopkg.in/yaml.v3                    v3.x   // YAML config loading

    // Stdlib only — NO external ML dependencies
    // NO torch / sentence-transformers / spacy / sklearn / igraph
)
```

**Intentionally absent:**
- No ML framework (all inference is Together.ai REST)
- No ORM (raw `database/sql` + `pgx/v5`)
- No web framework (stdlib `net/http` for the Together.ai client and any future HTTP handlers)

---

## 11. Definition of Done

The migration is complete when ALL of the following are true:

### Functional Completeness
- [ ] `go test ./...` passes with zero failures on a clean checkout
- [ ] `go test -race ./...` passes with zero race conditions
- [ ] All 7 dependency layers have their completion-criterion tests passing
- [ ] End-to-end integration test: load `index_config.yaml` → ingest 5 formats → 20 search queries → ≥1 relevant result each
- [ ] Golden-file diff gate: chunking, NER, search outputs match Python reference within tolerance

### Scope Boundaries
- [ ] All 5 target extractors (text, CSV, Markdown, HTML, DB schema) implemented
- [ ] Both storage backends (DuckDB, pgvector) implemented and tested
- [ ] All 7 target transports (local, HTTP, S3, SFTP, directory, SQL-query, web-crawler) implemented
- [ ] Together.ai client: Embed + Chat with retry/back-off + rate limiting

### Quality Gates
- [ ] `go vet ./...` — zero warnings
- [ ] `staticcheck ./...` — zero findings
- [ ] `gosec ./...` — zero high-severity findings
- [ ] NER F1 regression ≤ 5% vs Python spacy baseline (documented in `docs/ner-quality-baseline.md`)
- [ ] Storage ingestion throughput ≥ Python baseline
- [ ] Search P95 latency ≤ 2× Python baseline (documented in `docs/performance-baseline.md`)
- [ ] Binary size ≤ 100 MB (statically linked on Linux amd64)

### Documentation & Handoff
- [ ] `chonk-go/README.md` updated with build instructions and environment variable reference
- [ ] `docs/migration-retrospective.md` capturing lessons learned
- [ ] Python `main` branch frozen; `maintenance/python-v1` created
- [ ] `chonk-go` v0.1.0 GitHub release tagged with changelog

---

*End of migration plan. Last updated based on codebase analysis of `chonk` v0.1.0 (25,600 LOC, 116 source files).*

