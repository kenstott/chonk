# Conversion Plan — Python to Go

## How to Read This Document

Work proceeds layer by layer. Every package in a layer can be converted in parallel
because all of its dependencies are already done. Do not start a layer until all
packages in the previous layer have passing unit tests in Go.

---

## Dependency DAG (package level)

```
models ──────────────────────────────────────────────────────┐
schema ──────────────────┐                                   │
transports ──────────────┤                                   │
                         ▼                                   │
            _struct_inference                                │
            chunking     ◄──────────────────────────────────┤
            context      ◄──────────────────────────────────┤
            extractors   ◄──────────────────────────────────┤
            generation   ◄──────────────────────────────────┤
                         │
                         ▼
                       loader ──► graph ──► storage
                                    │          │
                                    │          ▼
                                    │      community
                                    │          │
                                    ▼          ▼
                                  cluster   indexer
                                    │      lifecycle
                                    ▼        ner
                                  search ────┘
                                    │
                                    ▼
                                  ingest
```

---

## ⚠ Cycle: graph → cluster → ner → graph

Python allows this because the imports inside `graph._context_graph`,
`cluster._cooccurrence`, and `ner._build` are lazy (inside function bodies).
Go does not permit circular package imports — the compiler rejects them.

**Resolution:** extract a new leaf package `graphtypes` containing only the
`ContextEdge` and `ContextGraph` structs from `graph/_context_graph.py`.
- `graph` imports `graphtypes` instead of defining those types itself.
- `cluster` and `ner` import `graphtypes` instead of `graph`.
- The cycle is eliminated. `graphtypes` has no dependencies beyond `models`.

This package is created in Layer 2, before any of the three cycle participants
are converted.

---

## Layer 0 — Pure Leaves

**Packages:** `models`, `schema`, `transports`

No internal dependencies. Start here.

| Package | Go file(s) | Notes |
|---|---|---|
| `models` | `models/models.go` | `DocumentChunk`, `ScoredChunk`, `Entity`, `EntityAssociation`, `ContextEdge`, `ContextGraphStats` as plain structs. |
| `schema` | `schema/schema.go` | Config schema types. No logic. |
| `transports` | `transports/*.go` | `Transport` interface + local, S3, SFTP, HTTP, directory, DB, web crawler implementations. Source-specific connectors (GitHub, Gmail, SharePoint, …) are out of scope — stub with `NotImplemented` or omit. |

**Completion criterion:** `go test ./models/... ./schema/... ./transports/...` passes.

**Unit tests:** See [unit-testing.md](unit-testing.md) — Layer 0 section. Templates: `tests/unit/test_context.py` (struct field preservation), `tests/integration/test_storage.py` (struct construction), `tests/integration/test_loader.py` (transport list/fetch).

---

## Layer 1 — Stateless Logic

**Packages:** `chunking`, `context`, `extractors`, `generation`, `_struct_inference`

All depend only on `models` and/or `schema`. No ML, no storage, no network.

| Package | Go file(s) | Notes |
|---|---|---|
| `chunking` | `chunking/chunking.go` | `ChunkDocument()`, heading promotion, section extraction, overlap, sentence split. Pure string + regexp. Compile all regexps at `init()`. |
| `context` | `context/context.go` | `EnrichChunk()`, `EnrichChunks()`. Breadcrumb prepend only. |
| `extractors` | `extractors/{text,csv,markdown,db,html}.go` | Five formats only — see `extraction-scope.md`. `Extractor` interface + `Registry`. |
| `generation` | `generation/{answer,context,prompt}.go` | `PromptBuilder`, `ContextBuilder` are pure string logic. `AnswerGenerator` makes Together.ai chat calls — stub the HTTP call behind an interface so unit tests don't need a live key. |
| `_struct_inference` | `structinfer/structinfer.go` | Renamed to valid Go identifier. Depends only on `schema`. |

**Completion criterion:** `go test ./chunking/... ./context/... ./extractors/... ./generation/... ./structinfer/...` passes with no network calls.

**Unit tests:** See [unit-testing.md](unit-testing.md) — Layer 1 section. Templates: `tests/unit/test_chunking.py`, `tests/unit/test_context.py`, `tests/unit/test_extractors.py`, `tests/unit/test_generation.py`.

---

## Layer 2 — Cycle Break

**Packages:** `graphtypes` (new)

Create this package before converting `graph`, `cluster`, or `ner`.

| Package | Go file(s) | Notes |
|---|---|---|
| `graphtypes` | `graphtypes/graphtypes.go` | `ContextEdge`, `ContextGraph`, `ContextGraphStats` structs only. Extracted from Python's `graph/_context_graph.py`. No methods, no logic. Imports only `models`. |

**Completion criterion:** `go test ./graphtypes/...` passes. `graph`, `cluster`, and `ner` must not exist yet.

**Unit tests:** See [unit-testing.md](unit-testing.md) — Layer 2 section. Template: `tests/unit/test_context_graph.py` (struct field access), `tests/unit/test_graph.py` (struct construction).

---

## Layer 3 — Graph & Loader

**Packages:** `graph`, `loader`

Can be converted in parallel once Layer 2 is done.

| Package | Go file(s) | Notes |
|---|---|---|
| `graph` | `graph/{svo,extractor,index,builder,entity_pipeline,context_graph,llm}.go` | `SVOExtractor` and triple extraction call Together.ai chat API. `ContextGraph` implementation uses `graphtypes` types. `GraphIndex` is pure in-memory. |
| `loader` | `loader/loader.go` | `DocumentLoader` — composes `extractors.Registry`, `transports.Transport`, `chunking`, `context`, `structinfer`. Mirrors Python `DocumentLoader.__init__` options. |

**Completion criterion:** `go test ./graph/... ./loader/...` passes. Graph LLM calls behind an injectable interface so tests use a stub.

**Unit tests:** See [unit-testing.md](unit-testing.md) — Layer 3 section. Templates: `tests/unit/test_graph.py`, `tests/unit/test_extractor.py`.

---

## Layer 4 — Storage & Community

**Packages:** `storage`, `community`

Can be converted in parallel once Layer 3 is done.

| Package | Go file(s) | Notes |
|---|---|---|
| `storage` | `storage/{store,vector,relational,schema,pool,pg,protocol}.go` | `Store` façade over DuckDB (`go-duckdb`) + optional pgvector (`pgx/v5`). `RebuildFTSIndex`, `AddDocument`, `RegisterNamespace`, `RegisterDomain`, `Search`. Thread safety via `sync.RWMutex`. Uses `graphtypes` (not `graph`) for `ContextEdge`. |
| `community` | `community/{index,builder,build,summarizer}.go` | `CommunityIndex`, `CommunityBuilder`. Leiden/Louvain implemented in pure Go (~120 lines). `CommunitySummarizer` calls Together.ai. Similarity-edge construction uses cosine over float32 vectors. |

**Completion criterion:** `go test ./storage/... ./community/...` passes. Storage tests use an in-memory DuckDB (`:memory:`). Community tests use synthetic embeddings.

**Unit tests:** See [unit-testing.md](unit-testing.md) — Layer 4 section. Templates: `tests/integration/test_storage.py`, `tests/unit/test_community_summarizer.py`, `tests/unit/test_context_graph.py`.

---

## Layer 5 — Parallel Group

**Packages:** `cluster`, `ner`, `indexer`, `lifecycle`

All unblocked once Layers 3 and 4 are complete. Convert in parallel.

| Package | Go file(s) | Notes |
|---|---|---|
| `cluster` | `cluster/{clusterer,cooccurrence,map}.go` | Agglomerative clustering and DBSCAN in pure Go. Imports `graphtypes`, not `graph`. `ClusterMap` uses `ner.EntityIndex` types — import `ner` for index access. |
| `ner` | `ner/{index,pipeline,build,schema,schema_vocab,vocabulary,merge,normalizer,spacy_labels}.go` | `SpacyMatcher` replaced by Together.ai structured chat output. `EntityIndex` is a pure in-memory + DuckDB index. Imports `graphtypes`, not `graph`. Vocabulary embedded via `//go:embed`. |
| `indexer` | `indexer/indexer.go` | Background goroutine indexer. Embedding via Together.ai. Abort via `context.CancelFunc`. `IndexHandle` wraps a channel. |
| `lifecycle` | `lifecycle/lifecycle.go` | `BuildNamespaceAsync` — orchestrates NER + community rebuild for a namespace. Mirrors Python's `build_namespace_async`. |

**Completion criterion:** `go test ./cluster/... ./ner/... ./indexer/... ./lifecycle/...` passes. ML calls behind injectable interfaces.

**Unit tests:** See [unit-testing.md](unit-testing.md) — Layer 5 section. Templates: `tests/unit/test_cluster.py`, `tests/unit/test_ner.py`, `tests/unit/test_ner_pipeline.py`, `tests/unit/test_indexer.py`.

---

## Layer 6 — Search

**Package:** `search`

Needs cluster, community, generation, graph, ner, and storage — all done in Layer 5.

| Package | Go file(s) | Notes |
|---|---|---|
| `search` | `search/{enhanced,graph,scoring,support}.go` | 4-lane cohort assembly: seed (vector), structural (adjacency), entity, cluster. Each lane independently toggle-able. `EnhancedSearch` struct mirrors Python constructor kwargs. BM25 via DuckDB FTS. |

**Completion criterion:** `go test ./search/...` passes with in-memory store and synthetic embeddings covering all four lanes independently and combined.

**Unit tests:** See [unit-testing.md](unit-testing.md) — Layer 6 section. Template: `tests/unit/test_enhanced_search.py`.

---

## Layer 7 — Orchestration

**Package:** `ingest`

The final package. Needs everything above.

| Package | Go file(s) | Notes |
|---|---|---|
| `ingest` | `ingest/{ingest,index,worker,phases}.go` | `Build(configPath string) (*Index, error)`. YAML config via `gopkg.in/yaml.v3`. Config schema is backwards-compatible with the Python YAML schema — existing `.yaml` files work unchanged. `Index` façade exposes `AddSource`, `RemoveSource`, `Search`, `AddNamespace`, `AddDomain`. Background worker via goroutine. |

**Completion criterion:** end-to-end integration test — load `index_config.yaml`, ingest the five supported formats, run a search query, verify results.

**Unit tests:** See [unit-testing.md](unit-testing.md) — Layer 7 section. Templates: `tests/integration/test_loader.py`, `tests/unit/test_indexer.py`. Test fixtures live under `ingest/testdata/`.

---

## Go Module Dependencies

```
module github.com/kenstott/chonk-go

go 1.22

require (
    github.com/marcboeker/go-duckdb     v1.x   // storage
    github.com/jackc/pgx/v5             v5.x   // storage (postgres)
    github.com/pgvector/pgvector-go     v0.x   // storage (postgres)
    github.com/aws/aws-sdk-go-v2        v2.x   // transports (S3)
    golang.org/x/crypto                 v0.x   // transports (SFTP)
    golang.org/x/net                    v0.x   // extractors (HTML)
    gopkg.in/yaml.v3                    v3.x   // ingest (config)
)
// NO torch  NO sentence-transformers  NO spacy  NO sklearn  NO igraph
```

---

## Work Breakdown Summary

| Layer | Packages | Parallel? | Estimated complexity |
|---|---|---|---|
| 0 | models, schema, transports | — | Low |
| 1 | chunking, context, extractors, generation, structinfer | All parallel | Medium |
| 2 | graphtypes | — | Low |
| 3 | graph, loader | Parallel | High |
| 4 | storage, community | Parallel | High |
| 5 | cluster, ner, indexer, lifecycle | All parallel | High |
| 6 | search | — | High |
| 7 | ingest | — | Medium |
