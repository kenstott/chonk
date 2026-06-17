# Code Conversion — Python to Go

## Approach

See [conversion-plan.md](conversion-plan.md) for the full layer-by-layer build order
derived from the dependency DAG.

Each Python sub-package becomes a Go package with the same name. Internal helpers
(`_foo.py`) become unexported Go functions or files within the same package. Public
symbols stay public; private helpers stay unexported.

Python `@dataclass` → Go `struct` with no methods unless behaviour is attached.
Python `Protocol` → Go `interface`.
Python optional/union types → Go pointer receivers or tagged unions where needed.
Python `threading.Thread` background workers → goroutine + channel.

## Circular Dependency — graphtypes

The Python codebase contains a cycle: `graph → cluster → ner → graph`. Python
permits this through lazy function-body imports; Go does not.

**Fix:** introduce a new leaf package `graphtypes` containing only the
`ContextEdge`, `ContextGraph`, and `ContextGraphStats` types from
`graph/_context_graph.py`. Convert this package in Layer 2 (before `graph`,
`cluster`, or `ner`). All three cycle participants then import `graphtypes`
rather than each other for these types.

## Package Conversion Table

### models

| Python | Go |
|---|---|
| `DocumentChunk` dataclass | `DocumentChunk` struct |
| `ScoredChunk` dataclass | `ScoredChunk` struct |
| `Entity`, `EntityAssociation` | unchanged struct mapping |
| `ContextEdge`, `ContextGraphStats` | unchanged struct mapping |

No external dependencies. Pure value types. Convert directly.

### chunking

**Python:** `chunking.py` — `chunk_document()`, heading promotion, section extraction,
overlap logic, table/list detection, sentence splitting.

**Go:** `chunking/` package.
- All regex patterns compile once at package init via `regexp.MustCompile`.
- `ChunkDocument(docName, content string, opts ChunkOptions) []DocumentChunk`
- `_split_at_sentences` → `splitAtSentences` using `strings` + boundary heuristics;
  no NLP library needed.
- `_promote_structural_levels`, `_promote_plain_text_headers` → unexported funcs,
  same logic with Go `regexp` and `strings`.

**No ML dependency.** Pure string processing.

### context

**Python:** `context.py` — `enrich_chunk()`, `enrich_chunks()` — prepends breadcrumb
to `embedding_content`.

**Go:** `context/` package.
- `EnrichChunk(chunk DocumentChunk) DocumentChunk`
- `EnrichChunks(chunks []DocumentChunk) []DocumentChunk`

**No ML dependency.** Pure string manipulation.

### extractors

See [extraction-scope.md](extraction-scope.md) for full details. Only five extractors
are carried forward.

**Go:** `extractors/` package with sub-files per format:
- `text.go`, `csv.go`, `markdown.go`, `db.go`, `html.go`
- Shared `Extractor` interface:
  ```go
  type Extractor interface {
      CanHandle(filename string) bool
      Extract(filename string, content []byte) ([]DocumentChunk, error)
  }
  ```
- `Registry` — maps file extension to `Extractor`; replaces Python's `_INGEST_FNS` dict.

### transports

**Python:** local, S3, SFTP, HTTP, directory crawler, DB schema, import crawler, web
crawler, and many source-specific transports (GitHub, Gmail, SharePoint, Cassandra,
MongoDB, Elasticsearch, …).

**Go:** carry forward the core transports; source-specific connectors are out of scope
for the initial Go port.

| Transport | Go status |
|---|---|
| local filesystem | `transports/local.go` |
| S3 | `transports/s3.go` (aws-sdk-go-v2) |
| SFTP | `transports/sftp.go` (golang.org/x/crypto/ssh) |
| HTTP/HTTPS | `transports/http.go` (net/http) |
| Directory crawler | `transports/directory.go` |
| DB schema (SQL) | `transports/db.go` (database/sql) |
| Web crawler | `transports/web.go` |
| Import crawler | out of scope (Python-specific) |
| GitHub, Gmail, SharePoint, etc. | out of scope for initial port |

**Interface:**
```go
type Transport interface {
    List(ctx context.Context) ([]FileRef, error)
    Fetch(ctx context.Context, ref FileRef) ([]byte, error)
}
```

### storage

**Python:** `Store` façade over `DuckDBVectorBackend` + `RelationalStore`;
optional `_pg.py` pgvector backend.

**Go:** `storage/` package.
- `DuckDB` backend via `github.com/marcboeker/go-duckdb`.
- `Postgres` backend via `pgx/v5` + `pgvector-go`.
- `Store` interface matches Python's public surface: `AddDocument`, `Search`,
  `RegisterNamespace`, `RegisterDomain`, `Count`, `RebuildFTSIndex`.
- Thread safety: `sync.RWMutex` on the DuckDB connection (same single-writer
  constraint as Python's `ThreadLocalDuckDB`).

### indexer

**Python:** `Indexer` — background thread, embed batch, abort flag.

**Go:** `indexer/` package.
- `Indexer` struct with `IndexSource(ctx, src)` and `IndexSourceAsync(ctx, src)`.
- Background work via goroutine; abort via `context.CancelFunc`.
- Embedding calls go to Together.ai (see [ml-services.md](ml-services.md)).

### ner

**Python:** spacy-based `SpacyMatcher`, `NERPipeline`, `EntityIndex`,
`_schema_vocab.py` vocabulary.

**Go:** `ner/` package.
- `SpacyMatcher` replaced by Together.ai chat completion (structured JSON output).
  See [ml-services.md](ml-services.md) — "NER Inference".
- `EntityIndex` — identical in-memory and DuckDB-backed structure, ported directly.
- Vocabulary and label sets remain as Go `const` blocks or JSON files embedded via
  `//go:embed`.

### graph

**Python:** `SVOExtractor`, `GraphBuilder`, `ContextGraph`, `EntityPipeline`,
`GraphIndex`, LLM-backed triple extraction.

**Go:** `graph/` package.
- `SVOExtractor` — LLM calls go to Together.ai chat API (same prompt structure).
- `ContextGraph`, `GraphIndex` — pure graph structs, no external dependency.
- `EntityPipeline` — orchestrates NER → dedup → store, identical flow.

### cluster

**Python:** `AgglomerativeClustering` and `DBSCAN` from scikit-learn; Leiden/Louvain
from igraph + leidenalg.

**Go:** `cluster/` package.
- `AgglomerativeClustering` — implement in pure Go using cosine similarity matrix.
  The algorithm is simple enough (~100 lines); no external library needed.
- `DBSCAN` — implement in pure Go or use `github.com/ericmort/godbscan`.
- Leiden/Louvain — see community package below.

### community

**Python:** Leiden community detection via `igraph`/`leidenalg`; `CommunityIndex`,
`CommunityBuilder`, summarisation via LLM.

**Go:** `community/` package.
- Graph construction (similarity edges) — pure Go, cosine similarity over embedding
  vectors from Together.ai.
- Leiden/Louvain partition — use `github.com/yourbasic/graph` or implement a
  simple Louvain variant. The Python code's `_run_leiden` / `_run_louvain` are
  ~60 lines each and can be ported directly once the graph adjacency structure
  is reproduced.
- `CommunitySummarizer` — LLM call to Together.ai chat API, same prompt.

### search

**Python:** `EnhancedSearch` — 4-lane cohort assembly (seed, structural, entity,
cluster) with scoring mixins.

**Go:** `search/` package.
- `EnhancedSearch` struct — same 4 lanes, each optional via config.
- Vector lane: cosine similarity against DuckDB VSS or pgvector.
- BM25 FTS lane: DuckDB full-text search.
- Structural, entity, cluster lanes: pure Go logic, no ML.

### generation

**Python:** `AnswerGenerator`, `PromptBuilder`, `ContextBuilder` — LLM-backed.

**Go:** `generation/` package.
- All LLM calls go to Together.ai chat API.
- `PromptBuilder` — string template logic, pure Go.
- `AnswerGenerator.Generate(ctx, question, chunks)` → calls Together.ai.

### ingest

**Python:** `build()`, `Index` façade, `_build_ingest_phase`, YAML config loading.

**Go:** `ingest/` package.
- `Build(configPath string) (*Index, error)` — reads YAML config (gopkg.in/yaml.v3),
  drives the same phases: ingest → embed → FTS → NER → community → SVO.
- `Index` struct — same `AddSource`, `RemoveSource`, `Search`, `AddNamespace`,
  `AddDomain` surface.
- Config schema stays identical to the Python YAML schema; existing `.yaml` files
  are reusable without modification.

## Go Module Layout

```
module github.com/kenstott/chonk-go

go 1.22

require (
    github.com/marcboeker/go-duckdb v1.x
    github.com/jackc/pgx/v5 v5.x
    github.com/pgvector/pgvector-go v0.x
    github.com/aws/aws-sdk-go-v2 v2.x
    golang.org/x/crypto v0.x   // SFTP
    gopkg.in/yaml.v3 v3.x
    // NO torch, NO sentence-transformers, NO spacy, NO sklearn
)
```

## Naming Conventions

| Python pattern | Go equivalent |
|---|---|
| `_private_func` | `privateFunc` (unexported) |
| `ClassName` | `ClassName` (exported struct) |
| `snake_case` args | `camelCase` args |
| `Optional[X]` | `*X` or `(X, bool)` return |
| `list[X]` | `[]X` |
| `dict[K, V]` | `map[K]V` |
| `@dataclass` | `struct` (no generated methods) |
| `Protocol` | `interface` |
| `threading.Thread` | goroutine + channel |
| `threading.Event` | `context.Context` cancellation |
