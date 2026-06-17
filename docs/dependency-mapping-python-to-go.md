# Dependency Mapping — Python → Go

> **Scope:** Chonk RAG pipeline rewrite from Python to Go 1.22+.
> Source: `chonk` Python package and its optional extras.
> Target: `github.com/kenstott/chonk-go` Go module.

---

## 1. Go Module Declaration

```go
module github.com/kenstott/chonk-go

go 1.22

require (
    github.com/marcboeker/go-duckdb     v1.x   // storage (DuckDB vector store)
    github.com/jackc/pgx/v5             v5.x   // storage (PostgreSQL)
    github.com/pgvector/pgvector-go     v0.x   // storage (pgvector extension)
    github.com/aws/aws-sdk-go-v2        v2.x   // transports (S3)
    golang.org/x/crypto                 v0.x   // transports (SFTP)
    golang.org/x/net                    v0.x   // extractors (HTML parsing)
    gopkg.in/yaml.v3                    v3.x   // ingest (YAML config)
)

// ELIMINATED — no local ML runtime on the host:
// NO torch  NO sentence-transformers  NO spacy  NO scikit-learn  NO igraph  NO leidenalg
```

---

## 2. Third-Party Dependency Mapping

### 2a. Storage Backends

| Python package | pip extra | Go equivalent | Notes |
|---|---|---|---|
| `duckdb` (Python binding) | `chonk[storage]` | `github.com/marcboeker/go-duckdb` | Direct CGo binding to the same DuckDB engine; SQL dialect identical. |
| `psycopg2` | `chonk[pgvector]` | `github.com/jackc/pgx/v5` | Pure-Go Postgres driver; replaces psycopg2's C extension. API differs — use `pgx.Connect` instead of `psycopg2.connect`. |
| `pgvector` (Python) | `chonk[pgvector]` | `github.com/pgvector/pgvector-go` | Companion to pgx; registers the `vector` type. Usage: `pgvector.NewVector([]float32{...})`. |

### 2b. Transports

| Python package | pip extra | Go equivalent | Notes |
|---|---|---|---|
| `boto3` | `chonk[s3]` | `github.com/aws/aws-sdk-go-v2` | AWS SDK v2 for Go. `s3.Client.GetObject` maps to `boto3.client("s3").get_object`. Session config via env vars (`AWS_REGION`, `AWS_ENDPOINT_OVERRIDE`) is identical. |
| `paramiko` | `chonk[sftp]` | `golang.org/x/crypto/ssh` + `github.com/pkg/sftp` | `golang.org/x/crypto` provides the SSH handshake; a thin SFTP client on top handles `sftp://` URIs. `AutoAddPolicy` equivalent: `ssh.InsecureIgnoreHostKey()` (mark with `// nosec`). |
| `requests` / `httpx` | `chonk[http]` | `net/http` (stdlib) | Go's standard `net/http` package fully replaces requests for HTTP/HTTPS transport. No extra module required. |
| `google-api-python-client`, `google-auth-oauthlib` | `chonk[gmail]` | **Out of scope** — Gmail transport is not ported. | Stub with `errors.New("not implemented")` if the interface slot must exist. |

### 2c. Extractors / Document Parsing

| Python package | pip extra | Go equivalent | Notes |
|---|---|---|---|
| `pypdf` / `pdfminer` | `chonk[pdf]` | **Dropped** | PDF extraction is out of scope in Go. Unsupported files are skipped gracefully. |
| `python-docx` | `chonk[docx]` | **Dropped** | DOCX extraction is out of scope. |
| `openpyxl` | `chonk[xlsx]` | **Dropped** | XLSX extraction is out of scope. |
| `python-pptx` | `chonk[pptx]` | **Dropped** | PPTX extraction is out of scope. |
| `PyYAML` | `chonk[yaml]` | **Dropped as extractor** | YAML _config_ is read via `gopkg.in/yaml.v3`; YAML _documents_ are not extracted. |
| `odfpy` | `chonk[odf]` | **Dropped** | ODF/ODS/ODT extraction is out of scope. |
| `pyarrow` / `pandas` (parquet) | `chonk[parquet]` | **Dropped** | Parquet/Arrow extraction is out of scope. |
| `pandas` (CSV) | core | `encoding/csv` (stdlib) | Go's `encoding/csv` replaces pandas for row-per-chunk CSV extraction. |
| `html.parser` / `lxml` | core | `golang.org/x/net/html` | HTML parsing; `golang.org/x/net` is the idiomatic Go replacement. |
| `markdown` / `mistune` | core | stdlib `strings` + `regexp` | Markdown heading detection uses pure regexp, no external parser required. |

### 2d. ML / AI Libraries (all replaced by Together.ai API calls)

| Python package | pip extra | What it did | Go replacement | Notes |
|---|---|---|---|---|
| `sentence-transformers` | core (heavy) | Text embedding via local model | `together.Client.Embed()` (HTTP call) | Default model: `togethercomputer/m2-bert-80M-8k-retrieval` (1024-dim). Batch size 256. Env var: `CHONK_EMBED_MODEL`. |
| `torch` | transitive dep | Tensor ops backing sentence-transformers | **Eliminated** | No tensors needed; Together.ai returns `[]float32` directly. |
| `spacy` | core | NER tagging | `together.Client.Chat()` with structured JSON prompt | Model: `meta-llama/Llama-3-8b-chat-hf`. Env var: `CHONK_NER_MODEL`. Label set embedded via `//go:embed`. |
| `scikit-learn` | `chonk[cluster]` | Agglomerative clustering, DBSCAN | Pure Go implementation (~150 lines in `cluster/clusterer.go`) | Cosine similarity matrix + linkage is straightforward without sklearn. |
| `igraph` | `chonk[leiden]` | Graph construction for Louvain/Leiden | Pure Go adjacency list (`graphtypes` package) | No external graph library needed for Louvain. |
| `leidenalg` | `chonk[leiden]` | Community detection | Pure Go Louvain (~120 lines in `community/`) | Louvain converges in < 10 iterations for typical RAG graph sizes. |
| `numpy` | transitive | Array/matrix ops | `[]float32` slices + stdlib `math` | All cosine similarity and dot product ops inline with native slices. |

### 2e. Config & Serialisation

| Python package | pip extra | Go equivalent | Notes |
|---|---|---|---|
| `PyYAML` / `pyyaml` | core | `gopkg.in/yaml.v3` | YAML config schema stays identical; existing `.yaml` files work unchanged. |
| `pydantic` | core | Plain Go `struct` + manual validation | No code-generation; validation is explicit `if` checks in constructors. |
| `tomli` / `tomllib` | core | `github.com/BurntSushi/toml` (optional) | Only if TOML benchmark config is ported; otherwise out of scope. |

---

## 3. Python Standard Library → Go Standard Library

| Python stdlib module | Used for | Go equivalent |
|---|---|---|
| `pathlib.Path` | File path manipulation | `path/filepath` |
| `re` | Regular expressions | `regexp` (compile once via `regexp.MustCompile` at `init()`) |
| `io.BytesIO` | In-memory byte buffer | `bytes.Buffer` / `bytes.Reader` |
| `urllib.parse` | URI parsing | `net/url` |
| `os` | Env vars, file I/O | `os` (same package name, similar API) |
| `json` | JSON encode/decode | `encoding/json` |
| `csv` | CSV parsing | `encoding/csv` |
| `threading.Thread` | Background workers | goroutine + `chan` |
| `threading.Event` | Cancellation signal | `context.Context` + `context.CancelFunc` |
| `threading.Lock` / `RLock` | Mutual exclusion | `sync.Mutex` / `sync.RWMutex` |
| `typing.Optional[X]` | Nullable value | `*X` (pointer) or `(X, bool)` return |
| `typing.Protocol` | Structural interface | `interface` |
| `dataclasses.dataclass` | Value type / record | `struct` (no generated methods) |
| `typing.list[X]` | Typed list | `[]X` slice |
| `typing.dict[K, V]` | Typed map | `map[K]V` |
| `abc.ABC` / `abstractmethod` | Abstract base class | `interface` |
| `logging` | Structured logging | `log/slog` (Go 1.21+) |
| `unittest` / `pytest` | Testing | `testing` stdlib + `github.com/stretchr/testify` |
| `contextlib.contextmanager` | Resource management | `defer` + explicit `Close()` |
| `functools.lru_cache` | Memoisation | `sync.Map` or package-level `map` with `sync.Once` |

---

## 4. Naming Convention Mapping

| Python pattern | Go equivalent |
|---|---|
| `_private_func` | `privateFunc` (unexported, lowercase first letter) |
| `ClassName` | `ClassName` (exported struct) |
| `snake_case` arguments | `camelCase` arguments |
| `Optional[X]` parameter | `*X` pointer parameter |
| `**kwargs` | Explicit `Options` struct passed by value |
| `@dataclass` | `struct` with no auto-generated methods |
| `Protocol` | `interface` |
| `threading.Thread` worker | goroutine launched with `go func()` |
| `threading.Event` | `context.Context` cancellation |
| `_foo.py` internal helper module | unexported file or function within the same package |
| `__init__.py` re-exports | package-level exported symbols in any `.go` file |

---

## 5. Dropped Features (not ported)

The following Python extras have **no Go equivalent** and are explicitly out of scope:

| Feature | Python extra | Reason dropped |
|---|---|---|
| PDF extraction | `chonk[pdf]` | Requires heavy native libs (poppler/ghostscript); not worth the CGo complexity |
| DOCX extraction | `chonk[docx]` | python-docx has no mature Go equivalent with the same API surface |
| XLSX extraction | `chonk[xlsx]` | openpyxl has no Go equivalent in scope |
| PPTX extraction | `chonk[pptx]` | python-pptx has no Go equivalent in scope |
| ODF extraction | `chonk[odf]` | Rarely used; dropped for simplicity |
| Parquet/Arrow | `chonk[parquet]` | Heavy dependency; out of scope for Go build |
| Gmail transport | `chonk[gmail]` | OAuth flow complexity; source-specific connector is out of scope |
| GitHub transport | (core) | Source-specific connector; stub with `NotImplemented` or omit |
| SharePoint transport | (core) | Source-specific connector; out of scope |
| FHIR extractor | (core) | Domain-specific; out of scope |
| EDGAR extractor | (core) | Domain-specific; out of scope |

---

## 6. Migration Order (Layer-by-Layer)

Work proceeds bottom-up through the dependency DAG. Every package in a layer can be
converted **in parallel**. Do not start a layer until all packages in the previous
layer have passing Go unit tests.

### Layer Summary

| Layer | Packages | Can parallelize? | Complexity |
|---|---|---|---|
| **0** | `models`, `schema`, `transports` | Start here | Low |
| **1** | `chunking`, `context`, `extractors`, `generation`, `structinfer` | All in parallel | Medium |
| **2** | `graphtypes` | — | Low |
| **3** | `graph`, `loader` | Both in parallel | High |
| **4** | `storage`, `community` | Both in parallel | High |
| **5** | `cluster`, `ner`, `indexer`, `lifecycle` | All in parallel | High |
| **6** | `search` | — | High |
| **7** | `ingest` | — (final integration) | Medium |

### Per-Layer Notes

**Layer 0 — Pure leaves (no internal deps)**
- `models`: Plain structs only. Zero external dependencies.
- `schema`: Config schema types. Zero logic. Zero external dependencies.
- `transports`: `Transport` interface + implementations. External deps: `aws-sdk-go-v2` (S3), `golang.org/x/crypto` (SFTP).

**Layer 1 — Stateless logic (depends only on models/schema)**
- No ML, no storage, no network calls in this layer.
- `extractors`: Five formats only (text, CSV, Markdown, SQL/DB schema, HTML). External dep: `golang.org/x/net/html` for HTML only.
- `generation`: `AnswerGenerator` makes Together.ai chat calls; inject HTTP interface so unit tests use a stub.
- `structinfer`: Renamed from `_struct_inference`; Go identifiers cannot begin with an underscore.

**Layer 2 — Shared graph types**
- `graphtypes`: Shared type definitions for graph/storage boundary. Only imports `models`. Required to break the `graph` vs `storage` import cycle.

**Layer 3 — Stateful pipeline assembly**
- `graph`: `SVOExtractor` calls Together.ai chat API. Inject client via interface for tests.
- `loader`: `DocumentLoader` composes `extractors.Registry`, `transports.Transport`, `chunking`, `context`, `structinfer`.

**Layer 4 — Persistence and community**
- `storage`: `Store` facade over DuckDB + optional pgvector. Thread safety via `sync.RWMutex`. BM25 via DuckDB FTS.
- `community`: Louvain in pure Go. `CommunitySummarizer` calls Together.ai.

**Layer 5 — Parallel ML-adjacent group**
- `ner`: `SpacyMatcher` replaced by Together.ai structured chat. Same label taxonomy.
- `cluster`: Agglomerative clustering and DBSCAN in pure Go. No scikit-learn.
- `indexer`: Background goroutine. Abort via `context.CancelFunc`.
- `lifecycle`: Orchestrates NER + community rebuild for a namespace asynchronously.

**Layer 6 — Search**
- `search`: 4-lane cohort assembly (vector seed, structural adjacency, entity, cluster). Mirrors Python `EnhancedSearch` constructor kwargs exactly.

**Layer 7 — Orchestration (final)**
- `ingest`: `Build(configPath string)`. YAML config via `gopkg.in/yaml.v3`. Config schema is **backwards-compatible** with the Python YAML schema.

---

## 7. API Compatibility Notes

| Area | Python signature | Go signature | Status |
|---|---|---|---|
| Config files | `.yaml` schema | Same schema (`gopkg.in/yaml.v3`) | Compatible — existing config files reusable unchanged |
| Chunk struct | `DocumentChunk` dataclass fields | `DocumentChunk` struct fields | Compatible — same field names and semantics |
| Extractor interface | `can_handle(uri) -> bool` / `extract(path) -> list[DocumentChunk]` | `CanHandle(string) bool` / `Extract(string, []byte) ([]DocumentChunk, error)` | Same semantics; error is explicit return value |
| Transport interface | `can_handle(uri) -> bool` / `fetch(uri, **kwargs) -> FetchResult` | `CanHandle(string) bool` / `Fetch(ctx, string, FetchOptions) (FetchResult, error)` | `**kwargs` becomes explicit `FetchOptions` struct; `context.Context` added |
| Embedding | `SentenceTransformer.encode(texts, batch_size=256) -> np.ndarray` | `together.Client.Embed(ctx, EmbedRequest) (EmbedResponse, error)` | Remote HTTP call; same 1024-dim float32 vectors |
| NER | `SpacyMatcher.match(text) -> list[Entity]` | `NERPipeline.ExtractEntities(ctx, text) ([]Entity, error)` | Remote HTTP call; same label taxonomy; explicit error return |
| Storage write | `store.add_document(chunks)` | `store.AddDocument(chunks []DocumentChunk) error` | Explicit error return added |
| Storage search | `store.search(vec, k, ...) -> list[ScoredChunk]` | `store.Search(ctx, vec []float32, k int, ...) ([]ScoredChunk, error)` | `context.Context` and explicit error added |
| Ingest facade | `Index.search(query_vec, k)` | `Index.Search(ctx, query string, k int) ([]ScoredChunk, error)` | Go version accepts query text and embeds internally |
| Concurrency | `threading.Thread` + `threading.Event.wait()` | goroutine + `context.Context` cancellation | Identical abort semantics; no GIL in Go |
| Optional values | `Optional[X]` / `None` | `*X` pointer (nil = absent) | Nil pointer checks replace `is None` checks |
| Error handling | `raise ValueError(...)` / `try/except` | `return nil, errors.New(...)` / `if err != nil` | Explicit propagation; no exception stack unwinding |

---
