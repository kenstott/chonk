# chonk — Python → Go Migration Status

## Overview

**Source:** Python 3.11 monolith (`chonk/` package, ~24,932 LOC)  
**Target:** Go 1.22 monolith (`go_migration/`, single binary)  
**Architecture:** Monolith → Monolith (same boundaries, different runtime)  
**Module:** `github.com/kennethstott/chonk`  
**ADR set:** `aipa_test_mcp_server/future_state_architecture/`

---

## Layer Build Order (ADR)

| Layer | Sprint | Packages | Status |
|-------|--------|----------|--------|
| G0 | Layer 0 | `models`, `schema`, `transports` (core 4) | ✅ COMPLETE |
| G1 | Layer 1 | `chunking`, `extractors` (5 kept), `generation`, `together` | ✅ COMPLETE |
| G2 | Layer 2 | `graphtypes` (new — cycle break) | ✅ COMPLETE |
| G3 | Layer 3 | `graph`, `loader` | 🔲 STUB |
| G4 | Layer 4 | `storage` (DuckDB + PG backends), `community` | 🔲 STUB |
| G5 | Layer 5 | `cluster`, `ner`, `indexer`, `lifecycle` | 🔲 STUB |
| G6 | Layer 6 | `search` (4-lane full implementation) | 🔲 STUB |
| G7 | Layer 7 | `ingest` (full orchestration) | 🔲 STUB |

Legend: ✅ COMPLETE · 🔲 STUB (interface defined, implementation pending sprint)

---

## Package-by-Package Status

### ✅ `internal/models`
- **Python source:** `chonk/models.py` (129 LOC)
- **Status:** COMPLETE
- All Python `@dataclass` types ported to Go structs with constructors.
- `DocumentChunk.__post_init__` source derivation replicated in `NewDocumentChunk()` + `WithChunkType()`.
- Python `field(default_factory=list)` → Go nil slices allocated on first use.
- Python `Optional[T]` → Go `*T` pointer types.

### ✅ `internal/schema`
- **Python source:** `chonk/schema.py` (~50 LOC)
- **Status:** COMPLETE
- `ColumnMeta`, `TableMeta`, `FieldMeta`, `EndpointMeta` ported exactly.

### ✅ `internal/graphtypes`
- **Python source:** NEW — does not exist in Python.
- **Purpose:** Breaks the import cycle `graph → cluster → ner → graph`.
- **Status:** COMPLETE — `ContextEdge`, `ContextGraph`, `ContextGraphStats` with lazy adjacency index.

### ✅ `internal/extractors`
- **Python source:** `chonk/extractors/` (kept 5 of 26 extractors per ADR)
- **Status:** COMPLETE (interface + all 5 kept extractors)
- `TextExtractor`, `CSVExtractor`, `MarkdownExtractor`, `HTMLExtractor`, `SQLExtractor`
- Registry with first-match semantics (`NormalizeType`, `DetectTypeFromSource`)
- **DROPPED (21 extractors per ADR):** PDF, DOCX, XLSX, PPTX, ODF, JSON, XML, YAML,
  Parquet, Python AST, TypeScript, Java, FHIR, EDGAR, email, MIME, ClinicalTrials,
  FDA, ATT&CK, CVE, CWE, NIST, NoSQL, lookup_table, renderer
- **BeautifulSoup → `golang.org/x/net/html`** (HTML tree walker with heading promotion)
- **pandas → `encoding/csv`** (stdlib CSV with dialect sniffing)

### ✅ `internal/transports`
- **Python source:** `chonk/transports/` (core 4 of 18 transports per ADR)
- **Status:** COMPLETE (interface + `LocalTransport`, `HttpTransport`, `S3Transport`, `SftpTransport`)
- **requests → `net/http` stdlib**
- **boto3 → `aws-sdk-go-v2`**
- **paramiko → `golang.org/x/crypto/ssh`** (SftpTransport uses SSH exec; `pkg/sftp` recommended for production)
- **DROPPED (14 transports per ADR):** GitHub, Gmail, SharePoint, Cassandra, MongoDB,
  Elasticsearch, Solr, DynamoDB, CosmosDB, Firestore, IMAP, FTP, SQLAlchemy, import_crawler

### ✅ `internal/together`
- **Python source:** NEW — replaces sentence-transformers + spaCy + OpenAI-compat LLM client
- **Status:** COMPLETE — `EmbedTexts()`, `Chat()`, `ChatOptions`, functional options
- All ML inference (embedding, NER, SVO, community summarisation, answer gen) routes here.

### ✅ `internal/chunking`
- **Python source:** `chonk/chunking.py` (936 LOC)
- **Status:** COMPLETE (interface + stubs); full markdown-aware block merging is G1 detail work
- `ChunkDocument`, `ExtractMarkdownSections`, `IsListLine`, `IsTableLine`, `MergeBlocks`,
  `PromotePlainTextHeaders`, `NovelStructuralLevels` all ported.
- Core naive paragraph split implemented; section-aware merge to follow.

### ✅ `internal/generation`
- **Python source:** `chonk/generation/` (168 LOC)
- **Status:** COMPLETE
- `Answer`, `AnswerContext`, `AnswerGenerator`, `PromptBuilder` fully ported.
- Python `Callable[[str], str]` → Go `LLMFunc = func(string) (string, error)`
- Token budget estimated at 4 chars/token (same heuristic as Python).

### ✅ `internal/ner`
- **Python source:** `chonk/ner/` (2,272 LOC)
- **Status:** COMPLETE (interface + `EntityIndex` + `NerPipeline`)
- **spaCy → Together.ai JSON-mode NER** (ADR ml-services.md)
- Two-pass pipeline (VocabularyMatcher + SpacyMatcher) → single Together.ai call.
- `EntityIndex`, `EntityMatch`, `MergeMatches`, `NormalizeSchemaTermm` ported.

### 🔲 `internal/storage`
- **Python source:** `chonk/storage/` (3,372 LOC)
- **Status:** Interface COMPLETE; DuckDB + PG backends are STUBS (G4)
- `VectorBackend` interface fully defined.
- `DuckDBVectorBackend` + `PgVectorBackend` skeleton with `ErrNotImplemented`.
- `Store` facade with `AddDocument()` + `Search()` wrappers.
- **G4 work:** Wire `go-duckdb`, implement VSS + BM25 RRF hybrid search, schema migration.

### 🔲 `internal/search`
- **Python source:** `chonk/search/` (1,871 LOC)
- **Status:** Interface COMPLETE; 4-lane assembly is STUB (G6)
- `EnhancedSearch`, `EnhancedSearchOptions`, `SearchRequest`, `SearchResponse`, `RetrievalTrace` defined.
- Seed lane (vector) wired through; Structural / Entity / Cluster / MMR scoring are G6.

### 🔲 `internal/ingest`
- **Python source:** `chonk/ingest.py` (900 LOC)
- **Status:** Config loading COMPLETE; pipeline wiring is STUB (G7)
- `ChonkConfig` + YAML loading with defaults fully ported.
- `Build()` + `Index.IngestBytes()` + `Index.Search()` + `Index.Ask()` skeleton wired.
- **G7 work:** NER indexing, community building, lifecycle, namespace refresher.

### 🔲 `internal/graph`
- **Python source:** `chonk/graph/` (1,375 LOC)
- **Status:** NOT STARTED (G3 sprint)
- `graphtypes` package already provides `ContextEdge`, `ContextGraph` (cycle break done).
- SVO extraction + LLM graph builder → Together.ai Chat (JSON-mode triples).

### 🔲 `internal/cluster`
- **Python source:** `chonk/cluster/` (478 LOC)
- **Status:** NOT STARTED (G5 sprint)
- scikit-learn `AgglomerativeClustering` / `DBSCAN` → pure Go (~100 LOC per ADR).

### 🔲 `internal/community`
- **Python source:** `chonk/community/` (1,158 LOC)
- **Status:** NOT STARTED (G4 sprint)
- igraph + leidenalg → pure Go Louvain (~120 LOC per ADR).

### 🔲 `internal/indexer`
- **Python source:** `chonk/indexer.py` (287 LOC)
- **Status:** NOT STARTED (G5 sprint)
- `threading.Thread` → `goroutine` + `sync.WaitGroup`
- `_registry: dict[str, Indexer]` → `sync.Map` or `sync.Mutex`-guarded map

### 🔲 `internal/lifecycle`
- **Python source:** `chonk/lifecycle.py` (260 LOC)
- **Status:** NOT STARTED (G5 sprint)

---

## Key Architecture Decisions

### ML Replacement (ADR: ml-services.md)
| Python | Go |
|--------|----|
| `sentence-transformers` + `torch` | `together.Client.EmbedTexts()` |
| `spacy` + `thinc` | `together.Client.Chat()` JSON-mode NER |
| `scikit-learn` clustering | Pure Go (G5) |
| `igraph` + `leidenalg` | Pure Go Louvain (G4) |
| OpenAI-compat LLM | `together.Client.Chat()` |

### Import Cycle Resolution (ADR: code-conversion.md)
Python resolves `graph → cluster → ner → graph` via lazy function-body imports.
Go rejects this at compile time. Resolution: `graphtypes` leaf package (G2).

### Storage Layer (ADR: storage-design)
- DuckDB: `go-duckdb` driver (same DuckDB engine, Go FFI)
- PostgreSQL: `pgx/v5` + `pgvector-go` (near 1:1 driver swap from psycopg2)
- `ThreadLocalDuckDB` (thread-local connections) → `sync.RWMutex` single-writer guard

### Dependency Map
| Python | Go | Type |
|--------|----|------|
| `duckdb` + `duckdb-engine` | `github.com/marcboeker/go-duckdb` | DROP-IN |
| `psycopg2` + `pgvector` | `github.com/jackc/pgx/v5` + `pgvector-go` | DROP-IN |
| `boto3` | `aws-sdk-go-v2` | DROP-IN |
| `paramiko` | `golang.org/x/crypto/ssh` | DROP-IN |
| `requests` + `certifi` | `net/http` (stdlib) | DROP-IN |
| `BeautifulSoup` (html) | `golang.org/x/net/html` | DROP-IN |
| `pandas` (CSV) | `encoding/csv` (stdlib) | DROP-IN |
| `pyyaml` | `gopkg.in/yaml.v3` | DROP-IN |
| `pytest` | `testing` + `testify/assert` | REWORK |
| All ML libraries | Together.ai REST API | REDESIGN (resolved) |

---

## Coverage Target

Python repo enforces `fail_under = 70` (`pyproject.toml`).
Go equivalent enforced by `Makefile cover` and CI `test` job.

---

## Running Tests

```bash
# Unit tests (no live services)
go test ./internal/... ./cmd/...

# With coverage check (>= 70%)
make cover

# Integration tests (requires CHONK_TEST_PG_DSN)
go test -tags integration ./tests/integration/...

# Lint
golangci-lint run ./...
```
