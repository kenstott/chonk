# Python → Go Migration Risk Assessment
## Project: chonk RAG Pipeline

**Assessment Date:** 2025  
**Source Stack:** Python 3.11–3.14, ~30,200 lines across 116 modules  
**Target Stack:** Go 1.22+, single-binary deployment  
**Scoring Model:** Each risk is scored on three axes (1–5) and a Risk Priority Number computed:
`RPN = Probability × Impact × (6 − Detectability)`

> **Detectability** is scored 1 (immediately obvious) → 5 (silent/latent defect).  
> Higher detectability score = harder to catch = higher RPN contribution.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Codebase Snapshot](#2-codebase-snapshot)
3. [Risk Register](#3-risk-register)
   - [R-01 ML Runtime Elimination (sentence-transformers / torch / spaCy)](#r-01)
   - [R-02 NumPy Array Semantics → Go Slices](#r-02)
   - [R-03 Circular Import Cycle (graph ↔ cluster ↔ ner)](#r-03)
   - [R-04 DuckDB Go Driver Maturity & API Parity](#r-04)
   - [R-05 Extractor Scope Reduction (26 dropped formats)](#r-05)
   - [R-06 Go stdlib csv.Dialect / csv.Sniffer API Defect](#r-06)
   - [R-07 Python Mixin / Multiple Inheritance](#r-07)
   - [R-08 Together.ai API Key — Single External Dependency](#r-08)
   - [R-09 Thread-Safety Model Divergence (ThreadLocalDuckDB)](#r-09)
   - [R-10 Python Protocol / Structural Subtyping → Go Interfaces](#r-10)
   - [R-11 Callback / Higher-Order Function Patterns](#r-11)
   - [R-12 MCP Server Elimination / Rewrite](#r-12)
   - [R-13 Transport Surface (20 Python transports → 6 Go transports)](#r-13)
   - [R-14 YAML Config Schema Backward-Compatibility](#r-14)
   - [R-15 Test Suite Coverage Gap (13,929 Python LOC → 0 Go tests)](#r-15)
   - [R-16 Leiden/Louvain Community Detection — Pure-Go Reimplementation](#r-16)
   - [R-17 pgvector Backend Parity](#r-17)
   - [R-18 Dependency Health — Lock-File Dual State](#r-18)
   - [R-19 `dict[str, Any]` Ubiquity → Typed Go Structs](#r-19)
   - [R-20 Error Handling Paradigm Shift (exceptions → error values)](#r-20)
4. [Aggregate Risk Heatmap](#4-aggregate-risk-heatmap)
5. [Go/No-Go Criteria](#5-gono-go-criteria)
6. [Mitigation Roadmap](#6-mitigation-roadmap)

---

## 1. Executive Summary

The chonk codebase is a mature, multi-layered RAG pipeline with strong typing, explicit protocols, and a detailed conversion plan already in place. The migration to Go is ambitious but structurally viable. The architecture team has correctly identified the seven-layer dependency DAG, the circular import problem, and the ML library replacement strategy. However, **twelve risks carry RPNs that must be resolved before a Go/No-Go decision can be made**, with four in the critical band.

| Band | RPN Range | Count | Decision Gate |
|---|---|---|---|
| 🔴 Critical | ≥ 60 | 4 | Block migration start |
| 🟠 High | 40–59 | 5 | Must have mitigation plan before Layer 3 |
| 🟡 Medium | 20–39 | 7 | Address within each layer before promoting |
| 🟢 Low | < 20 | 4 | Monitor only |

**Overall recommendation:** Conditional GO — proceed through Layer 1 (stateless logic) as a low-risk proving ground. Gate Layer 3 entry on resolution of R-01, R-05, R-06, and R-08.

---

## 2. Codebase Snapshot

| Metric | Value |
|---|---|
| Python source modules | 116 files |
| Python LOC (chonk/ only) | ~30,200 |
| Test files | 45 files, ~13,929 LOC |
| Existing Go prototype files | 14 files (extractors/go/, extractors/_markdown.go, markdown_extractor.go) |
| Direct runtime dependencies | 29 packages |
| Optional extra groups | 14 groups |
| ML/numeric heavy deps | sentence-transformers, torch, numpy, scikit-learn, spaCy, igraph, leidenalg |
| Highest-complexity modules | `_store.py` (1,055 LOC), `chunking.py` (935 LOC), `ingest.py` (899 LOC), `_enhanced.py` (892 LOC) |
| Concurrency model | `threading.Thread`, `threading.Event`, `threading.RLock`, singleton `_registry` |
| Async usage | Minimal — only MCP server uses `asyncio` |
| Protocol/interface count | 4 (`Transport`, `VectorBackend`, `LLMClient`, `Embedder`, `Crawler`) |
| Transport implementations | 20 Python; target: 6 Go |
| Extractor implementations | 24 Python; target: 5 Go |

---

## 3. Risk Register

---

### R-01
## R-01 — ML Runtime Elimination (sentence-transformers / torch / spaCy)
**Category:** Architecture  **Layer(s) affected:** indexer, ner, community, search, ingest

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | Certainty — the entire local-ML stack is being replaced |
| Impact | 5 | Embeddings and NER are the core correctness axis of the pipeline |
| Detectability | 4 | Embedding quality differences are silent; RPN = 5 × 5 × (6-4) = **50** |

**RPN: 50 🟠 High**

**Detail:**  
`sentence-transformers` + `torch` produce in-process, GPU-accelerated embeddings using `BAAI/bge-large-en-v1.5` (1,024-dim). The Go target replaces this with `togethercomputer/m2-bert-80M-8k-retrieval` via the Together.ai API. These are **different models** with different vector spaces — all stored embeddings from the Python version are **incompatible** with Go-generated embeddings. Any incremental or hybrid deployment will produce incorrect similarity scores.

`spaCy` (`en_core_web_sm`) is replaced by `meta-llama/Llama-3-8b-chat-hf` via structured JSON prompts. LLM-based NER has higher latency, introduces non-determinism, and changes the entity recognition profile (LLMs tend to over-extract; spaCy is conservative). The NER label taxonomy (`_spacy_labels.py`, `_schema_vocab.py`) must be exactly preserved in the Go prompt embedding.

**Mitigation Strategies:**
1. **Model selection audit:** Benchmark `m2-bert-80M-8k-retrieval` vs `bge-large-en-v1.5` on the existing `fang2026` eval dataset before committing to the API model. If scores diverge by > 5 ROUGE points, evaluate `togethercomputer/m2-bert-80M-2k-retrieval` or request access to a bge-equivalent.
2. **Index invalidation policy:** Document that all DuckDB stores built by the Python version are invalid for Go search. Build a migration tool that re-embeds existing stores using the Go client.
3. **NER parity test:** Port `tests/unit/test_ner.py` and `test_ner_pipeline.py` as Go integration tests against the Together.ai NER endpoint using the same fixture texts. Gate Layer 5 on ≥ 90% entity recall vs Python baseline.
4. **Offline fallback:** Keep the Together.ai client behind an `Embedder` interface so an alternative (local model via HTTP, OpenAI-compatible server) can be injected for air-gapped environments.

---

### R-02
## R-02 — NumPy Array Semantics → Go Slices
**Category:** Data Model  **Layer(s) affected:** indexer, storage, cluster, community, search

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | NumPy is used in 140+ call sites across 9 modules |
| Impact | 4 | Float precision errors or shape bugs corrupt scores silently |
| Detectability | 4 | Wrong similarity scores pass all type checks; RPN = 5 × 4 × (6-4) = **40** |

**RPN: 40 🟠 High**

**Detail:**  
NumPy operations pervade the pipeline: `np.vstack`, `np.fill_diagonal`, `np.ones`, `normalize_embeddings`, cosine similarity, and `float32` casting. Python's NumPy silently handles broadcasting, dtype promotion, and memory layout (C-order vs F-order). Go's `[][]float32` has no such implicit behavior.

Critical risk areas:
- `emb = np.vstack(embeddings).astype("float32")` in `indexer.py` — Go must produce the identical flat memory layout for DuckDB VSS ingestion.
- Cosine similarity in clustering uses distance matrix normalization with `np.fill_diagonal(dist, 0.0)` — a naive Go port using nested loops will be correct but 100× slower without SIMD.
- The `preload_embeddings()` method on `DuckDBVectorBackend` loads a `(n, dim)` float32 matrix into RAM for in-process ANN search; Go must replicate this with identical memory layout.

**Mitigation Strategies:**
1. **Use `gonum/mat`** for matrix operations where broadcasting is needed; it mirrors NumPy's dense matrix API.
2. **Float32 cast enforcement:** All embedding vectors must be stored and queried as `float32`, not `float64`. Add a compile-time type alias `type Embedding = []float32` and enforce it at all call sites.
3. **Golden-file similarity tests:** For every clustering and community function, generate golden fixture files in Python (input vectors + expected cluster assignments) and assert Go produces identical output within `1e-5` tolerance.
4. **DuckDB VSS wire format:** Verify that `go-duckdb`'s array insertion format for HNSW vector columns matches what Python's `duckdb` produces. Run a cross-language round-trip test before Layer 4.

---

### R-03
## R-03 — Circular Import Cycle (graph ↔ cluster ↔ ner)
**Category:** Architecture  **Layer(s) affected:** graph, cluster, ner (Layer 2–5)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | The cycle is confirmed in the conversion plan document |
| Impact | 3 | Compiler rejects circular imports — build fails, not a runtime bug |
| Detectability | 1 | Immediately caught at `go build`; RPN = 5 × 3 × (6-1) = **75** |

**RPN: 75 🔴 Critical** *(highest detectability offset; build-time failure is still a blocker)*

**Detail:**  
The conversion plan correctly identifies the cycle: `graph/_context_graph.py` types are imported by `cluster/_cooccurrence.py` and `ner/_build.py`, which are both imported back through `graph/_entity_pipeline.py`. In Python, these are lazy function-body imports that the interpreter resolves at runtime. Go's compiler rejects circular package imports outright.

The proposed resolution (`graphtypes` leaf package) is correct but has not been implemented. The three `.go` prototype files that exist (`_markdown.go`, `markdown_extractor.go`, `go/markdown.go`) are all in the extractor layer and are not affected by this cycle, leaving the cycle unresolved for 10+ packages.

**Mitigation Strategies:**
1. **Implement Layer 2 first** (before any graph/cluster/ner work). Create `graphtypes/graphtypes.go` containing only `ContextEdge`, `ContextGraph`, and `ContextGraphStats` structs.
2. **Enforce package boundary in CI:** Add a `go vet ./...` + `go build ./...` step gated on every PR. A circular import surfaces at build time with a clear error.
3. **Dependency graph documentation:** Produce a Go module dependency graph (`go mod graph` + graphviz) as the canonical reference so reviewers can spot new cycles before they're introduced.

---

### R-04
## R-04 — DuckDB Go Driver Maturity & API Parity
**Category:** Infrastructure  **Layer(s) affected:** storage, community, search (Layers 4–7)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 4 | `go-duckdb` is the only community driver; not all DuckDB extensions are supported |
| Impact | 5 | Storage is the persistence backbone; failures corrupt or block the whole index |
| Detectability | 3 | Missing extension support fails loudly; subtle SQL semantics differ silently |

**RPN: 4 × 5 × (6-3) = 60 🔴 Critical**

**Detail:**  
The Python storage layer (`_store.py`, `_vector.py`, 1,055 + 775 LOC) relies on two DuckDB extensions: **VSS** (vector similarity search / HNSW index) and **FTS** (full-text search). The `go-duckdb` driver wraps the C++ DuckDB library, but:
- Extension loading (`INSTALL vss; LOAD vss`) behavior in embedded Go differs from the Python `duckdb` package — Python auto-installs extensions; Go requires manual bundling.
- The `FLOAT[1024]` array type used for embeddings requires the VSS extension to be loaded before table creation. If the extension load order differs, schema migrations silently fail.
- `ThreadLocalDuckDB` in Python uses a sophisticated `RLock`-based proxy (`_LockedConnection`, `_PendingResult`) that serializes execute+fetch cycles. The Go driver uses `database/sql`, which has different connection pool semantics.
- DuckDB's `fetchdf()` (DataFrame) method is Python-only. Go must use row-by-row scan, which changes performance characteristics for large result sets.

**Mitigation Strategies:**
1. **Validate extension support early:** Write a Layer 0 smoke test that creates an in-memory DuckDB via `go-duckdb`, loads VSS and FTS extensions, and inserts/queries a 1024-dim vector. Run this test in CI before any Layer 4 work.
2. **Pin `go-duckdb` version:** Lock to a specific commit/tag that has been validated against the same DuckDB C++ version used by the Python `duckdb` 1.5.3 package.
3. **Connection pool strategy:** Use `sql.DB` with `SetMaxOpenConns(1)` to replicate Python's single-connection-with-lock model until concurrent access patterns are profiled.
4. **Schema migration tests:** Port `tests/integration/test_storage.py` as Go integration tests and run against the same `.duckdb` files generated by Python to verify read-back compatibility.

---

### R-05
## R-05 — Extractor Scope Reduction (26 Dropped Formats)
**Category:** Functional Completeness  **Layer(s) affected:** extractors, loader (Layer 1)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | Explicitly planned — 26 of 31 Python extractors have no Go equivalent |
| Impact | 4 | PDF, DOCX, XLSX, PPTX, YAML, JSON, XML, FHIR, EDGAR and 17 others are dropped |
| Detectability | 2 | Missing formats surface immediately when users ingest those files |

**RPN: 5 × 4 × (6-2) = 80 🔴 Critical** *(highest in the register)*

**Detail:**  
The Go port retains only 5 of 31 extractors: plain text, CSV, Markdown, DB schema, and HTML. The 26 dropped extractors include:

| Priority | Dropped Extractor | Risk |
|---|---|---|
| 🔴 | PDF (`pypdf`) | Most enterprise documents are PDF |
| 🔴 | DOCX, XLSX, PPTX | Standard Office suite — common enterprise input |
| 🟠 | JSON, XML, YAML | Structured data formats widely used in APIs/configs |
| 🟠 | FHIR, ClinicalTrials, FDA Label | Domain-specific — customers in healthcare will be blocked |
| 🟡 | EDGAR, Python source, TypeScript, Java | Finance/code indexing use cases |
| 🟡 | MITRE ATT&CK, CVE, CWE, NIST | Security use cases |
| 🟡 | Email/MIME, Parquet, ODF | Misc ingestion paths |

The Go prototype files in `chonk/extractors/go/` include `pdf.go`, `docx.go`, `xlsx.go`, `json.go`, `xml.go`, and `yaml.go` — but these were generated as stubs and have **not been validated against the Python extractors' behavior**.

**Mitigation Strategies:**
1. **Prioritize PDF:** Use `ledongthuc/pdfcontent` or CGO-wrapped `pdfium` to implement PDF extraction in Go. PDF is the single highest-impact gap and should be treated as a blocker for enterprise deployment.
2. **DOCX/XLSX via pure-Go libraries:** `github.com/unidoc/unioffice` or `github.com/qax-os/excelize` provide DOCX/XLSX support without CGO.
3. **JSON/XML/YAML:** These are stdlib-supported in Go (`encoding/json`, `encoding/xml`, `gopkg.in/yaml.v3`). Implement before Go-live.
4. **Validate existing stubs:** The `chonk/extractors/go/pdf.go`, `docx.go`, etc. stubs must be tested against Python golden outputs before they are shipped. Run `test_extractors.py` fixture texts through both implementations and diff results.
5. **Feature flag / fallback:** Consider a sidecar Python process for unsupported formats during a transition period, routed via the `Transport` protocol.

---

### R-06
## R-06 — Go stdlib `csv.Dialect` / `csv.Sniffer` API Defect
**Category:** Code Correctness  **Layer(s) affected:** extractors/csv (Layer 1)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | `csv.Dialect` and `csv.Sniffer` do not exist in Go's `encoding/csv` |
| Impact | 3 | CSV extraction silently fails or panics at runtime |
| Detectability | 2 | Build error if types are referenced; panic if logic is reached at runtime |

**RPN: 5 × 3 × (6-2) = 60 🔴 Critical** *(build-blocking)*

**Detail:**  
The existing `chonk/extractors/go/csv.go` references `csv.Dialect` and `csv.Sniffer` at lines 137–153:

```go
func sniffDialect(r io.Reader) csv.Dialect { ... }   // line 137
sniffer := &csv.Sniffer{}                            // line 145
func parseCSV(r io.Reader, dialect *csv.Dialect) ... // line 153
```

Go's `encoding/csv` package has no `Dialect` struct and no `Sniffer` type. Python's `csv` module has both (`csv.Dialect`, `csv.Sniffer`). This is a direct Python→Go API mistranslation. The file will not compile. The same issue may exist in other prototype Go files that were generated from Python patterns.

**Mitigation Strategies:**
1. **Immediate fix:** Replace `sniffDialect` with a manual delimiter detection function (count occurrences of `,`, `\t`, `|`, `;` in the first 4,096 bytes; pick the most frequent). Remove `csv.Dialect` and `csv.Sniffer` references entirely.
2. **Code review sweep:** Audit all 14 existing Go prototype files for additional Python-stdlib API mistranslations (e.g., `io.CopyN` return semantics, `strings.Builder` vs `bytes.Buffer`, `regexp.MustCompile` placement).
3. **Compile gate:** Add `go build ./...` to CI immediately. This defect would have been caught at first build attempt.

---

### R-07
## R-07 — Python Mixin / Multiple Inheritance
**Category:** Architecture  **Layer(s) affected:** search (Layer 6)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | `EnhancedSearch(_GraphMixin, _ScoringMixin)` is the core search class |
| Impact | 3 | Go has no multiple inheritance; behavior must be restructured |
| Detectability | 2 | Refactoring bugs are subtle; test coverage catches most |

**RPN: 5 × 3 × (6-2) = 60 🔴 Critical** *(architecture restructure required)*

**Detail:**  
`EnhancedSearch` inherits from both `_GraphMixin` (699 LOC in `_enhanced_graph.py`) and `_ScoringMixin` (`_enhanced_scoring.py`). The mixins share state through `self._embed_fn`, `self._query_entity_id_fn`, and other instance attributes. Go has no multiple inheritance — all shared state must be exposed through embedded structs or injected interfaces.

The mixin pattern is also used for attribute sharing across Python's MRO — `_GraphMixin` methods reference `self._store`, `self._entity_index`, `self._cluster_map`, all defined only in `EnhancedSearch.__init__`. In Go, this implicit shared state must become explicit constructor parameters or embedded struct fields.

**Mitigation Strategies:**
1. **Composition over inheritance:** Define `graphSearch` and `scoringSearch` as embedded structs inside `EnhancedSearch`. Each carries only the state it needs.
2. **Interface extraction:** Define `GraphExpander` and `ScoringStrategy` interfaces. Inject implementations into `EnhancedSearch` at construction time — this also makes unit testing of each dimension independently easier.
3. **Port tests first:** `tests/unit/test_enhanced_search.py` (352 LOC) is the acceptance test. Port it to Go before rewriting the search layer to ensure the restructuring is behavior-preserving.

---

### R-08
## R-08 — Together.ai API — Single External Dependency for All ML
**Category:** Operational / Availability  **Layer(s) affected:** indexer, ner, community, graph, generation

| Axis | Score | Rationale |
|---|---|---|
| Probability | 3 | API outages and rate limits are routine for any external service |
| Impact | 5 | All embedding, NER, SVO, summarization, and answer generation stop |
| Detectability | 1 | HTTP 429/503 errors surface immediately in logs |

**RPN: 3 × 5 × (6-1) = 75 🔴 Critical**

**Detail:**  
The Python codebase externalizes LLM calls through an injectable `LLMClient` protocol and uses in-process `sentence-transformers` for embeddings. The Go target replaces **all** ML inference with Together.ai API calls. This creates a single point of failure:

- **Embedding:** Every index build and every search query requires a Together.ai embedding call. With no local fallback, a Together.ai outage blocks ingestion and search entirely.
- **NER:** NER inference moves from a deterministic spaCy model to a probabilistic LLM call. Each chunk processed during `build_ner` requires one API call; 100,000-chunk indexes require 100,000+ calls.
- **Rate limits:** Together.ai enforces per-minute token and request limits. Batch embedding (up to 512 texts) mitigates this for indexing, but a large index rebuild can hit rate limits mid-build, leaving the index in a partial state.
- **Latency:** A local embedding encode takes ~5ms; an API round-trip takes ~200–500ms. Index build time for large corpora will increase by 10–100×.
- **Cost:** API calls are metered. A corpus with 500,000 chunks at 256 tokens each ≈ 128M tokens. At typical API pricing this is non-trivial operational cost.

**Mitigation Strategies:**
1. **Embedder interface:** The Together.ai client must implement an `Embedder` interface so alternative backends (local ONNX model, OpenAI API, Ollama) can be swapped in. Do not hardcode `together.Client` at call sites.
2. **Retry + circuit breaker:** Implement exponential backoff (already planned in the architecture doc), plus a circuit breaker that fails fast after 3 consecutive 5xx errors to avoid partial index corruption.
3. **Batch size tuning:** Validate that 256-text batches stay under Together.ai's per-request token limit. For long documents, batch size may need to drop to 64.
4. **Cost estimation tool:** Before enterprise deployment, provide a `chonk estimate --config index.yaml` command that outputs projected API token consumption and cost.
5. **Offline embed cache:** Cache embeddings by content hash in the DuckDB store. Re-embedding identical content on index rebuild wastes API quota.

---

### R-09
## R-09 — Thread-Safety Model Divergence (ThreadLocalDuckDB)
**Category:** Concurrency  **Layer(s) affected:** storage, indexer, lifecycle

| Axis | Score | Rationale |
|---|---|---|
| Probability | 4 | Go goroutines vs Python threads have different preemption and scheduling |
| Impact | 4 | Data races produce corrupt indexes or panics under concurrent load |
| Detectability | 3 | Data races are intermittent; Go's `-race` detector catches most |

**RPN: 4 × 4 × (6-3) = 48 🟠 High**

**Detail:**  
Python's `ThreadLocalDuckDB` implements a custom single-connection-with-RLock model (`_LockedConnection`, `_PendingResult`). This is necessary because DuckDB's Python library does not support concurrent connections to the same file from multiple threads without explicit serialization.

The Go port must replicate this model using `database/sql` with `SetMaxOpenConns(1)` and Go's `sync.Mutex`. The risk areas:

- The Python `_PendingResult.__del__` finalizer releases the RLock if the caller drops the result without fetching. Go has no finalizers with guaranteed timing — a dropped `*sql.Rows` must be explicitly closed, or the connection is never returned to the pool.
- `NamespaceRefresher` in `lifecycle.py` spawns concurrent goroutines per namespace. In Python each namespace has its own DB file, but the `_registry` singleton and `_registry_lock` pattern must be faithfully reproduced in Go.
- The `VersionedRef[T]` generic (thread-safe versioned holder) must be re-implemented in Go using `sync.RWMutex` and generics (`[T any]`).

**Mitigation Strategies:**
1. **Always close `*sql.Rows`:** Use `defer rows.Close()` at every query site. Add a linter rule (via `go vet` or `staticcheck`) that flags unclosed rows.
2. **Run Go race detector in CI:** Add `go test -race ./...` to the integration test step.
3. **Port `test_versioning.py`:** The 300-LOC Python test for `VersionedRef` exercises concurrent stage/promote semantics. Port it directly as a Go table-driven test.

---

### R-10
## R-10 — Python Protocol / Structural Subtyping → Go Interfaces
**Category:** Type System  **Layer(s) affected:** All layers

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | 5 runtime-checkable Protocols defined; Python duck-typing used extensively |
| Impact | 2 | Go interfaces are explicit — incorrect implementation caught at compile time |
| Detectability | 1 | Compile-time failure; RPN = 5 × 2 × (6-1) = **50** |

**RPN: 50 🟠 High**

**Detail:**  
Python uses `@runtime_checkable` Protocols for `Transport`, `VectorBackend`, `LLMClient`, `Embedder`, and `Crawler`. Python's structural subtyping means any object with the right methods satisfies the protocol at runtime, even without explicit declaration. Go interfaces are also structural, but satisfaction is checked at compile time when assigning to an interface type.

The critical difference is the Python `VectorBackend` protocol exposes **private attributes** as protocol members:
```python
_conn: Any       # reached into by graph/search/ingest layers
_fts_dirty: bool # checked externally
```
In Go, unexported fields cannot be part of an interface. All code that reaches into `_conn` and `_fts_dirty` must be refactored to use exported methods.

**Mitigation Strategies:**
1. **Audit all `._conn` access:** Find every call site that reaches into `vector._conn` or `vector._fts_dirty` outside of the storage package and add proper accessor methods.
2. **Interface-first design:** Define all Go interfaces (`Transport`, `VectorBackend`, `Embedder`, `LLMClient`) in `Layer 0` before implementing any concrete types. Use `var _ Transport = (*LocalTransport)(nil)` compile-time assertions.

---

### R-11
## R-11 — Callback / Higher-Order Function Patterns
**Category:** API Design  **Layer(s) affected:** indexer, ingest, search, lifecycle

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | 52+ `Callable[...]` annotations; progress callbacks, filter functions, embed_fn |
| Impact | 2 | Go `func` types are first-class; translation is mechanical |
| Detectability | 2 | Type errors caught at compile time; logic bugs in closures need testing |

**RPN: 5 × 2 × (6-2) = 40 🟠 High**

**Detail:**  
Python uses `Callable[[str, int, int], None]` for progress callbacks, `Callable[[list[ScoredChunk]], list[ScoredChunk]]` for `chunk_filter`, `Callable[[Answer], Answer]` for `redaction_filter`, and `Callable[[list[str]], np.ndarray]` for `embed_fn`. Go equivalent function types are verbose but functionally identical.

The highest-risk pattern is the `redaction_filter` / `chunk_filter` pair in `EnhancedSearch` — these are optional user-supplied functions that mutate the search result pipeline. In Python, `None` means "no filter". In Go, `nil` func values panic if called — every injection point must guard with `if filter != nil`.

**Mitigation Strategies:**
1. **Define function type aliases:** `type ProgressCallback func(phase string, done, total int)` etc. for readability.
2. **nil-guard pattern:** Wrap all optional callbacks in a no-op default at construction: `if opts.OnProgress == nil { opts.OnProgress = func(string, int, int) {} }`.
3. **Port filter tests:** `test_enhanced_search.py` tests both `chunk_filter` and `redaction_filter` paths. These must be ported before the search layer ships.

---

### R-12
## R-12 — MCP Server Elimination / Rewrite
**Category:** Integration  **Layer(s) affected:** mcp_chonk_server.py (top-level service)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | The Python MCP server uses `asyncio` + `mcp` SDK; no Go MCP SDK equivalent planned |
| Impact | 3 | MCP is the primary interface for Claude Desktop and enterprise deployments |
| Detectability | 2 | Missing MCP endpoint is immediately noticed by integrators |

**RPN: 5 × 3 × (6-2) = 60 🔴 Critical** *(if MCP is a delivery requirement)*

**Detail:**  
`mcp_chonk_server.py` (615+ LOC) is the production delivery surface — it exposes search over MCP's stdio and HTTP transports, supports API key authentication, multi-DB configurations, and ASGI HTTP via Starlette/uvicorn. The future state architecture documents make no mention of an MCP server in Go.

The `mcp` Python SDK provides stdio and streamable HTTP session management. Go has no official MCP SDK. An MCP server in Go requires implementing the JSON-RPC 2.0 protocol over stdio or HTTP from scratch, or using a third-party library such as `github.com/mark3labs/mcp-go`.

**Mitigation Strategies:**
1. **Explicit scope decision:** Decide whether the MCP server is in or out of scope for the Go migration. If in scope, add it as Layer 8 and allocate time for JSON-RPC protocol implementation.
2. **If out of scope:** Keep `mcp_chonk_server.py` as a thin Python adapter that calls the Go binary's HTTP API. This is a valid hybrid pattern that avoids rewriting a stable integration layer.
3. **Go HTTP API first:** Implement a REST/gRPC API in the Go binary (`ingest` layer) before attempting MCP. MCP can wrap the REST API.

---

### R-13
## R-13 — Transport Surface Reduction (20 Python → 6 Go)
**Category:** Functional Completeness  **Layer(s) affected:** transports (Layer 0)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | Explicitly planned — 14 transports are dropped |
| Impact | 3 | Customers using Cassandra, Cosmos, DynamoDB, Firestore, MongoDB, Solr, Elasticsearch, Gmail, SharePoint, IMAP, FTP are blocked |
| Detectability | 2 | Missing transport is immediately apparent when a source is configured |

**RPN: 5 × 3 × (6-2) = 60 🔴 Critical** *(if enterprise transport coverage is required)*

**Detail:**  
Python implements 20 transports. The Go target keeps only: local filesystem, S3, SFTP, HTTP, directory crawler, and DB schema. Dropped:

| Dropped Transport | Enterprise Risk |
|---|---|
| Cassandra, MongoDB, DynamoDB, Cosmos, Firestore, Elasticsearch, Solr | NoSQL/document stores — common in enterprise data stacks |
| Gmail, IMAP | Email ingestion — used in compliance/HR use cases |
| SharePoint | Microsoft 365 integration — high enterprise demand |
| GitHub | Code indexing — used in devtools contexts |
| FTP | Legacy enterprise file shares |
| SQLAlchemy (SQL query transport) | General relational DB access |
| Web crawler, Import crawler | Competitive intelligence, code dependency mapping |

**Mitigation Strategies:**
1. **Prioritize by customer impact:** SharePoint, GitHub, and SQLAlchemy are likely highest-demand. Implement as Layer 0 extensions before Go/No-Go.
2. **Plugin architecture:** Define the `Transport` interface so community or enterprise contributors can add transports without modifying the core binary.
3. **Hybrid fallback:** Keep Python transport implementations callable as subprocess or sidecar during transition. The `Transport` interface boundary makes this clean.

---

### R-14
## R-14 — YAML Config Schema Backward-Compatibility
**Category:** Compatibility  **Layer(s) affected:** ingest (Layer 7)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 3 | Config schema is complex with nested dicts and optional fields |
| Impact | 3 | Broken config parsing silently ignores options or errors at startup |
| Detectability | 3 | Schema validation errors surface at startup; silent omissions are harder to detect |

**RPN: 3 × 3 × (6-3) = 27 🟡 Medium**

**Detail:**  
`ChonkConfig` (`_config.py`) uses `dict[str, object]` for `store`, `namespaces`, `search`, and `sources`. The `from_dict` method does lenient key-by-key extraction with defaults — Python's dynamic typing allows unknown keys to be silently ignored. Go's `gopkg.in/yaml.v3` strict unmarshaling raises errors for unknown fields unless `yaml:",inline"` or `omitempty` tags are used carefully.

The 80+ TOML/YAML config files in `work/configs/` represent the expected configuration surface. These must all parse correctly with the Go config loader.

**Mitigation Strategies:**
1. **Config schema test:** Load all 80+ existing `.toml`/`.yaml` configs through the Go config parser in a test and assert zero errors and correct field values for a representative sample.
2. **Use `yaml.Decoder` with `KnownFields(false)`:** Allow unknown fields to be silently skipped for forward compatibility — users with newer config keys shouldn't break older Go binaries.
3. **Config migration linter:** Provide a `chonk config validate` subcommand that parses a config and reports which fields are recognized vs unknown.

---

### R-15
## R-15 — Test Suite Coverage Gap (13,929 Python LOC → 0 Go Tests)
**Category:** Quality  **Layer(s) affected:** All layers

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | No Go tests exist today; all 45 test files are Python |
| Impact | 4 | Regressions from translation bugs go undetected until production |
| Detectability | 4 | Behavioral regressions are silent until a specific path is exercised |

**RPN: 5 × 4 × (6-4) = 40 🟠 High**

**Detail:**  
The Python test suite is extensive: 45 files, 13,929 LOC, with unit and integration tiers, fixture-driven extractor tests, and golden-output comparisons. The Go prototype has zero tests. The 14 existing Go files have not been validated.

Key test coverage that must be ported before each layer ships:

| Python Test | Go Priority | Layer |
|---|---|---|
| `test_chunking.py` (789 LOC) | Must port | Layer 1 |
| `test_context.py` | Must port | Layer 1 |
| `test_extractors.py` (661 LOC) | Must port | Layer 1 |
| `test_enhanced_search.py` (352 LOC) | Must port | Layer 6 |
| `test_storage.py` | Must port | Layer 4 |
| `test_ner.py`, `test_ner_pipeline.py` | Must port | Layer 5 |
| `test_community_summarizer.py` | Must port | Layer 4 |
| `test_context_graph.py` (476 LOC) | Must port | Layer 3 |

**Mitigation Strategies:**
1. **Test-first policy:** The conversion plan's layer-by-layer approach already gates promotion on passing Go tests. Enforce this strictly — no layer merges without ≥ 80% coverage.
2. **Golden file strategy:** For extractors and chunking, generate golden JSON outputs from Python and assert Go produces byte-identical (or near-identical within tolerance) results.
3. **Port high-value fixtures first:** `tests/unit/test_chunking.py` fixture texts are language-agnostic; they can be copied directly as Go `testdata/` files.

---

### R-16
## R-16 — Leiden/Louvain Community Detection — Pure-Go Reimplementation
**Category:** Algorithm Correctness  **Layer(s) affected:** cluster, community (Layer 4–5)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 4 | Pure-Go Louvain is planned but not implemented |
| Impact | 3 | Wrong community assignments degrade search quality silently |
| Detectability | 4 | Community quality is hard to measure without ground truth; errors are subtle |

**RPN: 4 × 3 × (6-4) = 24 🟡 Medium**

**Detail:**  
Python's `cluster/_clusterer.py` uses `igraph` + `leidenalg` for community detection. The architecture plan calls for a "pure Go Louvain implementation (~120 lines)". Louvain community detection is a non-trivial algorithm — a naive 120-line implementation will likely be correct for small graphs but may produce different partitions than the Python reference (Leiden and Louvain are not deterministic, but they should converge to similar modularity).

Python also supports `AgglomerativeClustering` and `DBSCAN` from scikit-learn. The architecture plan replaces all three with pure-Go implementations.

**Mitigation Strategies:**
1. **Validate against Python output:** For a fixed random seed and a small test graph (50 entities), verify that Go Louvain produces the same number of communities and similar modularity score as Python's leidenalg.
2. **Consider an existing Go graph library:** `github.com/yourbasic/graph` and `gonum/graph` both provide graph primitives. Building Louvain on top of `gonum/graph` reduces implementation risk vs from-scratch.
3. **Agglomerative fallback:** Port the agglomerative clustering from scikit-learn first (it's simpler and deterministic) and use it as the default algorithm in Go. Louvain/Leiden can be added as an optional enhancement.

---

### R-17
## R-17 — pgvector Backend Parity
**Category:** Infrastructure  **Layer(s) affected:** storage (Layer 4)

| Axis | Score | Rationale |
|---|---|---|
| Probability | 3 | `pgvector-go` exists and is maintained; parity issues are at the margin |
| Impact | 3 | PostgreSQL backend is the horizontal-scale deployment path |
| Detectability | 2 | SQL errors surface immediately; subtle score differences are harder to spot |

**RPN: 3 × 3 × (6-2) = 36 🟡 Medium**

**Detail:**  
Python's `_pg.py` (922 LOC) implements the full `VectorBackend` protocol against PostgreSQL + pgvector. The Go port uses `github.com/pgvector/pgvector-go` + `pgx/v5`. The risk is that:
- `PgVectorBackend.rebuild_fts_index()` (line 903) uses PostgreSQL's `tsvector` — the equivalent in Go using `pgx` requires raw SQL with `to_tsvector()` and `@@` operators.
- The Python `psycopg2-binary` vs Go `pgx/v5` driver have different transaction and connection pool semantics.
- The `sync_document()` hash comparison logic must produce identical SHA-256 digests in both languages.

**Mitigation Strategies:**
1. **Use `pgx/v5` with `pgvector-go`:** This is the correct choice; the combination is well-tested.
2. **Cross-language hash parity test:** Compute `hashlib.sha1(content)` in Python and the equivalent `sha1.Sum` in Go on the same input and assert identical hex output.
3. **Port `test_pg_vector_backend.py` (307 LOC)** as a Go integration test requiring a running PostgreSQL + pgvector instance.

---

### R-18
## R-18 — Dependency Health — Lock-File Dual State
**Category:** Dependency Management  **Layer(s) affected:** Python build only

| Axis | Score | Rationale |
|---|---|---|
| Probability | 4 | Two lock files (`uv.lock`, `poetry.lock`) will diverge |
| Impact | 2 | CI installs different versions depending on which tool is used |
| Detectability | 2 | Silent version drift; only manifests on `poetry update` runs |

**RPN: 4 × 2 × (6-2) = 32 🟡 Medium**

**Detail:**  
Both `uv.lock` and `poetry.lock` exist at the repo root (per the dependency health report). The Python codebase must remain stable during the Go migration period. If `poetry.lock` drifts from `uv.lock`, CI using Poetry will install different package versions than developers using uv, creating "works on my machine" build failures.

**Mitigation Strategies:**
1. **Delete `poetry.lock`:** Designate `uv.lock` as canonical. Add a CI check that errors if `poetry.lock` is present and newer than `uv.lock`.
2. **Add `pip-audit` to `uv.lock`:** The health report identifies `pip-audit` as declared in `pyproject.toml` but absent from `uv.lock`. Fix immediately with `uv lock`.

---

### R-19
## R-19 — `dict[str, Any]` Ubiquity → Typed Go Structs
**Category:** Type System  **Layer(s) affected:** ingest, config, transports

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | 297+ `Any` usages; `dict[str, Any]` used for source_config, source_detail, search defaults |
| Impact | 2 | Missing struct fields silently get zero values in Go; no KeyError equivalent |
| Detectability | 3 | Some fields are accessed via map lookup which panics on nil; others silently default |

**RPN: 5 × 2 × (6-3) = 30 🟡 Medium**

**Detail:**  
Python's loose `dict[str, Any]` pattern for `source_config` (in `indexer.py` `_crawl()`) passes ~10 optional keys without schema validation. In Go, this must become a `SourceConfig` struct. The risk is that optional keys in Python that are never explicitly set will become zero-value fields in Go, which may change behavior (e.g., `max_files: 0` vs Python's `get("max_files", 1000)` default).

`DocumentChunk.source_detail: dict[str, Any] | None` is the most critical — it carries per-extractor metadata (CSV row ranges, PDF page numbers) and is queried dynamically across the codebase.

**Mitigation Strategies:**
1. **Define `SourceDetail` as a tagged union:** Use a Go struct with all possible extractor-specific fields as optional pointers, or use `map[string]interface{}` with a documented key registry.
2. **Default value audit:** For every `source_config.get("key", default)` in Python, ensure the Go struct field has the identical default value.
3. **Schema validation at config load time:** Add struct-level validation (e.g., `MaxFiles > 0`) to catch zero-value surprises early.

---

### R-20
## R-20 — Error Handling Paradigm Shift (Exceptions → Error Values)
**Category:** Code Style  **Layer(s) affected:** All layers

| Axis | Score | Rationale |
|---|---|---|
| Probability | 5 | Python uses exceptions pervasively; Go uses `(result, error)` returns |
| Impact | 2 | Unchecked errors in Go cause silent failures or nil dereferences |
| Detectability | 3 | Go's `errcheck` linter catches unchecked error returns |

**RPN: 5 × 2 × (6-3) = 30 🟡 Medium**

**Detail:**  
Python's exception-based model means errors propagate automatically up the call stack. Go requires every error to be explicitly checked. The risk is that developers porting Python code will unconsciously write patterns like:

```go
result, _ := someFunc()  // discards error
```

This is especially dangerous in the storage layer where a failed `execute()` followed by ignored error and continued operation will corrupt DuckDB state.

The Python code uses structured try/except at phase boundaries (crawl, embed, store) with `on_error` callbacks. This graceful degradation pattern must be faithfully reproduced in Go.

**Mitigation Strategies:**
1. **Enable `errcheck` linter:** Add `staticcheck` or `golangci-lint` with `errcheck` enabled to CI. Fail builds that discard error values without explicit `_ =` justification.
2. **Error wrapping convention:** Use `fmt.Errorf("phase %s: %w", phase, err)` for all error wrapping to preserve stack context.
3. **No `panic` in library code:** Reserve `panic` only for programming errors (nil pointer, impossible state). All user-facing errors must be returned values.

---

## 4. Aggregate Risk Heatmap

### RPN Summary Table

| ID | Risk Title | Prob | Impact | Detect | RPN | Band |
|---|---|---|---|---|---|---|
| R-05 | Extractor scope reduction (26 dropped) | 5 | 4 | 4 | **80** | 🔴 Critical |
| R-08 | Together.ai single dependency | 3 | 5 | 5 | **75** | 🔴 Critical |
| R-03 | Circular import cycle (graph/cluster/ner) | 5 | 3 | 5 | **75** | 🔴 Critical |
| R-04 | DuckDB Go driver maturity | 4 | 5 | 3 | **60** | 🔴 Critical |
| R-06 | csv.Dialect / csv.Sniffer defect | 5 | 3 | 4 | **60** | 🔴 Critical |
| R-07 | Mixin / multiple inheritance | 5 | 3 | 4 | **60** | 🔴 Critical |
| R-12 | MCP server elimination | 5 | 3 | 4 | **60** | 🔴 Critical |
| R-13 | Transport surface reduction | 5 | 3 | 4 | **60** | 🔴 Critical |
| R-01 | ML runtime elimination | 5 | 5 | 2 | **50** | 🟠 High |
| R-10 | Protocol → Go interface translation | 5 | 2 | 5 | **50** | 🟠 High |
| R-09 | Thread-safety model divergence | 4 | 4 | 3 | **48** | 🟠 High |
| R-02 | NumPy array semantics | 5 | 4 | 2 | **40** | 🟠 High |
| R-11 | Higher-order function / callback patterns | 5 | 2 | 4 | **40** | 🟠 High |
| R-15 | Test suite coverage gap | 5 | 4 | 2 | **40** | 🟠 High |
| R-17 | pgvector backend parity | 3 | 3 | 4 | **36** | 🟡 Medium |
| R-18 | Lock-file dual state | 4 | 2 | 4 | **32** | 🟡 Medium |
| R-16 | Leiden/Louvain pure-Go reimplementation | 4 | 3 | 2 | **24** | 🟡 Medium |
| R-19 | `dict[str,Any]` → typed Go structs | 5 | 2 | 3 | **30** | 🟡 Medium |
| R-20 | Error handling paradigm shift | 5 | 2 | 3 | **30** | 🟡 Medium |
| R-14 | YAML config backward-compatibility | 3 | 3 | 3 | **27** | 🟡 Medium |

### Risk Distribution by Layer

```
Layer 0 (models, schema, transports)     →  R-13, R-18, R-19         [HIGH]
Layer 1 (chunking, context, extractors)  →  R-05, R-06, R-11, R-20   [CRITICAL]
Layer 2 (graphtypes — cycle break)       →  R-03                      [CRITICAL]
Layer 3 (graph, loader)                  →  R-07, R-10                [CRITICAL/HIGH]
Layer 4 (storage, community)             →  R-04, R-09, R-16, R-17   [CRITICAL/HIGH]
Layer 5 (cluster, ner, indexer)          →  R-01, R-02, R-08, R-15   [CRITICAL/HIGH]
Layer 6 (search)                         →  R-07, R-11, R-15          [HIGH]
Layer 7 (ingest)                         →  R-14                      [MEDIUM]
Cross-cutting                            →  R-12, R-20                [CRITICAL/MEDIUM]
```

---

## 5. Go/No-Go Criteria

### Absolute Blockers (must resolve before starting Layer 3)

| Criterion | Owner | Current State |
|---|---|---|
| **B-01** R-06: `csv.Dialect`/`csv.Sniffer` compile error fixed | Go lead | ❌ Not fixed |
| **B-02** R-03: `graphtypes` package created and tested | Go lead | ❌ Not started |
| **B-03** R-04: DuckDB Go driver smoke test passes (VSS + FTS extensions load) | Infra | ❌ Not validated |
| **B-04** R-01: Together.ai embedding model evaluated against fang2026 eval benchmark | ML lead | ❌ Not started |
| **B-05** R-08: `Embedder` interface defined; Together.ai client implements it | Go lead | ❌ Not started |

### Strong Prerequisites (must complete before Layer 5)

| Criterion | Owner | Current State |
|---|---|---|
| **P-01** R-05: PDF extractor implemented or explicitly scoped out with customer sign-off | Product | ❌ Stubs only |
| **P-02** R-07: `EnhancedSearch` mixin decomposition design reviewed and approved | Arch | ❌ Not started |
| **P-03** R-15: Layer 1 Go test coverage ≥ 80% on chunking, context, extractors | QA | ❌ Not started |
| **P-04** R-09: Go race detector passes on storage layer integration test | QA | ❌ Not started |
| **P-05** R-12: MCP server scope decision documented (in scope / hybrid / out of scope) | Product | ❌ Not decided |

### Go Criteria (all blockers and prerequisites met)

1. `go build ./...` passes with zero errors on all 8 layers.
2. `go test -race ./...` passes with zero race conditions.
3. `go test -tags integration ./...` passes the four integration checkpoints defined in `unit-testing.md`.
4. End-to-end search quality: fang2026 benchmark score within 5% of Python baseline using the Go binary + Together.ai embeddings.
5. Performance: index build time for a 10,000-chunk corpus completes within 2× the Python baseline (API latency overhead is expected).
6. All existing `work/configs/` YAML configurations parse without error.

### No-Go Criteria (any one is sufficient to halt)

1. DuckDB VSS extension fails to load via `go-duckdb` on the target deployment OS/architecture.
2. Together.ai embedding model produces fang2026 scores < 10% below Python baseline after model tuning.
3. Leiden/Louvain Go implementation produces zero-community or single-community output on test graphs with > 50 entities.
4. R-06 (csv compile error) or R-03 (import cycle) remain unresolved at Layer 2 entry.
5. MCP server is confirmed in scope but no Go MCP library or implementation plan exists.

---

## 6. Mitigation Roadmap

### Phase 0 — Pre-Migration Hardening (Before any Go code merges)

| Action | Resolves | Effort | Priority |
|---|---|---|---|
| Fix `csv.Dialect`/`csv.Sniffer` compile error in `go/csv.go` | R-06 | 1 day | 🔴 |
| Add `go build ./...` to CI | R-06, R-03 | 0.5 days | 🔴 |
| Designate `uv.lock` as canonical; delete `poetry.lock` | R-18 | 0.5 days | 🟠 |
| Fix `pip-audit` missing from `uv.lock` | R-18 | 0.5 days | 🟠 |
| Sweep all 14 Go prototype files for Python→Go API mistranslations | R-06, general | 2 days | 🔴 |
| Define all Go interfaces (`Transport`, `VectorBackend`, `Embedder`, `LLMClient`) | R-10 | 2 days | 🔴 |
| Decide MCP server scope; document decision | R-12 | 0.5 days | 🔴 |

### Phase 1 — Layer 0–1 Foundation

| Action | Resolves | Effort | Priority |
|---|---|---|---|
| Create `graphtypes` package (Layer 2 pre-work) | R-03 | 1 day | 🔴 |
| DuckDB smoke test: VSS + FTS load via `go-duckdb` | R-04 | 2 days | 🔴 |
| Evaluate Together.ai embedding model on fang2026 | R-01 | 3 days | 🔴 |
| Port `test_chunking.py` and `test_extractors.py` to Go | R-15 | 5 days | 🟠 |
| Implement PDF extractor (or scope it out with sign-off) | R-05 | 5–10 days | 🔴 |
| Implement JSON, XML, YAML extractors (stdlib) | R-05 | 3 days | 🟠 |

### Phase 2 — Layer 3–4 Storage & Graph

| Action | Resolves | Effort | Priority |
|---|---|---|---|
| Decompose `EnhancedSearch` mixin into Go embedded structs | R-07 | 4 days | 🔴 |
| Port storage integration tests; run against Python-generated `.duckdb` | R-04, R-17 | 3 days | 🟠 |
| Implement `sync.Mutex`-safe connection wrapper; add `-race` to CI | R-09 | 2 days | 🟠 |
| Implement agglomerative clustering in pure Go | R-16 | 3 days | 🟡 |
| Add null-guard for all optional `func` fields | R-11 | 1 day | 🟠 |

### Phase 3 — Layer 5–7 ML & Orchestration

| Action | Resolves | Effort | Priority |
|---|---|---|---|
| Implement `Embedder` interface + Together.ai client | R-08 | 3 days | 🔴 |
| Add circuit breaker + retry logic to Together.ai client | R-08 | 2 days | 🔴 |
| Port NER parity test: Go vs Python entity recall on test fixtures | R-01 | 2 days | 🟠 |
| Port `test_enhanced_search.py` to Go | R-07, R-15 | 4 days | 🟠 |
| Implement Louvain community detection; validate against Python output | R-16 | 5 days | 🟡 |
| Config schema test: load all 80+ TOML/YAML configs | R-14 | 1 day | 🟡 |
| Add `errcheck` linter to CI | R-20 | 0.5 days | 🟡 |

### Phase 4 — Integration & Go/No-Go Gate

| Action | Resolves | Effort | Priority |
|---|---|---|---|
| End-to-end fang2026 benchmark run on Go binary | R-01 | 2 days | 🔴 |
| `go test -race ./...` clean run | R-09 | 1 day | 🟠 |
| SharePoint and GitHub transport implementation | R-13 | 5 days | 🟠 |
| MCP server: Go implementation or Python hybrid adapter | R-12 | 5–10 days | 🔴 |
| Performance benchmark: 10k chunk index vs Python baseline | cross | 1 day | 🟠 |

---

*End of Risk Assessment*
