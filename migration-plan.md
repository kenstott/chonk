# Chonk — Python → Go Migration Plan

## 0. DISCOVERY

### Source Language & Runtime
- **Language:** Python 3.11–3.14 (pyproject.toml `requires-python = ">=3.11,<3.15"`)
- **Build system:** Hatchling
- **Test framework:** pytest 9+ with pytest-cov, pytest-timeout, pytest-asyncio
- **Coverage target:** 70% minimum (enforced via `fail_under = 70`)
- **Linting/formatting:** ruff, black, pyright (standard mode)

### Codebase Size (chonk/ package only, excluding training & generate-training)
| Sub-package | LOC |
|---|---|
| transports | 5,050 |
| extractors | 4,757 |
| storage | 3,372 |
| ingest + indexer + lifecycle + loader + workers | 3,091 |
| ner | 2,272 |
| search | 1,871 |
| graph | 1,375 |
| community | 1,158 |
| chunking | 904 |
| cluster | 478 |
| generation | 168 |
| models + schema + context + struct_inference | 436 |
| **TOTAL** | **~24,932** |

Tests: ~27,810 LOC across unit/ and integration/ suites.

### Key Frameworks & Libraries
| Library | Role |
|---|---|
| sentence-transformers + torch | Text embedding (in-process ML) |
| spacy + thinc | Named entity recognition |
| scikit-learn | Agglomerative clustering, DBSCAN |
| igraph + leidenalg | Graph construction, Leiden community detection |
| duckdb + duckdb-engine | Vector + FTS storage backend |
| sqlalchemy | Relational ORM layer |
| psycopg2 + pgvector | PostgreSQL vector backend |
| pandas | CSV processing |
| pypdf, python-docx, openpyxl, python-pptx, odfpy | Document extraction |
| pyyaml, pyarrow | YAML and Parquet extraction |
| boto3, paramiko, requests | S3, SFTP, HTTP transports |
| google-api-python-client, office365-rest-python-client | Gmail, SharePoint transports |

### Knowledge Base — Future State ADRs (Already Committed)
The `aipa_test_mcp_server/future_state_architecture/` ADR set fully defines the Go target:
- **Runtime:** Go 1.22+, single binary, no Python interpreter
- **ML replacement:** All in-process ML (torch, spacy, sklearn, igraph, leidenalg) → Together.ai REST API
- **Extraction scope reduced:** Only 5 formats carried forward (text, CSV, Markdown, SQL/DB schema, HTML)
- **Storage:** DuckDB via go-duckdb + optional pgvector via pgx/v5
- **Circular dependency fix:** New `graphtypes` leaf package breaks graph→cluster→ner→graph cycle
- **Layer build order:** 8 layers (0–7) defined with explicit dependency DAG

---

## 1. INVENTORY

| Component | LOC | Language | External Deps | Internal Deps | Domain |
|---|---|---|---|---|---|
| `models` | 128 | Python | none | none | Core |
| `schema` | 50 | Python | none | none | Core |
| `context` | 83 | Python | none | models | Core |
| `_struct_inference` | 175 | Python | none | schema | Core |
| `_versioning` | 108 | Python | none | models | Core |
| `chunking` | 904 | Python | none | models, schema | Chunking |
| `extractors` (5 kept) | ~450 | Python | requests, pandas, lxml | models, chunking | Extraction |
| `extractors` (dropped) | ~4,307 | Python | pypdf, python-docx, openpyxl, pptx, odfpy, pyarrow, etc. | models, chunking | Extraction |
| `transports` (core 7) | ~800 | Python | boto3, paramiko, requests, duckdb | models, schema | Transport |
| `transports` (out-of-scope) | ~4,250 | Python | google-api, office365, pymongo, elasticsearch-py, etc. | models | Transport |
| `storage` | 3,372 | Python | duckdb, sqlalchemy, psycopg2, pgvector, numpy | models, schema | Storage |
| `generation` | 168 | Python | (LLM client) | models | Generation |
| `graph` | 1,375 | Python | (LLM client) | models, ner, cluster | Graph |
| `ner` | 2,272 | Python | spacy, inflect | models, graph | NER |
| `cluster` | 478 | Python | scikit-learn, numpy | models, graph, ner | Clustering |
| `community` | 1,158 | Python | igraph, leidenalg, (LLM) | models, graph, cluster | Community |
| `search` | 1,871 | Python | duckdb, numpy | models, storage, ner, cluster, community, generation | Search |
| `indexer` | 284 | Python | sentence-transformers, torch | models, storage, extractors, transports | Indexing |
| `ingest` + `_ingest_worker` | 1,388 | Python | pyyaml, sentence-transformers | all packages | Orchestration |
| `loader` | 748 | Python | none | extractors, transports, chunking, context, struct_inference | Loading |
| `lifecycle` | 260 | Python | none | ner, community, storage | Lifecycle |

**Discrepancies vs. ADR domain map:** None. The code sub-package boundaries exactly mirror the ADR's Go module layout. The `graphtypes` package does not yet exist in Python — it is a new package to be created in Go Layer 2 to break the import cycle.

---

## 2. FRAMEWORK CONTEXT

| Python Framework | Role | Go Equivalent | Complexity | Notes |
|---|---|---|---|---|
| sentence-transformers + torch | Text embedding | Together.ai Embeddings API | REDESIGN | ADR committed: remote API replaces in-process model |
| spacy + thinc | NER tagging | Together.ai Chat API (structured JSON) | REDESIGN | ADR committed: prompt-based NER replaces spacy pipeline |
| scikit-learn (AgglomerativeClustering, DBSCAN) | Clustering | Pure Go (~100 LOC) | REWORK | ADR committed: implement in Go, no library needed |
| igraph + leidenalg | Graph + Leiden community detection | Pure Go Louvain (~120 LOC) | REWORK | ADR committed: port algorithm directly |
| duckdb + duckdb-engine | Vector + FTS storage | go-duckdb | DROP-IN | Same DuckDB engine, Go driver |
| sqlalchemy | Relational ORM | database/sql (stdlib) | REWORK | Go uses stdlib, no ORM abstraction layer |
| psycopg2 + pgvector | PostgreSQL vector | pgx/v5 + pgvector-go | DROP-IN | Near 1:1 driver swap |
| pandas | CSV parsing | encoding/csv (stdlib) | DROP-IN | Standard library covers CSV use |
| pypdf, python-docx, openpyxl, python-pptx, odfpy | Document extraction | **DROPPED** | N/A | ADR: these formats not carried forward |
| pyyaml | YAML config parsing | gopkg.in/yaml.v3 | DROP-IN | Config schema unchanged |
| pyarrow / parquet | Parquet extraction | **DROPPED** | N/A | ADR: Parquet not in extraction scope |
| boto3 | S3 transport | aws-sdk-go-v2 | DROP-IN | Same AWS APIs |
| paramiko | SFTP transport | golang.org/x/crypto/ssh | DROP-IN | Same SSH/SFTP protocol |
| requests / certifi | HTTP transport | net/http (stdlib) | DROP-IN | Standard library |
| BeautifulSoup (via html.py) | HTML parsing | golang.org/x/net/html | DROP-IN | Same parse-tree model |
| google-api-python-client | Gmail transport | **OUT OF SCOPE** | N/A | ADR: source-specific connectors deferred |
| office365-rest-python-client | SharePoint transport | **OUT OF SCOPE** | N/A | ADR: source-specific connectors deferred |
| pymongo, elasticsearch-py, cassandra-driver, etc. | NoSQL transports | **OUT OF SCOPE** | N/A | ADR: source-specific connectors deferred |
| OpenAI-compatible LLM client (generation, graph) | Answer generation, SVO extraction | Together.ai Chat API | REWORK | ADR: all LLM calls → Together.ai |
| pytest | Test framework | testing (stdlib) + testify/assert | REWORK | Go test conventions differ; templates in ADR unit-testing.md |

**REDESIGN items (two) are both already resolved by ADR:** Together.ai replaces both sentence-transformers and spacy. No unresolved REDESIGN gaps exist.

---

## 3. DEPENDENCY ANALYSIS

### Dependency Graph (leaf → root order)

```
Layer 0 (leaves):  models  schema  transports(core)
Layer 1:           chunking  context  extractors(5)  generation  _struct_inference
Layer 2:           graphtypes  [NEW — cycle break]
Layer 3:           graph  loader
Layer 4:           storage  community
Layer 5:           cluster  ner  indexer  lifecycle
Layer 6:           search
Layer 7:           ingest  [root]
```

### Cycle: graph → cluster → ner → graph
Python permits this via lazy function-body imports. Go rejects circular package imports at compile time.
**ADR resolution (Layer 2):** Extract `graphtypes` package containing only `ContextEdge`, `ContextGraph`, `ContextGraphStats` structs. All three cycle participants import `graphtypes` instead of each other.

### Shared Utilities / Cross-Cutting Concerns
| Concern | Blocks | Go approach |
|---|---|---|
| `together` HTTP client | ner, graph, community, indexer, generation | Single `together` package, injected via interface |
| DuckDB connection pool | storage (all backends) | `sync.RWMutex` single-writer guard (mirrors Python's `ThreadLocalDuckDB`) |
| `DocumentChunk` struct | all packages | `models` package — already a leaf, no deps |

### Out-of-Scope Components (not migrated)
- **Dropped extractors** (PDF, DOCX, XLSX, PPTX, YAML, ODF, JSON, XML, Parquet, Python AST, TS, Java, FHIR, EDGAR, email, MIME, ClinicalTrials, FDA, ATT&CK, CVE, CWE, NIST, NoSQL, lookup_table, renderer): 21 files, ~4,307 LOC — ADR explicit.
- **Out-of-scope transports** (GitHub, Gmail, SharePoint, Cassandra, MongoDB, Elasticsearch, Solr, DynamoDB, CosmosDB, Firestore, IMAP, FTP, SQLAlchemy, import_crawler): ~4,250 LOC — ADR explicit.
- **`_versioning.py`**: Python-specific feature namespace lifecycle — Go equivalent deferred.

### Suggested Sprint Groupings
Groups follow ADR layers; each layer is one sprint unit.

| Group | Packages | Relative Size |
|---|---|---|
| G0 | models, schema, transports(core) | Small |
| G1 | chunking, context, extractors(5), generation, structinfer | Medium |
| G2 | graphtypes (new) | Small |
| G3 | graph, loader | Large |
| G4 | storage, community | Large |
| G5 | cluster, ner, indexer, lifecycle | Large |
| G6 | search | Large |
| G7 | ingest | Medium |

---

## 4. INTEROP BOUNDARY EVALUATION

The ADR set does not define any interop boundaries between the Python and Go systems. The architecture decision is a **full rewrite** with the Go binary replacing Python entirely. No REST/gRPC shim layer is described or implied.

| Grouping | Recommendation | Rationale |
|---|---|---|
| All G0–G7 packages | **INTEROP NOT RECOMMENDED** | The ADR mandates a single Go binary. No incremental dual-runtime operation is planned. The codebase has no external API surface (no HTTP server, no message queue consumer) — it is a library/CLI pipeline. Adding an interop boundary would introduce distributed system complexity with zero benefit: there are no external consumers to keep compatible during migration, and the Python and Go configs are file-compatible (same YAML schema). |
| Together.ai API calls | **ALREADY DECIDED** — external REST | ADR ml-services.md: all ML inference goes to Together.ai REST. This is the only external boundary, and it is already defined. Not an interop boundary between old and new systems — it is the permanent Go architecture. |

**Conclusion:** Migrate atomically layer by layer within the Go binary. No Python↔Go interop boundary is warranted.

---

## 5. BUSINESS VALUE SCORING (WSJF)

Scores reflect the Go rewrite delivering a production-deployable single binary: lower memory, no Python runtime, no model download, no venv.

| Component / Group | Business Value | Time Criticality | Risk Reduction | Effort (1=low) | WSJF | Flag |
|---|---|---|---|---|---|---|
| G0: models, schema, transports | 6 | 7 | 5 | 2 | **9.0** | FAIL FAST — must validate Go struct parity immediately |
| G1: chunking, context, extractors(5), generation, structinfer | 8 | 7 | 6 | 4 | **5.25** | Core chunking pipeline; high user-facing value |
| G2: graphtypes (cycle break) | 3 | 9 | 8 | 1 | **20.0** | FAIL FAST — unblocks all of G3–G5; near-zero effort |
| G3: graph, loader | 7 | 6 | 5 | 6 | **3.0** | LLM integration; graph triple extraction |
| G4: storage, community | 9 | 7 | 8 | 7 | **3.43** | DuckDB vector store is the production foundation |
| G5: cluster, ner, indexer, lifecycle | 8 | 6 | 7 | 8 | **2.63** | NER + clustering unlock semantic search quality |
| G6: search | 10 | 8 | 7 | 6 | **4.17** | Direct user-facing capability; 4-lane search |
| G7: ingest | 9 | 8 | 8 | 5 | **5.0** | End-to-end orchestration; delivers the binary |
| Dropped extractors (21 formats) | 2 | 1 | 1 | 1 | **4.0** | DEFER — ADR explicitly drops; migration excluded |
| Out-of-scope transports (14) | 2 | 1 | 1 | 1 | **4.0** | DEFER — ADR explicitly excludes; migration excluded |

**Priority order by WSJF:** G2 → G0 → G1 → G7 → G6 → G4 → G3 → G5

> Note: The ADR-mandated layer dependency ordering (G0 → G1 → G2 → G3 → G4 → G5 → G6 → G7) constrains WSJF ordering — you cannot execute G2 before G0 passes tests. The plan below reconciles WSJF scores with layer constraints.

---

## 6. TESTING STRATEGY

The ADR `unit-testing.md` fully defines Go test patterns per layer. The strategy below builds on it rather than replacing it.

### Dual-Run Validation Requirement
Every component must satisfy both:
1. **Python tests pass** against the existing Python source (baseline preserved)
2. **Go tests pass** against the Go output before the component is marked done

### Test Framework
- **Go unit tests:** `testing` stdlib + `github.com/stretchr/testify/assert` (optional)
- **Build tags:** `//go:build integration` guards network-dependent tests
- **Run unit (every commit):** `go test ./...`
- **Run integration (merge/nightly):** `go test -tags integration ./...`

### Per-Layer Testing Approach

| Layer | Unit | Functional Equivalence | Integration | Performance | Notes |
|---|---|---|---|---|---|
| G0: models, schema, transports | Struct field round-trip; interface compile-time assertions | Compare Python dataclass defaults vs Go zero values | LocalTransport list/fetch with temp dirs | N/A — no I/O path | Template: `test_context.py`, `test_transports.py` |
| G1: chunking, context, extractors, generation | 1:1 port of all Python unit cases (see unit-testing.md Layer 1) | Feed same fixtures to Python and Go; compare chunk count, section_path, embedding_content | Checkpoint 1: extract + chunk all 5 formats | Chunking throughput ≥ Python baseline | **REDESIGN flag:** generation LLM calls require manual equivalence review |
| G2: graphtypes | Struct construction, zero-value safety | N/A (pure structs) | N/A | N/A | Verify no deps beyond models via `go list -deps` |
| G3: graph, loader | SVOExtractor with stub client; loader with all 5 formats | Compare SVO triples on same text fixture | Checkpoint 2: DocumentLoader mixed formats | N/A | **REDESIGN flag:** LLM-backed SVO requires manual review of output quality |
| G4: storage, community | In-memory DuckDB `:memory:`; stub embedder | AddDocument→Search round-trip; count consistency | Checkpoint 3 (partial): index a directory | Storage write throughput; FTS query latency ≥ Python DuckDB baseline | Thread safety: 10-goroutine concurrent write test |
| G5: cluster, ner, indexer, lifecycle | Stub Together.ai client; stub embedder returning ones(n,1024) | NER entity labels on same text fixture; cluster membership on same embedding matrix | Checkpoint 3: full index with stub embedder | Indexer batch throughput ≥ Python baseline | **REDESIGN flag:** NER via LLM — manual review of entity label quality required |
| G6: search | In-memory store; synthetic float32 embeddings (dim 8) | 4-lane search results on same store/embeddings | Search with real DuckDB on indexed fixture corpus | Search latency ≤ Python baseline at k=10, k=30, k=50 | All 4 lanes independently and combined |
| G7: ingest | Stub embedder via BuildOptions; fixture YAML | End-to-end: same config → same chunk count | Checkpoint 4: full pipeline with real TOGETHER_API_KEY or go-vcr cassette | End-to-end pipeline throughput | Config YAML must be backward-compatible with Python schema |

### Regression Thresholds
- Chunking: chunk count within ±5% of Python output on same corpus
- Storage: `AddDocument` throughput within 20% of Python/DuckDB baseline
- Search: latency at k=30 ≤ Python latency (Go should be faster; flag regression if not)
- NER/SVO: entity recall within ±10% on held-out fixture set (requires manual baseline run)

---

## 7. MIGRATION PLAN

Priority is WSJF-within-layer-constraints. ADR layer order is mandatory; within a layer, highest WSJF goes first.

---

### Wave 1 — Foundation: Pure Types & Core Transports
**ADR Layer 0**

- **Components:** `models`, `schema`, `transports` (local, S3, SFTP, HTTP, directory, DB schema, web crawler)
- **Framework migrations:**
  - Python dataclass → Go struct (DROP-IN)
  - boto3 → aws-sdk-go-v2 (DROP-IN)
  - paramiko → golang.org/x/crypto/ssh (DROP-IN)
  - requests → net/http stdlib (DROP-IN)
- **Interop boundary:** INTEROP NOT RECOMMENDED — internal library layer
- **ADR references:** conversion-plan.md Layer 0; code-conversion.md §models, §transports
- **Testing approach:** Struct field round-trip unit tests; compile-time interface assertions (`var _ Transport = &LocalTransport{}`); LocalTransport list/fetch with `t.TempDir()`
- **Sprint estimate:** 1 sprint
- **Expected business value:** Unblocks all subsequent layers; validates Go module setup and build pipeline; no user-visible output yet
- **Risk:** LOW — pure value types and well-understood protocols; no ML, no storage

---

### Wave 2 — Stateless Pipeline: Chunking, Extraction, Context, Generation
**ADR Layer 1**

- **Components:** `chunking`, `context`, `extractors` (text, CSV, Markdown, DB schema, HTML only), `generation`, `_struct_inference`
- **Framework migrations:**
  - Python regex → Go `regexp` compiled at `init()` (DROP-IN)
  - pandas CSV → encoding/csv stdlib (DROP-IN)
  - BeautifulSoup HTML → golang.org/x/net/html (DROP-IN)
  - OpenAI-compatible LLM client → Together.ai Chat API behind injectable interface (REWORK)
  - 21 dropped extractors — no Go equivalent (OUT OF SCOPE per ADR)
- **Interop boundary:** INTEROP NOT RECOMMENDED
- **ADR references:** conversion-plan.md Layer 1; extraction-scope.md; ml-services.md §Answer Generation
- **Testing approach:** 1:1 port of `test_chunking.py`, `test_context.py`, `test_extractors.py`, `test_generation.py`; Checkpoint 1 integration test (extract + chunk all 5 formats); `AnswerGenerator` unit tests use `stubTogetherClient`; **REDESIGN flag on generation** — manual review of LLM output quality required
- **Sprint estimate:** 2 sprints
- **Expected business value:** Core chunking pipeline operational; all supported formats can be extracted and chunked; validates Go output matches Python chunk counts and section paths
- **Risk:** LOW-MEDIUM — pure string/regexp logic is low risk; generation stub pattern is well-defined in ADR

---

### Wave 3 — Cycle Break: graphtypes
**ADR Layer 2**

- **Components:** `graphtypes` (new Go package — does not exist in Python)
- **Framework migrations:** N/A — pure struct definitions
- **Interop boundary:** INTEROP NOT RECOMMENDED
- **ADR references:** conversion-plan.md Layer 2; code-conversion.md §Circular Dependency
- **Testing approach:** Struct construction and zero-value safety; `go list -deps ./graphtypes/...` verified to have no deps beyond models; must complete before Wave 4 begins
- **Sprint estimate:** 0.5 sprints (can overlap with end of Wave 2)
- **Expected business value:** Unblocks graph, cluster, and ner conversion; critical path item with near-zero effort (WSJF 20.0)
- **Risk:** LOW — pure data types with no logic

---

### Wave 4 — Graph Extraction & Document Loading
**ADR Layer 3**

- **Components:** `graph` (SVOExtractor, GraphBuilder, ContextGraph, EntityPipeline, GraphIndex, LLM triple extraction), `loader` (DocumentLoader)
- **Framework migrations:**
  - OpenAI-compatible LLM → Together.ai Chat API (REWORK)
  - Pure in-memory graph structs → Go structs (DROP-IN)
  - Python lazy circular imports resolved via `graphtypes` (REWORK — structural)
- **Interop boundary:** INTEROP NOT RECOMMENDED
- **ADR references:** conversion-plan.md Layer 3; code-conversion.md §graph, §loader; ml-services.md §SVO
- **Testing approach:** `SVOExtractor` unit tests with `stubTogetherClient`; `loader` tests for all 5 formats + unknown extension error; Checkpoint 2 integration (DocumentLoader mixed formats); **REDESIGN flag on SVO extraction** — LLM output quality requires manual validation against Python baseline triples
- **Sprint estimate:** 2 sprints
- **Expected business value:** DocumentLoader operational end-to-end; graph triple extraction wired; enables storage and community layers
- **Risk:** MEDIUM — LLM-backed graph extraction requires manual quality review; `EntityPipeline` orchestration complexity (~291 LOC Python source)

---

### Wave 5 — Storage & Community Detection
**ADR Layer 4**

- **Components:** `storage` (DuckDB + pgvector backends, Store façade, connection pool, relational store, FTS), `community` (CommunityIndex, CommunityBuilder, Louvain, CommunitySummarizer)
- **Framework migrations:**
  - duckdb-engine + sqlalchemy → go-duckdb + database/sql (REWORK)
  - psycopg2 + pgvector → pgx/v5 + pgvector-go (DROP-IN)
  - igraph + leidenalg → pure Go Louvain ~120 LOC (REWORK)
  - OpenAI-compatible LLM → Together.ai Chat API for summarization (REWORK)
  - numpy cosine similarity → pure Go float32 cosine (REWORK)
- **Interop boundary:** INTEROP NOT RECOMMENDED
- **ADR references:** conversion-plan.md Layer 4; code-conversion.md §storage, §community; ml-services.md §Community Summarisation
- **Testing approach:** All storage tests use `:memory:` DuckDB; thread-safety test (10 goroutines × 5 chunks); community tests with synthetic embeddings; Louvain with fully disconnected graph (N communities for N nodes); `CommunitySummarizer` with stub client; performance: write throughput and FTS latency benchmarks vs Python baseline
- **Sprint estimate:** 3 sprints
- **Expected business value:** Production storage layer operational; community detection enables graph-enriched search; highest business value component (WSJF 3.43, but foundational)
- **Risk:** HIGH — largest package group (storage 3,372 LOC + community 1,158 LOC); thread-safety bugs possible; Louvain port correctness must be validated against Python leiden output

---

### Wave 6 — NER, Clustering, Indexing, Lifecycle
**ADR Layer 5** — all four packages convert in parallel

- **Components:** `cluster` (AgglomerativeClustering, DBSCAN, CooccurrenceMatrix, ClusterMap), `ner` (EntityIndex, NERPipeline, VocabularyMatcher, normalizer, schema vocab), `indexer` (background goroutine indexer, embedding batching), `lifecycle` (BuildNamespaceAsync)
- **Framework migrations:**
  - scikit-learn AgglomerativeClustering + DBSCAN → pure Go ~100 LOC (REWORK)
  - spacy SpacyMatcher → Together.ai Chat API structured JSON (REDESIGN — ADR resolved)
  - sentence-transformers embedding → Together.ai Embeddings API (REDESIGN — ADR resolved)
  - Python threading.Thread → goroutine + context.CancelFunc (REWORK)
  - Python threading.Event → context.Context cancellation (REWORK)
  - inflect library → Go string manipulation or embedded rules (REWORK)
  - spacy vocabulary/label sets → Go `//go:embed` JSON files (DROP-IN data, REWORK tooling)
- **Interop boundary:** INTEROP NOT RECOMMENDED
- **ADR references:** conversion-plan.md Layer 5; code-conversion.md §cluster, §ner, §indexer; ml-services.md §NER Inference, §Embedding Service
- **Testing approach:** 1:1 port of `test_cluster.py` (co-occurrence matrix, Jaccard normalisation, ClusterMap groupings), `test_ner.py` (vocabulary matcher, alias match, frequency/position/span), `test_ner_pipeline.py` (stub client), `test_indexer.py` (stub embedder, async abort via context cancel); **REDESIGN flag on NER** — entity recall must be manually validated against Python/spacy baseline on held-out fixture set (±10% threshold); Checkpoint 3 integration: index a directory with stub embedder
- **Sprint estimate:** 3 sprints
- **Expected business value:** Full NER entity extraction and clustering operational; indexer enables document ingestion pipeline; lifecycle enables namespace rebuild
- **Risk:** HIGH — NER quality depends on LLM prompt design (no mechanical equivalence); clustering algorithm correctness requires numerical validation against sklearn output; indexer goroutine abort logic requires careful context propagation testing

---

### Wave 7 — Enhanced Search
**ADR Layer 6**

- **Components:** `search` (EnhancedSearch, 4-lane cohort assembly: seed/vector, structural, entity, cluster; BM25 FTS; scoring mixins; graph-enhanced search)
- **Framework migrations:**
  - Python cosine similarity + numpy → pure Go float32 (REWORK)
  - DuckDB FTS → go-duckdb FTS (DROP-IN — same SQL)
  - 4-lane search logic → Go struct with toggle-able lanes (REWORK)
- **Interop boundary:** INTEROP NOT RECOMMENDED
- **ADR references:** conversion-plan.md Layer 6; code-conversion.md §search
- **Testing approach:** In-memory DuckDB with synthetic float32 embeddings (dim 8); test each of 4 lanes independently and combined; deduplication across lanes; `ScoredChunk.Provenance` lane attribution; `top_k` enforcement; all-lanes-disabled → empty results; performance: latency at k=10, k=30, k=50 vs Python baseline (Go should be faster; flag any regression)
- **Sprint estimate:** 2 sprints
- **Expected business value:** Highest direct user-facing impact (WSJF 4.17); 4-lane search is the core retrieval capability that differentiates Chonk from vanilla vector search
- **Risk:** MEDIUM — search logic (1,871 LOC Python) is complex but deterministic; BM25/FTS is a DROP-IN; main risk is scoring mixin correctness across lane combinations

---

### Wave 8 — Ingest Orchestration (Final Binary)
**ADR Layer 7**

- **Components:** `ingest` (Build(), Index façade, YAML config loading, ingest phases: ingest→embed→FTS→NER→community→SVO), `_ingest_worker` (background worker goroutine)
- **Framework migrations:**
  - pyyaml → gopkg.in/yaml.v3 (DROP-IN — config schema identical)
  - Python threading workers → goroutine + channel (REWORK)
  - All phase orchestration → Go sequential/concurrent with context cancellation (REWORK)
- **Interop boundary:** INTEROP NOT RECOMMENDED
- **ADR references:** conversion-plan.md Layer 7; code-conversion.md §ingest
- **Testing approach:** Stub embedder via `BuildOptions` struct (no live API key required for unit tests); fixture YAML under `ingest/testdata/`; Checkpoint 4 integration: full pipeline with `TOGETHER_API_KEY` or go-vcr cassette; verify config YAML backward-compatibility with existing Python `.yaml` files; unsupported format (.pdf) skipped gracefully; `Index.Count()` matches ingested chunks
- **Sprint estimate:** 2 sprints
- **Expected business value:** Delivers the complete Go binary; existing Python `.yaml` config files work unchanged; single executable deployment with no pip, no venv, no model download
- **Risk:** MEDIUM — orchestration complexity (861 LOC ingest.py + 527 LOC _ingest_worker.py); phase ordering must be preserved exactly; YAML backward-compatibility is a hard requirement and must be regression-tested against all existing config files in `work/configs/`

---

## 8. RISKS AND ASSUMPTIONS

### Top Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **NER quality degradation** — spacy→Together.ai LLM NER produces different entity labels or misses entities present in Python output | HIGH | Establish Python/spacy baseline recall on held-out fixtures before starting Wave 6; gate Wave 6 completion on ±10% entity recall threshold; iterate on NER prompt if threshold not met |
| R2 | **Louvain community partition divergence** — pure Go Louvain implementation produces different community assignments than Python leidenalg | HIGH | Run Python leiden on benchmark corpus, record community memberships; compare Go Louvain output; acceptable threshold: same community count ± 10%, same top-3 members per community |
| R3 | **Storage thread-safety bugs** — DuckDB single-writer constraint violated under concurrent Go goroutines | HIGH | 10-goroutine concurrent write test (ADR unit-testing.md Layer 4); enforce `sync.RWMutex` consistently; do not expose DuckDB connection directly |
| R4 | **SVO triple extraction quality** — LLM-backed triple extraction may produce fewer or lower-confidence triples than Python baseline | MEDIUM | Manual review of triple output on same fixture set; ADR flags this as a REDESIGN requiring manual equivalence testing |
| R5 | **YAML config backward-compatibility breaks** — ingest config schema drift between Python and Go parsers | MEDIUM | Parse all 80+ existing `work/configs/*.toml` and `work/configs/**/*.yaml` files with Go parser in Wave 8; any parse error is a blocker |
| R6 | **Together.ai API latency / rate limits** introduce indexing bottleneck absent in Python (local ML) | MEDIUM | Implement exponential back-off + retry (defined in ADR ml-services.md); batch embedding at 256 texts/request; benchmark indexing throughput against Python baseline |
| R7 | **AgglomerativeClustering numerical divergence** — pure Go cosine + dendrogram may differ from sklearn implementation | LOW-MEDIUM | Compare cluster memberships on same synthetic embedding matrix; use sklearn output as reference; acceptable if cluster assignments match >90% |
| R8 | **go-duckdb version stability** — DuckDB C extension ABI changes between versions may require CGO rebuild | LOW | Pin go-duckdb to a specific DuckDB version; test on both darwin/arm64 and linux/amd64 CI runners |

### Assumptions

| # | Assumption | Should Be Validated |
|---|---|---|
| A1 | Together.ai API is available and `TOGETHER_API_KEY` is provisioned before Wave 4 integration tests | Confirm API key access before Wave 4 begins |
| A2 | Existing `work/configs/` YAML/TOML files are the canonical set of config schemas that must remain backward-compatible | Confirm no additional config files exist in production deployments |
| A3 | The 5 supported extraction formats (text, CSV, Markdown, SQL/DB schema, HTML) cover all production ingestion use cases | Confirm with stakeholders before committing to drop the other 21 format extractors |
| A4 | The Python `_versioning.py` feature namespace lifecycle is not required in the Go binary (no ADR mentions it) | Confirm feature namespace lifecycle is not a production dependency |
| A5 | Out-of-scope transports (Gmail, SharePoint, MongoDB, etc.) are not used in any production ingestion pipeline | Confirm no active pipelines depend on these transports before Wave 1 is finalized |
| A6 | go-duckdb supports the same VSS (vector similarity search) extension and FTS extension as the Python duckdb driver | Validate with a spike in Wave 5 before committing to storage implementation |

### Deferred Items (out of scope, revisit post-initial-release)
- 21 dropped extractor formats
- 14 out-of-scope transports (Gmail, SharePoint, MongoDB, Elasticsearch, etc.)
- `_versioning.py` feature namespace lifecycle
- Import crawler (Python-specific AST analysis)
- Existing benchmark tooling (`demo/`, `work/`) — Python-only, not migrated

---

## Summary: Migration Priority Order

| Priority | Wave | Theme | ADR Layer | Sprints | Risk | WSJF (lead component) |
|---|---|---|---|---|---|---|
| 1 | Wave 1 | Foundation — pure types & core transports | Layer 0 | 1 | LOW | 9.0 |
| 2 | Wave 2 | Stateless pipeline — chunking, extraction, context, generation | Layer 1 | 2 | LOW-MEDIUM | 5.25 |
| 3 | Wave 3 | Cycle break — graphtypes | Layer 2 | 0.5 | LOW | 20.0 |
| 4 | Wave 4 | Graph extraction & document loading | Layer 3 | 2 | MEDIUM | 3.0 |
| 5 | Wave 5 | Storage & community detection | Layer 4 | 3 | HIGH | 3.43 |
| 6 | Wave 6 | NER, clustering, indexing, lifecycle | Layer 5 | 3 | HIGH | 2.63 |
| 7 | Wave 7 | Enhanced 4-lane search | Layer 6 | 2 | MEDIUM | 4.17 |
| 8 | Wave 8 | Ingest orchestration — final binary | Layer 7 | 2 | MEDIUM | 5.0 |
| — | Deferred | Dropped extractors, out-of-scope transports | N/A | — | — | DEFER |

**Total estimated effort:** ~15.5 sprints (~8 months at 2-week sprints)

**Go module:** `github.com/kenstott/chonk-go` (Go 1.22+)

**ADR alignment:** All 8 waves align with the committed future state ADRs in `aipa_test_mcp_server/future_state_architecture/`. No wave conflicts with any existing architectural decision.
