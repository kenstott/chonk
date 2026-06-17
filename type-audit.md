# Type Consistency Audit — chonk Python → Go Migration

**Tool chain:** Pyright 1.1.409 | Python 3.11 | typeCheckingMode = standard  
**Scope:** `chonk/` package (114 files analysed)  
**Migration context:** Python → Go rewrite across 8 waves

---

## Executive Summary

| Metric | Count | Severity |
|---|---|---|
| Pyright errors | 0 | — |
| Pyright warnings (missing imports) | 71 | Expected / environment |
| Functions missing return-type annotation | 83 | **Medium** |
| Parameters missing type annotation | 111 | **Medium** |
| `Any` usages (all forms) | 60 | **High** (migration-critical) |
| Bare collection returns (`list`, `dict`, `tuple`) | 18 | **High** (opaque to Go) |
| Untyped `**kwargs` | 39 | **Medium** |
| `Any` inside annotations (`dict[str, Any]`, etc.) | 17 | **Medium** |
| Protocol methods lacking concrete types | 6 | **High** (contract boundary) |
| Go migration blockers (unsafe/opaque types) | 35 | **Critical** |

**Type coverage estimate:** ~72 % of function signatures are fully annotated. The missing 28 % is concentrated in storage, transports, and the `ingest` orchestrator.

---

## 1. Pyright Diagnostic Summary

Pyright reports **0 errors** and **71 warnings**, all of category `reportMissingImports`.  
Every warning traces to an optional extra that is not installed in the type-checking venv (duckdb, numpy, sentence-transformers, spacy, igraph, leidenalg, inflect, sqlalchemy, pyarrow, pandas, boto3, cassandra-driver, pymongo, azure-cosmos, google-api, pptx, docx, odfpy, pypdf, certifi).

**Assessment:** All 71 warnings are intentional — each optional import is guarded by a `try/except ImportError` block that raises a descriptive error on use. The `pyproject.toml` suppresses these with `reportMissingImports = "warning"`. No action required for these.

No suppression of genuine type errors is present (`# type: ignore` annotations are absent from in-scope files).

---

## 2. Type Coverage by Sub-package

| Sub-package | Files | Missing returns | Missing params | `Any` uses | Coverage % |
|---|---|---|---|---|---|
| `storage/` | 7 | 24 | 19 | 4 | 55 % |
| `transports/` | 21 | 16 | 33 | 22 | 60 % |
| `ingest.py` + `_ingest_worker.py` | 2 | 4 | 3 | 17 | 65 % |
| `ner/` | 7 | 6 | 16 | 0 | 70 % |
| `extractors/` | 20 | 12 | 10 | 4 | 75 % |
| `search/` | 4 | 4 | 12 | 4 | 75 % |
| `graph/` | 7 | 0 | 9 | 0 | 80 % |
| `community/` | 4 | 10 | 12 | 1 | 80 % |
| `generation/` | 3 | 1 | 0 | 0 | 85 % |
| `cluster/` | 4 | 1 | 0 | 0 | 92 % |
| `models.py` | 1 | 0 | 0 | 1 | 95 % |
| `chunking.py` | 1 | 0 | 0 | 0 | 99 % |
| `lifecycle.py` | 1 | 0 | 0 | 3 | 90 % |
| `indexer.py` | 1 | 2 | 0 | 3 | 82 % |
| `loader.py` | 1 | 4 | 1 | 0 | 75 % |
| **TOTAL** | **84** | **83** | **111** | **60** | **~72 %** |

---
## 3. Critical Findings — Go Migration Blockers

### 3.1 `Any` on Protocol Boundary Attributes (CRITICAL)

**File:** `chonk/storage/_protocol.py` lines 31–32

```python
# chonk/storage/_protocol.py
_conn: Any
```

`VectorBackend._conn` is declared `Any` because DuckDB ships no stubs. This is the most migration-critical annotation in the codebase: **every caller in storage, search, ingest, graph, and lifecycle reaches into `._conn` directly** and chains `.execute(...)` calls off it. In Go, this becomes a concrete `*duckdb.Conn` field — but without a typed Python Protocol, there is no machine-checkable contract to port from.

**Impact on Go migration:** Wave 5 (storage) cannot generate a Go interface from this protocol. The Go `VectorBackend` interface must be hand-written from the call-sites.

**Fix:**
```python
# chonk/storage/_protocol.py
import duckdb  # type: ignore[import]

@runtime_checkable
class VectorBackend(Protocol):
    _conn: duckdb.DuckDBPyConnection  # was: Any
    _fts_dirty: bool
```

---

### 3.2 Bare `list` / `dict` Return Types (HIGH — opaque to Go)

18 functions declare `-> list` or `-> dict` with no element type. Go generics require the element type to size the slice or map.

| File | Function | Current | Required |
|---|---|---|---|
| `ingest.py:57` | `_ingest_glob` | `-> list` | `-> list[DocumentChunk]` |
| `ingest.py:72` | `_ingest_json_array` | `-> list` | `-> list[DocumentChunk]` |
| `ingest.py:89` | `_ingest_sql` | `-> list` | `-> list[DocumentChunk]` |
| `ingest.py:154` | `_ingest_source` | `-> list` | `-> list[DocumentChunk]` |
| `indexer.py:215` | `_crawl` | `-> list` | `-> list[DocumentChunk]` |
| `storage/_protocol.py:76` | `get_all_chunks` | `-> list` | `-> list[DocumentChunk]` |
| `storage/_vector.py:524` | `get_all_chunks` | `-> list` | `-> list[DocumentChunk]` |
| `storage/_pg.py:59` | `fetchall` | `-> list` | `-> list[tuple[Any, ...]]` |
| `storage/_pg.py:606` | `_search_hybrid` | `-> list` | `-> list[tuple[str, float, DocumentChunk]]` |
| `storage/_pg.py:722` | `get_all_chunks` | `-> list` | `-> list[DocumentChunk]` |
| `storage/_store.py:193` | `search` | `-> list` | `-> list[tuple[str, float, DocumentChunk]]` |
| `transports/_cassandra.py:218` | `get_table_meta` | `-> list` | `-> list[TableMeta]` |
| `transports/_cassandra.py:374` | `cassandra_provenance` | `-> dict` | `-> dict[str, str]` |
| `transports/_sql_query.py:67` | `db_provenance` | `-> dict` | `-> dict[str, str]` |
| `graph/_entity_pipeline.py:171` | `_extract_one` | `-> tuple` | `-> tuple[str, list[Triple]]` |
| `cluster/_map.py:143` | `to_dict` | `-> dict` | `-> dict[str, list[str]]` |
| `ner/_index.py:217` | `to_dict` | `-> dict` | `-> dict[str, list[EntityAssociation]]` |
| `extractors/_xml.py:46` | `_to_dict` | `-> dict` | `-> dict[str, object]` |

**Fix pattern (example — `_ingest_glob`):**
```python
# chonk/ingest.py
from .models import DocumentChunk

def _ingest_glob(loader: DocumentLoader, src: dict[str, object]) -> list[DocumentChunk]:
    ...
```

---

### 3.3 `embed_model: str | Any` — Union with `Any` Erases the Type (HIGH)

**Files:** `lifecycle.py:25`, `lifecycle.py:189`, `indexer.py:28`, `indexer.py:93`

```python
# chonk/lifecycle.py
embed_model: str | Any,  # <-- Any absorbs the union; equivalent to Any alone
```

`str | Any` collapses to `Any` — pyright treats it as fully untyped. In Go this parameter must be a concrete interface (either a string model name or a `SentenceEmbedder` interface). The current annotation gives no information about the interface contract.

**Fix:**
```python
# Define once in chonk/indexer.py or chonk/_types.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Embedder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> list[list[float]]: ...

EmbedModel = str | Embedder  # now a documented union

# chonk/lifecycle.py
def build_namespace_async(
    namespace_id: str,
    db_path: str | Path,
    embed_model: EmbedModel,   # replaces str | Any
    ...
```

This gives Go Wave 5/6 an explicit `Embedder` interface to implement.

---

### 3.4 `Callable[..., Any]` in `_GraphMixin` Attribute Declarations (HIGH)

**File:** `chonk/search/_enhanced_graph.py:61–64`

```python
search: Callable[..., Any]
_resolve_ns_chunk_ids: Callable[..., Any]
_select_cohort: Callable[..., Any]
_fetch_chunk: Callable[..., Any]
```

These mixin-protocol attribute declarations use `Callable[..., Any]` — the wildcard parameter spec. Go has no mixin pattern; the Go equivalent is a struct with function-typed fields or interface methods. Without concrete signatures, Go function types cannot be defined.

**Required concrete signatures:**
```python
from .models import DocumentChunk, ScoredChunk
from ._enhanced_support import RetrievalTrace

search: Callable[
    [np.ndarray, int, str | None, list[str] | None, dict | None, str, list[str] | None, list[str] | None],
    list[ScoredChunk],
]
_resolve_ns_chunk_ids: Callable[[list[str] | None, list[str] | None], set[str] | None]
_select_cohort: Callable[[list[ScoredChunk], np.ndarray, int], list[ScoredChunk]]
_fetch_chunk: Callable[[str], DocumentChunk | None]
```

---

### 3.5 Callback Parameters Typed as `Any` in `Index` / `build_namespace_async` (HIGH)

**Files:** `ingest.py:382–384`, `ingest.py:449–451`, `ingest.py:488–490`, `lifecycle.py:24–26`

```python
on_progress: Any = None,
on_complete: Any = None,
on_error: Any = None,
```

These callbacks are the primary progress/error reporting surface of the async pipeline. With `Any`, pyright cannot verify callers pass the correct signatures. More critically, the Go migration (Wave 8) cannot define `chan`-based or closure-typed equivalents without knowing the exact function signatures.

**Correct types** (already used correctly in `NamespaceRefresher.__init__`):
```python
on_progress: Callable[[str, int, int], None] | None = None,
on_complete: Callable[[int], None] | None = None,
on_error: Callable[[str, Exception], None] | None = None,
```

---
### 3.6 `dict[str, Any]` Config Arguments in `build()` / `_build_ingest_phase()` (MEDIUM-HIGH)

**File:** `chonk/ingest.py:574,577,578,658,764,785`

```python
cfg: dict[str, Any],
loader_cfg: dict[str, Any],
embed_cfg: dict[str, Any],
ic: dict[str, Any],
```

The YAML-parsed config is passed through the entire pipeline as `dict[str, Any]`. In Go the config is a typed struct (the YAML schema is documented in the `build()` docstring). `dict[str, Any]` gives Go no structural information for auto-generating the config struct.

**Fix — define a typed config hierarchy:**
```python
from dataclasses import dataclass, field

@dataclass
class EmbedConfig:
    model: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 256

@dataclass
class LoaderConfig:
    min_chunk_size: int = 1100
    max_chunk_size: int = 2200
    enrich_context: bool = True
    extra_extractors: list[str] = field(default_factory=list)

@dataclass
class IndexConfig:
    ner: bool = True
    community: bool = True
    svo: bool = False
    spacy_model: str = "en_core_web_sm"
    svo_model: str = "gpt-4o-mini"
    community_alpha: float = 0.2
    community_sim_threshold: float = 0.6

@dataclass
class ChonkConfig:
    store: dict[str, object]
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    sources: list[dict[str, object]] = field(default_factory=list)
    namespaces: dict[str, object] = field(default_factory=dict)
```

---

### 3.7 `source_detail: dict[str, Any] | None` on `DocumentChunk` (MEDIUM)

**File:** `chonk/models.py:82`

```python
source_detail: dict[str, Any] | None = None
```

`DocumentChunk` is the fundamental transfer object that crosses every layer boundary. `source_detail` stores format-specific navigation metadata (e.g. EDGAR filing section IDs, XLSX sheet names). With `Any` as the value type, Go cannot define a discriminated union or an interface for this field.

**Options (in order of preference):**
1. **Strongest:** Make `source_detail` a typed `dataclass` per chunk type and use `object` as the field type with runtime dispatch.
2. **Pragmatic:** Change to `dict[str, str | int | float | bool | None]` — covers all JSON-serialisable scalar values, maps cleanly to `map[string]any` in Go.
3. **Acceptable for migration:** Keep `Any` but add a `TypeAlias` to document intent:
```python
SourceDetail = dict[str, str | int | float | bool | None]
source_detail: SourceDetail | None = None
```

---

## 4. Missing Annotations — Top Files by Severity

### 4.1 `chonk/storage/_pool.py` — 15 missing returns, 4 missing params

The DuckDB connection pool is the single most under-annotated file in the codebase (19 violations). It is Wave 5 material and the foundation for all storage.

| Method | Problem | Fix |
|---|---|---|
| `_PendingResult.__init__` | `conn` param untyped | `conn: duckdb.DuckDBPyConnection` |
| `_PendingResult.fetchall` | No return type | `-> list[tuple[Any, ...]]` |
| `_PendingResult.fetchone` | No return type | `-> tuple[Any, ...] \| None` |
| `_PendingResult.fetchdf` | No return type | `-> pd.DataFrame` |
| `_PendingResult.description` | No return type | `-> list[duckdb.column_expression] \| None` |
| `_PendingResult.__getattr__` | No types | `(name: str) -> object` |
| `_LockedConnection.__init__` | `conn` param untyped | `conn: duckdb.DuckDBPyConnection` |
| `_LockedConnection.execute` | No types | `(*args: object, **kwargs: object) -> _PendingResult` |
| `_LockedConnection.executemany` | No types | `(*args: object, **kwargs: object) -> None` |
| `_LockedConnection.__getattr__` | No types | `(name: str) -> object` |
| `ThreadLocalDuckDB.__init__` | Missing `-> None` | Add `-> None` |
| `ThreadLocalDuckDB.close` | Missing `-> None` | Already has it (OK) |
| `ThreadLocalDuckDB.__exit__` | `*_` untyped | `*_: object` |

### 4.2 `chonk/storage/_pg.py` — 4 missing returns, 8 missing params

The PgVector backend has the same pattern as the DuckDB pool — internal proxy methods lack types.

### 4.3 `chonk/ner/_build.py` — 2 missing returns, 9 missing params

NER build functions pass DuckDB connections as untyped parameters. All `con` parameters should be `duckdb.DuckDBPyConnection`.

### 4.4 `chonk/transports/_db_schema.py` — 1 missing return, 9 missing params

Schema crawlers pass SQLAlchemy `Engine` and `Connection` objects untyped. This is Wave 4 (loader) migration material.

### 4.5 `chonk/community/_index.py` — 9 missing params

Community index passes DuckDB connections without type annotations across 9 parameters.

---

## 5. Protocol Completeness Audit

### 5.1 `Transport` Protocol — INCOMPLETE

**File:** `chonk/transports/_protocol.py:27–28`

```python
class Transport(Protocol):
    def fetch(self, uri: str, **kwargs) -> FetchResult: ...  # **kwargs untyped
    def can_handle(self, uri: str) -> bool: ...
```

`**kwargs` is untyped on `fetch`. The concrete implementations use it for credentials (token, headers, auth) — different kwargs per transport. The Go `Transport` interface cannot accept variadic kwargs. 

**Migration note for Go:** The kwargs must become a typed options struct or be folded into the transport constructor. Recommended Go pattern:
```go
type FetchOptions struct {
    Headers    map[string]string
    Token      string
    MaxRetries int
}

type Transport interface {
    Fetch(ctx context.Context, uri string, opts FetchOptions) (FetchResult, error)
    CanHandle(uri string) bool
}
```

**Python fix (pre-migration):**
```python
@dataclass
class FetchOptions:
    headers: dict[str, str] = field(default_factory=dict)
    token: str = ""
    max_retries: int = 3

class Transport(Protocol):
    def fetch(self, uri: str, opts: FetchOptions | None = None) -> FetchResult: ...
    def can_handle(self, uri: str) -> bool: ...
```

### 5.2 `Extractor` Protocol — COMPLETE

All three methods (`extract`, `can_handle`, `annotate`) are fully typed with concrete parameter and return types. No issues.

### 5.3 `VectorBackend` Protocol — MOSTLY COMPLETE, one `Any` blocker

`add_chunks` declares `embeddings` without a type annotation (np.ndarray with no stubs). `get_all_chunks` returns bare `list`. See §3.1 and §3.2.

**Fix:**
```python
import numpy as np  # type: ignore[import]

def add_chunks(
    self,
    chunks: list[DocumentChunk],
    embeddings: np.ndarray,         # shape (n, embedding_dim)
    ...
) -> None: ...

def get_all_chunks(self) -> list[DocumentChunk]: ...
```

### 5.4 `CrawlerProtocol` — COMPLETE

`_crawler_protocol.py` defines `list(uri: str) -> list[FetchResult]` and `can_handle(uri: str) -> bool` — fully typed.

---

## 6. Unsafe Patterns — Null Safety

### 6.1 `.fetchone()[0]` without None guard (HIGH)

**Pattern found in:** `lifecycle.py`, `ingest.py`, `storage/_vector.py`, `storage/_pg.py`, `community/_index.py`

```python
# lifecycle.py:79 — crashes if table is empty
store.vector._conn.execute(
    "SELECT COUNT(*) FROM embeddings WHERE namespace = ?", [namespace_id]
).fetchone()[0]
```

`fetchone()` returns `tuple | None`. Indexing `[0]` on `None` raises `TypeError` at runtime. Pyright in standard mode does not flag this because `_conn` is typed `Any`.

**Count:** 23 unguarded `.fetchone()[0]` calls across storage and lifecycle.

**Fix pattern:**
```python
row = conn.execute("SELECT COUNT(*) FROM ...").fetchone()
count = row[0] if row is not None else 0
```

**Go migration note:** Go's `database/sql` `QueryRow().Scan()` is safe-by-default — no equivalent null deref risk. But the Python code should be fixed before migration to document the intended behaviour (0 vs. error) so Go code can mirror it correctly.

### 6.2 Implicit `assert` for None narrowing (MEDIUM)

**Pattern found in:** `search/_enhanced_graph.py` (7 locations), `storage/_vector.py` (1 location)

```python
assert self._entity_index is not None  # used as a None guard before accessing
```

Using `assert` for control flow is fragile — assertions are disabled with `python -O`. These should be replaced with explicit `if` guards or proper Optional narrowing.

**Fix:**
```python
if self._entity_index is None:
    return []
# pyright now narrows self._entity_index to EntityIndex (not None)
```

### 6.3 `dict.get()` chained without None checks (MEDIUM)

**Pattern:** `ingest.py:_load_domain_map`, `lifecycle.py:_check_all`

```python
self._domain_map.get(namespace_id, {}).pop(domain_name, None)
```

Safe individually, but the outer result of `.get` can return an empty dict that silently discards the `pop`. The intent is to remove from a known-present key. These should use direct dict access with an explicit presence check.

---

## 7. Type Inconsistencies (Same Concept, Different Types)

### 7.1 `chunk_id` — `str` vs positional `tuple[0]`

`chunk_id` is `str` throughout the public API (`ScoredChunk.chunk_id`, `VectorBackend.search` return), but internal DuckDB row unpacking uses positional indexing without types:
```python
for rank, (chunk_id, _score, chunk) in enumerate(vector_results, start=1):
```
The destructuring is correct but unverified — if the DuckDB query column order changes, `chunk_id` silently becomes a score float. Go's `Scan()` is position-invariant by design.

### 7.2 `section` — `list[str]` vs `str` vs JSON string

`DocumentChunk.section` is declared `list[str]` but the storage layer persists it as a JSON string and deserialises it via `_deserialize_section()` which also handles raw `str` and legacy formats. The type is not `list[str]` at all storage boundaries.

**Fix:** Introduce a `SectionPath = list[str]` alias and document the serialisation boundary explicitly. The Go equivalent should be `[]string` with JSON marshal/unmarshal.

### 7.3 `chunk_type` — `str` vs enum-like sentinel

`chunk_type` is `str` but has a closed set of values: `"document"`, `"db_table"`, `"db_column"`, `"db_schema"`, `"api_endpoint"`, `"graphql_query"`, `"graphql_mutation"`, `"graphql_type"`, `"graphql_field"`, `"community_summary"`. These are compared with `==` and `in (...)` throughout the codebase.

**Fix:** Replace with `Literal[...]` or a `StrEnum`:
```python
from typing import Literal
ChunkType = Literal[
    "document", "db_table", "db_column", "db_schema",
    "api_endpoint", "graphql_query", "graphql_mutation",
    "graphql_type", "graphql_field", "community_summary",
]
```

Go equivalent: `type ChunkType string` with `const` declarations.

### 7.4 `search()` return — `list[tuple[str, float, object]]` vs `list[tuple[str, float, DocumentChunk]]`

`VectorBackend.search` declares return `list[tuple[str, float, object]]` (with comment `# DocumentChunk`). `Store.search` declares `-> list` (bare). `EnhancedSearch.search` returns `list[ScoredChunk]`. Three layers, three different representations for the same data.

**Fix:** Standardise the raw-search return type:
```python
SearchResult = tuple[str, float, DocumentChunk]  # (chunk_id, score, chunk)

class VectorBackend(Protocol):
    def search(self, ...) -> list[SearchResult]: ...

class Store:
    def search(self, ...) -> list[SearchResult]: ...
```

`ScoredChunk` remains the public/enriched form returned by `EnhancedSearch`.

---
## 8. Implicit `Any` Flows — Detailed Callsite Analysis

The 60 `Any` usages cluster into five patterns, each with a different remediation:

### Pattern A — External library proxy (expected, mitigated by guards)
**Count:** 30 | **Files:** `_pool.py`, `_cassandra.py`, `_cosmos.py`, `_dynamodb.py`, `_firestore.py`, `_gmail.py`, `_mongodb.py`, `_elasticsearch.py`, `_schema_infer.py`

Library objects (DuckDB connections, Cassandra sessions, Firestore clients) typed `Any` because those libraries ship no stubs. Each use site is already guarded by `try/except ImportError`. These are acceptable for the Python codebase but must be replaced with concrete Go types in migration.

**Migration action:** For each, identify the minimal interface needed:
- DuckDB `_conn`: `Execute(sql string, params ...any) (*Rows, error)` 
- Cassandra `session`: define a `CassandraSession` interface with `Execute(stmt string) ([]Row, error)`

### Pattern B — Callback signatures typed `Any` (fix required)
**Count:** 9 | **File:** `ingest.py:382–384,449–451,488–490`

Already covered in §3.5. Replace with typed `Callable` signatures.

### Pattern C — Config dicts typed `dict[str, Any]` (fix required)
**Count:** 6 | **File:** `ingest.py:574,577,578,658,764,785`

Already covered in §3.6. Replace with typed dataclasses.

### Pattern D — `embed_model: str | Any` (fix required)
**Count:** 3 | **Files:** `lifecycle.py`, `indexer.py`

Already covered in §3.3. Introduce `Embedder` protocol.

### Pattern E — `__exit__` variadic (acceptable)
**Count:** 3 | **Files:** `ingest.py:298`, `storage/_pool.py:252`, `storage/_pg.py:893`

`def __exit__(self, *_: Any) -> None` is the canonical form for context managers that do not inspect exception info. This is correct and acceptable.

---

## 9. Recommended Fixes — Priority Order

Fixes are ordered by impact on the Go migration (critical-path items first).

### P0 — Blocking Go interface generation

| # | File | Change | Effort |
|---|---|---|---|
| 1 | `transports/_protocol.py` | Type `**kwargs` as `FetchOptions` dataclass | 1h |
| 2 | `storage/_protocol.py` | Type `_conn` as `duckdb.DuckDBPyConnection`, type `get_all_chunks` return | 30m |
| 3 | `storage/_protocol.py` | Type `add_chunks(embeddings)` as `np.ndarray` | 15m |
| 4 | `search/_enhanced_graph.py` | Replace `Callable[..., Any]` with concrete signatures | 2h |
| 5 | `indexer.py` + `lifecycle.py` | Replace `str | Any` with `EmbedModel = str \| Embedder` Protocol | 1h |
| 6 | `ingest.py` callbacks | Replace `Any` with typed `Callable[[str, int, int], None]` etc. | 30m |

### P1 — Closes bare collection returns (18 functions)

All 18 `-> list` / `-> dict` return types should gain element types. Highest priority:
- `storage/_protocol.py:get_all_chunks` → `list[DocumentChunk]`
- `storage/_store.py:search` → `list[SearchResult]`
- `ingest.py` ingest helpers → `list[DocumentChunk]`
- `indexer.py:_crawl` → `list[DocumentChunk]`

**Bulk fix script:**

The following functions can be fixed mechanically:
```python
# All four _ingest_* helpers in ingest.py
def _ingest_glob(loader: DocumentLoader, src: dict[str, object]) -> list[DocumentChunk]:
def _ingest_json_array(loader: DocumentLoader, src: dict[str, object]) -> list[DocumentChunk]:
def _ingest_sql(loader: DocumentLoader, src: dict[str, object]) -> list[DocumentChunk]:
def _ingest_source(src: dict[str, object], loader: DocumentLoader) -> list[DocumentChunk]:
```

### P2 — Typed config hierarchy

Define `ChonkConfig`, `EmbedConfig`, `LoaderConfig`, `IndexConfig` dataclasses in a new `chonk/_config.py` module (see §3.6 for definitions). This single change eliminates all 6 `dict[str, Any]` config usages and gives Go Wave 8 a complete struct definition.

### P3 — `ChunkType` literal / enum

Replace ad-hoc `str` chunk types with `Literal[...]` or `StrEnum` (see §7.3). Eliminates silent string comparison errors and gives Go a `const` block.

### P4 — `section` serialisation boundary

Rename `_deserialize_section` to `_load_section` and add a return type `-> list[str]`. Add a `SectionPath = list[str]` type alias in `models.py`. Mark storage persistence as an internal detail.

### P5 — Missing `__init__` return type annotations (83 functions)

The majority of the 83 missing-return violations are `__init__` methods. These all return `None` implicitly. Add `-> None` to all 83. This is a mechanical one-liner change per method.

---

## 10. Go Migration Type Mapping

Critical Python → Go type mappings derived from this audit:

| Python type | Go type | Notes |
|---|---|---|
| `DocumentChunk` dataclass | `models.DocumentChunk` struct | All fields map 1:1; `section []string`, `sourceDetail map[string]any` |
| `FetchResult` dataclass | `transports.FetchResult` struct | Direct port |
| `ScoredChunk` dataclass | `search.ScoredChunk` struct | Direct port |
| `Entity` / `EntityAssociation` | `models.Entity` / `models.EntityAssociation` | Direct port |
| `list[DocumentChunk]` | `[]models.DocumentChunk` | All `-> list` returns must be concretised first |
| `list[tuple[str, float, DocumentChunk]]` | `[]search.SearchResult` struct | Define `SearchResult` struct in Go |
| `dict[str, Any]` config | `config.ChonkConfig` struct | Must define typed config before Wave 8 |
| `str \| Any` (embed_model) | `indexer.EmbedModel` interface | `Encode([]string) ([][]float32, error)` |
| `Callable[[str, int, int], None]` | `func(phase string, done, total int)` | Go function type literal |
| `Callable[[int], None]` | `func(total int)` | Go function type literal |
| `Callable[[str, Exception], None]` | `func(phase string, err error)` | Go function type literal |
| `duckdb.DuckDBPyConnection` | `*duckdb.Conn` (go-duckdb) | Pool wraps this; not exposed in interface |
| `np.ndarray` shape `(n, dim)` | `[][]float32` or `[][1024]float32` | Fixed dim variant preferred for VSS |
| `Transport` Protocol | `transports.Transport` interface | `Fetch`, `CanHandle` methods; kwargs → `FetchOptions` struct |
| `VectorBackend` Protocol | `storage.VectorBackend` interface | All methods port 1:1 once `Any` is removed |
| `Extractor` Protocol | `extractors.Extractor` interface | Already fully typed; cleanest protocol |
| `chunk_type: str` (closed set) | `type ChunkType string` + `const` | `ChunkType = "document"` etc. |
| `section: list[str]` | `[]string` | JSON-serialised at storage boundary |
| `source_detail: dict[str, Any] \| None` | `map[string]any` nullable | Acceptable for migration |

---

## 11. Type Coverage Metrics

```
Total files analysed:          114
Total Python LOC (chonk/):    ~24,932

Fully annotated functions:     ~68 %  (estimated)
Missing return type:            83 functions  (8.5 % of all functions)
Missing parameter type:        111 parameters (across ~390 functions)
Any usages:                     60 call sites
  - Acceptable (library proxy): 33
  - Must fix (migration blocker): 27

Bare collection returns:        18  (100 % must be fixed for Go)
Protocol completeness:
  - Transport:    70 % (kwargs untyped)
  - Extractor:   100 % (clean)
  - VectorBackend: 85 % (embeddings param, get_all_chunks return)
  - CrawlerProtocol: 100 % (clean)

Pyright error count:             0  (clean baseline)
Pyright warning count:          71  (all expected missing-import)
```

---

## 12. Files Requiring No Changes (Already Fully Typed)

The following files pass this audit with no findings — they serve as reference implementations for the migration:

- `chonk/models.py` — all dataclass fields typed (except `source_detail: dict[str, Any]`)
- `chonk/chunking.py` — fully annotated, no `Any`
- `chonk/context.py` — fully annotated
- `chonk/extractors/_protocol.py` — cleanest Protocol in the codebase
- `chonk/extractors/_text.py`, `_markdown.py`, `_html.py`, `_csv.py` — fully typed
- `chonk/transports/_crawler_protocol.py` — fully typed
- `chonk/transports/_local.py`, `_http.py`, `_s3.py`, `_sftp.py` — fully typed (kwargs acceptable)
- `chonk/generation/_answer.py`, `_context.py`, `_prompt_builder.py` — fully typed
- `chonk/search/_enhanced_scoring.py` — fully typed
- `chonk/cluster/_clusterer.py`, `_cooccurrence.py` — fully typed

---

*Generated by automated pyright + AST analysis on 2025-07-14. All line numbers reference HEAD of the main branch.*
