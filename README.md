# Chonk

A pure-Python RAG/GraphRAG library for production-scale document corpora.

Designed for large, heterogenous, frequently updated enterprise collections — real-world
scenarios, not synthetic benchmarks or demos — where documents share structure, sections
share vocabulary, and naive chunking produces retrievals that are technically correct but
semantically wrong.

Chonk covers the full pipeline: transport and extraction, semantic chunking, contextual
enrichment, vector storage, entity and relationship indexing, community detection,
hybrid search, and answer generation. Every stage has tunable parameters and swappable
implementations.

Two capabilities distinguish it from simpler pipelines:

**Semantic boundary chunking.** Naive pipelines handle split boundaries with overlap —
repeating the tail of each chunk at the head of the next — which reduces missed splits
at the cost of redundant embeddings, index bloat, and duplicate retrievals that must be
deduplicated downstream. Chonk avoids the bad split in the first place. Chunks flush at
heading-level transitions. Tables split at row boundaries, lists at item boundaries,
prose at sentence boundaries. Plain-text documents without headings can have headers
promoted automatically from questions and short phrases before chunking begins. Because
every chunk corresponds to a complete unit of meaning — never a partial paragraph, never
a half-table — retrieved chunks are precise: what comes back is exactly the passage that
answers the query, with no leading or trailing noise from an adjacent context window.

**Graph-guided retrieval with completeness gates.** `EnhancedSearch` supports three
retrieval modes: vector-first (seed → structural → entity → cluster → community
expansion), graph-first (RelationshipIndex traversal with vector reranking), and global
(community summary search). After the top-k cohort is selected, a completeness gate
checks whether query entities are present in the results; if any are missing, the search
expands further until they appear or the budget is exhausted. Cohort reranking combines
relevance, priority, and marginal coverage into a composite score, so the final answer
context is both relevant and non-redundant.

---

## The problem with naive chunking

Most RAG pipelines embed raw chunk content and nothing else. This works when every
chunk contains enough distinctive vocabulary to describe itself. That is a narrow
special case.

In practice, almost every document type you want to retrieve from has repeating
structure:

- **Technical documentation** — every function reference has `Parameters`, `Returns`,
  `Raises` sections with the same words across every function in every library
- **Code** — every `__init__`, test setup, error handler, and config block shares
  vocabulary across the entire codebase
- **Contracts** — indemnification, limitation of liability, and governing law clauses
  are assembled from a shared clause library; the boilerplate is identical across
  every agreement
- **Regulatory filings** — every 10-K has the same Items in the same order; every
  company's Controls and Procedures section (Item 9A) is near-verbatim identical
- **Clinical protocols** — ECOG performance criteria, RECIST endpoints, and organ
  function thresholds appear word-for-word across hundreds of trials
- **Academic papers** — Abstract, Introduction, Methods, Results, Discussion; the
  heading hierarchy is fixed by convention

When sections share vocabulary, the embedding vectors for chunks from different
documents — or different sections of the same document — are indistinguishable.
Retrieval returns the wrong chunk, from the wrong document, for the wrong reason.

There are two places to inject this context. The document name and section path are
known at chunk time, so they can be prepended to the text that gets embedded:

```
[techcorp_msa_2024 > Limitation of Liability]

IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY…
```

Or they can be injected at answer generation time — included in the prompt context
alongside the retrieved chunk rather than baked into the embedding itself.

Which approach is better depends on the embedding model. Models that were trained on
structured prefixes can use them as a disambiguation signal and produce meaningfully
different vectors. Models that weren't may treat the prefix as noise, diluting the
content signal rather than sharpening it. Chonk supports both strategies and lets you
choose: `enrich_chunks()` handles embedding-time injection, and `AnswerGenerator` /
`PromptBuilder` handle generation-time injection.

---

## What Chonk does

Chonk is a document chunking and contextual enrichment pipeline. It:

1. **Fetches** documents from local disk, HTTP/HTTPS, S3, FTP, SFTP, or any custom
   source (SharePoint, Confluence, Google Drive, Notion). Built-in `WebCrawler` and
   `DirectoryCrawler` discover documents recursively from a root URI; custom crawlers
   plug in via the `Crawler` protocol.
2. **Extracts** text from PDF, DOCX, XLSX, PPTX, HTML, Markdown, plain text, SEC
   EDGAR inline XBRL, Python, TypeScript/JavaScript, Java, or any custom format
3. **Chunks** into semantically coherent pieces — never breaking mid-paragraph,
   keeping tables and lists atomic, tracking the full heading hierarchy
4. **Enriches** each chunk: sets `embedding_content` to
   `"[doc_name > section_path]\n\n<content>"` before it reaches your embedding model

The original `content` field is never modified. `embedding_content` is what you
embed. Everything downstream — your embedding model, vector store, retrieval
logic — is unchanged.

---

## Installation

Core (no optional dependencies):
```bash
pip install chonk
```

With specific extras:
```bash
pip install "chonk[http]"       # HTTP/HTTPS transport
pip install "chonk[s3]"         # Amazon S3 transport
pip install "chonk[sftp]"       # SFTP transport
pip install "chonk[pdf]"        # PDF extraction
pip install "chonk[docx]"       # DOCX extraction
pip install "chonk[xlsx]"       # XLSX extraction
pip install "chonk[pptx]"       # PPTX extraction
pip install "chonk[yaml]"       # YAML file extraction
pip install "chonk[odf]"        # ODF/ODS/ODT extraction
pip install "chonk[storage]"    # DuckDB vector store
pip install "chonk[cluster]"    # Entity clustering (scikit-learn)
pip install "chonk[leiden]"     # Leiden community detection (igraph + leidenalg)
pip install "chonk[parquet]"    # Parquet/Arrow/Feather structured file support
pip install "chonk[code]"       # Python/TS/JS/Java code chunking (stdlib only, no extra packages)
pip install "chonk[full]"       # Everything
```

---

## Quick start

```python
from chonk import DocumentLoader

loader = DocumentLoader()   # context_strategy="prefix" is the default

# Local file, URL, or raw bytes — same interface
chunks = loader.load("/path/to/report.pdf")
chunks = loader.load("https://example.com/docs/api.html")
chunks = loader.load_bytes(pdf_bytes, name="report", doc_type="pdf")
chunks = loader.load_text("Paragraph one.\n\nParagraph two.", name="notes")

for chunk in chunks:
    # chunk.content           — original text, unchanged (for display, storage)
    # chunk.embedding_content — "[doc > section]\n\n..." (embed this)
    # chunk.section           — ["Item 1A", "Risk Factors"] (list of heading levels)
    # chunk.document_name     — "aapl_10k_2025" (metadata, not in content)
    embed(chunk.embedding_content)
```

The section path and document name appear in both `chunk.section` / `chunk.document_name`
(as metadata, for filtering and display) **and** in `embedding_content` (as text, for
disambiguation during vector search). These are separate concerns. The metadata is
always present; `embedding_content` is what makes retrieval accurate.

---

## Pipeline

```
URI
 │
 ▼
Transport  (Local / HTTP / S3 / FTP / SFTP / custom)
 │  fetch(uri) → FetchResult(data: bytes, detected_mime, source_path)
 ▼
Extractor  (PDF / DOCX / XLSX / PPTX / HTML / Markdown / EDGAR / custom)
 │  extract(data) → str
 ▼
chunk_document(name, text, min_chunk_size, max_chunk_size)
 │  → list[DocumentChunk]  (content, section, document_name, breadcrumb,
 │                           embedding_content already set when include_breadcrumb=True)
 ▼
enrich_chunks(chunks, strategy="prefix")   [optional; re-enriches or enriches
 │  → list[DocumentChunk]                   chunks produced without a loader]
 ▼
Your embedding model / vector store
```

`chunk_document` sets `embedding_content` directly when `include_breadcrumb=True`
(the default). `DocumentLoader` calls `chunk_document` with `include_breadcrumb=True`
whenever `context_strategy` is not `None`, then passes the result through
`enrich_chunks` for the final enrichment step. Calling `enrich_chunks` on already-
enriched chunks is idempotent — it replaces `embedding_content` using the stored
`breadcrumb` field.

---

## API reference

### `DocumentChunk` fields

| Field | Type | Description |
|---|---|---|
| `document_name` | `str` | Source document name |
| `content` | `str` | Chunk text — original, never modified |
| `section` | `list[str]` | Ordered list of enclosing heading labels (`["Methods", "Table 1"]`) |
| `chunk_index` | `int` | Zero-based position within the document |
| `source_offset` | `int \| None` | Character offset of chunk start in source text |
| `source_length` | `int \| None` | Character length of chunk content |
| `embedding_content` | `str \| None` | Set by `chunk_document` / `enrich_chunks()` — embed this, not `content` |
| `chunk_type` | `str` | `"document"`, `"db_table"`, `"db_column"`, `"api_endpoint"`, `"api_graphql_query"`, `"api_graphql_mutation"`, `"api_graphql_type"` |
| `breadcrumb` | `str \| None` | Pre-formatted breadcrumb string (`"[doc > section]"`) used by `enrich_chunk` |
| `paragraph_continuation` | `bool` | True when this chunk is a continuation of a split paragraph |
| `source` | `str` | Origin class: `"document"`, `"schema"`, `"api"`, or `"community"` |
| `source_detail` | `dict \| None` | Format-specific navigation metadata — see [Source detail](#source-detail) |

### `chunk_document`

```python
chunk_document(
    name: str,
    content: str,
    min_chunk_size: int,
    max_chunk_size: int,
    overflow_margin: float = 0.15,
    include_breadcrumb: bool = True,
    include_doc_name: bool = True,
    promote_headings: bool = False,
    promote_questions: bool = True,
    promote_short_phrases: bool = True,
    max_header_words: int = 6,
    max_header_chars: int = 80,
    structural_levels: list[tuple[str, int]] | None = None,
    toc_proximity: int = 300,
    max_breadcrumb_chars: int | None = None,
    overlap_chars: int = 0,
) -> list[DocumentChunk]
```

Splits a document into semantically coherent chunks bounded by `min_chunk_size`
and `max_chunk_size`. Respects paragraph boundaries, keeps tables and lists atomic,
tracks heading hierarchy in `section`, and splits large blocks with continuation
markers (`[TABLE:start]` / `[TABLE:cont]` / `[TABLE:end]`, etc.).

When `include_breadcrumb=True` (default), sets `embedding_content` and `breadcrumb`
on every returned chunk.

### `enrich_chunk` / `enrich_chunks`

```python
enrich_chunk(chunk: DocumentChunk, strategy: str = "prefix") -> DocumentChunk
enrich_chunks(chunks: list[DocumentChunk], strategy: str = "prefix") -> list[DocumentChunk]
```

Returns new chunk(s) with `embedding_content` set. Never mutates input.

All three accepted strategy values (`"prefix"`, `"inline"`, `"breadcrumb"`) produce
the same output format:

```
[doc_name > Ancestor > Section]

<content>
```

The breadcrumb is taken from `chunk.breadcrumb` when present. When absent it is
rebuilt from `chunk.document_name` and `chunk.section`. If neither is available,
`embedding_content` is set to `chunk.content` unchanged.

The `strategy` parameter is validated and accepted for forward compatibility; it
does not currently alter the output format.

### `DocumentLoader`

```python
DocumentLoader(
    min_chunk_size: int = 600,
    max_chunk_size: int = 1500,
    overflow_margin: float = 0.15,
    context_strategy: str | None = "prefix",
    include_doc_name: bool = True,
    extra_transports: list | None = None,
    extra_extractors: list | None = None,
)
```

Full pipeline: fetch → extract → chunk → enrich. `context_strategy=None` disables
enrichment and is only useful as a baseline for benchmarking.

#### Core load methods

- `loader.load(uri, name=None)` — fetch from any supported URI (local path, `http(s)://`, `s3://`, `ftp://`, `sftp://`). Delegates to `load_structured_file()` for `.parquet`, `.arrow`, `.feather`, `.csv`, `.jsonl`, `.ndjson`.
- `loader.load_bytes(data, name, doc_type="auto", source_path=None)` — extract from raw bytes; `doc_type="auto"` detects from `source_path`.
- `loader.load_text(text, name)` — chunk and enrich pre-extracted text.

#### Structured / metadata loaders

- `loader.load_query(connection_url, query, name, params=None)` — execute a SQL query via SQLAlchemy, render results as a markdown table, and chunk. `connection_url` is any SQLAlchemy URL (e.g. `"sqlite:///data.db"`).
- `loader.load_schema(tables)` — build N+1 `DocumentChunk` objects per `TableMeta`: one `"db_table"` chunk summarising the table plus one `"db_column"` chunk per column.
- `loader.load_api(endpoints)` — build N+1 `DocumentChunk` objects per `EndpointMeta`: one `"api_endpoint"` / `"api_graphql_query"` / `"api_graphql_mutation"` / `"api_graphql_type"` chunk plus one `"api_field"` chunk per field.
- `loader.load_structured_file(path_or_uri, name=None)` — infer schema from `.csv`, `.json`, `.jsonl`/`.ndjson`, `.parquet`, `.arrow`, or `.feather` and delegate to `load_schema()`. Returns the same N+1 layout.
- `loader.load_imap(uri, *, include_attachments=False, limit=None)` — fetch messages from an IMAP mailbox. `uri` format: `imaps://user:pass@host/MAILBOX`. Each message becomes a separate set of chunks; attachments are optionally extracted inline.
- `loader.load_from_db(connection, queries)` — execute one or more SQL queries or views against a live DB connection and load the results as document chunks. Each query becomes a separate document. `queries` is a `dict[name, sql]` or `list[tuple[name, sql]]`. The same connection used for schema introspection and NER data vocab can be passed here — no second authentication needed.

#### Crawl methods

- `loader.load_site(url, max_pages=50, max_depth=3, same_domain=True, exclude_patterns=None, include_pattern=None, crawler=None)` — crawl a website and load all discovered HTML pages.
- `loader.load_directory(path, extensions=None, recursive=True, max_files=1000, crawler=None)` — load all documents in a local directory or S3 prefix. Code extensions (`.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.java`) are included by default.
- `loader.load_crawl(uri, crawler=None, **crawler_kwargs)` — generic entry point; `load_site` and `load_directory` are convenience wrappers.

---

## Code indexing

Python, TypeScript/JavaScript, and Java files are first-class document types. The
extractor converts source structure into Markdown headings — classes become `#`, methods
become `##` — then feeds the result through the standard chunker unchanged.

```python
loader = DocumentLoader()

# Single file
chunks = loader.load("src/auth/token.py")

# Entire repository
chunks = loader.load_directory("./src")

for chunk in chunks:
    # chunk.section        — ["TokenService", "validate"]
    # chunk.source_detail  — {"line_start": 42, "line_end": 67, "symbol": "TokenService.validate"}
    embed(chunk.embedding_content)
```

Docstrings and JSDoc/Javadoc comments are emitted as plain-text paragraphs before the
code fence, giving the embedding model natural language to anchor on. Import blocks are
collected under a single `## Imports` heading so they do not dilute the method-level
chunks.

### `ImportCrawler`

Discovers transitive dependencies starting from a seed file, bounded by depth or
repository root. Use it to index a module and everything it imports without having to
enumerate files manually.

```python
from chonk.transports import ImportCrawler

crawler = ImportCrawler(root_path="./src", max_depth=3)
uris = crawler.crawl("src/auth/token.py")   # seed + all reachable imports within src/

chunks = loader.load_crawl("src/auth/token.py", crawler=crawler)
```

`root_path` prevents crawling outside the repository. `max_depth=0` returns only the
seed file; `max_depth=1` adds its direct imports. Bare module specifiers (`react`,
`java.util.*`) are skipped — only local relative imports and resolvable package paths
are followed.

---

## Live DB queries as document chunks

`load_from_db()` materialises SQL queries or views against an existing DB connection
and feeds the results through the standard CSV extractor pipeline. This closes the loop
on "find everything we know about customer X": structured docs, schema metadata, NER
entity vocab, and now live relational data — all in one retrieval index.

```python
from chonk import DocumentLoader

loader = DocumentLoader()

# Same engine used for load_schema() and NerPipeline.add_from_db()
chunks = loader.load_from_db(
    connection=engine,          # SQLAlchemy Engine, Connection, URL string, or any .execute()
    queries={
        "customer_360":  "SELECT * FROM v_customer_360",
        "open_invoices": "SELECT customer_name, amount, due_date "
                         "FROM invoices WHERE status = 'open'",
        "risk_flags":    "SELECT * FROM vw_customer_risk_flags",
    },
)
# Each key becomes a separate document_name in the returned chunks
```

Accepts the same connection types as `NerPipeline.add_from_db()` and
`load_schema()` — pass the same object, no second authentication needed.

Queries that return zero rows produce no chunks (empty result sets are silently
skipped). Column names become the CSV header row and appear in the chunk text.

### Chunk provenance

Every chunk produced by `load_from_db()` carries a `source_detail` dict with
enough information to locate the original rows — without any credentials:

```python
chunk.source_detail == {
    "db_dialect": "postgresql+psycopg2",  # SQLAlchemy drivername
    "db_host":    "prod-db.internal",
    "db_port":    5432,
    "db_name":    "warehouse",
    "query":      "SELECT * FROM v_customer_360",
    "row_start":  12,   # 1-based data row index (header = row 0)
    "row_end":    47,
}
```

`row_start` / `row_end` are 1-based indices into the data rows (excluding the
header). Re-run the query with `LIMIT`/`OFFSET` or `WHERE rownum BETWEEN` to
retrieve exactly the rows the chunk came from.

`db_host`, `db_port`, and `db_name` are omitted when not present in the connection
(e.g. SQLite in-memory). Credentials (`username`, `password`) are never included.

Plain CSV files loaded via `loader.load()` also receive `row_start` / `row_end`
(but not the DB fields).

### `SqlQueryTransport`

`load_from_db()` is a convenience wrapper around `SqlQueryTransport`, which can be
used directly when you need fine-grained control or want to integrate with the
transport registry:

```python
from chonk.transports import SqlQueryTransport

transport = SqlQueryTransport(engine)
result = transport.fetch("sqlquery://customer_360",
                         sql="SELECT * FROM v_customer_360")
# result.data          — UTF-8 CSV bytes
# result.detected_mime — "text/csv"
# result.source_path   — "customer_360"
```

---

## Source detail

Every `DocumentChunk` already carries `section` (heading breadcrumb path) and, for
text-based formats, `source_offset` / `source_length` (byte offsets into the extracted
text). `source_detail` adds **format-specific navigation on top of those** — the kind of
sub-location that breadcrumbs alone cannot express.

How much additional detail is useful varies by format:

| Format | What breadcrumbs give you | What `source_detail` adds |
|--------|--------------------------|--------------------------|
| Markdown | Heading path | Char offsets (already in `source_offset`/`source_length`) — `source_detail` is `None` |
| XLSX | Sheet + named range (if any) | `sheet`, `row_start`, `row_end` — useful when a sheet has thousands of rows |
| DOCX | Heading section path | `paragraph_start`, `paragraph_end`, `section` — pin-points exact paragraph range |
| PDF | None (no heading extraction) | `page` or `page_start` / `page_end` |
| PPTX | None | `slide`, `shape` |
| Python | Class / method heading | `line_start`, `line_end`, `symbol` (e.g. `"MyClass.run"`) — IDE jump-to-line |
| TypeScript / JavaScript | Class / function heading | `line_start`, `line_end`, `symbol` |
| Java | Class / method heading | `line_start`, `line_end`, `symbol` |

`source_detail` is **not embedded** — it lives on the chunk as metadata only. Use it to
build source links, IDE jump-to-definition integrations, or citation footnotes.

Custom extractors populate `source_detail` by implementing `annotate()` (see
[Extending Chonk](#extending-chonk)).

---

## Extending Chonk

### Custom extractor

All extractors implement three methods: `can_handle`, `extract`, and `annotate`.

```python
class MyExtractor:
    def can_handle(self, doc_type: str) -> bool:
        return doc_type == "myformat"

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        return data.decode()  # return plain text for chunk_document()

    def annotate(self, chunks: list, data: bytes, source_path: str | None = None) -> list:
        # Optionally stamp chunk.source_detail with navigation metadata.
        # Called by the loader after chunking; return chunks unchanged if not needed.
        return chunks

loader = DocumentLoader(extra_extractors=[MyExtractor()])
chunks = loader.load_bytes(raw_bytes, name="doc", doc_type="myformat")
```

`annotate()` receives the chunks produced by `chunk_document()` and the original raw
bytes. It runs after chunking and before enrichment. The no-op implementation (just
`return chunks`) is correct for formats where source navigation is not meaningful.

### Custom transport

```python
from chonk.transports._protocol import FetchResult

class SharePointTransport:
    def can_handle(self, uri): return uri.startswith("sharepoint://")
    def fetch(self, uri, **kwargs):
        data = ...  # your fetch logic
        return FetchResult(data=data, detected_mime="text/html", source_path=uri)

loader = DocumentLoader(extra_transports=[SharePointTransport()])
chunks = loader.load("sharepoint://site/document")
```

---

## Storage

Requires `pip install "chonk[storage]"`.

```python
import numpy as np
from chonk import DocumentLoader
from chonk.storage import Store

loader = DocumentLoader()
chunks = loader.load("report.pdf")

embeddings = your_model.encode([c.embedding_content for c in chunks])

with Store("index.duckdb", embedding_dim=1024) as store:
    store.add_document(chunks, np.array(embeddings, dtype=np.float32))

    results = store.search(your_model.encode(["primary outcomes"])[0], limit=5)
    for chunk_id, score, chunk in results:
        print(f"{score:.3f}  [{chunk.document_name} > {chunk.section}]")
        print(f"       {chunk.content[:80]}")
```

### `Store.search()` parameters

```python
store.search(
    query_embedding,          # np.ndarray shape (dim,)
    limit: int = 5,
    query_text: str | None = None,       # enables hybrid FTS + vector search
    namespaces: list[str] | None = None, # restrict to named partitions
    chunk_types: list[str] | None = None # restrict to specific chunk types
) -> list[tuple[str, float, DocumentChunk]]
```

Returns a list of `(chunk_id, score, DocumentChunk)` tuples. Each returned
`DocumentChunk` includes `source_detail` (page, slide, paragraph, line numbers,
etc.) when the original document was annotated — `source_detail` is persisted
to the DuckDB store and round-trips through search.

### `chonk.storage` exports

| Name | Description |
|---|---|
| `Store` | High-level facade: vector search + relational entity storage over a single DuckDB file |
| `DuckDBVectorBackend` | Low-level DuckDB VSS + FTS backend implementing `VectorBackend` |
| `RelationalStore` | SQLAlchemy-based relational store for entity and chunk-entity link tables |
| `VectorBackend` | `typing.Protocol` — implement to plug in an alternative vector store |

---

## Additional top-level exports

### `VersionedRef[T]`

Thread-safe versioned holder for any object. Supports immediate swap (`ref.update(value)`) and stage-then-promote (`ref.stage(value)` / `ref.promote()`) patterns. The `version` counter increments on every `update` or `promote` call.

```python
from chonk import VersionedRef

ref: VersionedRef[list] = VersionedRef(initial=[])
ref.update(new_list)   # atomic swap, version++
```

### `EnhancedSearch`

4-dimensional cohort assembler for retrieval. Assembles candidates across vector similarity (seed), structural adjacency (next/prev/parent chunks), entity adjacency (`EntityIndex`), and cluster adjacency (`ClusterMap`). Each dimension can be independently enabled or disabled.

```python
from chonk import EnhancedSearch
search = EnhancedSearch(store, entity_index=ei, cluster_map=cm)
results: list[ScoredChunk] = search.search(query_embedding, k=5)
```

### Generation layer

| Name | Description |
|---|---|
| `AnswerContext` | Retrieval output (ranked `ScoredChunk` list, query, optional community context) packaged for prompt assembly |
| `PromptBuilder` | Assembles a prompt string from an `AnswerContext` |
| `Answer` | Structured answer returned by `AnswerGenerator` |
| `AnswerGenerator` | Sends the assembled prompt to an LLM and returns an `Answer` |

### Graph layer

| Name | Description |
|---|---|
| `SVOTriple` | Subject-verb-object triple extracted from text |
| `SVOExtractor` | Extracts `SVOTriple` objects from chunk content |
| `RelationshipIndex` | In-memory index of SVO triples keyed by entity |
| `RelationshipIndexBuilder` | Builds a `RelationshipIndex` from chunks, optionally using an LLM |
| `LLMClient` | Thin protocol / adapter for LLM calls used by the graph and generation layers |
| `VERB_SET` | Default set of relation verbs used by `SVOExtractor` |

### Community layer

| Name | Description |
|---|---|
| `CommunityIndex` | Index of Leiden communities over the entity co-occurrence graph |
| `CommunityIndexBuilder` | Builds a `CommunityIndex` from an `EntityIndex` and `ClusterMap` |
| `CommunitySummarizer` | Generates natural-language summaries of each community via an LLM |

### NER / vocabulary layer

#### Three-layer NER pipeline

`NerPipeline` runs up to three matcher layers and merges the results. Schema
and data vocab both suppress overlapping spaCy hits.

| Layer | What it matches | Normalisation |
|-------|----------------|---------------|
| **Schema vocab** (`db_enrich=True`) | Table/column/API identifier terms | camelCase / snake_case / SCREAMING_SNAKE → prose |
| **Data vocab** (`add_from_db` / `add_entities`) | Actual values from your DBs: customer names, employee names, counterparties, tickers | Verbatim, case-insensitive |
| **spaCy NER** (`spacy_entities=True`) | Generic statistical NER | — |

```python
from chonk.ner import NerPipeline, SpacyLabel

pipeline = NerPipeline(
    db_enrich=True,         # match schema/column/API identifier terms
    spacy_entities=True,    # run spaCy NER
    spacy_entity_types=[SpacyLabel.ORG, SpacyLabel.PERSON, SpacyLabel.GPE],
)

# --- Schema vocab: identifier names (normalised) ---
pipeline.add_tables(table_meta_list)             # TableMeta objects
pipeline.add_sql(open("schema.sql").read())      # raw DDL
pipeline.add_chunks(loader.load_schema(tables))  # chunks from load_schema()

# --- Data vocab: real values from your DB (reuse existing connection) ---
pipeline.add_from_db(
    engine,   # SQLAlchemy Engine, Connection, or URL string
    queries={
        "customer":     "SELECT name      FROM customers   WHERE active = true",
        "employee":     "SELECT full_name FROM employees",
        "counterparty": "SELECT name      FROM counterparties",
    },
    row_limit=50_000,   # max rows per query (default 10 000)
)

# --- Data vocab: plain list (CRM export, config file, spreadsheet, etc.) ---
pipeline.add_entities(["Acme Corp", "Globex"], entity_type="customer")

# Run against document chunks
matches = pipeline.match(chunk.content)

# Or index a whole batch at once
pipeline.run_on_chunks(chunks, entity_index)
```

`add_from_db` rules:
- Each SQL query **must return exactly one column** — `ValueError` is raised otherwise.
- Nulls are dropped and values are deduplicated before being added.
- Data values are matched verbatim (case-insensitive) — no camelCase splitting.
  "Acme Corp" matches "Acme Corp", not "acme corp".

Schema identifiers are normalised: `firstName`, `first_name`, and `FIRST_NAME`
all match the prose form `"first name"`, surfacing connections between your
relational data model and the documents that reference it.

#### Primitives (for custom scenarios)

| Name | Description |
|---|---|
| `NerPipeline` | Three-layer NER pipeline: schema vocab + data vocab + spaCy |
| `SchemaVocabBuilder` | Builds matchers from `TableMeta`, SQL DDL, `load_schema()` chunks, DB queries, or plain lists |
| `VocabularyMatcher` | Rule-based entity matcher using a user-supplied vocabulary |
| `EntityIndex` | Index mapping entity IDs to the chunks they appear in |
| `SpacyMatcher` | spaCy-backed NER matcher — `entity_types` restricts to a label subset |
| `SchemaMatcher` | Matches schema-derived terms (table/column names) against chunk text |
| `normalize_schema_term` | `"firstName"` / `"first_name"` / `"FIRST_NAME"` → `"first name"` |
| `merge_matches` | Merge vocab and spaCy hits; vocab wins on span overlap |
| `CooccurrenceMatrix` | Tracks entity co-occurrence counts across chunks |
| `ClusterMap` | Maps entity IDs to cluster IDs after `cluster_entities()` |
| `SpacyLabel` | Enum of the 18 standard spaCy English entity labels |
| `ALL_SPACY_LABELS` | Default label list used when `entity_types` is `None` |

---

## Demos

```bash
# Synthetic multi-section docs (ops reports, product catalogs, incident logs)
python demo/contextual_vs_naive.py

# Real SEC EDGAR 10-K filings (AAPL, MSFT, AMZN, CRM) — requires internet
python demo/edgar_demo.py

# Real ClinicalTrials.gov Phase 2/3 oncology protocols — requires internet
python demo/clinicaltrials_demo.py

# Python standard library documentation — requires internet
python demo/python_docs_demo.py
```

---

## Ethics & sourcing

- **Sustainably harvested tokens** — no embeddings computed, stored, or billed without your consent
- **Free-range paragraphs** — chunks never split mid-sentence against their will
- **Cage-free section breadcrumbs** — every chunk knows where it came from
- **Conflict-free text extraction** — no third-party cloud APIs consulted without consent
- **Non-GMO transport layer** — no monkey-patching of built-ins
- **Fair trade** — MIT licensed, attribution appreciated

---

## License

MIT
