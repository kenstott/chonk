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

**For teams that want to expose an existing index to LLM tooling:** Chonk ships an
[MCP server](#mcp-server) (`mcp_chonk_server.py`) that speaks the
[Model Context Protocol](https://modelcontextprotocol.io/). Run it locally with
`CHONK_TRANSPORT=stdio` for individual use, or deploy it centrally with
`CHONK_TRANSPORT=http` so every user in your organisation can point Claude Desktop,
Cursor, or any MCP-compatible host at a shared index by URL — no local Python
environment, no file access needed on the client side.

Two capabilities distinguish it from simpler pipelines:

**Semantic boundary chunking.** Naive pipelines handle split boundaries with overlap —
repeating the tail of each chunk at the head of the next — which reduces missed splits
at the cost of redundant embeddings, index bloat, and duplicate retrievals that must be
deduplicated downstream. Chonk avoids the bad split in the first place. Chunks flush at
heading-level transitions, tables at row boundaries, lists at item boundaries, prose at
sentence boundaries. Plain-text documents without headings can have headers promoted
automatically from questions and short phrases before chunking begins.

Because every chunk corresponds to a complete unit of meaning — never a partial
paragraph, never a half-table — what comes back is exactly the passage that answers the
query, with no leading or trailing noise from an adjacent context window.

**Graph-guided retrieval with completeness gates.** `EnhancedSearch` supports three
retrieval modes: vector-first (seed → structural → entity → cluster → community
expansion), graph-first (RelationshipIndex traversal with vector reranking), and global
(community summary search). After the top-k cohort is assembled, a completeness gate
checks whether query entities are present. Missing ones trigger further expansion until
they appear or the budget is exhausted. Completeness, relevance, priority, and marginal
coverage are combined into a composite reranking score. The result: answer context that
is both on-topic and non-redundant.

The completeness gate is only as good as the entity vocabulary it searches for. On
high-quality vocabularies built from schema identifiers, structured files, or API
endpoints, it reliably finds the right chunks. On raw spaCy output against domain text
it will chase false positives and degrade both quality and latency. The custom vocab
layer exists specifically to make the gate production-viable — see
[NER / vocabulary layer](#ner--vocabulary-layer).

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

When sections share vocabulary, the embedding vectors are indistinguishable. Retrieval
returns the wrong chunk, from the wrong document, for the wrong reason.

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

1. **Fetches** documents from local disk, HTTP/HTTPS, S3, FTP, SFTP, or any custom
   source (SharePoint, Confluence, Google Drive, Notion). Built-in `WebCrawler`,
   `DirectoryCrawler`, and `GitHubCrawler` discover documents recursively from a root
   URI; custom crawlers plug in via the `Crawler` protocol.
2. **Extracts** text from PDF, DOCX, XLSX, PPTX, HTML, Markdown, plain text, SEC
   EDGAR inline XBRL, Python, TypeScript/JavaScript, Java, or any custom format
3. **Chunks** into semantically coherent pieces — never breaking mid-paragraph,
   keeping tables and lists atomic, tracking the full heading hierarchy
4. **Enriches** each chunk: sets `embedding_content` to
   `"[doc_name > section_path]\n\n<content>"` before it reaches your embedding model

The original `content` field is never modified. `embedding_content` is what you embed. Everything downstream — your embedding model, vector store, retrieval logic — is unchanged.

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
pip install "chonk[pgvector]"  # PostgreSQL + pgvector vector store
pip install "chonk[cluster]"    # Entity clustering (scikit-learn)
pip install "chonk[leiden]"     # Leiden community detection (igraph + leidenalg)
pip install "chonk[parquet]"    # Parquet/Arrow/Feather structured file support
pip install "chonk[code]"       # Python/TS/JS/Java code chunking (stdlib only, no extra packages)
pip install "chonk[gmail]"      # Gmail transport (google-api-python-client, google-auth-oauthlib)
pip install "chonk[full]"       # Everything
```

---

## Quick start

```python
from chonk import DocumentLoader

loader = DocumentLoader()   # enrich_context=True is the default

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
chunk_document(name, content, min_chunk_size, max_chunk_size)
 │  → list[DocumentChunk]  (content, section, document_name, breadcrumb,
 │                           embedding_content already set when include_breadcrumb=True)
 ▼
enrich_chunks(chunks)   [optional; re-enriches or enriches chunks produced without a loader]
 │  → list[DocumentChunk]
 ▼
Your embedding model / vector store
```

`chunk_document` sets `embedding_content` directly when `include_breadcrumb=True`
(the default). `DocumentLoader` calls `chunk_document` with `include_breadcrumb=True`
when `enrich_context=True`, then passes the result through `enrich_chunks` for the
final enrichment step. Calling `enrich_chunks` on already-enriched chunks is
idempotent — it replaces `embedding_content` using the stored `breadcrumb` field.

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
| `rendered_source` | `str \| None` | Per-record Markdown set by domain renderers (CWE, CVE, ATT&CK, etc.) for visualization |

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
enrich_chunk(chunk: DocumentChunk) -> DocumentChunk
enrich_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]
```

Returns new chunk(s) with `embedding_content` set. Never mutates input.

Output format:

```
[doc_name > Ancestor > Section]

<content>
```

The breadcrumb is taken from `chunk.breadcrumb` when present. When absent it is
rebuilt from `chunk.document_name` and `chunk.section`. If neither is available,
`embedding_content` is set to `chunk.content` unchanged.

### `DocumentLoader`

```python
DocumentLoader(
    min_chunk_size: int = 600,
    max_chunk_size: int = 1500,
    overflow_margin: float = 0.15,
    enrich_context: bool = True,
    include_doc_name: bool = True,
    extra_transports: list | None = None,
    extra_extractors: list | None = None,
)
```

Full pipeline: fetch → extract → chunk → enrich. `enrich_context=False` disables
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
- `loader.load_imap(uri, *, include_attachments=False, limit=None)` — fetch messages from an IMAP mailbox. Each message becomes a separate set of chunks; attachments are optionally extracted inline.

  URI format: `imap[s]://user:pass@host[:port]/MAILBOX[?search=CRITERIA&limit=N]`

  `search=` passes RFC 3501 criteria directly to the server — filtering happens before any bytes are transferred. Criteria are space-separated (implicit AND). Common values:

  | Criterion | Meaning |
  |---|---|
  | `ALL` | every message (default) |
  | `UNSEEN` / `SEEN` | unread / read |
  | `FROM addr` | sender address |
  | `SUBJECT text` | subject contains text |
  | `BODY text` | body contains text |
  | `SINCE date` | on or after date (e.g. `01-Jan-2025`) |
  | `BEFORE date` | before date |
  | `FLAGGED` | starred / flagged messages |
  | `LARGER n` / `SMALLER n` | size threshold in bytes |

  `limit=N` (also a kwarg) caps results to the N most-recent messages by UID after server-side filtering. Use Gmail/O365 app passwords — OAuth2 is not supported.

  ```python
  # Unread messages with PDF attachments from the last 90 days
  chunks = loader.load_imap(
      "imaps://me@example.com:app-pass@imap.gmail.com/INBOX"
      "?search=UNSEEN%20SINCE%2001-Feb-2025",
      include_attachments=True,
      limit=100,
  )
  ```
- `loader.load_from_db(connection, queries)` — execute one or more SQL queries or views against a live DB connection and load the results as document chunks. Each query becomes a separate document. `queries` is a `dict[name, sql]` or `list[tuple[name, sql]]`. The same connection used for schema introspection and NER data vocab can be passed here — no second authentication needed.

#### Crawl methods

- `loader.load_site(url, max_pages=50, max_depth=3, same_domain=True, exclude_patterns=None, include_pattern=None, crawler=None)` — crawl a website and load all discovered HTML pages.
- `loader.load_directory(path, extensions=None, recursive=True, max_files=1000, crawler=None)` — load all documents in a local directory or S3 prefix. Code extensions (`.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.java`) are included by default.
- `loader.load_crawl(uri, crawler=None, **crawler_kwargs)` — generic entry point; `load_site` and `load_directory` are convenience wrappers.

### `GitHubCrawler`

`GitHubCrawler` indexes a GitHub repository — or every repo a token can reach — without cloning. It calls the GitHub REST API to get the file tree, then returns `raw.githubusercontent.com` URLs. `HttpTransport` fetches each blob on demand; no local storage is needed.

```python
from chonk.transports import GitHubCrawler
from chonk import DocumentLoader

crawler = GitHubCrawler(token="ghp_...")  # or set GITHUB_TOKEN env var
loader = DocumentLoader()
```

Set `GITHUB_TOKEN` in the environment and omit the `token` argument to avoid hardcoding credentials.

#### Full index of a single repo

```python
chunks = loader.load_crawl("https://github.com/org/repo", crawler=crawler)

# Persist the watermark — needed for incremental updates
sha = crawler.current_sha
```

`crawler.current_sha` is set after every `crawl()` call. Store it; it is the input to the next incremental run.

#### Incremental update

Pass the previously saved SHA as `since_sha`. Only files added, modified, renamed, or copied since that commit are returned.

```python
chunks = loader.load_crawl(
    "https://github.com/org/repo",
    crawler=crawler,
    since_sha=sha,
)
sha = crawler.current_sha  # update the watermark
```

When `since_sha` equals the current HEAD, `crawl()` returns an empty list immediately — nothing changed.

#### All accessible repos with `crawl_all`

`list_repos()` paginates `/user/repos` with `affiliation=owner,collaborator,organization_member`, covering personal repos, org repos, and repos shared with the token. `crawl_all()` calls `list_repos()` then `crawl()` on each, returning `(urls, current_shas)`.

```python
import json, pathlib

shas_file = pathlib.Path("github_shas.json")
since_shas = json.loads(shas_file.read_text()) if shas_file.exists() else {}

crawler = GitHubCrawler(
    repo_include=r"org/",        # only repos whose URL matches this regex
    repo_exclude=r"-archived$",  # skip repos matching this regex
)
urls, current_shas = crawler.crawl_all(since_shas=since_shas)

# Load all discovered URLs
chunks = []
for url in urls:
    chunks.extend(loader.load(url))

# Persist watermarks for the next run
shas_file.write_text(json.dumps(current_shas, indent=2))
```

Repos that fail (bad token scope, private with insufficient access) are skipped with a warning rather than aborting the entire run.

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `token` | `str \| None` | `None` | GitHub personal access token. Falls back to `GITHUB_TOKEN` env var. Public repos work without a token but are rate-limited to 60 requests/hour. |
| `extensions` | `list[str] \| None` | See below | File extensions to include. Leading `.` is optional. |
| `branch` | `str \| None` | `None` | Branch or tag to crawl. Defaults to the repo's default branch. |
| `max_files` | `int` | `2000` | Maximum files returned per repo. |
| `repo_include` | `str \| None` | `None` | Regex — only repos whose `https://github.com/{owner}/{repo}` URL matches are crawled. |
| `repo_exclude` | `str \| None` | `None` | Regex — repos whose URL matches are skipped. Applied after `repo_include`. |

Default extensions: `.md`, `.txt`, `.rst`, `.html`, `.htm`, `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.csv`, `.json`, `.xml`, `.yaml`, `.yml`, `.py`, `.pyw`, `.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.java`.

`GitHubCrawler` is exported from `chonk.transports`.

---

### `DatabaseSchemaCrawler`

`DatabaseSchemaCrawler` indexes stored procedures, functions, views, and triggers from a live database as searchable document chunks. Each object's SQL definition becomes its own chunk, with a `dbschema://` URI as the document name.

Unlike `load_schema()` — which describes table structure (column names, types, relationships) — `DatabaseSchemaCrawler` captures the actual SQL logic: the `CREATE VIEW` body, the procedure parameter list and code, trigger firing conditions. Together they cover the full picture of what a database does and how.

The class implements both the `Crawler` and `Transport` protocols. Pass the same instance as both `crawler=` and in `extra_transports=`:

```python
from chonk.transports import DatabaseSchemaCrawler
from chonk import DocumentLoader

crawler = DatabaseSchemaCrawler("postgresql://user:pass@host/db")
loader = DocumentLoader(extra_transports=[crawler])
chunks = loader.load_crawl("postgresql://user:pass@host/db", crawler=crawler)
```

`crawl()` connects to the database, fetches all matching schema objects, and caches their definitions keyed by `dbschema://` URIs. `load_crawl()` then calls `fetch()` for each URI — which reads from that cache, not the database again.

#### Supported dialects

| Dialect | Views | Procedures / Functions | Triggers |
|---------|-------|----------------------|---------|
| PostgreSQL | Yes | Yes | Yes |
| MySQL / MariaDB | Yes | Yes | Yes |
| SQL Server | Yes | Yes (`P`, `FN`, `IF`, `TF`) | Yes |
| SQLite | Yes | No (SQLite has no stored procedures) | Yes |

For dialects not in this list, `crawl()` logs a warning and indexes views only (via SQLAlchemy inspection, which works across all dialects).

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_url` | `str` | required | SQLAlchemy connection URL |
| `include_procs` | `bool` | `True` | Include stored procedures and functions |
| `include_views` | `bool` | `True` | Include views |
| `include_triggers` | `bool` | `True` | Include triggers |
| `schemas` | `list[str] \| None` | `None` | Restrict to these schema names. `None` indexes all non-system schemas. |

#### Basic usage — index everything from a PostgreSQL database

```python
from chonk.transports import DatabaseSchemaCrawler
from chonk import DocumentLoader

crawler = DatabaseSchemaCrawler("postgresql://user:pass@prod-db/warehouse")
loader = DocumentLoader(extra_transports=[crawler])

chunks = loader.load_crawl("postgresql://user:pass@prod-db/warehouse", crawler=crawler)

for chunk in chunks:
    # chunk.document_name — e.g. "VIEW: reporting.v_customer_360"
    # chunk.content       — the SQL definition, prefixed with "-- VIEW: ..."
    embed(chunk.embedding_content)
```

#### Selective indexing — procs and views only, restricted to named schemas

```python
crawler = DatabaseSchemaCrawler(
    "mssql+pyodbc://sa:pass@sqlserver/OperationsDB?driver=ODBC+Driver+18+for+SQL+Server",
    include_procs=True,
    include_views=True,
    include_triggers=False,
    schemas=["dbo", "reporting"],
)
loader = DocumentLoader(extra_transports=[crawler])
chunks = loader.load_crawl(
    "mssql+pyodbc://sa:pass@sqlserver/OperationsDB?driver=ODBC+Driver+18+for+SQL+Server",
    crawler=crawler,
)
```

#### Combined pipeline — GitHub source code + database schema

The motivating use case for `DatabaseSchemaCrawler` is cross-referencing application code against the database objects it calls. A view named `v_customer_risk` in the DB and a function called `fetch_customer_risk` in the Python codebase land in the same retrieval index. A query for "how is customer risk calculated?" pulls both.

```python
from chonk.transports import GitHubCrawler, DatabaseSchemaCrawler
from chonk import DocumentLoader

github = GitHubCrawler(token="ghp_...")
db = DatabaseSchemaCrawler("postgresql://user:pass@prod-db/warehouse")

loader = DocumentLoader(extra_transports=[db])

# Index the application source code
code_chunks = loader.load_crawl("https://github.com/org/risk-service", crawler=github)

# Index the database schema objects — views, procs, triggers
db_chunks = loader.load_crawl("postgresql://user:pass@prod-db/warehouse", crawler=db)

all_chunks = code_chunks + db_chunks
# embed and store as usual
```

`DatabaseSchemaCrawler` is exported from `chonk.transports`.

---

### `SharePointCrawler`

`SharePointCrawler` indexes a SharePoint site — document libraries, generic lists, calendar/events lists, and site pages — and produces searchable chunks from all of them. Three authentication modes cover the full range of SharePoint deployments: Azure AD for Microsoft 365, legacy Add-in auth for older cloud tenants, and NTLM for on-premises servers.

Like `DatabaseSchemaCrawler`, the class implements both the `Crawler` and `Transport` protocols. Pass the same instance as both `crawler=` and in `extra_transports=`:

```python
from chonk.transports import SharePointCrawler
from chonk import DocumentLoader

crawler = SharePointCrawler(
    site_url="https://contoso.sharepoint.com/sites/mysite",
    auth_mode="azure_ad",
    tenant_id="your-tenant-id",
    client_id="your-client-id",
    client_secret="your-client-secret",
)
loader = DocumentLoader(extra_transports=[crawler])
chunks = loader.load_crawl(
    "https://contoso.sharepoint.com/sites/mysite",
    crawler=crawler,
)
```

`crawl()` authenticates, enumerates all configured artifact types, and returns a list of `spitem://` URIs. Documents are registered as pending — not downloaded. `fetch()` downloads each document on demand when `load_crawl()` processes the URI list. Lists, calendars, and pages are serialized to text during `crawl()` and cached; they are read from that cache during `fetch()`, not re-fetched.

#### Authentication modes

**`"azure_ad"`** — Microsoft 365 / SharePoint Online with an Azure AD app registration. Uses MSAL to acquire a client-credentials token and calls the Microsoft Graph API. Requires `pip install msal`.

**`"legacy"`** — SharePoint Add-in OAuth for tenants that cannot use Azure AD app registrations. Acquires a token from the Azure ACS endpoint (`accounts.accesscontrol.windows.net`) and calls the SharePoint REST API (`/_api/`). Requires only `requests`.

**`"ntlm"`** — On-premises SharePoint Server. Authenticates with Windows NTLM credentials and calls the SharePoint REST API. Requires `pip install requests-ntlm`.

#### Artifact types

All four types are enabled by default. Pass `artifacts=` to restrict:

| Artifact | Default | How it is fetched | Content |
|---|---|---|---|
| `"documents"` | Yes | Lazily, in `fetch()` | Raw file bytes — same extractors as `loader.load()` |
| `"lists"` | Yes | During `crawl()`, cached | List item fields serialized as plain text |
| `"calendars"` | Yes | During `crawl()`, cached | Event fields serialized as plain text (Title, EventDate, EndDate, Location, Description first) |
| `"pages"` | Yes | During `crawl()`, cached | Site page HTML |

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `site_url` | `str` | required | Full URL of the SharePoint site |
| `auth_mode` | `str` | `"azure_ad"` | `"azure_ad"`, `"legacy"`, or `"ntlm"` |
| `tenant_id` | `str \| None` | `None` | Azure AD tenant ID or domain (azure_ad and legacy modes) |
| `client_id` | `str \| None` | `None` | App client ID (azure_ad and legacy modes) |
| `client_secret` | `str \| None` | `None` | App client secret (azure_ad and legacy modes) |
| `username` | `str \| None` | `None` | Windows username including domain, e.g. `DOMAIN\user` (ntlm) |
| `password` | `str \| None` | `None` | Password (ntlm) |
| `artifacts` | `list[str] \| None` | `None` | Artifact types to crawl. `None` enables all four. |
| `max_items` | `int` | `5000` | Maximum list items fetched per list |

#### Azure AD — full site crawl

Use this for any Microsoft 365 / SharePoint Online tenant where you can register an Azure AD application.

```python
from chonk.transports import SharePointCrawler
from chonk import DocumentLoader

crawler = SharePointCrawler(
    site_url="https://contoso.sharepoint.com/sites/legal",
    auth_mode="azure_ad",
    tenant_id="contoso.onmicrosoft.com",
    client_id="a1b2c3d4-...",
    client_secret="your-secret",
)
loader = DocumentLoader(extra_transports=[crawler])
chunks = loader.load_crawl(
    "https://contoso.sharepoint.com/sites/legal",
    crawler=crawler,
)

for chunk in chunks:
    embed(chunk.embedding_content)
```

#### Legacy Add-in auth

When the tenant does not support Azure AD app registrations, register a SharePoint Add-in via `appregnew.aspx` and use `auth_mode="legacy"`. The `tenant_id` field accepts either a GUID or a domain like `contoso.onmicrosoft.com` — the actual tenant GUID is read from the `WWW-Authenticate` header automatically, so the domain form works.

```python
crawler = SharePointCrawler(
    site_url="https://contoso.sharepoint.com/sites/operations",
    auth_mode="legacy",
    tenant_id="contoso.onmicrosoft.com",
    client_id="your-addin-client-id",
    client_secret="your-addin-client-secret",
)
loader = DocumentLoader(extra_transports=[crawler])
chunks = loader.load_crawl(
    "https://contoso.sharepoint.com/sites/operations",
    crawler=crawler,
)
```

#### NTLM — on-premises SharePoint Server

For SharePoint Server deployments behind corporate firewalls. Supply Windows credentials as `DOMAIN\username`.

```python
crawler = SharePointCrawler(
    site_url="https://sharepoint.corp.example.com/sites/projects",
    auth_mode="ntlm",
    username=r"CORP\svc_indexer",
    password="service-account-password",
)
loader = DocumentLoader(extra_transports=[crawler])
chunks = loader.load_crawl(
    "https://sharepoint.corp.example.com/sites/projects",
    crawler=crawler,
)
```

#### Selective artifacts — documents and pages only

Pass `artifacts=` to skip artifact types you do not need. Omitting lists and calendars is common when the site contains mostly documents and wiki pages.

```python
crawler = SharePointCrawler(
    site_url="https://contoso.sharepoint.com/sites/wiki",
    auth_mode="azure_ad",
    tenant_id="contoso.onmicrosoft.com",
    client_id="a1b2c3d4-...",
    client_secret="your-secret",
    artifacts=["documents", "pages"],   # skip lists and calendars
)
loader = DocumentLoader(extra_transports=[crawler])
chunks = loader.load_crawl(
    "https://contoso.sharepoint.com/sites/wiki",
    crawler=crawler,
)
```

`SharePointCrawler` is exported from `chonk.transports`.

---

### `GmailCrawler`

`GmailCrawler` indexes Gmail messages via the Gmail REST API. It authenticates with OAuth2 and pages through a mailbox label, returning one chunk set per message. The message subject becomes the `document_name`; the body is the plain-text content.

Like `SharePointCrawler`, the class implements both the `Crawler` and `Transport` protocols. Pass the same instance as both `crawler=` and in `extra_transports=`:

```python
from chonk.transports import GmailCrawler
from chonk import DocumentLoader

crawler = GmailCrawler(
    client_id="your-client-id",
    client_secret="your-client-secret",
    # token_path defaults to ~/.chonk/gmail_token.json
)
loader = DocumentLoader(extra_transports=[crawler])
chunks = loader.load_crawl("gmail://me/INBOX", crawler=crawler)
```

`crawl()` calls the Gmail API to list message IDs and returns `gmsg://` URIs — one per message. No message content is downloaded at this stage. `fetch()` downloads each message lazily when `load_crawl()` processes the URI list, and caches the result so a second call to `fetch()` for the same URI hits the cache.

#### First-run authentication

On the first run the browser opens for an OAuth2 consent screen (read-only Gmail scope). The resulting token is written to `token_path` (default `~/.chonk/gmail_token.json`) and reused on every subsequent run. Expired tokens are refreshed automatically without user interaction.

Run the bundled helper script once to complete the consent flow before using the crawler in a pipeline:

```bash
python scripts/gmail_auth.py
```

Credentials can also be passed via environment variables rather than constructor arguments:

```bash
export GOOGLE_EMAIL_CLIENT_ID=your-client-id
export GOOGLE_EMAIL_CLIENT_SECRET=your-client-secret
```

#### URI scheme

| URI | Mailbox |
|---|---|
| `gmail://me/INBOX` | Inbox |
| `gmail://me/SENT` | Sent mail |
| `gmail://me/DRAFTS` | Drafts |
| `gmail://me/SPAM` | Spam |
| `gmail://me/TRASH` | Trash |
| `gmail://me/ALL` | All mail |

`crawl()` returns internal `gmsg://<key>/<message_id>` URIs. These are opaque — pass them back to `fetch()` or `load_crawl()` unchanged.

#### Filtering with Gmail search queries

The `query` parameter accepts any Gmail search string. When `query` is supplied the label in the URI is ignored; the search covers all mail.

```python
# Unread messages since the start of 2025
chunks = loader.load_crawl(
    "gmail://me/INBOX",
    crawler=crawler,
    query="is:unread after:2025/01/01",
    limit=50,
)

# Messages from a specific sender
chunks = loader.load_crawl(
    "gmail://me/INBOX",
    crawler=crawler,
    query="from:alice@example.com",
    limit=100,
)
```

#### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client_id` | `str \| None` | `None` | Google OAuth2 client ID. Falls back to `GOOGLE_EMAIL_CLIENT_ID` env var. |
| `client_secret` | `str \| None` | `None` | Google OAuth2 client secret. Falls back to `GOOGLE_EMAIL_CLIENT_SECRET` env var. |
| `token_path` | `str \| Path \| None` | `~/.chonk/gmail_token.json` | Path to read/write the OAuth2 token. |
| `user_id` | `str` | `"me"` | Gmail user ID. `"me"` always refers to the authenticated account. |
| `redirect_port` | `int` | `8000` | Local port used for the OAuth2 redirect during the consent flow. |

#### Full inbox crawl

```python
from chonk.transports import GmailCrawler
from chonk import DocumentLoader

crawler = GmailCrawler(
    client_id="your-client-id",
    client_secret="your-client-secret",
)
loader = DocumentLoader(extra_transports=[crawler])

chunks = loader.load_crawl("gmail://me/INBOX", crawler=crawler, limit=200)

for chunk in chunks:
    # chunk.document_name — message subject (or gmsg:// URI if subject is absent)
    # chunk.content       — From/To/Subject/Date headers + plain-text body
    embed(chunk.embedding_content)
```

#### Sent mail

```python
chunks = loader.load_crawl("gmail://me/SENT", crawler=crawler, limit=100)
```

`GmailCrawler` is exported from `chonk.transports`.

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

## Unified DB pattern: find everything about entity X

Most enterprise knowledge lives in three places simultaneously: unstructured documents
(contracts, reports, emails, filings), relational database schema (what data exists and
how it is structured), and relational database content (the actual records). Naive RAG
pipelines index one of these. Chonk indexes all three through a single DB connection.

```
engine = create_engine("postgresql://prod-db/warehouse")

┌─────────────────────────────────────────────────────────────┐
│  1. Schema as chunks          loader.load_schema(tables)    │
│     What data exists, column names, types, relationships    │
│     chunk_type = "db_table" / "db_column"                   │
├─────────────────────────────────────────────────────────────┤
│  2. Entity vocab from DB      pipeline.add_from_db(engine)  │
│     Known entity names → NER vocab for all document types   │
│     "Acme Corp" tagged as customer in contracts, emails,    │
│     filings — linked to the same entity ID everywhere       │
├─────────────────────────────────────────────────────────────┤
│  3. Live data as chunks       loader.load_from_db(engine)   │
│     Query results materialised as searchable document chunks│
│     Provenance: db_host, db_name, query, row_start, row_end │
└─────────────────────────────────────────────────────────────┘
```

All three use the same connection object. No second authentication, no credential
duplication.

```python
from sqlalchemy import create_engine
from chonk import DocumentLoader
from chonk.ner import NerPipeline, SpacyLabel
from chonk.storage import Store

engine = create_engine("postgresql+psycopg2://prod-db/warehouse")

loader = DocumentLoader()
pipeline = NerPipeline(db_enrich=True, spacy_entities=True)

# 1. Schema chunks — index what data exists and how it is structured
schema_chunks = loader.load_schema(tables)          # from TableMeta introspection

# 2. Entity vocab — teach NER about your actual customers, employees, counterparties
pipeline.add_from_db(engine, queries={
    "customer":     "SELECT name      FROM customers   WHERE active = true",
    "employee":     "SELECT full_name FROM employees",
    "counterparty": "SELECT name      FROM counterparties",
})
pipeline.add_tables(tables)                         # schema identifiers normalised

# 3. Live data chunks — make actual records searchable
data_chunks = loader.load_from_db(engine, queries={
    "customer_360":  "SELECT * FROM v_customer_360",
    "open_invoices": "SELECT customer_name, amount, due_date "
                     "FROM invoices WHERE status = 'open'",
})

# 4. Unstructured docs — NER now links these to the same entities as the DB data
doc_chunks = loader.load_directory("./documents")
pipeline.run_on_chunks(doc_chunks, entity_index)

# Everything lands in one index (DuckDB or PgVectorBackend — same interface)
all_chunks = schema_chunks + data_chunks + doc_chunks
with Store("index.duckdb", embedding_dim=1024) as store:
    store.add_document(all_chunks, embeddings)
# or: PgVectorBackend("postgresql://prod-db/warehouse").add_chunks(all_chunks, embeddings)
```

A query for "Acme Corp payment terms" now retrieves: the contract clause (unstructured
doc), the `payment_terms` column definition (schema chunk), and the matching rows from
the invoices view (live data chunk) — all linked through the same `ent_acme_corp`
entity ID in `EntityIndex`.

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

### Domain renderers

`JsonExtractor` and `XmlExtractor` support a `Renderer` plug-in interface for
domain-specific document formats that have known schemas. Instead of falling back
to the generic key-path walk, a matching renderer takes over rendering and annotation
entirely. This co-locates all fields that belong together in a single chunk, rather
than splitting them across separate key-path sections.

#### Renderer contract

```python
class Renderer(Protocol):
    def can_render(self, source_path: str | None, obj: object) -> bool:
        """Return True if this renderer handles the parsed document object."""

    def render(self, obj: object) -> str:
        """Convert the parsed object to Markdown. H1 headings mark record boundaries."""

    def annotate(self, chunks: list[DocumentChunk], obj: object) -> list[DocumentChunk]:
        """Stamp chunk.source_detail and chunk.rendered_source after chunking."""
```

`render()` returns Markdown with one `# Heading` per logical record (one CVE, one
ATT&CK technique, one control, one trial).  `chunk_document()` splits at those
headings, so each chunk maps to a complete record or a named subsection of one.

`annotate()` receives chunks produced from the rendered Markdown and the original
parsed object.  It sets two fields on each chunk:

- **`source_detail`** — record-level metadata (IDs, scores, status) for filtering
  and citation.  Not embedded.
- **`rendered_source`** — the full per-record Markdown for that chunk's parent
  record.  Useful for visualization: render it with any Markdown viewer to see
  the complete record alongside the retrieved chunk.

#### Built-in renderers

| Renderer | Format | Source | `source_detail` keys |
|---|---|---|---|
| `CveRenderer` | NVD CVE JSON (API v2) | `JsonExtractor` | `cve_id`, `cvss_score`, `severity`, `published` |
| `AttackRenderer` | MITRE ATT&CK STIX 2.x bundles | `JsonExtractor` | `attack_id`, `name`, `tactics`, `platforms`, `is_subtechnique`, `parent_id` |
| `NistRenderer` | NIST SP 800-53 OSCAL JSON | `JsonExtractor` | `control_id`, `title`, `group` |
| `ClinicalTrialRenderer` | ClinicalTrials.gov API v2 | `JsonExtractor` | `nct_id`, `title`, `status`, `phases`, `conditions` |
| `FdaLabelRenderer` | openFDA drug label JSON | `JsonExtractor` | `application_id`, `brand_name`, `generic_name`, `manufacturer` |
| `FhirRenderer` | FHIR R4 Bundle JSON | `JsonExtractor` | `resource_type`, `resource_id`, `code`, `subject` |
| `CweRenderer` | MITRE CWE XML catalog | `XmlExtractor` | `cwe_id`, `name`, `platforms` |

All renderers are pre-registered. Pass a document as `doc_type="json"` or
`doc_type="xml"` and the right renderer is selected automatically.

```python
from chonk import DocumentLoader

loader = DocumentLoader()

# NVD CVE feed
chunks = loader.load("https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=log4j")
annotated = [c for c in chunks if c.source_detail]
print(annotated[0].source_detail)      # {"cve_id": "CVE-2021-44228", "cvss_score": 10.0, ...}
print(annotated[0].rendered_source)    # full Markdown for CVE-2021-44228

# ATT&CK STIX bundle
chunks = loader.load("https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json")
print(chunks[0].source_detail["attack_id"])   # "T1059.001"
print(chunks[0].rendered_source[:200])         # # T1059.001 PowerShell ...
```

To add a custom renderer for a new JSON format:

```python
from chonk.extractors import JsonExtractor

class MyRenderer:
    def can_render(self, source_path, obj):
        return isinstance(obj, dict) and "myKey" in obj

    def render(self, obj):
        return "\n\n".join(f"# {r['id']}\n\n{r['text']}" for r in obj["myKey"])

    def annotate(self, chunks, obj):
        for chunk in chunks:
            chunk.source_detail = {"id": "..."}
            chunk.rendered_source = "# ..."
        return chunks

loader = DocumentLoader(extra_extractors=[JsonExtractor(renderers=[MyRenderer()])])
```

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

Chonk ships two vector backends. Both implement the same `VectorBackend` protocol and
return identical `(chunk_id, score, DocumentChunk)` results.

### DuckDB (default)

Requires `pip install "chonk[storage]"`. Stores everything in a single local file.
Uses DuckDB VSS (HNSW cosine index) for vector search and DuckDB FTS (BM25) for
hybrid reranking.

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

### PostgreSQL + pgvector

Requires `pip install "chonk[pgvector]"`. Stores chunks in a PostgreSQL table with
a `vector(dim)` column. Uses pgvector's HNSW cosine index for ANN search. The right
choice when your team is already running PostgreSQL and wants the vector index in the
same managed database as the rest of your data.

```python
from chonk.storage import PgVectorBackend
import numpy as np

backend = PgVectorBackend(
    dsn="postgresql://user:pass@prod-db:5432/warehouse",
    embedding_dim=1024,
    table="chonk_embeddings",   # default; created automatically
)

backend.add_chunks(chunks, np.array(embeddings, dtype=np.float32))

results = backend.search(query_vec, limit=5, namespaces=["project_alpha"])
for chunk_id, score, chunk in results:
    print(f"{score:.3f}  {chunk.content[:80]}")

backend.close()
```

`PgVectorBackend` implements the full `VectorBackend` protocol: `add_chunks`,
`search`, `delete_by_document`, `count`, `clear`. The schema is created on
first instantiation; subsequent connections reuse it. `namespace` and `chunk_type`
filters work identically to the DuckDB backend.

### Document registry and incremental sync

`DuckDBVectorBackend` maintains a `documents` table that tracks a content
fingerprint for every indexed document.  Use it to avoid re-downloading and
re-embedding content that hasn't changed.

#### `sync_document()`

```python
from chonk.storage import sync_document

result = sync_document(backend, document_name, raw_bytes)
```

Returns a `SyncResult(action, document_name, content_hash, chunk_count,
previous_chunk_count)`.

| `action` | Meaning |
|---|---|
| `"skipped"` | Stored hash matches; index is current. Nothing was changed. |
| `"added"` | Document not previously indexed. |
| `"updated"` | Document changed; all old chunks have been deleted. |

On `"added"` or `"updated"` the caller re-embeds and calls
`register_document()` to complete the update.  `result.content_hash` carries
the hash so you don't compute it twice.

#### Three calling patterns

**1. Mutable document, no server hint** — download first, then check:

```python
raw = requests.get(url).content
result = sync_document(backend, "nvd-feed", raw)
if result.action != "skipped":
    chunks = loader.load_bytes(raw, name="nvd-feed", doc_type="json")
    embeddings = embed(chunks)
    backend.add_chunks(chunks, embeddings)
    backend.register_document("nvd-feed", result.content_hash,
                               source_uri=url, chunk_count=len(chunks))
```

**2. Server provides ETag or Last-Modified** — skip the download entirely if
the hash already matches:

```python
etag = requests.head(url).headers.get("ETag", "")
result = sync_document(backend, "attack-enterprise", content_hash=etag)
if result.action != "skipped":
    raw = requests.get(url).content
    chunks = loader.load_bytes(raw, name="attack-enterprise", doc_type="json")
    # ... embed, add_chunks, register_document ...
```

**3. Immutable / versioned URL** — the URL itself is the fingerprint:

```python
versioned_url = "https://example.com/data/cwe-4.15.xml"
result = sync_document(backend, "cwe", content_hash=versioned_url)
if result.action != "skipped":
    raw = requests.get(versioned_url).content
    # ...
```

#### Detecting deleted documents

`list_documents()` returns every registered document.  Compare against your
source list to find documents that should be removed:

```python
known = {"nvd-feed", "attack-enterprise", "cwe"}
for doc in backend.list_documents():
    if doc["document_name"] not in known:
        backend.delete_by_document(doc["document_name"])
```

`delete_by_document()` removes both the chunks and the registry entry.
`store.vector.clear()` removes everything — chunks and registry. (`Store` does not wrap `clear()` on the facade; call it on `store.vector` directly.)

### `Store.add_document()` parameters

```python
store.add_document(
    chunks: list[DocumentChunk],
    embeddings,                    # np.ndarray shape (n, dim)
    namespace: str | None = None,  # optional partition key
)
```

`namespace` partitions the index so different corpora can share a single DuckDB file and be searched independently. Pass the same value to `store.search(namespaces=[...])` to restrict retrieval to that partition. `None` means no namespace — all chunks are visible to namespace-unrestricted searches.

### `Store.search()` parameters

```python
store.search(
    query_embedding,          # np.ndarray shape (dim,)
    limit: int = 5,
    query_text: str | None = None,       # enables hybrid FTS + vector (DuckDB only)
    namespaces: list[str] | None = None, # restrict to named partitions
    chunk_types: list[str] | None = None # restrict to specific chunk types
) -> list[tuple[str, float, DocumentChunk]]
```

Returns a list of `(chunk_id, score, DocumentChunk)` tuples. Each returned
`DocumentChunk` includes `source_detail` (page, slide, paragraph, line numbers,
etc.) when the original document was annotated — `source_detail` is persisted
and round-trips through search on both backends.

`query_text` hybrid BM25 reranking is supported by `DuckDBVectorBackend` only;
`PgVectorBackend` accepts the parameter for API compatibility but performs
pure vector search.

### `Store` constructor

```python
Store(
    db_path: str | Path = ":memory:",
    embedding_dim: int = 1024,
    read_only: bool = False,
)
```

`read_only=True` opens the DuckDB file without acquiring a write lock, allowing multiple concurrent readers against the same index. Relational entity features are unavailable in read-only mode.

### `chonk.storage` exports

| Name | Description |
|---|---|
| `Store` | High-level facade: vector search + relational entity storage over a single DuckDB file |
| `DuckDBVectorBackend` | Low-level DuckDB VSS (HNSW) + FTS (BM25) backend implementing `VectorBackend` |
| `PgVectorBackend` | PostgreSQL + pgvector HNSW cosine backend implementing `VectorBackend` |
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

**Progressive adoption.** Vector-only is a valid production configuration — pass `store` with no additional arguments and every other dimension is disabled. Each layer is an upgrade you earn by building the prerequisite index:

| Layer | Prerequisite | What you gain |
|---|---|---|
| Vector seed | `Store` with embeddings | Baseline semantic retrieval |
| Structural adjacency | Chunking with section/parent metadata | Adjacent context pulled automatically |
| Entity + cluster | `EntityIndex` + `ClusterMap` (custom vocab recommended) | Completeness gate; entity-linked expansion |
| Graph communities | `CommunityIndex` (Leiden, zero LLM) | Cluster expansion guided by co-occurrence graph |
| Global summaries | `CommunitySummarizer` (opt-in LLM) | `global` mode: query answered from community summaries |

Each layer degrades gracefully to the one below it when its index is absent. You do not need to build all layers before going to production.

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

Community detection (`CommunityIndexBuilder` + Leiden algorithm) runs at index time and makes zero LLM calls. The Leiden graph partition is pure graph math. In `vector_first` mode, cluster expansion uses that partition to pull co-occurring chunks — graph-guided retrieval at zero per-query LLM cost.

`CommunitySummarizer` is a separate, opt-in step that calls an LLM once per community to produce a natural-language summary. It is only needed for `global` retrieval mode, which answers queries from community summaries rather than raw chunks. Traditional GraphRAG pipelines make this summarization pass mandatory; here it is optional and gated behind an explicit call. If you never use `global` mode, `CommunitySummarizer` never runs.

### NER / vocabulary layer

#### The problem with naive entity extraction

Generic NER models — spaCy, BERT-NER, cloud APIs — are trained on public corpora.
They reliably find people and places in news articles. They are poor at the entities
that actually matter in enterprise retrieval:

- **Schema identifiers**: `customerRiskScore`, `cpty_id`, `EFFECTIVE_DT` — your
  internal column names appear in documents ("the customer risk score is reviewed
  quarterly") but no generic model was trained on your data dictionary
- **Known entities from your databases**: "Acme Corp" is in your CRM; spaCy may or
  may not tag it as `ORG`; it will never tag it as `customer` with the right
  canonical ID to join back to your database
- **Ambiguous short names**: "Mercury" is a planet, a car brand, a record label,
  and possibly your internal code name for a project — spaCy cannot distinguish them
  without your context
- **Domain vocabulary**: drug names, ticker symbols, legal clause labels, internal
  project codes — statistical models generalise poorly to narrow domains

The result: NER links the wrong chunks to the wrong entities, or misses the link
entirely, producing entity graphs that look plausible but silently fail on real queries.

#### How Chonk solves it: three-layer NER

`NerPipeline` runs three matcher layers in order and merges the results. Both
vocabulary layers suppress overlapping spaCy hits — your known entities take
precedence over statistical guesses.

| Layer | What it matches | Source |
|-------|----------------|--------|
| **Schema vocab** | Table names, column names, API field identifiers | Your DDL, `TableMeta`, `load_schema()` chunks |
| **Data vocab** | Actual entity values: customer names, employee names, counterparties, tickers | Live DB queries or plain lists |
| **spaCy NER** | Generic statistical NER for entities not covered above | Pre-trained spaCy model |

Schema identifiers are normalised before matching: `customerRiskScore`,
`customer_risk_score`, and `CUSTOMER_RISK_SCORE` all match the prose form
`"customer risk score"`. This surfaces structural connections — a document that says
"the customer risk score" and a table column called `CUSTOMER_RISK_SCORE` are linked
through the same entity, without any manual synonym list.

Data values are matched verbatim (case-insensitive). "Acme Corp" matches "Acme Corp"
in text, and the match carries the entity type (`customer`) and a stable canonical ID
that joins back to your database.

#### Usage

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

## MCP server

`mcp_chonk_server.py` exposes Chonk search over the
[Model Context Protocol](https://modelcontextprotocol.io/) so any MCP-compatible
host (Claude Desktop, Cursor, VS Code Copilot, etc.) can query your index directly.

```bash
pip install "chonk[storage]" mcp
```

### Transports

Two transports are supported via `CHONK_TRANSPORT`:

| Value | When to use |
|-------|-------------|
| `stdio` (default) | Local / developer use. The MCP host manages the subprocess. Each user runs their own server process against a locally accessible DuckDB file. |
| `http` | Enterprise / centralised deployment. One server process shared by all users. Users connect by URL — no Python install, no file access required on the user side. |

### stdio — local use

```bash
export CHONK_DB_PATH=/data/index.duckdb
export CHONK_EMBEDDING_DIM=1024   # default if omitted
python mcp_chonk_server.py
```

Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "chonk": {
      "command": "python",
      "args": ["/path/to/mcp_chonk_server.py"],
      "env": {
        "CHONK_DB_PATH": "/data/index.duckdb",
        "CHONK_EMBEDDING_DIM": "1024"
      }
    }
  }
}
```

### http — centralised enterprise deployment

The enterprise team builds and maintains the index. End users connect to the
server by URL with a shared API key — no local Python environment needed.

```bash
export CHONK_TRANSPORT=http
export CHONK_DB_PATH=/data/index.duckdb
export CHONK_API_KEY=your-secret-key   # omit to disable auth (not recommended)
export CHONK_HOST=0.0.0.0              # default
export CHONK_PORT=8000                 # default
python mcp_chonk_server.py
# → Uvicorn running on http://0.0.0.0:8000
```

Claude Desktop config for end users (no local server process):

```json
{
  "mcpServers": {
    "chonk": {
      "url": "http://chonk.internal:8000/mcp",
      "headers": {"Authorization": "Bearer your-secret-key"}
    }
  }
}
```

All requests must carry `Authorization: Bearer <key>` when `CHONK_API_KEY` is set.
Requests with a missing or wrong key are rejected with HTTP 401 before any MCP
session is established.

### Multiple named DBs

`CHONK_DB_CONFIG` takes precedence over `CHONK_DB_PATH` and works with both
transports. Each named DB is independently searchable; `search_chunks` merges
results across all stores when no `db` parameter is supplied.

```bash
export CHONK_DB_CONFIG='{
  "main":    {"path": "/data/main.duckdb"},
  "archive": {"path": "/data/archive.duckdb", "embedding_dim": 768}
}'
```

### Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_chunks` | `query_embedding` (required), `query_text`, `limit`, `db`, `namespaces`, `chunk_types` | Hybrid vector + BM25 search. Omit `db` to search all stores and merge by score. |
| `get_chunk` | `chunk_id` (required), `db`, `include_neighbors`, `neighbor_radius` | Fetch a chunk by ID, optionally with adjacent chunks from the same document. |
| `expand_chunk_graph` | `chunk_id` (required), `db` | Stub — wire to your `EntityIndex` / `RelationshipIndex` to expand a chunk into entity and relation overlays. |

---

## Benchmark runner

`demo/graphrag_bench.py` runs the full GraphRAG retrieval benchmark against a stratified question corpus.

**Additional requirements:**

```bash
# S3-compatible storage (Cloudflare R2 used in this project)
pip install "chonk[s3]"   # boto3

# rclone — sync large result files (checkpoints, run DBs, embeddings) to/from R2
brew install rclone
# Configure R2 remote named "chonk":
#   rclone config create chonk s3 provider Cloudflare \
#     access_key_id <key> secret_access_key <secret> \
#     endpoint https://<account>.r2.cloudflarestorage.com
```

Large files (`work/results/*.jsonl`, `work/data/runs/*.duckdb`, `work/data/*.npy/npz`) are not tracked in git. Sync them with:

```bash
rclone sync work/results r2:chonk/results
rclone sync work/data    r2:chonk/data
```

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
