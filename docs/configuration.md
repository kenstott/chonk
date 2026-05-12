# Configuration reference

Chonk has three configuration layers. They are applied in this order: hardcoded defaults in the library, then a TOML file (benchmark CLI only), then CLI flags. Each layer overrides only the keys it sets; the others fall through.

**Library API.** Pass arguments directly to constructors. No file on disk, no global state. This is the right layer for application code and tests.

**TOML config file.** Used exclusively by `demo/graphrag_bench.py`. A config file is loaded with `--config path/to/file.toml`, deep-merged over the hardcoded defaults, and then CLI flags override any key that was explicitly passed. A config file may declare `extends = "base.toml"` to inherit from a parent — chains up to depth 5.

**CLI flags.** Override any TOML value. Only explicitly-passed flags win; a flag that was not passed leaves the TOML value in place.

---

## Data sources

Chonk fetches documents through **Transport** and **Crawler** objects. Transports handle single URIs; crawlers discover and enumerate many URIs from a root. Both implement the same `fetch(uri) → FetchResult` protocol and are used by `DocumentLoader`.

### Available transports and crawlers

| Type | Class | URI pattern | Key constructor args |
|---|---|---|---|
| `directory` | `DirectoryCrawler` | local path / `file://` | `extensions`, `recursive`, `max_files`, `exclude_dirs` |
| `github` | `GitHubCrawler` | `https://github.com/org/repo` | `token`, `branch`, `extensions`, `max_files`, `repo_include`, `repo_exclude` |
| `web` | `WebCrawler` | `https://…` | `max_pages`, `max_depth`, `same_domain`, `exclude_patterns`, `include_pattern` |
| `db_schema` | `DatabaseSchemaCrawler` | SQLAlchemy URL | `connection_url`, `include_procs`, `include_views`, `include_triggers`, `schemas` |
| `sharepoint` | `SharePointCrawler` | SharePoint site URL | `site_url`, `auth_mode`, `tenant_id`, `client_id`, `client_secret`, `artifacts`, `max_items` |
| `gmail` | `GmailCrawler` | `gmail://…` | `client_id`, `client_secret`, `token_path`, `user_id` |
| `s3` | `S3Transport` | `s3://bucket/…` | boto3 env credentials |
| `http` | `HttpTransport` | `https://…` (single fetch) | — |
| `ftp` | `FtpTransport` | `ftp://…` | — |
| `sftp` | `SftpTransport` | `sftp://…` | — |
| `sql_query` | `SqlQueryTransport` | `sqlquery://<name>` | `connection` (SQLAlchemy), `sql` passed to `fetch()` |

`DatabaseSchemaCrawler` also populates chunks with `chunk_type` values (`db_table`, `db_column`, etc.) that `SchemaVocabBuilder.add_chunks()` uses to build schema-aware NER vocabulary. Set `[index.features] schema_vocab = true` when indexing database or API sources.

### TOML source declarations

Each `[[source]]` block in the config declares one data source. Multiple sources are processed in order and merged into a single index.

```toml
# Local filesystem
[[source]]
type       = "directory"
uri        = "/path/to/docs"
extensions = [".md", ".txt", ".pdf"]
recursive  = true
max_files  = 1000

# GitHub repository
[[source]]
type       = "github"
uri        = "https://github.com/myorg/myrepo"
branch     = "main"
extensions = [".py", ".md"]

# Database schema (also enables schema_vocab NER)
[[source]]
type             = "db_schema"
uri              = "postgresql://user:pass@host/db"
include_views    = true
schemas          = ["public"]

# SharePoint site
[[source]]
type       = "sharepoint"
uri        = "https://myorg.sharepoint.com/sites/mysite"
auth_mode  = "azure_ad"
artifacts  = ["documents", "lists", "pages"]

# Gmail inbox
[[source]]
type   = "gmail"
uri    = "gmail://inbox"
query  = "label:important after:2024/01/01"
limit  = 500

# Web crawl
[[source]]
type        = "web"
uri         = "https://docs.example.com"
max_pages   = 100
max_depth   = 3
same_domain = true

# SQL query result
[[source]]
type = "sql_query"
uri  = "sqlquery://my_report"
sql  = "SELECT title || ' ' || body AS content FROM articles"

# namespace is optional on any [[source]] block.
# Chunks from this source are tagged with the namespace value.
# Use --namespaces (CLI) or retrieval.namespaces (TOML) to restrict queries to
# specific namespaces at retrieval time.
[[source]]
type      = "directory"
uri       = "/path/to/internal-docs"
namespace = "internal"

[[source]]
type      = "github"
uri       = "https://github.com/myorg/public-wiki"
namespace = "public"
```

### Library usage

```python
from chonk import DocumentLoader
from chonk.transports import DirectoryCrawler, DatabaseSchemaCrawler

loader = DocumentLoader()

# Crawl a directory
chunks = loader.load_crawl(DirectoryCrawler(extensions=[".md", ".txt"]), "/path/to/docs")

# Crawl a database schema (produces db_table / db_column chunks)
db_crawler = DatabaseSchemaCrawler("postgresql://user:pass@host/db")
schema_chunks = loader.load_crawl(db_crawler, "postgresql://user:pass@host/db")

# Single URL
chunks += loader.load("https://example.com/doc.pdf")
```

---

## Entity vocabulary

Entity vocabulary extends the NER pipeline with domain-specific names that spaCy would not recognize. These are added via `SchemaVocabBuilder` before NER runs.

There are two sources:

**Static list** — a fixed set of names provided verbatim. Matched case-insensitively at query time.

**DB query** — executes a SQL `SELECT` and adds each result row as an entity name. Useful for populating vocabularies from live databases (customer names, product SKUs, employee names, etc.).

Data values are matched verbatim — no camelCase splitting is applied. `SchemaVocabBuilder.add_entities()` and `SchemaVocabBuilder.add_from_db()` both use `build_data_matcher()` to produce a `VocabularyMatcher` that runs alongside the schema matcher; `merge_matches` gives schema hits precedence over spaCy hits, and data-matcher hits supplement the combined result.

### TOML vocab declarations

```toml
# Static entity list
[[vocab.entities]]
type        = "static"
entity_type = "customer"
names       = ["Acme Corp", "Globex Inc", "Initech"]

# DB-query populated vocabulary
[[vocab.entities]]
type        = "db_query"
entity_type = "employee"
connection  = "postgresql://user:pass@host/db"
sql         = "SELECT full_name FROM employees WHERE active = true"

[[vocab.entities]]
type        = "db_query"
entity_type = "product"
connection  = "postgresql://user:pass@host/db"
sql         = "SELECT product_name FROM products"
```

`entity_type` is the label assigned to matched spans (e.g. `"customer"`, `"employee"`, `"product"`). It is stored alongside each entity hit in `chunk_entities`.

### Library usage

```python
from chonk.ner import SchemaVocabBuilder

builder = SchemaVocabBuilder()

# Static names
builder.add_entities("customer", ["Acme Corp", "Globex Inc"])

# From a DB query
import duckdb
con = duckdb.connect("my.duckdb")
builder.add_from_db("employee", con, "SELECT full_name FROM employees WHERE active = true")

data_matcher = builder.build_data_matcher()

# Use alongside schema_matcher and spacy in NER loop
from chonk.ner import merge_matches
for chunk in loader_chunks:
    schema_hits = schema_matcher.match(chunk.content)
    data_hits   = data_matcher.match(chunk.content)
    spacy_hits  = spacy.match(chunk.content)
    combined    = merge_matches(schema_hits + data_hits, spacy_hits, source_text=chunk.content)
    entity_index.index_chunk(chunk_id, chunk.content, combined)
```

---

## Index features

These are built once against a DuckDB file and loaded at query time. None are required; each one enables an additional retrieval capability.

### NER (`build-ner`)

Runs `SpacyMatcher` over every chunk in the index, writes the results to a `chunk_entities` table, and persists them for later loading into an `EntityIndex`. The `EntityIndex` is then used by `EnhancedSearch` for entity-adjacency expansion and completeness gating.

Two optional flags extend what NER builds:

**`--with-embeddings`** — after NER, embeds all unique entity name strings and stores them in an `entity_embeddings` table. Required for `--ner-x` mode at query time, which uses ANN search over entity embeddings to find additional related entities when the literal-match expansion produces too few results.

**`--with-schema-vocab`** — before running spaCy, builds a `SchemaMatcher` from any chunks whose `chunk_type` is `db_table`, `db_column`, `api_endpoint`, `api_graphql_query`, `api_graphql_mutation`, or `api_graphql_type`. The schema matcher runs first; its hits suppress overlapping spaCy hits. Use this when your index contains schema or API chunks (from `loader.load_schema()` or `loader.load_api()`).

### SchemaMatcher and SchemaVocabBuilder

`SchemaVocabBuilder` extracts schema identifiers from several sources and compiles them into a `SchemaMatcher`. All identifiers go through `normalize_schema_term` before matching — `customerRiskScore`, `customer_risk_score`, and `CUSTOMER_RISK_SCORE` all produce the surface form `"customer risk score"` and match as the same entity.

`add_chunks()` inspects each chunk's `chunk_type` and `document_name`:

- `chunk_type in ("db_table", "db_column")` and `document_name` starts with `"schema:"` → extracts table and column names from the dotted path
- `chunk_type in ("api_endpoint", "api_graphql_query", "api_graphql_mutation", "api_graphql_type")` and `document_name` starts with `"api:"` → extracts the endpoint path
- `chunk_type == "api_field"` and `document_name` starts with `"api:"` → extracts the field name

Variants generated per term (all lowercase): normalized form, singular form (trailing `s` stripped), underscore form (if original contains `_`), and joined forms with spaces removed. First-registration wins on variant collision.

`build()` returns a `SchemaMatcher` for schema/API/column terms. `build_data_matcher()` returns a `VocabularyMatcher` for data-value terms (customer names, employee names, etc.) that were added via `add_entities()` or `add_from_db()`. Data values are matched verbatim — no camelCase splitting.

### Community index (`build-community`)

Embeds chunk breadcrumbs (heading vectors), computes a weighted average with content vectors (`alpha` controls heading weight, default `0.2`), builds a cosine similarity graph over chunks, and runs Louvain community detection. Community summaries are stored as `community_summary` chunks. Required for `mode="global"` retrieval.

### SVO triples (`build-svo`)

`SVOExtractor` calls an LLM once per chunk to extract subject-verb-object triples. The LLM is constrained to a fixed verb vocabulary (`VERB_SET`). Valid triples are added to a `RelationshipIndex` and written to a `svo_triples` table via `RelationshipIndex.save_to_db()`.

At query time, `RelationshipIndex.load_from_db()` loads the table into memory. This is required for `mode="graph_first"`.

The `cmd_build_svo` function in `graphrag_bench.py` runs extraction concurrently across chunks using a `ThreadPoolExecutor`.

### Namespace filtering

Each `[[source]]` block accepts an optional `namespace` string. Chunks ingested from that source are tagged with that value in the `embeddings` table.

At query time, pass `namespaces` as a list to restrict results to matching rows:

- **TOML**: set `retrieval.namespaces = ["internal", "public"]`
- **CLI**: `--namespaces internal public`
- **Library**: pass `namespaces=["internal"]` to `EnhancedSearch.search()` or `store.vector.search()`

When `namespaces` is `None` (the default), all namespaces are searched — fully backwards-compatible.

---

## Retrieval modes

Pass `mode=` to `EnhancedSearch.search()`.

### `vector_first` (default)

Seed pool = `k × seed_pool_multiplier` (default `3`). From the seed:

1. Structural expansion pulls the previous and next chunk from each seed chunk's document.
2. Entity expansion looks up all entity IDs present in the seed+structural pool, then fetches up to `entity_expansion_top_n` (default `3`) chunks per entity from `EntityIndex`. When `lane_entity_min_sim` is set, entity-linked chunks below that cosine similarity threshold are dropped before entering the pool.
3. Cluster expansion adds cluster-adjacent chunks, budget-limited to `cluster_budget` (default `2 × k`).
4. Greedy MMR selects the final top-k using a composite score: `rw × relevance + pw × priority + cw × coverage`, where `coverage = relevance - λ × max_sim_to_selected`.

After selection, if `entity_ref_expansion=True` and query entities are missing from the result text, an additional search loop fetches chunks for the missing entities.

### `graph_first`

Requires `build-svo` to have run (and a `RelationshipIndex` loaded). Requires either `query_entities` passed directly to `search()`, or a `query_ner_fn` set on the `EnhancedSearch` constructor.

Steps: run NER on query text → traverse `RelationshipIndex` 1-hop forward (`get_objects`) and backward (`get_subjects`) for each query entity → collect chunks linked to related entities via `EntityIndex` → augment with vector seeds → rerank with `_select_cohort`.

Falls back to `vector_first` silently when prerequisites are absent (no `relationship_index`, no `entity_index`, or NER produces no entity hits).

### `global`

Searches only chunks with `chunk_type = "community_summary"`. Requires `build-community` to have run. Returns the top-k community summary chunks ranked by vector similarity.

---

## Key retrieval parameters

These are constructor arguments on `EnhancedSearch` unless noted otherwise.

| Parameter | Default | Description |
|---|---|---|
| `seed_pool_multiplier` | `3` | Seed pool size = `k × multiplier` |
| `entity_expansion_top_n` | `3` | Max chunks fetched per entity in entity expansion |
| `cluster_budget` | `2 × k` (resolved at call time) | Max cluster-adjacent candidates |
| `lambda_diversity` | `0.3` | MMR redundancy penalty weight |
| `relevance_weight` | `0.5` | Composite score weight for relevance |
| `priority_weight` | `0.2` | Composite score weight for source priority |
| `coverage_weight` | `0.3` | Composite score weight for marginal coverage |
| `lane_entity_min_sim` | `None` | Drop entity-linked chunks below this cosine similarity |
| `entity_ref_expansion` | `False` | Enable post-selection gap-fill for missing entities |
| `entity_ref_expansion_k` | `20` | Total expansion pool size for entity-ref gap-fill |
| `entity_ref_expansion_per_k` | `None` | Chunks fetched per missing entity (overrides k÷n split) |
| `entity_ref_expansion_min_sim` | `None` | Drop expansion hits below this cosine similarity |
| `structural_expansion` | `True` | Enable prev/next chunk expansion |
| `entity_expansion` | `True` | Enable entity-adjacency expansion |
| `cluster_expansion` | `True` | Enable cluster-adjacency expansion |

`top_k` and `fetch_k` (in `graphrag_bench.py`) correspond to the `k` argument to `search()` and the upstream reranker pool size, not `EnhancedSearch` constructor args.

Source priority constants (used in composite scoring, not configurable):

| Provenance | Priority |
|---|---|
| `seed` | 1.0 |
| `structural` | 0.9 |
| `entity_adjacent` | 0.7 |
| `cluster_adjacent` | 0.5 |

---

## Generation variants

These are features of `graphrag_bench.py`, not the library itself.

### Plain generation

Default. Retrieves context, sends to LLM, returns the text answer.

### `--sr` (structured response)

Instructs the LLM to return JSON with three fields: `answer`, `key_claims`, and `evidence_used`. No additional retrieval. The structured output is useful as a chain-of-thought signal and produces parseable evidence citations.

### `--srr` (structured response with coverage check)

SR plus a coverage-check loop (up to 2 rounds). After the first structured response, the key entities in the query are embedded and compared against the `evidence_used` field using cosine similarity (threshold `0.35`). Entities not covered trigger `_srr_gap_fill`, which retrieves additional chunks for each uncovered entity and regenerates. The loop runs at most 2 rounds.

`--srr` requires the embedding model at query time (to compute entity–evidence coverage scores). A cheaper model can be specified for coverage checks only via `--srr-model` and `--srr-provider`.

---

## TOML config (benchmark CLI)

`demo/graphrag_bench.py` accepts `--config path/to/file.toml`. The canonical base config is at `work/configs/base.toml`; per-run configs are in `work/configs/runs/`. Both are tracked in git.

### Schema

```toml
# Extend a parent config — path relative to this file
extends = "base.toml"   # optional

# Data sources — zero or more; processed in order
[[source]]
type      = "directory"
uri       = "/path/to/docs"
namespace = "internal"   # optional

[[source]]
type      = "github"
uri       = "https://github.com/myorg/myrepo"
namespace = "public"     # optional

# Entity vocabulary — zero or more; extends spaCy NER
[[vocab.entities]]
type        = "static"
entity_type = "customer"
names       = ["Acme Corp", "Globex Inc"]

[[vocab.entities]]
type        = "db_query"
entity_type = "employee"
connection  = "postgresql://user:pass@host/db"
sql         = "SELECT full_name FROM employees WHERE active = true"

[index]
out_dir            = "work"
db_name            = "chunkymonkey_nobc_1100_2200.duckdb"
embed_model        = "BAAI/bge-large-en-v1.5"
spacy_model        = "en_core_web_sm"
min_chunk          = 1100
max_chunk          = 2200
embed_content_only = true

[index.features]
ner            = true
ner_embeddings = true
schema_vocab   = false
community      = true
svo            = false

[rerank]
enabled  = false
provider = "local"
model    = "BAAI/bge-reranker-large"

[retrieval]
top_k                = 5
fetch_k              = 50
search_mode          = "vector_first"
enhanced             = false
entity_ref_expansion = false
lane_entity_min_sim  = null
redundancy_threshold = null
cluster              = false
vanilla              = false
namespaces           = null   # optional list of namespace strings to restrict retrieval

[retrieval.community]
enabled       = false
min_coherence = 0.5

[gen]
provider    = "openai"
model       = "gpt-4o-mini"
temperature = 0.0

[sr]
enabled = false

[srr]
enabled  = false
provider = null
model    = null

[eval]
judge       = "gpt-4o-mini"
rpm         = 8000
batch_size  = 20
concurrency = 50
nan_limit   = 136
```

### `extends` chain

A config may declare `extends = "relative/path/to/parent.toml"`. The parent is loaded first, then the child is deep-merged over it. Chains resolve recursively up to depth 5. Keys absent from the child fall through from the parent unchanged; keys present in the child override the parent at any nesting level.

### Resolution order

Hardcoded defaults (constants at the top of `graphrag_bench.py`) < TOML config file < CLI flags.

`_apply_config()` applies TOML values only when the corresponding CLI flag was not explicitly set. This means passing `--top-k 10` on the command line beats a `top_k = 5` in TOML, but omitting `--top-k` lets TOML win.

### Batch runs

```bash
python demo/graphrag_bench.py run-all --config-dir work/configs/
```

`run-all` discovers every `.toml` file in the directory and runs each one sequentially as a separate `run` invocation.

---

## Library usage example

```python
import numpy as np
from chonk import Store, EnhancedSearch
from chonk.ner import (
    EntityIndex,
    SchemaVocabBuilder,
    SpacyMatcher,
    merge_matches,
)

# ── Index phase (run once) ────────────────────────────────────────────────
loader_chunks = [...]   # DocumentChunk list from DocumentLoader
embeddings    = np.array([...], dtype=np.float32)  # shape (n, 1024)

with Store("my.duckdb", embedding_dim=1024) as store:
    store.add_document(loader_chunks, embeddings)

# ── Build NER index (run once, write to DB) ───────────────────────────────
schema_chunks = [c for c in loader_chunks
                 if c.chunk_type in ("db_table", "db_column")]
builder = SchemaVocabBuilder()
builder.add_chunks(schema_chunks)
schema_matcher = builder.build()

spacy = SpacyMatcher(model="en_core_web_sm", strip_numeric=True)
entity_index = EntityIndex()

for chunk in loader_chunks:
    schema_hits = schema_matcher.match(chunk.content)
    spacy_hits  = spacy.match(chunk.content)
    combined    = merge_matches(schema_hits, spacy_hits, source_text=chunk.content)
    # chunk_id must match the ID used when calling store.add_document()
    entity_index.index_chunk(chunk_id, chunk.content, combined)

entity_index.recompute_scores()

# ── Query phase ───────────────────────────────────────────────────────────
query_ner_fn = lambda text: [m.display_name for m in spacy.match(text)]

with Store("my.duckdb", embedding_dim=1024, read_only=True) as store:
    search = EnhancedSearch(
        store,
        entity_index=entity_index,
        query_ner_fn=query_ner_fn,
        lane_entity_min_sim=0.45,
        entity_ref_expansion=True,
        entity_ref_expansion_k=20,
    )
    results = search.search(
        query_vec,          # np.ndarray shape (1024,)
        k=10,
        query_text="...",
        mode="vector_first",
    )
    for sc in results:
        print(sc.score, sc.chunk.content[:80])
```

All `EnhancedSearch` constructor parameters and their defaults are listed in the [key retrieval parameters](#key-retrieval-parameters) table above.

`graph_first` mode additionally requires:

```python
from chonk.graph import RelationshipIndex

# At index time, after build-svo:
rel_index = RelationshipIndex.load_from_db(duckdb_connection)

search = EnhancedSearch(
    store,
    entity_index=entity_index,
    relationship_index=rel_index,
    query_ner_fn=query_ner_fn,
)
results = search.search(query_vec, k=10, query_text="...", mode="graph_first")
```
