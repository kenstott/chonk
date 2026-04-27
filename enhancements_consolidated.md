# ChunkyMonkey Enhancements — Consolidated

**Already implemented (excluded):** Enhanced semantic similarity search (NER + cluster + cohort assembly), cross-encoder reranking, breadcrumb-weighted embeddings, entity reference expansion, community detection / Louvain, community coherence gating.

**Excluded as operational (not feature enhancements):** Constat migration compatibility.

All remaining items follow the coherence-gating pattern: measure a proxy signal, gate on a threshold, default to inactive.

---

## Tier 1 — Retrieval Quality (direct benchmark impact)

### 1. Passage Redundancy Pruning

**Signal:** Pairwise cosine similarity among retrieved chunks before LLM context assembly.

**Mechanism:** If two retrieved chunks exceed a similarity threshold (cosine > 0.92), keep only the higher-scoring one. Applied after reranking, before prompt assembly.

**Why:** Duplicate or near-duplicate passages waste context window tokens and can confuse the LLM by presenting the same information as if it were corroborating evidence. Always-on quality gate — pruning a duplicate never removes unique information.

**Tunable parameter:** `redundancy_threshold` (default: 0.92)

**Priority:** Implement first — no downside, zero false-positive risk.

---

### 2. Lane Entry Sensitivity Gating

**Signal:** Cosine similarity of each candidate chunk to the query, computed before pool admission.

**Mechanism:** In multi-lane retrieval (semantic + entity expansion + diversity expansion), apply a per-lane minimum similarity threshold before a chunk enters the reranker pool. Entity expansion (≥ 0.45) and diversity expansion (≥ 0.40) are slightly looser than semantic (already sorted by similarity).

**Why:** The reranker is not immune to pool dilution. A pool flooded with weakly-relevant chunks raises the average score floor — genuinely relevant chunks get crowded out when the reranker takes a fixed top-N. The entry gate is cheap (cosine, already computed during retrieval) and protects pool composition before the cross-encoder runs.

**Tunable parameters:** `lane_entity_min_sim` (default: 0.45), `lane_diversity_min_sim` (default: 0.40)

---

### 3. Retrieval Concentration Gating (auto-trigger NER gap-fill)

**Signal:** Fraction of top-K retrieved chunks from the same grouping unit.

**Mechanism:** If top-K chunks exceed a concentration threshold (e.g., >60% from the same unit), automatically trigger entity-reference expansion (NER gap-fill) to broaden retrieval coverage.

**Why:** Semantic similarity retrieval is biased toward the dominant topic of the query. Concentration indicates locally relevant but globally incomplete retrieval. NER gap-fill treats query entities as first-class retrieval keys, independent of embedding proximity.

**Grouping unit design:** The optimal grouping unit is a property of the **data source**, not inferable from chunk content at query time. Different sources have fundamentally different natural boundaries:

| Source type | Natural boundary | Rationale |
|---|---|---|
| PubMed / journal articles | Document | Each article is an independent unit; 10–30 chunks |
| Novels / long-form books | Section (chapter) | Entire book is one document; chapter is the meaningful unit |
| Legal regulations (CFR, statutes) | Section | A regulation has hundreds of sections; section is the citable unit |
| SEC filings (10-K, 10-Q) | Section | Filing has distinct named sections (Item 1A, MD&A, etc.) |
| Emails / IMAP messages | Document | Each message is an independent unit |
| Internal policies | Subsection | Policies have numbered clauses; subsection is the operative unit |

**Full design (requires Enhancement #10 — Document Registry):** Each source registered with a `concentration_unit` field at ingest time. At query time: `chunk → document_id → source_uri → transport config → concentration_unit`. The grouping key is then computed per-chunk using the source-appropriate level. This is the only design that handles a heterogeneous corpus correctly.

**Fallback (without registry — grid tests):** `"adaptive"` mode uses the breadcrumb as a proxy: if `chunk.section` is non-empty, group by `document_name::section[0]`; otherwise group by `document_name`. Best available heuristic when source type is unknown. Under-performs on deep hierarchies (subsection-level sources) and on corpora mixing short articles with long books where breadcrumb depth is inconsistent.

**Complements:** Reranking (quality control); concentration gating is coverage control.

**Tunable parameters:** `concentration_threshold` (default: 0.6); `concentration_unit` (`"document"` | `"section"` | `"subsection"` | `"adaptive"`, default: `"adaptive"` for grid, per-source config for production)

---

### 4. NER-Based Community Topic Labels with Embedding-Based Synonym Merging

**Signal:** Named entities extracted from chunk texts within each community, deduplicated via entity embeddings, computed at index time.

**Mechanism:** Replace `_top_terms()` word-frequency labels with two steps:

1. **NER extraction:** Run the existing NER pipeline over all chunk texts in the community (already executed at index time for entity expansion — no additional model load).
2. **Embedding-based synonym merging:** Cluster entity mentions by cosine similarity of their entity embeddings (already stored in `entity_embeddings` table). Within each cluster, select the most-frequent surface form as canonical. Top-K canonical forms by cluster size become the community label.

This collapses synonyms and abbreviation variants without any domain knowledge — `"DLBCL"` ↔ `"diffuse large B-cell lymphoma"`, `"Mr. Darcy"` ↔ `"Fitzwilliam Darcy"`, `"EU"` ↔ `"European Union"` — driven entirely by the embedding space, domain-agnostic by construction.

**Why:** Word-frequency labels are dominated by common domain terms (`"cancer, treatment, cells"` for the entire oncology corpus) with no discriminative signal. Entity-based labels name specific concepts a community is about.

**Observed failure mode:** `nobc_ner_rerank_community_gated_full` — 667-chunk Medical community generated the constant label `"cancer, treatment, cells, blood, care"` injected identically into every Medical query context.

**Zero additional cost:** NER output and entity embeddings already computed at index time.

**Tunable parameters:** `community_label_strategy` (`"term_freq"` | `"ner_embedding"`, default: `"ner_embedding"`); `community_label_entity_types`; `community_label_top_k` (default: 5); `community_label_synonym_threshold` (default: 0.85)

---

## Tier 2 — Adaptive Pipeline (routing and calibration)

### 5. Query Complexity Routing

**Signal:** Query embedding entropy or syntactic parse depth (clause count, entity count, question type classifier).

**Mechanism:** Simple factoid queries (low entropy, single entity) skip community context injection and NER expansion entirely. Multi-hop or comparative queries (high entropy, multiple entities, "relationship between X and Y" patterns) activate the full pipeline.

**Why:** Community context adds noise to simple fact lookups. Routing avoids this while reducing cost per query on easy questions.

**Tunable parameter:** `query_complexity_threshold` (default: auto-detected via entropy percentile on first N queries)

---

### 6. Corpus Heterogeneity Score → Auto-tune sim_threshold / alpha

**Signal:** Inter-cluster variance of the full corpus embedding space, computed at index time.

**Mechanism:** Low variance (homogeneous corpus) → raise `sim_threshold` (tighter communities) and lower `alpha` (breadcrumbs less useful). High variance (heterogeneous corpus) → lower `sim_threshold` (more cross-topic edges) and raise `alpha` (structural position more informative).

**Why:** A single fixed `sim_threshold` and `alpha` can't be optimal across corpora as different as a single-topic textbook vs. a cross-agency document store.

**Tunable parameter:** `auto_calibrate` flag (default: True); overridden by explicit `sim_threshold` / `alpha` if provided.

**Complements:** Lane Entry Sensitivity Gating (#2) — homogeneous corpus → tighter lane gates; heterogeneous → looser.

---

### 7. Entity Density Threshold

**Signal:** Number of chunks in which a given entity type appears, computed at index time.

**Mechanism:** Only include entity types in the NER expansion index that appear in more than N chunks (e.g., N=5). Sparse entities (hapax legomena, OCR artifacts, one-off proper nouns) are excluded from the expansion index.

**Why:** Sparse entities in the NER index generate noisy gap-fill retrievals — they fire rarely and when they do, the retrieved chunks are often tangential.

**Tunable parameter:** `entity_min_chunks` (default: 5)

---

### 8. Structural Depth Confidence Gating

**Signal:** Per-document heading parse confidence score, computed at index time from heading depth variance and heading/content ratio.

**Mechanism:** Breadcrumb-weighted embeddings (alpha > 0) only applied to documents where heading structure is reliably detected. Flat documents (CSVs, emails, transcripts) get alpha=0 automatically; structured documents (textbooks, reports, regulations) get the configured alpha.

**Why:** Applying breadcrumb weighting to documents without meaningful heading structure injects noise into the embedding.

**Tunable parameter:** `min_heading_confidence` (default: 0.5); per-document override at index time.

---

## Tier 3 — Knowledge Graph Extension

### 9. Entity Relationships and SVO Triples

**Signal:** Subject-Verb-Object triples extracted from text via spaCy dependency parse, stored as typed relationship records.

**Mechanism:** After entity extraction, run a lightweight SVO extractor over chunks and store triples in an `entity_relationships` table. Enables graph traversal as a fifth cohort-assembly dimension beyond the current four (semantic, structural, entity, cluster).

**Why:** Builds a knowledge graph from document ingestion without an LLM graph-extraction step. Typed relationships enable query patterns that pure entity co-occurrence misses (e.g., "X acquired Y", "X reports to Y").

**Storage:** `entity_relationships(subject_entity_id, verb_category, object_entity_id, chunk_id, confidence, user_edited)`.

---

## Tier 4 — Ingest and Infrastructure

### 10. Document Registry with Source URI, Freshness, and Deduplication

**Mechanism:** New `documents` table with surrogate PK; `embeddings.document_name` replaced by `embeddings.document_id FK`. Every chunk has a canonical `source_uri` (`file://`, `s3://`, `imap://`, etc.). Transport stores `freshness_meta` JSON (mtime, etag, UID). Loader calls `check_freshness()` before re-ingest; if unchanged, skips. Supports `reload_document(chunk)` and `reconstruct_document(document_id)` from stored chunks.

**Why:** Without source URI and freshness tracking, every ingest is a full re-index. No way to detect stale chunks, avoid redundant embedding, or trace a chunk back to its source.

---

### 11. Content Hash Freshness

**Extends:** #10. Adds SHA-256 `content_hash` of extracted text to `freshness_meta`. Catches cases where mtime changes but content does not (file copy, touch) and cases where content changes without mtime update (S3 overwrites preserving timestamp). If hash unchanged, skip re-ingest regardless of timestamp.

---

### 12. Parallel Document Loading

**Mechanism:** Add `max_workers: int = 1` parameter to the loader. When `> 1`, use `ThreadPoolExecutor` to fetch and extract concurrently. Chunk and embed remain sequential per document (embedding model is not thread-safe without locking). IMAP specifically benefits from per-message streaming.

**Why:** Significant latency reduction for I/O-bound sources (S3, HTTP, IMAP).

---

### 13. Image Classification for OCR

**Mechanism:** Before choosing extraction strategy: run local OCR (Tesseract); if `word_count >= 50` AND `confidence >= 60%` → use OCR text as text-primary. Otherwise → send to LLM vision for description + tag extraction. Image tags fed into NER as pseudo-entities, making image content searchable via entity adjacency.

---

### 14. Additional Extractors

**RTF:** `striprtf` — legal and healthcare legacy systems.

**MSG:** `extract-msg` — Outlook `.msg` files. Attachments produce child `source_uri` entries under the #10 hierarchy.

**ZIP / TAR:** Container extractors — unpack and recursively dispatch each member to the appropriate extractor. Each member produces a child `source_uri`. Nested archives recursed up to configurable depth (default 2). Password-protected archives fail loudly.

---

### 15. Additional Transports

**Azure Blob Storage:** `az://container/blob-path`. Freshness via `Last-Modified` + `ETag`. `azure-storage-blob`.

**Google Cloud Storage:** `gs://bucket/object-path`. Freshness via `updated` + `etag`. `google-cloud-storage`.

**Google Drive:** `gdrive://file-id`. Freshness via `modifiedTime`. Exports Google Workspace formats to PDF or plain text. Folder traversal via crawler variant.

**SharePoint Online:** `sharepoint://tenant/site/drive/item-id`. Microsoft Graph API. OAuth 2.0 client credentials. `msal`.

**SharePoint On-Premises:** `sharepoint-onprem://host/site/item-path`. NTLM/Kerberos via `requests-ntlm`.

**Git Repository:** `git://repo-path@ref/file-path`. Freshness via commit SHA of last commit touching the file. Two modes: working tree (delegates to local transport) or bare/remote (reads blob objects via `gitpython`).

**IMAP Folder Selection:** `folders: list[str]` parameter; `LIST` command to enumerate available folders. URI scheme extended to include folder: `imap://user@host/FOLDER-NAME/<message-id>`. Covers sent mail without a separate SMTP transport.

---

### 16. Federated Libraries / Multi-Store

**Mechanism:** `UseCase("compliance", stores=[sec_store, policy_store, contract_store])` holds references to N libraries and builds aggregated indexes across them. Federated search: fan-out query to each library's HNSW index in parallel, collect top-k per library, RRF merge across libraries. Cross-library NER and cluster indexes surface connections that per-library indexes miss. Library stores are unmodified; use-case indexes are derived and disposable.

**Why:** Document collections that partition into independent libraries (SEC filings, internal policies, contracts) require a use-case aggregation layer for cross-library NER and clustering to be meaningful.

---

## Design Principles

All Tier 1 and Tier 2 features share the coherence-gating pattern:

1. Measure a proxy signal (at index time or query time)
2. Compare against a threshold
3. Activate only when the signal justifies it
4. Default to inactive (do no harm on corpora where the feature doesn't apply)

This makes the full pipeline **corpus-adaptive by construction**.

The multi-lane retrieval architecture (semantic + entity expansion + diversity expansion) layers on top: lanes pool candidates, entry gates (#2) protect pool quality, redundancy pruning (#1) deduplicates, and the reranker makes the final selection. Each concern handled at the cheapest possible stage.

---

## Grid Redesign: Deltas to Existing Results

### Existing runs that would change

All four completed full runs are affected by at least one enhancement:

| Run | All | Affected by |
|---|---|---|
| `nobc_ner_ref_rerank_full` | 0.662 | #1 (redundancy prune), #7 (entity density), #2 (lane gating) |
| `nobc_ner_rerank_community_gated_full` | 0.657 | #1, #7, #4 (community labels) |
| `nobc_ner_rerank_full` | 0.657 | #1, #7, #2 |
| `vanilla_256_rerank_full` | 0.660 | #1 only |

`vanilla_256_rerank_full` is the cleanest control — only redundancy pruning applies.

### New runs: isolation tier

| Run name | New flags vs current best | Tests |
|---|---|---|
| `nobc_ner_ref_rerank_pruned_full` | `--redundancy-threshold 0.92` | #1 alone — baseline lift, no downside |
| `nobc_ner_ref_rerank_laned_full` | `--lane-entity-min-sim 0.45` | #2 alone — pool quality control |
| `nobc_ner_ref_rerank_conc_full` | `--concentration-threshold 0.6` | #3 — conditional vs always-on NER expansion |

### New runs: community fix tier

| Run name | Adds over community run | Tests |
|---|---|---|
| `nobc_ner_ref_rerank_community_v2_full` | #4 (NER labels) + #5 (query routing) | Can community context beat ref_expansion with discriminative labels? |

### New runs: cumulative stacks

| Run name | Feature set | Hypothesis |
|---|---|---|
| `nobc_ner_ref_rerank_v2_full` | current best + #1 + #7 + #2 | Best non-community v2 stack |
| `nobc_ner_ref_rerank_community_v2_full` | above + #4 + #5 | Best community v2 stack |
| `nobc_ner_ref_rerank_v2_conc_full` | v2 stack + #3 | Conditional vs always-on NER in v2 context |

### Questions this grid answers

1. Whether `ref_expansion` (always-on NER) beats `concentration_gating` (conditional NER).
2. Whether community context recovers from its −0.005 deficit once labels are fixed (#4 + #5).
3. Whether #1 + #2 alone close the gap to vanilla (0.660) without entity expansion complexity.
4. The marginal value of each enhancement in isolation before stacking.

**Minimum new runs to answer the key questions: 3** — `pruned`, `laned`, `community_v2`.

---

## Execution Plan

### Phase 1 — Complete in-flight runs (parallel with implementation)

- **Let `bc_1100_2200_ner_rerank_full` complete.** Already 2h invested; retrieval nearly done. bc result determines whether bc-indexed runs belong in the v2 grid. LLM generation (~4-6h) runs while Tier 1 features are implemented.
- **Skip `nobc_1100_2200_ner_ref_rerank_community_full`.** Crashed, needs full restart. Tests community context with known-broken word-frequency labels — superseded immediately by `community_v2`. Not worth 6-8h.

### Phase 2 — Implement features (while bc runs)

Implement in dependency order:

**2a. Already done:**
- `--redundancy-threshold` — post-rerank cosine dedup ✅
- `--lane-entity-min-sim` — entity expansion pool gate ✅

**2b. Document Registry (prerequisite for concentration gating):**

Concentration gating requires knowing the correct grouping unit per data source. Without the registry, the implementation degrades to the `"adaptive"` breadcrumb heuristic, which is incorrect for heterogeneous corpora. The registry must be built first.

Schema additions:
```sql
CREATE TABLE documents (
    id              INTEGER PRIMARY KEY,
    source_uri      TEXT UNIQUE,
    display_name    TEXT NOT NULL,
    concentration_unit TEXT DEFAULT 'adaptive',  -- 'document'|'section'|'subsection'|'adaptive'
    freshness_meta  TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
ALTER TABLE embeddings ADD COLUMN document_id INTEGER REFERENCES documents(id);
```

For existing indexes: backfill `documents` from distinct `document_name` values in `embeddings`. Infer `concentration_unit` from corpus characteristics at backfill time (e.g., Medical articles → `"document"`, Novel chapters → `"section"`).

Add `build-registry` subcommand to `graphrag_bench.py` that backfills existing DBs and sets `concentration_unit` per document pattern.

**2c. Concentration gating (after registry):**
- `--concentration-threshold` — query-time check; grouping key read from `documents.concentration_unit` per hit
- Falls back to `"adaptive"` breadcrumb heuristic if document_id not populated

**2d. NER community labels:**
- `--community-label-strategy ner_embedding` — index-time, requires `build-community` rerun after registry is in place

### Phase 3 — Grid (9 candidates, run overnight after bc completes)

Existing grid runs (scores on 300-question subset, do not rerun):

| Grid run | avg_ac | Notes |
|---|---|---|
| `nobc_ner_ref_rerank` | 0.685 | current best baseline |
| `nobc_ner_rerank` | 0.672 | NER without ref expansion |
| `nobc_ner_rerank_community_gated` | 0.668 | community baseline |
| `bc_ner_rerank` | 0.664 | bc breadcrumbs, no ref expansion |
| `nobc_rerank` | 0.650 | rerank only, no NER |

New grid runs required (need implementation first):

| # | Grid run | New flags | Needs |
|---|---|---|---|
| 1 | `nobc_ner_ref_rerank_pruned` | `--redundancy-threshold 0.92` | ✅ Done |
| 2 | `nobc_ner_ref_rerank_laned` | `--lane-entity-min-sim 0.45` | ✅ Done |
| 3 | `nobc_ner_ref_rerank_v2` | `--redundancy-threshold 0.92 --lane-entity-min-sim 0.45` | ✅ Done |
| 4 | `nobc_ner_ref_rerank_conc` | `--concentration-threshold 0.6` | Needs 2b + 2c |
| 5 | `bc_ner_ref_rerank` | bc index + `--entity-ref-expansion` | No code change; new run only |

`nobc_ner_ref_rerank_community_v2` deferred until NER community labels (#4) implemented (requires `build-community` rerun).

Grid runs use the existing question subset (~300 questions, ~3h per run).

### Phase 4 — Full eval on top 5 grid candidates

Pick the 5 highest grid scores. Run full eval (4,072 questions, ~6h each, sequential). Expected top 5 (predicted before running):

1. `nobc_ner_ref_rerank_v2`
2. `nobc_ner_ref_rerank_pruned`
3. `nobc_ner_ref_rerank_laned`
4. `nobc_ner_ref_rerank_conc` (pending registry)
5. `bc_ner_ref_rerank`

Grid runs 1–3 can start as soon as bc completes (implementations already done). Run 4 starts after Document Registry and concentration gating are implemented. `nobc_ner_ref_rerank_community_v2` deferred until NER labels implemented.
