# Experiment Ideas

Goal: beat traditional GraphRAG (deterministic, incremental, no LLM extraction at index time).
Leaderboard target: >0.661 overall (G-reasoner), starting from vanilla_rag_rerank baseline of 0.652.

---

## Strategy 1: Smart Boundaries (pure chunking improvement)

Isolates the value of section-aware, paragraph-respecting chunk splits over fixed-size vanilla chunks.
No breadcrumbs anywhere — content-only embedding, content-only generator context.

| Experiment | Embed | Generator context | Status |
|---|---|---|---|
| `vanilla_rag_v2` | content (256-tok fixed) | content | done |
| `contextual_plain_nobc` | content only | content only | pending (needs nobc index) |

**Key question**: does smarter boundary alone improve over vanilla?

---

## Strategy 2: Smart Boundaries + Breadcrumb

Breadcrumb = `[Doc > Section > Subsection]` prepended to content.
Hypothesis: disambiguates generic section headings at embedding or generation time.
Requires empirical tuning — chunk size and embedding model dimensions affect the dilution/enrichment tradeoff.

### 2a. Breadcrumb in embedding only

| Experiment | Embed | Generator context | Rerank | Status |
|---|---|---|---|---|
| `contextual_plain` | bc+content | content | no | done |
| `contextual_plain_rerank` | bc+content | content | yes | running |

**Hypothesis**: bc adds discriminating signal for ambiguous chunks; may dilute for large chunks with rich content. Rerank may recover retrieval loss from bc dilution.

### 2b. Breadcrumb in generator only

| Experiment | Embed | Generator context | Status |
|---|---|---|---|
| `contextual_plain_nobc` | content only | content | no | running (eval) |
| `contextual_plain_nobc_rerank` | content only | content | yes | running |
| `contextual_plain_bc_only` | content only | bc+content | no | pending |

**Hypothesis**: bc helps generator produce more accurate answers (section context); no embedding dilution.

### 2c. Breadcrumb in both

| Experiment | Embed | Generator context | Status |
|---|---|---|---|
| `contextual_plain_bc` | bc+content | bc+content | pending |

### 2d. Embedding model + chunk size interaction

Hypothesis: bc-in-embedding only pays off when (a) chunks are large enough that bc tokens are a small fraction, and (b) embedding model has enough dimensions to carry both signals without dilution.

| Experiment | Model | Dims | Chunk size | bc in embed |
|---|---|---|---|---|
| baseline | BGE-large | 1024 | 256 tok (vanilla) | no |
| current | BGE-large | 1024 | 400–1200 chars | yes |
| planned | text-embedding-3-large | 3072 | 400–1200 chars | yes |
| planned | text-embedding-3-large | 3072 | 400–1200 chars | no |

---

## Strategy 3: Chunk Search → Entity/Cluster Expansion → Assembly

Vector retrieval as entry point, expanded via NER + cluster graph. Deterministic and incremental.

**Pipeline**:
1. Embed query → vector search → top-K candidate chunks
2. (Optional) Rerank candidates
3. NER on retrieved chunks (or query) → extract entities
4. Entity → cluster lookup → find related entities in cluster
5. Pull additional chunks associated with cluster members
6. Assemble: retrieved chunks + cluster-expanded chunks → generator

**Open questions**:
- Rerank before expansion (cleaner top-K as seeds) or after (rerank full expanded set)?
- How many cluster hops? (1-hop vs 2-hop expansion)
- Cap on expanded chunks to avoid context bloat?

| Experiment | Retrieval | Rerank | Expansion | Status |
|---|---|---|---|---|
| `contextual_enhanced` | vector | no | NER+cluster | done (old index) |
| `contextual_rerank` | vector | yes | none | done (old index) |
| planned: `contextual_expand_postrerank` | vector | yes, before expand | NER+cluster | pending |
| planned: `contextual_expand_prererank` | vector | expand first | rerank expanded set | pending |

---

## Strategy 4: NER-Anchored Semantic Search

Retrieval starts from entity space rather than chunk space.

**Pipeline**:
1. NER on query → extract entity terms
2. Embed entity terms → semantic search against entity index (not chunk index)
3. Entity hits → associated chunks (via chunk_entities table)
4. (Optional) Cluster expansion from matched entities
5. Assemble chunks for generator

**Advantage over Strategy 3**: handles query/document terminology mismatch — finds "myocardial infarction" chunks when query says "heart attack". Entity index is smaller and faster than chunk index.

| Experiment | Entry point | Expansion | Status |
|---|---|---|---|
| planned: `ner_semantic` | entity vector search | none | pending |
| planned: `ner_semantic_expand` | entity vector search | cluster | pending |

---

## Assembly / Generator Context Experiments

Independent of retrieval strategy — what to include in the generator context window:

- **Content only** (current baseline)
- **bc + content** — section path gives generator location context
- **bc + content + NER annotations** — entity labels grounded in text
- **bc + content + cluster summary** — document-level context around retrieved chunk
- **All of the above**

These stack on top of any retrieval strategy and can be tested independently.

---

## Notes

- Reranking is assumed to always operate on the top-K clusters returned by the primary retrieval step
- All strategies are deterministic (no LLM at index time) and incrementally updatable
- Current leaderboard gap to beat: G-reasoner 0.661 overall; our best so far: vanilla_rag_rerank 0.652
- Medical and Novel subsets should be tracked separately — Novel's lack of meaningful section names limits bc value
