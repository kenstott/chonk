# Auto-Adaptation Enhancements (v2)

Six self-regulating features that activate automatically based on corpus or query signals.
Each follows the coherence-gating pattern: measure a proxy, gate on a threshold, do no harm when inactive.

---

## 1. Retrieval Concentration Gating (auto-trigger NER gap-fill)

**Signal:** Fraction of top-K retrieved chunks from the same source document.

**Mechanism:** If top-K chunks exceed a concentration threshold (e.g., >60% from one doc), automatically trigger entity-reference expansion (NER gap-fill) to broaden retrieval coverage.

**Why:** Semantic similarity retrieval is biased toward the dominant topic of the query. Concentration indicates locally relevant but globally incomplete retrieval — the retriever found a lot of one document but may have missed the document that contains the answer to the other half of the query. NER gap-fill treats query entities as first-class retrieval keys, independent of embedding proximity.

**Complements:** Reranking (quality control on what was found); concentration gating is coverage control (did you find everything?).

**Tunable parameter:** `concentration_threshold` (default: 0.6)

---

## 2. Corpus Heterogeneity Score → Auto-tune sim_threshold / alpha

**Signal:** Inter-cluster variance of the full corpus embedding space, computed at index time.

**Mechanism:** Measure embedding spread across the corpus. Low variance (homogeneous) → raise `sim_threshold` (fewer, tighter communities) and lower `alpha` (breadcrumbs less useful). High variance (heterogeneous) → lower `sim_threshold` (more cross-topic edges) and raise `alpha` (structural position more informative).

**Why:** A single fixed `sim_threshold` and `alpha` can't be optimal across corpora as different as a single-topic textbook vs. a cross-agency document store. Auto-calibration at index time means the system self-configures for the data it sees.

**Tunable parameter:** `auto_calibrate` flag (default: True); overridden by explicit `sim_threshold`/`alpha` if provided.

---

## 3. Query Complexity Routing

**Signal:** Query embedding entropy or syntactic parse depth (number of clauses, entity count, question type classifier).

**Mechanism:** Simple factoid queries (low entropy, single entity) skip community context injection and NER expansion entirely. Multi-hop or comparative queries (high entropy, multiple entities, "relationship between X and Y" patterns) activate the full pipeline.

**Why:** Community context adds noise to simple fact lookups — the LLM receives irrelevant topic scaffolding. Routing avoids this while also reducing cost per query on easy questions.

**Tunable parameter:** `query_complexity_threshold` (default: auto-detected via entropy percentile on first N queries)

---

## 4. Passage Redundancy Pruning

**Signal:** Pairwise cosine similarity among retrieved chunks before LLM context assembly.

**Mechanism:** If two retrieved chunks exceed a similarity threshold (e.g., cosine > 0.92), keep only the higher-scoring one. Applied after reranking, before prompt assembly.

**Why:** Duplicate or near-duplicate passages waste context window tokens and can confuse the LLM by presenting the same information as if it were corroborating evidence. Always-on quality gate with no false-positive risk (pruning a duplicate never removes unique information).

**Tunable parameter:** `redundancy_threshold` (default: 0.92)

---

## 5. Structural Depth Confidence Gating

**Signal:** Per-document heading parse confidence score, computed at index time from heading depth variance and heading/content ratio.

**Mechanism:** Breadcrumb-weighted embeddings (alpha > 0) only applied to documents where heading structure is reliably detected. Flat documents (CSVs, emails, transcripts) get alpha=0 automatically; structured documents (textbooks, reports, regulations) get the configured alpha.

**Why:** Applying breadcrumb weighting to documents without meaningful heading structure injects noise into the embedding. A flat doc's "headings" are often just the first line of each chunk, which carries no structural signal.

**Tunable parameter:** `min_heading_confidence` (default: 0.5); per-document override at index time.

---

## 6. Entity Density Threshold

**Signal:** Number of chunks in which a given entity type appears, computed at index time.

**Mechanism:** Only include entity types in the NER expansion index that appear in more than N chunks (e.g., N=5). Sparse entities (hapax legomena, OCR artifacts, proper nouns appearing once) are excluded from the expansion index.

**Why:** Sparse entities in the NER index generate noisy gap-fill retrievals — they fire rarely and when they do, the retrieved chunks are often tangential. Excluding them reduces index size and improves precision of gap-fill retrieval.

**Tunable parameter:** `entity_min_chunks` (default: 5)

---

## 7. Lane Entry Sensitivity Gating

**Signal:** Cosine similarity of each candidate chunk to the query, computed before pool admission.

**Mechanism:** In multi-lane retrieval (semantic + entity expansion + diversity expansion), apply a per-lane minimum similarity threshold before a chunk enters the reranker pool. Chunks below threshold are excluded regardless of which lane nominated them. Thresholds differ by lane intent: entity expansion (e.g., ≥ 0.45) and diversity expansion (e.g., ≥ 0.40) are slightly looser than semantic (which is already sorted by similarity).

**Why:** The reranker is not immune to pool dilution. A pool flooded with weakly-relevant chunks raises the average score floor — genuinely relevant chunks get crowded out when the reranker takes a fixed top-N. The entry gate is cheap (cosine, already computed during retrieval) and protects pool composition before the cross-encoder ever runs. The reranker then rank-orders quality candidates rather than filtering noise.

**Complements:** Passage Redundancy Pruning (#4, post-rerank dedup) and Corpus Heterogeneity Score (#2, which can auto-tune thresholds — homogeneous corpus → tighter gates, heterogeneous → looser).

**Tunable parameters:** `lane_semantic_min_sim` (default: implicit via top-K), `lane_entity_min_sim` (default: 0.45), `lane_diversity_min_sim` (default: 0.40)

---

## 8. NER-Based Community Topic Labels with Embedding-Based Synonym Merging

**Signal:** Named entities extracted from chunk texts within each community, deduplicated via entity embeddings, computed at index time.

**Mechanism:** Replace the current `_top_terms()` word-frequency label (top-5 non-stopword token counts) with two steps:

1. **NER extraction**: Run the existing NER pipeline (already executed at index time for entity expansion — no additional model load) over all chunk texts in the community. Collect all entity surface-form mentions.

2. **Embedding-based synonym merging**: Cluster entity mentions by cosine similarity of their entity embeddings (already stored in the `entity_embeddings` table). Within each cluster, select the most-frequent surface form as the canonical label. Top-K canonical forms by cluster size become the community label.

This collapses synonyms and abbreviation variants without any domain knowledge: `"DLBCL"` ↔ `"diffuse large B-cell lymphoma"`, `"Mr. Darcy"` ↔ `"Darcy"` ↔ `"Fitzwilliam Darcy"`, `"EU"` ↔ `"European Union"` — driven entirely by the embedding space, domain-agnostic by construction.

**Why:** Word-frequency labels are dominated by common domain terms that appear in every chunk (`"cancer, treatment, cells"` for the entire oncology corpus) and carry no discriminative signal. Entity-based labels name the specific concepts a community is actually about, making topic context injection meaningful for the LLM rather than redundant.

**Observed failure mode:** In the `nobc_ner_rerank_community_gated_full` benchmark run, the single large Medical community (667 chunks, coherence=0.629) generated the constant label `"cancer, treatment, cells, blood, care"` — injected identically into every Medical query context. Synonym-merged entity labels would produce discriminative per-community labels (cancer subtypes, specific drugs, treatment modalities) enabling genuine topic disambiguation.

**Zero additional cost:** Both the NER output and entity embeddings are already computed at index time for entity expansion. Label generation reuses these artifacts — no new models, no new passes over the corpus.

**Complements:** Corpus Heterogeneity Score (#2) — a heterogeneous corpus benefits most from precise labels; Community Coherence Gating (existing) — high-coherence communities have tighter entity clusters and more informative labels.

**Tunable parameters:** `community_label_strategy` (`"term_freq"` | `"ner_embedding"`, default: `"ner_embedding"`); `community_label_entity_types` (spaCy entity types to include, default: all); `community_label_top_k` (default: 5); `community_label_synonym_threshold` (cosine similarity for merging, default: 0.85)

---

## Design Principle

All eight features share the same architecture as the existing `--community-min-coherence` gate:

1. Measure a proxy signal (at index time or query time)
2. Compare against a threshold
3. Activate the feature only when the signal justifies it
4. Default to inactive (do no harm on corpora where the feature doesn't apply)

This makes the full pipeline **corpus-adaptive by construction**: ship one system, it tunes itself to the data.

The multi-lane retrieval architecture (semantic + entity expansion + diversity expansion) layers on top: lanes pool candidates, dedup removes duplicates, entry gates protect pool quality, and the reranker makes the final selection. Each concern handled at the cheapest possible stage.

Enhancement #8 (NER-Based Community Labels) is a zero-cost upgrade to an existing index-time step: the NER pipeline already runs for entity expansion, so routing its output to label generation adds no model overhead.

---

## Grid Redesign: Deltas to Existing Results

### Existing runs that would change

All four completed full runs are affected by at least one enhancement. None are clean baselines for v2:

| Run | All | Affected by |
|---|---|---|
| `nobc_ner_ref_rerank_full` | 0.662 | #4 (redundancy prune), #6 (entity density), #7 (lane gating) |
| `nobc_ner_rerank_community_gated_full` | 0.657 | #4, #6, #8 (community labels) |
| `nobc_ner_rerank_full` | 0.657 | #4, #6, #7 |
| `vanilla_256_rerank_full` | 0.660 | #4 only |

`vanilla_256_rerank_full` is the least affected — only redundancy pruning applies. It remains the cleanest control.

### New runs: isolation tier (one new dimension each)

Add exactly one enhancement over the current best (`nobc_ner_ref_rerank_full`, 0.662) to measure marginal lift:

| Run name | New flags vs current best | Tests |
|---|---|---|
| `nobc_ner_ref_rerank_pruned_full` | `--redundancy-threshold 0.92` | #4 alone — baseline lift, no downside |
| `nobc_ner_ref_rerank_laned_full` | `--lane-entity-min-sim 0.45` | #7 alone — does pool quality control help? |
| `nobc_ner_ref_rerank_conc_full` | `--concentration-threshold 0.6` | #1 — conditional vs always-on NER expansion |

### New runs: community fix tier

Test whether community context recovers once its known failure mode (word-frequency labels, noise on simple queries) is addressed:

| Run name | Adds over community run | Tests |
|---|---|---|
| `nobc_ner_ref_rerank_community_v2_full` | #8 (NER labels) + #3 (query routing) | Can community context beat ref_expansion with discriminative labels? |

### New runs: cumulative stacks

Bundle all validated easy wins:

| Run name | Feature set | Hypothesis |
|---|---|---|
| `nobc_ner_ref_rerank_v2_full` | current best + #4 + #6 + #7 | Best non-community v2 stack |
| `nobc_ner_ref_rerank_community_v2_full` | above + #8 + #3 | Best community v2 stack |
| `nobc_ner_ref_rerank_v2_conc_full` | v2 stack + #1 | Conditional vs always-on NER in v2 context |

### Questions this grid answers

1. Whether `ref_expansion` (always-on NER) beats `concentration_gating` (conditional NER) — currently untested.
2. Whether community context recovers from its −0.005 deficit once labels are fixed (#8 + #3).
3. Whether #4 + #7 alone close the gap to vanilla (0.660) without entity expansion complexity.
4. The marginal value of each enhancement in isolation before stacking.

**Minimum new runs to answer the key questions: 3** — `pruned`, `laned`, `community_v2`. Together they determine whether the community loss is label quality or pipeline architecture.
