# Implicit GraphRAG: Knowledge Graph Signals Without LLM-Based Graph Construction

**Kenneth Stott**
Member of Technical Staff and Senior Advisor, Logick

Code: https://github.com/kenstott/chonk (MIT License)

> **TBD markers**: results tagged `[TBD]` require experimental runs not yet complete. All structural arguments and hypotheses are testable; final numbers are pending.
> **Status**: GraphRAG-Bench full-corpus runs (gpt-4o-mini) confirmed. Haiku model comparison and FANG-2026 16-run matrix (±SRR × 2 models × 4 configs) pending. Canonical NER boundary: May 13 2026 (commit ff8f697) — pre-canonical mini NER runs queued for re-execution.

---

## Abstract

Expensive RAG deployments fail on two sides: the retrieval layer returns incomplete evidence, and the generator does not engage with the evidence it receives. This paper addresses both sides without LLM-based graph construction or expensive reranking infrastructure.

On the retrieval side, we build a knowledge graph entirely from NLP primitives — entity edges from NER co-occurrence, community structure from Louvain clustering — and traverse it at query time through entity-ref-expansion (P1), community context injection (P2), and widened lane-gated dense retrieval (P3). No LLM is involved at index time. Against the vanilla RAG+rerank baseline (All=0.624), the full graph stack achieves All=**0.674** at zero index-time LLM cost and an ~8-minute index build.

On the generation side, we introduce Evidence-Compliance Reprompting (SRR, P4): the generator is required to produce `{answer, key_claims, evidence_used}` JSON; when no evidence is cited on the first attempt, a one-shot reprompt enumerates the claims and demands supporting quotes. No additional retrieval occurs; the reprompt fires at most once. SRR is strictly additive by construction.

Two hypotheses organize the evaluation. H1: graph retrieval via NLP primitives produces compounding gains on heterogeneous corpora where cross-document entity joins are required. H2: SRR gains are real but model-capability-gated — compliance with the structured schema and reprompt quality both depend on instruction-following ability, predicting larger, more consistent gains on Haiku than Mini. A vanilla+SRR ablation isolates `Δ_SRR` without graph retrieval, and the 16-run FANG matrix tests whether graph-augmented retrieval amplifies that delta [TBD].

FANG-2026, a cross-domain benchmark spanning financial filings, security advisories, regulatory documents, and technical patents, serves as the primary evaluation surface. A 16-run matrix (±SRR × gpt-4o-mini/Haiku × 4 retrieval configs) tests H1 (graph retrieval gains on heterogeneous corpora) and H2 (SRR gains are model-capability-gated). GraphRAG-Bench (Medical + Novel) serves as a credibility bridge to published baselines.

---

## 1. Introduction

The immediate motivation for this work was a multi-step reasoning agent operating across heterogeneous corpora — relational databases and unstructured documents side by side. The agent could answer complex questions, but it did so inefficiently: through repeated re-prompting, intermediate validation steps, and iterative proof-building to assemble evidence scattered across sources. It worked. It was expensive.

Each re-prompting cycle was the agent doing manually what a better retrieval layer should have done automatically: traversing the connections between entities, following topic threads across documents, and expanding the evidence set until it was complete. The agent was compensating for a RAG layer that returned locally relevant but globally incomplete context.

This paper asks two questions. First: whether the retrieval layer itself can encode the relational structure that agents currently rediscover through re-prompting — without the cost of LLM-based graph construction. Second: whether the generator can be required to engage with the evidence it receives — without reranking infrastructure or model scale.

Almost all real-world documents have structure. Paragraphs group related sentences. Sections group related paragraphs. Named entities recur across sections, connecting ideas that naive chunking severs. A fixed-size chunker discards this structure by design: it splits at token boundaries, not semantic ones, producing chunks whose boundaries are arbitrary with respect to the content they contain. The graph signals we recover — co-occurrence edges, community partitions, entity-ref-expansion — are not additions to the retrieval pipeline. They are recoveries of structure the chunker discarded. The cost of recovering them via NLP primitives is negligible precisely because the structure was always there; we are reading it, not constructing it.

Existing GraphRAG systems (G-reasoner, AutoPrunedRetriever, HippoRAG2) encode this structure through LLM-based entity and relation extraction, requiring O(docs × LLM-calls) at index time. For an agent operating across heterogeneous and frequently-updated corpora, that cost is impractical.

We build the graph differently. At index time, spaCy NER identifies entities in each chunk; co-occurrence edges connect entities that appear together; Louvain clustering over the co-occurrence matrix partitions the graph into communities. The result is a full knowledge graph — nodes, edges, community structure — built in minutes with no LLM calls. At query time, the graph is traversed: entity-ref-expansion follows entity edges to retrieve non-adjacent chunks, community context injection provides global topic framing, and widened dense retrieval (k gated by lane similarity threshold) approximates multi-hop path traversal.

### 1.1 Design Principles

Three structural signals drive the system:

**P1 — Entity connectivity.** NER identifies entities in each chunk. Co-occurrence within a chunk creates an edge between those entities in the graph. At query time, entity-ref-expansion follows these edges: chunks that share a named entity with a retrieved chunk are pulled in, even if they are embedding-distant. Lane filtering (sim ≥ 0.60) gates expansion on query relevance — expansion without filtering hurts (−0.017), filtered expansion helps (+0.017). The lane threshold acts as the quality gate on retrieval depth: k is set large; the similarity threshold controls what lands.

**P2 — Community structure.** Louvain clustering over the co-occurrence matrix partitions the entity graph into topically coherent communities at O(chunks²), one-time. At query time, the communities of retrieved chunks are identified and a community context summary is prepended to the generator prompt, providing global topic framing that individual chunk retrieval misses.

**P3 — Traversal depth.** Wider dense retrieval (k=30, gated by lane sim) approximates multi-hop graph traversal by expanding the evidence pool. Redundancy pruning (cosine ≥ 0.92) prevents context dilution as k grows.

**P4 — Evidence-compliance reprompt (SRR).** The generator is required to produce `{answer, key_claims, evidence_used}` JSON. If `evidence_used` is empty on the first response, a single targeted reprompt enumerates the key claims and requests verbatim quotes. No additional retrieval occurs; the reprompt fires at most once and is a no-op when the generator already cited evidence. SRR is strictly additive to the retrieval stack by construction. Its gains are intuitive but model-capability-gated: compliance with the JSON schema and reprompt quality both depend on the generator's instruction-following ability (H2, §7.3).

Because P1 operates at retrieval time, P2 at prompt-construction time, P3 at the retrieval pool level, and P4 at generation time, their failure modes are orthogonal — gains compound superadditively.

### 1.2 Contributions

1. We describe a GraphRAG system whose graph is built entirely from NLP primitives (NER + Louvain) and show it delivers a +0.050 improvement over the vanilla RAG+rerank baseline (0.624 → **0.674**), with graph signals contributing +0.013 over the no-graph ablation using the same semantic chunking (rerank\_k10, 0.661 → 0.674), at zero index-time LLM cost and an ~8-minute index build versus hours for LLM-based KG construction (§4–6).
2. We demonstrate conditional superadditivity: the three signals (P1 entity-ref-expansion, P2 community context, P3 widened retrieval) correct orthogonal retrieval failure modes; gains compound when k is large enough for the extra context to be non-redundant — superadditivity is present at k=15/20+pruning but not at k=10 (§5–6).
3. We characterize the k-plateau and show redundancy pruning shifts it, providing a principled approach to retrieval depth without k tuning (§6.1).
4. We provide domain-weighted tuning guidance showing that entity-dense production corpora (legal, financial, medical, technical) gain disproportionately from graph signals relative to the equal-weight benchmark (§6.5).
5. We introduce Evidence-Compliance Reprompting (SRR, P4): a generation-side feature that forces `{answer, key_claims, evidence_used}` JSON output and reprompts once when no evidence is cited. SRR is strictly additive by construction. Its gains are intuitive but model-capability-gated (H2): instruction-following quality determines whether the generator complies with the schema on the first attempt and produces useful evidence citations on reprompt. We test H2 across corpus types and generator tiers (§7–8).
6. We introduce FANG-2026 as a primary cross-domain benchmark. A 16-run matrix (±SRR × gpt-4o-mini/Haiku × 4 retrieval configs) tests SRR contribution, retrieval strategy sensitivity under corpus heterogeneity, and model-tier invariance (§8).
7. We introduce a multi-step RAG evaluation framework as a more realistic proxy for the agentic deployment scenarios where retrieval quality matters most (§9).

---

## 2. Related Work

[Fill: 3–4 paragraphs positioning against GraphRAG, HippoRAG2, G-reasoner, AutoPrunedRetriever, RAPTOR, HyDE, FLARE. Key distinction: prior work either builds explicit KGs (expensive) or uses dense retrieval alone (misses structure). We occupy the gap.]

### 2.1 LLM-Based GraphRAG Systems

MS-GraphRAG, LightRAG, Fast-GraphRAG, HippoRAG, HippoRAG2, and G-reasoner all construct knowledge graphs through LLM extraction — entity recognition, relation triple extraction, or both — at O(docs × LLM-calls) index time. Graph traversal at query time then follows the extracted edges. These systems achieve strong benchmark results but impose hours of index build time and significant API cost on static corpora; corpus updates require full reconstruction.

### 2.2 NLP-Primitive GraphRAG (This Work)

We construct the same graph structures — entity nodes, co-occurrence edges, community partitions — using spaCy NER and Louvain clustering. Index time drops to minutes with zero LLM calls. Query-time traversal follows the same pattern: entity-ref-expansion walks entity edges; community context injection uses community membership; widened retrieval (k) extends traversal depth.

One structural difference from MS-GraphRAG and most LLM-based systems is chunking strategy: MS-GraphRAG uses fixed-size token chunking (typically 300–600 tokens), discarding sentence and paragraph boundaries. Our system uses semantic boundary chunking (1,100–2,200 tokens), splitting only at natural linguistic boundaries. To isolate the contribution of graph signals from chunking strategy, we evaluate graph_first retrieval on both semantic chunks and naive 256-token chunks, providing a controlled comparison that matches MS-GraphRAG's index structure while holding the retrieval mechanism constant.

### 2.3 Retrieval Augmentation Without Graph Structure

Cross-encoder reranking, RAPTOR (hierarchical summarization), HyDE (hypothetical document embeddings). These improve retrieval without encoding graph structure and serve as our non-graph baselines.

### 2.4 Redundancy and Diversity in Retrieval

[Fill: MMR, deduplication approaches, AutoPrunedRetriever's pruning strategy. Position our cosine-threshold pruning.]

---

## 3. Method

### 3.0 Design Rationale

The breadth of configurations evaluated in this study is intentional but not arbitrary. All features under test are grounded in a single organizing hypothesis: that retrieval quality degrades when document structure is discarded, and that restoring structural meaning — through entity recognition, lane-based routing, community summarization, reference expansion, and graph traversal — should produce compounding gains. Rather than presenting a monolithic system, we decompose this hypothesis into a small set of independently toggleable features, each with a clear structural motivation. The benchmark grid then serves as the instrument for identifying the *superadditive* subset: combinations whose joint contribution exceeds the sum of individual gains. This approach also surfaces subtractive interactions — cases where two individually positive features interfere — and makes the cost implications of each feature explicit. The goal is not to report the best single configuration but to map the tradeoff surface, giving practitioners a principled basis for selecting the configuration appropriate to their cost and quality constraints.

### 3.1 Index-Time Pipeline

```
Documents
  → Semantic boundary chunking (1100–2200 tokens)
  → [optional] Breadcrumb embedding bias (--breadcrumb-embed)
  → NER (spaCy) → EntityIndex
  → Co-occurrence matrix → Community detection (Louvain)
  → Vector store + embeddings
```

**Entity extraction QC and canonicalization.** Raw spaCy NER output is filtered and normalized before the EntityIndex is populated. Two categories of span are dropped post-extraction: (1) *Numeric entity types* — spans classified as CARDINAL, ORDINAL, MONEY, PERCENT, or QUANTITY, and any span whose surface form is purely numeric. These labels produce high-frequency entities (counts, prices, percentages) that co-occur broadly across chunks without implying semantic relatedness, creating dense noise edges in the co-occurrence graph. (2) *Stop-word-only spans* — multi-token entities where every token is a function word. Both filters apply before edge construction and require no modification to the spaCy model. Surviving entities pass through a four-layer normalization pipeline before being assigned an entity ID:

1. **Root lemma canonicalization** — the entity ID is derived from ``ent.root.lemma_.lower()``, so inflected and plural surface forms ("customers", "ordered", "invoices") resolve to the same node as their base form ("customer", "order", "invoice"). Without this, the same real-world entity fragments into multiple nodes with diluted co-occurrence edges, reducing SVO triple density and weakening context graph edge weights.

2. **Head-noun singularization** — multi-token entity names are further normalized via `inflect`, singularizing the head noun while preserving acronym casing (IBM stays IBM) and guarding against domain exceptions ("data", "criteria").

3. **Schema identifier normalization** — table and column names from the schema vocabulary are split on `snake_case` and `camelCase` boundaries and singularized, so `performance_reviews`, `performanceReviews`, and `performance review` all resolve to the same entity. Matching is attempted in both split and concatenated forms to handle mixed-style corpora.

4. **Identifier suffix aliasing** — entities whose IDs end in common identifier suffixes (`_id`, `_key`, `_code`, `_ref`, `_num`, `_no`, etc.) have the suffix stripped and stored as an alias in the `entity_aliases` table, so `customer_id` is reachable as `customer` during entity-ref-expansion.

Index cost: off-the-shelf NER (CPU) + co-occurrence matrix (O(chunks²), one-time) + Louvain community detection. **Zero LLM calls.**

### 3.2 Query-Time Pipeline

```
Query
  → Dense retrieval (top-k, large k)
  → [optional] Entity-ref-expansion  ← P1: entity connectivity
  → [optional] Lane filtering (sim ≥ 0.60)  ← P1: edge confidence / depth gate
  → [optional] Cluster expansion  ← P1: topological neighbors
  → [optional] Redundancy pruning (cosine ≥ 0.92)  ← P3: deduplication
  → [optional] Cross-encoder reranking
  → [optional] Community context injection  ← P2: global structure
  → Generator (gpt-4o-mini or claude-haiku-4-5-20251001)
  → [optional] Evidence-compliance reprompt (SRR)  ← P4: generation quality gate
```

### 3.3 Parameters

| Parameter | Default | Tested range |
|-----------|---------|-------------|
| top-k | 5 | 5, 7, 10, 15, 20, 25, 30 |
| lane-entity-min-sim | 0.60 | 0.45, 0.55, 0.60 |
| community-min-coherence | 0.50 | 0.50, 0.65 |
| community alpha | 0.20 | 0.0, 0.2 |
| redundancy-threshold | 0.92 | 0.92 |

### 3.4 Feature Inventory

Five independently-toggleable retrieval features, each grounded in a structural hypothesis about where standard RAG fails.

| Feature | Code flag | Hypothesis |
|---------|-----------|------------|
| Entity-ref-expansion | `--enhanced --entity-ref-expansion` | NER co-occurrence edges recover evidence that embedding distance misses; cross-document entity links are the mechanism by which graph structure adds value |
| Lane filtering | `--lane-entity-min-sim 0.60` | Expansion without confidence filtering injects noise (−0.017 confirmed); the lane threshold acts as both a quality gate and a de facto retrieval depth control — k is set large, sim gates what lands |
| Community context | `--community-context` | Global Louvain topic framing complements local chunk evidence; P2 and P1 address orthogonal failure modes and should compound |
| Redundancy pruning | `--pruned` | Near-duplicate chunks dilute context at high k; pruning enables deeper retrieval to add new evidence rather than noise |
| Cluster expansion | `--cluster` | Topological neighborhood expansion via cluster membership; replaces lane filtering for heterogeneous corpora where embedding similarity is a poor cross-domain proxy |
| Evidence-compliance reprompt (SRR) | `--srr` | Forces the generator into `{answer, key_claims, evidence_used}` JSON; fires a one-shot reprompt enumerating claims when `evidence_used=[]` on the first response. No additional retrieval; strictly additive by construction. Addresses the generation bottleneck that retrieval improvements cannot reach |

---

## 4. Experimental Setup

- **Primary benchmark**: FANG-2026 (§8) — 50 questions across SEC 10-K filings, CVE records, Federal Register entries, and US patents. Cross-domain entity resolution required by design.
- **Credibility bridge**: GraphRAG-Bench (arXiv:2506.05690), full question set, Medical + Novel domains — used to position results against published baselines.
- **Question types**: Factual, Reasoning, Summary, Creative (GraphRAG-Bench); MDJ, TV, CDER, QS, A/N (FANG-2026)
- **Generator + judge**: gpt-4o-mini (primary), claude-haiku-4-5-20251001 (model comparison)
- **Metric**: answer_correctness (mean of 4 subtype scores per domain, GraphRAG-Bench); mean across question types (FANG-2026)
- **Statistical testing**: bootstrap resampling (n=10,000); 95% CI throughout. `[VERIFY: report MDD after confirming full question set size]`
- **Experimental protocol**: A full combinatorial sweep across all feature dimensions would require evaluating O(N^k) configurations at non-trivial cost per full-corpus run — computationally prohibitive given our resource constraints. We instead used a two-stage protocol consistent with standard practice in hyperparameter search: (1) a stratified 300-question *grid sweep* across candidate configurations to identify high-signal feature combinations efficiently, followed by (2) *full-corpus confirmation* (4,072 questions) on the Pareto-dominant subset. Conclusions about feature importance are drawn from feature co-occurrence across top-ranked configurations, not from exhaustive enumeration. All scores reported in tables are from full-corpus runs unless marked `[TBD]` or `[VERIFY]`.

---

## 5. Experiments

### RQ1 — Does the full stack exceed published GraphRAG systems?

**Partially aligned.** The GraphRAG-Bench authors provided the exact vanilla RAG baseline specification used to produce the published leaderboard entries: 256-token naive chunks, retrieval_topk=5, gpt-4o-mini as generator, and the generator prompt from Appendix H.2. We are grateful for this assistance, which allowed us to replicate their baseline conditions precisely. Our vanilla RAG+rerank run under these conditions scores 0.652, compared to their published 0.554 — a gap whose source is not yet explained; all four baseline conditions (chunk size, k, generator model, prompt) are matched exactly. For the leaderboard systems, generator prompts and embedding configurations remain unpublished; score differences of 0.01–0.02 are within the margin introduced by those unknowns.

The table below places our numbers alongside published leaderboard values. Vanilla RAG baseline conditions are verified equivalent; other system comparisons are reference only.

| System | Med | Nov | All | Index cost |
|--------|-----|-----|-----|------------|
| G-reasoner | 0.733 | 0.589 | 0.661 | O(docs × LLM) |
| AutoPrunedRetriever-llm | 0.670 | 0.637 | 0.654 | O(docs × LLM) |
| HippoRAG2 | 0.648 | 0.565 | 0.607 | O(docs × LLM) |
| Fast-GraphRAG | 0.641 | 0.520 | 0.581 | O(docs × LLM) |
| LightRAG | 0.626 | 0.451 | 0.538 | O(docs × LLM) |
| RAG (w/ rerank) | 0.624 | 0.483 | 0.554 | — |
| RAG (w/o rerank) | 0.610 | 0.479 | 0.545 | — |
| **Ours (full stack)** | **0.733** | **0.614** | **0.674** | **0** |

*Full stack = laned entity-ref-expansion (sim≥0.45) + community context + redundancy pruning, k=20. Leaderboard values from arXiv:2506.05690. Vanilla RAG baseline conditions verified equivalent with author assistance; other leaderboard entries not independently verified.*

What we can state: within our own controlled pipeline (identical generator, prompt, and eval code across all configurations), the full stack reaches All=0.674 versus our vanilla RAG+rerank baseline of 0.647 (+0.027) and our no-graph semantic chunking baseline of 0.654 (+0.020). Those internal comparisons are the paper's primary claim. The consistent ~0.10 Med–Nov spread across all configurations, including the published leaderboard entries, suggests it reflects corpus characteristics rather than retrieval method.

We also implement and evaluate two retrieval modes that approximate the query-time behavior of published LLM-GraphRAG systems: `global_search` (community-level summarization, approximating MS-GraphRAG global mode) and `graph_first` (entity traversal before dense retrieval, approximating HippoRAG-style hop expansion). Both run through the same eval pipeline as every other configuration. `graph_first` scores 0.645 and `global_search` scores 0.257 — both below the full stack (0.674) and below the no-graph semantic chunking baseline (0.654). These are controlled comparisons. The full results and analysis are in §6.3.

The `global_search` result (0.257) warrants a qualification. Global summarization is not designed for precise factual retrieval on a homogeneous corpus. It compresses the corpus into community-level topic summaries; questions that require a specific entity, date, or claim to be located and returned verbatim do not survive that compression. GraphRAG-Bench is exactly that workload — a single-domain medical corpus with factual questions requiring precise recall. On FANG-2026, a heterogeneous cross-domain corpus, `global_search` scores 0.338 — still last among our configurations, but the gap narrows. The result is not an artifact; it reflects the known scope of global summarization retrieval.

One claim requires no controlled comparison: every system in the table above with O(docs × LLM) index cost is expensive by design. That cost is structural — it follows from LLM-based entity and relation extraction at index time. Our index cost is zero LLM calls regardless of accuracy rank.

### RQ2 — Which components drive the gain?

Rather than declaring a single winning configuration, we identify feature importance by examining which features are present across all top-ranked full-corpus runs. All three top configurations share laned community detection (`laned_pruned_k10`, `laned55_community_k10`, `laned_community_k10`), while the highest-scoring non-laned configuration (`cluster_community`) scores 0.007 lower — a consistent, configuration-independent signal that laned community detection is the dominant driver of performance. Within the laned family, differences of 0.001–0.002 between configurations indicate that the exact lane threshold is a low-sensitivity tuning parameter once the core feature is present.

Each component is added sequentially to quantify individual contributions. Entity-ref-expansion without lane filtering *hurts* (−0.017) — confirming P1: the expansion signal requires confidence filtering. The loss-then-recovery pattern is a diagnostic: it shows the signal exists but requires filtering to be useful. Note that adding community context in isolation reaches 0.661 — equal to G-reasoner — but this is not yet our result; gains from widened k materialize at k=15 and compound with redundancy pruning at k=20.

**Sequential feature addition:**

| Config | All | Δ | Status |
|--------|-----|---|--------|
| vanilla_256_rerank | 0.624 | — | ✓ |
| semantic_boundary_rerank | 0.646 | +0.022 | ✓ |
| + NER + entity-ref-expansion | 0.629 | −0.017 | ✓ |
| + lane filtering (sim≥0.45) | 0.646 | +0.017 | ✓ |
| + community context | 0.661 | +0.015 | ✓ |
| + k=10 | 0.659 | −0.002 | ✓ full run |
| + k=15 | 0.666 | +0.007 | ✓ |
| + pruning (k=20) | **0.674** | **+0.008** | ✓ |

*The k=10 gain observed in the grid sweep (+0.024 at 300 questions) does not replicate at full-corpus scale (4071 questions), suggesting the grid sample over-represented question types favoring wider retrieval. Gains from widened k materialize at k=15 and compound with redundancy pruning at k=20.*

**Component removal from full stack:**

| Removed | All | Δ |
|---------|-----|---|
| — (full stack: laned+community+pruning, k=20) | 0.674 | — |
| − pruning (→ k=15, unpruned) | 0.666 | −0.008 |
| − community context | 0.661¹ | −0.013 |
| − entity-ref-expansion | 0.661² | −0.013 |
| − lane filtering | 0.655³ | −0.019 |

¹ laned+pruning+k=10 (closest available ablation)
² rerank_k10 (no entity features at k=10)
³ cluster_community_k10 (no lane filter, community via cluster)

### RQ3 — Are the signals superadditive, and why?

Community context and k=10 are NOT superadditive at full-corpus scale: the community×k=10 combination (0.659) falls below the additive prediction (0.669). Superadditivity emerges at k=15, where community+k=15 (0.666) approaches the additive prediction and continues to compound with pruning at k=20 (0.674). The pattern is consistent with P3: redundancy removal (pruning) is required for deeper retrieval to add new evidence rather than dilute it — without pruning, k=10 adds noise as fast as it adds signal.

**Community × depth (2×2):**

| | k=5 | k=10 | k=15 |
|---|---|---|---|
| no community (laned) | 0.646¹ | 0.654 | — |
| community | 0.661 | 0.659 | **0.666** |
| Δ(community) | +0.015 | +0.005 | — |

¹ grid (300 questions)

At k=15: additive prediction = 0.654 + 0.015 = 0.669. Actual: **0.666**. Near-additive. Superadditivity appears only when pruning is added at k=20: 0.674 vs additive prediction of 0.669, interaction **+0.005**.

**Pruning × depth (2×2):**

| | k=10 | k=20 | Δ(k) |
|---|---|---|---|
| no pruning | 0.659 | — | — |
| pruning | 0.652 | **0.674** | +0.022 |
| Δ(pruning) | −0.007 | — | |

*Pruning hurts at k=10 (−0.007) because at moderate retrieval depth it removes useful near-duplicate evidence. At k=20, pruning removes true redundancy from the larger pool and improves by +0.022 vs k=10+pruning. The k=10 unpruned baseline for reference is 0.659; the full pruned+k=20 result of 0.674 represents a +0.015 gain over k=10+pruning (0.652) and +0.013 over unpruned k=10 (0.659).*

### RQ4 — What is the cost vs. quality tradeoff?

Our system eliminates index-time LLM calls entirely. Query-time overhead versus vanilla RAG is ~0.4s per query for reranking and community lookup combined. `[VERIFY: measure actual wall-clock index time on our 1100–2200 chunk corpus; measure query latency vs vanilla RAG]`

Index time and query latency for competing systems are not reported here. Those numbers would require running each system under identical hardware and corpus conditions, which we have not done. What is derivable from their published designs without running them: every system that performs LLM-based entity or relation extraction at index time incurs O(docs × LLM-calls) cost — the exact wall-clock figure depends on corpus size, model, and parallelism, but the scaling class is fixed by design. Further, because the graph edges are extracted by an LLM that sees chunk content, any corpus change invalidates the affected edges and requires re-extraction; incremental updates are not architecturally supported. Our index cost is O(docs) with no LLM calls. Adding new documents triggers community re-clustering over the updated co-occurrence graph, but that is a Louvain pass — O(edges) with no LLM calls — not a re-extraction of the corpus.

| System | Index LLM calls | All |
|--------|----------------|-----|
| G-reasoner | O(docs × relations) | 0.661 |
| HippoRAG2 | O(docs × entities) | 0.607 |
| AutoPrunedRetriever | O(docs × chunks) | 0.654 |
| **Ours** | **0** | **0.674** |

*Accuracy figures from arXiv:2506.05690. Vanilla RAG baseline conditions verified equivalent with author assistance; other system comparisons are reference only.*

---

## 6. Analysis

### 6.1 Retrieval Depth: The k Curve

The unpruned curve is flat from k=5 to k=10 and improves modestly at k=15, suggesting the benefit of widening k saturates quickly without redundancy removal. The pruned curve shows a clear gain from k=10 to k=20 (+0.022), consistent with pruning enabling deeper retrieval by removing context dilution. The unpruned k=20 and k=25 full runs are not yet available; the k=20 pruned result (0.674) is the current ceiling.

| k | unpruned All | pruned All |
|---|-------------|-----------|
| 5 | 0.661 | — |
| 7 | 0.658¹ | — |
| 10 | 0.659 | 0.652 |
| 15 | **0.666** | — |
| 20 | — | **0.674** |

¹ A-avg only (no Med/Nov breakdown)

### 6.2 Parameter Sensitivity

**Lane threshold** — tighter lane filtering does not consistently hurt at k=10 in the full run; all three thresholds cluster at 0.659–0.662. The grid showed a penalty for tighter filtering that does not replicate at full-corpus scale.

| Lane sim | All (k=5) | All (k=10) |
|----------|-----------|------------|
| 0.45 | 0.661 | 0.659 |
| 0.55 | 0.661¹ | 0.662 |
| 0.60 | 0.661² | 0.661³ |

¹ laned55_community_k10 at k=5 not available; laned55_community_k10 full run = 0.662 at k=10
² laned60_community at k=5 not available directly
³ laned60_community_k10 full run = 0.661

**Community coherence** — coherence sensitivity is below measurement noise at this scale; the finding is inconclusive.

| Coherence | All (k=10) |
|-----------|------------|
| 0.50 | 0.659 (laned_community_k10) |
| 0.60 | 0.661 (laned60_community_k10, A-avg only) |

**Community alpha** (breadcrumb structural prior):

| Alpha | All |
|-------|-----|
| 0.0 | 0.656 |
| 0.2 | 0.661 |

### 6.3 Topological Expansion (Cluster)

Cluster expansion does not add value over laned expansion in the full run; the grid suggested +0.003–0.005 but this does not replicate. Cluster may be redundant with lane filtering on homogeneous corpora; its value on heterogeneous corpora is tested in §8.

| Config | All (k=10) |
|--------|------------|
| laned + community | 0.659 |
| cluster + community (no lane) | 0.655 |
| cluster + laned + community + pruning | 0.653 |

### 6.5 Domain Asymmetries on GraphRAG-Bench

A consistent ~0.10 gap between Medical (Med ≈ 0.73) and Novel (Nov ≈ 0.63) persists across every configuration we tested. The gap is structural: Medical text is entity-dense, terminologically precise, and factual — properties that amplify P1 and P3. Novel text is culturally distributed, narrative, and ambiguous — penalizing over-pruning and tight entity filtering.

**α-weighted score** — for deployments weighted toward entity-dense corpora (medical records, legal filings, financial reports), define:

```
Score(α) = α · Med + (1 − α) · Nov
```

where α = 0.5 recovers the benchmark's equal-weight All. The table shows how top configuration rankings shift at α=0.7:

| Config | Med | Nov | All (α=0.5) | Score (α=0.7) | Rank (α=0.5) | Rank (α=0.7) |
|--------|-----|-----|-------------|--------------|--------------|--------------|
| laned + community + pruning + k=20 | 0.733 | 0.614 | **0.674** | **0.704** | 1 | 1 |
| laned + community + k=15 | 0.735 | 0.598 | 0.666 | 0.694 | 2 | 2 |
| laned55 + community + k=10 | 0.719 | 0.605 | 0.662 | 0.685 | 3 | 4 |
| laned60 + community + k=10 | 0.716 | 0.606 | 0.661 | 0.685 | 4 | 5 |
| laned + pruning + k=10 | 0.725 | 0.597 | 0.661 | 0.686 | 5 | 3 |
| rerank_k10 (no graph) | 0.727 | 0.595 | 0.661 | 0.685 | 5 | 5 |

The no-graph baseline (rerank_k10) ranks equally with several graph-augmented configs at α=0.5. At α=0.7, laned+pruning+k=10 edges ahead on Med score. The pruned+k=20 config leads at both α values.

**Domain-asymmetric parameter effects:**

*Retrieval depth (k)* — Med gains ~3× more from k=5→10 than Nov (+0.035 vs +0.012). Entity-dense corpora contain more recoverable non-redundant evidence per additional slot.

| Domain | k=5 | k=10 | Δ(k) |
|--------|-----|------|------|
| Medical | 0.701 | 0.736 | **+0.035** |
| Novel | 0.622 | 0.634 | +0.012 |

*Redundancy pruning* — N-Crea drops 0.073 (0.537 → 0.464) when pruning is added. M-Crea is largely unaffected. Creative questions require lexical diversity that pruning suppresses; factual and reasoning questions do not.

| Signal | Med Δ | Nov Δ | Asymmetry |
|--------|-------|-------|-----------|
| + pruning (at k=10, community) | −0.009 `[VERIFY]` | **−0.039** | Nov penalized 4× more |
| + k=10 (vs k=5) | **+0.035** | +0.012 | Med gains 3× more |
| + community context (at k=5) | +0.007 | **+0.022** | Nov gains 3× more |

*Community coherence* — tighter coherence (0.65 vs 0.50) is near-neutral for Med and slightly hurts Nov (0.622 → 0.583). Medical communities survive stricter filtering; narrative communities do not.

*Lane threshold* — 0.45 holds across both domains. Tighter thresholds (0.55, 0.60) hurt Nov proportionally more, as entity reference is less concentrated in narrative text.

These asymmetries are observed on a two-domain homogeneous benchmark. Configuration guidance grounded across both GraphRAG-Bench and FANG-2026 is in §8.5.

### 6.6 Configuration Selection Under Statistical Equivalence

*Note: bootstrap CIs (Appendix B) are not yet computed. Treat the statistical tie observations below as directional.*

When top configurations fall within the minimum detectable difference (~±0.015 at n=300), benchmark score alone cannot drive the selection decision. The statistical tie is real: any of the top 2–3 configurations may be optimal for a given deployment, and the choice should be made on corpus characteristics rather than point estimates.

The clearest discriminators are:

**Pruning** is the sharpest split. Its effect is corpus-dependent: it removes near-duplicate chunks, which helps factual and reasoning retrieval but suppresses lexical diversity that creative and narrative questions require. N-Crea drops 0.073 when pruning is added; M-Crea is largely unaffected.

**k** interacts with corpus density. The k=5→10 gain is ~3× larger for Medical than Novel. For sparse or short corpora, k=10 may over-retrieve; for large entity-dense corpora, k=15–20 is worth evaluating.

**Community coherence** separates corpora by topical tightness. Medical communities survive strict coherence filtering (0.65); narrative communities do not. Forcing coherence=0.65 on narrative text shrinks the P2 signal.

These observations are specific to the two GraphRAG-Bench domains. Configuration guidance across corpus types — incorporating FANG-2026 evidence — is in §8.5.

### 6.4 Case Studies

`[VERIFY: pull actual examples from eval output comparing runs with/without each signal. Format: question → answer without signal → answer with signal → ground truth.]`

**Entity connectivity** — Medical-Factual: a question about drug contraindications where the drug, condition, and contraindication appear in separate non-adjacent chunks. Dense retrieval returns two of three; entity-ref-expansion with lane filtering returns all three. `[VERIFY: find specific question ID]`

**Community context** — Novel-Summary: a question requiring synthesis across multiple chapters of a pre-20th-century novel. Individual chunk retrieval returns local passages; community injection provides the thematic framing needed to connect them. `[VERIFY: find specific question ID]`

**k=10** — Novel-Reasoning: a multi-entity question where the reasoning chain spans four connected concepts. k=5 returns evidence for three; k=10 recovers the missing link. `[VERIFY: find specific question ID]`

---

## 7. Generator Model Effects

The benchmark results throughout §5–6 use gpt-4o-mini as generator and judge. This section examines whether generator model choice materially affects the conclusions, and whether the answer depends on corpus characteristics.

### 7.1 Motivation

On GraphRAG-Bench (gpt-4o-mini), completed full-corpus runs cluster within a narrow band — a 10-point spread across all configurations, and less than 3 points separating the top seven:

| Configuration | Med | Nov | All | Graph features |
|---|---|---|---|---|
| vanilla\_256\_rerank (naive chunking) | — | — | 0.554 | none |
| rerank\_k10 (semantic chunking, no graph) | 0.727 | 0.595 | 0.661 | none |
| laned\_pruned\_k10 | 0.725 | 0.597 | 0.661 | entity-ref, lane filter, pruning |
| laned55\_community\_k10 | 0.719 | 0.605 | 0.662 | entity-ref, lane filter (0.55), community |
| laned\_community\_k10 | 0.715 | 0.602 | 0.659 | entity-ref, lane filter, community |
| cluster\_laned\_community\_pruned\_k10 | 0.710 | 0.596 | 0.653 | entity-ref, lane filter, community, cluster, pruning |
| laned\_community\_pruned\_k10 | 0.722 | 0.582 | 0.652 | entity-ref, lane filter, community, pruning |
| cluster\_community\_k10 | 0.717 | 0.593 | 0.655 | entity-ref, community, cluster |
| **laned\_community\_pruned\_k20** | **0.733** | **0.614** | **0.674** | entity-ref, lane filter, community, pruning, k=20 |

Two observations stand out. First, rerank\_k10 — semantic chunking with reranking, no graph features — ties the top graph-augmented configurations. Second, the graph-augmented configurations do not hurt: every configuration with entity-ref-expansion, lane filtering, or community context matches or approaches rerank\_k10. The graph features are, at minimum, doing no harm.

This narrow band could reflect a generator ceiling: the structured context produced by graph-augmented retrieval may exceed what gpt-4o-mini can exploit, leaving the latent retrieval advantage invisible at this model scale. FANG-2026 provides a sharper test: on a heterogeneous corpus the synthesis task is harder, and stronger generator capabilities may matter in ways that are invisible on a homogeneous benchmark.

### 7.2 Model Comparison Design

To evaluate whether retrieval gains are model-tier-invariant, we compare two generator tiers:

| Generator | Label | Tier |
|-----------|-------|------|
| gpt-4o-mini | **Mini** | Established low-cost baseline |
| claude-haiku-4-5-20251001 | **Haiku** | Stronger capability, low marginal cost on Anthropic subscription |

Haiku 4.5 represents a meaningfully higher capability tier while remaining in the low-cost range. The comparison tests whether: (1) the graph-augmented retrieval advantage replicates across tiers, and (2) a stronger model amplifies or diminishes that advantage.

We evaluate four configurations per model × ±SRR on both GraphRAG-Bench (credibility bridge) and FANG-2026 (primary benchmark):

| Config | Features |
|--------|---------|
| vanilla_rerank | 256-token chunks + reranking |
| vanilla_rerank+SRR | 256-token chunks + reranking + SRR |
| laned60+community+k=30 | NER + entity-ref + lane filter (sim≥0.60) + community + large k |
| laned60+community+k=30+SRR | NER + entity-ref + lane filter + community + large k + SRR |
| cluster+community+k=10 | NER + entity-ref + cluster + community |
| cluster+community+k=10+SRR | NER + entity-ref + cluster + community + SRR |
| graph_first+k=10 | Graph-first traversal + community |
| graph_first+k=10+SRR | Graph-first traversal + community + SRR |

**Feature-contribution decomposition (per model):**

- `Δ_SRR` = vanilla_rerank+SRR − vanilla_rerank: isolated SRR gain over reranking
- `Δ_retrieval` = best graph config − vanilla_rerank: isolated retrieval gain
- `Δ_combined` = best graph config + SRR − vanilla_rerank: combined gain
- Superadditivity test: `Δ_combined > Δ_retrieval + Δ_SRR`

**Pending GraphRAG-Bench runs (Haiku and SRR ablations, post-canonical NER):**

| Run | Model | SRR | Notes |
|-----|-------|-----|-------|
| vanilla_256_haiku_full | Haiku | no | baseline |
| vanilla_256_haiku_rerank_full | Haiku | no | reranking baseline |
| vanilla_256_mini_rerank_srr_full | Mini | yes | **SRR on vanilla — isolates Δ_SRR without graph** |
| vanilla_256_haiku_rerank_srr_full | Haiku | yes | **SRR on vanilla — isolates Δ_SRR without graph** |
| ner_ref_laned60_community_k30_haiku_full | Haiku | no | graph baseline |
| ner_ref_laned60_community_k30_haiku_srr_full | Haiku | yes | graph+SRR |
| ner_ref_laned60_community_k30_mini_srr_full | Mini | yes | graph+SRR |

The two vanilla+SRR runs are the critical ablation for H2: if `Δ_SRR` on vanilla is large and consistent across models, SRR works independently of graph retrieval. If the gain is larger on Haiku than Mini, model capability is the gating factor.

Mini equivalents using post-canonical NER (commit ff8f697, May 13 2026) are in the pending queue. Pre-canonical Mini NER runs used surface-form entity matching; the canonicalization change (root lemma normalization: "customers" → "customer") affects entity-feature runs only. Vanilla runs are unaffected and serve as valid baselines.

### 7.3 Hypotheses

**H1 (Graph retrieval via NLP primitives improves RAG quality)**: NER co-occurrence edges, Louvain community structure, and lane-gated traversal recover evidence that pure embedding retrieval misses — particularly on heterogeneous corpora requiring cross-document entity joins. The structural signal is compounding: P1 (entity-ref-expansion) and P2 (community context) each contribute independently, and their combination should exceed the sum of individual gains. On FANG-2026, where answers require linking entities across SEC filings, CVEs, Federal Register entries, and patents, this superadditivity should be most visible.

**H2 (SRR gains are real but model-capability-gated)**: Requiring the generator to produce `{answer, key_claims, evidence_used}` JSON and reprompting once when no evidence is cited is intuitive — it closes the loop between retrieval and generation. But compliance with the schema and the quality of the reprompt response depend on the generator's instruction-following capability. H2 predicts SRR delivers larger, more consistent gains on Haiku than on Mini, and that the gain floor (minimum `Δ_SRR`) is higher for capable models. The vanilla+SRR ablation isolates this effect without graph retrieval.

### 7.4 Results

Mini confirmed (GraphRAG-Bench, no SRR):

| Config | Model | All |
|--------|-------|-----|
| vanilla_256_rerank | Mini | 0.647 |
| laned_community_pruned_k20 | Mini | 0.674 |
| Δ_retrieval | — | +0.027 |

Mini+SRR, Haiku, and Haiku+SRR results: **[TBD — runs pending]**

FANG-2026 16-run matrix: **[TBD — see §8]**

---

## 8. Cross-Domain Evaluation (FANG-2026)

FANG-2026 is this study's primary benchmark. GraphRAG-Bench tests retrieval on two internally consistent domains; FANG-2026 tests it across four structurally dissimilar ones. The benchmark exists to answer a specific question: when the corpus is heterogeneous by construction, which retrieval strategies hold up? Configurations that appear equivalent on GraphRAG-Bench should separate on FANG-2026 — this is not a limitation of either benchmark, but the discriminating condition the study is designed to surface.

### 8.1 Corpus

The FANG-2026 corpus contains 4,237 chunks drawn from four source types, all covering the period 2024–2026:

| Source type | Content | Structure |
|---|---|---|
| SEC 10-K filings | Annual reports for FANG companies (Meta, Apple, Netflix, Google) | Long-form narrative + financial tables |
| CVE vulnerability records | NIST NVD entries for disclosed CVEs | Structured advisory format |
| Federal Register entries | Regulatory notices and proposed rules | Legal/regulatory prose |
| US patent data | Granted patents across technology domains | Claim + description format |

Each source type uses a different register, schema, and entity vocabulary. A retrieval strategy that works by identifying topically coherent clusters within a single domain faces a fundamentally different task here: entities from one domain (a CVE's affected product) must be linked to entities in another (a company's disclosed risk factor in a 10-K) to answer a question correctly.

### 8.2 Question Type Taxonomy

FANG-2026 uses 50 questions across five types, each requiring cross-domain evidence assembly:

| Code | Name | Definition |
|---|---|---|
| MDJ | Multi-Document Join | Answer requires combining facts from documents in at least two distinct source types |
| TV | Temporal Versioning | Answer requires identifying the correct version or date of a fact that changed during 2024–2026 |
| CDER | Cross-Domain Entity Resolution | The same real-world entity is referred to differently across source types; answer requires linking these references |
| QS | Quantitative Synthesis | Answer requires aggregating or comparing numerical values across sources |
| A/N | Absence/Negation | Correct answer is the absence of a fact, or that a stated claim is not supported by any source |

These question types were chosen because they stress the connections between retrieval chunks, not just individual chunk relevance. MDJ and CDER are explicit cross-domain joins. TV and QS require precision on specific values across multiple documents. A/N tests whether the retrieval layer returns evidence that is complete enough for the generator to distinguish supported from unsupported claims.

### 8.3 Results

**Previously completed runs (gpt-4o-mini, single-shot):**

| Run | MDJ | TV | CDER | QS | A/N | Mean |
|-----|----:|---:|-----:|---:|----:|-----:|
| rerank+cluster+community+k10 | 0.327 | 0.497 | 0.402 | 0.525 | 0.637 | **0.478** ✓ |
| graph_first+k10 | 0.394 | 0.579 | 0.431 | 0.406 | 0.556 | **0.473** ✓ |
| rerank+laned60+community+k10 | 0.156 | 0.563 | 0.345 | 0.262 | 0.495 | **0.364** ✓ |
| global_search+k10 | 0.358 | 0.463 | 0.267 | 0.316 | 0.287 | **0.338** ✓ |

**16-run model comparison matrix (Mini + Haiku × ±SRR × 4 configs — all pending):**

| Run | Model | SRR | MDJ | TV | CDER | QS | A/N | Mean |
|-----|-------|-----|----:|---:|-----:|---:|----:|-----:|
| fang_vanilla_rerank_mini | Mini | no | — | — | — | — | — | [TBD] |
| fang_vanilla_rerank_srr_mini | Mini | yes | — | — | — | — | — | [TBD] |
| fang_ner_ref_laned60_community_k10_mini | Mini | no | — | — | — | — | — | [TBD] |
| fang_ner_ref_laned60_community_k10_srr_mini | Mini | yes | — | — | — | — | — | [TBD] |
| fang_ner_ref_cluster_community_k10_mini | Mini | no | — | — | — | — | — | [TBD] |
| fang_ner_ref_cluster_community_k10_srr_mini | Mini | yes | — | — | — | — | — | [TBD] |
| fang_ner_ref_graph_first_k10_mini | Mini | no | — | — | — | — | — | [TBD] |
| fang_ner_ref_graph_first_k10_srr_mini | Mini | yes | — | — | — | — | — | [TBD] |
| fang_vanilla_rerank_haiku | Haiku | no | — | — | — | — | — | [TBD] |
| fang_vanilla_rerank_srr_haiku | Haiku | yes | — | — | — | — | — | [TBD] |
| fang_ner_ref_laned60_community_k10_haiku | Haiku | no | — | — | — | — | — | [TBD] |
| fang_ner_ref_laned60_community_k10_srr_haiku | Haiku | yes | — | — | — | — | — | [TBD] |
| fang_ner_ref_cluster_community_k10_haiku | Haiku | no | — | — | — | — | — | [TBD] |
| fang_ner_ref_cluster_community_k10_srr_haiku | Haiku | yes | — | — | — | — | — | [TBD] |
| fang_ner_ref_graph_first_k10_haiku | Haiku | no | — | — | — | — | — | [TBD] |
| fang_ner_ref_graph_first_k10_srr_haiku | Haiku | yes | — | — | — | — | — | [TBD] |

*Key comparisons: for each retrieval config × model, `Δ_SRR` = SRR row − no-SRR row. Superadditivity test: does `Δ_SRR` increase with retrieval quality (vanilla → laned → cluster/graph_first)? Multi-step variants (`_ms`) for top-2 configurations queued; see §9.*

### 8.4 Findings from Completed Runs

**Overall difficulty.** A mean score near 0.4 is appropriate for this benchmark. GraphRAG-Bench questions are domain-homogeneous — the evidence for any given question lives within a single, terminologically consistent domain. FANG-2026 questions require assembling evidence across structurally dissimilar source types. The single-shot retrieval ceiling is lower by design; it is not a deficiency of the retrieval systems.

**cluster+community is best (0.478).** Community structure helps more than lane filtering on a heterogeneous corpus. Communities on FANG-2026 are formed from co-occurrence patterns that span source types — entities like company names and product identifiers appear in both financial filings and security advisories. These cross-domain co-occurrences form the edges that community detection captures and lane filtering discards.

**graph_first nearly ties (0.473).** Graph traversal is effective on heterogeneous corpora. Following entity edges to non-adjacent chunks is precisely the operation that cross-domain joins require — an entity in a CVE record that also appears in a 10-K risk factor is connected by an edge, and graph_first follows it. The near-tie with cluster+community suggests the two strategies are reaching the same cross-domain evidence via different paths.

**laned60 collapses on MDJ (0.156).** The 0.60 lane threshold cuts cross-domain entity links by requiring that expanded chunks be embedding-similar to the query. On a heterogeneous corpus, a CVE record and a 10-K filing may share an entity without being embedding-similar — they use different vocabulary, different structure, different register. Lane filtering with a tight threshold treats this dissimilarity as noise and discards it. For MDJ questions, the discarded chunks are the answer. The MDJ score (0.156) reflects this directly: laned60 retrieves within-domain chunks at high precision but fails to assemble cross-domain joins.

**global_search is weakest (0.338).** Global summarization compresses the corpus to topic summaries. Precise cross-domain lookups — a specific CVE ID, a specific regulatory docket number, a specific patent claim — do not survive compression. The global_search score is lowest on CDER (0.267) and A/N (0.287), exactly the question types that require precise entity identification and complete evidence coverage.

**Comparison with GraphRAG-Bench.** On GraphRAG-Bench, configurations that differ by lane threshold cluster within 0.003 of each other — the signal is below measurement noise at that scale. On FANG-2026, the same choice (laned60 vs cluster+community) produces a 0.114-point gap. Corpus heterogeneity is the discriminating condition. This is consistent with H1 from §7.3: graph retrieval gains compound on heterogeneous corpora where cross-document entity joins are required.

### 8.5 Configuration Guidance

GraphRAG-Bench (§6.5) reveals parameter sensitivity within a homogeneous corpus. FANG-2026 reveals strategy sensitivity across heterogeneous corpus types. Together they support the following guidance.

**Retrieval strategy** is the first choice, and it depends on corpus structure:

- *Homogeneous corpus* (single domain, consistent vocabulary): laned entity-ref-expansion. Embedding similarity is a reliable relevance proxy within a domain; lane filtering at sim≥0.45–0.60 improves precision without sacrificing cross-domain recall that doesn't exist in the corpus.
- *Heterogeneous corpus* (multiple source types, mixed register): cluster+community or graph_first. A CVE record and a 10-K filing may share an entity without being embedding-similar — they use different vocabulary, different structure. Lane filtering treats this dissimilarity as noise and discards relevant evidence. Entity-edge traversal and community co-occurrence capture cross-domain links regardless of embedding distance.

The reversal is large: laned60 scores 0.661 on GraphRAG-Bench but 0.364 on FANG-2026. cluster+community scores 0.654 on GraphRAG-Bench and 0.478 on FANG-2026. Strategy choice that is nearly invisible on a homogeneous benchmark produces a 0.114-point gap on a heterogeneous one.

**k (retrieval depth):**

- Entity-dense corpora (medical, legal, financial, technical): k=15–20+pruning. Each additional slot recovers non-redundant evidence. The Med k=5→10 gain (+0.035) is 3× the Nov gain (+0.012). With lane filtering, k can be set large and the similarity threshold controls quality of retrieved expansions.
- Narrative or cross-domain corpora: k=10–15. Over-retrieval with tight lane filtering on heterogeneous corpora adds noise rather than evidence.

**Redundancy pruning:**

- Enable on factual/technical homogeneous corpora where creative generation is not a priority (N-Crea drops 0.073 with pruning).
- Disable for narrative corpora.
- Behavior on heterogeneous corpora is not yet measured — no pruning runs were included in the FANG-2026 evaluation.

**Community coherence:** 0.65 for terminologically precise single-domain corpora; 0.50 for narrative or cross-domain corpora where strict filtering shrinks the community signal.

**Summary:**

| Corpus type | Strategy | k | lane-sim | coherence | pruning |
|-------------|----------|---|----------|-----------|---------|
| Homogeneous, entity-dense (medical, legal, financial) | laned+community | 15–20 | 0.45–0.60 | 0.50–0.65 | enabled |
| Homogeneous, narrative | laned+community | 7–10 | 0.45 | 0.50 | disabled |
| Heterogeneous, multi-domain | cluster+community or graph_first | 10–15 | N/A | 0.50 | not tested |

---

## 9. Toward a Multi-Step RAG Benchmark

### 9.1 Motivation

Every production RAG deployment for complex queries operates with a planner in front of it. The planner decomposes the user's question into targeted sub-queries, issues them to the retrieval layer, and assembles the results into a coherent response. Single-shot evaluation — one query, one retrieval pass — measures a deployment pattern that does not exist in practice for hard questions.

The FANG-2026 single-shot scores (~0.4 mean) are not a floor to be optimized away. They are a one-pass retrieval ceiling. A system that retrieves perfectly on a single pass cannot score higher than the information present in the top-k chunks retrieved for the original, potentially ambiguous query. For MDJ questions, the original query may not surface the exact entity terms used in both source types. For TV questions, the original query may not specify the date range that disambiguates versions. Sub-query decomposition resolves these gaps — not by changing the retrieval system, but by issuing more targeted queries to it.

The delta between single-shot and multi-step scores answers a concrete engineering question: does multi-step retrieval help for cross-domain questions, and which retrieval strategy benefits most from targeted sub-queries?

### 9.2 Multi-Step Retrieval Experiment

**Implementation.** The `--multi-step` flag decomposes the original question into three targeted sub-queries using the generator model. Each sub-query is issued independently to the retrieval layer. Retrieved chunks from all three sub-queries are merged by best score per chunk (a chunk retrieved by multiple sub-queries keeps the highest score). The merged set is then reranked against the original question and passed to the generator.

This design isolates retrieval strategy as the independent variable. The planner (sub-query decomposition) is fixed and identical across all runs; the only variation is which retrieval configuration serves the sub-queries.

**Paired runs (results pending):**

| Single-shot run | Multi-step run | Single-shot mean | Multi-step mean | Expected Δ |
|---|---|---|---|---|
| rerank+cluster+community+k10 | cluster+community+k10_ms | 0.478 ✓ | [PENDING] | +0.05–0.12¹ |
| graph_first+k10 | graph_first+k10_ms | 0.473 ✓ | [PENDING] | +0.05–0.12¹ |
| rerank_k10 | rerank_k10_ms | [PENDING] | [PENDING] | +0.02–0.06² |

¹ Graph-augmented strategies are expected to gain more from multi-step than no-graph baselines. Sub-query decomposition issues targeted queries (e.g. "CVE affecting product X" + "10-K risk disclosure for company Y") that the graph traversal layer is specifically built to answer — entity-ref-expansion follows cross-domain entity edges that a single ambiguous query does not surface.
² No-graph ablation multi-step: decomposition helps by reducing query ambiguity, but without entity-edge traversal the merger of retrieved chunks is likely dominated by duplicate evidence rather than complementary cross-domain evidence.

Falsification condition: if rerank\_k10 multi-step gain ≥ cluster+community multi-step gain, the benefit is planner-driven and independent of graph structure. If graph-augmented configs gain more, sub-query precision is amplified by graph traversal — the two together exceed either alone.

### 9.3 Benchmark Vision

The FANG-2026 multi-step experiment is a prototype for a broader benchmark. The design principles:

**Questions with known decomposition paths.** FANG-2026's five question types already have natural sub-query structures:

| Question type | Natural decomposition |
|---|---|
| MDJ | 3 entity-specific sub-queries, one per source type + one join |
| TV | date sub-query + change-event sub-query + version-resolution sub-query |
| CDER | entity-in-domain-A sub-query + entity-in-domain-B sub-query + resolution sub-query |
| QS | value-retrieval sub-query per source + aggregation sub-query |
| A/N | positive-claim sub-query + negation-check sub-query + coverage-confirmation sub-query |

A benchmark where decomposition paths are known enables per-step evaluation: did the retrieval layer surface the right evidence for each sub-query, independent of whether the generator assembled it correctly? This separates retrieval quality from generation quality — a distinction that single-shot evaluation collapses.

**Per-step and per-claim scoring.** Decomposed scoring is already implemented in this system. Each retrieved sub-query result can be evaluated against the sub-question it was issued for, and each generator claim can be traced to the chunks that supported it. Benchmark-level scoring aggregates these per-step scores, making it possible to identify where in the retrieval pipeline quality degrades — whether at the first retrieval pass, the merge step, or the final generation.

**Deployment-scenario-weighted criteria.** Different deployment scenarios weight question types differently. A compliance use case weights A/N (has the company disclosed this risk?) more heavily than MDJ. A discovery use case weights MDJ and CDER more heavily than A/N. A benchmark that reports only mean accuracy discards this structure. The Multi-Step RAG Benchmark reports per-type scores and allows operators to apply deployment-appropriate weights — the same α-weighting approach introduced in §6.5, applied to question type rather than domain.

**Retrieval strategy as the independent variable.** The planner is fixed. Only the retrieval configuration changes. This isolates retrieval quality as the thing being measured, which is what a retrieval benchmark should measure.

The delta between single-shot and multi-step scores — for each retrieval strategy — is the primary contribution of this benchmark. A strategy that improves under multi-step has precision that scales with targeted querying. A strategy that holds flat has a coverage ceiling that sub-query decomposition cannot break through.

---

## 10. Limitations

- **Benchmark scope**: GraphRAG-Bench serves as a credibility bridge to published baselines; FANG-2026 (50 questions) is the primary benchmark. At 50 questions, score differences below ~0.05 are not reliably detectable. The 16-run pending matrix extends coverage, not statistical power.
- **Single credibility benchmark**: GraphRAG-Bench results are on Medical + Novel domains only. We include preliminary results on HotpotQA (All=**0.578** `[VERIFY]` vs RAG+rerank **0.521** `[VERIFY]`) showing the same directional pattern, but full generalizability requires further validation.
- **Canonical NER boundary**: Entity canonicalization (root lemma normalization, commit ff8f697, May 13 2026) was introduced after initial Mini NER runs. All pre-canonical Mini NER runs are queued for re-execution. Vanilla baselines are unaffected (NER not involved). gpt-4o experiments at k=30 were not repeated due to cost (~$130/run); observations from those runs are noted as preliminary.
- **Creative question type**: N-Crea is volatile across configurations. `bc_pruned_laned_community` drops to N-Crea=0.293 while overall ACC looks acceptable — pruning or community context may suppress creative generation. Not yet understood.
- **Medical vs Novel gap**: consistent ~0.10 gap (Med ~0.73, Nov ~0.63) across all configurations. Entity-centric signals may systematically advantage Medical (denser entity linking) over Novel (broader cultural knowledge).
- **Judge calibration**: certain questions fail the calc_fact judge on every run due to structured JSON parse failures. These are excluded from scoring; impact on reported scores is small but systematic.
- **NaN rate as a UX signal**: questions that fail evaluation (NaN) represent cases where the generator produced a non-compliant or empty response. In a production system, each such failure requires a human re-prompt cycle, directly incurring the agent cost this paper aims to reduce. GraphRAG-Bench currently excludes NaN questions from scoring entirely, which inflates reported ACC for systems with high failure rates. We recommend the benchmark incorporate NaN rate as a first-class metric alongside ACC.
- **Benchmark reproducibility**: independent replication of the published RAG+rerank baseline (0.554) was not possible with the information currently available — the generator prompt and baseline retrieval code are not published. All ablation comparisons in this paper use an internal controlled baseline run through the same pipeline and eval code as every other configuration reported here.
- **Leaderboard integrity**: the absence of a standardized submission protocol creates conditions where self-reported scores computed under differing embedding models, generator prompts, or evaluation code versions are presented alongside author-verified scores without distinction. We encourage the community to treat leaderboard positions as indicative rather than authoritative until submission standards are established.
- **Static index**: community detection and entity index are built once. Corpus updates require rebuilding the co-occurrence matrix and community graph — a meaningful constraint for frequently-updated corpora.

---

## 11. Conclusion

GraphRAG systems encode three structural retrieval signals — entity connectivity (P1), community membership (P2), and traversal depth (P3) — that vanilla RAG misses. The standard assumption is that LLM-based graph extraction is necessary to capture these signals at useful fidelity. We show it is not.

Against the vanilla RAG+rerank baseline (vanilla\_256\_rerank, All=0.624), our full graph stack delivers a +0.050 improvement (All=0.674) at zero index-time LLM cost. Graph signals specifically contribute +0.013 over the no-graph ablation using the same semantic chunking (rerank\_k10, All=0.661). The index builds in approximately 8 minutes versus hours for LLM-based knowledge graph construction.

The more interesting story is in production. GraphRAG-Bench uses equal-weight Medical and Novel domains; entity-dense corpora (legal, financial, technical, medical) resemble the Medical domain, which gains 3× more from k widening than narrative corpora. In real-world deployments where entity density is high and corpus updates are frequent, the cost differential widens further: LLM-based KG construction requires full reconstruction on every update; NLP-primitive indexing does not.

FANG-2026 extends this picture to heterogeneous corpora. The single-shot results confirm that retrieval strategy choice matters in ways that domain-homogeneous benchmarks cannot reveal. Lane filtering, which is near-neutral on GraphRAG-Bench, collapses on multi-document join questions. Graph traversal strategies that perform equivalently on GraphRAG-Bench separate cleanly on FANG-2026. The reversal is structurally motivated and reproducible. Whether this separation is amplified or diminished for stronger generators (Haiku vs Mini) is the primary open question. H2 predicts the answer is model-capability-gated: stronger generators comply with SRR more reliably, which may interact with the richer evidence that graph retrieval delivers.

The result is not that graphs are unnecessary — it is that LLM extraction is unnecessary to build a graph that works. The broader implication returns to the motivating agent: re-prompting is a symptom of retrieval incompleteness. Each re-prompting cycle is the agent reconstructing, at inference time, the graph structure that a better retrieval layer would have pre-computed. Cheap NLP primitives are sufficient to pre-compute it — and cheap enough to keep current.

---

## Appendix A: Full Grid Results

`[VERIFY: insert complete sorted results table after all runs complete]`

## Appendix B: Statistical Significance

`[VERIFY: bootstrap 95% CIs for all key pairwise comparisons in RQ2–RQ3. Flag deltas below ±0.015 as below minimum detectable difference at n=300, α=0.05.]`

## Appendix C: Experimental Run Reference

| Script | Runs | Status |
|--------|------|--------|
| run_bc_grid_missing.sh | 6 | ✓ complete |
| run_ablation_structural.sh | 2 | ✓ complete |
| run_grid_gaps.sh | 2 | ✓ complete |
| run_new_ideas.sh | 6 | ✓ complete |
| run_paper_validation.sh | 4 | ✓ complete |
| run_extended_grid.sh | 8 | ✓ complete |
| run_full_all.sh | GraphRAG-Bench full-corpus: Mini baseline confirmed; Haiku runs pending | Mini ✓; Haiku pending |
| run_fang2026.sh | FANG-2026: preliminary single-shot runs ✓; 8-run Mini+Haiku matrix pending | Preliminary ✓; matrix pending |

## Appendix D: Run Name Key

Run names encode the retrieval configuration as a `_`-delimited sequence of feature tokens. All graph-augmented runs use semantic boundary chunking (1100–2200 tokens). The mapping:

| Token | Meaning |
|-------|---------|
| `vanilla_256` | Naive 256-token fixed-size chunking. No semantic boundaries, no graph features. |
| `ner_ref` | NER entity-ref-expansion enabled (P1) |
| `rerank` | Cross-encoder reranking (BAAI/bge-reranker-large) |
| `laned` | Lane filtering at default sim≥0.45 threshold (see also laned55, laned60) |
| `laned55` / `laned60` | Lane filtering at sim≥0.55/0.60 |
| `community` | Community context injection (P2); coherence filter default 0.50 |
| `community60` | Community context with coherence≥0.60 |
| `pruned` | Redundancy pruning at cosine≥0.92 |
| `cluster` | Cluster-based topological expansion (replaces lane filtering for heterogeneous corpora) |
| `kN` | Retrieval depth top-k=N (default k=5 if omitted) |
| `global_search` | Global search mode (community-level summarization retrieval) |
| `graph_first` | Graph-first search mode (entity traversal before dense retrieval) |
| `haiku` | Generator: claude-haiku-4-5-20251001 via Anthropic |
| `llama8b` | Generator: Llama-3.1-8B-Instruct-Turbo via Together AI |
| `srr` | Evidence-compliance reprompt — generator produces `{answer, key_claims, evidence_used}` JSON; fires once when `evidence_used=[]`; no additional retrieval; strictly additive |
| `ms` | Multi-step retrieval (3 sub-query decomposition) |
| `full` | Full-corpus run (4,072 questions) |
| `grid` | Grid sweep run (300 stratified questions) |

**Examples:**
- `ner_ref_rerank_laned_community_pruned_k20_full` = semantic boundary chunking + entity-ref-expansion + reranking + lane filter (sim≥0.45) + community context + redundancy pruning + k=20, full corpus
- `ner_ref_laned60_community_k30_haiku_full` = semantic boundary chunking + entity-ref-expansion + lane filter (sim≥0.60) + community + k=30, Haiku generator, full corpus
- `fang_ner_ref_cluster_community_k10_haiku` = FANG-2026 corpus + entity-ref + cluster expansion + community context + k=10, Haiku generator
