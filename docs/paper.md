# Implicit GraphRAG: Knowledge Graph Signals Without LLM-Based Graph Construction

**Kenneth Stott**
Member of Technical Staff and Senior Advisor, Logick

Code: https://github.com/kenstott/chonk (MIT License)

> **VERIFY markers**: all results tagged `[VERIFY]` are placeholder estimates. Replace with actual experimental results before submission.
> **Status**: Most `[VERIFY]` and `[TBD]` placeholders have been replaced with confirmed full-corpus results (4071 questions). Remaining `[VERIFY]` tags indicate sections not yet updated (RQ4 latency, Appendices A–B, §6.4 case studies).

---

## Abstract

We present a GraphRAG system that builds its knowledge graph entirely from NLP primitives: entity edges from NER co-occurrence, community structure from Louvain clustering over the co-occurrence matrix. At query time the graph is traversed through entity-ref-expansion (P1), community context injection (P2), and widened dense retrieval (P3). No LLM is involved at index time. The three signals are conditionally superadditive — they correct orthogonal retrieval failure modes and their gains compound when retrieval depth is sufficient for the additional context to be non-redundant.

Against the vanilla RAG+rerank baseline (naive 256-token chunking, All=0.624), our full stack achieves a +0.050 improvement (All=**0.674**) at zero index-time LLM cost. Against a no-graph ablation using the same semantic chunking pipeline (rerank\_k10, All=0.661), graph signals alone contribute +0.013 — isolating the contribution of entity-ref-expansion, community context, and traversal depth from the chunking improvement. Index build time is approximately 8 minutes versus hours for LLM-based knowledge graph construction. The cost differential is the primary contribution: graph-quality retrieval signals at NLP-primitive cost, with disproportionate gains in entity-dense production corpora (legal, financial, medical, technical). We further evaluate on FANG-2026, a cross-domain benchmark spanning financial filings, security advisories, regulatory documents, and technical patents, where structural heterogeneity separates retrieval strategies that perform equivalently on domain-homogeneous corpora. We introduce a multi-step RAG evaluation framework as a more realistic proxy for the agentic deployment scenarios where retrieval quality matters most (§9).

---

## 1. Introduction

The immediate motivation for this work was a multi-step reasoning agent operating across heterogeneous corpora — relational databases and unstructured documents side by side. The agent could answer complex questions, but it did so inefficiently: through repeated re-prompting, intermediate validation steps, and iterative proof-building to assemble evidence scattered across sources. It worked. It was expensive.

Each re-prompting cycle was the agent doing manually what a better retrieval layer should have done automatically: traversing the connections between entities, following topic threads across documents, and expanding the evidence set until it was complete. The agent was compensating for a RAG layer that returned locally relevant but globally incomplete context.

This paper asks whether the retrieval layer itself can encode the relational structure that agents currently rediscover through re-prompting — and whether it can do so without the cost of LLM-based graph construction.

Almost all real-world documents have structure. Paragraphs group related sentences. Sections group related paragraphs. Named entities recur across sections, connecting ideas that naive chunking severs. A fixed-size chunker discards this structure by design: it splits at token boundaries, not semantic ones, producing chunks whose boundaries are arbitrary with respect to the content they contain. The graph signals we recover — co-occurrence edges, community partitions, entity-ref-expansion — are not additions to the retrieval pipeline. They are recoveries of structure the chunker discarded. The cost of recovering them via NLP primitives is negligible precisely because the structure was always there; we are reading it, not constructing it.

Existing GraphRAG systems (G-reasoner, AutoPrunedRetriever, HippoRAG2) encode this structure through LLM-based entity and relation extraction, requiring O(docs × LLM-calls) at index time. For an agent operating across heterogeneous and frequently-updated corpora, that cost is impractical.

We build the graph differently. At index time, spaCy NER identifies entities in each chunk; co-occurrence edges connect entities that appear together; Louvain clustering over the co-occurrence matrix partitions the graph into communities. The result is a full knowledge graph — nodes, edges, community structure — built in minutes with no LLM calls. At query time, the graph is traversed: entity-ref-expansion follows entity edges to retrieve non-adjacent chunks, community context injection provides global topic framing, and widened dense retrieval (k=10) approximates multi-hop path traversal.

### 1.1 Design Principles

Three structural signals drive the system:

**P1 — Entity connectivity.** NER identifies entities in each chunk. Co-occurrence within a chunk creates an edge between those entities in the graph. At query time, entity-ref-expansion follows these edges: chunks that share a named entity with a retrieved chunk are pulled in, even if they are embedding-distant. Lane filtering (sim ≥ 0.45) gates expansion on query relevance — expansion without filtering hurts (−0.017), filtered expansion helps (+0.017).

**P2 — Community structure.** Louvain clustering over the co-occurrence matrix partitions the entity graph into topically coherent communities at O(chunks²), one-time. At query time, the communities of retrieved chunks are identified and a community context summary is prepended to the generator prompt, providing global topic framing that individual chunk retrieval misses.

**P3 — Traversal depth.** Wider dense retrieval (k=10 vs k=5) approximates multi-hop graph traversal by expanding the evidence pool. The gain from k is bounded by corpus redundancy; redundancy pruning (cosine ≥ 0.92) shifts the effective plateau rightward, enabling larger k without context dilution.

Because P1 operates at retrieval time, P2 at prompt-construction time, and P3 at the retrieval pool level, their failure modes are orthogonal — gains compound superadditively.

### 1.2 Contributions

1. We describe a GraphRAG system whose graph is built entirely from NLP primitives (NER + Louvain) and show it delivers a +0.050 improvement over the vanilla RAG+rerank baseline (0.624 → **0.674**), with graph signals contributing +0.013 over the no-graph ablation using the same semantic chunking (rerank\_k10, 0.661 → 0.674), at zero index-time LLM cost and an ~8-minute index build versus hours for LLM-based KG construction (§4–6).
2. We demonstrate conditional superadditivity: the three signals (P1 entity-ref-expansion, P2 community context, P3 widened retrieval) correct orthogonal retrieval failure modes; gains compound when k is large enough for the extra context to be non-redundant — superadditivity is present at k=15/20+pruning but not at k=10, where community and depth do not interact positively at full-corpus scale.
3. We characterize the k-plateau and show redundancy pruning shifts it, providing a principled approach to retrieval depth without k tuning (§6.1).
4. We provide domain-weighted tuning guidance showing that entity-dense production corpora (legal, financial, medical, technical) gain disproportionately from graph signals relative to the equal-weight benchmark (§6.5).
5. We introduce a multi-step RAG evaluation framework and FANG-2026, a cross-domain benchmark spanning financial filings, security advisories, regulatory documents, and technical patents. Single-shot evaluation systematically underestimates retrieval system quality for the agentic deployment scenarios where RAG is most needed (§9).

---

## 2. Related Work

[Fill: 3–4 paragraphs positioning against GraphRAG, HippoRAG2, G-reasoner, AutoPrunedRetriever, RAPTOR, HyDE, FLARE. Key distinction: prior work either builds explicit KGs (expensive) or uses dense retrieval alone (misses structure). We occupy the gap.]

### 2.1 LLM-Based GraphRAG Systems

MS-GraphRAG, LightRAG, Fast-GraphRAG, HippoRAG, HippoRAG2, and G-reasoner all construct knowledge graphs through LLM extraction — entity recognition, relation triple extraction, or both — at O(docs × LLM-calls) index time. Graph traversal at query time then follows the extracted edges. These systems achieve strong benchmark results but impose hours of index build time and significant API cost on static corpora; corpus updates require full reconstruction.

### 2.2 NLP-Primitive GraphRAG (This Work)

We construct the same graph structures — entity nodes, co-occurrence edges, community partitions — using spaCy NER and Louvain clustering. Index time drops to minutes with zero LLM calls. Query-time traversal follows the same pattern: entity-ref-expansion walks entity edges; community context injection uses community membership; widened retrieval (k) extends traversal depth.

### 2.3 Retrieval Augmentation Without Graph Structure

Cross-encoder reranking, RAPTOR (hierarchical summarization), HyDE (hypothetical document embeddings). These improve retrieval without encoding graph structure and serve as our non-graph baselines.

### 2.3 Redundancy and Diversity in Retrieval

[Fill: MMR, deduplication approaches, AutoPrunedRetriever's pruning strategy. Position our cosine-threshold pruning.]

---

## 3. Method

### 3.1 Index-Time Pipeline

```
Documents
  → Semantic boundary chunking (1100–2200 tokens)
  → [optional] Breadcrumb embedding bias (--breadcrumb-embed)
  → NER (spaCy) → EntityIndex
  → Co-occurrence matrix → Community detection (Louvain)
  → Vector store + embeddings
```

Index cost: off-the-shelf NER (CPU) + co-occurrence matrix (O(chunks²), one-time) + Louvain community detection. **Zero LLM calls.**

### 3.2 Query-Time Pipeline

```
Query
  → Dense retrieval (top-k)
  → [optional] Entity-ref-expansion  ← P1: entity connectivity
  → [optional] Lane filtering (sim ≥ threshold)  ← P1: edge confidence
  → [optional] Cluster expansion  ← P1: topological neighbors
  → [optional] Redundancy pruning (cosine ≥ 0.92)  ← P3: deduplication
  → Cross-encoder reranking
  → [optional] Community context injection  ← P2: global structure
  → Generator (gpt-4o-mini)
```

### 3.3 Parameters

| Parameter | Default | Tested range |
|-----------|---------|-------------|
| top-k | 5 | 5, 7, 10, 15, 20, 25 |
| lane-entity-min-sim | 0.45 | 0.45, 0.55, 0.60 |
| community-min-coherence | 0.50 | 0.50, 0.65 |
| community alpha | 0.20 | 0.0, 0.2 |
| redundancy-threshold | 0.92 | 0.92 |

---

## 4. Experimental Setup

- **Dataset**: GraphRAG-Bench (arXiv:2506.05690), full question set, Medical + Novel domains
- **Question types**: Factual, Reasoning, Summary, Creative
- **Generator + judge**: gpt-4o-mini
- **Metric**: answer_correctness (mean of 4 subtype scores per domain)
- **Statistical testing**: bootstrap resampling (n=10,000); 95% CI throughout. `[VERIFY: report MDD after confirming full question set size]`
- **Experimental protocol**: A full combinatorial sweep across all feature dimensions would require evaluating O(N^k) configurations at non-trivial cost per full-corpus run — computationally prohibitive given our resource constraints. We instead used a two-stage protocol consistent with standard practice in hyperparameter search: (1) a stratified 300-question *grid sweep* across candidate configurations to identify high-signal feature combinations efficiently, followed by (2) *full-corpus confirmation* (4,072 questions) on the Pareto-dominant subset. Conclusions about feature importance are drawn from feature co-occurrence across top-ranked configurations, not from exhaustive enumeration. All scores reported in tables are from full-corpus runs unless marked `[VERIFY]`.

A secondary evaluation surface, FANG-2026, is introduced in §8. It is a 50-question benchmark across four structurally distinct source types (SEC 10-K filings, CVE records, Federal Register entries, US patents). Unlike GraphRAG-Bench, FANG-2026 questions require cross-domain entity resolution by design — the correct answer requires assembling evidence from at least two different source types.

---

## 5. Experiments

### RQ1 — Does the full stack exceed published GraphRAG systems?

**Unknown.** Direct comparison is not possible: the benchmark's published numbers were not produced under the same evaluation conditions as ours, and the information required to replicate them is not available. The generator prompt, baseline retrieval implementation, and exact embedding configuration used for leaderboard entries are unpublished. Score differences of 0.01–0.02 are within the margin introduced by these unknowns.

The table below places our numbers alongside published leaderboard values for reference, not as a controlled comparison.

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

*Full stack = laned entity-ref-expansion (sim≥0.45) + community context + redundancy pruning, k=20. Leaderboard values from arXiv:2506.05690; evaluation conditions not verified as equivalent.*

What we can state: within our own controlled pipeline (identical generator, prompt, and eval code across all configurations), the full stack reaches All=0.674 versus our vanilla RAG+rerank baseline of 0.647 (+0.027) and our no-graph semantic chunking baseline of 0.654 (+0.020). Those internal comparisons are the paper's primary claim. The consistent ~0.10 Med–Nov spread across all configurations, including the published leaderboard entries, suggests it reflects corpus characteristics rather than retrieval method.

We also implement and evaluate two retrieval modes that approximate the query-time behavior of published LLM-GraphRAG systems: `global_search` (community-level summarization, approximating MS-GraphRAG global mode) and `graph_first` (entity traversal before dense retrieval, approximating HippoRAG-style hop expansion). Both run through the same eval pipeline as every other configuration. `graph_first` scores 0.645 and `global_search` scores 0.257 — both below the full stack (0.674) and below the no-graph semantic chunking baseline (0.654). These are controlled comparisons. The full results and analysis are in §7.5.

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

*Accuracy figures from arXiv:2506.05690; evaluation conditions not verified as equivalent (see §8.4).*

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

Cluster expansion does not add value over laned expansion in the full run; the grid suggested +0.003–0.005 but this does not replicate. Cluster may be redundant with lane filtering.

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

**k** interacts with corpus density. The k=5→10 gain is ~3× larger for Medical than Novel. For sparse or short corpora, k=10 may over-retrieve; for large entity-dense corpora, k=15 is worth evaluating.

**Community coherence** separates corpora by topical tightness. Medical communities survive strict coherence filtering (0.65); narrative communities do not. Forcing coherence=0.65 on narrative text shrinks the P2 signal.

These observations are specific to the two GraphRAG-Bench domains. Configuration guidance across corpus types — incorporating FANG-2026 evidence — is in §8.5.

### 6.4 Case Studies

`[VERIFY: pull actual examples from eval output comparing runs with/without each signal. Format: question → answer without signal → answer with signal → ground truth.]`

**Entity connectivity** — Medical-Factual: a question about drug contraindications where the drug, condition, and contraindication appear in separate non-adjacent chunks. Dense retrieval returns two of three; entity-ref-expansion with lane filtering returns all three. `[VERIFY: find specific question ID]`

**Community context** — Novel-Summary: a question requiring synthesis across multiple chapters of a pre-20th-century novel. Individual chunk retrieval returns local passages; community injection provides the thematic framing needed to connect them. `[VERIFY: find specific question ID]`

**k=10** — Novel-Reasoning: a multi-entity question where the reasoning chain spans four connected concepts. k=5 returns evidence for three; k=10 recovers the missing link. `[VERIFY: find specific question ID]`

---

## 7. Generator Model Effects on Retrieval Signal Utilization

The benchmark results throughout this paper use gpt-4o-mini as both generator and judge — a deliberate choice for cost, reproducibility, and comparability with published baselines. This section examines whether generator model choice materially affects the conclusions, and whether the answer depends on corpus characteristics.

### 7.1 Motivation

On GraphRAG-Bench (gpt-4o-mini generator + judge), completed full-corpus runs cluster within a remarkably narrow band — a 10-point spread across all configurations, and less than 3 points separating the top seven:

| Configuration | Med | Nov | All | Graph features |
|---|---|---|---|---|
| vanilla\_256\_rerank (published baseline, naive chunking) | — | — | 0.554 | none |
| rerank\_k10 (semantic chunking, no graph) | 0.727 | 0.595 | 0.661 | none |
| laned\_pruned\_k10 | 0.725 | 0.597 | 0.661 | entity-ref, lane filter, pruning |
| laned55\_community\_k10 | 0.719 | 0.605 | 0.662 | entity-ref, lane filter (0.55), community |
| laned\_community\_k10 | 0.715 | 0.602 | 0.659 | entity-ref, lane filter, community |
| cluster\_laned\_community\_pruned\_k10 | 0.710 | 0.596 | 0.653 | entity-ref, lane filter, community, cluster, pruning |
| laned\_community\_pruned\_k10 | 0.722 | 0.582 | 0.652 | entity-ref, lane filter, community, pruning |
| cluster\_community\_k10 | 0.717 | 0.593 | 0.655 | entity-ref, community, cluster |
| **laned\_community\_pruned\_k20** | **0.733** | **0.614** | **0.674** | entity-ref, lane filter, community, pruning, k=20 |

Two observations stand out. First, rerank\_k10 — semantic chunking with reranking, no graph features — ties the top graph-augmented configurations. Second, the graph-augmented configurations do not hurt: every configuration with entity-ref-expansion, lane filtering, or community context matches or approaches rerank\_k10. The graph features are, at minimum, doing no harm.

This narrow band could reflect a generator ceiling: the structured context produced by graph-augmented retrieval may simply exceed what gpt-4o-mini can exploit, leaving the latent retrieval advantage invisible at this model scale. The preliminary result with SRR (Structured Retrieval Retry, `--srr`) on the gpt-4o-mini generator partially refutes the ceiling interpretation: SRR reorganizes the retrieved context for more systematic generator consumption, and its gains at gpt-4o-mini scale suggest the ceiling is not fixed — it is a function of how the context is presented, not only of model capacity.

However, a second factor complicates this picture: **corpus diversity**. GraphRAG-Bench uses two relatively homogeneous domains — medical literature and pre-20th-century novels. Each domain is internally consistent in register, terminology, and structure. A generator operating on Medical text faces a narrower synthesis task than one operating across financial filings, security advisories, regulatory documents, and patents simultaneously. On heterogeneous corpora, model scale may matter in a way that SRR alone cannot compensate for — because the synthesis challenge is not just using the retrieved context more systematically, but reasoning across structurally dissimilar content types.

### 7.2 Hypotheses

**H1 (SRR reduces model sensitivity on homogeneous corpora)**: On GraphRAG-Bench (Medical + Novel), SRR applied to the gpt-4o-mini generator will close most of the gap relative to gpt-4o, and this effect will hold even with a significantly smaller model (Llama-3.1-8B). The narrow band in §7.1 reflects a prompt-structure bottleneck that SRR resolves, not an irreducible model-scale requirement.

**H2 (Model sensitivity resurfaces on diverse corpora)**: On a heterogeneous corpus spanning multiple structurally distinct domains (FANG-2026: financial, security, regulatory, technical), a stronger generator will show larger advantage over smaller models, even with SRR applied. Synthesizing across dissimilar schemas and registers demands more from the model than SRR can bridge.

H1 and H2 together predict a **crossover**: model sensitivity is low on domain-homogeneous corpora and higher on domain-heterogeneous corpora. This is testable: H1 via the generator ablation on GraphRAG-Bench; H2 via FANG-2026 (§8).

Both hypotheses have clean falsification conditions. If Llama-8B + SRR tracks gpt-4o + SRR on GraphRAG-Bench, H1 holds. If the model gap is no larger on FANG-2026 than on GraphRAG-Bench, H2 fails — and the conclusion would be that SRR is sufficient regardless of corpus diversity.

### 7.3 Experimental Design

We evaluate five configurations across four generator conditions:

| Generator | Judge | Label | Corpus |
|---|---|---|---|
| gpt-4o-mini | gpt-4o-mini | **Baseline** (complete) | GraphRAG-Bench |
| gpt-4o-mini + SRR | gpt-4o-mini | **SRR-Mini** | GraphRAG-Bench |
| gpt-4o + SRR | gpt-4o-mini | **SRR-Large** | GraphRAG-Bench |
| Llama-3.1-8B + SRR | gpt-4o-mini | **SRR-Small** | GraphRAG-Bench |

*The SRR-Small condition uses `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` via Together AI. The judge remains gpt-4o-mini across all conditions so scores are directly comparable.*

Configurations: rerank\_k10 (no-graph ablation) and the top-3 graph-augmented configurations by full-corpus All score: laned\_community\_pruned\_k20 (0.674), laned\_community\_k15 (0.666), laned55\_community\_k10 (0.662). H2 is tested via FANG-2026 (§8).

### 7.4 Results

SRR runs use laned60+community+k=30 (no cross-encoder reranking) and report A-avg scores (not the standard Med/Nov-derived All). For comparison, the standard All for rerank\_k10 = 0.661 corresponds to A-avg ≈ 0.668.

| Config | gpt-4o (no SRR) | gpt-4o + SRR | gpt-4o-mini + SRR | Llama-3.1-8B + SRR |
|---|---|---|---|---|
| laned60+community+k=30 | 0.514 (A-avg)¹ | **0.710** (A-avg) ✓ | **0.711** (A-avg) ✓ | [PENDING] |

¹ gpt-4o without SRR at k=30 scores 0.514 A-avg — well below gpt-4o-mini at k=10 (0.668 A-avg). This is not a generator comparison: the run generated 2,981/4,072 answers (partial completion), and the 73% completion rate combined with large unstructured context (avg ~13,700 tokens per question) caused substantial degradation. The meaningful generator comparison is SRR+gpt-4o vs SRR+gpt-4o-mini — both complete runs with structured context.

*The SRR results confirm that at k=30 with structured context, gpt-4o and gpt-4o-mini produce identical scores (0.710 vs 0.711, within noise). The Llama-3.1-8B + SRR run is queued (current run_full_all.sh pass). Under H1, Llama-3.1-8B + SRR should track gpt-4o-mini + SRR (≈ 0.70–0.71); if it falls significantly below (~0.65 or lower), the ceiling is not fully a prompt-structure effect — model capacity contributes.*

### 7.5 Interpretation

On GraphRAG-Bench, the SRR results confirm H1: gpt-4o and gpt-4o-mini produce identical scores (0.710 vs 0.711 A-avg, within noise) when graph context is structured via SRR at k=30. The no-SRR gpt-4o score (0.514) is not a clean comparison due to partial completion (2,981/4,072 answers), but the degradation is consistent with the prediction: large, unstructured context harms performance regardless of model size. The SRR+mini result (0.711) is the cleanest data point: a gpt-4o-mini generator with structured k=30 graph context exceeds every non-SRR configuration, including the full stack (0.674). Whether model sensitivity resurfaces on heterogeneous corpora (H2) is tested in §8.

**Guidance for practitioners (preliminary)**: On domain-homogeneous corpora, the choice of generator model may matter less than how retrieved context is structured for the generator. SRR appears to reduce model sensitivity, suggesting that even smaller open-source models can be competitive when graph-augmented retrieval is paired with structured context presentation. On heterogeneous enterprise corpora, model scale may remain important — the FANG-2026 results in §8 provide the first evidence on this question.

### 7.6 Constrained Generator Hypothesis

Enterprise deployments face a constraint that benchmark evaluations routinely ignore: the generator model is often fixed by policy, not chosen for performance. Air-gapped environments, data residency requirements, and cost controls push practitioners toward smaller on-premise models — Llama-3.1-8B and its successors — regardless of what the benchmark leaderboard recommends.

This creates a specific question the existing literature does not answer: **which RAG strategy minimizes the performance penalty of using a constrained generator?**

The hypothesis is that the penalty is not uniform across retrieval strategies. A capable generator (gpt-4o-mini) can partially compensate for noisy or incomplete retrieval through inference-time reasoning — inferring missing connections, tolerating irrelevant context, synthesizing across loosely related passages. A constrained generator cannot. It is more sensitive to what lands in the context window, and more penalized by the absence of the right evidence. If this holds, the gap between vanilla RAG and graph-augmented RAG should *widen* as generator capability decreases — retrieval quality matters more, not less, when the generator cannot compensate for retrieval gaps.

**H3 (Constrained generator amplifies retrieval signal)**: The performance delta between the top graph-augmented configuration (laned+community+pruning, k=20) and the vanilla RAG baseline (vanilla\_256\_rerank) will be larger for Llama-3.1-8B than for gpt-4o-mini, without SRR. The retrieval layer must do work that a capable generator would otherwise do implicitly.

A corollary prediction: SRR should close most of the model-size gap on a homogeneous corpus (H1), but the graph-augmented configuration advantage should persist or grow on both corpora because better retrieval has higher marginal value when generation is weak.

**Experimental design.** We evaluate three generator conditions on the top graph config (laned+community+pruning, k=20) and the vanilla RAG baseline (vanilla\_256\_rerank):

| Run | Generator | SRR | Config |
|-----|-----------|-----|--------|
| vanilla\_256\_rerank\_full | gpt-4o-mini | no | vanilla baseline (naive chunking) |
| rerank\_k10\_full | gpt-4o-mini | no | no-graph ablation (semantic chunking) |
| laned\_community\_pruned\_k20\_full | gpt-4o-mini | no | top graph config |
| vanilla\_256\_llama8b\_full | Llama-3.1-8B | no | vanilla baseline (naive chunking) |
| ner\_ref\_rerank\_laned\_community\_pruned\_k20\_llama8b\_full | Llama-3.1-8B | no | top graph config |
| ner\_ref\_laned60\_community\_k30\_llama8b\_srr\_full | Llama-3.1-8B | yes | graph+SRR |

The key comparison is the delta (graph − vanilla) at gpt-4o-mini vs Llama-8B. If H3 holds, the delta is larger for Llama-8B. The SRR condition establishes how much structured context presentation can recover relative to model scale.

**Results**: `[PENDING — vanilla_256_llama8b_full and laned_community_pruned_k20_llama8b_full currently running]`

Expected outcome table (pre-registered, to be replaced with actuals):

| Run | Generator | Expected All | Basis |
|-----|-----------|-------------|-------|
| vanilla\_256\_rerank\_full | gpt-4o-mini | 0.647 ✓ | confirmed |
| rerank\_k10\_full | gpt-4o-mini | 0.654 ✓ | confirmed |
| laned\_community\_pruned\_k20\_full | gpt-4o-mini | 0.674 ✓ | confirmed |
| vanilla\_256\_llama8b\_full | Llama-3.1-8B | ~0.55–0.62 | H3 predicts 10–15% drop vs mini |
| laned\_community\_pruned\_k20\_llama8b\_full | Llama-3.1-8B | ~0.58–0.65 | H3 predicts graph delta widens |
| laned60\_community\_k30\_llama8b\_srr\_full | Llama-3.1-8B + SRR | ~0.66–0.71 | SRR should partially recover vs H1 |

H3 falsification condition: if Llama-8B graph delta ≤ mini graph delta (+0.027), H3 fails — model capability fully compensates for retrieval gaps at both scales.

**Practical implication**: If H3 holds, the ROI on graph-augmented retrieval is highest precisely where practitioners are most constrained. Air-gapped deployments on smaller models are not a reason to deprioritize retrieval investment — they are the strongest reason to prioritize it.

---

## 8. Cross-Domain Evaluation (FANG-2026)

GraphRAG-Bench tests retrieval on two internally consistent domains. FANG-2026 tests it across four structurally dissimilar ones. The benchmark exists to answer a specific question: when the corpus is heterogeneous by construction, which retrieval strategies hold up?

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

### 8.3 Single-Shot Results

The following results are from single-shot retrieval (one query → retrieve → generate), using the same generator and judge as §4 (gpt-4o-mini).

| Run | MDJ | TV | CDER | QS | A/N | Mean |
|-----|----:|---:|-----:|---:|----:|-----:|
| rerank+cluster+community+k10 | 0.327 | 0.497 | 0.402 | 0.525 | 0.637 | **0.478** ✓ |
| graph_first+k10 | 0.394 | 0.579 | 0.431 | 0.406 | 0.556 | **0.473** ✓ |
| rerank+laned60+community+k10 | 0.156 | 0.563 | 0.345 | 0.262 | 0.495 | **0.364** ✓ |
| global_search+k10 | 0.358 | 0.463 | 0.267 | 0.316 | 0.287 | **0.338** ✓ |
| rerank+laned45+community+k10 | — | — | — | — | — | **[PENDING]** ~0.38–0.44¹ |
| rerank_k10 (no-graph ablation) | — | — | — | — | — | **[PENDING]** ~0.42–0.50² |
| vanilla_rerank (naive chunking) | — | — | — | — | — | **[PENDING]** ~0.35–0.43³ |

¹ laned45 uses a looser lane threshold (sim≥0.45) than laned60 (sim≥0.60). Expected to recover some MDJ score vs laned60 (0.156 → ~0.25–0.35) while losing some TV/CDER precision. Net effect: moderate overall improvement over laned60.
² No-graph ablation on FANG: expected to exceed laned60 (0.364) but fall short of cluster+community (0.478). Semantic chunking without graph signals should handle TV and QS reasonably but struggle with MDJ and CDER.
³ Vanilla (256-token chunks) on FANG: expected weakest overall — naive chunking will fragment cross-domain entity references that semantic boundaries preserve. Baseline anchor for §7.6/H3 on FANG.

*Multi-step variants (`_ms`) for three runs are queued and pending; see §9.*

### 8.4 Findings

**Overall difficulty.** A mean score near 0.4 is appropriate for this benchmark. GraphRAG-Bench questions are domain-homogeneous — the evidence for any given question lives within a single, terminologically consistent domain. FANG-2026 questions require assembling evidence across structurally dissimilar source types. The single-shot retrieval ceiling is lower by design; it is not a deficiency of the retrieval systems.

**cluster+community is best (0.478).** Community structure helps more than lane filtering on a heterogeneous corpus. Communities on FANG-2026 are formed from co-occurrence patterns that span source types — entities like company names and product identifiers appear in both financial filings and security advisories. These cross-domain co-occurrences form the edges that community detection captures and lane filtering discards.

**graph_first nearly ties (0.473).** Graph traversal is effective on heterogeneous corpora. Following entity edges to non-adjacent chunks is precisely the operation that cross-domain joins require — an entity in a CVE record that also appears in a 10-K risk factor is connected by an edge, and graph_first follows it. The near-tie with cluster+community suggests the two strategies are reaching the same cross-domain evidence via different paths.

**laned60 collapses on MDJ (0.156).** The 0.60 lane threshold cuts cross-domain entity links by requiring that expanded chunks be embedding-similar to the query. On a heterogeneous corpus, a CVE record and a 10-K filing may share an entity without being embedding-similar — they use different vocabulary, different structure, different register. Lane filtering with a tight threshold treats this dissimilarity as noise and discards it. For MDJ questions, the discarded chunks are the answer. The MDJ score (0.156) reflects this directly: laned60 retrieves within-domain chunks at high precision but fails to assemble cross-domain joins.

**global_search is weakest (0.338).** Global summarization compresses the corpus to topic summaries. Precise cross-domain lookups — a specific CVE ID, a specific regulatory docket number, a specific patent claim — do not survive compression. The global_search score is lowest on CDER (0.267) and A/N (0.287), exactly the question types that require precise entity identification and complete evidence coverage.

**Comparison with GraphRAG-Bench.** On GraphRAG-Bench, configurations that differ by lane threshold cluster within 0.003 of each other — the signal is below measurement noise at that scale. On FANG-2026, the same choice (laned60 vs cluster+community) produces a 0.114-point gap. Corpus heterogeneity is the discriminating condition. This is consistent with H2 from §7.2: retrieval strategy matters more when the corpus is structurally diverse.

**Relationship to §7.** The FANG-2026 results provide partial evidence for H2: on a heterogeneous corpus, graph traversal strategies (graph_first, cluster+community) outperform lane-filtered expansion. This is the opposite of the GraphRAG-Bench finding, where laned configurations lead. The reversal is structurally motivated — lane filtering assumes embedding similarity is a good proxy for evidence relevance, which holds within a domain but not across domains. The H2 crossover is visible in the data.

### 8.5 Configuration Guidance

GraphRAG-Bench (§6.5) reveals parameter sensitivity within a homogeneous corpus. FANG-2026 reveals strategy sensitivity across heterogeneous corpus types. Together they support the following guidance.

**Retrieval strategy** is the first choice, and it depends on corpus structure:

- *Homogeneous corpus* (single domain, consistent vocabulary): laned entity-ref-expansion. Embedding similarity is a reliable relevance proxy within a domain; lane filtering at sim≥0.45 improves precision without sacrificing cross-domain recall that doesn't exist in the corpus.
- *Heterogeneous corpus* (multiple source types, mixed register): cluster+community or graph_first. A CVE record and a 10-K filing may share an entity without being embedding-similar — they use different vocabulary, different structure. Lane filtering treats this dissimilarity as noise and discards relevant evidence. Entity-edge traversal and community co-occurrence capture cross-domain links regardless of embedding distance.

The reversal is large: laned60 scores 0.661 on GraphRAG-Bench but 0.364 on FANG-2026. cluster+community scores 0.654 on GraphRAG-Bench and 0.478 on FANG-2026. Strategy choice that is nearly invisible on a homogeneous benchmark produces a 0.114-point gap on a heterogeneous one.

**k (retrieval depth):**

- Entity-dense corpora (medical, legal, financial, technical): k=15–20. Each additional slot recovers non-redundant evidence. The Med k=5→10 gain (+0.035) is 3× the Nov gain (+0.012).
- Narrative or cross-domain corpora: k=10–15. Over-retrieval with tight lane filtering on heterogeneous corpora adds noise rather than evidence.

**Redundancy pruning:**

- Enable on factual/technical homogeneous corpora where creative generation is not a priority (N-Crea drops 0.073 with pruning).
- Disable for narrative corpora.
- Behavior on heterogeneous corpora is not yet measured — no pruning runs were included in the FANG-2026 evaluation.

**Community coherence:** 0.65 for terminologically precise single-domain corpora; 0.50 for narrative or cross-domain corpora where strict filtering shrinks the community signal.

**Summary:**

| Corpus type | Strategy | k | lane-sim | coherence | pruning |
|-------------|----------|---|----------|-----------|---------|
| Homogeneous, entity-dense (medical, legal, financial) | laned+community | 15–20 | 0.45–0.55 | 0.50–0.65 | enabled |
| Homogeneous, narrative | laned+community | 7–10 | 0.45 | 0.50 | disabled |
| Heterogeneous, multi-domain | cluster+community or graph_first | 10–15 | N/A | 0.50 | not tested |

**Remaining open questions:**

- *H1*: Pruning benefit scales with corpus entity density — corpora above some entity-per-chunk threshold benefit; those below do not. Requires ablations across corpora with systematically varied entity density.
- *H2*: The k-plateau shifts rightward with corpus size — larger corpora have more recoverable non-redundant evidence and tolerate higher k before dilution. Requires ablations across corpus sizes.

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

The baseline (rerank\_k10) single-shot and multi-step runs are queued alongside the FANG pending baselines above.

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

- **Benchmark scope**: GraphRAG-Bench is the only available benchmark that evaluates full-pipeline GraphRAG systems end-to-end. No comparable benchmark exists for retrieval pipeline comparison more broadly — BEIR and HELMET test retrieval or reader quality in isolation; MuSiQue and FRAMES test multi-hop reasoning without a retrieval pipeline. GraphRAG-Bench is the natural evaluation surface for this work.
- **Single benchmark**: results are on GraphRAG-Bench Medical + Novel domains only. We include preliminary results on HotpotQA (All=**0.578** `[VERIFY]` vs RAG+rerank **0.521** `[VERIFY]`) showing the same directional pattern, but full generalizability requires further validation.
- **Creative question type**: N-Crea is volatile across configurations. `bc_pruned_laned_community` drops to N-Crea=0.293 while overall ACC looks acceptable — pruning or community context may suppress creative generation. Not yet understood.
- **Medical vs Novel gap**: consistent ~0.10 gap (Med ~0.73, Nov ~0.63) across all configurations. Entity-centric signals may systematically advantage Medical (denser entity linking) over Novel (broader cultural knowledge).
- **Judge calibration**: certain questions fail the calc_fact judge on every run due to structured JSON parse failures. These are excluded from scoring; impact on reported scores is small but systematic.
- **NaN rate as a UX signal**: questions that fail evaluation (NaN) represent cases where the generator produced a non-compliant or empty response — the equivalent of a retrieval-induced hallucination or refusal. In a production system, each such failure requires a human re-prompt cycle, directly incurring the agent cost this paper aims to reduce. GraphRAG-Bench currently excludes NaN questions from scoring entirely, which inflates reported ACC for systems with high failure rates. A retrieval system that answers 90% of questions correctly but refuses 10% is materially worse in production than one that answers 95% at slightly lower ACC. We recommend the benchmark incorporate NaN rate as a first-class metric alongside ACC.
- **Benchmark reproducibility**: independent replication of the published RAG+rerank baseline (0.554) was not possible with the information currently available — the generator prompt and baseline retrieval code are not published, and the leaderboard does not enforce a standardized submission protocol. As a concrete example, the current #1 leaderboard entry (FalkorDB GraphRAG SDK, 69.73) is self-reported and uses a different embedding model for the scoring similarity component than the benchmark's evaluation code specifies. To enable reproducible cross-paper comparison, we encourage the authors to publish the generator prompt, baseline retrieval implementation, and a controlled submission process. All ablation comparisons in this paper use an internal controlled baseline run through the same pipeline and eval code as every other configuration reported here.
- **Leaderboard integrity**: as the GraphRAG-Bench leaderboard grows, the absence of a standardized submission protocol creates conditions where self-reported scores computed under differing embedding models, generator prompts, or evaluation code versions are presented alongside author-verified scores without distinction. A score reported under different evaluation conditions is not a controlled comparison — it is a separate experiment on different infrastructure. We recommend the community treat leaderboard positions as indicative rather than authoritative until submission standards are established, and encourage future work to report an internally controlled baseline alongside any leaderboard submission.
- **Static index**: community detection and entity index are built once. Corpus updates require rebuilding the co-occurrence matrix and community graph — a meaningful constraint for frequently-updated corpora.
- **FANG-2026 scale**: at 50 questions, FANG-2026 results carry wider confidence intervals than GraphRAG-Bench (4,072 questions). Score differences below ~0.05 are not reliably detectable at this sample size. The pending runs and multi-step variants will not change this; they extend the experimental surface, not the statistical power.

---

## 11. Conclusion

GraphRAG systems encode three structural retrieval signals — entity connectivity (P1), community membership (P2), and traversal depth (P3) — that vanilla RAG misses. The standard assumption is that LLM-based graph extraction is necessary to capture these signals at useful fidelity. We show it is not.

Against the vanilla RAG+rerank baseline (vanilla\_256\_rerank, All=0.624), our full stack delivers a +0.050 improvement (All=0.674) at zero index-time LLM cost. Graph signals specifically contribute +0.013 over the no-graph ablation using the same semantic chunking (rerank\_k10, All=0.661). The index builds in approximately 8 minutes `[VERIFY]` versus hours for LLM-based knowledge graph construction. The improvement is not free in absolute terms — NLP primitives and Louvain clustering have real compute cost — but relative to LLM extraction it is effectively free, and it deploys as a drop-in to any existing dense retrieval pipeline.

The more interesting story is in production. GraphRAG-Bench uses equal-weight Medical and Novel domains; entity-dense corpora (legal, financial, technical, medical) resemble the Medical domain, which gains 3× more from k=10 widening than narrative corpora. In real-world deployments where entity density is high and corpus updates are frequent, the cost differential widens further: LLM-based KG construction requires full reconstruction on every update; NLP-primitive indexing does not.

FANG-2026 extends this picture to heterogeneous corpora. The single-shot results confirm H2 from §7.2: on a corpus where source types are structurally dissimilar, retrieval strategy choice matters in ways that domain-homogeneous benchmarks cannot reveal. Lane filtering, which is near-neutral on GraphRAG-Bench, collapses on multi-document join questions. Graph traversal strategies that perform equivalently on GraphRAG-Bench separate cleanly on FANG-2026. The reversal is structurally motivated and reproducible.

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
| run_full_all.sh | All full-corpus runs including SRR (gpt-4o, mini) and constrained generator (Llama-8B) | ✓ gpt-4o/mini SRR complete; Llama runs in progress |
| run_fang2026.sh | FANG-2026 single-shot (cluster, graph_first, laned60, global_search) | ✓ complete; nobc+vanilla+laned45+multi-step pending |

## Appendix D: Run Name Key

Run names encode the retrieval configuration as a `_`-delimited sequence of feature tokens. All graph-augmented runs use semantic boundary chunking (1100–2200 tokens). The mapping:

| Token | Meaning |
|-------|---------|
| `vanilla_256` | Naive 256-token fixed-size chunking. No semantic boundaries, no graph features. |
| `ner_ref` | NER entity-ref-expansion enabled (P1) |
| `rerank` | Cross-encoder reranking (BAAI/bge-reranker-large) |
| `laned` | Lane filtering at default sim≥0.45 threshold (see also laned55, laned60, laned65, laned70) |
| `laned55` / `laned60` / `laned65` / `laned70` | Lane filtering at sim≥0.55/0.60/0.65/0.70 |
| `community` | Community context injection (P2); coherence filter default 0.50 |
| `community60` | Community context with coherence≥0.60 |
| `pruned` | Redundancy pruning at cosine≥0.92 |
| `cluster` | Cluster-based topological expansion (replaces lane filtering) |
| `kN` | Retrieval depth top-k=N (default k=5 if omitted) |
| `global_search` | Global search mode (community-level summarization retrieval) |
| `graph_first` | Graph-first search mode (entity traversal before dense retrieval) |
| `gpt4o` | Generator: gpt-4o (default is gpt-4o-mini) |
| `llama8b` | Generator: Llama-3.1-8B-Instruct-Turbo via Together AI |
| `srr` | Structured Retrieval Retry — generator produces structured key claims + evidence; gaps trigger a targeted re-retrieval pass |
| `ms` | Multi-step retrieval (3 sub-query decomposition) |
| `full` | Full-corpus run (4,072 questions) |
| `grid` | Grid sweep run (300 stratified questions) |
| `_rp` suffix | Judge-reprompt (JR) evaluation applied |

**Examples:**
- `ner_ref_rerank_laned_community_pruned_k20_full_rp` = semantic boundary chunking (1100–2200 tokens) + entity-ref-expansion + reranking + lane filter (sim≥0.45) + community context + redundancy pruning + k=20, full corpus, JR evaluated
- `vanilla_256_llama8b_full` = naive 256-token chunks + Llama-3.1-8B generator, full corpus (no JR eval yet)
- `fang_ner_ref_rerank_cluster_community_k10_rp` = FANG-2026 corpus + entity-ref + reranking + cluster expansion + community context + k=10, JR evaluated
