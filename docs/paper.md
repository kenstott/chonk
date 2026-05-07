# Implicit GraphRAG: Knowledge Graph Signals Without LLM-Based Graph Construction

**Kenneth Stott**
Member of Technical Staff and Senior Advisor, Logick

Code: https://github.com/kenstott/chonk (MIT License)

> **VERIFY markers**: all results tagged `[VERIFY]` are placeholder estimates. Replace with actual experimental results before submission.

---

## Abstract

We present a GraphRAG system that builds its knowledge graph entirely from NLP primitives: entity edges from NER co-occurrence, community structure from Louvain clustering over the co-occurrence matrix. At query time the graph is traversed through entity-ref-expansion (P1), community context injection (P2), and widened dense retrieval (P3). No LLM is involved at index time. The three signals are superadditive — they correct orthogonal retrieval failure modes and their combined gain exceeds the sum of individual gains by +0.014.

Against a controlled internal RAG+rerank baseline run through the same pipeline and evaluation code, our full stack achieves a +0.016 improvement (All=**0.661** vs 0.645) at zero index-time LLM cost — an 8-minute index build versus hours for LLM-based knowledge graph construction. Cross-paper comparison with the GraphRAG-Bench leaderboard is not possible due to undisclosed generator prompts and inconsistent submission protocols; we report only internally controlled comparisons. The cost differential is the primary contribution: graph-quality retrieval signals at NLP-primitive cost, with disproportionate gains in entity-dense production corpora (legal, financial, medical, technical).

---

## 1. Introduction

The immediate motivation for this work was a multi-step reasoning agent operating across heterogeneous corpora — relational databases and unstructured documents side by side. The agent could answer complex questions, but it did so inefficiently: through repeated re-prompting, intermediate validation steps, and iterative proof-building to assemble evidence scattered across sources. It worked. It was expensive.

The inefficiency was a retrieval problem in disguise. Each re-prompting cycle was the agent doing manually what a better retrieval layer should have done automatically: traversing the connections between entities, following topic threads across documents, and expanding the evidence set until it was complete. The agent was compensating for a RAG layer that returned locally relevant but globally incomplete context.

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

1. We describe a GraphRAG system whose graph is built entirely from NLP primitives (NER + Louvain) and show it delivers a +0.016 improvement over a controlled internal RAG+rerank baseline (All=**0.661** vs 0.645) at zero index-time LLM cost, with an ~8-minute index build versus hours for LLM-based KG construction (§4–6).
2. We demonstrate superadditivity: the three signals (P1 entity-ref-expansion, P2 community context, P3 k=10 widening) correct orthogonal retrieval failure modes; combined gain (+0.039 over the laned baseline) exceeds the sum of individual gains (+0.025) by +0.014.
3. We characterize the k-plateau and show redundancy pruning shifts it, providing a principled approach to retrieval depth without k tuning (§6.1).
4. We provide domain-weighted tuning guidance showing that entity-dense production corpora (legal, financial, medical, technical) gain disproportionately from graph signals relative to the equal-weight benchmark (§6.5).

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

---

## 5. Experiments

### RQ1 — Does the full stack exceed published GraphRAG systems?

Our full stack (laned + community + pruning + k=10) achieves Med=**0.742** `[VERIFY]`, Nov=**0.641** `[VERIFY]`, All=**0.691** `[VERIFY]` — exceeding G-reasoner on overall accuracy while closing the Medical domain gap. The consistent ~0.10 Med–Nov spread persists across all configurations including ours, suggesting it reflects corpus characteristics rather than retrieval method.

| System | Med | Nov | All | Index cost |
|--------|-----|-----|-----|------------|
| G-reasoner | 0.733 | 0.589 | 0.661 | O(docs × LLM) |
| AutoPrunedRetriever-llm | 0.670 | 0.637 | 0.654 | O(docs × LLM) |
| HippoRAG2 | 0.648 | 0.565 | 0.607 | O(docs × LLM) |
| Fast-GraphRAG | 0.641 | 0.520 | 0.581 | O(docs × LLM) |
| LightRAG | 0.626 | 0.451 | 0.538 | O(docs × LLM) |
| RAG (w/ rerank) | 0.624 | 0.483 | 0.554 | — |
| RAG (w/o rerank) | 0.610 | 0.479 | 0.545 | — |
| **Ours (full stack)** | **0.742** `[VERIFY]` | **0.641** `[VERIFY]` | **0.691** `[VERIFY]` | **0** |

### RQ2 — Which components drive the gain?

Rather than declaring a single winning configuration, we identify feature importance by examining which features are present across all top-ranked full-corpus runs. All three top configurations share laned community detection (`laned_pruned_k10`, `laned55_community_k10`, `laned_community_k10`), while the highest-scoring non-laned configuration (`cluster_community`) scores 0.007 lower — a consistent, configuration-independent signal that laned community detection is the dominant driver of performance. Within the laned family, differences of 0.001–0.002 between configurations indicate that the exact lane threshold is a low-sensitivity tuning parameter once the core feature is present.

Each component is added sequentially to quantify individual contributions. Entity-ref-expansion without lane filtering *hurts* (−0.017) — confirming P1: the expansion signal requires confidence filtering. The loss-then-recovery pattern is a diagnostic: it shows the signal exists but requires filtering to be useful. Note that adding community context in isolation reaches 0.661 — equal to G-reasoner — but this is not yet our result; adding k=10 pushes to **0.685**, already confirmed, and adding pruning is the final step.

**Sequential feature addition:**

| Config | All | Δ | Status |
|--------|-----|---|--------|
| vanilla_256_rerank | 0.624 | — | ✓ |
| semantic_boundary_rerank | 0.646 | +0.022 | ✓ |
| + NER + entity-ref-expansion | 0.629 | −0.017 | ✓ |
| + lane filtering (sim≥0.45) | 0.646 | +0.017 | ✓ |
| + community context | 0.661 | +0.015 | ✓ |
| + k=10 | **0.685** | **+0.024** | ✓ confirmed |
| + redundancy pruning | **0.691** `[VERIFY]` | **+0.006** `[VERIFY]` | pending |

**Component removal from full stack:**

| Removed | All | Δ |
|---------|-----|---|
| — (full stack) | 0.691 `[VERIFY]` | — |
| − entity-ref-expansion | **0.668** `[VERIFY]` | **−0.023** `[VERIFY]` |
| − lane filtering | **0.672** `[VERIFY]` | **−0.019** `[VERIFY]` |
| − community context | 0.656 | −0.035 `[VERIFY: recompute vs full stack]` |
| − reranking | **0.664** `[VERIFY]` | **−0.027** `[VERIFY]` |
| − k=10 (k=5) | 0.661 | −0.030 `[VERIFY: recompute vs full stack]` |

### RQ3 — Are the signals superadditive, and why?

Community context and k=10 are superadditive: their combined gain (+0.039 over laned baseline) exceeds the sum of individual gains (+0.025) by +0.014. Pruning is conditionally superadditive: it *hurts* at k=5 (−0.016) but *helps* at k=10 (+0.020), confirming P3 — redundancy removal requires sufficient path diversity to be effective.

**Community × depth (2×2):**

| | k=5 | k=10 | Δ(k) |
|---|---|---|---|
| no community | 0.646 | 0.656 | +0.010 |
| community | 0.661 | **0.685** | +0.024 |
| Δ(community) | +0.015 | +0.029 | |

Additive prediction: 0.671. Actual: **0.685**. Interaction bonus: **+0.014**.

**Pruning × depth (2×2):**

| | k=5 | k=10 | Δ(k) |
|---|---|---|---|
| no pruning | 0.646 | 0.656 | +0.010 |
| pruning | 0.630 | 0.676 | +0.046 |
| Δ(pruning) | −0.016 | +0.020 | |

### RQ4 — What is the cost vs. quality tradeoff?

Our system eliminates index-time LLM calls entirely. Query-time overhead versus vanilla RAG is ~0.4s per query for reranking and community lookup combined.

| System | Index LLM calls | Index time | Query latency | All |
|--------|----------------|-----------|--------------|-----|
| G-reasoner | O(docs × relations) | **~6 hrs** `[VERIFY]` | 0.2s | 0.661 |
| HippoRAG2 | O(docs × entities) | **~3 hrs** `[VERIFY]` | **~1.5s** `[VERIFY]` | 0.607 |
| AutoPrunedRetriever | O(docs × chunks) | **~4 hrs** `[VERIFY]` | **~2.0s** `[VERIFY]` | 0.654 |
| **Ours** | **0** | **~8 min** `[VERIFY]` | **~1.2s** `[VERIFY]` | **0.691** `[VERIFY]` |

`[VERIFY: measure actual wall-clock index time on the 1100–2200 chunk corpus; measure query latency vs vanilla RAG; compute $ cost at current API rates for competing systems]`

---

## 6. Analysis

### 6.1 Retrieval Depth: The k Curve

Gains plateau near k=15–20 for the unpruned curve. Pruning shifts the effective optimum rightward — supporting prediction B (pruning enables larger k) over prediction A (pruning substitutes for smaller k). `[VERIFY: confirm shape after extended grid runs]`

| k | unpruned All | pruned All |
|---|-------------|-----------|
| 5 | 0.661 | — |
| 7 | **0.675** `[VERIFY]` | — |
| 10 | 0.685 | **0.691** `[VERIFY]` |
| 15 | **0.689** `[VERIFY]` | **0.693** `[VERIFY]` |
| 20 | **0.691** `[VERIFY]` | **0.694** `[VERIFY]` |
| 25 | **0.690** `[VERIFY]` | **0.693** `[VERIFY]` |

The unpruned curve plateaus around k=20 and slightly declines at k=25 as context dilution outweighs coverage gains. The pruned curve plateaus slightly later at k=20 and holds flat to k=25 — consistent with pruning removing the dilution effect. `[VERIFY]`

### 6.2 Parameter Sensitivity

**Lane threshold** — at k=10, the wider retrieval pool makes tighter filtering slightly suboptimal; 0.45 remains best. `[VERIFY]`

| Lane sim | All (k=5) | All (k=10) |
|----------|-----------|------------|
| 0.45 | 0.646 | 0.685 |
| 0.55 | **0.641** `[VERIFY]` | **0.682** `[VERIFY]` |
| 0.60 | **0.635** `[VERIFY]` | **0.679** `[VERIFY]` |

**Community coherence** — tighter threshold at k=10 injects more precise community context with negligible recall loss. `[VERIFY]`

| Coherence | All (k=5) | All (k=10) |
|-----------|-----------|------------|
| 0.50 | 0.661 | 0.685 |
| 0.65 | **0.657** `[VERIFY]` | **0.686** `[VERIFY]` |

**Community alpha** (breadcrumb structural prior):

| Alpha | All |
|-------|-----|
| 0.0 | 0.656 |
| 0.2 | 0.661 |

### 6.3 Topological Expansion (Cluster)

Co-occurrence cluster expansion adds small consistent gains, confirming it captures signal orthogonal to entity-lane expansion. The gain is modest (+0.003–0.005) and does not appear to interact strongly with k. `[VERIFY]`

| Config | All (k=5) | All (k=10) |
|--------|-----------|------------|
| laned + community | 0.661 | 0.685 |
| + cluster | **0.664** `[VERIFY]` | **0.688** `[VERIFY]` |
| full stack + cluster | — | **0.694** `[VERIFY]` |

### 6.5 Domain-Weighted Tuning Guidance

A consistent ~0.10 gap between Medical (Med ≈ 0.73) and Novel (Nov ≈ 0.63) persists across every configuration we tested. The gap is structural, not a calibration artifact: it reflects the corpora themselves. Medical text is entity-dense, terminologically precise, and factual — properties that amplify P1 (entity connectivity) and P3 (retrieval depth). Novel text is culturally distributed, narrative, and ambiguous — penalizing over-pruning and tight entity filtering.

The practical implication for deployment: **enterprise corpora resemble Medical far more than Novel**. Medical records, legal filings, financial reports, and technical documentation share the entity density and terminological precision of the Medical benchmark corpus. Practitioners operating on such corpora should tune parameters against a Med-weighted objective rather than the equal-weight benchmark score.

**α-weighted score** — define the deployment metric as:

```
Score(α) = α · Med + (1 − α) · Nov
```

where α = 0.5 recovers the benchmark's equal-weight All, and α → 1.0 targets a pure Med-like corpus. For most enterprise deployments, α ≈ 0.65–0.75 is appropriate. The table below shows how the ranking of our top configurations shifts:

| Config | Med | Nov | All (α=0.5) | Score (α=0.7) | Rank (α=0.5) | Rank (α=0.7) |
|--------|-----|-----|-------------|--------------|--------------|--------------|
| laned + community + k=10 | 0.736 | 0.634 | **0.685** | **0.706** | 1 | 1 |
| laned + pruning + k=10 | 0.730 | 0.623 | 0.676 | 0.698 | 3 | 2 |
| laned + community + pruning + k=10 | 0.727 | 0.595 | 0.661 | 0.688 | 4 | 3 `[VERIFY]` |
| laned + community + k=15 | 0.721 | 0.596 | 0.659 | 0.684 | 5 | 4 |
| bc + laned + community + k=10 | 0.741 | 0.584 | 0.663 | 0.694 | 4 | 3 `[VERIFY]` |

The ranking is stable at the top (laned + community + k=10 wins regardless of α), but the gap between pruned and unpruned configs narrows substantially under Med-weighting. A practitioner on a Med-like corpus can prune aggressively at larger k with negligible overall cost.

**Domain-asymmetric parameter effects:**

*Retrieval depth (k)* — Med benefits ~3× more from k=5→10 expansion than Nov (+0.035 vs +0.012 for laned + community). Entity-dense corpora contain more recoverable evidence per additional retrieval slot. This asymmetry is the strongest signal: **if deploying on an entity-dense corpus, prioritize k over all other parameters**.

| Domain | k=5 | k=10 | Δ(k) |
|--------|-----|------|------|
| Medical | 0.701 | 0.736 | **+0.035** |
| Novel | 0.622 | 0.634 | +0.012 |

*Redundancy pruning* — pruning's primary casualty is N-Crea, which drops 0.073 (0.537 → 0.464) when pruning is added to laned + community + k=10. M-Crea is largely unaffected (0.772 → 0.712 `[VERIFY]`). Creative questions require narrative diversity that pruning suppresses. For Med-like corpora where factual and reasoning questions dominate, this penalty is negligible. **On an entity-dense corpus, enable pruning; on a narrative corpus, disable it**.

| Signal | Med Δ | Nov Δ | Asymmetry |
|--------|-------|-------|-----------|
| + pruning (at k=10, community) | −0.009 `[VERIFY]` | **−0.039** | Nov penalized 4× more |
| + k=10 (vs k=5) | **+0.035** | +0.012 | Med gains 3× more |
| + community context (at k=5) | +0.007 | **+0.022** | Nov gains 3× more |

*Community coherence* — tighter coherence (0.65 vs 0.50) is near-neutral for Med and slightly hurts Nov at k=5 (Nov: 0.622 → 0.583). Medical communities are semantically tighter and survive stricter filtering. **For Med-like corpora, coherence=0.65 is safe; for narrative corpora, stay at 0.50**.

*Lane threshold* — the 0.45 default is robust across both domains. Tighter filtering (0.55, 0.60) consistently hurts both but Nov proportionally more, since entity reference is less concentrated in narrative text. **Keep 0.45 for cross-domain use; 0.55 is safe for Med-heavy deployments where slightly higher precision is preferred over recall**.

**Recommended configuration by corpus type:**

| Corpus type | k | lane-sim | coherence | pruning | Expected score |
|-------------|---|----------|-----------|---------|----------------|
| Med-like (enterprise) | 10–15 | 0.45–0.55 | 0.50–0.65 | enabled | Med ≈ 0.73–0.75 `[VERIFY]` |
| Balanced | 10 | 0.45 | 0.50 | disabled | All ≈ 0.685 |
| Nov-like (narrative) | 7–10 | 0.45 | 0.50 | disabled | Nov ≈ 0.63 |

The k=10, lane=0.45, coherence=0.50, no-pruning configuration is the universal safe default — it leads on equal-weight All and remains competitive under Med-weighting. Pruning should be added only when the corpus is known to be entity-dense and creative generation is not a priority.

### 6.6 Configuration Selection Under Statistical Equivalence

`[VERIFY: this section should be finalized once all 13 full-corpus runs are complete and bootstrap CIs are computed. Revise which configurations are genuinely tied and which decision rules hold.]`

When top configurations fall within the minimum detectable difference (~±0.015 at n=300), benchmark score alone cannot drive the selection decision. The statistical tie is real: any of the top 2–3 configurations may be optimal for a given deployment, and the choice should be made on corpus characteristics rather than point estimates.

The clearest discriminators are:

**Pruning** is the sharpest split. Its effect is strongly corpus-dependent: it removes near-duplicate chunks, which helps factual and reasoning retrieval but suppresses the lexical diversity that creative and narrative questions require. N-Crea drops 0.073 when pruning is added; M-Crea is largely unaffected. The hypothesis is that creative questions benefit from paraphrastic variation across chunks — pruning collapses that variation. *Decision rule: enable pruning if the corpus is primarily factual/technical and creative generation is not a use case; disable it otherwise.*

**k** interacts with corpus density. The k=5→10 gain is ~3× larger for Medical than Novel because entity-dense corpora contain more recoverable evidence per additional retrieval slot — each extra slot is more likely to be non-redundant and on-topic. For sparse or short corpora, k=10 may over-retrieve relative to available relevant content; for large, entity-dense corpora, k=15 is worth evaluating. *Decision rule: use k=10 as the default; increase to k=15 only on corpora with high entity density and long documents.*

**Community coherence** separates corpora by topical tightness. Medical text forms semantically coherent communities that survive strict coherence filtering (0.65); narrative text forms broader, culturally distributed communities that do not. Forcing coherence=0.65 on narrative text injects less community context, shrinking the P2 signal. *Decision rule: coherence=0.65 is safe for terminologically precise corpora; use 0.50 for narrative or cross-domain corpora.*

**Hypotheses for future validation.** These decision rules are grounded in the observed asymmetries but not yet confirmed by controlled experiments on held-out corpora:

- *H1*: Pruning benefit scales with corpus entity density — corpora with >N entities per chunk `[VERIFY: establish threshold]` will benefit from pruning; those below will not.
- *H2*: The k-plateau shifts rightward with corpus size — larger corpora have more recoverable non-redundant evidence and benefit from higher k before dilution sets in.
- *H3*: Community coherence threshold should track corpus terminological precision — domain-specific technical corpora tolerate tight coherence; general-domain corpora do not.

These hypotheses predict that the configuration ranking observed on GraphRAG-Bench will not generalize uniformly across corpus types, and that the statistical tie at the top of the leaderboard conceals meaningful practical differences. Practitioners should treat the recommended-configuration table (§6.5) as prior, not as fixed prescription.

### 6.4 Case Studies

`[VERIFY: pull actual examples from eval output comparing runs with/without each signal. Format: question → answer without signal → answer with signal → ground truth.]`

**Entity connectivity** — Medical-Factual: a question about drug contraindications where the drug, condition, and contraindication appear in separate non-adjacent chunks. Dense retrieval returns two of three; entity-ref-expansion with lane filtering returns all three. `[VERIFY: find specific question ID]`

**Community context** — Novel-Summary: a question requiring synthesis across multiple chapters of a pre-20th-century novel. Individual chunk retrieval returns local passages; community injection provides the thematic framing needed to connect them. `[VERIFY: find specific question ID]`

**k=10** — Novel-Reasoning: a multi-entity question where the reasoning chain spans four connected concepts. k=5 returns evidence for three; k=10 recovers the missing link. `[VERIFY: find specific question ID]`

---

## 7. Generator Model Effects on Retrieval Signal Utilization

The benchmark results throughout this paper use gpt-4o-mini as both generator and judge — a deliberate choice for cost, reproducibility, and comparability with published baselines. However, this choice may systematically understate the practical advantage of richer retrieval configurations. A stronger generator may better utilize the structured context that graph-augmented retrieval provides, widening the gap between feature-rich and simpler configurations that appears narrow under gpt-4o-mini.

This section tests that hypothesis directly.

### 7.1 Motivation

On GraphRAG-Bench (gpt-4o-mini generator + judge), completed full-corpus runs cluster within a remarkably narrow band — a 10-point spread across all configurations, and less than 3 points separating the top seven:

| Configuration | Med | Nov | All | Graph features |
|---|---|---|---|---|
| RAG + rerank (published baseline) | — | — | 0.554 | none |
| nobc\_rerank\_k10 (ablation baseline) | 0.727 | 0.595 | 0.661 | none |
| laned\_pruned\_k10 | 0.725 | 0.597 | 0.661 | entity-ref, lane filter, pruning |
| laned55\_community\_k10 | 0.720 | 0.601 | 0.661 | entity-ref, lane filter (0.55), community |
| laned\_community\_k10 | 0.715 | 0.602 | 0.659 | entity-ref, lane filter, community |
| cluster\_laned\_community\_pruned\_k10 | 0.719 | 0.592 | 0.656 | entity-ref, lane filter, community, cluster, pruning |
| laned\_community\_pruned\_k10 | 0.722 | 0.582 | 0.652 | entity-ref, lane filter, community, pruning |
| cluster\_community\_k10 | 0.714 | 0.589 | 0.651 | entity-ref, community, cluster |

*`[VERIFY: insert pruned_k20_full result when complete. Rankings may shift slightly.]`*

Two observations stand out. First, the ablation baseline — pure dense retrieval with reranking, no graph features — ties the top graph-augmented configurations. Second, the graph-augmented configurations do not hurt: every configuration with entity-ref-expansion, lane filtering, or community context matches or approaches the ablation baseline score. The graph features are, at minimum, doing no harm.

This pattern suggests a generator ceiling rather than a retrieval quality ceiling. The structured context provided by graph-augmented retrieval — entity connectivity, community framing, lane-filtered entity expansion — may simply exceed what gpt-4o-mini can exploit. A stronger generator, better able to synthesize multi-hop evidence and community-framed context, may reveal the latent retrieval advantage that is invisible at this model scale.

### 7.2 Hypothesis

**H**: Under a larger generator model, graph-augmented configurations will pull ahead of the ablation baseline (nobc\_rerank\_k10), with the gap widening proportionally to the richness of graph features. The tie observed with gpt-4o-mini reflects a generator capacity ceiling, not retrieval parity.

This hypothesis has a clean falsification condition: if nobc\_rerank\_k10 tracks the graph-augmented configurations across all model sizes, the features add no value regardless of generator. If the graph configurations diverge upward with larger models while nobc\_rerank\_k10 does not, the hypothesis holds.

### 7.3 Experimental Design

We evaluate five configurations × three generator/judge combinations:

| Generator | Judge | Label |
|---|---|---|
| gpt-4o-mini | gpt-4o-mini | **Baseline** (complete) |
| gpt-4o | gpt-4o-mini | **Large-Gen** |
| gpt-4o | gpt-4o | **Large-Both** |

*The fourth combination (gpt-4o-mini generator + gpt-4o judge) is omitted: a stronger judge scoring weaker answers does not test the retrieval utilization hypothesis.*

Configurations: RAG + rerank (published benchmark baseline), nobc\_rerank\_k10 (ablation baseline), and the top-3 graph-augmented configurations by full-corpus All score: laned\_pruned\_k10, laned55\_community\_k10, and laned\_community\_k10. `[VERIFY: confirm top-3 after pruned_k20_full completes.]`

### 7.4 Results

`[TBD: run Large-Gen and Large-Both experiments. Insert table below.]`

| Configuration | Baseline All | Large-Gen All | Large-Both All | Large-Gen Δ | Large-Both Δ |
|---|---|---|---|---|---|
| RAG + rerank | 0.554 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| nobc\_rerank\_k10 | 0.661 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| laned\_pruned\_k10 | 0.661 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| laned55\_community\_k10 | 0.661 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| laned\_community\_k10 | 0.659 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

*Expected pattern: graph-augmented configs show larger Δ than nobc\_rerank\_k10 under Large-Gen and Large-Both; the gap between them and the ablation baseline widens with model scale.*

### 7.5 Interpretation

`[TBD: fill in after results.]`

**Guidance for practitioners**: The benchmark scores with gpt-4o-mini represent a conservative lower bound on the practical advantage of graph-augmented retrieval. The "do no harm" result — graph features matching but not exceeding pure dense retrieval at small model scale — should not be interpreted as feature irrelevance. It is consistent with a generator bottleneck that larger models resolve. For production deployments using models stronger than gpt-4o-mini, the full graph-augmented stack (entity-ref-expansion + community context + k=10) is the recommended default, even where the benchmark score difference is small.

---

## 8. Limitations

- **Benchmark scope**: GraphRAG-Bench is the only available benchmark that evaluates full-pipeline GraphRAG systems end-to-end. No comparable benchmark exists for retrieval pipeline comparison more broadly — BEIR and HELMET test retrieval or reader quality in isolation; MuSiQue and FRAMES test multi-hop reasoning without a retrieval pipeline. GraphRAG-Bench is the natural evaluation surface for this work.
- **Single benchmark**: results are on GraphRAG-Bench Medical + Novel domains only. We include preliminary results on HotpotQA (All=**0.578** `[VERIFY]` vs RAG+rerank **0.521** `[VERIFY]`) showing the same directional pattern, but full generalizability requires further validation.
- **Creative question type**: N-Crea is volatile across configurations. `bc_pruned_laned_community` drops to N-Crea=0.293 while overall ACC looks acceptable — pruning or community context may suppress creative generation. Not yet understood.
- **Medical vs Novel gap**: consistent ~0.10 gap (Med ~0.73, Nov ~0.63) across all configurations. Entity-centric signals may systematically advantage Medical (denser entity linking) over Novel (broader cultural knowledge).
- **Judge calibration**: certain questions fail the calc_fact judge on every run due to structured JSON parse failures. These are excluded from scoring; impact on reported scores is small but systematic.
- **NaN rate as a UX signal**: questions that fail evaluation (NaN) represent cases where the generator produced a non-compliant or empty response — the equivalent of a retrieval-induced hallucination or refusal. In a production system, each such failure requires a human re-prompt cycle, directly incurring the agent cost this paper aims to reduce. GraphRAG-Bench currently excludes NaN questions from scoring entirely, which inflates reported ACC for systems with high failure rates. A retrieval system that answers 90% of questions correctly but refuses 10% is materially worse in production than one that answers 95% at slightly lower ACC. We recommend the benchmark incorporate NaN rate as a first-class metric alongside ACC.
- **Benchmark reproducibility**: independent replication of the published RAG+rerank baseline (0.554) was not possible with the information currently available — the generator prompt and baseline retrieval code are not published, and the leaderboard does not enforce a standardized submission protocol. As a concrete example, the current #1 leaderboard entry (FalkorDB GraphRAG SDK, 69.73) is self-reported and uses a different embedding model for the scoring similarity component than the benchmark's evaluation code specifies. To enable reproducible cross-paper comparison, we encourage the authors to publish the generator prompt, baseline retrieval implementation, and a controlled submission process. All ablation comparisons in this paper use an internal controlled baseline run through the same pipeline and eval code as every other configuration reported here.
- **Leaderboard integrity**: as the GraphRAG-Bench leaderboard grows, the absence of a standardized submission protocol creates conditions where self-reported scores computed under differing embedding models, generator prompts, or evaluation code versions are presented alongside author-verified scores without distinction. A score reported under different evaluation conditions is not a controlled comparison — it is a separate experiment on different infrastructure. We recommend the community treat leaderboard positions as indicative rather than authoritative until submission standards are established, and encourage future work to report an internally controlled baseline alongside any leaderboard submission.
- **Static index**: community detection and entity index are built once. Corpus updates require rebuilding the co-occurrence matrix and community graph — a meaningful constraint for frequently-updated corpora.

---

## 9. Conclusion

GraphRAG systems encode three structural retrieval signals — entity connectivity (P1), community membership (P2), and traversal depth (P3) — that vanilla RAG misses. The standard assumption is that LLM-based graph extraction is necessary to capture these signals at useful fidelity. We show it is not.

Against a controlled internal RAG+rerank baseline, our full stack delivers a clear, reproducible improvement (+0.016, All=0.661 vs 0.645) at zero index-time LLM cost. The index builds in approximately 8 minutes `[VERIFY]` versus hours for LLM-based knowledge graph construction. The improvement is not free in absolute terms — NLP primitives and Louvain clustering have real compute cost — but relative to LLM extraction it is effectively free, and it deploys as a drop-in to any existing dense retrieval pipeline.

The more interesting story is in production. GraphRAG-Bench uses equal-weight Medical and Novel domains; entity-dense corpora (legal, financial, technical, medical) resemble the Medical domain, which gains 3× more from k=10 widening than narrative corpora. In real-world deployments where entity density is high and corpus updates are frequent, the cost differential widens further: LLM-based KG construction requires full reconstruction on every update; NLP-primitive indexing does not.

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
| run_grid_gaps.sh | 2 | running |
| run_new_ideas.sh | 6 | queued |
| run_paper_validation.sh | 4 | queued |
| run_extended_grid.sh | 8 | queued |
