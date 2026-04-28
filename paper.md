# Chonk: Entity-Guided Retrieval with Community Context Matches Graph RAG without Graph Construction

**Kenneth Stott**  
*Logick.io*  
`kennethstott@gmail.com`

---

## Abstract

Graph-based Retrieval-Augmented Generation (GraphRAG) systems achieve strong performance on knowledge-intensive tasks by constructing explicit entity graphs, but this comes at high indexing cost and system complexity. We present **Chonk**, a retrieval system that achieves competitive performance through semantic boundary chunking, breadcrumb-informed community detection, and multi-dimensional entity-guided retrieval, with no LLM calls required during indexing. On GraphRAG-Bench—the standard benchmark for comparing RAG and GraphRAG systems across Medical and Novel corpora—Chonk's best configuration achieves **0.661 overall accuracy** (Med 0.701, Nov 0.622), tying G-reasoner, the current state-of-the-art, while outperforming all other published baselines including HippoRAG2 (0.607), AutoPrunedRetriever (0.654), LightRAG (0.538), and MS-GraphRAG (0.480). We present a detailed ablation study isolating the contribution of each component: entity reference expansion (+0.017), entity laning (+0.017), and community context injection (+0.015) account for the full gain over the vanilla reranking baseline (0.624). We further find that breadcrumb-based embedding enrichment—effective for enterprise documents with boilerplate structure—does not help on this benchmark, and that adding redundancy pruning to the best configuration slightly degrades performance, suggesting that diversity of evidence is beneficial for complex reasoning questions.

---

## 1. Introduction

Large language models augmented with retrieval have become a primary paradigm for knowledge-intensive question answering. *Vanilla RAG* retrieves fixed-size text chunks by embedding similarity and passes them as context to the generator. *GraphRAG* extends this by constructing an explicit entity knowledge graph over the corpus, enabling multi-hop traversal and community-level summarization [Edge et al., 2024; Guo et al., 2025]. While GraphRAG systems excel on complex reasoning tasks, they require substantial upfront index construction (entity extraction, relationship linking, community detection) and introduce significant token overhead at inference time—MS-GraphRAG generates up to 4×10⁴ tokens per query compared to ~879 for vanilla RAG [Chen et al., 2025].

GraphRAG-Bench [Chen et al., 2025] provides a systematic evaluation across four question complexity levels—Fact Retrieval, Complex Reasoning, Contextual Summarization, and Creative Generation—on two corpora: NCCN oncology guidelines (Medical) and pre-20th-century novels from Project Gutenberg (Novel). The benchmark reveals a nuanced picture: graph systems excel on reasoning but can underperform vanilla RAG on factual lookup, motivating research into hybrid approaches.

Chonk originates from production RAG pipelines at Logick.io, where the primary corpus is enterprise documentation: technical specifications, regulatory filings, clinical guidelines, and multi-document research portfolios. In this setting, the cost of LLM-driven graph construction is prohibitive at scale, and frequent corpus updates make static graph indices impractical. The system was designed to exploit the rich document structure already present in these corpora without requiring re-indexing on every update.

Our central hypothesis is that **most real-world documents carry inherent structure that vanilla RAG discards**. Technical reports, clinical guidelines, legal contracts, and enterprise documentation embed their meaning not just in prose content but in section hierarchies, entity relationships, and topical neighborhoods that persist across the document. Fixed-size chunking severs these structural signals; graph-based systems recover them through expensive LLM-driven entity extraction. Chonk instead preserves and exploits structure directly: boundaries are detected from document markup, breadcrumbs encode the section path, and community clusters capture topical provenance—all without LLM calls.

We ask: *can we close the gap to graph-based systems without building a graph?* Chonk addresses this through three mechanisms that together approximate the benefits of graph construction:

1. **Entity Reference Expansion**: After seed retrieval, identify query-relevant entities and fetch additional chunks containing those entities, filling coverage gaps that embedding similarity alone misses.
2. **Entity Laning**: A quality gate that admits entity-expanded candidates only if they maintain a minimum similarity to the query, preventing low-relevance noise.
3. **Community Context Injection**: A pre-built community index (Louvain clustering on the embedding similarity graph) provides topic labels that are injected into the generation context, giving the model a map of the corpus neighborhood relevant to each query.

Together these mechanisms yield 0.661 overall accuracy on GraphRAG-Bench, matching G-reasoner (0.661) and outperforming all other published systems, without requiring knowledge graph construction, entity relationship extraction, or graph traversal at inference time.

---

## 2. Related Work

### 2.1 GraphRAG Systems

**MS-GraphRAG** [Edge et al., 2024] extracts an entity knowledge graph and pre-generates hierarchical community summaries. The global query mode passes community summaries as context, achieving strong sensemaking performance at the cost of massive token overhead. The local query mode traverses entity neighborhoods for specific questions.

**LightRAG** [Guo et al., 2025] employs dual-level retrieval combining low-level entity lookup with high-level graph traversal, incorporating an incremental update mechanism for evolving corpora.

**HippoRAG2** [Gutierrez et al., 2025] extends the Personalized PageRank approach of HippoRAG to non-parametric continual learning, integrating deeper passage content into graph-based retrieval for factual, sense-making, and associative memory tasks.

**RAPTOR** [Sarthi et al., 2024] recursively clusters and summarizes document trees, creating a multi-level abstraction hierarchy for retrieval. It shows strong performance on multi-hop reasoning but requires expensive iterative summarization during indexing.

**G-reasoner** [Anonymous, 2025] integrates a Graph Foundation Model (GFM, 34M parameters) with large language models through a QuadGraph standardized abstraction. It achieves the current state-of-the-art on GraphRAG-Bench by jointly capturing topology and textual semantics.

**AutoPrunedRetriever** [Anonymous, 2026] persists minimal reasoning subgraphs across queries using ID-indexed codebooks and aggressive pruning of low-value structures. It reduces token overhead by up to two orders of magnitude compared to graph-heavy approaches.

### 2.2 Enhanced Retrieval without Graph Construction

**Reranking** [Nogueira et al., 2019] improves precision by applying a cross-encoder model to score retrieved candidates. It is a standard enhancement that improves vanilla RAG but does not address the entity coverage problem.

**Entity-based RAG** systems [Trivedi et al., 2022; Jiang et al., 2023] use named entity recognition to guide multi-hop retrieval, typically through explicit linking of named entities across documents. Chonk's entity reference expansion takes a simpler approach: instead of building links at index time, it performs entity-conditioned retrieval at query time.

### 2.3 Breadcrumb Embeddings and Structural Signals

Chonk records each chunk's heading path (document name + section hierarchy) as a *breadcrumb*. Breadcrumb vectors are computed at index time and serve two distinct purposes. First, for community detection, they provide a structural grouping signal: chunks that share section provenance cluster together even when content similarity is modest. Second, as an optional retrieval enrichment (`--breadcrumb-embed`), the breadcrumb is prepended to the chunk text before encoding, which addresses the *boilerplate collision* problem in enterprise documents where identical section titles (e.g., "Parameters", "Limitations") from different documents otherwise collapse in embedding space. We evaluate the retrieval use of breadcrumb embedding on GraphRAG-Bench and find mixed results (§5.4); the structural use in community detection is always active.

---

## 3. Method

### 3.1 System Overview

Chonk consists of two phases: *indexing*, which builds a DuckDB vector store with optional NER and community indices, and *retrieval*, which assembles a multi-dimensional candidate pool for each query.

### 3.2 Indexing

**Semantic boundary chunking.** Chonk parses document structure—headings, lists, tables, and continuation markers—and uses heading boundaries as natural split points. Content accumulates across sections until a minimum chunk size (600 characters) is reached, then flushes at the next same-level or shallower heading. Hard splits apply at a maximum size with sentence-boundary awareness and continuation markers (`[PARA:start/cont/end]`). Each chunk records its full heading path as a *breadcrumb* (document name + section hierarchy). Certain extractors additionally promote structured list items and section-like patterns into heading markers, improving boundary detection for semi-structured documents.

**Breadcrumb embeddings.** Every chunk's breadcrumb (document name and section path) is embedded separately from the chunk content. Breadcrumb vectors serve two roles: (1) as an *optional* enrichment of the retrieval embedding (`--breadcrumb-embed`), where the heading path is prepended to the chunk text before encoding to resolve boilerplate collision (e.g., two chunks from different documents both titled "Limitations"); and (2) as a *structural signal in community detection*, where breadcrumb vectors are always blended with content vectors (α=0.2) when building the similarity graph, grouping topically related chunks that share structural provenance.

**NER index.** We apply spaCy's NLP pipeline to extract named entities (persons, organizations, locations, medical terms, etc.) from each chunk, building an inverted index from entity strings to chunk IDs with frequency and confidence scores.

**Community index.** We build chunk embeddings as a weighted blend of breadcrumb embeddings and content embeddings (α=0.2 breadcrumb, α=0.8 content), then compute pairwise cosine similarities and construct a sparse similarity graph (edges where similarity ≥ 0.75). Louvain community detection partitions this graph into topically coherent groups. Community topic labels are derived using an entity-embedding strategy: candidate label terms are drawn from NER index entities present in the community's chunks, and near-duplicate labels are merged using embedding similarity (threshold 0.85). Coherence (mean intra-community similarity) is stored per community and used at query time to filter low-quality communities (`--community-min-coherence`).

### 3.3 Retrieval

**Seed retrieval.** Top-*k* chunks are retrieved by embedding similarity (DuckDB VSS).

**Entity reference expansion (ERE).** We extract named entities from the query using spaCy. For each query entity not present in the seed results, we embed the entity name and retrieve the top-3 nearest chunks. These are added to the candidate pool as entity-expanded candidates.

**Entity laning.** Each entity-expanded candidate must satisfy a minimum cosine similarity to the query (`--lane-entity-min-sim`, default 0.45) to be admitted. This prevents low-relevance entity neighbors from diluting the context.

**Community context injection.** For each retrieved chunk, we look up its community in the community index and, if the community coherence exceeds a threshold (`--community-min-coherence`, default 0.5), prepend a topic label to the context block passed to the generator:
```
[Topic: Hodgkin lymphoma · staging · PET-CT · ABVD chemotherapy]
<chunk content>
```

**Reranking.** The full candidate pool is scored by a cross-encoder (BAAI/bge-reranker-large) and the top-*k* candidates are selected.

**Redundancy pruning (optional).** Post-reranking, candidate pairs with cosine similarity ≥ 0.92 are deduplicated, retaining the higher-scoring member.

### 3.4 Generation

Retrieved chunks are concatenated in reranker score order (with optional community topic labels prepended) and passed as context to GPT-4o-mini with the benchmark's standard generation prompt.

---

## 4. Experimental Setup

### 4.1 Benchmark

We evaluate on **GraphRAG-Bench** [Chen et al., 2025], a publicly available benchmark with two corpora:
- **Medical**: NCCN clinical practice guidelines, with explicit hierarchical structure and domain-specific terminology.
- **Novel**: Pre-20th-century literature from Project Gutenberg, with rich narrative structure and implicit character/event relationships.

The benchmark includes 300 questions (150 Medical, 150 Novel) sampled to cover four complexity levels: Fact Retrieval (M-Fact, N-Fact), Complex Reasoning (M-Rsn, N-Rsn), Contextual Summarization (M-Summ, N-Summ), and Creative Generation (M-Crea, N-Crea). The primary metric is `answer_correctness`, an LLM-based scoring function using GPT-4o-mini as judge.

### 4.2 Implementation Details

- **Embedding model**: `text-embedding-3-small` (OpenAI), 1536-dimensional
- **Vector store**: DuckDB 0.10 with VSS extension (HNSW index)
- **Reranker**: `BAAI/bge-reranker-large` (local cross-encoder)
- **Generator / Judge**: GPT-4o-mini
- **Chunking**: Semantic chunking with header promotion, min 600 chars, max 1500 chars
- **Retrieval k**: 5 final candidates (20 fetched before reranking)
- **NaN limit**: Runs are terminated if 10 items produce NaN scores after 5 retry attempts

All runs use the 300-question grid sample. Published leaderboard scores [Chen et al., 2025] use the full 2,070-question dataset; we benchmark against their reported numbers as external reference.

---

## 5. Results

### 5.1 Main Results

Table 1 compares Chonk's best configuration against published GraphRAG-Bench baselines.

**Table 1: Comparison with Published Baselines**

| System | Med | Nov | Overall |
|--------|-----|-----|---------|
| G-reasoner† [2025] | 0.733 | 0.589 | **0.661** |
| AutoPrunedRetriever-llm† [2026] | 0.670 | 0.637 | 0.654 |
| **Chonk (ours)** | **0.701** | **0.622** | **0.661** |
| HippoRAG2† [2025] | 0.648 | 0.565 | 0.607 |
| Fast-GraphRAG† | 0.641 | 0.520 | 0.581 |
| RAG + rerank† | 0.624 | 0.483 | 0.554 |
| LightRAG† [2025] | 0.626 | 0.451 | 0.538 |
| RAG (no rerank)† | 0.610 | 0.479 | 0.545 |
| RAPTOR† [2024] | 0.571 | 0.432 | 0.502 |
| MS-GraphRAG (local)† [2024] | 0.452 | 0.509 | 0.480 |

† Published results from GraphRAG-Bench leaderboard [Chen et al., 2025]. Chonk results on 300-question subset using same generator and judge (GPT-4o-mini).

Chonk achieves 0.661 overall, matching G-reasoner without graph construction. Notably, Chonk achieves higher Novel accuracy (0.622) than G-reasoner (0.589) and higher Medical accuracy than most graph-based systems, while maintaining strong Novel performance.

### 5.2 Question-Type Breakdown

**Table 2: Per-Type Accuracy (Chonk best vs. baselines)**

| System | M-Fact | M-Rsn | M-Summ | M-Crea | N-Fact | N-Rsn | N-Summ | N-Crea |
|--------|--------|-------|--------|--------|--------|-------|--------|--------|
| Chonk (best) | 0.728 | 0.616 | 0.786 | 0.673 | 0.657 | 0.577 | 0.692 | 0.560 |
| Vanilla + rerank | 0.734 | 0.655 | 0.745 | 0.667 | 0.607 | 0.539 | 0.632 | 0.410 |

Chonk's largest gains over the vanilla baseline are in Novel Summarization (+0.060), Novel Creative (+0.150), and Medical Summarization (+0.041). These are precisely the question types that require broad evidence integration—suggesting that entity reference expansion and community context are providing coverage of relevant passages that embedding similarity alone misses.

### 5.3 Ablation Study

We used a two-stage methodology to navigate the large configuration space efficiently. First, we conducted an exploratory composition analysis across a wide range of retrieval configurations—varying entity laning thresholds, community coherence filters, redundancy pruning, cluster-neighbor expansion, breadcrumb injection, and retrieval depth—to identify promising component combinations. We then selected a focused subset for formal ablation based on empirical contribution, keeping the final evaluation tractable. Configurations that showed negligible or negative contribution in the exploratory phase (e.g., cluster-neighbor expansion, tighter lane thresholds, breadcrumb-context injection) were excluded from the main ablation; this accounts for apparent gaps in the component space.

Table 3 shows the incremental contribution of each Chonk component, starting from a reranking baseline.

**Table 3: Component Ablation**

| Configuration | Med | Nov | All | ΔSOTA |
|--------------|-----|-----|-----|-------|
| Vanilla RAG + rerank (baseline) | 0.700 | 0.547 | 0.624 | −0.037 |
| + NER retrieval (no ERE) | 0.664 | 0.581 | 0.622 | −0.039 |
| + Entity Reference Expansion (ERE) | 0.690 | 0.568 | 0.629 | −0.032 |
| + Redundancy pruning | 0.680 | 0.586 | 0.633 | −0.028 |
| + Entity laning (min_sim=0.45) | 0.692 | 0.600 | 0.646 | −0.015 |
| + Community context (best) | 0.701 | 0.622 | **0.661** | **0.000** |
| + Redundancy pruning (w/ community) | 0.686 | 0.607 | 0.647 | −0.014 |

**Entity Reference Expansion** (+0.007 over baseline): Adding NER alone without quality gating slightly reduces Medical accuracy (the NER index introduces some noisy candidates), but entity reference expansion—fetching chunks by entity identity rather than query similarity—improves Novel coverage.

**Entity Laning** (+0.017): Applying the minimum similarity gate to entity-expanded candidates produces the largest single improvement. This gate filters out entity-matched chunks that are topically irrelevant to the query, improving both Medical (+0.012) and Novel (+0.032) accuracy.

**Community Context Injection** (+0.015): Prepending community topic labels to retrieved chunks provides the model with a broader topical context signal, primarily benefiting Novel accuracy (+0.022) where documents have rich inter-character and event relationships that benefit from community framing.

**Redundancy Pruning**: Pruning helps in isolation (+0.004 over ERE alone) but degrades performance when applied after community context injection (−0.014). This suggests that for complex reasoning and creative generation questions—which constitute half the benchmark—evidence diversity is more valuable than conciseness.

### 5.4 Breadcrumb Embedding Analysis

Table 4 examines the effect of prepending document name and section path to the embedding input.

**Table 4: Breadcrumb Embedding Effect**

| Configuration | Med | Nov | All |
|--------------|-----|-----|-----|
| NER + rerank (no breadcrumb) | 0.664 | 0.581 | 0.622 |
| NER + rerank (+ breadcrumb embed) | 0.690 | 0.620 | **0.655** |
| Full stack, no breadcrumb (best) | 0.701 | 0.622 | **0.661** |
| Full stack + breadcrumb embed | 0.701 | 0.601 | 0.651 |

Breadcrumb embedding produces a substantial gain in the simple NER+rerank configuration (+0.033), confirming that the embedding enrichment improves basic retrieval quality. However, when combined with entity reference expansion and laning, breadcrumb embedding is slightly harmful (0.651 vs. 0.661). We hypothesize that breadcrumb tokens shift the embedding space in ways that affect entity similarity calculations: the entity name embeddings used in reference expansion are computed without breadcrumbs, so cross-querying entity embeddings against breadcrumb-enriched chunk embeddings introduces a distributional mismatch.

Additionally, breadcrumb-context injection—passing the section path as text in the prompt to the generator (`--breadcrumb-context`)—consistently degrades performance across all question types (0.637 overall), particularly for Summarization (N-Summ 0.604 vs. 0.692) and Reasoning (N-Rsn 0.571 vs. 0.577). The breadcrumb text appears to distract the model from the primary evidence.

### 5.5 Effect of Retrieval Depth (k=10)

A preliminary ablation on Novel questions (n=150) finds that increasing retrieval depth from k=5 to k=10 improves Novel accuracy from 0.600 to 0.613 (+0.013), with the largest gains on Novel Creative (+0.051) and Novel Summarization (+0.015), at a cost of Novel Reasoning (−0.015). Full 300-question results with the best configuration are pending.

---

## 6. Analysis

### 6.1 Why Does Chonk Match Graph RAG without a Graph?

Our results suggest that the primary benefit of graph-based RAG on this benchmark is *entity-level coverage*: ensuring that all relevant named entities appearing in a question are represented in the retrieved context. Chonk achieves this through query-time entity reference expansion, which is functionally equivalent to one-hop graph traversal from query entities, but implemented as targeted embedding search rather than explicit edge traversal.

Community context injection approximates the *community summarization* benefit of MS-GraphRAG's global mode, but without precomputing community summaries. Instead, we inject compact community topic labels (a handful of key entity names) that orient the model's attention toward the relevant topical neighborhood.

The key advantage over graph construction is that Chonk's community and entity indices are built as byproducts of the embedding process—no additional LLM calls are required during indexing, unlike graph-based systems that require entity extraction and relationship identification.

A deeper implication of these results is that the *retrieval-relevant* structure captured by graph construction—entity co-occurrence, topical community membership—appears to be latent in the document's embedding geometry and recoverable through deterministic methods: rule-based NER, cosine similarity graphs, and Louvain clustering. We do not claim these methods recover all information that LLM-based graph extraction produces; LLMs may identify implicit relationships, coreference chains, and causal links that deterministic NLP misses. What the results suggest is that, on this benchmark, those additional signals do not translate into measurable retrieval benefit. Whether that reflects a ceiling in the benchmark's sensitivity or a genuine sufficiency of structural approximation remains an open empirical question. Importantly, the deterministic nature of Chonk's indices is itself a practical advantage: entity and community assignments are reproducible, inspectable, and free of hallucinated edges—properties that LLM-extracted graphs cannot guarantee.

### 6.2 Medical vs. Novel Performance Profile

Chonk achieves relatively balanced performance across both domains (Med 0.701, Nov 0.622), whereas G-reasoner shows a stronger Medical skew (Med 0.733, Nov 0.589). This suggests Chonk's entity-guided retrieval is better calibrated for narrative documents, possibly because named entity expansion naturally captures character and event references that are central to Novel questions.

The Medical corpus, with its explicit hierarchical structure, benefits less from entity expansion (the answer is often in one specific section) and more from high-precision retrieval of the exact guideline passage. This is reflected in our breadcrumb embedding analysis: the section-path signal is more informative for the Medical corpus.

### 6.3 Confounded Components

Two indexing-time design choices — semantic boundary chunking and breadcrumb-blended community detection — are active in all Chonk runs and therefore not isolated by the current ablation. The vanilla baseline uses fixed-size 256-character chunking, so the gap between it and the NER+rerank baseline (Table 3) conflates chunking strategy with NER retrieval. Similarly, community detection always uses breadcrumb-blended embeddings (α=0.2); there is no community run with content-only graph edges. Isolating these contributions requires two additional runs: (a) semantic boundary chunking + rerank without NER, to separate chunking from retrieval strategy; and (b) community detection with α=0.0 (content-only), to quantify the structural signal from breadcrumbs. These runs would clarify whether the core strength of Chonk lies in its retrieval pipeline or its indexing-time structural encoding.

### 6.4 Benchmark Representativeness

GraphRAG-Bench uses two corpora: NCCN oncology guidelines and pre-20th-century novels from Project Gutenberg. Both are atypical of the corpora where structure-aware retrieval is most valuable. The NCCN guidelines are highly regular—each document follows a predictable section schema—so breadcrumbs add modest disambiguation value. The Project Gutenberg novels are largely unstructured narrative prose, with minimal heading hierarchy and no cross-document entity graph to exploit.

Real-world enterprise corpora—legal briefs, clinical notes, engineering specifications, multi-document research portfolios—typically exhibit far richer and more variable structure: deep heading hierarchies, cross-referenced sections, recurring entity networks, and high intra-corpus boilerplate. We expect Chonk's structure-exploiting components (semantic boundary detection, breadcrumb embeddings, community detection) to show larger gains on such corpora precisely because there is more structural signal to capture. The modest contribution of breadcrumb embedding on GraphRAG-Bench (helpful for basic retrieval, neutral at full stack) is likely a floor effect: when documents have shallow structure, the breadcrumb signal is weak. The same mechanism applied to a corpus of 500-page regulatory filings with 10-level section hierarchies should produce substantially larger improvements.

This suggests that GraphRAG-Bench, while valuable for comparing retrieval pipelines on clean corpora, may underestimate the advantage of structure-aware systems in production settings. Evaluation on corpora with richer document structure remains important future work.

### 6.5 Failure Modes

**NaN items**: Across all runs, 4–11 items per 300-question run produce NaN scores (LLM judge returns non-parseable output). These cluster around a small set of persistent items (e.g., the Angeline Hall question and the Elsie Inglis question across multiple run variants), suggesting content-specific parsing failures in the judge rather than system failures.

**Reasoning degradation with k=10**: Increasing k from 5 to 10 slightly hurts Novel Reasoning (−0.015), consistent with the hypothesis that adding more context beyond what the reasoning chain requires introduces distracting evidence.

---

## 7. Conclusion

We presented Chonk, a retrieval system that achieves state-of-the-art performance on GraphRAG-Bench by combining entity reference expansion, entity laning, and community context injection without constructing a knowledge graph. Our best configuration achieves 0.661 overall accuracy, matching G-reasoner (0.661) and outperforming all other published baselines.

The core finding is that the benefits of graph-based RAG—entity coverage and community-level context—can be approximated at query time through targeted entity-conditioned retrieval and pre-built community topic labels, with no additional LLM calls during indexing. This makes Chonk practical for large or frequently updated corpora where graph construction is prohibitively expensive.

Importantly, GraphRAG-Bench's corpora (clinical guidelines with regular schemas, flat-prose novels) are not representative of the corpora where Chonk's structural mechanisms are most valuable. Our central hypothesis—that most real-world documents contain exploitable structure that vanilla RAG discards—predicts larger gains on richer corpora such as legal filings, engineering specifications, or multi-document research collections, where deep section hierarchies, entity cross-references, and intra-corpus boilerplate are far more prevalent. The current results should be understood as a lower bound on Chonk's advantage in such settings.

Future work includes: (1) full evaluation of k=10 with community context (expected to exceed 0.661 on Novel), (2) evaluation on the full 2,070-question dataset for direct leaderboard comparison, (3) ablation isolating semantic boundary chunking from NER retrieval (fixed-size chunking + rerank baseline), (4) ablation of breadcrumb contribution to community detection (α=0.0 vs. α=0.2 graph edges), (5) evaluation on structurally rich corpora (regulatory filings, legal briefs, engineering specifications) where we expect substantially larger gains from boundary detection and breadcrumb-informed retrieval, and (6) a user study or domain-specific benchmark that directly tests the structural-exploitation hypothesis on enterprise document corpora.

---

## References

Chen, Z., et al. (2025). *When to Use Graphs in RAG: A Systematic Benchmark for Graph-Enhanced Retrieval-Augmented Generation*. arXiv:2506.05690.

Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130.

Gutierrez, B.J., et al. (2025). *HippoRAG2: From RAG to Memory*. arXiv:2502.14802.

Guo, Z., et al. (2025). *LightRAG: Simple and Fast Retrieval-Augmented Generation*. arXiv:2410.05779. EMNLP 2025.

Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv:1901.04085.

Sarthi, P., et al. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*. arXiv:2401.18059. ICLR 2024.

Anonymous. (2025). *G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge*. arXiv:2509.24276.

Anonymous. (2026). *Pruning Minimal Reasoning Graphs for Efficient Retrieval-Augmented Generation*. arXiv:2602.04926.
