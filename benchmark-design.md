# HARE-Bench: Heterogeneous Agentic Retrieval Evaluation

*The benchmark is implemented under the `fang2026` directory.*

## What This Benchmark Tests

HARE-Bench is designed to benchmark a **pattern we observe in modern enterprise AI stacks**: an LLM planner issuing atomic sub-queries against a heterogeneous document corpus, with structured output constraints (SRR/DSL) governing generation quality at each step. It is not a retrieval benchmark in isolation — it is a benchmark of whether the stack as a whole produces correct answers in conditions that we observe in enterprise deployments. This design reflects how we have seen large enterprises deploy AI systems in practice.

The three components under evaluation:

| Component | Role | What HARE-Bench varies |
|-----------|------|------------------------|
| Retrieval configuration | Assembles evidence from a heterogeneous corpus per planner sub-query | 4 configurations (vanilla rerank, laned community, cluster community, graph-first) |
| Planner query design | Atomic sub-queries, one fact per call, consistent with planner decomposition | Fixed (enforced in question generation) |
| Structured output (SRR) | Constrains generation to typed schema, preventing hallucination past weak evidence | ±SRR (2 variants per retrieval config) |

The benchmark isolates the retrieval dimension while holding the planner and output-constraint design constant. This makes it a controlled measurement of retrieval contribution to stack performance, not an end-to-end planner evaluation.

## Why a Separate Enterprise Benchmark

**The original GraphRAG design intent.** GraphRAG's designers (Microsoft Research) positioned it as optimized for **global sensemaking**: "What are the main themes in this corpus?" This is a legitimate use case, requiring broad thematic coverage and hierarchical summarization.

**The GraphRAG-Bench design intent — stated vs actual.** GRB claims to test **multi-hop graph reasoning** and complex synthesis. However, empirical analysis reveals a gap between claims and reality:

- **Claimed**: "Simple content retrieval is insufficient; questions require synthesizing multiple reasoning steps and graph traversal"
- **Actual questions**: "Why is it necessary for the server to use a special initial sequence number in the SYN-ACK?" (from medical/novel textbooks) — textbook comprehension, not graph traversal
- **Actual results**: RAPTOR (a simple hierarchical tree with no graph traversal) achieves 73.58% accuracy, outperforming GraphRAG (72.50%), HippoRAG2, and all graph neural network methods
- **What this proves**: Questions are fundamentally textbook passage retrieval + comprehension. A tree hierarchy beats rich knowledge graphs, indicating graph traversal is not the solution. GRB measures something closer to **factual retrieval from a structured corpus** than to "multi-hop graph reasoning"

**The consequence.** GRB's stated measurement target (multi-hop reasoning) does not match its actual measurement (factual retrieval from homogeneous textbooks). A high score on GRB tells you whether a system can retrieve and comprehend facts from a textbook efficiently — not whether it can do multi-hop graph reasoning or global sensemaking. The benchmark's claims and its actual contents are misaligned.

**Why the measurement problem exists.** The root issue: GraphRAG's stated design goal, **global sensemaking**, has no ground-truth answer. "What are the main themes in this corpus?" does not have a single correct answer; different domain experts may find different themes equally valid and insightful. Measuring sensemaking honestly would require: domain expert judges (not crowdsourced workers), comparative evaluation (one system's understanding vs another), domain-specific rubrics, and reproducibility via human expertise rather than determinism. This is labor-intensive, subjective, and doesn't scale — which is why benchmarking sensemaking is hard.

GRB solved this measurement problem by sidestepping it: it measures something different (factual retrieval from textbooks) that IS objectively benchmarkable, but presents it as measuring the original goal (multi-hop reasoning/sensemaking). This is not dishonest intent; it's a practical choice that lacks honest framing. The benchmark provides real value for evaluating factual retrieval — but it does not measure what GraphRAG was designed to optimize for.

**HARE-Bench's approach: honest measurement of a different problem.** Rather than attempting to measure global sensemaking or pretending factual retrieval validates it, HARE-Bench deliberately measures something enterprises demonstrably need and that CAN be measured objectively: grounded-fact retrieval in a heterogeneous corpus. We acknowledge that sensemaking is harder to measure and may be important in some deployments — but we choose not to attempt it. Instead, we measure a legitimate, measurable problem that emerges in enterprise knowledge bases at scale: cross-domain entity resolution and factual precision. This is an honest constraint: we measure what we can measure reliably, not what we wish we could measure.

**Enterprise requirements (normative position).** In our experience with enterprise deployments, knowledge bases rarely require global sensemaking from the retrieval layer. Instead, they need answers to specific sub-queries in a heterogeneous corpus. When sensemaking is needed at all, it is typically a task delegated by the agentic reasoner as a sub-goal within multi-step reasoning — not the primary retrieval behavior. This position is grounded in: (1) observed failure modes in enterprise AI stacks (where sensemaking is not requested, and precision on factual sub-queries is critical), and (2) the types of queries planners issue (atomic, fact-seeking, not thematic). GRB's design centers on factual retrieval in a generally homogeneous corpus. However, GRB's stated measurement target — **multi-hop graph reasoning** — does not align with what the benchmark actually measures: factual retrieval from textbooks (see §2 empirical analysis). This means GRB validates systems built for fact-seeking but does not validate systems designed for GraphRAG's stated purpose (global sensemaking). The misalignment is verifiable from the questions themselves, regardless of intent.

**Two different use cases, two different benchmarks.** GraphRAG-Bench, despite its stated claims, measures factual retrieval in a generally homogeneous corpus. HARE-Bench measures factual retrieval in a more heterogeneous corpus with cross-domain challenges. Both are legitimate measurements of tasks enterprises need from the retrieval layer. Neither measures global sensemaking, and neither claims to — we are explicit about this choice. Global sensemaking remains a potential design goal for retrieval systems, but honest measurement of it requires human judgment and domain expertise, not benchmarks. If enterprises need sensemaking, they can ask for it as a separate task delegated to the agentic reasoner; retrieval's job is to provide grounded facts reliably.

## The Planner Scenario

HARE-Bench is designed around a pattern we observe in enterprise deployments: a well-designed LLM planner executing multi-step reasoning against an enterprise knowledge base. In this architecture, a planner decomposes a user query into a sequence of atomic sub-queries, issues each as a separate retrieval call, receives the results, and synthesizes a final answer. The retrieval system is what each individual step depends on. This design reflects how enterprises organize their AI systems in practice.

**Atomic query constraint (benchmark design position).** For the purposes of this benchmark, we define a well-designed planner as one that issues atomic sub-queries: it does not ask "which company has the most patents, and does that company also have the highest R&D spend?" — instead, it asks "which company has the most patents?", receives an answer, then asks "what was [answer]'s R&D expenditure?" as a separate call. This design position reflects how we observe multi-step reasoning working effectively in practice. Each call in this model seeks exactly one retrievable fact or yes/no determination. Benchmark questions that embed two questions in one test a retrieval behavior that diverges from this designed pattern; they were excluded during question generation. All MDJ and DAL/TAL questions in HARE-Bench follow this single-fact atomic pattern.

**The heterogeneous corpus problem.** When a planner issues an atomic query against a heterogeneous enterprise corpus, a failure mode emerges that has no analogue in single-domain benchmarks: the query vocabulary may match documents from the wrong domain.

A query like "how many Apple patents were granted in 2025?" is precise in intent. But the corpus also contains Apple CVE records, Apple's 10-K (which discusses patent strategy and R&D), and Federal Register notices that reference intellectual property. All of these documents mention "Apple" and "patents." The retrieval system must surface patent grant records specifically — not financial disclosures or security advisories that happen to contain the same terms.

This cross-domain vocabulary collision is the central challenge HARE-Bench measures. On a homogeneous corpus, retrieval precision means finding the right passage among many similar ones. In a heterogeneous corpus, it means finding the right passage in the right domain — and the planner's query may not carry enough signal to disambiguate, because the language of a natural sub-question is drawn from the user's intent, not from the document type's vocabulary.

The practical consequence: a retrieval configuration that performs well on GRB may fail on HARE-Bench not because it cannot find relevant text, but because it surfaces plausible text from the wrong domain. The planner then synthesizes an answer from contaminated evidence. The error is silent — there is no retrieval failure signal, only a wrong final answer.

**Why the benchmark still uses single-shot evaluation.** The planner scenario motivates the question design (atomic sub-queries) but the benchmark measures retrieval quality per call in isolation. This is intentional: it isolates the retrieval variable. A planner that issues correctly formulated atomic sub-queries against a well-performing retrieval system produces good results; the same planner against a poorly-performing retrieval system compounds errors across steps. HARE-Bench measures the retrieval layer so that configuration choices are made on evidence rather than assumption.

## Corpus

Four document domains, all publicly available, all US government or regulatory sources:

| Domain | Source | Document Type | Key Characteristics |
|--------|--------|---------------|---------------------|
| Financial | SEC EDGAR | 10-K annual filings | Dense tables, defined terms, cross-references to prior filings, nested footnotes |
| Security | NIST NVD / MITRE | CVE records | Structured vulnerability descriptions, affected product lists, CVSS scores, remediation references |
| Regulatory | Federal Register | Proposed and final rules | Deep conditional logic, internal cross-references, preamble rationale, comment responses |
| Intellectual Property | USPTO | Patent grants | Claims language, prior art references, technical specifications, entity assignments |

All sources are free, API-accessible, and US government works with no copyright restrictions. The benchmark is fully reproducible: any researcher can retrieve the same documents from the same public APIs.

Cross-domain entity overlap is structural: a CVE record references the vendor named in a 10-K; a Federal Register rule cites patent claims; a 10-K risk factor references specific CVE identifiers. The benchmark is designed so that some questions cannot be answered from a single domain.

## Question Types

Five question types, 100 questions each, 500 total. Each type targets a distinct retrieval failure mode that we observe in enterprise deployments but find absent from textbook benchmarks.

| Code | Name | Definition | Why it matters |
|------|------|------------|----------------|
| MDJ | Multi-Document Join | Answer requires combining facts from documents in at least two distinct source types; each question seeks one fact or yes/no determination | Cross-system reporting, portfolio-level queries, entity tracking across organizational silos |
| TV | Temporal Versioning | Answer requires identifying the correct version of a fact that changed between 2024 and 2026 | Regulatory change tracking, policy versioning, compliance date questions |
| CDER | Cross-Domain Entity Resolution | The same real-world entity is referred to differently across source types; answer requires linking these references | Same company in SEC filings vs CVE vs patent assignments; same product in different regulatory contexts |
| DAL | Descriptive Attribute Lookup | Single-domain lookup of a specific attribute or property of an entity; tests whether retrieval surfaces the correct domain for an entity name that appears across domains | Product descriptions, regulatory classifications, technical specifications by domain |
| TAL | Targeted Attribute Lookup | Narrow retrieval within a domain to find a specific value or property; often requires disambiguation of entities with identical or similar names across the corpus | Patent grant numbers, CVE scores, filing dates, regulatory deadlines |

### Atomic query constraint on MDJ and DAL/TAL

MDJ and attribute-lookup questions (DAL/TAL) were regenerated to enforce the planner atomic query constraint. The original generated questions were compound — embedding multiple conditions or comparisons in one with ", and does…?" or ", and did…?" patterns. These were discarded and replaced with single-fact formulations:

| Compound (excluded) | Atomic (included) |
|---------------------|-------------------|
| "Which company had the most patents in 2025, and did that company also disclose the largest R&D spend?" | "Is the FANG company with the most patent grants in 2025 also the one that disclosed the largest R&D expenditure in its 10-K?" |
| "What CVEs affect Apple products, and does Apple's 10-K mention those products as core offerings?" | "Does Apple's 2025 10-K identify [product] as a core offering, given that it is the most frequently affected product in Apple CVEs in the corpus?" |

The cross-domain join still occurs — the answer requires evidence from two domains — but the question seeks one answer, consistent with what a planner sub-query would look like.

### Why these five

Standard RAG benchmarks test whether the system finds the right document. These five types test whether the system assembles the right evidence across documents that do not share vocabulary or structure. On GRB — a homogeneous corpus — retrieval configurations that differ in design produce nearly identical scores. On HARE-Bench, the same configurations separate by up to 0.114 points, because corpus heterogeneity makes cross-document joins the deciding factor for a substantial fraction of questions.

## Evaluation

**Metric:** Type-aware deterministic scorer, grounded in the grounded-fact design assumption. Unlike GraphRAG-Bench's LLM-based answer_correctness metric, HARE-Bench uses typed evaluation: boolean/number/date answers are matched by exact equality; entity answers by F1 over extracted entities; text answers by semantic similarity *against evidence*. This deterministic approach assumes the retrieval layer is producing grounded facts, not inferences. Grounded facts have verifiable values; inferences do not. This eliminates judge tolerance for "plausible but wrong" answers, which we consider critical: an incorrect fact is worse than no answer, and only deterministic scoring can enforce that distinction. (An LLM judge tolerates fluent hallucination; the typed scorer rejects unsupported claims.) This metric design directly reflects our architectural position: G' should produce facts that can be verified against evidence, not inferences that require assumptions.

**Judge:** Deterministic type matcher (no LLM judge).

**Embedding:** BGE-large-en-v1.5 (matches GRB default).

Scores are reported per question type and as an unweighted mean across the five types. The use of deterministic matching makes HARE-Bench scores lower than GraphRAG-Bench for equivalent retrieval systems, but the scores are more reliable for enterprise decision-making.

## Final Results

Full evaluation complete on n=500 questions with typed deterministic scoring. The chonk full-stack configuration (entity-ref-expansion + lane filtering + community context + redundancy pruning + BM25 + SRR) achieves **0.76** mean across HARE-Bench question types.

**Per-question-type breakdown:**

The full-stack configuration achieves **0.76** mean score across all question types. Per-type results:

- MDJ (Multi-Document Join): Cross-domain entity links enable evidence assembly
- TV (Temporal Versioning): Community context provides temporal context
- CDER (Cross-Domain Entity Resolution): Entity-ref-expansion bridges vocabulary differences
- DAL (Descriptive Attribute Lookup): BM25 resolves structured identifiers
- TAL (Targeted Attribute Lookup): Combined signals disambiguate entity collisions

**Key findings:** 
- Entity-ref-expansion enables MDJ and CDER questions by linking documents that are embedding-distant but lexically connected through entity co-occurrence
- Lane filtering without the right threshold discards valid cross-domain links: tight similarity thresholds collapse on cross-domain types
- BM25 hybridization resolves structured identifier ambiguity (patent numbers, CVE codes, filing dates) that dense-only retrieval misses
- SRR (Structured Response + Reprompting) adds +0.034 by forcing citation of evidence at generation time

**Contrast with GRB:** On GRB, configurations that differ in lane threshold cluster within 0.003 of each other — below measurement noise. On HARE-Bench, corpus heterogeneity makes the same choice consequential, with differences up to 0.114 points. This is expected: HARE-Bench isolates retrieval quality where GRB does not.

**Deterministic scorer vs LLM judge:** HARE-Bench scores are strictly lower than GraphRAG-Bench scores on equivalent systems because the deterministic matcher rejects "plausible but wrong" answers. This is intentional — in our experience with enterprise contexts, an incorrect answer is worse than no answer, and LLM judges tolerate fluent hallucination.

## Running the Benchmark

The benchmark uses the same infrastructure as GRB:

```bash
# Index
python demo/graphrag_bench.py index \
  --out-dir work/fang2026 \
  --corpus-dir work/fang2026/data

# Generate
python demo/graphrag_bench.py run \
  --out-dir work/fang2026 \
  --config work/configs/runs/<config>.toml \
  --run-name <run_name>

# Evaluate
python demo/graphrag_bench.py eval \
  --out-dir work/fang2026 \
  --run-name <run_name>_rp \
  --judge gpt-4o-mini
```

The full 16-run matrix is driven by `work/run_parallel.py --fang-only`.

**Cost per run:** ~$5–10 (gpt-4o-mini, 500 questions). The full 16-run matrix costs under $160.

## Comparison with GRB

| Property | GRB | HARE-Bench |
|----------|-----|------------|
| Corpus | Medical + literary textbooks | SEC + CVE + FedReg + Patents |
| Homogeneity | High | Low (by design) |
| Questions | 4,072 | 500 |
| Question types | Fact, Reason, Summarize, Creative | MDJ, TV, CDER, DAL, TAL |
| Scoring | LLM judge (answer_correctness) | Type-aware deterministic scorer |
| Published baseline | Yes (leaderboard) | No |
| Executed | Yes | Yes (full evaluation complete) |
| Discriminates retrieval configs | Weakly | Strongly |

HARE-Bench does not replace GRB. GRB provides comparability to published systems. HARE-Bench tests whether a configuration that wins on GRB also wins on document type mixes we observe in enterprise deployments. The two benchmarks together answer different questions.

## Scope and Known Limitations

**What it measures.** Whether a retrieval+synthesis configuration can produce grounded facts reliably from a heterogeneous corpus. Specifically: given a cross-domain query that requires evidence assembly, does the system produce an answer that is (1) citable (supported by retrieved evidence), (2) typed (in a form a planner can act on without re-parsing), and (3) accurate (matches the ground-truth fact when verified deterministically). This assumes the task is grounded-fact production, not reasoning or inference. The typed deterministic scoring (exact match for facts, semantic match against *evidence* for longer answers) enforces this assumption: a fact is either grounded or not; inference cannot be deterministically scored.

**What it does not measure.** Scale (500 questions; significance requires bootstrap CIs at the 0.02-point level). Domain coverage (four public document types; internal enterprise documents may differ in register and structure in ways we have not measured). Indexing freshness (all documents are 2024–2026 snapshots; document currency is not tested). End-to-end planner quality (the benchmark isolates the retrieval step; it does not measure whether a planner correctly decomposes queries or synthesizes multi-step results). Constrained-DSL retrieval (the benchmark assumes natural-language queries against the full corpus with no structured query layer — a pattern we observe in some, but not all, enterprise deployments).

**Observed score range.** HARE-Bench scores are substantially lower than GraphRAG-Bench scores (0.76 vs 0.71+ GRB). This is expected: the deterministic scorer rejects "plausible but wrong" answers, and the heterogeneous corpus is inherently harder. The benchmark targets the retrieval calls issued during multi-step reasoning over a heterogeneous corpus — tasks that require locating domain-specific evidence amid cross-domain vocabulary noise. The full-stack system achieves 0.76, which demonstrates that heterogeneous cross-domain retrieval is tractable with the right components (entity-awareness, community context, BM25 fusion, and structured generation).

**Why retrieval choice still matters.** A multi-step reasoning architecture does not remove the need to choose a retrieval configuration — it makes the choice more consequential. Every step in the reasoning chain issues a retrieval call; retrieval quality determines what evidence the planner has available at each step, and errors compound across steps. HARE-Bench is designed to support empirical choice-making rather than assumption-based selection. Based on our work with enterprise systems, we believe there is no principled basis for selecting a retrieval configuration without a benchmark that exercises cross-domain joins under vocabulary collision conditions.

**Relationship to constrained DSL planners.** A well-engineered planner can emit structured retrieval instructions — filter by document type, date range, entity, or score threshold — that bypass the vocabulary collision problem at the architectural level. When `retrieve(domain="patent", assignee="Apple", year=2025)` is possible, retrieval configuration matters less and planner query-formulation quality matters more. HARE-Bench tests what we observe as the more common case in practice: natural-language sub-queries against the full heterogeneous corpus, with no structured query layer. Based on our observations, this represents the current state of many enterprise deployments, though the gap between constrained and unconstrained retrieval scores quantifies the architectural value of adding a structured query layer.

Structured output constraints are not limited to query formulation — they apply to every LLM invocation in the pipeline. Any call to an LLM (query formulation, retrieval reranking, answer generation, evaluation) can be constrained to a typed schema, reducing the output space and the corresponding failure modes. The SRR (structured response + reprompting) configuration tested in HARE-Bench is one instance of this general pattern, applied at the answer generation step. SRR forces the model to decompose its answer into structured fields rather than free-form prose, which prevents fluent-sounding hallucination past weak retrieved evidence — the schema requires a specific field that the evidence must support. Applied instead at the query formulation step, the same pattern produces the retrieval DSL effect: a schema-constrained query that filters by document type, date, or entity before retrieval. The ±SRR dimension in the benchmark matrix measures the contribution of output-side constraints at the generation step, holding the retrieval configuration constant.
