# Enterprise Cross-Domain Retrieval Benchmark (ECDR-Bench)

*Working title. The benchmark is implemented under the `fang2026` directory.*

## What This Benchmark Tests

ECDR-Bench benchmarks the **modern enterprise AI stack**: an LLM planner issuing atomic sub-queries against a heterogeneous document corpus, with structured output constraints (SRR/DSL) governing generation quality at each step. It is not a retrieval benchmark in isolation — it is a benchmark of whether the stack as a whole produces correct answers in conditions that match how large enterprises actually deploy AI systems.

The three components under evaluation:

| Component | Role | What ECDR-Bench varies |
|-----------|------|------------------------|
| Retrieval configuration | Assembles evidence from a heterogeneous corpus per planner sub-query | 4 configurations (vanilla rerank, laned community, cluster community, graph-first) |
| Planner query design | Atomic sub-queries, one fact per call, consistent with planner decomposition | Fixed (enforced in question generation) |
| Structured output (SRR) | Constrains generation to typed schema, preventing hallucination past weak evidence | ±SRR (2 variants per retrieval config) |

The benchmark isolates the retrieval dimension while holding the planner and output-constraint design constant. This makes it a controlled measurement of retrieval contribution to stack performance, not an end-to-end planner evaluation.

## Why a Separate Enterprise Benchmark

GraphRAG-Bench (GRB) uses medical and literary textbooks. Those corpora are homogeneous by design: each document comes from the same domain, uses consistent vocabulary, and covers the same subject matter. Retrieval on a homogeneous corpus rewards recall precision — finding the right passage among many similar ones. That is not the dominant failure mode in enterprise AI stacks.

Enterprise knowledge bases span organizational boundaries: SEC filings, regulatory guidance, security advisories, patent claims, clinical protocols, internal policy. The same entity — a company, a chemical compound, a software package — appears in multiple document types with different terminology, structure, and register. A planner issuing sub-queries against this corpus faces vocabulary collision at every step: the query terms that describe one domain also appear in adjacent domains for different reasons.

GRB measures whether a retrieval system finds the right passage. ECDR-Bench measures whether the retrieval component of an enterprise AI stack assembles the right evidence across document types that do not naturally point at each other — under the query conditions that a real planner would produce.

## The Planner Scenario

ECDR-Bench is designed around the retrieval calls issued by a well-designed LLM planner executing multi-step reasoning against an enterprise knowledge base. In this architecture, a planner decomposes a user query into a sequence of atomic sub-queries, issues each as a separate retrieval call, receives the results, and synthesizes a final answer. The retrieval system is what each individual step depends on.

**Atomic query constraint.** A well-designed planner never issues compound questions. It does not ask "which company has the most patents, and does that company also have the highest R&D spend?" — it asks "which company has the most patents?", receives an answer, then asks "what was [answer]'s R&D expenditure?" as a separate call. Each call seeks exactly one retrievable fact or yes/no determination. Benchmark questions that embed two questions in one test a retrieval behavior no planner would produce; they were excluded during question generation. All MDJ and QS questions in ECDR-Bench are single-fact atomic queries.

**The heterogeneous corpus problem.** When a planner issues an atomic query against a heterogeneous enterprise corpus, a failure mode emerges that has no analogue in single-domain benchmarks: the query vocabulary may match documents from the wrong domain.

A query like "how many Apple patents were granted in 2025?" is precise in intent. But the corpus also contains Apple CVE records, Apple's 10-K (which discusses patent strategy and R&D), and Federal Register notices that reference intellectual property. All of these documents mention "Apple" and "patents." The retrieval system must surface patent grant records specifically — not financial disclosures or security advisories that happen to contain the same terms.

This cross-domain vocabulary collision is the central challenge ECDR-Bench measures. On a homogeneous corpus, retrieval precision means finding the right passage among many similar ones. In a heterogeneous corpus, it means finding the right passage in the right domain — and the planner's query may not carry enough signal to disambiguate, because the language of a natural sub-question is drawn from the user's intent, not from the document type's vocabulary.

The practical consequence: a retrieval configuration that performs well on GRB may fail on ECDR-Bench not because it cannot find relevant text, but because it surfaces plausible text from the wrong domain. The planner then synthesizes an answer from contaminated evidence. The error is silent — there is no retrieval failure signal, only a wrong final answer.

**Why the benchmark still uses single-shot evaluation.** The planner scenario motivates the question design (atomic sub-queries) but the benchmark measures retrieval quality per call in isolation. This is intentional: it isolates the retrieval variable. A planner that issues correctly formulated atomic sub-queries against a well-performing retrieval system produces good results; the same planner against a poorly-performing retrieval system compounds errors across steps. ECDR-Bench measures the retrieval layer so that configuration choices are made on evidence rather than assumption.

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

Five question types, 100 questions each, 500 total. Each type targets a distinct retrieval failure mode that is common in enterprise deployments but absent from textbook benchmarks.

| Code | Name | Definition | Why it matters |
|------|------|------------|----------------|
| MDJ | Multi-Document Join | Answer requires combining facts from documents in at least two distinct source types; each question seeks one fact or yes/no determination | Cross-system reporting, portfolio-level queries, entity tracking across organizational silos |
| TV | Temporal Versioning | Answer requires identifying the correct version of a fact that changed between 2024 and 2026 | Regulatory change tracking, policy versioning, compliance date questions |
| CDER | Cross-Domain Entity Resolution | The same real-world entity is referred to differently across source types; answer requires linking these references | Same company in SEC filings vs CVE vs patent assignments; same compound in FDA label vs clinical trial |
| QS | Quantitative Synthesis | Answer requires aggregating or comparing numerical values drawn from one or more sources; each question seeks one number, count, ratio, or comparison result | Capital requirement calculations, exposure aggregation, patent count by assignee |
| A/N | Absence/Negation | Correct answer is the absence of a fact, or that a stated claim is not supported by any source | Compliance scope exclusions, products not subject to a rule, claims not present in any filing |

### Atomic query constraint on MDJ and QS

MDJ and QS questions were regenerated to enforce the planner atomic query constraint. The original generated questions were 75% and 88% compound respectively — embedding two questions in one with ", and does…?" or ", and did…?" patterns. These were discarded and replaced with single-fact formulations:

| Compound (excluded) | Atomic (included) |
|---------------------|-------------------|
| "Which company had the most patents in 2025, and did that company also disclose the largest R&D spend?" | "Is the FANG company with the most patent grants in 2025 also the one that disclosed the largest R&D expenditure in its 10-K?" |
| "What CVEs affect Apple products, and does Apple's 10-K mention those products as core offerings?" | "Does Apple's 2025 10-K identify [product] as a core offering, given that it is the most frequently affected product in Apple CVEs in the corpus?" |

The cross-domain join still occurs — the answer requires evidence from two domains — but the question seeks one answer, consistent with what a planner sub-query would look like.

### Why these five

Standard RAG benchmarks test whether the system finds the right document. These five types test whether the system assembles the right evidence across documents that do not share vocabulary or structure. On GRB — a homogeneous corpus — retrieval configurations that differ in design produce nearly identical scores. On ECDR-Bench, the same configurations separate by up to 0.114 points, because corpus heterogeneity makes cross-document joins the deciding factor for a substantial fraction of questions.

## Evaluation

**Metric:** `answer_correctness` — the GraphRAG-Bench metric, which decomposes both the generated answer and the ground truth into statements and computes F1 over TP/FP/FN classifications. Scores are on a 0–1 scale.

**Judge:** gpt-4o-mini (matches the GRB evaluation protocol; directly comparable).

**Embedding:** BGE-large-en-v1.5 (matches GRB default).

Scores are reported per question type and as an unweighted mean across the five types.

## Preliminary Results

*16-run matrix (±SRR × gpt-4o-mini/Haiku × 4 retrieval configs) pending. Grid-sweep results below are from an earlier pass and should be treated as directional.*

| Configuration | MDJ | TV | CDER | QS | A/N | Mean |
|---------------|----:|---:|-----:|---:|----:|-----:|
| Rerank + cluster community k=10 | 0.327 | 0.497 | 0.402 | 0.525 | 0.637 | **0.478** |
| Graph-first k=10 | 0.394 | 0.579 | 0.431 | 0.406 | 0.556 | **0.473** |
| Laned60 + community k=10 | 0.156 | 0.563 | 0.345 | 0.262 | 0.495 | **0.364** |
| Global search k=10 | 0.358 | 0.463 | 0.267 | 0.316 | 0.287 | **0.338** |

**Key finding:** The laned60 configuration — highest-scoring on GRB — collapses on MDJ (0.156). The 0.60 entity similarity threshold discards cross-domain links: a CVE record and a 10-K filing may reference the same vendor without being embedding-similar, because they use different vocabulary and structure. Lane filtering treats that dissimilarity as noise and discards it. MDJ questions require exactly those discarded chunks.

**Contrast with GRB:** On GRB, configurations that differ in lane threshold cluster within 0.003 of each other — below measurement noise. On ECDR-Bench, the same choice produces a 0.114-point gap. Corpus heterogeneity is the discriminating condition.

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

| Property | GRB | ECDR-Bench |
|----------|-----|------------|
| Corpus | Medical + literary textbooks | SEC + CVE + FedReg + Patents |
| Homogeneity | High | Low (by design) |
| Questions | 4,072 | 500 |
| Question types | Fact, Reason, Summarize, Creative | MDJ, TV, CDER, QS, A/N |
| Published baseline | Yes (leaderboard) | No |
| Executed | Yes | Partially (16-run matrix in progress) |
| Discriminates retrieval configs | Weakly | Strongly |

ECDR-Bench does not replace GRB. GRB provides comparability to published systems. ECDR-Bench tests whether a configuration that wins on GRB also wins on the document type mix that large enterprises actually deploy. The two benchmarks together answer different questions.

## Scope and Known Limitations

**What it measures.** Whether a retrieval configuration assembles cross-domain evidence correctly on a corpus that structurally resembles enterprise knowledge management content, evaluated against the retrieval calls a well-designed planner would issue.

**What it does not measure.** Scale (500 questions; significance requires bootstrap CIs at the 0.02-point level). Domain coverage (four public document types; internal enterprise documents differ in register and structure). Indexing freshness (all documents are 2024–2026 snapshots; document currency is not tested). End-to-end planner quality (the benchmark isolates the retrieval step; it does not measure whether a planner correctly decomposes queries or synthesizes multi-step results). Constrained-DSL retrieval (the benchmark assumes natural-language queries against the full corpus with no structured query layer).

**Expected score range.** Absolute scores on ECDR-Bench will be lower than on GRB. The benchmark targets the retrieval calls issued during multi-step reasoning over a heterogeneous corpus — tasks that require locating domain-specific evidence amid cross-domain vocabulary noise. No single-pass retrieval fully satisfies these questions. This is expected and not a defect.

**Why retrieval choice still matters.** A multi-step reasoning architecture does not remove the need to choose a retrieval configuration — it makes the choice more consequential. Every step in the reasoning chain issues a retrieval call; retrieval quality determines what evidence the planner has available at each step, and errors compound across steps. ECDR-Bench is the mechanism for making that choice empirically rather than by assumption. Without a benchmark that exercises cross-domain joins under vocabulary collision conditions, there is no principled basis for selecting a retrieval configuration in enterprise deployments.

**Relationship to constrained DSL planners.** A well-engineered planner can emit structured retrieval instructions — filter by document type, date range, entity, or score threshold — that bypass the vocabulary collision problem at the architectural level. When `retrieve(domain="patent", assignee="Apple", year=2025)` is possible, retrieval configuration matters less and planner query-formulation quality matters more. ECDR-Bench tests the harder case: natural-language sub-queries against the full heterogeneous corpus, with no structured query layer. This represents either the realistic state of many enterprise deployments, or a measurement of how much work the constrained DSL is doing — the gap between constrained and unconstrained retrieval scores quantifies the value of adding that layer.

Structured output constraints are not limited to query formulation — they apply to every LLM invocation in the pipeline. Any call to an LLM (query formulation, retrieval reranking, answer generation, evaluation) can be constrained to a typed schema, reducing the output space and the corresponding failure modes. The SRR (structured response + reprompting) configuration tested in ECDR-Bench is one instance of this general pattern, applied at the answer generation step. SRR forces the model to decompose its answer into structured fields rather than free-form prose, which prevents fluent-sounding hallucination past weak retrieved evidence — the schema requires a specific field that the evidence must support. Applied instead at the query formulation step, the same pattern produces the retrieval DSL effect: a schema-constrained query that filters by document type, date, or entity before retrieval. The ±SRR dimension in the benchmark matrix measures the contribution of output-side constraints at the generation step, holding the retrieval configuration constant.
