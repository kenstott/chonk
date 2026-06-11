# Methodology — Frontier-Assisted Training Data Generation for AIPA Fine-Tuning

**Status:** Draft
**Date:** 2026-06-09
**Author:** WTE Pre-Engagement Research

---

## Overview

This methodology uses Claude Code as an intelligent corpus analyst to generate high-quality training datasets for AIPA fine-tuning. The goal is to produce open-weight models that perform at frontier-model quality across the full breadth of AIPA skills in use for a specific engagement — without relying on a frontier model at inference time.

---

## The Core Theory

Frontier models perform well on software engineering tasks because they have seen vast amounts of code across many patterns and paradigms. Open-weight models lag not because they are fundamentally less capable, but because they have not been exposed to the specific patterns, conventions, and architecture of a particular codebase.

Fine-tuning on examples derived from that codebase closes this gap — combining frontier-quality task execution with deep familiarity with the specific legacy system. The key insight: use the frontier model to **generate** training data, not to run at inference time. This transfers frontier knowledge into the fine-tuned model permanently, making inference cheap and local.

---

## Skill Clusters

AIPA skills span fundamentally different input/output formats. A single fine-tuned model trained across all skill types will trade off quality — a model optimised for code transformation will produce weaker documentation and weaker audit reports. Skills are therefore grouped into clusters, each potentially warranting a separate fine-tuned model.

| Cluster | Skills | Input | Output | Recommended model tier |
|---|---|---|---|---|
| A — Code Transformation | Code Review, Bug Detection, Refactoring, Performance, API Design, Legacy Modernization | Code | Transformed/improved code | 14B–34B |
| B — Documentation | Documentation, Intelligent Documentation Generator | Code | Markdown documentation | 7B–13B |
| C — Test Generation | Test Generation | Code | Test code | 14B–34B |
| D — Analysis & Audit | Security Audit, Quality Metrics, Dependency Scanner, Bundle Analyzer, Git History Analyzer | Code / metadata | Structured report | 7B–13B |
| E — Data & SQL | SQL Export, Data Analysis | SQL / schema | Optimised queries / analysis | 7B–13B |

See `skill_clusters.md` for full cluster definitions, additional Judge dimensions per cluster, and cluster-specific manifest fields.

### Model Strategy

**Option 1 — Single model:** One fine-tune run across all clusters. Simple but quality trades off across skill types. Use only when one cluster strongly dominates.

**Option 2 — One model per cluster:** Separate fine-tune per cluster. Maximum quality but full operational overhead.

**Option 3 — Hybrid (recommended default):** One model for Cluster A (code transformation — highest complexity). One combined model for Clusters B, C, D (all produce structured text — formats are similar enough to share). Two models cover all clusters at practical operational cost.

---

## Three-Skill Architecture

| Skill | Input | Output | Runs |
|---|---|---|---|
| Interviewer | Codebase + user interview + `manifest_delta_*.md` (iteration 2+) | `modernization_manifest_{cluster}.md` per cluster | Once per engagement or per iteration |
| Generator | `modernization_manifest_{cluster}.md` + codebase | Raw training dataset per cluster | Once per cluster per iteration |
| Judge | Raw dataset + manifest | Annotated dataset + quality report + review queue | Once per cluster per iteration |

---

## Alignment Algorithm Flag

The manifest carries an `Alignment algorithm` field — SFT, DPO, or GRPO — that cascades through Generator and Judge.

| Algorithm | Data structure | Generator produces | Judge additions |
|---|---|---|---|
| SFT | instruction, input, output | Correct task examples | Standard rubric D1–D6 + cluster dimensions |
| DPO | prompt, chosen, rejected | Correct `chosen` + plausible-but-wrong `rejected` | D7 rejected plausibility, D8 chosen/rejected contrast |
| GRPO | prompt + reward_criteria | Prompts with checkable reward criteria | D9 reward criteria quality; D1/D2/D4 not applicable |

**SFT is the recommended starting point for all clusters.**

---

## The Iterative Loop (per cluster)

```
Interviewer
  ← skill audit (which clusters are in scope?)
  ← modernization_manifest_{cluster}.md (new or refined)
  ← manifest_delta_{cluster}.md (iteration 2+)
        ↓
Generator (run once per in-scope cluster)
  → training_data_{algorithm}_{cluster}_{source}_{target}_{date}.csv / .jsonl
        ↓
Judge (run once per cluster)
  → annotated_dataset_{cluster}_{date}.csv
  → judge_report_{cluster}_{date}.md
  → review_queue_{cluster}_{date}.md
        ↓
Human Review (per cluster)
  → annotated decisions
  → manifest_delta_{cluster}.md
        ↓
Back to Interviewer
        ↓
... repeat per cluster until pass rate ≥ 90% ...
        ↓
Load clean datasets into AIPA (one dataset per model)
```

Human review is a **learning signal**, not a repair step. Decisions feed `manifest_delta_{cluster}.md`, which refines the manifest for the next iteration. Convergence typically occurs in 2–3 iterations per cluster.

---

## Quality Convergence

| Iteration | Expected pass rate | Primary failure mode |
|---|---|---|
| 1 | ~60–70% | Manifest gaps surface here |
| 2 | ~75–85% | Specific pattern gaps from delta |
| 3 | ~88–95% | Residual edge cases |

**Threshold for loading into AIPA: ≥ 90% pass rate with no systemic failure modes.**

---

## The Manifest as a Living Deliverable

By production quality, each cluster's manifest contains human-validated rules for every significant task pattern in the codebase. Beyond training data, it serves as onboarding documentation, code review reference, versioned record of intent, and client deliverable.

---

## Frontier Model Access — Go / No-Go

**Anonymization is not viable.** Training examples must contain real class names, method signatures, and identifiers.

| Scenario | Path |
|---|---|
| Enterprise API agreement acceptable | Claude API / GPT-4o with enterprise data processing terms |
| External API not acceptable, self-hosted frontier model available | Self-hosted Llama 405B or equivalent |
| No external or frontier model access permitted | Methodology not applicable |

---

## Reference Files

| File | Produced by | Consumed by |
|---|---|---|
| `skill_clusters.md` | This document (static reference) | Interviewer |
| `modernization_manifest_{cluster}.md` | Interviewer | Generator, Judge |
| `manifest_delta_{cluster}.md` | Human Review | Interviewer (next iteration) |
| `training_data_{algorithm}_{cluster}_{source}_{target}_{date}.csv/jsonl` | Generator | Judge, AIPA |
| `judge_report_{cluster}_{date}.md` | Judge | Human Review |
| `review_queue_{cluster}_{date}.md` | Judge | Human Review |
| `annotated_dataset_{cluster}_{date}.csv` | Judge | Human Review, AIPA (final) |

---

## Midmarket Relevance

Midmarket companies have real AI ambitions and proprietary codebases but no on-premise GPU infrastructure and budget constraints that preclude frontier inference at scale. This methodology combined with an external training backend (e.g. Together.ai) removes both barriers. The hybrid model strategy (Option 3) is particularly well-suited to midmarket — two models cover the full AIPA skill breadth at a cost and complexity level appropriate for the segment.

---

## Revision note — 2026-06-10

Three refinements now reflected in the skills and manifest schema:
1. **Verified, not judged, outputs** — the Generator compiles every code output against the manifest's build command (and runs gold I/O where present) before it enters the dataset; the Judge's D1/D2 consume those results. Unverified frontier output is not treated as gold.
2. **Objective & validation capture** — each code-cluster manifest opens with a §0 recording the objective, scope mode (e.g. pure language port), bulk target, and the tests-green success metric.
3. **Consistency check** — the Judge flags the same source pattern transformed inconsistently across examples (M2), preventing a confidently-inconsistent adapter.
