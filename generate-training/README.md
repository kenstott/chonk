# AIPA Training Pipeline

A Claude Code skill suite for generating high-quality fine-tuning training data for AIPA Tuning, grounded in a real codebase rather than invented examples.

**Author:** Kenneth Stott

---

## What This Is

AIPA Tuning fine-tunes open-weight LLMs to perform software engineering tasks. The quality of a fine-tuned model is bounded by the quality of its training data. Generic or invented training examples produce generic results.

This pipeline uses a frontier model (Claude, GPT-4o, or a self-hosted equivalent) to read the actual codebase and generate instruction-tuned training examples that are grounded in real patterns, real idioms, and real transformation decisions from that specific project. The frontier model generates the training data once; the fine-tuned open-weight model runs at inference time — cheap, local, and codebase-aware.

---

## Contents

```
setup_training.sh                   — run this first
skill_interviewer.md                → installs as aipa-training-interviewer/SKILL.md
skill_generator.md                  → installs as aipa-training-generator/SKILL.md
skill_judge.md                      → installs as aipa-training-judge/SKILL.md
skill_clusters.md                   — ref file: cluster definitions, model strategy
modernization_manifest_schema.md    — ref file: manifest structure and field guidance
manifest_delta_schema.md            — ref file: delta file schema with worked examples
methodology.md                      — full pipeline overview (human reading)
README.md                           — this file
```

---

## Prerequisites

- macOS with bash 3.2+ (the macOS default — no Homebrew required)
- Claude Code active in the target codebase
- `ANTHROPIC_API_KEY` set (injected automatically by Claude Code)
- Frontier model API access approved for the codebase (see Governance below)
- Git repository (recommended but not required)

---

## Installation

Drop all files from this package into a single flat directory, then from the root of the codebase you want to train on:

```bash
chmod +x /path/to/setup_training.sh
/path/to/setup_training.sh              # normal run — prompts if manifests exist
/path/to/setup_training.sh --restart    # back up and clear all manifests for a fresh start
```

The script installs three Claude Code skills, a shared refs directory, and creates a `training/` directory for runtime artifacts:

```
{codebase-root}/
  .claude/
    refs/
      skill_clusters.md
      modernization_manifest_schema.md
      manifest_delta_schema.md
    skills/
      aipa-training-interviewer/
        SKILL.md
      aipa-training-generator/
        SKILL.md
      aipa-training-judge/
        SKILL.md
  training/
    methodology.md
    skill_clusters.md
    .gitignore
```

Ref files are referenced from within skills using project-root paths, e.g. `.claude/refs/skill_clusters.md`. On re-runs, the script overwrites all skill and ref files with the latest versions.

---

## The Three Skills

### aipa-training-interviewer

**Run once per engagement, and at the start of each subsequent iteration.**

Reads the codebase, conducts a skill audit to determine which AIPA skill clusters are in scope, interviews you per cluster, self-critiques the resulting manifest for gaps and ambiguity, and produces one `training/modernization_manifest_{cluster}.md` per in-scope cluster. On iteration runs, also ingests `training/manifest_delta_{cluster}.md` files to incorporate lessons from the previous cycle.

Invoke in Claude Code: `use the aipa-training-interviewer skill`

### aipa-training-generator

**Run once per in-scope cluster per iteration.**

Reads the cluster manifest and traverses the full codebase to produce a raw training dataset. Output structure adapts to the alignment algorithm declared in the manifest (SFT, DPO, or GRPO). Asks no questions — all configuration comes from the manifest.

Invoke in Claude Code: `use the aipa-training-generator skill` (specify which cluster)

### aipa-training-judge

**Run once per cluster per iteration.**

Evaluates every example in the raw dataset against a structured rubric. Core dimensions apply to all clusters; additional cluster-specific dimensions apply automatically (documentation completeness, test coverage, finding quality, etc.). Produces an annotated dataset, a quality report, and a human review queue. Routes the pipeline to the correct next action explicitly.

Invoke in Claude Code: `use the aipa-training-judge skill` (specify which cluster)

---

## Skill Clusters

AIPA skills span fundamentally different input/output formats. Training data is generated per cluster, and separate fine-tuned models are recommended per cluster (or at minimum, one model for code transformation and one for structured text output).

| Cluster | AIPA Skills | Output format | Recommended model tier |
|---|---|---|---|
| A — Code Transformation | Code Review, Bug Detection, Refactoring, Performance, API Design, Legacy Modernization | Code | 14B–34B |
| B — Documentation | Documentation, Intelligent Documentation Generator | Markdown | 7B–13B |
| C — Test Generation | Test Generation | Test code | 14B–34B |
| D — Analysis & Audit | Security Audit, Quality Metrics, Dependency Scanner, Bundle Analyzer, Git History Analyzer | Structured report | 7B–13B |
| E — Data & SQL | SQL Export, Data Analysis | SQL / analysis | 7B–13B |

See `skill_clusters.md` for full definitions, model strategy options, and cluster-specific Judge dimensions.

---

## The Iterative Loop

```
Interviewer  →  manifest(s)
Generator    →  raw dataset (per cluster)
Judge        →  quality report + review queue (per cluster)
Human review →  manifest_delta (per cluster)
↑___________________________|
repeat until Judge pass rate ≥ 90% per cluster
↓
load clean dataset(s) into AIPA Dataset Builder
fine-tune in AIPA Tuning
```

Human review is a learning signal, not a repair step. Review decisions are captured in `manifest_delta_{cluster}.md` and fed back into the Interviewer at the start of the next iteration, producing a progressively sharper manifest. Convergence typically occurs in 2–3 iterations.

---

## Alignment Algorithms

The manifest carries an `Alignment algorithm` field that drives Generator output structure and Judge evaluation criteria.

| Algorithm | Data structure | When to use |
|---|---|---|
| SFT | instruction, input, output | Default — always start here |
| DPO | prompt, chosen, rejected | After a strong SFT baseline; improves preference alignment |
| GRPO | prompt, reward_criteria | Requires reward function design experience; not recommended for first runs |

---

## Runtime Files (in `training/`)

| File | Produced by | Consumed by |
|---|---|---|
| `skill_audit.md` | Interviewer | Interviewer (subsequent runs) |
| `modernization_manifest_{cluster}.md` | Interviewer | Generator, Judge |
| `manifest_delta_{cluster}.md` | Human review | Interviewer (next iteration) |
| `training_data_{algorithm}_{cluster}_{source}_{target}_{date}.csv/jsonl` | Generator | Judge, AIPA |
| `annotated_dataset_{cluster}_{date}.csv` | Judge | Human review, AIPA (final) |
| `judge_report_{cluster}_{date}.md` | Judge | Human review |
| `review_queue_{cluster}_{date}.md` | Judge | Human review |

The `training/.gitignore` (created by the setup script) excludes generated datasets and reports but commits manifests, deltas, and the skill audit — the artifacts that should be versioned.

---

## Governance

This methodology requires a frontier model to read the real codebase at data generation time. **Anonymization is not viable** — training examples must contain real class names, method signatures, and architectural identifiers to be useful.

| Scenario | Path |
|---|---|
| Enterprise API agreement acceptable | Claude API / GPT-4o with enterprise data processing terms |
| External API not acceptable, self-hosted frontier model available | Self-hosted Llama 405B or equivalent |
| No external or frontier model access permitted | Methodology not applicable |

Confirm data governance approval before running the Interviewer.

---

## Reference File Structure

Shared reference files live in `.claude/refs/` and are referenced from within skills using project-root paths:

```
.claude/refs/skill_clusters.md
.claude/refs/modernization_manifest_schema.md
.claude/refs/manifest_delta_schema.md
```

This works because Claude Code resolves file references relative to the project root, not relative to the skill directory. A single copy in `.claude/refs/` is sufficient — no duplication needed across skill directories. Re-running `setup_training.sh` updates all files in place.

---

## Loading into AIPA

Once the Judge pass rate reaches ≥ 90% for a cluster:

1. Open AIPA Tuning → Data → Dataset Builder
2. Upload the `annotated_dataset_{cluster}_{date}.csv` (accepted examples only)
3. Column mapping: `instruction` → instruction, `input` → input, `output` → output
4. Proceed to the Fine-tuning wizard
5. Select base model appropriate for the cluster tier (see table above)
6. Select training method (LoRA recommended; QLoRA for limited VRAM; full fine-tune via external backend e.g. Together.ai for largest models)

---

## Further Reading

- `methodology.md` — full pipeline narrative with rationale
- `skill_clusters.md` — cluster definitions, decision matrix, model strategy options, cluster-specific Judge dimensions
- `modernization_manifest_schema.md` — complete manifest schema with field guidance
- `manifest_delta_schema.md` — delta file schema with five worked examples

---

## Revisions — 2026-06-10

- **skill_generator.md → 2.1:** complexity band (min/max construct size), output execution gate (compile + gold-I/O, replacing prompt self-check), dedup registry + duplication histogram.
- **skill_interviewer.md → 7.1:** pure-language-port handling (architecture held constant, behaviour preserved), build/test command capture, §0 objective & validation, self-critique checks.
- **skill_judge.md → 2.1:** D1/D2 consume execution results for code clusters; cross-example consistency check (M2).
- **modernization_manifest_schema.md:** §0 Objective & Validation, §2 build/test commands, §3 output-correctness rules, §4 operational complexity band.
- **methodology.md:** revision note summarising the above.
- Unchanged: skill_clusters.md, manifest_delta_schema.md, setup_training.sh.
- Added (separate, not installed by setup): modernization_manifest_A.md — a filled Cluster A manifest for the Python→Go pure-port AIPA evaluation.
