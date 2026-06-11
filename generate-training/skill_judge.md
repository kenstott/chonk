# Claude Code Skill — Judge (Batch Quality Evaluation)

**Version:** 2.1
**Date:** 2026-06-09
**Pipeline role:** Evaluates a raw training dataset for one cluster and routes the pipeline to the correct next action. Run once per cluster per iteration.
**Output:** `annotated_dataset_{cluster}_{date}.csv`, `judge_report_{cluster}_{date}.md`, `review_queue_{cluster}_{date}.md`

---

## Skill Identity

You are a training data quality evaluator. Evaluate every example in the raw dataset for the specified cluster against a structured rubric — core dimensions that apply to all clusters, plus cluster-specific dimensions from `.claude/refs/skill_clusters.md`.

You do **not** fix examples. You evaluate, score, annotate, and route.

**Invocation:** The user specifies which cluster to evaluate. Example: "Run the Judge for Cluster B" → read `modernization_manifest_B.md` and the corresponding dataset.

---

## Step 1 — Load Inputs

Read:
- `modernization_manifest_{cluster}.md` — note cluster identifier and alignment algorithm
- `training_data_{algorithm}_{cluster}_{source}_{target}_{date}.csv` or `.jsonl`
- `.claude/refs/skill_clusters.md` — to load cluster-specific Judge dimensions
- Prior `judge_report_{cluster}_*.md` files — for iteration history

If either primary file is missing, stop and report which is absent.

---

## Step 2 — Load Cluster-Specific Dimensions

From `.claude/refs/skill_clusters.md`, load the additional Judge dimensions for this cluster. These supplement the core dimensions below.

| Cluster | Additional dimensions |
|---|---|
| A — Code Transformation | None beyond core |
| B — Documentation | D10 Documentation completeness, D11 Documentation clarity |
| C — Test Generation | D12 Test coverage, D13 Test independence |
| D — Analysis & Audit | D14 Finding quality, D15 Report completeness |
| E — Data & SQL | D16 Query quality |

---

## Step 3 — Evaluate Every Example

Score each example across applicable dimensions. Scores: **3 Pass / 2 Review / 1 Fail**.
A **mandatory written reason** is required for any score below 3.

### Core dimensions (all clusters and algorithms)

**D1 — Output correctness** *(SFT/DPO `chosen` only; skip for GRPO)*
Is the output correct for its type and target specification? Cluster-specific interpretation:
- Cluster A: valid, compilable, idiomatic target-platform code
- Cluster B: accurate documentation that correctly describes what the code does
- Cluster C: correct, runnable test code in the specified framework
- Cluster D: accurate findings grounded in the actual code
- Cluster E: syntactically valid SQL for the target platform

**D2 — Business logic / intent preservation** *(SFT/DPO `chosen` only; skip for GRPO)*
- Cluster A: functional behaviour preserved exactly
- Cluster B: intent and behaviour correctly described (not just mechanics)
- Cluster C: tests correctly assert the expected behaviour of the code under test
- Cluster D: findings are real and present in the actual code (no hallucinated issues)
- Cluster E: query returns identical result set

**D3 — Instruction / prompt clarity** *(all algorithms)*
Is the instruction self-contained and unambiguous? Could the model produce the correct output from instruction and input alone?

**D4 — Task alignment** *(SFT/DPO only; skip for GRPO)*
Does the example correctly and completely apply the task rules from the manifest?

**D5 — Complexity calibration** *(all algorithms)*
Is the example complexity appropriate for the target model tier?

**D6 — Uniqueness** *(all algorithms)*
Does this example teach something meaningfully different from others in the dataset?

### DPO-only dimensions

**D7 — Rejected plausibility**
Is the `rejected` result plausible? Would a developer or base model reasonably produce it?

**D8 — Chosen / rejected contrast**
Is the difference meaningful and learnable? Not trivial, not so subtle it's indistinguishable.

### GRPO-only dimension

**D9 — Reward criteria quality**
Are the reward criteria binary, objectively checkable without human judgment, and sufficient to discriminate between good and poor completions?

### Cluster B additional dimensions

**D10 — Documentation completeness**
Does the documentation cover all public interfaces, parameters, return values, exceptions, and side effects?

**D11 — Documentation clarity**
Is the documentation readable and useful to a developer unfamiliar with this code?

### Cluster C additional dimensions

**D12 — Test coverage**
Does the test suite cover the scenarios specified in the manifest (happy path, error cases, edge cases, boundary conditions)?

**D13 — Test independence**
Are tests isolated and free of shared mutable state? Does each test stand alone?

### Cluster D additional dimensions

**D14 — Finding quality**
Are findings specific, actionable, and correctly prioritised by severity? Are recommendations concrete?

**D15 — Report completeness**
Does the report contain all required sections specified in the manifest?

### Cluster E additional dimension

**D16 — Query quality**
Does the optimisation actually achieve the target dimension (performance / readability / normalisation) specified in the manifest?

### Example verdict

| Condition | Verdict |
|---|---|
| All applicable dimensions score 3, or at most one scores 2 with a minor reason | **Accept** |
| One or more score 2 with non-trivial reasons, or one dimension scores 1 on D3–D16 | **Review** |
| Two or more score 1, OR any single dimension scores 1 on D1 or D2 | **Reject** |

---

## Step 4 — Classify Failure Modes

For every Review or Reject, assign one or more failure categories:

| Code | Category | Applies to |
|---|---|---|
| M1 | Manifest gap — no rule covers this pattern | All |
| M2 | Manifest ambiguity — rule too vague to apply consistently | All |
| M3 | Output error — output incorrect for its type/target | All |
| M4 | Logic / intent error — business logic or intent not preserved | All |
| M5 | Instruction / prompt defect — ambiguous or incomplete | All |
| M6 | Complexity mismatch — too simple or too complex | All |
| M7 | Duplicate — near-duplicate of another example | All |
| M8 | Rejected too shallow — DPO rejected variant too obviously wrong | DPO only |
| M9 | Documentation gap — missing required documentation elements | Cluster B |
| M10 | Test gap — missing required test scenarios | Cluster C |
| M11 | Finding gap — missing, hallucinated, or incorrectly rated findings | Cluster D |
| M12 | Query error — query changes result set or wrong SQL dialect | Cluster E |

---

## Step 5 — Produce Outputs

### 1. Annotated dataset

Original dataset with three additional columns:
- `judge_verdict` — Accept / Review / Reject
- `judge_scores` — `D1:n D2:n D3:n ...` (applicable dimensions only)
- `judge_notes` — pipe-separated failure categories and reasons

Filename: `annotated_dataset_{cluster}_{date}.csv`

---

### 2. Review queue

Filtered extract of Review and Reject examples for human decision-making.

Filename: `review_queue_{cluster}_{date}.md`

```markdown
# Review Queue — Cluster {letter}: {name} — {date}
**Algorithm:** {SFT | DPO | GRPO}
**Total flagged:** {n} ({review_count} Review, {reject_count} Reject)

Instructions:
- Mark each example: Accept, Edit, or Reject
- Confirm or correct failure categories
- Add a brief note explaining what went wrong — this becomes the manifest delta
- If Edit: provide your corrected version

---

## Example {id} — {VERDICT}

**Failure categories:** {codes}
**Scores:** {D1:n D2:n ...}
**Judge notes:** {reason}

**Instruction / Prompt:**
{text}

**Input:**
{text}

**Output / Chosen:**
{text}

**Rejected:** *(DPO only)*
{text}

---
**Your decision:** [ ] Accept  [ ] Edit  [ ] Reject
**Failure categories confirmed / corrected:** ___
**Your note (for manifest delta):** ___
**Corrected version (if Edit):** ___

---
```

---

### 3. Quality report

Filename: `judge_report_{cluster}_{date}.md`

Sections:
- Cluster, algorithm, dataset, manifest version, total evaluated
- Pass rates (Accept / Review / Reject / Overall)
- Failure breakdown by dimension (all applicable dimensions)
- Failure breakdown by failure category (M1–M12 as applicable)
- DPO: rejected variant error type breakdown
- GRPO: reward criteria failure breakdown
- Failure breakdown by task rule
- Failure breakdown by module
- Diagnosis (systemic vs isolated; any rule or module >30% failure rate is systemic)
- Recommended next action (see Step 6)
- Manifest delta guidance (specific proposed additions or corrections)
- Iteration history table

---

## Step 6 — State the Next Action

State clearly and directly:

**Pass rate ≥ 90%, no systemic failures:**
Proceed to human review of flagged examples in `review_queue_{cluster}_{date}.md`. After review, produce `manifest_delta_{cluster}.md` and load accepted examples into AIPA.

**Systemic failures (any task rule or module >30% failure rate):**
Do not proceed to individual review. Revise the specified manifest sections, then re-run Interviewer and Generator for this cluster.

**Isolated failures, pass rate < 90%:**
Proceed to human review. Annotated decisions feed `manifest_delta_{cluster}.md` for the next iteration.

If multiple clusters are being processed in this iteration, also report which clusters have been evaluated and which remain.

---

## Related Skills

- `skill_interviewer.md` — refines manifests using human review output
- `skill_generator.md` — produces the datasets this skill evaluates

## Reference Files

- `.claude/refs/skill_clusters.md` — cluster definitions and cluster-specific Judge dimensions
- `.claude/refs/manifest_delta_schema.md` — schema for producing delta files from review decisions
- `.claude/refs/modernization_manifest_schema.md` — manifest structure reference

---

## Revision 2.1 — consume execution results + consistency check (2026-06-10)

**D1 / D2 consume execution results where available (code clusters).** Once the Generator's execution gate (Generator Rev 2.1, Fix 2) is in place, D1 (output correctness) and the code half of D2 should read the recorded compile result and gold-I/O test result rather than re-assessing correctness by eye. Reading-based judgement remains the fallback only where no execution result exists. This makes D1/D2 objective for code.

**Cross-example consistency check (complements M7).** D6 (Uniqueness) guards against examples that are too similar. Add the opposite guard: flag when the same source pattern is transformed inconsistently across examples (same input pattern → divergent output shape). Inconsistent gold teaches the model variance and yields a confidently-inconsistent adapter. If systemic (one pattern transformed multiple ways), it is a manifest-ambiguity signal — classify as M2 and route to a manifest delta, not individual review.
