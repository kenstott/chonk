# Claude Code Skill — Generator (Training Data Generation)

**Version:** 2.1
**Date:** 2026-06-09
**Pipeline role:** Consumes one `modernization_manifest_{cluster}.md` and the full codebase to produce a raw training dataset for that cluster. Run once per in-scope cluster.
**Output:** `training_data_{algorithm}_{cluster}_{source}_{target}_{date}.csv` and/or `.jsonl`

---

## Skill Identity

You are a training data generation specialist. Read the specified cluster manifest and traverse the full codebase to produce a training dataset for that cluster's AIPA skills.

You ask **no questions**. All configuration comes from the manifest. If the manifest is missing, incomplete, or frontier model access is not confirmed, stop and direct the user to run the Interviewer skill first.

You produce a **raw dataset** for the Judge to evaluate — not a final clean dataset.

**Invocation:** The user specifies which cluster to run. Example: "Run the Generator for Cluster A" → read `modernization_manifest_A.md`.

---

## Step 1 — Identify and Validate the Manifest

Determine which cluster manifest to read based on user instruction or context. Read `modernization_manifest_{cluster}.md`.

Confirm all of the following are present:

- [ ] Cluster identifier (A / B / C / D / E)
- [ ] Target language / platform (Cluster A, C, E) or documentation format (Cluster B) or report format (Cluster D)
- [ ] At least two task rules / patterns
- [ ] Target model name
- [ ] Alignment algorithm (SFT | DPO | GRPO)
- [ ] Example count
- [ ] Output format
- [ ] Frontier model access: confirmed

If anything is missing, **stop** and report exactly what is absent.

---

## Step 2 — Apply Model Calibration

Derive example complexity from the model tier in the manifest:

| Model tier | Complexity | Max input tokens | Max output tokens |
|---|---|---|---|
| 3B–7B | Atomic — single concern | 300 | 300 |
| 8B–13B | Moderate — 1–2 concerns | 600 | 600 |
| 14B–34B | Complex — multi-concern | 1200 | 1200 |
| 34B+ | Full method or small class | 2000 | 2000 |

---

## Step 3 — Plan Coverage

Before generating, produce a coverage plan:
- List all modules and layers relevant to this cluster
- Allocate example budget proportionally (weighted toward frequently occurring patterns)
- Map task rules from the manifest to specific modules and layers
- Mark excluded modules

Track against this plan. Report deviations in the summary.

---

## Step 4 — Generate Examples

Output structure depends on the alignment algorithm in the manifest. Apply cluster-specific generation guidance below.

---

### SFT — instruction / input / output

**instruction** — Self-contained task description. Cluster-specific instruction patterns:

**Cluster A (Code Transformation):**
"[Transform/Review/Refactor/Optimise/Fix] this [source language] [construct type] [to target platform / for quality dimension], applying [specific rule from manifest]. [Any specific preservation or convention requirement.]"

**Cluster B (Documentation):**
"Generate [documentation type — inline comments / module docstring / README section / API reference] for this [source language] [construct type] following [style guide from manifest]. [Audience and format requirements.]"

**Cluster C (Test Generation):**
"Generate a [test type — unit / integration / edge case] test suite for this [source language] [construct type] using [test framework from manifest], covering [coverage requirements from manifest]. Follow [naming convention]."

**Cluster D (Analysis & Audit):**
"Perform a [audit type — security / quality / dependency / git history] analysis of this [scope] and produce a structured report with [required sections from manifest]. Use severity scale: [scale from manifest]."

**Cluster E (Data & SQL):**
"Optimise this [database platform] query for [optimisation focus from manifest]. Follow [conventions from manifest]. Preserve the result set exactly."

**input** — Source material extracted directly from the codebase. Never invented. Within token limits. Free of credentials, secrets, PII. Cluster-specific:
- Clusters A, B, C, E: code from the codebase
- Cluster D: code, dependency manifest content, or git log excerpt

**output** — Correctly executed task result. Cluster-specific quality standards:
- Cluster A: correct, compilable, idiomatic target-platform code preserving business logic
- Cluster B: accurate, complete documentation in the specified format and style
- Cluster C: correct, runnable test code in the specified framework covering the specified scenarios
- Cluster D: structured Markdown report following the specified format with accurate findings
- Cluster E: correct, optimised SQL for the specified platform returning identical results

For all clusters: add `// Note:` or `<!-- Note: -->` comments for ambiguous decisions.

---

### DPO — prompt / chosen / rejected

**prompt** — Combined instruction and input as a single string.

**chosen** — Correctly executed task result. Same quality standard as SFT output.

**rejected** — A plausible-but-wrong result. Cluster-specific rejected patterns:

**Cluster A:** Partial transformation (signature converted but body not), wrong target idiom, preserved anti-pattern, missing error handling carry-over.

**Cluster B:** Documentation that describes what the code does mechanically but misses the intent; incorrect parameter descriptions; missing exception documentation; wrong format for the style guide.

**Cluster C:** Tests that only cover the happy path when edge cases are required; tests with shared mutable state; wrong assertion style for the framework; missing error case coverage.

**Cluster D:** Report with correct finding titles but wrong severity ratings; findings that are real but not grounded in the actual code; missing the required report sections; vague recommendations without actionable specifics.

**Cluster E:** Query that returns the same results but without the specified optimisation; query that improves performance but changes the result set; correct optimisation but wrong SQL dialect for the target platform.

---

### GRPO — prompt / reward_criteria

**prompt** — Combined instruction and input.

**reward_criteria** — Binary, objectively checkable conditions. Cluster-specific examples:

**Cluster A:** "Output compiles for [target platform]", "Output applies rule [Rnn]", "Method signature matches original", "No legacy constructs remain"

**Cluster B:** "All public methods have docstrings", "Parameter descriptions match actual parameters", "Return value documented", "Follows [style guide] format"

**Cluster C:** "Tests compile and run", "At least one test covers error case", "Test names follow [convention]", "No shared mutable state between tests"

**Cluster D:** "Report contains all required sections: [list]", "Each finding includes severity rating", "Severity uses defined scale: [list]", "Each finding includes a recommendation"

**Cluster E:** "Query is syntactically valid for [platform]", "Query returns identical result set", "Query applies [specified optimisation]"

---

## Step 5 — Output

**Write output in batches, not as one large string.** Passing hundreds of examples in a single agent prompt will overflow the context window. Use this pattern instead:

1. Each generation batch writes its examples to a numbered temp file:
   `training/tmp/batch_{batch_id}.jsonl`

2. After all batches complete, concatenate with bash:
   ```bash
   cat training/tmp/batch_*.jsonl > training/training_data_{algorithm}_{cluster}_{source}_{target}_{date}.jsonl
   rm -rf training/tmp/
   ```

3. Verify line count matches expected example count.

Filenames include cluster identifier:

### CSV
```
instruction,input,output    (SFT)
prompt,chosen,rejected      (DPO)
```
Filename: `training_data_{algorithm}_{cluster}_{source}_{target}_{date}.csv`

### JSONL
```
{"instruction": "...", "input": "...", "output": "..."}    (SFT)
{"prompt": "...", "chosen": "...", "rejected": "..."}      (DPO)
{"prompt": "...", "reward_criteria": [...]}                (GRPO)
```
Filename: `training_data_{algorithm}_{cluster}_{source}_{target}_{date}.jsonl`

All CSV fields double-quoted. Internal quotes escaped as `""`.

**If a batch exceeds the output token limit:** split it into two smaller batches of equal size rather than truncating. Never truncate examples. Log any splits in the generation summary.

---

## Step 6 — Generation Summary

- Cluster processed
- Algorithm used
- Total examples generated vs target
- Coverage breakdown by module and layer
- Coverage breakdown by task rule
- Average token lengths (instruction / input / output)
- Modules under-sampled, skipped, or excluded and why
- DPO: breakdown of rejected variant error types used
- GRPO: summary of reward criteria defined
- Recommended AIPA training configuration (method, epochs, rank)
- Recommended next step: run Judge skill for this cluster

---

## Related Skills

- `skill_interviewer.md` — produces the manifests this skill consumes
- `skill_judge.md` — evaluates the datasets this skill produces

## Reference Files

- `.claude/refs/skill_clusters.md` — cluster definitions and cluster-specific generation guidance
- `.claude/refs/modernization_manifest_schema.md` — manifest structure reference

---

## Revision 2.1 — Judge-feedback round 1 + execution gate (2026-06-10)

Applied after a Cluster A Judge run scored 57.8% pass with three dominant, isolated failure modes (no single dimension > 30%): M6 complexity mismatch, M3 output error, M7 duplicate. Most relevant to code clusters (A, C, E). Feedback assessed as sound; the M3 remedy is upgraded from "prompt self-check" to an actual execution gate.

**Fix 1 — Complexity band (M6 / D5).** Step 2's token caps bound the top but not the floor, so datasets fill with trivial constructs and occasional oversized pastes. Amend Steps 2–3 input selection:
- Minimum meaningful construct: >= 10 lines AND at least one branch or loop (at least one non-trivial transformation decision). Skip pure getters/setters, one-line delegations, and bare constant declarations — the base model already handles these and they dilute the set.
- Maximum per example: ~60 lines, never exceeding the tier's token cap. Split oversized classes into method-level or cohesive-block examples rather than pasting the whole class.

**Fix 2 — Output validation gate (M3 / D1).** Do NOT rely on a prompt instruction to self-check compilation — models are unreliable compilers, which is how non-compiling output (missing imports, wrong type signatures, source-language artifacts such as Python slice notation left in Go) got through. Add an execution gate AFTER generation, BEFORE the example enters the dataset:
- Compile every output / DPO `chosen` against the manifest's Target build command (e.g. `go build ./...` / `go vet`; `tsc`; `javac`). Reject non-compiling examples and regenerate.
- Lint for source-language artifacts that cannot be valid in the target (e.g. Python `[a:b]` slices, f-strings, `self`, `None`). Reject on hit.
- Where gold I/O pairs exist (Target test command), run them and require green — not merely compile-clean.
- Record compile-pass and test-pass rates in the Step 6 summary. This converts "claimed correct" into "verified correct."
- Applies to code clusters. For text clusters (B, D) the analogous gate is format/lint validation, not compilation.

**Fix 3 — Deduplication + repetition signal (M7 / D6).** Near-duplicates concentrate in large modules where parallel batches sample the same common pattern. Amend Step 3:
- Maintain a registry of source functions/constructs already emitted (normalized signature / AST hash). Exclude near-duplicates of already-emitted sources before generation.
- Keep enough instances per pattern for robust learning, but drop redundant near-identical ones; prioritise variety within a pattern over repetition of it.
- Emit the duplication histogram (instances per source pattern) in the Step 6 summary. High duplication is the repetition signal indicating which modules are most adapter-tractable — keep the count even as you drop the duplicates.
