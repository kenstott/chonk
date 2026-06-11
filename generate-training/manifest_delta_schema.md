# Reference — Manifest Delta Schema

**File:** `manifest_delta_{cluster}.md` (one per cluster)
**Produced by:** Human review of `review_queue_{cluster}_{date}.md`
**Consumed by:** Interviewer skill (next iteration)
**Location:** Repository root or `/training/`

---

## Purpose

`manifest_delta_{cluster}.md` captures lessons learned from a human review cycle for one cluster. It is not a repair list of bad examples — it is a structured set of instructions for how that cluster's manifest should be refined before the next Generator run.

The Interviewer ingests all `manifest_delta_*.md` files at the start of the next iteration and incorporates every item into the appropriate manifest section.

**Do not write a delta item for every rejected example.** Look for patterns. Ten examples failing with M1 (manifest gap) on the same pattern → one delta item.

---

## Schema

```markdown
# Manifest Delta — Cluster {letter}: {name}
**Iteration:** {integer}
**Date:** {ISO date}
**Review queue source:** {review_queue_{cluster}_{date}.md}
**Algorithm used this iteration:** {SFT | DPO | GRPO}
**Judge pass rate this iteration:** {%}

---

## Delta Items

### DELTA-{n} — {short description}

**Type:** {Gap | Ambiguity | Correction | Coverage | Complexity | DPO-Rejected | GRPO-Criteria | Cluster-Specific}
**Manifest section:** {e.g. Section 3 — Task Rules}
**Failure category:** {M1–M12}
**Affected examples:** {IDs or "multiple — see pattern"}
**Failure pattern observed:** {description}

**Proposed manifest change:**
> {Exact text to add, replace, or remove. Be precise.}

**Rationale:** {Why this prevents recurrence}

---
```

---

## Delta Item Types

| Type | When to use |
|---|---|
| Gap | Pattern exists in codebase but no manifest rule covers it (M1) |
| Ambiguity | Rule exists but too vague — examples interpreted it differently (M2) |
| Correction | Rule is incorrect — examples produce wrong output (M3, M4) |
| Coverage | Module or layer under-represented — needs more examples next run |
| Complexity | Examples too simple or too complex for the model tier (M6) |
| DPO-Rejected | `rejected` variants recurrently too shallow — need better error type guidance (M8) |
| GRPO-Criteria | Reward criteria vague or uncheckable (D9 failures) |
| Cluster-Specific | Documentation gap (M9), test gap (M10), finding gap (M11), query error (M12) |

---

## Worked Examples

### DELTA-1 (Cluster A) — Add rule for batch processing patterns

**Type:** Gap
**Manifest section:** Section 3 — Task Rules
**Failure category:** M1
**Affected examples:** ex_047, ex_112, ex_203 and 7 others
**Failure pattern observed:** Codebase has batch processing logic (PERFORM UNTIL loops) that the Generator attempted to transform with no manifest guidance, producing inconsistent output — some used Spring Batch, others used plain Java streams.

**Proposed manifest change:**
> Add to Section 3 Task Rules table:
> | R07 | Batch processing | PERFORM UNTIL file read loops → Spring Batch ItemReader/ItemProcessor/ItemWriter. Use FlatFileItemReader for file-based input. Do not use Java streams for batch — Spring Batch required for all batch processing to maintain restart/retry capability. |

**Rationale:** Without this rule the Generator had no basis for choosing between Spring Batch and streams, producing 10 inconsistent examples the Judge correctly rejected on D4.

---

### DELTA-2 (Cluster B) — Specify constructor documentation requirement

**Type:** Gap
**Manifest section:** Section 3 — Task Rules
**Failure category:** M9
**Affected examples:** ex_031, ex_055, ex_089 and 4 others
**Failure pattern observed:** 7 documentation examples covered public methods but not constructors. The manifest rule said "document all public interfaces" but constructors were consistently omitted.

**Proposed manifest change:**
> Amend Rule R01 from:
> "Document all public methods with JavaDoc including parameters, return values, and exceptions"
> To:
> "Document all public methods AND constructors with JavaDoc. Required fields: @param for every parameter, @return for non-void methods, @throws for every checked exception. Constructors must document their purpose, all parameters, and any side effects (e.g. resource initialisation)."

**Rationale:** Makes the constructor requirement explicit, preventing the recurring omission.

---

### DELTA-3 (Cluster C) — Improve test coverage for null inputs

**Type:** Gap
**Manifest section:** Section 3 — Task Rules
**Failure category:** M10
**Affected examples:** ex_018, ex_067, ex_134 and 9 others
**Failure pattern observed:** 12 test generation examples only covered happy path and basic error cases. Null input scenarios were never tested despite the manifest requiring edge case coverage.

**Proposed manifest change:**
> Add to Section 3 Task Rules table:
> | R04 | Null input tests | Every public method with reference type parameters must include at least one test for null input. If the method has null guard logic, assert the expected exception. If it does not, note this in a @Disabled test with a TODO comment. |

**Rationale:** Closes the null input gap that caused 12 examples to fail D12 (test coverage).

---

### DELTA-4 (Cluster D) — Tighten severity rating guidance

**Type:** Ambiguity
**Manifest section:** Section 3 — Task Rules
**Failure category:** M2, M11
**Affected examples:** ex_022, ex_089, ex_145 and 6 others
**Failure pattern observed:** 9 security audit examples rated SQL injection vulnerabilities inconsistently — some as Critical, some as High, based on the same underlying pattern. The manifest defined the severity scale but not the criteria for each level.

**Proposed manifest change:**
> Add to Section 2 (Task Target) under Severity scale:
>
> Severity criteria:
> - Critical: directly exploitable with no authentication required; immediate data loss or system compromise possible
> - High: exploitable with standard user access; significant data exposure or privilege escalation possible
> - Medium: requires specific conditions or elevated access; limited impact
> - Low: theoretical risk; difficult to exploit; minimal impact
> - Info: best practice improvement; no direct security risk
>
> SQL injection is always Critical. XSS in output contexts is High. Missing input validation without direct injection path is Medium.

**Rationale:** Eliminates the ambiguity that caused inconsistent severity ratings across 9 examples.

---

### DELTA-5 (Cluster A, DPO) — Improve rejected variant guidance

**Type:** DPO-Rejected
**Manifest section:** Section 4 — Algorithm-specific configuration
**Failure category:** M8
**Affected examples:** ex_034, ex_067 and 14 others
**Failure pattern observed:** 16 of 60 DPO examples had `rejected` variants that were just the unmodified legacy code — too obviously wrong to teach anything.

**Proposed manifest change:**
> Add to Section 4 DPO rejected variant guidance:
> "Do not use unmodified legacy code as a rejected variant. Required error types: (1) partial transformation — method signature converted to target platform but body left in legacy style; (2) wrong Spring interface — CrudRepository used instead of JpaRepository; (3) missing error handling — transformation applied but try/catch not carried over to target idiom."

**Rationale:** Prevents the trivial rejected examples that make D7 fail.
