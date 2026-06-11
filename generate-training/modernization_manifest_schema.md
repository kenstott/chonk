# Reference — Modernization Manifest Schema

**File:** `modernization_manifest_{cluster}.md`
**Produced by:** Interviewer skill
**Consumed by:** Generator skill, Judge skill
**Location:** Repository root or `/training/`
**One file per in-scope cluster:** `_A`, `_B`, `_C`, `_D`, `_E`

---

## Full Schema

```markdown
# Modernization Manifest — Cluster {letter}: {name}
**Version:** {integer — increment each iteration}
**Generated:** {ISO date}
**Engagement:** {engagement or project name}
**Cluster:** {A — Code Transformation | B — Documentation | C — Test Generation | D — Analysis & Audit | E — Data & SQL}
**Status:** Draft — pending review | Approved

---

## 1. Source Codebase

**Primary language:** {language and version}
**Frameworks / libraries:** {list}
**Architecture:** {pattern}
**Naming conventions:** {conventions observed}
**Scale:** {file count, estimated LOC}
**Module structure:** {description}
**Notable patterns / debt relevant to this cluster:** {observations}

---

## 2. Task Target

### Cluster A — Code Transformation
**Target language / platform:** {e.g. Java 21 + Spring Boot 3.2}
**Target architecture:** {e.g. microservices with clean architecture}
**Required frameworks / libraries:** {list}
**Target conventions / style guide:** {reference or description}

### Cluster B — Documentation
**Documentation types:** {inline comments | module docstrings | README sections | API reference | architecture docs}
**Target format / style guide:** {e.g. JavaDoc, Google style, NumPy style, custom}
**Audience:** {internal developers | external API consumers | onboarding}
**Format requirements:** {any specific structural requirements}

### Cluster C — Test Generation
**Target test framework:** {e.g. JUnit 5, pytest, NUnit, Jest}
**Test types required:** {unit | integration | edge cases | performance | security}
**Coverage requirements:** {e.g. all public methods, happy path + error cases, boundary conditions}
**Test naming convention:** {e.g. methodName_condition_expectedResult}
**Mocking framework:** {e.g. Mockito, unittest.mock, Jest mocks — or none}

### Cluster D — Analysis & Audit
**Audit types in scope:** {security | quality metrics | dependency health | bundle size | git history}
**Required report sections:** {list}
**Severity scale:** {e.g. Critical / High / Medium / Low / Info}
**Finding format:** {finding title, description, affected code reference, recommendation}

### Cluster E — Data & SQL
**Database platform:** {e.g. PostgreSQL 15, MySQL 8, SQL Server 2019}
**Optimisation focus:** {performance | readability | normalisation}
**Query conventions:** {e.g. CTE style, aliasing standards, formatting rules}

---

## 3. Task Rules

Each rule must be specific enough that two developers would apply it identically.

### Cluster A
| Rule ID | Pattern | Description |
|---|---|---|
| R01 | {short name} | {specific: what to transform and exactly how} |

**Preserve exactly:**
- {item — specific, e.g. "All public API method signatures on CustomerService"}

**Intentionally change:**
- {item — specific, e.g. "Replace System.err logging with SLF4J at WARN level"}

**Exclude from transformation:**
| Module / pattern | Reason |
|---|---|
| {path or glob} | {reason} |

### Cluster B
| Rule ID | Documentation type | Format requirement | Completeness requirement |
|---|---|---|---|
| R01 | {type} | {format} | {what must be documented} |

### Cluster C
| Rule ID | Test type | Coverage requirement | Framework-specific notes |
|---|---|---|---|
| R01 | {type} | {what must be covered} | {framework-specific patterns to use} |

### Cluster D
| Rule ID | Audit type | What to look for | Report format |
|---|---|---|---|
| R01 | {type} | {specific patterns to identify} | {format of findings} |

### Cluster E
| Rule ID | Query type | Optimisation to apply | Constraints |
|---|---|---|---|
| R01 | {type} | {specific optimisation} | {must preserve result set, must use CTEs, etc.} |

---

## 4. Fine-Tuning Configuration

**Target model:** {specific named model — e.g. `Qwen2.5-Coder-14B-Instruct`, `Qwen2.5-7B-Instruct`, `CodeLlama-34B-Instruct`, `DeepSeek-Coder-33B-Instruct`. A tier description (e.g. "14B–34B") is not valid here — the Generator validates this field and will block on a tier label.}
**Model tier:** {3B–7B | 8B–13B | 14B–34B | 34B+}
**Alignment algorithm:** {SFT | DPO | GRPO}
**Example count:** {number, default 500}
**Output format:** {CSV | JSONL | both}
**Example complexity:** {atomic | moderate | complex | full-method}

*Example complexity is derived from model tier — do not set manually.*

### Algorithm-specific configuration

**If DPO:**
**Rejected variant guidance:** {specific error types to use — e.g. "partial transformations, wrong framework interface, missing error handling carry-over"}

**If GRPO:**
**Reward criteria template:** {list the standard criteria to apply — must be binary and objectively checkable}

---

## 5. Data Governance

**Frontier model access approved:** {yes | no | conditional}
**API agreement reference:** {reference}
**Excluded files / modules:** {list}
**Other constraints:** {compliance, security, data residency}

---

## 6. Iteration History

| Version | Date | Algorithm | Summary of changes | Trigger |
|---|---|---|---|---|
| 1 | {date} | {algorithm} | Initial manifest | First run |
| 2 | {date} | {algorithm} | {delta items applied} | Judge pass rate {n}% — {failure mode} |

---

## 7. Review Sign-Off

| Reviewer | Role | Status | Date |
|---|---|---|---|
| | WTE Consultant | Pending | |
| | Client Architect | Pending | |
```

---

## Field Guidance

### Task rules (section 3)
The most critical section. Rules must be specific, scoped, and unambiguous. "Replace EXEC SQL SELECT with JPA repository findById() calls" is good. "Modernize database access" is not. Apply the same specificity standard to all clusters — documentation format requirements, test coverage requirements, and audit finding formats must be equally precise.

### Cluster-specific mandatory fields
- Cluster B: documentation types, format, audience — all required
- Cluster C: test framework, test types, coverage requirements, naming convention — all required
- Cluster D: audit types, report sections, severity scale, finding format — all required
- Cluster E: database platform, optimisation focus — required

### Example complexity (section 4)
Derived automatically by the Interviewer from the model tier. Do not override.

### Iteration history (section 6)
Maintained automatically by the Interviewer. Do not edit manually.

---

## Revision — 2026-06-10 (objective capture, output correctness, complexity band)

Additions to the schema, primarily for code clusters (A; also C, E where outputs are code/SQL).

**Add §0 Objective & Validation (top of the manifest, before §1):**

~~~markdown
## 0. Objective & Validation
**Objective:** {one line — what the fine-tuned model/adapter is for}
**Scope mode:** {pure language port | re-platform | re-architecture | mixed}
**In scope:** {what transforms}
**Out of scope:** {what does NOT — e.g. architecture, business logic}
**Bulk target:** {whole codebase | named subset}
**Validation method:** {one of the three oracle paths below — the oracle is required; how you obtain it is not}
- `existing tests` — a CI suite or test file already validates the source at the external-contract level; port it alongside the code and re-run against the target. Most common path.
- `characterization tests` — no adequate tests exist; generate them from the running source before porting (Characterizer skill, if available). Captures the external contract as golden I/O.
- `golden I/O` — manually curated input/output pairs at the public API boundary; use when a running source environment is unavailable or impractical.
The Characterizer skill is optional and only needed when neither existing tests nor manual golden I/O is available.
**Success metric:** first-pass tests-green % and after-fix-loop tests-green %
**Target build command:** {e.g. go build ./...}
**Target test command:** {e.g. go test ./...}
~~~

For a **pure language port**: Target architecture = unchanged (mirror source 1:1); Intentionally change = none.

**§2 Task Target (Cluster A) — add two fields:**
- **Target build command:** {command the Generator uses to verify every output compiles}
- **Target test command:** {command used to verify behavioural equivalence where gold I/O exists}

**§3 Task Rules (Cluster A) — add an Output Correctness Rules subsection:**
- Every output must compile against the Target build command (verified by execution, not assertion).
- No source-language artifacts may survive into the target (enumerate the source→target traps for this pair).
- Imports/dependencies complete and correct; type signatures valid in the target type system.
- Where gold I/O pairs exist, output must pass them.

**§4 Fine-Tuning Configuration — operational complexity band** (defines what the tier-derived complexity means; not a manual override):
- Minimum construct: >= 10 lines AND at least one branch or loop. Exclude pure getters/setters, one-line delegations, bare constants.
- Maximum construct: ~60 lines, never exceeding the tier token cap; split larger classes into method-level examples.
