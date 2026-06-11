# Modernization Manifest — Cluster A: Code Transformation
**Version:** 1
**Generated:** 2026-06-10
**Engagement:** AIPA evaluation — Python→Go pure language port
**Cluster:** A — Code Transformation
**Status:** Draft — pending review

> Scenario: a **pure language port** — translate the Python codebase to Go, preserving observable
> behaviour exactly, with **no architectural refactoring**. Bulk-convert, build, generate/port tests,
> confirm green. This is the AIPA test vehicle: the cleanest case for a fine-tuned local specialist,
> because the work is convention-bound (mechanical 1:1 mapping), not capability-bound.

---

## 0. Objective & Validation

**Objective:** Fine-tune a local code-transformation adapter that ports Python constructs to idiomatic Go 1.22, one-to-one, preserving behaviour — so bulk conversion runs cheaply on-prem instead of against a frontier model.
**Scope mode:** pure language port
**In scope:** language-level translation of functions, classes, control flow, error handling, data structures, standard-library idioms.
**Out of scope:** architecture (package/module boundaries mirror the source), business logic, public contracts, data schemas. Nothing is re-designed.
**Bulk target:** {TODO — whole codebase | named subset, e.g. `core/` + `storage/`}
**Validation method:** `existing tests` — port the Python test suite to Go alongside the source code. Run both the Python tests against the Python source (establishes the baseline) and the Go tests against the Go output. Both must be green. Dual-run confirms behavioural equivalence and catches test-porting errors independently of source-porting errors.
**Success metric:** report **two numbers** — first-pass tests-green % (Go output before any fix loop) and after-fix-loop tests-green % (after the iterate-to-green cycle).
**Target build command:** `go build ./...` (and `go vet ./...`)
**Target test command:** `go test ./...`

---

## 1. Source Codebase

**Primary language:** Python {TODO — version, e.g. 3.11; check pyproject.toml / .python-version}
**Frameworks / libraries:** {TODO — list with versions; flag any with no clean Go equivalent}
**Architecture:** {TODO — observed pattern; this is MIRRORED, not changed}
**Naming conventions:** {TODO — e.g. snake_case functions, PascalCase classes}
**Scale:** {TODO — file count, estimated LOC}
**Module structure:** {TODO — package layout that the Go tree will mirror 1:1}
**Notable patterns / debt relevant to this cluster:** {TODO — duplication hotspots (from the Generator's histogram) are the most adapter-tractable modules; note them here}

---

## 2. Task Target

### Cluster A — Code Transformation
**Target language / platform:** Go 1.22 — standard library first; add a dependency only where the Python original used a non-trivial third-party library with a direct Go counterpart.
**Target architecture:** **unchanged — mirror the source package structure 1:1.** No re-layering, no consolidation, no splitting.
**Required frameworks / libraries:** {TODO — only those needed to mirror existing behaviour}
**Target conventions / style guide:** `gofmt`-clean; standard Go idioms (exported identifiers PascalCase, errors as values, `context.Context` only where the source already threaded a cancellation/timeout concept).
**Target build command:** `go build ./...` && `go vet ./...`
**Target test command:** `go test ./...`

---

## 3. Task Rules

Each rule is specific enough that two developers would apply it identically. These are language-mapping rules — the whole point of a pure port is that they are mechanical and consistent.

### Cluster A
| Rule ID | Pattern | Description |
|---|---|---|
| R01 | Class → struct + methods | Python `class` → Go `struct` for fields + methods with pointer receivers. No inheritance: model base classes as embedded structs; mixins as embedded interfaces. |
| R02 | `__init__` → constructor | `__init__` → `NewX(...) (*X, error)` returning a pointer; validation that raised in `__init__` returns an error. |
| R03 | Exceptions → errors | `try/except` → `(T, error)` returns + `if err != nil` checks. Map each caught exception type to a sentinel error or typed error; never panic for control flow. Re-raise → `fmt.Errorf("...: %w", err)`. |
| R04 | Comprehensions → loops | List/dict/set comprehensions and generator expressions → explicit `for` loops building the target slice/map. Preserve ordering. |
| R05 | dict / list / set → map / slice | `dict` → `map[K]V`; `list` → `[]T`; `set` → `map[T]struct{}`. Preserve element types precisely; no `interface{}` unless the source was genuinely heterogeneous. |
| R06 | `None` → zero value / nil | `None` → `nil` for pointers/slices/maps/interfaces, or the typed zero value otherwise. Optional scalars that distinguish "unset" → pointer types (`*int`). |
| R07 | `with` → `defer` | Context managers → acquire + `defer release()`. `__enter__/__exit__` → explicit open + `defer Close()`. |
| R08 | f-strings → fmt | f-strings / `.format()` / `%` → `fmt.Sprintf`. No interpolation syntax may survive. |
| R09 | Slicing → Go slices | `a[i:j]`, `a[:n]`, `a[-1]` → Go slice expressions / index arithmetic. **Python slice syntax must never appear in Go output.** Negative indices → `len(a)-n`. |
| R10 | Iterables → range | `for x in xs` → `for _, x := range xs`; `enumerate` → `for i, x := range xs`; `dict.items()` → `for k, v := range m`. |
| R11 | Truthiness → explicit | Python truthiness (`if x:`) → explicit checks (`if len(x) > 0`, `if x != nil`, `if x != ""`). No implicit truthiness. |
| R12 | Keyword/default args | Default args → either an options struct or distinct constructors; never silently drop. Document the choice in a `// Note:` if non-obvious. |

**Preserve exactly:**
- All observable behaviour and outputs for every public entry point (the golden I/O oracle is the arbiter).
- Public function/method signatures' semantics (names map to exported Go identifiers; shapes preserved).
- {TODO — CLI surface, on-disk formats, wire/JSON field names, DB schemas}

**Intentionally change:**
- None. (Pure port — formatting normalises to `gofmt`, but nothing is re-designed.)

**Exclude from transformation:**
| Module / pattern | Reason |
|---|---|
| C-extensions / `ctypes` / native bindings | Not a language-level port; needs a separate strategy — route to frontier/human. |
| Metaclasses, dynamic `eval`/`exec`, monkeypatching, runtime attribute injection | No mechanical Go equivalent; would force architectural decisions — out of scope for the adapter. |
| `docs/`, build/CI scripts | Not source under port. |
| {TODO — any module already slated for rebuild rather than port} | Rebuild ≠ port; do not feed to the adapter. |

**Note — test porting:** `tests/` is **in scope** for this engagement but handled as a separate Cluster C pass, not Cluster A. The Python tests are ported to Go (same framework-mapping rules as source code), then both the Python tests (against Python source) and the Go tests (against Go output) are run. Both must be green — this is the dual-run oracle. Include test-porting rules in the Cluster C manifest, not here.

**Output Correctness Rules (verified by execution, not assertion):**
- Every output compiles under `go build ./...` and passes `go vet ./...`.
- No source-language artifacts survive: no Python slices (`a[i:j]`), f-strings, `self`, `None`, `True`/`False`, `def`, `elif`, colons-as-blocks, or comprehension syntax in the Go output.
- Imports complete and used; no undefined symbols; type signatures valid in Go's type system.
- Where a golden I/O pair exists for the construct, the Go output passes it (`go test ./...` green), not merely compiles.

---

## 4. Fine-Tuning Configuration

**Target model:** {TODO — recommend starting Qwen2.5-Coder-7B-Instruct; move to 14B if first-pass green % is weak and budget allows}
**Model tier:** {7B–13B start → 14B–34B if needed}
**Alignment algorithm:** SFT
**Example count:** 500
**Output format:** both (CSV + JSONL)
**Example complexity:** complex — derived from tier (do not set manually)

**Operational complexity band (per Generator Rev 2.1):**
- Minimum construct: >= 10 lines AND at least one branch or loop. Skip pure getters/setters, one-line delegations, bare constants — the base model already ports these and they dilute the set.
- Maximum construct: ~60 lines, never exceeding the tier token cap; split larger classes into method-level examples.

### Algorithm-specific configuration
SFT first run — no DPO/GRPO config yet. (Once a strong SFT baseline exists, DPO `rejected` guidance for this pair would target: Python idioms left intact, panic-for-control-flow, `interface{}` overuse, dropped error returns.)

---

## 5. Data Governance

**Frontier model access approved:** {TODO — yes | no | conditional. Generator needs frontier access to produce the gold set; confirm before running.}
**API agreement reference:** {TODO}
**Excluded files / modules:** see §3 exclusion table.
**Other constraints:** {TODO — data residency / on-prem requirement that motivates the local adapter in the first place}

---

## 6. Iteration History

| Version | Date | Algorithm | Summary of changes | Trigger |
|---|---|---|---|---|
| 1 | 2026-06-10 | SFT | Initial manifest — Python→Go pure port | First run |

---

## 7. Review Sign-Off

| Reviewer | Role | Status | Date |
|---|---|---|---|
| Kenneth Stott | WTE Consultant | Pending | |
| | Client Architect | Pending | |
