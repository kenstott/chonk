# Modernization Manifest — Cluster A: Code Transformation
**Version:** 4
**Generated:** 2026-06-10
**Engagement:** chonk — RAG pipeline library Python→Go port
**Cluster:** A — Code Transformation
**Status:** Draft — pending review

---

## 0. Objective & Validation

**Objective:** Fine-tune a local code-transformation adapter that ports Python constructs to idiomatic Go 1.22, one-to-one, preserving behaviour — so bulk conversion runs cheaply on-prem instead of against a frontier model.
**Scope mode:** pure language port
**In scope:** language-level translation of functions, classes, control flow, error handling, data structures, standard-library idioms — across all 114 source files in `chonk/`.
**Out of scope:** architecture (package/module boundaries mirror the source 1:1), business logic, public contracts, data schemas. Nothing is re-designed.
**Bulk target:** whole `chonk/` package — 114 source files, ~24,616 LOC
**Validation method:** `existing tests` — port the Python pytest suite (45 test files) to Go alongside the source code. Run both the Python tests against the Python source (establishes the baseline) and the Go tests against the Go output. Both must be green. Dual-run confirms behavioural equivalence and catches test-porting errors independently of source-porting errors.
**Success metric:** report **two numbers** — first-pass tests-green % (Go output before any fix loop) and after-fix-loop tests-green % (after the iterate-to-green cycle).
**Target build command:** `go build ./...` (and `go vet ./...`)
**Target test command:** `go test ./...`

---

## 1. Source Codebase

**Primary language:** Python 3.11 (requires-python ≥3.11,<3.15)
**Frameworks / libraries:** sentence-transformers ≥5.5.1, torch ≥2.12, DuckDB ≥1.5.3, SQLAlchemy ≥2.0.50, duckdb-engine ≥0.17, numpy ≥2.4.6, scikit-learn ≥1.9, igraph ≥0.11.9, leidenalg ≥0.10.2, pyarrow ≥24.0, spacy ≥3.8.14, pandas ≥2.3.3, psycopg2-binary ≥2.9.12, pgvector ≥0.4.2, inflect ≥7.5, requests ≥2.34.2, boto3 ≥1.43.23, paramiko ≥3.5.1, pypdf ≥6.12.2, python-docx ≥1.2, openpyxl ≥3.1.5, python-pptx ≥1.0.2, pyyaml ≥6.0.3
**Architecture:** modular RAG pipeline library — chunking → ingestion → indexing → search, with pluggable transports, storage backends, NER, and community/graph layers
**Naming conventions:** snake_case functions and variables; PascalCase classes; `_private` prefix for internal modules and symbols; `__init__.py` re-exports for public surface
**Scale:** 114 source files, ~24,616 LOC
**Module structure:**
- `chonk/` — top-level: chunking.py, context.py, indexer.py, ingest.py, lifecycle.py, loader.py, models.py, schema.py, _ingest_worker.py, _struct_inference.py, _versioning.py
- `chonk/cluster/` — _clusterer.py, _cooccurrence.py, _map.py
- `chonk/community/` — community detection
- `chonk/extractors/` — document field extractors
- `chonk/generation/` — LLM generation helpers
- `chonk/graph/` — graph construction
- `chonk/ner/` — _build.py, _index.py, _merge.py, _normalizer.py, _pipeline.py, _schema.py, _schema_vocab.py, _spacy.py, _spacy_labels.py, _vocabulary.py
- `chonk/search/` — vector search and retrieval
- `chonk/storage/` — DuckDB/SQLAlchemy backends
- `chonk/transports/` — S3, SFTP, IMAP, Cosmos, DynamoDB, etc.

**Notable patterns / debt relevant to this cluster:** heavy use of Python dataclasses and TypedDicts (map well to Go structs); 288 raw SQL occurrences in storage/ and transports/ via SQLAlchemy ORM (require go-duckdb / pgx mapping); several ML library calls (sentence-transformers, torch, spacy) with no direct Go stdlib equivalent — these are the highest-risk port targets and are flagged for cgo wrapper or frontier/human review.

---

## 2. Task Target

### Cluster A — Code Transformation
**Target language / platform:** Go 1.22 — standard library first; add a dependency only where the Python original used a non-trivial third-party library with a direct Go counterpart
**Target architecture:** unchanged — mirror the source package structure 1:1. No re-layering, no consolidation, no splitting.
**Required frameworks / libraries:** go-duckdb (DuckDB driver); pgx v5 + pgvector-go (pgvector); ML embeddings via cgo wrapper or pure-Go equivalent (TBD — flag for frontier/human on first pass); standard library for HTTP, JSON, I/O, sync
**Target conventions / style guide:** `gofmt`-clean; exported identifiers PascalCase; errors as values (`(T, error)` returns, `if err != nil`); `context.Context` only where the Python source already threads a cancellation/timeout concept; no generics where plain interfaces suffice
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
| R06 | `None` → zero value / nil | `None` → `nil` for pointers/slices/maps/interfaces, or the typed zero value otherwise. Optional scalars that distinguish "unset" → pointer types (`*int`, `*string`). |
| R07 | `with` → `defer` | Context managers → acquire + `defer release()`. `__enter__/__exit__` → explicit open + `defer Close()`. |
| R08 | f-strings → fmt | f-strings / `.format()` / `%` → `fmt.Sprintf`. No interpolation syntax may survive. |
| R09 | Slicing → Go slices | `a[i:j]`, `a[:n]`, `a[-1]` → Go slice expressions / index arithmetic. **Python slice syntax must never appear in Go output.** Negative indices → `len(a)-n`. |
| R10 | Iterables → range | `for x in xs` → `for _, x := range xs`; `enumerate` → `for i, x := range xs`; `dict.items()` → `for k, v := range m`. |
| R11 | Truthiness → explicit | Python truthiness (`if x:`) → explicit checks (`if len(x) > 0`, `if x != nil`, `if x != ""`). No implicit truthiness. |
| R12 | Keyword/default args | Default args → either an options struct or distinct constructors; never silently drop. Document the choice in a `// Note:` if non-obvious. When a parameter has a valid zero value (e.g. `confidence=0.0`, `timeout=0`), use a pointer type (`*float64`, `*int`) rather than treating `0` as "unset" — the zero-value sentinel pattern silently overrides legitimate caller-supplied zeros. |
| R13 | Python regex → RE2 | Go's `regexp` package uses RE2, which does not support lookbehind assertions (`(?<=...)`, `(?<!...)`), backreferences (`\1`–`\9`), or possessive quantifiers. For each such Python pattern: produce a behaviourally equivalent RE2 alternative where one exists; otherwise emit a `// TODO: RE2 does not support lookbehind — manual port required` comment. Never silently drop the regex behaviour or substitute a non-equivalent pattern that changes results. |

**Preserve exactly:**
- All observable behaviour and outputs for every public entry point (the dual-run oracle is the arbiter)
- Public function/method signature semantics (names map to exported Go identifiers; shapes preserved)
- On-disk formats (DuckDB schema, Parquet layouts, on-disk JSON)
- JSON/wire field names (no field name changes — match Python's serialised output exactly)
- DB schemas (DuckDB table definitions and column types)
- CLI surface (if any), file path conventions, environment variable names

**Intentionally change:**
- None. (Pure port — formatting normalises to `gofmt`, but nothing is re-designed.)

**Exclude from transformation:**
| Module / pattern | Reason |
|---|---|
| C-extensions / `ctypes` / native bindings | Not a language-level port; needs a separate strategy — route to frontier/human. |
| Metaclasses, dynamic `eval`/`exec`, monkeypatching, runtime attribute injection | No mechanical Go equivalent; would force architectural decisions — out of scope for the adapter. |
| ML library calls (sentence-transformers, torch, spacy inference) | No direct Go stdlib or community equivalent — flag for cgo wrapper or frontier/human review. |
| `docs/` | Not source under port. |
| `generate-training/` | Not source under port. |
| `training/` | Not source under port. |
| Build/CI scripts | Not source under port. |

**Note — test porting:** `tests/` is **in scope** for this engagement but handled as a separate Cluster C pass, not Cluster A. See `modernization_manifest_C.md`.

**Output Correctness Rules (verified by execution, not assertion):**
- Every output compiles under `go build ./...` and passes `go vet ./...`.
- No source-language artifacts survive: no Python slices (`a[i:j]`), f-strings (f"..."), `self`, `None`, `True`/`False`, `def`, `elif`, colon-terminated blocks, or comprehension syntax (`[x for x in ...]`) in the Go output.
- Imports complete and used; no undefined symbols; type signatures valid in Go's type system.
- Where a golden I/O pair exists for the construct, the Go output passes it (`go test ./...` green), not merely compiles.

**Completeness rule:** When the instruction covers N methods or constructs, the output must contain all N. Partial ports (M < N methods ported) must not be written as examples — regenerate until complete. A `// TODO: port X` comment does not satisfy this requirement.

**No silent value drops:** `_ = variable` in Go output is always wrong when the discarded value was computed from a Python assignment that was stored or returned. Every field write, DB column, return value, and struct initialisation in the Python source must have a corresponding write in the Go output.

---

## 4. Fine-Tuning Configuration

**Target model:** `Qwen2.5-Coder-14B-Instruct`
**Model tier:** 14B–34B
**Alignment algorithm:** SFT
**Example count:** 500
**Output format:** both (CSV + JSONL)
**Example complexity:** complex — derived from tier (do not set manually)

**Operational complexity band (per Generator Rev 2.1, amended v2):**

The complexity floor is a **hard gate**, not a guideline. Reject any candidate construct that does not pass both conditions before generating the example:

1. **Line floor:** the Python source block must be **≥ 10 lines** (blank lines and comments count; decorator lines count).
2. **Branch/loop requirement (raised v4):** the Python source block must satisfy **at least one** of:
   - **Option A:** ≥2 distinct statement-level `if`/`elif`/`else`, `for`, `while`, or `try`/`except` constructs within the selected block, OR
   - **Option B:** ≥1 `for` or `while` loop with a body of ≥5 lines within the selected block.
   A single bare `if err is not None: return` or a single `try/except` with a one-line body does **not** satisfy this requirement. Count by scanning the selected lines only — not in called helper functions, not transitively reachable.

**What COUNTS as a branch/loop (at statement level):**
| Pattern | Counts? |
|---|---|
| `if`/`elif`/`else` with a body of ≥2 lines | YES |
| `for` loop with a body of ≥2 lines | YES |
| `while` loop | YES |
| `try`/`except` block with ≥2-line body in each branch | YES |
| Nested `if` inside an `if` body | YES (each counts independently) |

**What does NOT count:**
| Pattern | Why |
|---|---|
| List/dict/set comprehension as the ONLY loop: `[x for x in ...]` | Inline expression, not a statement-level loop |
| Generator in a function call: `sum(x for x in ...)` | Not a statement-level loop |
| Ternary expression: `a if cond else b` | Not a statement-level branch |
| `try`/`except` where every branch is a single assignment | Too trivial — not meaningful branching |
| `@property` decorated getter/setter regardless of line count | Structural, not algorithmic |
| Method body = one `return` statement + field reads, even if 15 lines | No branching, decorator lines pad the count |
| `__init__` with only field assignments (e.g. `self.x = x`) even if 12+ fields | Pure assignment — zero branching |
| Method that only calls `super().method(*args)` | One-line delegation |
| All lines are dataclass field declarations (`x: int = 0`) | Struct definition, no computation |
| Single list comprehension spanning multiple lines (line-wrapped) | Still one expression |

Constructs that fail either condition must be skipped or combined with a related sibling method to form a compound example that clears both gates. Do not write sub-floor examples to the batch file.

- Maximum construct: ~60 lines input, never exceeding the tier token cap; split larger classes into method-level examples.

### Generator Quality Gates (added v2)

Apply all four gates per-example before writing to the batch file. An example that fails any gate must be discarded and regenerated from a different source construct.

**Gate 1 — Input purity (M5 prevention):**
Scan the `input` field for Go syntax markers: `fmt.`, `:=`, `func`, `go`, `chan`, `append(`, `make(`, `var`, `strings.`, `return nil`, `return err`. Any match inside what should be a Python source block indicates the Generator read a partially-ported file. Reject the example and read the source from the unmodified Python file instead.

**Gate 2 — Rule-citation accuracy (M4b prevention):**
Each transformation rule cited in the `instruction` field (R01–R13) must have at least one visible instance in the `input` Python code. Use the table below to verify each citation. Remove or replace any cited rule that has no corresponding input pattern — do not cite a rule "in case it applies".

| Rule | Required input pattern (at least one must be present) |
|---|---|
| R01 | `class ` declaration |
| R02 | `def __init__` |
| R03 | `try:` or `except ` or `raise ` |
| R04 | `[x for`, `{k:` or `{k for`, `(x for` — comprehension or generator |
| R05 | `{}` dict literal, `[]` list literal, or `set()` / `{x,` set literal |
| R06 | `= None` or `is None` or `if x is None` |
| R07 | `with ` context manager |
| R08 | `f"` or `f'` or `.format(` or `% (` or `%s` |
| R09 | `[i:j]` or `[-` or `[:n]` or `[n:]` — Python slice syntax |
| R10 | `for x in ` or `enumerate(` or `.items()` or `.values()` or `.keys()` |
| R11 | `if x:` or `if not x:` or `if xs:` — bare truthiness check |
| R12 | `def method(self, x=default` — parameter with default value |
| R13 | `(?<=` or `(?<!` or `\1` through `\9` — lookbehind or backreference in a regex string |

**Gate 3 — Output completeness (M4 prevention):**
Count the number of methods, functions, or constructs named in the instruction. The output must contain the same count. If the instruction says "port these three methods", the output must contain all three. Use `grep`/count to verify before committing.

**Gate 4 — Deduplication (cross-run registry):**
Before sampling any construct, read `training/gen_registry_v4.txt` (one `ClassName.method_name` per line; create if absent). If the normalised signature of the candidate construct is already in the file, skip it and pick a different construct. After writing each example to the batch file, immediately append the construct's normalised signature to `training/gen_registry_v4.txt`. The registry is shared across all batch agents in the same run — do not maintain only an in-memory per-batch set. At the start of a new Generator run, delete and recreate the registry file; do not carry over signatures from prior runs.

**Gate 5 — Multi-rule floor for high-failure modules (added v3, reinforced v4):**
For examples drawn from `chonk/cluster/`, `chonk/graph/`, `chonk/ner/`, `chonk/extractors/`, or `chonk/transports/`, the selected construct must trigger at least **2 distinct rules** (R01–R13) that are visibly present in the input. Single-rule examples from these modules are structurally too trivial for the 14B model tier and consistently fail D5 (complexity calibration). If a candidate construct only triggers one rule, combine it with an adjacent sibling method (sharing the same class or logical group) to form a compound example that triggers ≥2 rules. Do not write single-rule examples from these modules.

**Additional Gate 5 requirement for `chonk/cluster/` (added v4):** The construct must contain at least one `for` or `while` loop that iterates over a graph structure (nodes, edges, adjacency, membership arrays). Error-only constructs (`if x is None / raise`) without any collection iteration do not qualify regardless of line count. If no method in `_clusterer.py`, `_cooccurrence.py`, or `_map.py` meets this, combine two methods that together form a loop+branch compound.

**Gate 6 — Module-specific construct targeting (added v3):**
Apply the per-module guidance below when selecting constructs. This guidance supplements, not replaces, the hard complexity gate.

| Module | Prefer (high yield) | Avoid (consistently fail D5) |
|---|---|---|
| `chonk/cluster/` | `_clusterer.py`: `fit()`, `_compute_distance_matrix()`, `_assign_clusters()`; `_cooccurrence.py`: matrix computation loops; `_map.py`: mapping loops. **Must include ≥1 for/while loop iterating over nodes, edges, a distance matrix row, or a membership array.** | `__init__` field assignments; property getters; any method whose body is a single numpy/sklearn call; any method with only error checks and no collection iteration |
| `chonk/graph/` | `_builder.py`: edge-building loops, adjacency construction, graph traversal methods | Dataclass field declarations; one-line delegation wrappers; methods that only call `igraph` with remapped args |
| `chonk/ner/` | `_pipeline.py` stage dispatch methods with ≥1 for loop over tokens/spans; `_merge.py` conflict resolution; `_normalizer.py` rule-application loops; `_build.py` index construction loops | Schema field definitions (`_schema.py`, `_schema_vocab.py`); `_vocabulary.py` lookup one-liners; `_spacy_labels.py` constants; simple string-only methods without iteration |
| `chonk/extractors/` | Methods with try/except chains + field extraction conditionals; multi-format dispatch methods | Pure field mappers with one assignment per field; pass-through wrappers |
| `chonk/transports/` | S3/SFTP/IMAP connection methods with error-propagation chains; retry loops; credential resolution with conditional fallbacks | Single-line send/receive wrappers; config field constructors |
| `chonk/` main-cont (`_ingest_worker.py`, `ingest.py`) | Batch processing loops (`for chunk in`), error-recovery paths, conditional dispatch on content type | Field-only `__init__`; single-assignment methods |
| `chonk/generation/` | Methods with prompt construction loops, conditional model dispatch, retry-on-error chains | One-line LLM call wrappers; config accessors |

### Algorithm-specific configuration
SFT first run — no DPO/GRPO config yet. (Once a strong SFT baseline exists, DPO `rejected` guidance for this pair would target: Python idioms left intact, panic-for-control-flow, `interface{}` overuse, dropped error returns, Python slice syntax surviving into Go output.)

---

## 5. Data Governance

**Frontier model access approved:** yes
**API agreement reference:** enterprise API agreement in place
**Excluded files / modules:** see §3 exclusion table
**Other constraints:** none recorded

---

## 6. Iteration History

| Version | Date | Algorithm | Summary of changes | Trigger |
| --- | --- | --- | --- | --- |
| 1 | 2026-06-10 | SFT | Initial manifest — Python→Go pure port, whole chonk/ package | First run |
| 2 | 2026-06-10 | SFT | Added R13 (RE2 compatibility); R12 amended (pointer types for zero-value-ambiguous defaults); complexity floor made hard gate with explicit branch/loop check; added Generator Quality Gates §4 (input purity, rule-citation accuracy, output completeness, deduplication); added completeness rule and no-silent-drops rule to §3 Output Correctness | Judge v1 pass rate 49.4% — D5 systemic at 32.6%, M5 contaminated inputs 7% |
| 3 | 2026-06-11 | SFT | Complexity gate: added explicit "counts / does not count" table for branch/loop check (inline vs statement-level distinction); Gate 2 strengthened with per-rule citation pattern table (R01–R13); Gate 4 upgraded from per-batch in-memory set to cross-run persistent registry file (`gen_registry_v3.txt`); Gate 5 added — multi-rule floor (≥2 distinct rules) for cluster/, graph/, ner/, extractors/, transports/ modules; Gate 6 added — per-module construct targeting table for the 7 systemic-failure modules | Judge v2 pass rate 61.0% — 7/11 modules systemic (cluster 56%, graph 52%, ner 49%), M6 dominant at 18.6%, M7 5.8% cross-batch |
| 4 | 2026-06-11 | SFT | Raised complexity floor: ≥2 statement-level branches/loops OR ≥1 for/while with ≥5-line body (up from ≥1 branch); Gate 4 registry updated to `gen_registry_v4.txt`; Gate 5 cluster reinforced: requires ≥1 for/while over graph structure; Gate 6 cluster and ner rows updated with explicit iteration requirement; replaced 85 failed examples from v4 judge run | Judge v3 pass rate 48.2% was regression caused by verdict computation bug (any-2=Review) and D5/D6 over-scoring; v4 judge criteria fixed (one-2=Accept, recalibrated D5/D6) → 83.0% pass rate, cluster (36%) only systemic module |

---

## 7. Review Sign-Off

| Reviewer | Role | Status | Date |
|---|---|---|---|
| Kenneth Stott | WTE Consultant | Pending | |
| | Client Architect | Pending | |
