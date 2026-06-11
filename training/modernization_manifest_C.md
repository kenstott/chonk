# Modernization Manifest — Cluster C: Test Generation
**Version:** 1
**Generated:** 2026-06-10
**Engagement:** chonk — RAG pipeline library Python→Go port
**Cluster:** C — Test Generation
**Status:** Draft — pending review

---

## 0. Objective & Validation

**Objective:** Fine-tune a local test-generation adapter that ports the existing Python pytest suite to idiomatic Go tests (standard testing + testify/assert), enabling the dual-run oracle for the Cluster A port — Go tests against Go output must be green, Python tests against Python source must remain green.
**Scope mode:** pure language port (test porting mirrors Cluster A — no new test logic invented, existing assertions translated)
**In scope:** porting Python test functions (pytest) to Go test functions (testing + testify/assert); generating additional Go tests for modules prioritised by coverage gap.
**Out of scope:** architecture (test file structure mirrors the source pytest layout 1:1); adding new test scenarios beyond what the Python suite already covers (that is a separate test-authoring task).
**Bulk target:** `tests/` — 45 test files; priority modules: `chonk/storage/`, `chonk/search/`, `chonk/chunking.py` + `chonk/context.py`, `chonk/ner/`, `chonk/cluster/`
**Validation method:** `existing tests` — the Python pytest suite is the oracle. Port each test file alongside the source. Dual-run: `pytest tests/` against Python source (green baseline), then `go test ./...` against Go output (must also be green). Both suites must pass independently.
**Success metric:** report **two numbers** — first-pass Go tests-green % (before any fix loop) and after-fix-loop tests-green %.
**Target build command:** `go build ./...` && `go vet ./...`
**Target test command:** `go test ./...`

---

## 1. Source Codebase

**Primary language:** Python 3.11 (requires-python ≥3.11,<3.15)
**Frameworks / libraries:** pytest ≥9.0.3, pytest-asyncio ≥1.4, pytest-cov ≥7.1, pytest-timeout ≥2.4, pytest-docker ≥3.2.5, respx ≥0.23.1, unittest.mock (stdlib)
**Architecture:** test files in `tests/` mirroring the `chonk/` module structure; fixtures in conftest.py files; async tests use `@pytest.mark.asyncio`
**Naming conventions:** test functions `test_*`; test files `test_*.py`; fixtures snake_case
**Scale:** 45 test files
**Module structure:** mirrors `chonk/` package layout — one test file per source module (where coverage exists)
**Notable patterns / debt relevant to this cluster:** pytest fixtures are the primary dependency-injection mechanism (map to Go `TestMain` or helper constructors); async tests with `@pytest.mark.asyncio` require `go test` goroutine equivalents; `respx` HTTP mocking requires Go `httptest` equivalents; `pytest-docker` integration tests require Go `testcontainers` or manual Docker setup

---

## 2. Task Target

### Cluster C — Test Generation
**Target test framework:** Go standard `testing` package + `testify/assert` v1 (github.com/stretchr/testify)
**Test types required:** unit (ported from pytest unit tests); integration (ported from pytest-docker tests, using `testcontainers-go` or manual Docker); edge cases (preserve all boundary assertions from the Python suite)
**Coverage requirements:** all public methods covered by the Python suite must be covered by the Go suite; happy path + all error cases present in Python tests must be preserved; boundary conditions (nil inputs, empty slices, zero values) must be explicitly tested
**Test naming convention:** `Test{FunctionName}_{Condition}` (e.g. `TestChunker_EmptyInput`, `TestStorage_InsertDuplicate`)
**Mocking framework:** `net/http/httptest` for HTTP mocking (replaces `respx`); `testify/mock` for interface mocking (replaces `unittest.mock`); `testcontainers-go` for Docker-dependent integration tests (replaces `pytest-docker`)

---

## 3. Task Rules

### Cluster C
| Rule ID | Test type | Coverage requirement | Framework-specific notes |
|---|---|---|---|
| R01 | pytest function → Go test function | One Python `test_*` function → one Go `Test*` function in the same package. Preserve all assertion logic. | `assert x == y` → `assert.Equal(t, y, x)` (testify); `assert x is None` → `assert.Nil(t, x)` |
| R02 | pytest fixture → Go helper / TestMain | `@pytest.fixture` → Go helper constructor or `TestMain(m *testing.M)` for suite-wide setup/teardown. Scoped fixtures (module/session) → `TestMain`; function-scoped → in-test setup. | Do not use global mutable state between tests. |
| R03 | `@pytest.mark.asyncio` → goroutine | Async pytest tests → synchronous Go tests that invoke the function and block with a channel or `sync.WaitGroup`; or use `context.Background()` with a timeout. | No external async test framework needed in Go. |
| R04 | `pytest.raises` → assert error | `with pytest.raises(SomeError):` → call function, capture `(_, err)`, `assert.ErrorIs(t, err, expectedErr)` or `assert.Error(t, err)`. | Typed errors → `assert.ErrorAs`. |
| R05 | `respx` HTTP mock → `httptest.Server` | HTTP mock via `respx` → `httptest.NewServer(handler)` returning a test URL; inject URL into the component under test. | Close the server with `defer ts.Close()`. |
| R06 | `unittest.mock.patch` → testify/mock | `mock.patch` on an interface → `testify/mock` mock struct implementing the interface; inject via constructor. | Never mock concrete structs — extract interface first if needed. |
| R07 | `pytest-docker` → testcontainers-go | Docker-dependent tests → `testcontainers-go` container setup in `TestMain` or per-test setup. | Tag integration tests with `//go:build integration` build tag so `go test ./...` skips them by default; `go test -tags integration ./...` to run them. |
| R08 | Parametrize → table-driven | `@pytest.mark.parametrize` → Go table-driven test with `[]struct{ name, input, expected }` and `t.Run(tc.name, ...)`. | Use subtests so each case is independently reported. |
| R09 | Priority modules — coverage gap | Generate additional Go tests for: `chonk/storage/` (DuckDB/SQLAlchemy layer), `chonk/search/` (vector retrieval), `chonk/chunking.py` + `chonk/context.py` (chunking pipeline), `chonk/ner/` + `chonk/cluster/` (NER and clustering). | For these modules, generate tests even where no Python test exists — happy path + at least one error case per public function. |

**Preserve exactly:**
- All assertion logic from the Python test suite — every `assert`/`assertEqual`/`pytest.raises` maps to a Go equivalent
- Test isolation — no shared mutable state between test cases
- Integration test scope (Docker-dependent tests remain integration; unit tests remain unit)

**Intentionally change:**
- Test file extension: `.py` → `_test.go`
- Test naming: `test_*` → `Test*`
- Assertion style: pytest assert → testify/assert
- Async pattern: `@pytest.mark.asyncio` + `await` → synchronous Go with `context.Background()` + timeout

**Exclude from transformation:**
| Module / pattern | Reason |
|---|---|
| `generate-training/` | Not test source. |
| `training/` | Not test source. |
| `docs/` | Not test source. |

---

## 4. Fine-Tuning Configuration

**Target model:** `Qwen2.5-Coder-14B-Instruct`
**Model tier:** 14B–34B
**Alignment algorithm:** SFT
**Example count:** 500
**Output format:** both (CSV + JSONL)
**Example complexity:** complex — derived from tier (do not set manually)

**Operational complexity band (per Generator Rev 2.1):**
- Minimum construct: >= 10 lines AND at least one assertion beyond a trivial equality check. Skip single-assertion smoke tests — the base model already handles these and they dilute the set.
- Maximum construct: ~60 lines (one test function or one table-driven test block); split larger test files into per-function examples.

### Algorithm-specific configuration
SFT first run — no DPO/GRPO config yet. (Once a strong SFT baseline exists, DPO `rejected` guidance for this pair would target: pytest syntax surviving in Go output, missing error-path assertions, shared mutable state between tests, missing `defer ts.Close()` on test servers.)

---

## 5. Data Governance

**Frontier model access approved:** yes
**API agreement reference:** enterprise API agreement in place
**Excluded files / modules:** see §3 exclusion table
**Other constraints:** none recorded

---

## 6. Iteration History

| Version | Date | Algorithm | Summary of changes | Trigger |
|---|---|---|---|---|
| 1 | 2026-06-10 | SFT | Initial manifest — Python pytest→Go testing+testify port | First run |

---

## 7. Review Sign-Off

| Reviewer | Role | Status | Date |
|---|---|---|---|
| Kenneth Stott | WTE Consultant | Pending | |
| | Client Architect | Pending | |
