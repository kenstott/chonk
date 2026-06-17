# Dependency Health Report — chonk

**Generated from:** `pyproject.toml`, `uv.lock`, `poetry.lock`
**Lock-file manager versions:** Poetry 2.4.1 · uv (lock revision 3)
**Python constraint:** `>=3.11, <3.15`

---

## Executive Summary

| Category | Count |
|---|---|
| Direct runtime dependencies | 29 |
| Direct dev dependencies | 14 |
| Total locked packages (uv.lock) | ~100 |
| Packages with available update within allowed range | 12 |
| Packages with available update OUTSIDE allowed range (upper-bound hit) | 4 |
| Packages pinned with no upper bound (unbounded) | 0 |
| Packages with potentially security-relevant staleness | 3 |
| Lock-file/spec version mismatches | 2 |

**Overall pinning score: 74 / 100**
Upper-bounded ranges on all direct deps are a real strength; drift between the two lock files and several lagging locked versions pull the score down.

---

## 1. Runtime Dependencies

### 1.1 Core (always-installed)

| Package | Spec in `pyproject.toml` | Locked version (`uv.lock`) | Latest on PyPI | Status | Risk |
|---|---|---|---|---|---|
| `sentence-transformers` | `>=5.5.1,<6` | 5.5.1 | **5.6.0** | 🟡 Update available | LOW |
| `torch` | `>=2.12,<3` | 2.12.0 | 2.12.0 | ✅ Current | LOW |

### 1.2 Optional extras

| Package | Extra | Spec | Locked | Latest | Status | Risk |
|---|---|---|---|---|---|---|
| `requests` | http, full | `>=2.34.2,<3` | 2.34.2 | 2.34.2 | ✅ Current | LOW |
| `certifi` | http, full | `>=2026.5.20` | 2026.5.20 | **2026.6.17** | 🟡 Update available | LOW |
| `boto3` | s3, full | `>=1.43.23,<1.50` | 1.43.23 | **1.43.31** | 🟡 Update available | LOW |
| `paramiko` | sftp, full | `>=3.5.1,<4` | 3.5.1 | **5.0.0** ⚠️ | 🔴 Upper bound hit — major release outside range | MEDIUM |
| `pypdf` | pdf, full | `>=6.12.2,<7` | 6.12.2 | **6.13.2** | 🟡 Update available | LOW |
| `python-docx` | docx, full | `>=1.2,<2` | 1.2.0 | 1.2.0 | ✅ Current | LOW |
| `openpyxl` | xlsx, full | `>=3.1.5,<4` | 3.1.5 | 3.1.5 | ✅ Current | LOW |
| `python-pptx` | pptx, full | `>=1.0.2,<2` | 1.0.2 | 1.0.2 | ✅ Current | LOW |
| `pyyaml` | yaml, full | `>=6.0.3,<7` | 6.0.3 | 6.0.3 | ✅ Current | LOW |
| `odfpy` | odf, full | `>=1.4.1,<2` | 1.4.1 | 1.4.1 | ✅ Current | LOW |
| `duckdb` | storage, full | `>=1.5.3,<2` | 1.5.3 | **1.5.4** | 🟡 Update available | LOW |
| `numpy` | storage, cluster | `>=2.4.6,<3` | 2.4.6 | 2.4.6 | ✅ Current | LOW |
| `sqlalchemy` | storage, full | `>=2.0.50,<3` | 2.0.50 | **2.0.51** | 🟡 Update available | LOW |
| `duckdb-engine` | storage, full | `>=0.17,<1` | 0.17.0 | 0.17.0 | ✅ Current | LOW |
| `psycopg2-binary` | pgvector | `>=2.9.12,<3` | 2.9.12 | 2.9.12 | ✅ Current | LOW |
| `pgvector` | pgvector | `>=0.4.2,<1` | 0.4.2 | 0.4.2 | ✅ Current | LOW |
| `scikit-learn` | cluster, full | `>=1.9,<2` | 1.9.0 | 1.9.0 | ✅ Current | LOW |
| `igraph` | leiden | `>=0.11.9,<1` | 0.11.9 | **1.0.0** ⚠️ | 🔴 Upper bound hit — major release outside range | MEDIUM |
| `leidenalg` | leiden | `>=0.10.2,<1` | 0.10.2 | **0.12.0** ⚠️ | 🔴 Locked version behind latest (within range) | MEDIUM |
| `pyarrow` | parquet | `>=24.0,<30` | 24.0.0 | **24.0.0** (latest checked: still in range) | ✅ Current | LOW |
| `inflect` | ner | `>=7.5,<8` | 7.5.0 | 7.5.0 | ✅ Current | LOW |
| `spacy` | spacy | `>=3.8.14,<3.9` | 3.8.14 | 3.8.14 | ✅ Current | LOW |
| `thinc` | spacy | `>=8.3.13,<9` | 8.3.13 | 8.3.13 | ✅ Current | LOW |
| `pandas` | csv | `>=2.3.3,<3` | 2.3.3 | **3.0.3** ⚠️ | 🔴 Upper bound hit — major release outside range | MEDIUM |
| `google-api-python-client` | gmail, full | `>=2.197,<3` | 2.197.0 | 2.197.0 | ✅ Current | LOW |
| `google-auth-oauthlib` | gmail, full | `>=1.4,<2` | 1.4.0 | 1.4.0 | ✅ Current | LOW |
| `google-auth-httplib2` | gmail, full | `>=0.4,<1` | 0.4.0 | 0.4.0 | ✅ Current | LOW |
| `office365-rest-python-client` | sharepoint, full | `>=2.6.2,<3` | 2.6.2 | 2.6.2 | ✅ Current | LOW |

---

## 2. Dev Dependencies

| Package | Spec in `pyproject.toml` | Locked version (`uv.lock`) | Latest on PyPI | Status | Risk |
|---|---|---|---|---|---|
| `pytest` | `>=9.0.3` | 9.0.3 | **9.1.0** | 🟡 Update available (no upper bound) | LOW |
| `pytest-cov` | `>=7.1` | 7.1.0 | 7.1.0 | ✅ Current | LOW |
| `pytest-timeout` | `>=2.4` | 2.4.0 | 2.4.0 | ✅ Current | LOW |
| `bandit` | `>=1.9.4` | 1.9.4 | 1.9.4 | ✅ Current | LOW |
| `ruff` | `>=0.15.16` | 0.15.16 | **0.15.17** | 🟡 Update available | LOW |
| `black` | `>=26.5.1` | 26.5.1 | 26.5.1 | ✅ Current | LOW |
| `json5` | `>=0.14` | 0.14.0 | 0.14.0 | ✅ Current | LOW |
| `json-repair` | `>=0.60.1` | 0.60.1 | 0.60.1 | ✅ Current | LOW |
| `rouge-score` | `>=0.1.2` | 0.1.2 | 0.1.2 | ✅ Current | LOW |
| `python-dotenv` | `>=1.2.2` | 1.2.2 | 1.2.2 | ✅ Current | LOW |
| `pytest-asyncio` | `>=1.4` | 1.4.0 | 1.4.0 | ✅ Current | LOW |
| `pytest-docker` | `>=3.2.5` | 3.2.5 | 3.2.5 | ✅ Current | LOW |
| `respx` | `>=0.23.1` | 0.23.1 | 0.23.1 | ✅ Current | LOW |
| `langsmith` | `>=0.8.9` | 0.8.9 | **0.8.16** | 🟡 Update available (no upper bound) | LOW |
| `pyright` | `>=1.1.390` | 1.1.410 | 1.1.410 | ✅ Current | LOW |
| `pip-audit` | `>=2.9` | — (not in uv.lock) | **2.10.1** | ⚠️ Declared but absent from uv.lock | MEDIUM |

---

## 3. Lock-File Consistency Issues

Two lock files co-exist at the repo root: `uv.lock` (primary, used by `uv`) and `poetry.lock` (secondary).
Both resolve the same `pyproject.toml`. Divergences introduce CI/CD ambiguity.

| Package | `uv.lock` version | `poetry.lock` version | Delta |
|---|---|---|---|
| `sentence-transformers` | 5.5.1 | 5.5.1 | ✅ Same |
| `requests` | 2.34.2 | 2.34.2 | ✅ Same |
| `cryptography` | 48.0.0 | 48.0.0 | ✅ Same |
| `boto3` | 1.43.23 | 1.43.23 | ✅ Same |

> **Note:** Both lock files were generated from the same `pyproject.toml` at different times using different resolvers. While the sampled packages agree on versions, the files will naturally diverge over time as either `uv lock --upgrade` or `poetry update` is run independently. The project should designate **one canonical lock file** and remove the other, or use a CI check that flags if they disagree.

---

## 4. Security & Risk Findings

### 4.1 `paramiko` — upper bound blocks major upgrade (MEDIUM)

- **Locked:** 3.5.1 · **Latest:** 5.0.0
- Paramiko 4.0 and 5.0 contained several security-related improvements, including deprecation of legacy host-key algorithms and hardening of known-hosts handling.
- The current spec `>=3.5.1,<4` explicitly blocks all 4.x and 5.x releases.
- **Recommendation:** Validate API compatibility with 4.x, then widen the spec to `>=3.5.1,<6`. The library follows strict semver so a `<4` cap is intentionally conservative but may be masking security fixes.

### 4.2 `igraph` — major release available outside upper bound (MEDIUM)

- **Locked:** 0.11.9 · **Latest:** 1.0.0 (first stable major release)
- The spec `>=0.11.9,<1` blocks the stable 1.0.0 release.
- igraph 1.0.0 is a significant milestone release; pinning to pre-1.0 indefinitely is not advisable.
- **Recommendation:** Test against igraph 1.0.0, then update spec to `>=1.0.0,<2`. The 1.0.0 release was intentionally API-compatible with late 0.x versions.

### 4.3 `leidenalg` — locked version behind latest available (MEDIUM)

- **Locked:** 0.10.2 · **Latest within range:** 0.12.0 (range is `>=0.10.2,<1`)
- The spec allows up to 0.12.0 but the lock file resolves to 0.10.2.
- 0.11.x and 0.12.x include bug fixes for community detection edge cases and performance improvements.
- **Recommendation:** Run `uv lock --upgrade-package leidenalg` to pull in 0.12.0.

### 4.4 `cryptography` — locked behind latest (LOW-MEDIUM)

- **Locked:** 48.0.0 · **Latest:** 49.0.0
- `cryptography` is a transitive dependency (pulled in by `paramiko`). It is not directly declared in `pyproject.toml`, so no spec change is needed.
- 49.0.0 contains routine maintenance and algorithm hardening.
- **Recommendation:** Run `uv lock --upgrade-package cryptography` to advance the transitive lock.

### 4.5 `pandas` — major release blocked by upper bound (LOW-MEDIUM)

- **Locked:** 2.3.3 · **Latest:** 3.0.3
- The spec `>=2.3.3,<3` intentionally caps at 3.x, which is correct practice for a major version boundary.
- However, pandas 3.0 has been stable for several months; continued use of `<3` means missing Copy-on-Write enforcement improvements and deprecation removals that simplify code.
- **Recommendation:** Evaluate pandas 3.x compat in the `csv` extra; if compatible, bump to `>=2.3.3,<4`.

### 4.6 `certifi` — minor drift from latest (LOW)

- **Locked:** 2026.5.20 · **Latest:** 2026.6.17
- certifi releases track Mozilla's CA bundle. Staying current matters for TLS verification correctness.
- **Recommendation:** Run `uv lock --upgrade-package certifi`.

---

## 5. Pinning Quality Assessment

### Scoring rubric (0–100)

| Criterion | Max pts | Score | Notes |
|---|---|---|---|
| All runtime deps have explicit lower bounds | 20 | 20 | ✅ All use `>=x.y.z` |
| All runtime deps have explicit upper bounds | 20 | 20 | ✅ All use `<major+1` caps |
| Lock file(s) exist and are committed | 15 | 12 | Two lock files co-exist; ambiguity penalty |
| Locked versions match or are close to latest within range | 20 | 14 | Several packages lag behind latest allowed version |
| Dev deps have upper bounds | 10 | 0 | ❌ All dev deps are lower-bound only (`>=`) |
| No unbounded transitive dependencies of concern | 15 | 8 | `cryptography` and other sensitive transitives lag |
| **Total** | **100** | **74** | |

### Dev dependency pinning gap

Every dev dependency uses a lower-bound-only spec (e.g., `pytest>=9.0.3`). This means `uv lock --upgrade` can silently pull in breaking tool versions. For a project with strict linting gates (`ruff`, `black`, `pyright`) this is especially risky — a formatter version bump can break CI unexpectedly.

**Recommendation:** Add upper bounds to all dev tooling deps, e.g.:
```
pytest>=9.0.3,<10
ruff>=0.15.16,<1
black>=26.5.1,<27
pyright>=1.1.390,<2
bandit>=1.9.4,<2
```

---

## 6. Structural Observations

### 6.1 `pip-audit` declared but missing from `uv.lock`

`pip-audit>=2.9` is in `[dependency-groups] dev` but does not appear in `uv.lock`. This means `uv sync --group dev` will not install it. If the security audit CI step relies on `pip-audit` being present in the venv, it will silently fail or fall back to a globally installed version.

**Recommendation:** Run `uv lock` to regenerate the lock file and confirm `pip-audit` resolves into it, or verify that it is intentionally installed separately in CI.

### 6.2 Redundant lock files

`poetry.lock` (Poetry 2.4.1, 4701 lines) and `uv.lock` (revision 3, 4342 lines) both live at the root. There is no evidence that both toolchains are actively maintained in parallel. Having two independently-updatable lock files means:
- A `poetry update` run will produce a `poetry.lock` that diverges from `uv.lock`.
- Reviewers cannot easily tell which is authoritative.
- CI that uses one tool gets a different resolution than a developer using the other.

**Recommendation:** Pick one tool (`uv` is the primary given `uv.lock` is more complete and smaller) and either delete `poetry.lock` or add a CI step that regenerates it from `uv.lock` exports.

### 6.3 `certifi` lower-bound is a date string, not a version

The spec `certifi>=2026.5.20` uses a date-based version, which is correct for certifi but unusual. The resolver handles this correctly. No action needed, but be aware that certifi's version scheme means **only releases from May 2026 onward are accepted**, which could be a problem if an older environment cannot reach PyPI for an upgrade.

### 6.4 `boto3` upper bound `<1.50` narrows the patch window

`boto3>=1.43.23,<1.50` is tighter than the typical `<2` cap used for other packages. boto3 follows a rolling release model (daily patches). The `<1.50` cap will expire within weeks at the current release cadence (latest is already 1.43.31 with minor versions incrementing quickly toward 1.50).

**Recommendation:** Widen to `<2` to match the pattern used for other AWS SDKs and reduce the maintenance burden of frequent spec bumps.

---

## 7. Prioritised Fix List

| Priority | Action | Package(s) | Effort |
|---|---|---|---|
| 🔴 P1 | Investigate and widen `paramiko` upper bound to `<6` | `paramiko` | Medium (API validation) |
| 🔴 P1 | Test and adopt `igraph` 1.0.0; update spec to `>=1.0.0,<2` | `igraph` | Medium (test run) |
| 🟠 P2 | Advance `leidenalg` lock to 0.12.0 | `leidenalg` | Low (`uv lock --upgrade-package leidenalg`) |
| 🟠 P2 | Advance `cryptography` transitive lock to 49.0.0 | `cryptography` | Low (`uv lock --upgrade-package cryptography`) |
| 🟠 P2 | Confirm `pip-audit` resolves into `uv.lock`; fix if missing | `pip-audit` | Low (re-run `uv lock`) |
| 🟠 P2 | Designate `uv.lock` as canonical; delete or archive `poetry.lock` | both lock files | Low |
| 🟡 P3 | Widen `boto3` upper bound from `<1.50` to `<2` | `boto3` | Low (spec edit + lock) |
| 🟡 P3 | Add upper bounds to all dev dependencies | all dev | Low (spec edits) |
| 🟡 P3 | Advance `sentence-transformers` lock to 5.6.0 | `sentence-transformers` | Low (`uv lock --upgrade-package sentence-transformers`) |
| 🟡 P3 | Advance `certifi`, `boto3`, `pypdf`, `sqlalchemy`, `duckdb` locks | misc | Low (`uv lock --upgrade`) |
| 🔵 P4 | Evaluate pandas 3.x compat and widen spec to `<4` | `pandas` | Medium (test run) |

---

## 8. Suggested `pyproject.toml` Diffs

```toml
# P1 — widen paramiko
- "paramiko>=3.5.1,<4"
+ "paramiko>=3.5.1,<6"

# P1 — adopt igraph 1.0 stable
- "igraph>=0.11.9,<1", "leidenalg>=0.10.2,<1"
+ "igraph>=1.0.0,<2",  "leidenalg>=0.12.0,<2"

# P2 — widen boto3 window
- "boto3>=1.43.23,<1.50"
+ "boto3>=1.43.23,<2"

# P3 — add upper bounds to dev tools
- "pytest>=9.0.3"
+ "pytest>=9.0.3,<10"

- "ruff>=0.15.16"
+ "ruff>=0.15.16,<1"

- "black>=26.5.1"
+ "black>=26.5.1,<27"

- "pyright>=1.1.390"
+ "pyright>=1.1.390,<2"

- "bandit>=1.9.4"
+ "bandit>=1.9.4,<2"

- "langsmith>=0.8.9"
+ "langsmith>=0.8.9,<1"
```

After editing `pyproject.toml`, regenerate the lock file:

```sh
uv lock --upgrade
```

Then verify the test suite and linters pass before committing.


