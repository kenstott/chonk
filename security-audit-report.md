# Security Audit Report — Input Validation
**Standard:** SOC2 | **Date:** 2025  
**Codebase:** chonk (RAG pipeline + MCP servers)  
**Scope:** All Python source under `chonk/`, `mcp_chonk_server.py`, `aipa_test_mcp_server/server.py`

---

## Executive Summary

| Severity | Count |
|----------|-------|
| 🔴 Critical (CVSS 9.0+) | 1 |
| 🟠 High (CVSS 7.0–8.9) | 3 |
| 🟡 Medium (CVSS 4.0–6.9) | 4 |
| 🔵 Low (CVSS 1.0–3.9) | 3 |

**SOC2 relevance:** All findings map to the **Availability**, **Confidentiality**, and **Processing Integrity** trust service criteria. Unresolved High/Critical findings are non-compliant with CC6.1 (Logical Access), CC7.1 (System Monitoring), and CC8.1 (Change Management).

---

## Findings

---

### VULN-01 — SQL Injection via `query_text` in PgVectorBackend BM25 Search
**File:** `chonk/storage/_pg.py` (approx. line 638)  
**Severity:** 🔴 Critical — CVSS 9.1 (AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H)  
**OWASP:** A03:2021 — Injection  
**SOC2 Criteria:** CC6.1, CC7.1

**Description:**  
The `_search_hybrid()` method in `PgVectorBackend` manually escapes `query_text` by replacing single quotes (`'` → `''`), then interpolates the result directly into a PostgreSQL SQL string using an f-string. This is a textbook SQL injection pattern — single-quote doubling is not a safe escaping strategy in all contexts (e.g., dollar-quoted strings, `$$..$$`), and bypasses are well documented.

The vulnerable sink is reached whenever any caller invokes `store.search(query_text=...)`, which flows from user-controlled input in both MCP server tools.

**Vulnerable code:**
```python
# chonk/storage/_pg.py
safe_query = query_text.replace("'", "''")
bm25_where = (
    where + " AND " if where else "WHERE "
) + f"fts_vec @@ plainto_tsquery('english', '{safe_query}')"

# ...
cur.execute(
    f"""
    SELECT chunk_id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank(
                   fts_vec, plainto_tsquery('english', '{safe_query}')
               ) DESC
           ) AS rank
    FROM {t} {bm25_where}
    LIMIT %s
    """,
    bm25_params,
)
```

**Attack vector:**  
A malicious `query_text` value such as:
```
') OR 1=1; DROP TABLE embeddings; --
```
or via dollar-quote bypass:
```
') OR 1=1$$
```

**Remediation — use parameterized queries exclusively:**
append test

```python
# FIXED: chonk/storage/_pg.py — _search_hybrid()
# Replace all manual string interpolation of query_text with plainto_tsquery parameter binding.

# BM25 ranking — parameterized
bm25_filter_params = filter_params + [query_text, candidate_limit]
bm25_where_clause = (where + " AND " if where else "WHERE ") + \
    "fts_vec @@ plainto_tsquery('english', %s)"

with self._pgconn.cursor() as cur:
    cur.execute(
        f"""
        SELECT chunk_id,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank(fts_vec, plainto_tsquery('english', %s)) DESC
               ) AS rank
        FROM {t} {bm25_where_clause}
        LIMIT %s
        """,
        [query_text] + filter_params + [query_text, candidate_limit],
    )
    bm25_ranks = {row[0]: row[1] for row in cur.fetchall()}
```

> **Note:** `{t}` (table name) is safe here — it is set once at construction from a trusted string literal default `"embeddings"`, never from user input. If this ever changes, use `psycopg2.sql.Identifier` for table name quoting.

---

### VULN-02 — DuckDB ATTACH Path Injection in `promote_domain` / `attach_global`
**File:** `chonk/storage/_store.py` (lines 531, 626)  
**Severity:** 🟠 High — CVSS 8.1 (AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:N)  
**OWASP:** A03:2021 — Injection  
**SOC2 Criteria:** CC6.1, CC8.1

**Description:**  
Two methods pass a caller-supplied file path directly into a DuckDB `ATTACH` DDL statement via an f-string, without any sanitization or path validation:

```python
# promote_domain() — line 531
conn.execute(f"ATTACH '{target_db_path}' AS _promote_target")

# attach_global() — line 626
conn.execute(f"ATTACH '{global_db_path}' AS global_db (READ_ONLY)")
```

A path containing a single quote can terminate the string literal and inject arbitrary DuckDB DDL. In addition, even without quote injection, an attacker with any influence over `target_db_path` (e.g., via a malicious TOML config or API parameter) can point the ATTACH at an adversarially-crafted DuckDB file, mount a side-channel read of arbitrary database paths, or cause a denial-of-service by attaching an enormous file.

**Attack vector (injection):**  
```python
store.promote_domain(
    "domain",
    "ns",
    "ns2",
    target_db_path="legit.duckdb'; COPY (SELECT * FROM embeddings) TO '/tmp/exfil.csv'; ATTACH 'x"
)
```

**Remediation:**

```python
# FIXED: chonk/storage/_store.py — promote_domain() and attach_global()
import re
from pathlib import Path

def _validate_db_path(path: str | Path) -> str:
    """Validate and normalize a DuckDB file path.
    
    Raises ValueError if the path contains shell-injectable characters.
    Only allow printable ASCII minus single-quote, semicolons, and null bytes.
    """
    resolved = str(Path(path).resolve())
    # Disallow characters that could break the ATTACH string literal
    if re.search(r"['\x00;]", resolved):
        raise ValueError(
            f"Invalid database path {path!r}: path must not contain "
            "single quotes, semicolons, or null bytes."
        )
    return resolved

# In promote_domain():
target_db_path = _validate_db_path(target_db_path)
conn.execute(f"ATTACH '{target_db_path}' AS _promote_target")

# In attach_global():
global_db_path = _validate_db_path(global_db_path)
conn.execute(f"ATTACH '{global_db_path}' AS global_db (READ_ONLY)")
```

> **Longer-term:** DuckDB's Python API does not yet support parameterized identifiers for `ATTACH`. Once it does, switch to that API. Until then, `_validate_db_path` provides defense-in-depth.

---

### VULN-03 — Timing-Attack Vulnerability on API Key Comparison (MCP HTTP Server)
**Files:** `mcp_chonk_server.py` (line 645), `aipa_test_mcp_server/server.py` (line 587)  
**Severity:** 🟠 High — CVSS 7.5 (AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:N)  
**OWASP:** A07:2021 — Identification and Authentication Failures  
**SOC2 Criteria:** CC6.1

**Description:**  
Both MCP HTTP servers compare the incoming bearer token against `CHONK_API_KEY` using Python's `!=` string operator. This comparison is not constant-time, allowing a remote attacker to determine the correct API key one character at a time via response timing differences (timing side-channel).

```python
# mcp_chonk_server.py — line 645
if auth != f"Bearer {_API_KEY}":
    response = JSONResponse({"error": "Unauthorized"}, status_code=401)
```

```python
# aipa_test_mcp_server/server.py — line 587
if auth != f"Bearer {_API_KEY}":
    response = JSONResponse({"error": "Unauthorized"}, status_code=401)
```

**Remediation — use `hmac.compare_digest`:**

```python
# FIXED: both MCP servers — replace != comparison with constant-time digest compare
import hmac

# Replace:
#   if auth != f"Bearer {_API_KEY}":
# With:
if not hmac.compare_digest(auth.encode(), f"Bearer {_API_KEY}".encode()):
    response = JSONResponse({"error": "Unauthorized"}, status_code=401)
    await response(scope, receive, send)
    return
```

---

### VULN-04 — SSRF via Unvalidated URL in WebCrawler / HttpTransport
**Files:** `chonk/transports/_web_crawler.py`, `chonk/transports/_http.py`  
**Severity:** 🟠 High — CVSS 8.6 (AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:N/A:N)  
**OWASP:** A10:2021 — Server-Side Request Forgery  
**SOC2 Criteria:** CC6.1, CC7.1

**Description:**  
`HttpTransport.fetch()` and `WebCrawler._http_get()` will follow any `http://` or `https://` URL without validating whether it targets a private or internal network address. An attacker who can supply a URI (e.g., via the MCP `search_chunks` tool with a crafted URI, a user-controlled `load_site()` call, or a poisoned link followed during a crawl) can cause the server to issue requests to:

- `http://169.254.169.254/latest/meta-data/` (AWS/GCP/Azure IMDS metadata endpoint)
- `http://localhost:6379` (Redis, internal services)
- `http://10.0.0.1/admin` (internal infrastructure)

`WebCrawler` has a `same_domain` flag (default `True`), but this is opt-out and does not block the initial seed URL itself. `HttpTransport` has **no** host validation at all.

**Remediation:**

```python
# NEW FILE: chonk/transports/_url_validation.py
"""Shared URL validation utilities for all transports."""
from __future__ import annotations
import ipaddress
import socket
from urllib.parse import urlparse

# Ranges blocked by default (RFC 1918, loopback, link-local, IMDS)
_BLOCKED_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),    # AWS/GCP/Azure IMDS
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

def is_ssrf_safe(url: str) -> bool:
    """Return True iff url resolves to a public IP address."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
        return not any(addr in net for net in _BLOCKED_RANGES)
    except (socket.gaierror, ValueError):
        return False

def assert_ssrf_safe(url: str) -> None:
    """Raise ValueError if url targets a private/internal address."""
    if not is_ssrf_safe(url):
        raise ValueError(
            f"SSRF guard: {url!r} resolves to a private/internal address. "
            "Only public URLs are permitted."
        )
```

```python
# FIXED: chonk/transports/_http.py — add SSRF guard to fetch()
from ._url_validation import assert_ssrf_safe

class HttpTransport:
    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        assert_ssrf_safe(uri)   # <-- add this line
        session = _get_http_session()
        # ... rest unchanged
```

```python
# FIXED: chonk/transports/_web_crawler.py — guard the seed URL
from ._url_validation import assert_ssrf_safe

class WebCrawler:
    def crawl(self, uri: str, **kwargs: object) -> list[str]:
        assert_ssrf_safe(uri)   # <-- guard the root URL
        # ... rest unchanged
```

---

### VULN-05 — IMAP Search Criteria Injection
**File:** `chonk/transports/_imap.py` (line 87)  
**Severity:** 🟡 Medium — CVSS 5.3 (AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)  
**OWASP:** A03:2021 — Injection  
**SOC2 Criteria:** CC6.1

**Description:**  
The `search_criteria` parameter is read directly from the `?search=` query string of a user-supplied IMAP URI and passed verbatim to `imaplib.IMAP4.search()`:

```python
search_criteria = qs.get("search", ["ALL"])[0]
# ...
typ, data = conn.search(None, search_criteria)
```

IMAP search criteria follow RFC 3501 syntax, which supports arbitrary search flags. A malicious search string can enumerate all messages (`ALL`), cause excessive server load (unbounded search on huge mailboxes), or — on some servers — trigger server-side behaviour outside of what the application intends. There is no allow-list of permitted search terms.

**Remediation:**

```python
# FIXED: chonk/transports/_imap.py
import re

# Allow-list of safe IMAP search keywords (RFC 3501 subset)
_SAFE_IMAP_SEARCH_RE = re.compile(
    r"^(ALL|UNSEEN|SEEN|RECENT|NEW|FLAGGED|UNFLAGGED|DELETED|UNDELETED"
    r"|ANSWERED|UNANSWERED"
    r"|FROM\s+\S+"
    r"|TO\s+\S+"
    r"|SUBJECT\s+\S+"
    r"|SINCE\s+\d{1,2}-\w{3}-\d{4}"
    r"|BEFORE\s+\d{1,2}-\w{3}-\d{4}"
    r"|ON\s+\d{1,2}-\w{3}-\d{4}"
    r")$",
    re.IGNORECASE,
)

search_criteria = qs.get("search", ["ALL"])[0]
if not _SAFE_IMAP_SEARCH_RE.match(search_criteria):
    raise ValueError(
        f"ImapTransport: unsafe IMAP search criteria {search_criteria!r}. "
        "Only standard RFC 3501 keywords are permitted."
    )
```

---

### VULN-06 — SFTP Host Key Validation Disabled (MITM Exposure)
**File:** `chonk/transports/_sftp.py` (line 44)  
**Severity:** 🟡 Medium — CVSS 5.9 (AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N)  
**OWASP:** A02:2021 — Cryptographic Failures  
**SOC2 Criteria:** CC6.1, CC6.7

**Description:**  
`SftpTransport` uses `paramiko.AutoAddPolicy()`, which silently accepts **any** SSH host key — including one presented by an attacker performing a man-in-the-middle attack. This is explicitly flagged `nosec B507` in the source, acknowledging the risk but leaving it unmitigated.

```python
client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())  # nosec B507
```

An attacker on the network path between the chonk server and the SFTP host can intercept the connection and serve malicious documents that get indexed.

**Remediation:**

```python
# FIXED: chonk/transports/_sftp.py
# Option A — use a known_hosts file (preferred for production)
import os
known_hosts = options.known_hosts_path if options else None
if known_hosts and os.path.isfile(known_hosts):
    client.load_host_keys(known_hosts)
    client.set_missing_host_key_policy(_paramiko.RejectPolicy())
else:
    # Option B — allow explicit fingerprint pinning via FetchOptions
    # For now, default to WarningPolicy (logs but does not block) rather than
    # silently trusting. Remove AutoAddPolicy entirely.
    client.set_missing_host_key_policy(_paramiko.WarningPolicy())

# Also add known_hosts_path to FetchOptions:
# @dataclass
# class FetchOptions:
#     ...
#     known_hosts_path: str | None = None
```

---

### VULN-07 — Unbounded `limit` Integer in MCP Server Tools (Denial of Service)
**Files:** `mcp_chonk_server.py` (lines 468, 527), `aipa_test_mcp_server/server.py`  
**Severity:** 🟡 Medium — CVSS 5.3 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H)  
**OWASP:** A05:2021 — Security Misconfiguration  
**SOC2 Criteria:** CC7.1, A1.1 (Availability)

**Description:**  
Both MCP servers declare `minimum: 1, maximum: 50` for `limit` in their JSON Schema tool definitions. However, the **JSON Schema is advisory only** — the MCP framework does not enforce it at runtime. The actual handler code performs no bounds check:

```python
# mcp_chonk_server.py
limit = int(args.get("limit", 5))   # No maximum enforcement
```

A malicious client can send `limit: 999999`, causing the server to:
1. Execute a database query requesting millions of rows.
2. Allocate unbounded memory for the result.
3. Block the event loop during serialization.

**Remediation:**

```python
# FIXED: both MCP servers — clamp limit to a safe maximum
_MAX_LIMIT = 50

def _clamp_limit(args: dict[str, Any], default: int = 5) -> int:
    raw = args.get("limit", default)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        val = default
    return max(1, min(val, _MAX_LIMIT))

# Replace:  limit = int(args.get("limit", 5))
# With:     limit = _clamp_limit(args, default=5)
```

---

### VULN-08 — Unsanitized SQL Passed as Config Value in `_ingest_sql`
**File:** `chonk/ingest.py` (lines 93–116)  
**Severity:** 🟡 Medium — CVSS 4.9 (AV:L/AC:L/PR:H/UI:N/S:U/C:H/I:N/A:N)  
**OWASP:** A03:2021 — Injection  
**SOC2 Criteria:** CC6.1, CC8.1

**Description:**  
The `_ingest_sql()` function executes raw SQL taken from a TOML/YAML config file:

```python
query = src["query"].strip()
# ...
conn = duckdb.connect(db_path, read_only=True)
rows = conn.execute(query).fetchall()   # raw SQL from config
```

While this is limited to operator-controlled config files, the `read_only=True` argument to DuckDB is **not** a strong security boundary — DuckDB `read_only` mode still permits `PRAGMA` statements, file `ATTACH`, and extension loading that can access the filesystem or create side effects. If config is sourced from an untrusted location (CI artifact, remote config store), this becomes an arbitrary code execution vector.

Additionally, `column_names.index(name_col)` will raise an unhandled `ValueError` if the column specified in config does not exist in the query result, which may expose internal query structure in error messages.

**Remediation:**

```python
# FIXED: chonk/ingest.py — _ingest_sql()

# 1. Validate the SQL is a SELECT statement only
def _assert_select_only(sql: str) -> None:
    first_token = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_token != "SELECT":
        raise ValueError(
            f"ingest sql source: only SELECT statements are permitted, got {first_token!r}"
        )

# 2. Validate column names exist before indexing
def _ingest_sql(loader: DocumentLoader, src: dict[str, Any]) -> list[DocumentChunk]:
    connection = src["connection"]
    query = src["query"].strip()
    _assert_select_only(query)            # <-- guard
    name = src.get("name", "sql_source")
    name_col = src.get("name_col")
    content_col = src.get("content_col")
    if name_col and content_col:
        import duckdb
        db_path = connection.replace("duckdb:///", "")
        conn = duckdb.connect(db_path, read_only=True)
        rows = conn.execute(query).fetchall()
        col_names = [d[0] for d in conn.description]
        conn.close()
        if name_col not in col_names:
            raise ValueError(
                f"ingest sql source: name_col {name_col!r} not in query columns {col_names}"
            )
        if content_col not in col_names:
            raise ValueError(
                f"ingest sql source: content_col {content_col!r} not in query columns {col_names}"
            )
        name_idx = col_names.index(name_col)
        content_idx = col_names.index(content_col)
        # ... rest unchanged
```

---

### VULN-09 — Local Transport Allows Arbitrary File Read (No Path Restriction)
**File:** `chonk/transports/_local.py` (lines 37–47)  
**Severity:** 🔵 Low — CVSS 3.5 (AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N)  
**OWASP:** A01:2021 — Broken Access Control  
**SOC2 Criteria:** CC6.1

**Description:**  
`LocalTransport.fetch()` reads any file path without restricting access to a declared root:

```python
path = Path(uri)
return FetchResult(data=path.read_bytes(), ...)
```

In a multi-tenant deployment where users supply URIs, this allows reading `/etc/passwd`, SSH private keys, or other sensitive system files. The `DirectoryCrawler` filters by extension, but `LocalTransport.fetch()` is called directly and enforces no such constraint.

**Remediation:**

```python
# FIXED: chonk/transports/_local.py
class LocalTransport:
    def __init__(self, allowed_root: str | Path | None = None) -> None:
        self._allowed_root = Path(allowed_root).resolve() if allowed_root else None

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        path = Path(uri[len("file://"):] if uri.startswith("file://") else uri).resolve()
        if self._allowed_root and not str(path).startswith(str(self._allowed_root)):
            raise PermissionError(
                f"LocalTransport: path {path!r} is outside allowed root "
                f"{self._allowed_root!r}"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return FetchResult(data=path.read_bytes(), detected_mime=None, source_path=str(path))
```

---

### VULN-10 — FTP Credentials Visible in URI Error Logs
**File:** `chonk/transports/_ftp.py` (line 36)  
**Severity:** 🔵 Low — CVSS 3.1 (AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N)  
**OWASP:** A09:2021 — Security Logging and Monitoring Failures  
**SOC2 Criteria:** CC7.2

**Description:**  
When `FtpTransport.fetch()` raises a `ValueError` for a missing host, it includes the full URI in the error message:

```python
raise ValueError(f"FtpTransport: could not parse host from URI: {uri!r}")
```

An FTP URI includes credentials in plaintext: `ftp://user:password@host/path`. If this exception propagates to a centralized log aggregator or is returned to an API caller, credentials are exposed.

**Remediation:**

```python
# FIXED: chonk/transports/_ftp.py
from urllib.parse import urlparse

def _redact_uri(uri: str) -> str:
    """Return URI with credentials replaced by ***."""
    parsed = urlparse(uri)
    if parsed.password:
        uri = uri.replace(parsed.password, "***", 1)
    if parsed.username:
        uri = uri.replace(parsed.username, "***", 1)
    return uri

# In fetch():
if host is None:
    raise ValueError(
        f"FtpTransport: could not parse host from URI: {_redact_uri(uri)!r}"
    )
```

The same pattern should be applied to `SftpTransport`, `ImapTransport`, and any other transport that logs or raises with `uri` in the message.

---

### VULN-11 — MCP HTTP Server Binds to `0.0.0.0` by Default
**File:** `mcp_chonk_server.py` (line 662), `aipa_test_mcp_server/server.py` (approx. line 580)  
**Severity:** 🔵 Low — CVSS 2.7 (AV:N/AC:H/PR:N/UI:N/S:U/C:L/I:N/A:N)  
**OWASP:** A05:2021 — Security Misconfiguration  
**SOC2 Criteria:** CC6.1

**Description:**  
When started in HTTP mode, the MCP server defaults to binding on all interfaces (`0.0.0.0`):

```python
host = os.environ.get("CHONK_HOST", "0.0.0.0")
```

If `CHONK_API_KEY` is not set (which is only a warning, not an error), the server is accessible to any host on the network without authentication.

**Remediation:**

```python
# FIXED: both MCP servers
_DEFAULT_HOST = "127.0.0.1"   # Bind localhost by default; require explicit opt-in for 0.0.0.0
host = os.environ.get("CHONK_HOST", _DEFAULT_HOST)

# Also: raise an error if HTTP mode is selected without an API key
if os.environ.get("CHONK_TRANSPORT") == "http" and not os.environ.get("CHONK_API_KEY"):
    raise RuntimeError(
        "CHONK_API_KEY must be set when running in HTTP transport mode. "
        "Set CHONK_API_KEY=<secret> or restrict access at the network layer."
    )
```

---

## Remediation Priority Matrix

| ID | Title | CVSS | Priority | Effort |
|----|-------|------|----------|--------|
| VULN-01 | SQL injection in PgVectorBackend BM25 | 9.1 | P0 — Fix immediately | Low (2 LOC change) |
| VULN-02 | DuckDB ATTACH path injection | 8.1 | P0 — Fix immediately | Low (add validator fn) |
| VULN-03 | Timing attack on API key comparison | 7.5 | P1 — Fix this sprint | Trivial (1 import + 1 LOC) |
| VULN-04 | SSRF via unvalidated URLs | 8.6 | P1 — Fix this sprint | Medium (new validation module) |
| VULN-05 | IMAP search criteria injection | 5.3 | P2 — Fix next sprint | Low (regex allow-list) |
| VULN-06 | SFTP host key validation disabled | 5.9 | P2 — Fix next sprint | Low (change policy) |
| VULN-07 | Unbounded limit (DoS) | 5.3 | P2 — Fix next sprint | Trivial (clamp function) |
| VULN-08 | Unsanitized SQL from config | 4.9 | P2 — Fix next sprint | Low (SELECT-only guard) |
| VULN-09 | Unrestricted local file read | 3.5 | P3 — Fix when convenient | Low (allowed_root param) |
| VULN-10 | Credentials in error log/messages | 3.1 | P3 — Fix when convenient | Low (redact helper) |
| VULN-11 | Insecure default bind address | 2.7 | P3 — Fix when convenient | Trivial (change default) |

---

## SOC2 Compliance Gap Summary

| SOC2 Criterion | Status | Gaps |
|----------------|--------|------|
| **CC6.1** Logical and Physical Access | ❌ Non-compliant | VULN-01, VULN-02, VULN-03, VULN-04 unmitigated |
| **CC6.7** Transmission Integrity | ⚠️ Partial | SFTP MITM (VULN-06) unmitigated |
| **CC7.1** Threat and Vulnerability Management | ⚠️ Partial | No input validation on search path; SQL injection (VULN-01) |
| **CC7.2** Monitoring | ⚠️ Partial | Credential exposure in logs (VULN-10) |
| **CC8.1** Change Management | ⚠️ Partial | Path injection from config (VULN-02, VULN-08) |
| **A1.1** Availability | ⚠️ Partial | No rate-limiting or input size bounds on MCP tools (VULN-07) |

**Attestation readiness:** The codebase is **not ready** for a SOC2 Type II audit in its current state. VULN-01 and VULN-02 represent direct violations of CC6.1 (protection of information assets from unauthorized access) and must be remediated before any audit window opens.
