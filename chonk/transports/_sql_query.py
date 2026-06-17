# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b906d571-b4e2-4c69-becf-709df8b05d02
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SqlQueryTransport — execute SQL queries / views against a live DB connection.

Wraps an existing engine or connection so no second authentication is needed.
Results are returned as CSV bytes and flow through the existing CsvExtractor.

Usage via transport protocol::

    transport = SqlQueryTransport(engine)
    loader = DocumentLoader(extra_transports=[transport])
    chunks = loader.load("sqlquery://customer_summary",
                         sql="SELECT * FROM v_customer_360")

Usage via DocumentLoader convenience method (recommended)::

    chunks = loader.load_from_db(
        connection=engine,
        queries={
            "customer_summary": "SELECT * FROM v_customer_360",
            "open_invoices":    "SELECT customer_name, amount FROM invoices "
                                "WHERE status = 'open'",
        },
    )

Accepts the same connection types as NerPipeline.add_from_db():
- SQLAlchemy URL string
- SQLAlchemy Engine (has .connect())
- SQLAlchemy Connection or any object with .execute()
"""

from __future__ import annotations

import csv
import io
from typing import Any

from ._protocol import FetchResult

SCHEME = "sqlquery"


def _resolve(connection: Any) -> Any:  # noqa: ANN401
    """Return a usable connection.  Accepts URL str, Engine, or Connection."""
    if isinstance(connection, str):
        try:
            import sqlalchemy as sa
        except ImportError as exc:
            raise ImportError(
                "sqlalchemy is required for SqlQueryTransport. Install with: pip install sqlalchemy"
            ) from exc
        return sa.create_engine(connection).connect()
    if hasattr(connection, "connect") and not hasattr(connection, "execute"):
        return connection.connect()
    if hasattr(connection, "execute"):
        return connection
    raise TypeError(
        f"connection must be a URL string, SQLAlchemy Engine, or an object "
        f"with .execute(). Got: {type(connection)}"
    )


def db_provenance(connection: Any) -> dict[str, object]:  # noqa: ANN401
    """Return safe (no-credential) DB location info from a connection object.

    Extracts dialect, host, port, and database name from SQLAlchemy Engine,
    Connection, or URL string.  Returns an empty dict for DBAPI-only objects.
    """
    url = None
    try:
        if isinstance(connection, str):
            import sqlalchemy as sa

            url = sa.make_url(connection)
        elif hasattr(connection, "url"):  # Engine
            url = connection.url
        elif hasattr(connection, "engine"):  # SQLAlchemy Connection
            url = connection.engine.url
    except Exception:
        return {}

    if url is None:
        return {}

    info: dict[str, object] = {}
    if getattr(url, "drivername", None):
        info["db_dialect"] = url.drivername
    if getattr(url, "host", None):
        info["db_host"] = url.host
    if getattr(url, "port", None):
        info["db_port"] = url.port
    if getattr(url, "database", None):
        info["db_name"] = url.database
    return info


def _maybe_close(conn: Any, original: Any) -> None:  # noqa: ANN401
    if isinstance(original, str) or (
        hasattr(original, "connect") and not hasattr(original, "execute")
    ):
        try:
            conn.close()
        except Exception:
            pass


def _query_to_csv(conn: Any, sql: str) -> bytes:  # noqa: ANN401
    """Execute *sql* and return result as UTF-8 CSV bytes."""
    try:
        import sqlalchemy as sa

        result = conn.execute(sa.text(sql))
    except Exception:
        result = conn.execute(sql)
    columns = list(result.keys())
    rows = result.fetchall()
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


class SqlQueryTransport:
    """Transport that executes SQL queries / views against a live DB connection.

    Args:
        connection: SQLAlchemy URL string, Engine, Connection, or any object
            with ``.execute()``.  The same connection used for schema
            introspection and NerPipeline.add_from_db() can be passed here
            — no second authentication needed.
    """

    def __init__(self, connection: Any) -> None:  # noqa: ANN401
        self._connection = connection

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(f"{SCHEME}://")

    def fetch(self, uri: str, sql: str | None = None, **kwargs: object) -> FetchResult:
        """Fetch query results as CSV bytes.

        Args:
            uri:  ``sqlquery://<document_name>``  — the path becomes the
                  chunk ``document_name``.
            sql:  SQL query or view reference to execute.  Required unless
                  passed as a ``?sql=`` query parameter in the URI.

        Returns:
            FetchResult with CSV bytes and ``detected_mime="text/csv"``.
        """
        from urllib.parse import parse_qs, unquote_plus, urlparse

        parsed = urlparse(uri)
        qs = parse_qs(parsed.query)

        if "sql" in qs:
            sql = unquote_plus(qs["sql"][0])
        if not sql:
            raise ValueError(
                f"SqlQueryTransport: no SQL provided for {uri!r}. "
                "Pass sql= to fetch() or include ?sql=... in the URI."
            )

        doc_name = parsed.netloc or parsed.path.lstrip("/") or "query"

        conn = _resolve(self._connection)
        try:
            csv_bytes = _query_to_csv(conn, sql)
        finally:
            _maybe_close(conn, self._connection)

        return FetchResult(
            data=csv_bytes,
            detected_mime="text/csv",
            source_path=doc_name,
        )
