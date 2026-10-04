# Copyright (c) 2025 Kenneth Stott. MIT License.

"""DatabaseSchemaCrawler — index tables, views, stored procedures, and triggers via SQLAlchemy.

Implements both the ``Crawler`` and ``Transport`` protocols. Pass the same
instance as both ``crawler=`` and in ``extra_transports=``.

Supported dialects: PostgreSQL, MySQL/MariaDB, SQL Server, SQLite.

Usage::

    from chonk.transports import DatabaseSchemaCrawler
    from chonk.loader import DocumentLoader

    crawler = DatabaseSchemaCrawler("postgresql://user:pass@host/db")
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl("postgresql://user:pass@host/db", crawler=crawler)

Each table becomes a document listing its comment and its columns (type, key
role, comment). ``get_table_meta()`` returns the same tables as ``TableMeta``, for
``DocumentLoader.load_schema`` (one chunk per table and per column) and
``SchemaVocabBuilder.add_tables`` / ``NerPipeline.add_tables`` (column names as
entities)::

    crawler.crawl()
    schema_chunks = loader.load_schema(crawler.get_table_meta())

Filter to specific object types::

    crawler = DatabaseSchemaCrawler(
        "mssql+pyodbc://...",
        include_procs=True,
        include_views=True,
        include_triggers=False,
        schemas=["dbo", "reporting"],
    )
"""

from __future__ import annotations

import hashlib
import logging
import textwrap
from typing import Any

from ..schema import ColumnMeta, TableMeta
from ._protocol import FetchOptions, FetchResult  # noqa: F401

_log = logging.getLogger(__name__)

# ── Dialect-specific queries ─────────────────────────────────────────────────

_PG_ROUTINES = """
SELECT n.nspname AS schema_name, p.proname AS routine_name,
       CASE p.prokind WHEN 'f' THEN 'FUNCTION'
            WHEN 'p' THEN 'PROCEDURE' ELSE 'FUNCTION' END AS routine_type,
       pg_get_functiondef(p.oid) AS definition
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
  AND p.prokind IN ('f', 'p')
"""

_PG_TRIGGERS = """
SELECT trigger_schema AS schema_name, trigger_name,
       event_manipulation || ' ON ' || event_object_schema || '.' || event_object_table AS context,
       action_statement AS definition
FROM information_schema.triggers
WHERE trigger_schema NOT IN ('pg_catalog', 'information_schema')
"""

_MYSQL_ROUTINES = """
SELECT ROUTINE_SCHEMA AS schema_name, ROUTINE_NAME AS routine_name,
       ROUTINE_TYPE AS routine_type, ROUTINE_DEFINITION AS definition
FROM information_schema.ROUTINES
WHERE ROUTINE_SCHEMA NOT IN ('sys', 'information_schema', 'performance_schema', 'mysql')
  AND ROUTINE_DEFINITION IS NOT NULL
"""

_MYSQL_TRIGGERS = """
SELECT TRIGGER_SCHEMA AS schema_name, TRIGGER_NAME AS trigger_name,
       CONCAT(EVENT_MANIPULATION, ' ON ', EVENT_OBJECT_TABLE) AS context,
       ACTION_STATEMENT AS definition
FROM information_schema.TRIGGERS
WHERE TRIGGER_SCHEMA NOT IN ('sys', 'information_schema', 'performance_schema', 'mysql')
"""

# SQL Server — one query covers procs, functions, views, and triggers via sys.sql_modules
_MSSQL_OBJECTS = """
SELECT OBJECT_SCHEMA_NAME(o.object_id) AS schema_name,
       o.name AS object_name,
       o.type_desc AS object_type,
       m.definition
FROM sys.sql_modules m
JOIN sys.objects o ON o.object_id = m.object_id
WHERE o.type IN ('P', 'FN', 'IF', 'TF', 'V', 'TR')
"""


# Schemas that hold the database's own catalog, not user tables.
_SYSTEM_SCHEMAS = frozenset(
    {"information_schema", "pg_catalog", "pg_toast", "sys", "mysql", "performance_schema"}
)


class DatabaseSchemaCrawler:
    """Crawl database schema objects (tables, views, procs, triggers) as indexed documents.

    Implements both ``Crawler`` and ``Transport`` — pass the same instance to both::

        crawler = DatabaseSchemaCrawler("postgresql://...")
        loader = DocumentLoader(extra_transports=[crawler])
        chunks = loader.load_crawl("postgresql://...", crawler=crawler)

    Args:
        connection_url:    SQLAlchemy connection URL.
        include_procs:     Include stored procedures and functions (default True).
        include_views:     Include views (default True).
        include_triggers:  Include triggers (default True).
        include_tables:    Include tables, with columns, keys, and comments (default True).
        schemas:           Restrict to these schema names. None = all non-system schemas.
    """

    def __init__(
        self,
        connection_url: str,
        include_procs: bool = True,
        include_views: bool = True,
        include_triggers: bool = True,
        schemas: list[str] | None = None,
        include_tables: bool = True,
    ) -> None:
        self._url = connection_url
        self.include_procs = include_procs
        self.include_views = include_views
        self.include_triggers = include_triggers
        self.include_tables = include_tables
        self._table_meta: list[TableMeta] = []
        self._schemas = set(schemas) if schemas else None
        self._cache: dict[str, FetchResult] = {}
        self._url_key = hashlib.md5(connection_url.encode(), usedforsecurity=False).hexdigest()[:8]

    # ── Transport + Crawler Protocol ─────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(f"dbschema://{self._url_key}/") or self._looks_like_connection_url(
            uri
        )

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        if uri not in self._cache:
            raise KeyError(f"DatabaseSchemaCrawler: unknown URI {uri!r} — call crawl() first")
        return self._cache[uri]

    def crawl(self, uri: str = "", **__: object) -> list[str]:  # noqa: ARG002  # uri ignored; connection set in constructor
        """Connect to the database and index all schema objects.

        Args:
            uri:     SQLAlchemy connection URL (same as constructor).
            **_kw:   Ignored (protocol compatibility).

        Returns:
            List of ``dbschema://`` URIs, one per schema object.
        """
        try:
            import sqlalchemy as sa
        except ImportError:
            raise ImportError(
                "sqlalchemy is required for DatabaseSchemaCrawler. "
                "Install with: pip install sqlalchemy"
            ) from None

        engine = sa.create_engine(self._url)
        dialect = engine.dialect.name  # 'postgresql', 'mysql', 'mssql', 'sqlite', ...

        self._cache.clear()
        self._table_meta.clear()
        if self.include_tables:
            self._index_tables(sa.inspect(engine))
        with engine.connect() as conn:
            if self.include_views:
                self._index_views(conn, engine, sa)
            if dialect == "postgresql":
                self._run_query(
                    conn,
                    sa,
                    _PG_ROUTINES,
                    "routine_name",
                    "routine_type",
                    skip_procs=not self.include_procs,
                )
                if self.include_triggers:
                    self._run_query(conn, sa, _PG_TRIGGERS, "trigger_name", "TRIGGER")
            elif dialect in ("mysql", "mariadb"):
                if self.include_procs:
                    self._run_query(conn, sa, _MYSQL_ROUTINES, "routine_name", "routine_type")
                if self.include_triggers:
                    self._run_query(conn, sa, _MYSQL_TRIGGERS, "trigger_name", "TRIGGER")
            elif dialect == "mssql":
                self._run_mssql(conn, sa)
            elif dialect == "sqlite":
                self._run_sqlite(conn, sa)
            else:
                _log.warning(
                    "DatabaseSchemaCrawler: dialect %r not supported — views only", dialect
                )

        _log.info("DatabaseSchemaCrawler: indexed %d object(s)", len(self._cache))
        return list(self._cache.keys())

    def get_table_meta(self) -> list[TableMeta]:
        """Return the crawled tables as ``TableMeta``. Call after ``crawl()``.

        Pass to ``DocumentLoader.load_schema`` for one chunk per table and per
        column, or to ``SchemaVocabBuilder.add_tables`` / ``NerPipeline.add_tables``
        to make column names entities.
        """
        return list(self._table_meta)

    # ── Indexing helpers ─────────────────────────────────────────────────────

    def _index_tables(self, insp: Any) -> None:  # noqa: ANN401
        """Index every table with its comment, columns, and keys."""
        schemas: list[str] = (
            sorted(self._schemas)
            if self._schemas
            else [s for s in insp.get_schema_names() if s not in _SYSTEM_SCHEMAS]
        )
        for schema in schemas:
            for name in insp.get_table_names(schema=schema):
                table = self._table(insp, schema, name)
                self._table_meta.append(table)
                self._store(table.schema_name or "", name, "TABLE", _table_text(table))

    @staticmethod
    def _table(insp: Any, schema: str, name: str) -> TableMeta:  # noqa: ANN401
        try:
            comment = insp.get_table_comment(name, schema=schema).get("text")
        except NotImplementedError:  # the dialect stores no table comments (e.g. SQLite)
            comment = None
        primary = set(insp.get_pk_constraint(name, schema=schema).get("constrained_columns") or [])
        foreign: dict[str, str] = {}
        for fk in insp.get_foreign_keys(name, schema=schema):
            # "table.column" within the schema; "schema.table.column" across schemas
            ref_schema = fk.get("referred_schema")
            if ref_schema in (schema, "main"):
                ref_schema = None
            target = ".".join(p for p in (ref_schema, fk["referred_table"]) if p)
            for col, ref in zip(fk["constrained_columns"], fk["referred_columns"], strict=True):
                foreign[col] = f"{target}.{ref}"
        columns = [
            ColumnMeta(
                name=c["name"],
                data_type=str(c["type"]),
                description=c.get("comment") or None,
                nullable=bool(c.get("nullable", True)),
                is_primary_key=c["name"] in primary,
                is_foreign_key=c["name"] in foreign,
                foreign_key_ref=foreign.get(c["name"]),
            )
            for c in insp.get_columns(name, schema=schema)
        ]
        return TableMeta(
            name=name,
            schema_name=schema,
            description=comment or None,
            columns=columns,
        )

    def _store(self, schema: str, name: str, obj_type: str, definition: str) -> None:
        if self._schemas and schema not in self._schemas:
            return
        if not definition or not definition.strip():
            return
        qualified = f"{schema}.{name}" if schema and schema != "main" else name
        uri = f"dbschema://{self._url_key}/{obj_type.lower()}/{qualified}"
        header = f"-- {obj_type}: {qualified}\n"
        text = (header + textwrap.dedent(definition).strip() + "\n").encode("utf-8")
        self._cache[uri] = FetchResult(
            data=text,
            detected_mime="text/x-sql",
            source_path=f"{obj_type}: {qualified}",
        )

    def _index_views(self, _conn: Any, engine: Any, sa: Any) -> None:  # noqa: ANN401
        insp = sa.inspect(engine)
        for schema in self._schemas or [None]:  # type: ignore[list-item]
            try:
                names = insp.get_view_names(schema=schema)
            except Exception as exc:
                _log.debug("get_view_names(%s) failed: %s", schema, exc)
                continue
            for vname in names:
                try:
                    defn = insp.get_view_definition(vname, schema=schema)
                    self._store(schema or "public", vname, "VIEW", defn or "")
                except Exception as exc:
                    _log.debug("get_view_definition(%s) failed: %s", vname, exc)

    def _run_query(
        self,
        conn: Any,  # noqa: ANN401
        sa: Any,  # noqa: ANN401
        sql: str,
        name_col: str,
        type_col: str,
        skip_procs: bool = False,
    ) -> None:
        try:
            rows = conn.execute(sa.text(sql)).mappings().all()
        except Exception as exc:
            _log.warning("DatabaseSchemaCrawler query failed: %s", exc)
            return
        for row in rows:
            obj_type = str(row.get(type_col, type_col)).upper()
            if skip_procs and obj_type in ("FUNCTION", "PROCEDURE"):
                continue
            self._store(
                str(row.get("schema_name", "")),
                str(row.get(name_col, "")),
                obj_type,
                str(row.get("definition", "") or ""),
            )

    def _run_mssql(self, conn: Any, sa: Any) -> None:  # noqa: ANN401
        try:
            rows = conn.execute(sa.text(_MSSQL_OBJECTS)).mappings().all()
        except Exception as exc:
            _log.warning("DatabaseSchemaCrawler mssql query failed: %s", exc)
            return
        type_map = {
            "SQL_STORED_PROCEDURE": ("PROCEDURE", self.include_procs),
            "SQL_SCALAR_FUNCTION": ("FUNCTION", self.include_procs),
            "SQL_INLINE_TABLE_VALUED_FUNCTION": ("FUNCTION", self.include_procs),
            "SQL_TABLE_VALUED_FUNCTION": ("FUNCTION", self.include_procs),
            "VIEW": ("VIEW", self.include_views),
            "SQL_TRIGGER": ("TRIGGER", self.include_triggers),
        }
        for row in rows:
            obj_type_raw = str(row.get("object_type", "")).strip()
            entry = type_map.get(obj_type_raw)
            if not entry:
                continue
            obj_type, include = entry
            if not include:
                continue
            self._store(
                str(row.get("schema_name", "dbo")),
                str(row.get("object_name", "")),
                obj_type,
                str(row.get("definition", "") or ""),
            )

    def _run_sqlite(self, conn: Any, sa: Any) -> None:  # noqa: ANN401
        rows_v = rows_t = []
        if self.include_views:
            try:
                rows_v = (
                    conn.execute(
                        sa.text(
                            "SELECT 'main' schema_name, name object_name,"
                            " 'VIEW' object_type, sql definition"
                            " FROM sqlite_master WHERE type='view' AND sql IS NOT NULL"
                        )
                    )
                    .mappings()
                    .all()
                )
            except Exception as exc:
                _log.warning("DatabaseSchemaCrawler sqlite views failed: %s", exc)
        if self.include_triggers:
            try:
                rows_t = (
                    conn.execute(
                        sa.text(
                            "SELECT 'main' schema_name, name object_name,"
                            " 'TRIGGER' object_type, sql definition"
                            " FROM sqlite_master WHERE type='trigger' AND sql IS NOT NULL"
                        )
                    )
                    .mappings()
                    .all()
                )
            except Exception as exc:
                _log.warning("DatabaseSchemaCrawler sqlite triggers failed: %s", exc)
        for row in list(rows_v) + list(rows_t):
            self._store(
                str(row["schema_name"]),
                str(row["object_name"]),
                str(row["object_type"]),
                str(row["definition"] or ""),
            )

    @staticmethod
    def _looks_like_connection_url(uri: str) -> bool:
        return any(
            uri.startswith(p)
            for p in (
                "postgresql://",
                "postgres://",
                "mysql://",
                "mssql://",
                "sqlite://",
                "mariadb://",
            )
        )


def _table_text(table: TableMeta) -> str:
    """A table as a document: its comment, then one line per column."""
    lines = [f"Comment: {table.description}"] if table.description else []
    lines.append("Columns:")
    for col in table.columns:
        role = []
        if col.is_primary_key:
            role.append("primary key")
        if col.foreign_key_ref:
            role.append(f"references {col.foreign_key_ref}")
        line = f"  {col.name} {col.data_type}"
        if role:
            line += f" [{', '.join(role)}]"
        if col.description:
            line += f" -- {col.description}"
        lines.append(line)
    return "\n".join(lines)
