# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SchemaVocabBuilder — two-pass NER vocabulary extraction.

Builds a SchemaMatcher from structured schema metadata (TableMeta/ColumnMeta)
or raw SQL DDL text.  All identifiers are normalized via normalize_schema_term
before being added to the vocab, so camelCase, snake_case, and SCREAMING_SNAKE
all resolve to "first name" style surface forms.

Typical two-pass usage::

    from chonk.ner import SchemaVocabBuilder, SpacyMatcher, merge_matches

    # Pass 1: build vocab from DB metadata or DDL files
    builder = SchemaVocabBuilder()
    builder.add_tables(table_meta_list)          # from DB introspection
    builder.add_sql(ddl_text)                    # from .sql files
    builder.add_chunks(schema_chunks)            # from loader.load_schema()
    schema_matcher = builder.build()

    # Pass 2: combined NER on all document chunks
    spacy = SpacyMatcher()
    for chunk_id, chunk in indexed_chunks:
        schema_hits = schema_matcher.match(chunk.content)
        spacy_hits  = spacy.match(chunk.content)
        combined    = merge_matches(schema_hits, spacy_hits, source_text=chunk.content)
        entity_index.index_chunk(chunk_id, chunk.content, combined)
"""

from __future__ import annotations

import re

from ._schema import SchemaMatcher

# ---------------------------------------------------------------------------
# SQL DDL parsing helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?"
    r"(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r'(?:"?\w+"?\.)?"?([a-zA-Z_]\w*)"?',
    re.IGNORECASE,
)

_ALTER_TABLE_RE = re.compile(
    r'ALTER\s+TABLE\s+(?:"?\w+"?\.)?"?([a-zA-Z_]\w*)"?',
    re.IGNORECASE,
)

# SQL primitive type keywords — column lines must start with an identifier
# followed by one of these to be recognised as a column definition.
_SQL_TYPES = (
    "INTEGER",
    "INT",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "NUMERIC",
    "REAL",
    "MONEY",
    "BOOLEAN",
    "BOOL",
    "TEXT",
    "VARCHAR",
    "CHAR",
    "NCHAR",
    "NVARCHAR",
    "CLOB",
    "BLOB",
    "BYTEA",
    "BINARY",
    "VARBINARY",
    "DATE",
    "TIME",
    "TIMESTAMP",
    "TIMESTAMPTZ",
    "INTERVAL",
    "DATETIME",
    "UUID",
    "JSON",
    "JSONB",
    "XML",
    "SERIAL",
    "BIGSERIAL",
    "SMALLSERIAL",
    "AUTOINCREMENT",
    "ARRAY",
    "HSTORE",
    "GEOMETRY",
    "GEOGRAPHY",
)

# Compiled: column-start line pattern
_COL_RE = re.compile(
    r'^\s*"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s+'
    r"(?:" + "|".join(_SQL_TYPES) + r")",
    re.IGNORECASE | re.MULTILINE,
)

# Lines beginning with these are DDL constraint/index clauses, not columns.
_SKIP_STARTS = re.compile(
    r"^\s*(?:PRIMARY|FOREIGN|UNIQUE|CHECK|INDEX|KEY|CONSTRAINT|REFERENCES|"
    r"COMMENT|ON\s|WITH\s|USING\s)",
    re.IGNORECASE,
)


def _extract_sql_terms(sql: str) -> tuple[list[str], list[str]]:
    """Return (table_names, column_names) extracted from SQL DDL text."""
    tables: list[str] = []
    columns: list[str] = []

    for m in _CREATE_TABLE_RE.finditer(sql):
        tables.append(m.group(1))
    for m in _ALTER_TABLE_RE.finditer(sql):
        name = m.group(1)
        if name not in tables:
            tables.append(name)

    for m in _COL_RE.finditer(sql):
        line = sql[m.start() : sql.find("\n", m.start())]
        if _SKIP_STARTS.match(line):
            continue
        columns.append(m.group(1))

    return tables, columns


# ---------------------------------------------------------------------------
# SchemaVocabBuilder
# ---------------------------------------------------------------------------


class SchemaVocabBuilder:
    """Accumulate schema identifiers and produce a SchemaMatcher.

    All identifiers are deduplicated (by raw value) before being passed to
    SchemaMatcher, which applies normalize_schema_term internally.

    Args:
        min_term_length: Raw identifier length below which terms are ignored
            (default 2 — filters out single-letter columns like ``i`` or ``n``).
    """

    def __init__(self, min_term_length: int = 2):
        self._min = min_term_length
        self._tables: set[str] = set()
        self._columns: set[str] = set()
        self._api_terms: set[str] = set()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_tables(self, tables: list) -> SchemaVocabBuilder:
        """Add terms from a list of TableMeta objects (and their ColumnMeta).

        Accepts any object with a ``.name`` attribute and an optional
        ``.columns`` iterable of objects with a ``.name`` attribute.
        """
        for table in tables:
            name = getattr(table, "name", None)
            if name and len(name) >= self._min:
                self._tables.add(name)
            for col in getattr(table, "columns", None) or []:
                col_name = getattr(col, "name", None)
                if col_name and len(col_name) >= self._min:
                    self._columns.add(col_name)
        return self

    def add_endpoints(self, endpoints: list) -> SchemaVocabBuilder:
        """Add terms from a list of EndpointMeta / FieldMeta objects."""
        for ep in endpoints:
            path = getattr(ep, "path", None)
            if path and len(path) >= self._min:
                self._api_terms.add(path)
            for field in getattr(ep, "fields", None) or []:
                name = getattr(field, "name", None)
                if name and len(name) >= self._min:
                    self._columns.add(name)
        return self

    def add_sql(self, ddl: str) -> SchemaVocabBuilder:
        """Extract and add table/column names from raw SQL DDL text."""
        tables, columns = _extract_sql_terms(ddl)
        for t in tables:
            if len(t) >= self._min:
                self._tables.add(t)
        for c in columns:
            if len(c) >= self._min:
                self._columns.add(c)
        return self

    def add_chunks(self, chunks: list) -> SchemaVocabBuilder:
        """Extract terms from DocumentChunk objects produced by load_schema() / load_api().

        Document names follow the pattern:
        - ``schema:<db>.<table>``       → table term
        - ``schema:<db>.<table>.<col>`` → column term
        - ``api:<api>.<path>.<field>``  → api / column term

        Chunks with other document_name patterns are silently skipped.
        """
        for chunk in chunks:
            doc_name: str = getattr(chunk, "document_name", "") or ""
            chunk_type: str = str(getattr(chunk, "chunk_type", "") or "")

            if chunk_type in ("db_table", "db_column") and doc_name.startswith("schema:"):
                parts = doc_name[len("schema:") :].split(".")
                # parts: [db, table] or [db, table, column]
                if len(parts) >= 2:
                    table = parts[1]
                    if len(table) >= self._min:
                        self._tables.add(table)
                if len(parts) >= 3:
                    col = parts[2]
                    if len(col) >= self._min:
                        self._columns.add(col)

            elif chunk_type in (
                "api_endpoint",
                "api_graphql_query",
                "api_graphql_mutation",
                "api_graphql_type",
            ):
                if doc_name.startswith("api:"):
                    parts = doc_name[len("api:") :].split(".", 1)
                    if len(parts) >= 2:
                        path = parts[1]
                        if len(path) >= self._min:
                            self._api_terms.add(path)

            elif chunk_type == "api_field" and doc_name.startswith("api:"):
                parts = doc_name[len("api:") :].split(".")
                if len(parts) >= 3:
                    field = parts[-1]
                    if len(field) >= self._min:
                        self._columns.add(field)

        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> SchemaMatcher:
        """Return a SchemaMatcher populated with all accumulated terms.

        SchemaMatcher applies normalize_schema_term internally, so
        ``firstName``, ``first_name``, and ``FIRST_NAME`` all produce the
        surface form "first name" and match as the same entity.
        """
        return SchemaMatcher(
            schema_terms=list(self._tables) + list(self._columns),
            api_terms=list(self._api_terms),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def table_count(self) -> int:
        return len(self._tables)

    def column_count(self) -> int:
        return len(self._columns)

    def api_term_count(self) -> int:
        return len(self._api_terms)
