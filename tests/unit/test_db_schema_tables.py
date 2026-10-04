# Copyright (c) 2025 Kenneth Stott. MIT License.

"""DatabaseSchemaCrawler indexes tables: columns, types, keys, and comments.

Each table becomes a document (its comment, then one line per column with its
type, key role, and comment), and ``get_table_meta()`` returns the same tables as
``TableMeta`` for ``DocumentLoader.load_schema`` and ``SchemaVocabBuilder``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

sa = pytest.importorskip("sqlalchemy")

from chonk.transports._db_schema import DatabaseSchemaCrawler  # noqa: E402


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE TABLE companies (cik TEXT PRIMARY KEY, legal_name TEXT NOT NULL)")
        )
        conn.execute(
            sa.text(
                "CREATE TABLE filings (accession_number TEXT PRIMARY KEY,"
                " cik TEXT REFERENCES companies(cik), filing_date TEXT, paragraph_text TEXT)"
            )
        )
        conn.execute(sa.text("CREATE VIEW recent AS SELECT * FROM filings"))
    engine.dispose()
    return url


def _docs(crawler: DatabaseSchemaCrawler) -> dict[str, str]:
    return {uri: crawler.fetch(uri).data.decode() for uri in crawler.crawl()}


def test_tables_are_indexed_with_columns_and_keys(sqlite_url: str) -> None:
    crawler = DatabaseSchemaCrawler(sqlite_url, include_procs=False, include_triggers=False)

    docs = _docs(crawler)
    tables = {uri.rsplit("/", 1)[-1]: text for uri, text in docs.items() if "/table/" in uri}

    assert set(tables) == {"companies", "filings"}
    filings = tables["filings"]
    assert "accession_number" in filings and "primary key" in filings
    assert "cik" in filings and "companies.cik" in filings
    assert "paragraph_text" in filings
    assert any("/view/" in uri for uri in docs), "views are still indexed"


def test_table_meta_matches_the_indexed_tables(sqlite_url: str) -> None:
    crawler = DatabaseSchemaCrawler(sqlite_url, include_procs=False, include_triggers=False)
    crawler.crawl()

    meta = {t.name: t for t in crawler.get_table_meta()}

    assert set(meta) == {"companies", "filings"}
    cols = {c.name: c for c in meta["filings"].columns}
    assert cols["accession_number"].is_primary_key
    assert cols["cik"].is_foreign_key and cols["cik"].foreign_key_ref == "companies.cik"
    assert not cols["filing_date"].is_primary_key


def test_include_tables_false_skips_tables(sqlite_url: str) -> None:
    crawler = DatabaseSchemaCrawler(
        sqlite_url, include_procs=False, include_triggers=False, include_tables=False
    )

    assert not any("/table/" in uri for uri in crawler.crawl())
    assert crawler.get_table_meta() == []


class _CommentingInspector:
    """An inspector for a dialect that stores table and column comments."""

    def get_table_names(self, schema: Any = None) -> list[str]:  # noqa: ANN401
        return ["mda_sections"]

    def get_table_comment(self, table: str, schema: Any = None) -> dict[str, str]:  # noqa: ANN401
        return {"text": "Management's discussion and analysis, one row per paragraph."}

    def get_columns(self, table: str, schema: Any = None) -> list[dict[str, Any]]:  # noqa: ANN401
        return [
            {"name": "cik", "type": "VARCHAR", "nullable": False, "comment": "Central Index Key"},
            {"name": "paragraph_text", "type": "VARCHAR", "nullable": True, "comment": None},
        ]

    def get_pk_constraint(self, table: str, schema: Any = None) -> dict[str, Any]:  # noqa: ANN401
        return {"constrained_columns": []}

    def get_foreign_keys(self, table: str, schema: Any = None) -> list[dict[str, Any]]:  # noqa: ANN401
        return []


def test_table_and_column_comments_are_kept() -> None:
    crawler = DatabaseSchemaCrawler("postgresql://unused", schemas=["sec"])

    crawler._index_tables(_CommentingInspector())

    (table,) = crawler.get_table_meta()
    assert table.schema_name == "sec"
    assert table.description == "Management's discussion and analysis, one row per paragraph."
    assert {c.name: c.description for c in table.columns} == {
        "cik": "Central Index Key",
        "paragraph_text": None,
    }
    (text,) = [crawler.fetch(uri).data.decode() for uri in crawler._cache]
    assert "Management's discussion and analysis" in text
    assert "Central Index Key" in text
