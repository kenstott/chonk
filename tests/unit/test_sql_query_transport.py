# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holders.

"""Tests for SqlQueryTransport and DocumentLoader.load_from_db()."""

from __future__ import annotations

import pytest

sa = pytest.importorskip("sqlalchemy")


@pytest.fixture()
def engine():
    """In-memory SQLite engine with a small test table."""
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(
            sa.text("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT)")
        )
        conn.execute(
            sa.text(
                "INSERT INTO customers VALUES (1, 'Acme Corp', 'West'), "
                "(2, 'Globex', 'East'), (3, 'Initech', 'West')"
            )
        )
        conn.commit()
    return engine


# ---------------------------------------------------------------------------
# SqlQueryTransport
# ---------------------------------------------------------------------------


class TestSqlQueryTransport:
    def test_can_handle_scheme(self):
        from chonk.transports import SqlQueryTransport

        t = SqlQueryTransport("sqlite:///:memory:")
        assert t.can_handle("sqlquery://my_doc")
        assert not t.can_handle("sqlalchemy://my_doc")
        assert not t.can_handle("file:///foo.csv")

    def test_fetch_returns_csv_bytes(self, engine):
        from chonk.transports import SqlQueryTransport

        t = SqlQueryTransport(engine)
        result = t.fetch("sqlquery://customers", sql="SELECT * FROM customers")
        assert result.detected_mime == "text/csv"
        text = result.data.decode()
        assert "Acme Corp" in text
        assert "Globex" in text

    def test_fetch_doc_name_from_uri(self, engine):
        from chonk.transports import SqlQueryTransport

        t = SqlQueryTransport(engine)
        result = t.fetch("sqlquery://my_report", sql="SELECT name FROM customers")
        assert result.source_path == "my_report"

    def test_fetch_csv_has_header_row(self, engine):
        from chonk.transports import SqlQueryTransport

        t = SqlQueryTransport(engine)
        result = t.fetch("sqlquery://q", sql="SELECT name, region FROM customers")
        lines = result.data.decode().splitlines()
        assert lines[0] == "name,region"

    def test_fetch_no_sql_raises(self, engine):
        from chonk.transports import SqlQueryTransport

        t = SqlQueryTransport(engine)
        with pytest.raises(ValueError, match="no SQL provided"):
            t.fetch("sqlquery://customers")

    def test_accepts_url_string(self):
        from chonk.transports import SqlQueryTransport

        t = SqlQueryTransport("sqlite:///:memory:")
        # Can construct without error; fetch on empty db still works
        result = t.fetch(
            "sqlquery://test",
            sql="SELECT 1 AS val",
        )
        assert b"val" in result.data

    def test_accepts_existing_connection(self, engine):
        from chonk.transports import SqlQueryTransport

        with engine.connect() as conn:
            t = SqlQueryTransport(conn)
            result = t.fetch("sqlquery://q", sql="SELECT COUNT(*) AS n FROM customers")
        assert b"n" in result.data

    def test_invalid_connection_raises(self):
        from chonk.transports import SqlQueryTransport

        t = SqlQueryTransport(object())
        with pytest.raises(TypeError, match="connection must be"):
            t.fetch("sqlquery://q", sql="SELECT 1")


# ---------------------------------------------------------------------------
# DocumentLoader.load_from_db
# ---------------------------------------------------------------------------


class TestLoadFromDb:
    def test_returns_chunks(self, engine):
        from chonk import DocumentLoader

        loader = DocumentLoader(context_strategy=None)
        chunks = loader.load_from_db(
            engine,
            queries={"customers": "SELECT name, region FROM customers"},
        )
        assert len(chunks) >= 1

    def test_document_name_matches_key(self, engine):
        from chonk import DocumentLoader

        loader = DocumentLoader(context_strategy=None)
        chunks = loader.load_from_db(
            engine,
            queries={"my_view": "SELECT name FROM customers"},
        )
        assert all(c.document_name == "my_view" for c in chunks)

    def test_multiple_queries_produce_separate_documents(self, engine):
        from chonk import DocumentLoader

        loader = DocumentLoader(context_strategy=None)
        chunks = loader.load_from_db(
            engine,
            queries={
                "customers": "SELECT name FROM customers",
                "west": "SELECT name FROM customers WHERE region = 'West'",
            },
        )
        doc_names = {c.document_name for c in chunks}
        assert "customers" in doc_names
        assert "west" in doc_names

    def test_accepts_list_of_tuples(self, engine):
        from chonk import DocumentLoader

        loader = DocumentLoader(context_strategy=None)
        chunks = loader.load_from_db(
            engine,
            queries=[("q1", "SELECT name FROM customers")],
        )
        assert len(chunks) >= 1
        assert chunks[0].document_name == "q1"

    def test_chunk_content_contains_data(self, engine):
        from chonk import DocumentLoader

        loader = DocumentLoader(context_strategy=None)
        chunks = loader.load_from_db(
            engine,
            queries={"c": "SELECT name FROM customers"},
        )
        combined = " ".join(c.content for c in chunks)
        assert "Acme Corp" in combined

    def test_empty_result_produces_no_chunks(self, engine):
        from chonk import DocumentLoader

        loader = DocumentLoader(context_strategy=None)
        chunks = loader.load_from_db(
            engine,
            queries={"empty": "SELECT name FROM customers WHERE 1=0"},
        )
        assert chunks == []
