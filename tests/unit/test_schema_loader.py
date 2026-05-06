# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Unit tests for load_schema() and load_api() — Phase 1.2."""
from __future__ import annotations

import pytest

from chonk import DocumentLoader, ColumnMeta, TableMeta, FieldMeta, EndpointMeta
from chonk.models import DocumentChunk


@pytest.fixture
def loader():
    return DocumentLoader()


# ---------------------------------------------------------------------------
# load_schema
# ---------------------------------------------------------------------------

class TestLoadSchema:
    def test_table_only_produces_one_chunk(self, loader):
        table = TableMeta(name="users", source_db="mydb")
        chunks = loader.load_schema([table])
        assert len(chunks) == 1

    def test_table_with_columns_produces_n_plus_one(self, loader):
        table = TableMeta(
            name="orders",
            source_db="mydb",
            columns=[
                ColumnMeta(name="id", data_type="INTEGER"),
                ColumnMeta(name="amount", data_type="NUMERIC"),
                ColumnMeta(name="customer_id", data_type="INTEGER"),
            ],
        )
        chunks = loader.load_schema([table])
        assert len(chunks) == 4  # 1 table + 3 columns

    def test_table_chunk_type_and_document_name(self, loader):
        table = TableMeta(name="products", source_db="shop")
        chunks = loader.load_schema([table])
        t = chunks[0]
        assert t.chunk_type == "db_table"
        assert t.document_name == "schema:shop.products"
        assert t.section == ["table_description"]

    def test_column_chunk_type_and_document_name(self, loader):
        table = TableMeta(
            name="products",
            source_db="shop",
            columns=[ColumnMeta(name="price", data_type="NUMERIC")],
        )
        chunks = loader.load_schema([table])
        col = chunks[1]
        assert col.chunk_type == "db_column"
        assert col.document_name == "schema:shop.products.price"
        assert col.section == ["column_description"]

    def test_source_db_none_uses_fallback_prefix(self, loader):
        table = TableMeta(name="events")  # source_db=None
        chunks = loader.load_schema([table])
        assert chunks[0].document_name == "schema:db.events"

    def test_table_chunk_content_includes_metadata(self, loader):
        table = TableMeta(
            name="invoices",
            source_db="erp",
            description="All issued invoices",
            row_count=42000,
            columns=[ColumnMeta(name="invoice_id", data_type="TEXT")],
        )
        chunks = loader.load_schema([table])
        content = chunks[0].content
        assert "Table: invoices" in content
        assert "Source DB: erp" in content
        assert "Description: All issued invoices" in content
        assert "Row count: 42000" in content
        assert "invoice_id" in content  # column listed

    def test_column_chunk_content_pk_fk(self, loader):
        table = TableMeta(
            name="order_lines",
            source_db="shop",
            columns=[
                ColumnMeta(name="id", data_type="INTEGER", is_primary_key=True),
                ColumnMeta(
                    name="order_id", data_type="INTEGER",
                    is_foreign_key=True, foreign_key_ref="orders.id",
                ),
            ],
        )
        chunks = loader.load_schema([table])
        id_chunk = next(c for c in chunks if c.document_name.endswith(".id"))
        assert "Primary key: yes" in id_chunk.content

        fk_chunk = next(c for c in chunks if c.document_name.endswith(".order_id"))
        assert "Foreign key: orders.id" in fk_chunk.content

    def test_column_chunk_nullable_false(self, loader):
        table = TableMeta(
            name="t",
            source_db="db",
            columns=[ColumnMeta(name="col", data_type="TEXT", nullable=False)],
        )
        chunks = loader.load_schema([table])
        assert "Nullable: no" in chunks[1].content

    def test_multiple_tables_all_chunks_present(self, loader):
        tables = [
            TableMeta(name="a", source_db="db", columns=[ColumnMeta("x", "TEXT")]),
            TableMeta(name="b", source_db="db", columns=[ColumnMeta("y", "INT"), ColumnMeta("z", "INT")]),
        ]
        chunks = loader.load_schema(tables)
        assert len(chunks) == 2 + 3  # (1+1) + (1+2)

    def test_chunks_are_enriched(self, loader):
        table = TableMeta(name="t", source_db="db")
        chunks = loader.load_schema([table])
        assert chunks[0].embedding_content is not None
        assert "schema:db.t" in chunks[0].embedding_content

    def test_chunk_index_ordering(self, loader):
        table = TableMeta(
            name="t", source_db="db",
            columns=[ColumnMeta("a", "TEXT"), ColumnMeta("b", "TEXT")],
        )
        chunks = loader.load_schema([table])
        assert chunks[0].chunk_index == 0   # table
        assert chunks[1].chunk_index == 1   # first column
        assert chunks[2].chunk_index == 2   # second column


# ---------------------------------------------------------------------------
# load_api
# ---------------------------------------------------------------------------

class TestLoadApi:
    def test_endpoint_only_produces_one_chunk(self, loader):
        ep = EndpointMeta(path="/users", method="GET", source_api="myapi")
        chunks = loader.load_api([ep])
        assert len(chunks) == 1

    def test_endpoint_with_fields_produces_n_plus_one(self, loader):
        ep = EndpointMeta(
            path="/orders",
            source_api="myapi",
            fields=[
                FieldMeta(name="order_id", field_type="string"),
                FieldMeta(name="amount", field_type="number"),
            ],
        )
        chunks = loader.load_api([ep])
        assert len(chunks) == 3  # 1 endpoint + 2 fields

    def test_endpoint_chunk_type_rest(self, loader):
        ep = EndpointMeta(path="/items", endpoint_type="rest", source_api="shop")
        chunks = loader.load_api([ep])
        assert chunks[0].chunk_type == "api_endpoint"

    def test_endpoint_chunk_type_graphql_query(self, loader):
        ep = EndpointMeta(path="getUser", endpoint_type="graphql_query", source_api="gql")
        chunks = loader.load_api([ep])
        assert chunks[0].chunk_type == "api_graphql_query"

    def test_endpoint_chunk_type_graphql_mutation(self, loader):
        ep = EndpointMeta(path="createUser", endpoint_type="graphql_mutation", source_api="gql")
        chunks = loader.load_api([ep])
        assert chunks[0].chunk_type == "api_graphql_mutation"

    def test_endpoint_chunk_type_graphql_type(self, loader):
        ep = EndpointMeta(path="User", endpoint_type="graphql_type", source_api="gql")
        chunks = loader.load_api([ep])
        assert chunks[0].chunk_type == "api_graphql_type"

    def test_endpoint_document_name(self, loader):
        ep = EndpointMeta(path="/users/{id}", source_api="myapi")
        chunks = loader.load_api([ep])
        assert chunks[0].document_name == "api:myapi./users/{id}"

    def test_field_chunk_type_and_document_name(self, loader):
        ep = EndpointMeta(
            path="/users",
            source_api="myapi",
            fields=[FieldMeta(name="email", field_type="string", required=True)],
        )
        chunks = loader.load_api([ep])
        field_chunk = chunks[1]
        assert field_chunk.chunk_type == "api_field"
        assert field_chunk.document_name == "api:myapi./users.email"

    def test_field_content_required(self, loader):
        ep = EndpointMeta(
            path="/login",
            source_api="auth",
            fields=[FieldMeta(name="password", field_type="string", required=True)],
        )
        chunks = loader.load_api([ep])
        assert "Required: yes" in chunks[1].content

    def test_field_content_not_required(self, loader):
        ep = EndpointMeta(
            path="/search",
            source_api="api",
            fields=[FieldMeta(name="page", field_type="integer", required=False)],
        )
        chunks = loader.load_api([ep])
        assert "Required: no" in chunks[1].content

    def test_source_api_none_uses_fallback(self, loader):
        ep = EndpointMeta(path="/ping")  # source_api=None
        chunks = loader.load_api([ep])
        assert chunks[0].document_name == "api:api./ping"

    def test_endpoint_content_includes_metadata(self, loader):
        ep = EndpointMeta(
            path="/products",
            method="GET",
            description="List all products",
            source_api="shop",
            endpoint_type="rest",
            fields=[FieldMeta(name="limit", field_type="integer")],
        )
        chunks = loader.load_api([ep])
        content = chunks[0].content
        assert "Endpoint: /products" in content
        assert "Method: GET" in content
        assert "Description: List all products" in content
        assert "Source API: shop" in content
        assert "limit" in content  # field listed

    def test_chunks_are_enriched(self, loader):
        ep = EndpointMeta(path="/ping", source_api="api")
        chunks = loader.load_api([ep])
        assert chunks[0].embedding_content is not None

    def test_multiple_endpoints_all_chunks_present(self, loader):
        endpoints = [
            EndpointMeta(path="/a", source_api="api", fields=[FieldMeta("x", "string")]),
            EndpointMeta(path="/b", source_api="api", fields=[FieldMeta("y", "int"), FieldMeta("z", "int")]),
        ]
        chunks = loader.load_api(endpoints)
        assert len(chunks) == 2 + 3  # (1+1) + (1+2)
