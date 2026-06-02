# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 1246319b-6461-4ccc-a574-6b7a5972e749

"""Unit tests for ParquetExtractor and load_structured_file() — Phase 1.3."""
from __future__ import annotations

import io
import json
import tempfile
import os

import pytest

from chonk import DocumentLoader
from chonk.extractors._parquet import ParquetExtractor


# ---------------------------------------------------------------------------
# Fixtures — build tiny in-memory files
# ---------------------------------------------------------------------------

@pytest.fixture
def parquet_bytes():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({
        "id": pa.array([1, 2, 3], type=pa.int64()),
        "name": pa.array(["alice", "bob", "carol"], type=pa.string()),
        "score": pa.array([9.1, 8.5, 7.2], type=pa.float64()),
    })
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@pytest.fixture
def arrow_bytes():
    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    table = pa.table({"x": pa.array([10, 20], type=pa.int32())})
    buf = io.BytesIO()
    with ipc.new_stream(buf, table.schema) as writer:
        writer.write_table(table)
    return buf.getvalue()


@pytest.fixture
def feather_bytes():
    pa = pytest.importorskip("pyarrow")
    feather = pytest.importorskip("pyarrow.feather")
    table = pa.table({"a": pa.array([True, False], type=pa.bool_())})
    buf = io.BytesIO()
    feather.write_feather(table, buf)
    return buf.getvalue()


@pytest.fixture
def csv_bytes():
    return b"city,population,area\nLondon,9000000,1572\nParis,2161000,105\n"


@pytest.fixture
def jsonl_bytes():
    lines = [
        json.dumps({"user": "alice", "age": 30, "active": True}),
        json.dumps({"user": "bob", "age": 25, "active": False}),
    ]
    return "\n".join(lines).encode()


@pytest.fixture
def json_array_bytes():
    data = [{"product": "widget", "price": 1.99}, {"product": "gadget", "price": 4.99}]
    return json.dumps(data).encode()


@pytest.fixture
def loader():
    return DocumentLoader()


# ---------------------------------------------------------------------------
# ParquetExtractor — schema mode
# ---------------------------------------------------------------------------

class TestParquetExtractorSchema:
    def test_schema_mode_lists_columns(self, parquet_bytes):
        text = ParquetExtractor(mode="schema").extract(parquet_bytes, "data.parquet")
        assert "id" in text
        assert "name" in text
        assert "score" in text

    def test_schema_mode_includes_row_count(self, parquet_bytes):
        text = ParquetExtractor(mode="schema").extract(parquet_bytes, "data.parquet")
        assert "Rows: 3" in text

    def test_schema_mode_includes_sample(self, parquet_bytes):
        text = ParquetExtractor(mode="schema").extract(parquet_bytes, "data.parquet")
        assert "Sample" in text
        assert "alice" in text

    def test_default_mode_is_schema(self, parquet_bytes):
        text = ParquetExtractor().extract(parquet_bytes, "data.parquet")
        assert "Rows:" in text

    def test_can_handle_parquet(self):
        assert ParquetExtractor().can_handle("parquet")

    def test_can_handle_arrow(self):
        assert ParquetExtractor().can_handle("arrow")

    def test_can_handle_feather(self):
        assert ParquetExtractor().can_handle("feather")

    def test_arrow_file(self, arrow_bytes):
        text = ParquetExtractor(mode="schema").extract(arrow_bytes, "data.arrow")
        assert "x" in text

    def test_feather_file(self, feather_bytes):
        text = ParquetExtractor(mode="schema").extract(feather_bytes, "data.feather")
        assert "a" in text


# ---------------------------------------------------------------------------
# ParquetExtractor — data mode
# ---------------------------------------------------------------------------

class TestParquetExtractorData:
    def test_data_mode_markdown_table(self, parquet_bytes):
        text = ParquetExtractor(mode="data").extract(parquet_bytes, "data.parquet")
        assert "| id | name | score |" in text
        assert "alice" in text
        assert "bob" in text

    def test_data_mode_has_separator_row(self, parquet_bytes):
        text = ParquetExtractor(mode="data").extract(parquet_bytes, "data.parquet")
        lines = text.splitlines()
        assert any("---" in line for line in lines)


# ---------------------------------------------------------------------------
# load_structured_file — schema inference
# ---------------------------------------------------------------------------

class TestLoadStructuredFile:
    def _write_tmp(self, data: bytes, suffix: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.write(data)
        f.flush()
        f.close()
        return f.name

    def test_parquet_n_plus_one_chunks(self, loader, parquet_bytes):
        path = self._write_tmp(parquet_bytes, ".parquet")
        try:
            chunks = loader.load_structured_file(path)
            # 1 table + 3 columns
            assert len(chunks) == 4
        finally:
            os.unlink(path)

    def test_parquet_chunk_types(self, loader, parquet_bytes):
        path = self._write_tmp(parquet_bytes, ".parquet")
        try:
            chunks = loader.load_structured_file(path)
            assert chunks[0].chunk_type == "db_table"
            assert all(c.chunk_type == "db_column" for c in chunks[1:])
        finally:
            os.unlink(path)

    def test_parquet_column_types_mapped(self, loader, parquet_bytes):
        path = self._write_tmp(parquet_bytes, ".parquet")
        try:
            chunks = loader.load_structured_file(path)
            id_chunk = next(c for c in chunks if "id" in c.document_name and c.chunk_type == "db_column")
            assert "INTEGER" in id_chunk.content
            score_chunk = next(c for c in chunks if "score" in c.document_name)
            assert "FLOAT" in score_chunk.content
        finally:
            os.unlink(path)

    def test_arrow_produces_schema_chunks(self, loader, arrow_bytes):
        path = self._write_tmp(arrow_bytes, ".arrow")
        try:
            chunks = loader.load_structured_file(path)
            assert len(chunks) == 2  # 1 table + 1 column
            assert chunks[1].chunk_type == "db_column"
        finally:
            os.unlink(path)

    def test_feather_produces_schema_chunks(self, loader, feather_bytes):
        path = self._write_tmp(feather_bytes, ".feather")
        try:
            chunks = loader.load_structured_file(path)
            assert len(chunks) == 2
        finally:
            os.unlink(path)

    def test_csv_n_plus_one_chunks(self, loader, csv_bytes):
        path = self._write_tmp(csv_bytes, ".csv")
        try:
            chunks = loader.load_structured_file(path)
            assert len(chunks) == 4  # 1 table + 3 columns
        finally:
            os.unlink(path)

    def test_csv_column_names_inferred(self, loader, csv_bytes):
        path = self._write_tmp(csv_bytes, ".csv")
        try:
            chunks = loader.load_structured_file(path)
            names = [c.document_name for c in chunks]
            assert any("city" in n for n in names)
            assert any("population" in n for n in names)
        finally:
            os.unlink(path)

    def test_jsonl_n_plus_one_chunks(self, loader, jsonl_bytes):
        path = self._write_tmp(jsonl_bytes, ".jsonl")
        try:
            chunks = loader.load_structured_file(path)
            assert len(chunks) == 4  # 1 table + 3 columns
        finally:
            os.unlink(path)

    def test_jsonl_column_types_inferred(self, loader, jsonl_bytes):
        path = self._write_tmp(jsonl_bytes, ".jsonl")
        try:
            chunks = loader.load_structured_file(path)
            age_chunk = next(c for c in chunks if "age" in c.document_name)
            assert "INTEGER" in age_chunk.content
        finally:
            os.unlink(path)

    def test_json_array_inferred(self, loader, json_array_bytes):
        path = self._write_tmp(json_array_bytes, ".json")
        try:
            chunks = loader.load_structured_file(path)
            assert len(chunks) == 3  # 1 table + 2 columns
        finally:
            os.unlink(path)

    def test_name_override(self, loader, parquet_bytes):
        path = self._write_tmp(parquet_bytes, ".parquet")
        try:
            chunks = loader.load_structured_file(path, name="my_table")
            assert "my_table" in chunks[0].document_name
        finally:
            os.unlink(path)

    def test_all_chunks_enriched(self, loader, parquet_bytes):
        path = self._write_tmp(parquet_bytes, ".parquet")
        try:
            chunks = loader.load_structured_file(path)
            assert all(c.embedding_content is not None for c in chunks)
        finally:
            os.unlink(path)

    def test_unsupported_extension_raises(self, loader):
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_structured_file("/tmp/file.xyz")


# ---------------------------------------------------------------------------
# load() auto-dispatch
# ---------------------------------------------------------------------------

class TestLoadAutoDispatch:
    def _write_tmp(self, data: bytes, suffix: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.write(data)
        f.flush()
        f.close()
        return f.name

    def test_load_parquet_auto_dispatches(self, loader, parquet_bytes):
        path = self._write_tmp(parquet_bytes, ".parquet")
        try:
            chunks = loader.load(path)
            assert chunks[0].chunk_type == "db_table"
        finally:
            os.unlink(path)

    def test_load_csv_auto_dispatches(self, loader, csv_bytes):
        path = self._write_tmp(csv_bytes, ".csv")
        try:
            chunks = loader.load(path)
            assert chunks[0].chunk_type == "db_table"
        finally:
            os.unlink(path)

    def test_load_jsonl_auto_dispatches(self, loader, jsonl_bytes):
        path = self._write_tmp(jsonl_bytes, ".jsonl")
        try:
            chunks = loader.load(path)
            assert chunks[0].chunk_type == "db_table"
        finally:
            os.unlink(path)

    def test_load_arrow_auto_dispatches(self, loader, arrow_bytes):
        path = self._write_tmp(arrow_bytes, ".arrow")
        try:
            chunks = loader.load(path)
            assert chunks[0].chunk_type == "db_table"
        finally:
            os.unlink(path)

    def test_load_feather_auto_dispatches(self, loader, feather_bytes):
        path = self._write_tmp(feather_bytes, ".feather")
        try:
            chunks = loader.load(path)
            assert chunks[0].chunk_type == "db_table"
        finally:
            os.unlink(path)
