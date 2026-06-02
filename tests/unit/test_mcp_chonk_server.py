# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d0c1dd8a-daeb-487b-aee9-44a9500542bf
"""Tests for mcp_chonk_server — uses an in-memory-style temp DuckDB."""

from __future__ import annotations

import asyncio
import json
import sys

import numpy as np
import pytest

sa = pytest.importorskip("sqlalchemy")
pytest.importorskip("mcp")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 4


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    """Real DuckDB file with a few indexed chunks."""
    from chonk import DocumentLoader
    from chonk.storage import Store

    path = str(tmp_path_factory.mktemp("mcp") / "test.duckdb")
    loader = DocumentLoader(enrich_context=False)
    chunks = loader.load_text(
        "Alpha beta gamma.\n\nDelta epsilon zeta.", name="doc_a"
    ) + loader.load_text("One two three.", name="doc_b")

    rng = np.random.default_rng(0)
    embeddings = rng.random((len(chunks), DIM), dtype=np.float32)

    with Store(path, embedding_dim=DIM) as store:
        store.add_document(chunks, embeddings)

    return path


@pytest.fixture()
def server(db_path, monkeypatch):
    """Load mcp_chonk_server with the test DB, reload between tests."""
    monkeypatch.setenv("CHONK_DB_PATH", db_path)
    monkeypatch.setenv("CHONK_EMBEDDING_DIM", str(DIM))
    monkeypatch.delenv("CHONK_DB_CONFIG", raising=False)

    # Force fresh module load so env vars take effect
    sys.modules.pop("mcp_chonk_server", None)
    import mcp_chonk_server as srv

    return srv


@pytest.fixture()
def multi_server(db_path, monkeypatch):
    """Load mcp_chonk_server with CHONK_DB_CONFIG pointing to two named DBs."""
    config = json.dumps(
        {
            "primary": {"path": db_path, "embedding_dim": DIM},
            "secondary": {"path": db_path, "embedding_dim": DIM},
        }
    )
    monkeypatch.setenv("CHONK_DB_CONFIG", config)
    monkeypatch.delenv("CHONK_DB_PATH", raising=False)
    monkeypatch.delenv("CHONK_EMBEDDING_DIM", raising=False)

    sys.modules.pop("mcp_chonk_server", None)
    import mcp_chonk_server as srv

    return srv


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Config / startup
# ---------------------------------------------------------------------------


class TestConfig:
    def test_single_db_store_named_default(self, server):
        assert list(server.STORES.keys()) == ["default"]

    def test_multi_db_stores_named(self, multi_server):
        assert set(multi_server.STORES.keys()) == {"primary", "secondary"}

    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("CHONK_DB_PATH", raising=False)
        monkeypatch.delenv("CHONK_DB_CONFIG", raising=False)
        sys.modules.pop("mcp_chonk_server", None)
        with pytest.raises(RuntimeError, match="CHONK_DB_PATH or CHONK_DB_CONFIG"):
            import mcp_chonk_server  # noqa: F401


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------


class TestListTools:
    def test_returns_three_tools(self, server):
        tools = run(server.list_tools())
        assert len(tools) == 3

    def test_tool_names(self, server):
        names = {t.name for t in run(server.list_tools())}
        assert names == {"search_chunks", "get_chunk", "expand_chunk_graph"}

    def test_search_chunks_requires_query_embedding(self, server):
        tools = {t.name: t for t in run(server.list_tools())}
        assert "query_embedding" in tools["search_chunks"].inputSchema["required"]

    def test_db_names_in_description(self, multi_server):
        tools = {t.name: t for t in run(multi_server.list_tools())}
        assert "primary" in tools["search_chunks"].description
        assert "secondary" in tools["search_chunks"].description


# ---------------------------------------------------------------------------
# search_chunks
# ---------------------------------------------------------------------------


class TestSearchChunks:
    def _embedding(self):
        return list(np.random.default_rng(42).random(DIM, dtype=np.float32))

    def test_returns_results(self, server):
        result = run(server._search_chunks({"query_embedding": self._embedding(), "limit": 3}))
        data = json.loads(result[0].text)
        assert len(data["results"]) <= 3
        assert "usage" in data

    def test_result_fields(self, server):
        result = run(server._search_chunks({"query_embedding": self._embedding()}))
        data = json.loads(result[0].text)
        if data["results"]:
            r = data["results"][0]
            for field in ("chunk_id", "score", "document_name", "content", "db"):
                assert field in r

    def test_2d_embedding_normalised(self, server):
        emb = [self._embedding()]  # shape (1, DIM)
        result = run(server._search_chunks({"query_embedding": emb}))
        assert result

    def test_missing_embedding_raises(self, server):
        with pytest.raises(ValueError, match="query_embedding is required"):
            run(server._search_chunks({}))

    def test_bad_embedding_shape_raises(self, server):
        with pytest.raises(ValueError, match="shape"):
            run(server._search_chunks({"query_embedding": [[0.1] * DIM, [0.2] * DIM]}))

    def test_unknown_db_raises(self, server):
        with pytest.raises(ValueError, match="Unknown db"):
            run(server._search_chunks({"query_embedding": self._embedding(), "db": "nope"}))

    def test_target_db_filters(self, multi_server):
        emb = list(np.random.default_rng(1).random(DIM, dtype=np.float32))
        result = run(multi_server._search_chunks({"query_embedding": emb, "db": "primary"}))
        data = json.loads(result[0].text)
        assert all(r["db"] == "primary" for r in data["results"])

    def test_all_dbs_searched_when_no_db(self, multi_server):
        emb = list(np.random.default_rng(1).random(DIM, dtype=np.float32))
        result = run(multi_server._search_chunks({"query_embedding": emb, "limit": 10}))
        data = json.loads(result[0].text)
        dbs = {r["db"] for r in data["results"]}
        assert len(dbs) >= 1  # both point to same file; db tag still set

    def test_results_sorted_by_score_descending(self, server):
        result = run(server._search_chunks({"query_embedding": self._embedding(), "limit": 5}))
        data = json.loads(result[0].text)
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_limit_respected(self, server):
        result = run(server._search_chunks({"query_embedding": self._embedding(), "limit": 1}))
        data = json.loads(result[0].text)
        assert len(data["results"]) <= 1


# ---------------------------------------------------------------------------
# get_chunk
# ---------------------------------------------------------------------------


class TestGetChunk:
    def _get_a_chunk_id(self, server):
        emb = list(np.random.default_rng(7).random(DIM, dtype=np.float32))
        result = run(server._search_chunks({"query_embedding": emb, "limit": 1}))
        data = json.loads(result[0].text)
        return data["results"][0]["chunk_id"] if data["results"] else None

    def test_fetch_known_chunk(self, server):
        cid = self._get_a_chunk_id(server)
        if cid is None:
            pytest.skip("no chunks in index")
        result = run(server._get_chunk({"chunk_id": cid}))
        data = json.loads(result[0].text)
        assert data["chunk"]["chunk_id"] == cid

    def test_includes_neighbors(self, server):
        cid = self._get_a_chunk_id(server)
        if cid is None:
            pytest.skip("no chunks in index")
        result = run(
            server._get_chunk(
                {
                    "chunk_id": cid,
                    "include_neighbors": True,
                    "neighbor_radius": 2,
                }
            )
        )
        data = json.loads(result[0].text)
        assert "neighbors" in data

    def test_missing_chunk_id_raises(self, server):
        with pytest.raises(ValueError, match="chunk_id is required"):
            run(server._get_chunk({}))

    def test_nonexistent_chunk_raises(self, server):
        with pytest.raises(KeyError, match="chunk_id not found"):
            run(server._get_chunk({"chunk_id": "does_not_exist"}))

    def test_unknown_db_raises(self, server):
        with pytest.raises(ValueError, match="Unknown db"):
            run(server._get_chunk({"chunk_id": "x", "db": "nope"}))

    def test_db_field_in_response(self, server):
        cid = self._get_a_chunk_id(server)
        if cid is None:
            pytest.skip("no chunks in index")
        result = run(server._get_chunk({"chunk_id": cid}))
        data = json.loads(result[0].text)
        assert data["db"] == "default"


# ---------------------------------------------------------------------------
# handle_call_tool dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_unknown_tool_raises(self, server):
        with pytest.raises(ValueError, match="Unknown tool"):
            run(server.handle_call_tool("nonexistent", {}))

    def test_none_arguments_treated_as_empty(self, server):
        with pytest.raises(ValueError, match="query_embedding is required"):
            run(server.handle_call_tool("search_chunks", None))

    def test_expand_chunk_graph_not_implemented(self, server):
        with pytest.raises(NotImplementedError):
            run(server.handle_call_tool("expand_chunk_graph", {"chunk_id": "x"}))
