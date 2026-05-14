# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for chunk-level context graph: build_chunk_clusters and build_context_graph_edges."""

from __future__ import annotations

import duckdb
import pytest


def _setup_schema(conn):
    """Create all required tables in-memory."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT PRIMARY KEY,
            document_name TEXT NOT NULL,
            section TEXT,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            breadcrumb TEXT,
            chunk_type TEXT NOT NULL DEFAULT 'document',
            source_offset INTEGER,
            source_length INTEGER,
            namespace TEXT,
            embedding FLOAT[4]
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_entities (
            chunk_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            frequency INTEGER NOT NULL DEFAULT 1,
            positions_json TEXT NOT NULL DEFAULT '[]',
            score REAL NOT NULL DEFAULT 0.0,
            namespace TEXT,
            PRIMARY KEY (chunk_id, entity_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS svo_triples (
            chunk_id VARCHAR,
            subject_id VARCHAR NOT NULL,
            verb VARCHAR NOT NULL,
            object_id VARCHAR NOT NULL,
            confidence FLOAT NOT NULL DEFAULT 1.0,
            namespace VARCHAR,
            description TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_clusters (
            chunk_id TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            namespace TEXT NOT NULL DEFAULT 'global',
            PRIMARY KEY (chunk_id, namespace, cluster_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_graph_edges (
            source_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            namespace TEXT NOT NULL DEFAULT 'global',
            weight REAL NOT NULL,
            svo_signal REAL NOT NULL DEFAULT 0.0,
            cooccur_signal REAL NOT NULL DEFAULT 0.0,
            cluster_signal REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (source_entity_id, target_entity_id, namespace)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_graph_cache (
            namespace TEXT PRIMARY KEY,
            chunk_fingerprint TEXT NOT NULL,
            entity_count INTEGER NOT NULL,
            edge_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _insert_chunks(conn, chunk_ids, namespace="global"):
    for i, cid in enumerate(chunk_ids):
        conn.execute(
            "INSERT INTO embeddings (chunk_id, document_name, chunk_index, content, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            [cid, "doc.txt", i, f"content {cid}", namespace],
        )


def _insert_chunk_entities(conn, mapping, namespace="global"):
    """mapping: {chunk_id: [entity_id, ...]}"""
    for chunk_id, entity_ids in mapping.items():
        for eid in entity_ids:
            conn.execute(
                "INSERT INTO chunk_entities (chunk_id, entity_id, namespace) VALUES (?, ?, ?)",
                [chunk_id, eid, namespace],
            )


def _insert_svo(conn, subject_id, object_id, namespace="global", chunk_id="c1"):
    conn.execute(
        "INSERT INTO svo_triples (chunk_id, subject_id, verb, object_id, confidence, namespace) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [chunk_id, subject_id, "type_of", object_id, 1.0, namespace],
    )


class TestBuildChunkClusters:
    def test_below_min_chunks_returns_empty(self):
        from chonk.graph._context_graph import build_chunk_clusters

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        # Insert only 3 chunks (min_chunks default is 10)
        _insert_chunks(conn, ["c1", "c2", "c3"])
        _insert_chunk_entities(conn, {"c1": ["e1", "e2"], "c2": ["e2", "e3"], "c3": ["e1"]})

        result = build_chunk_clusters(conn, namespace="global", min_chunks=10, force=True)
        assert result == {}

    def test_produces_valid_cluster_assignments(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_chunk_clusters

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        chunk_ids = [f"c{i}" for i in range(12)]
        _insert_chunks(conn, chunk_ids)
        # e1+e2 share chunks c0-c5; e3+e4 share chunks c6-c11
        mapping = {f"c{i}": ["e1", "e2"] for i in range(6)}
        mapping.update({f"c{i}": ["e3", "e4"] for i in range(6, 12)})
        _insert_chunk_entities(conn, mapping)

        result = build_chunk_clusters(conn, namespace="global", min_chunks=10, force=True)
        assert len(result) == 12
        for chunk_id, cluster_id in result.items():
            assert isinstance(cluster_id, int)

    def test_cache_hit_skips_rebuild(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_chunk_clusters

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        chunk_ids = [f"c{i}" for i in range(12)]
        _insert_chunks(conn, chunk_ids)
        mapping = {f"c{i}": ["e1", "e2"] for i in range(12)}
        _insert_chunk_entities(conn, mapping)

        # First build — populates cache
        result1 = build_chunk_clusters(conn, namespace="global", min_chunks=10, force=True)
        # Second build with force=False — should use cache
        result2 = build_chunk_clusters(conn, namespace="global", min_chunks=10, force=False)
        assert result1 == result2

    def test_force_true_rebuilds(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_chunk_clusters

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        chunk_ids = [f"c{i}" for i in range(12)]
        _insert_chunks(conn, chunk_ids)
        mapping = {f"c{i}": ["e1", "e2"] for i in range(12)}
        _insert_chunk_entities(conn, mapping)

        build_chunk_clusters(conn, namespace="global", min_chunks=10, force=True)
        # force=True should not raise and should return valid result
        result = build_chunk_clusters(conn, namespace="global", min_chunks=10, force=True)
        assert isinstance(result, dict)
        assert len(result) == 12


class TestBuildContextGraphEdges:
    def _setup_two_entity_pair(self, conn, n_shared_chunks=2, namespace="global"):
        """e1 and e2 share n_shared_chunks chunks. Total 12 chunks."""
        chunk_ids = [f"c{i}" for i in range(12)]
        _insert_chunks(conn, chunk_ids, namespace=namespace)
        mapping = {}
        for i in range(n_shared_chunks):
            mapping[f"c{i}"] = ["e1", "e2"]
        for i in range(n_shared_chunks, 12):
            mapping[f"c{i}"] = ["e1"]
        _insert_chunk_entities(conn, mapping, namespace=namespace)

    def test_no_svo_triples_svo_signal_zero(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._setup_two_entity_pair(conn, n_shared_chunks=3)

        stats = build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        rows = conn.execute(
            "SELECT svo_signal FROM context_graph_edges WHERE namespace = 'global'"
        ).fetchall()
        assert len(rows) > 0
        for (svo_signal,) in rows:
            assert svo_signal == 0.0

    def test_svo_signal_set_when_triple_exists(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._setup_two_entity_pair(conn, n_shared_chunks=3)
        _insert_svo(conn, "e1", "e2", namespace="global")

        build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        rows = conn.execute(
            "SELECT svo_signal FROM context_graph_edges "
            "WHERE source_entity_id = 'e1' AND target_entity_id = 'e2'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 1.0

    def test_cooccur_signal_capped_at_0_8(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        # 10 shared chunks — cooccur signal would be 10*0.8 uncapped, capped at 0.8
        self._setup_two_entity_pair(conn, n_shared_chunks=10)

        build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        rows = conn.execute(
            "SELECT cooccur_signal FROM context_graph_edges WHERE namespace = 'global'"
        ).fetchall()
        assert len(rows) > 0
        for (cooccur_signal,) in rows:
            # DuckDB REAL is 32-bit; 0.8 round-trips as ~0.8000000119
            assert cooccur_signal <= 0.8 + 1e-6

    def test_edge_symmetry(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._setup_two_entity_pair(conn, n_shared_chunks=3)
        _insert_svo(conn, "e1", "e2")

        build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        e1_to_e2 = conn.execute(
            "SELECT weight FROM context_graph_edges "
            "WHERE source_entity_id = 'e1' AND target_entity_id = 'e2'"
        ).fetchone()
        e2_to_e1 = conn.execute(
            "SELECT weight FROM context_graph_edges "
            "WHERE source_entity_id = 'e2' AND target_entity_id = 'e1'"
        ).fetchone()
        assert e1_to_e2 is not None
        assert e2_to_e1 is not None
        assert abs(e1_to_e2[0] - e2_to_e1[0]) < 1e-9

    def test_min_weight_filters_edges(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        # e1+e2 share chunks, e3 alone
        chunk_ids = [f"c{i}" for i in range(12)]
        _insert_chunks(conn, chunk_ids)
        mapping = {f"c{i}": ["e1", "e2"] for i in range(6)}
        mapping.update({f"c{i}": ["e3"] for i in range(6, 12)})
        _insert_chunk_entities(conn, mapping)

        build_context_graph_edges(conn, namespace="global", min_weight=0.9, force=True)
        rows = conn.execute(
            "SELECT weight FROM context_graph_edges WHERE namespace = 'global'"
        ).fetchall()
        for (w,) in rows:
            assert w >= 0.9

    def test_normalized_weight_in_0_1(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._setup_two_entity_pair(conn, n_shared_chunks=5)
        _insert_svo(conn, "e1", "e2")

        build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        rows = conn.execute(
            "SELECT weight FROM context_graph_edges WHERE namespace = 'global'"
        ).fetchall()
        for (w,) in rows:
            assert 0.0 <= w <= 1.0 + 1e-9

    def test_stats_returned(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import ContextGraphStats, build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._setup_two_entity_pair(conn, n_shared_chunks=3)

        stats = build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        assert isinstance(stats, ContextGraphStats)
        assert stats.edge_count >= 0
        assert stats.entity_count >= 0

    def test_force_false_uses_cache(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._setup_two_entity_pair(conn, n_shared_chunks=3)

        stats1 = build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        stats2 = build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=False)
        # Second call returns same edge_count from cache
        assert stats2.edge_count == stats1.edge_count

    def test_force_true_bypasses_cache(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import ContextGraphStats, build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._setup_two_entity_pair(conn, n_shared_chunks=3)

        build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        # Add more data after first build
        conn.execute(
            "INSERT INTO embeddings (chunk_id, document_name, chunk_index, content, namespace) "
            "VALUES ('c99', 'doc.txt', 99, 'extra', 'global')"
        )
        conn.execute(
            "INSERT INTO chunk_entities (chunk_id, entity_id, namespace) VALUES ('c99', 'e1', 'global')"
        )
        conn.execute(
            "INSERT INTO chunk_entities (chunk_id, entity_id, namespace) VALUES ('c99', 'e2', 'global')"
        )
        # force=True must rebuild (should not raise)
        stats = build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        assert isinstance(stats, ContextGraphStats)

    def test_cluster_signal_between_zero_and_0_4(self):
        pytest.importorskip("sklearn")
        from chonk.graph._context_graph import build_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._setup_two_entity_pair(conn, n_shared_chunks=3)

        build_context_graph_edges(conn, namespace="global", min_weight=0.0, force=True)
        rows = conn.execute(
            "SELECT cluster_signal FROM context_graph_edges WHERE namespace = 'global'"
        ).fetchall()
        for (cs,) in rows:
            assert 0.0 <= cs <= 0.4 + 1e-9


class TestGetContextGraph:
    def _build_graph(self, conn):
        conn.execute(
            "INSERT INTO context_graph_edges "
            "(source_entity_id, target_entity_id, namespace, weight, svo_signal, cooccur_signal, cluster_signal) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["e1", "e2", "global", 0.9, 1.0, 0.8, 0.2],
        )
        conn.execute(
            "INSERT INTO context_graph_edges "
            "(source_entity_id, target_entity_id, namespace, weight, svo_signal, cooccur_signal, cluster_signal) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["e1", "e3", "global", 0.5, 0.0, 0.5, 0.1],
        )
        conn.execute(
            "INSERT INTO context_graph_edges "
            "(source_entity_id, target_entity_id, namespace, weight, svo_signal, cooccur_signal, cluster_signal) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["e2", "e1", "global", 0.9, 1.0, 0.8, 0.2],
        )

    def test_get_context_graph_sorted_by_weight_desc(self):
        from chonk.graph._context_graph import get_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._build_graph(conn)

        edges = get_context_graph_edges(conn, "e1", namespace="global", min_weight=0.0)
        assert len(edges) == 2
        assert edges[0].weight >= edges[1].weight

    def test_get_context_graph_filters_by_source(self):
        from chonk.graph._context_graph import get_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._build_graph(conn)

        edges = get_context_graph_edges(conn, "e1", namespace="global", min_weight=0.0)
        for edge in edges:
            assert edge.source_entity_id == "e1"

    def test_get_context_graph_min_weight_filter(self):
        from chonk.graph._context_graph import get_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._build_graph(conn)

        edges = get_context_graph_edges(conn, "e1", namespace="global", min_weight=0.8)
        assert all(e.weight >= 0.8 for e in edges)

    def test_get_context_graph_returns_context_edge(self):
        from chonk.graph._context_graph import ContextEdge, get_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        self._build_graph(conn)

        edges = get_context_graph_edges(conn, "e1", namespace="global", min_weight=0.0)
        assert all(isinstance(e, ContextEdge) for e in edges)
        assert edges[0].svo_signal == 1.0
        # DuckDB REAL is 32-bit; 0.8 round-trips as ~0.8000000119
        assert abs(edges[0].cooccur_signal - 0.8) < 1e-6

    def test_lazy_miss_returns_empty_and_logs(self, caplog):
        import logging

        from chonk.graph._context_graph import get_context_graph_edges

        conn = duckdb.connect(":memory:")
        _setup_schema(conn)
        # No edges inserted — graph never built

        with caplog.at_level(logging.DEBUG, logger="chonk.graph._context_graph"):
            edges = get_context_graph_edges(conn, "e1", namespace="global")

        assert edges == []
        assert any("context graph not built" in r.message for r in caplog.records)

    def test_lazy_miss_returns_empty_when_table_absent(self):
        from chonk.graph._context_graph import get_context_graph_edges

        conn = duckdb.connect(":memory:")
        # No schema at all
        edges = get_context_graph_edges(conn, "e1", namespace="global")
        assert edges == []


class TestStoreContextGraph:
    def test_store_build_and_get_context_graph(self):
        pytest.importorskip("sklearn")
        from chonk.storage._store import Store

        with Store(":memory:") as store:
            conn = store._db.conn
            _setup_schema(conn)
            chunk_ids = [f"c{i}" for i in range(12)]
            _insert_chunks(conn, chunk_ids)
            mapping = {f"c{i}": ["e1", "e2"] for i in range(6)}
            mapping.update({f"c{i}": ["e3", "e4"] for i in range(6, 12)})
            _insert_chunk_entities(conn, mapping)

            stats = store.build_context_graph(namespace="global", min_weight=0.0, force=True)
            from chonk.graph._context_graph import ContextGraphStats
            assert isinstance(stats, ContextGraphStats)

            edges = store.get_context_graph("e1", namespace="global", min_weight=0.0)
            for e in edges:
                assert e.source_entity_id == "e1"
