# Copyright (c) 2025 Kenneth Stott. MIT License.
"""Unit tests for EntityGraphPipeline."""

from __future__ import annotations

import json
from pathlib import Path

from chonk.graph import (
    PHASE_EXTRACT,
    PHASE_LOAD,
    PHASE_PERSIST_TRIPLES,
    EntityGraphPipeline,
    EntityGraphStats,
    SVOExtractor,
)
from chonk.graph._llm import LLMClient


class StubLLM(LLMClient):
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, prompt: str) -> str:
        return self._payload


def _ea_response(triples=None, descriptions=None, aliases=None):
    return json.dumps(
        {
            "triples": triples or [],
            "descriptions": descriptions or {},
            "aliases": aliases or {},
        }
    )


def _make_store(tmp_path: Path):
    from chonk.storage._store import Store

    return Store(tmp_path / "t.duckdb")


def _seed_store(store, entity_ids: list[str], chunk_content: str = "some text"):
    """Insert one chunk and entities with chunk_entities join rows."""
    conn = store._db.conn
    conn.execute(
        "INSERT INTO embeddings(chunk_id, document_name, chunk_index, content, chunk_type)"
        " VALUES (?, ?, ?, ?, ?)",
        ["chunk1", "doc1", 0, chunk_content, "document"],
    )
    for eid in entity_ids:
        conn.execute(
            "INSERT OR IGNORE INTO entities(id, name, display_name, entity_type)"
            " VALUES (?, ?, ?, ?)",
            [eid, eid, eid, "concept"],
        )
    for eid in entity_ids:
        conn.execute(
            "INSERT OR IGNORE INTO chunk_entities(chunk_id, entity_id) VALUES (?, ?)",
            ["chunk1", eid],
        )


class TestEntityGraphPipelineBasic:
    def test_returns_stats_dataclass(self, tmp_path):
        llm = StubLLM(_ea_response())
        pipeline = EntityGraphPipeline(SVOExtractor(llm))
        with _make_store(tmp_path) as store:
            _seed_store(store, ["EntityA", "EntityB"])
            stats = pipeline.build(store, force=True)
        assert isinstance(stats, EntityGraphStats)

    def test_skips_chunks_with_fewer_than_two_entities(self, tmp_path):
        llm = StubLLM(_ea_response())
        pipeline = EntityGraphPipeline(SVOExtractor(llm))
        with _make_store(tmp_path) as store:
            _seed_store(store, ["EntityA"])  # only one entity
            stats = pipeline.build(store, force=True)
        assert stats.chunks_skipped == 1
        assert stats.chunks_processed == 0

    def test_processes_chunk_with_two_entities(self, tmp_path):
        payload = _ea_response(
            descriptions={"EntityA": "Desc for A"},
            aliases={"EntityA": ["EA"]},
        )
        llm = StubLLM(payload)
        pipeline = EntityGraphPipeline(SVOExtractor(llm))
        with _make_store(tmp_path) as store:
            _seed_store(store, ["EntityA", "EntityB"])
            stats = pipeline.build(store, force=True)
        assert stats.chunks_processed == 1

    def test_descriptions_persisted(self, tmp_path):
        payload = _ea_response(descriptions={"EntityA": "A is a concept"})
        llm = StubLLM(payload)
        pipeline = EntityGraphPipeline(SVOExtractor(llm))
        with _make_store(tmp_path) as store:
            _seed_store(store, ["EntityA", "EntityB"])
            stats = pipeline.build(store, force=True)
            descs = store.get_entity_descriptions(["EntityA"])
        assert descs["EntityA"] == "A is a concept"
        assert stats.descriptions_written >= 1

    def test_aliases_persisted(self, tmp_path):
        payload = _ea_response(aliases={"EntityA": ["EA", "Alpha"]})
        llm = StubLLM(payload)
        pipeline = EntityGraphPipeline(SVOExtractor(llm))
        with _make_store(tmp_path) as store:
            _seed_store(store, ["EntityA", "EntityB"])
            stats = pipeline.build(store, force=True)
            aliases = store.get_entity_aliases("EntityA")
        assert "EA" in aliases
        assert "Alpha" in aliases
        assert stats.aliases_written >= 2

    def test_force_false_skips_when_triples_exist(self, tmp_path):
        from chonk.graph import RelationshipIndex, SVOTriple

        llm = StubLLM(_ea_response())
        pipeline = EntityGraphPipeline(SVOExtractor(llm))
        with _make_store(tmp_path) as store:
            _seed_store(store, ["EntityA", "EntityB"])
            # Pre-populate svo_triples via RelationshipIndex so the table exists
            ri = RelationshipIndex()
            ri.add(SVOTriple("EntityA", "part_of", "EntityB", 0.9))
            ri.save_to_db(store._db.conn)
            stats = pipeline.build(store, force=False)
        # Should have returned early without processing
        assert stats.chunks_processed == 0


class TestEntityGraphPipelineProgress:
    def test_progress_callback_fires(self, tmp_path):
        payload = _ea_response(descriptions={"EntityA": "desc"})
        llm = StubLLM(payload)
        pipeline = EntityGraphPipeline(SVOExtractor(llm))
        calls: list[tuple] = []

        def on_progress(phase, done, total):
            calls.append((phase, done, total))

        with _make_store(tmp_path) as store:
            _seed_store(store, ["EntityA", "EntityB"])
            pipeline.build(store, progress=on_progress, force=True)

        phases_seen = {c[0] for c in calls}
        assert PHASE_LOAD in phases_seen
        assert PHASE_EXTRACT in phases_seen
        assert PHASE_PERSIST_TRIPLES in phases_seen

    def test_progress_extract_done_equals_total(self, tmp_path):
        llm = StubLLM(_ea_response())
        pipeline = EntityGraphPipeline(SVOExtractor(llm))
        extract_calls: list[tuple] = []

        def on_progress(phase, done, total):
            if phase == PHASE_EXTRACT:
                extract_calls.append((done, total))

        with _make_store(tmp_path) as store:
            _seed_store(store, ["EntityA", "EntityB"])
            pipeline.build(store, progress=on_progress, force=True)

        assert extract_calls, "no PHASE_EXTRACT callbacks fired"
        final = extract_calls[-1]
        assert final[0] == final[1]  # done == total at end
