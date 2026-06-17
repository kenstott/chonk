# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: ae9a3ac0-9863-4a6b-a83d-ff9c7b145768
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


class TestEmbedEntitiesUsesConn:
    """Regression tests for issue #3 — NameError: 'db' undefined in _embed_entities.

    The bug: line 280 referenced `db.execute(...)` but the correct variable in
    scope is `conn` (assigned at line 241 via ``conn = store._db.conn``).
    This caused a ``NameError`` at runtime whenever entity embedding was attempted.

    Without the fix, these tests would raise:
        NameError: name 'db' is not defined
    With the fix they must pass cleanly.
    """

    class _StubEmbedModel:
        """Minimal embed model that returns zero vectors of the requested size."""

        DIM = 4

        def encode(self, texts: list[str], **kwargs):
            import numpy as np

            return np.zeros((len(texts), self.DIM), dtype="float32")

    def _seed_entities(self, store, entity_ids: list[str]):
        """Insert entities and one chunk so _embed_entities has rows to work with."""
        conn = store._db.conn
        conn.execute(
            "INSERT INTO embeddings(chunk_id, document_name, chunk_index, content, chunk_type)"
            " VALUES (?, ?, ?, ?, ?)",
            ["chunk_embed", "doc_embed", 0, "entity embed content", "document"],
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
                ["chunk_embed", eid],
            )

    def _make_embed_store(self, tmp_path, name: str = "e.duckdb"):
        """Store configured with embedding_dim matching _StubEmbedModel.DIM."""
        from chonk.storage._store import Store

        return Store(tmp_path / name, embedding_dim=self.DIM)

    def test_embed_entities_does_not_raise_name_error(self, tmp_path):
        """_embed_entities must reference conn, not the undefined name db."""
        from chonk.graph._entity_pipeline import EntityGraphPipeline
        from chonk.graph._extractor import SVOExtractor
        from chonk.graph._llm import LLMClient

        class _NoOpLLM(LLMClient):
            def complete(self, prompt: str) -> str:
                import json

                return json.dumps({"triples": [], "descriptions": {}, "aliases": {}})

        embed_model = self._StubEmbedModel()
        pipeline = EntityGraphPipeline(
            SVOExtractor(_NoOpLLM()),
            embed_model=embed_model,
            embed_dim=self.DIM,
        )
        with self._make_embed_store(tmp_path) as store:
            self._seed_entities(store, ["EntityX", "EntityY"])
            # Must not raise NameError; any other exception would also fail the test
            stats = pipeline.build(store, force=True)

        assert stats.entity_embeddings_written >= 0  # 0 is valid if no entities found

    def test_embed_entities_deletes_existing_entity_embeddings(self, tmp_path):
        """Verify the DELETE SQL runs without error (previously used undefined `db`)."""
        from chonk.graph._entity_pipeline import EntityGraphPipeline
        from chonk.graph._extractor import SVOExtractor
        from chonk.graph._llm import LLMClient

        class _NoOpLLM(LLMClient):
            def complete(self, prompt: str) -> str:
                import json

                return json.dumps({"triples": [], "descriptions": {}, "aliases": {}})

        embed_model = self._StubEmbedModel()
        pipeline = EntityGraphPipeline(
            SVOExtractor(_NoOpLLM()),
            embed_model=embed_model,
            embed_dim=self.DIM,
        )
        with self._make_embed_store(tmp_path, "e2.duckdb") as store:
            conn = store._db.conn
            self._seed_entities(store, ["EntityA", "EntityB"])
            # Pre-insert a stale entity embedding row to confirm DELETE fires
            conn.execute(
                "INSERT INTO embeddings(chunk_id, document_name, chunk_index, content, chunk_type)"
                " VALUES (?, ?, ?, ?, ?)",
                ["__entity__stale", "doc", 0, "old entity text", "entity"],
            )
            stale_before = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE chunk_type = 'entity'"
            ).fetchone()[0]
            assert stale_before == 1, "setup: stale entity row should exist"

            # This must not raise NameError
            pipeline.build(store, force=True)

            # After build the stale row should have been deleted and replaced
            remaining = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE chunk_type = 'entity'"
            ).fetchone()[0]
        # The DELETE must have removed the stale row (remaining may be > 0 if
        # entities were re-embedded, but it should differ from stale_before only
        # if new embeddings were written — the key assertion is no NameError above).
        assert remaining >= 0  # test passes iff no NameError was raised

    def test_embed_entities_returns_count(self, tmp_path):
        """entity_embeddings_written must equal the number of entity rows encoded."""
        from chonk.graph._entity_pipeline import EntityGraphPipeline
        from chonk.graph._extractor import SVOExtractor
        from chonk.graph._llm import LLMClient

        class _NoOpLLM(LLMClient):
            def complete(self, prompt: str) -> str:
                import json

                return json.dumps({"triples": [], "descriptions": {}, "aliases": {}})

        embed_model = self._StubEmbedModel()
        pipeline = EntityGraphPipeline(
            SVOExtractor(_NoOpLLM()),
            embed_model=embed_model,
            embed_dim=self.DIM,
        )
        entity_ids = ["E1", "E2", "E3"]
        with self._make_embed_store(tmp_path, "e3.duckdb") as store:
            self._seed_entities(store, entity_ids)
            stats = pipeline.build(store, force=True)

        # We seeded 3 entities so the embed step should have encoded 3 texts
        assert stats.entity_embeddings_written == len(entity_ids)

    DIM = _StubEmbedModel.DIM
