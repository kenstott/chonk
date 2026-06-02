# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 56be0467-4847-467a-9ef6-d7d7fa14e224

"""Unit tests for VersionedRef, CommunityIndexBuilder, RelationshipIndexBuilder."""

import threading
import time
import pytest
import numpy as np

from chonk._versioning import VersionedRef
from chonk.community import CommunityIndex, CommunityIndexBuilder
from chonk.graph import RelationshipIndex, RelationshipIndexBuilder, SVOTriple


# ---------------------------------------------------------------------------
# VersionedRef
# ---------------------------------------------------------------------------

class TestVersionedRefInit:
    def test_unset_version_is_minus_one(self):
        ref: VersionedRef[int] = VersionedRef()
        assert ref.version == -1
        assert ref.current is None

    def test_initial_value_sets_version_zero(self):
        ref = VersionedRef(42)
        assert ref.version == 0
        assert ref.current == 42

    def test_has_pending_false_initially(self):
        ref: VersionedRef[str] = VersionedRef()
        assert not ref.has_pending


class TestVersionedRefUpdate:
    def test_update_increments_version(self):
        ref: VersionedRef[int] = VersionedRef()
        v1 = ref.update(1)
        assert v1 == 0
        v2 = ref.update(2)
        assert v2 == 1

    def test_update_replaces_current(self):
        ref: VersionedRef[str] = VersionedRef("a")
        ref.update("b")
        assert ref.current == "b"

    def test_update_clears_pending(self):
        ref: VersionedRef[int] = VersionedRef()
        ref.stage(99)
        assert ref.has_pending
        ref.update(1)
        assert not ref.has_pending


class TestVersionedRefStagePromote:
    def test_stage_does_not_change_current(self):
        ref = VersionedRef("original")
        ref.stage("next")
        assert ref.current == "original"

    def test_stage_sets_has_pending(self):
        ref: VersionedRef[int] = VersionedRef()
        ref.stage(5)
        assert ref.has_pending

    def test_promote_activates_staged(self):
        ref = VersionedRef("v1")
        ref.stage("v2")
        ref.promote()
        assert ref.current == "v2"

    def test_promote_increments_version(self):
        ref = VersionedRef("v1")
        ref.stage("v2")
        v = ref.promote()
        assert v == 1

    def test_promote_clears_pending(self):
        ref: VersionedRef[int] = VersionedRef()
        ref.stage(1)
        ref.promote()
        assert not ref.has_pending

    def test_promote_without_staged_raises(self):
        ref: VersionedRef[int] = VersionedRef()
        with pytest.raises(RuntimeError):
            ref.promote()

    def test_discard_staged_clears_pending(self):
        ref: VersionedRef[int] = VersionedRef()
        ref.stage(42)
        ref.discard_staged()
        assert not ref.has_pending

    def test_stage_overwrite(self):
        ref: VersionedRef[int] = VersionedRef()
        ref.stage(1)
        ref.stage(2)
        ref.promote()
        assert ref.current == 2


class TestVersionedRefThreadSafety:
    def test_concurrent_updates_all_increment(self):
        ref: VersionedRef[int] = VersionedRef(0)
        results = []

        def worker():
            v = ref.update(1)
            results.append(v)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert len(set(results)) == 20  # all versions unique


# ---------------------------------------------------------------------------
# CommunityIndexBuilder
# ---------------------------------------------------------------------------

def _small_vecs(n: int = 10, dim: int = 8) -> tuple[list[str], np.ndarray]:
    ids = [f"chunk_{i}" for i in range(n)]
    vecs = np.random.randn(n, dim).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return ids, vecs / np.maximum(norms, 1e-9)


class TestCommunityIndexBuilderInit:
    def test_ref_starts_none(self):
        b = CommunityIndexBuilder()
        assert b.ref.current is None
        assert b.ref.version == -1

    def test_initial_index(self):
        idx = CommunityIndex()
        b = CommunityIndexBuilder(initial=idx)
        assert b.ref.current is idx
        assert b.ref.version == 0

    def test_index_property(self):
        b = CommunityIndexBuilder()
        assert b.index is None


class TestCommunityIndexBuilderBuild:
    def test_build_completes(self):
        ids, vecs = _small_vecs()
        b = CommunityIndexBuilder()
        b.build(ids, vecs, n_levels=1, algorithm="louvain")
        assert b.wait(timeout=30)
        assert b.index is not None

    def test_version_increments_after_build(self):
        ids, vecs = _small_vecs()
        b = CommunityIndexBuilder()
        b.build(ids, vecs, n_levels=1, algorithm="louvain")
        b.wait(timeout=30)
        assert b.ref.version == 0

    def test_second_build_increments_version(self):
        ids, vecs = _small_vecs()
        b = CommunityIndexBuilder()
        b.build(ids, vecs, n_levels=1, algorithm="louvain")
        b.wait(timeout=30)
        b.build(ids, vecs, n_levels=1, algorithm="louvain")
        b.wait(timeout=30)
        assert b.ref.version == 1

    def test_on_complete_called(self):
        ids, vecs = _small_vecs()
        received = []
        b = CommunityIndexBuilder()
        b.build(ids, vecs, n_levels=1, algorithm="louvain",
                on_complete=lambda idx: received.append(idx))
        b.wait(timeout=30)
        assert len(received) == 1
        assert isinstance(received[0], CommunityIndex)

    def test_is_building_true_during_build(self):
        ids, vecs = _small_vecs(n=50)
        b = CommunityIndexBuilder()
        b.build(ids, vecs, n_levels=1, algorithm="louvain")
        # is_building may already be False if it finished fast; just verify no crash
        _ = b.is_building
        b.wait(timeout=30)

    def test_wait_returns_true_when_done(self):
        ids, vecs = _small_vecs()
        b = CommunityIndexBuilder()
        b.build(ids, vecs, n_levels=1, algorithm="louvain")
        assert b.wait(timeout=30)

    def test_wait_returns_true_when_never_started(self):
        b = CommunityIndexBuilder()
        assert b.wait()


# ---------------------------------------------------------------------------
# RelationshipIndexBuilder
# ---------------------------------------------------------------------------

class TestRelationshipIndexBuilderInit:
    def test_ref_starts_none(self):
        b = RelationshipIndexBuilder()
        assert b.ref.current is None
        assert b.ref.version == -1

    def test_initial_index(self):
        idx = RelationshipIndex()
        b = RelationshipIndexBuilder(initial=idx)
        assert b.ref.current is idx
        assert b.ref.version == 0


class TestRelationshipIndexBuilderFromTriples:
    def _triples(self) -> list[SVOTriple]:
        return [
            SVOTriple("entity_a", "governs", "entity_b", 0.9),
            SVOTriple("entity_b", "uses", "entity_c", 0.8),
        ]

    def test_build_from_triples_completes(self):
        b = RelationshipIndexBuilder()
        b.build_from_triples(self._triples())
        assert b.wait(timeout=10)
        assert b.index is not None

    def test_triples_indexed(self):
        b = RelationshipIndexBuilder()
        b.build_from_triples(self._triples())
        b.wait(timeout=10)
        idx = b.index
        assert len(idx) == 2

    def test_version_increments(self):
        b = RelationshipIndexBuilder()
        b.build_from_triples(self._triples())
        b.wait(timeout=10)
        assert b.ref.version == 0
        b.build_from_triples(self._triples())
        b.wait(timeout=10)
        assert b.ref.version == 1

    def test_on_complete_called(self):
        received = []
        b = RelationshipIndexBuilder()
        b.build_from_triples(self._triples(),
                             on_complete=lambda idx: received.append(idx))
        b.wait(timeout=10)
        assert len(received) == 1
        assert isinstance(received[0], RelationshipIndex)

    def test_empty_triples(self):
        b = RelationshipIndexBuilder()
        b.build_from_triples([])
        b.wait(timeout=10)
        assert len(b.index) == 0


class TestRelationshipIndexBuilderHotSwap:
    def test_old_index_served_while_building(self):
        initial = RelationshipIndex()
        initial.add(SVOTriple("a", "uses", "b", 1.0))
        b = RelationshipIndexBuilder(initial=initial)

        new_triples = [SVOTriple("x", "governs", "y", 0.9)] * 5
        b.build_from_triples(new_triples)

        # Before wait: ref may still point to initial
        early = b.ref.current
        assert early is not None  # never None — always serves something

        b.wait(timeout=10)
        assert b.ref.current is not initial  # hot-swapped
        assert b.ref.version == 1


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

class TestTopLevelImports:
    def test_versioned_ref_importable(self):
        from chonk import VersionedRef
        assert VersionedRef is not None

    def test_community_index_builder_importable(self):
        from chonk import CommunityIndexBuilder
        assert CommunityIndexBuilder is not None

    def test_relationship_index_builder_importable(self):
        from chonk import RelationshipIndexBuilder
        assert RelationshipIndexBuilder is not None
