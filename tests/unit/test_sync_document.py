# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: c06a66dc-d084-4067-8814-c82c6a954c3b
"""Tests for sync_document, SyncResult, and DuckDBVectorBackend document registry."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from chonk.storage._vector import SyncResult, sync_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend():
    from chonk.storage import Store

    with Store(":memory:", embedding_dim=4) as store:
        yield store.vector


def _make_chunk(name: str = "doc", idx: int = 0, content: str = "hello world"):
    from chonk.models import DocumentChunk

    return DocumentChunk(document_name=name, content=content, chunk_index=idx)


def _embeddings(n: int = 1, dim: int = 4):
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------


class TestSyncResult:
    def test_default_fields(self):
        r = SyncResult(action="added", document_name="doc")
        assert r.chunk_count == 0
        assert r.previous_chunk_count == 0
        assert r.content_hash == ""

    def test_all_fields(self):
        r = SyncResult(
            action="updated",
            document_name="doc",
            content_hash="abc",
            chunk_count=5,
            previous_chunk_count=3,
        )
        assert r.action == "updated"
        assert r.content_hash == "abc"
        assert r.chunk_count == 5
        assert r.previous_chunk_count == 3


# ---------------------------------------------------------------------------
# Registry methods
# ---------------------------------------------------------------------------


class TestDocumentRegistry:
    def test_get_hash_unknown_returns_none(self, backend):
        assert backend.get_document_hash("nonexistent") is None

    def test_register_and_get(self, backend):
        backend.register_document("doc", "abc123", source_uri="s3://bucket/doc", chunk_count=7)
        assert backend.get_document_hash("doc") == "abc123"

    def test_register_updates_existing(self, backend):
        backend.register_document("doc", "hash1", chunk_count=3)
        backend.register_document("doc", "hash2", chunk_count=5)
        assert backend.get_document_hash("doc") == "hash2"

    def test_list_documents_empty(self, backend):
        assert backend.list_documents() == []

    def test_list_documents_returns_all(self, backend):
        backend.register_document("a", "h1", source_uri="uri-a", chunk_count=2)
        backend.register_document("b", "h2", source_uri="uri-b", chunk_count=4)
        docs = backend.list_documents()
        assert len(docs) == 2
        names = {d["document_name"] for d in docs}
        assert names == {"a", "b"}

    def test_list_documents_fields(self, backend):
        backend.register_document("doc", "myhash", source_uri="https://example.com", chunk_count=9)
        docs = backend.list_documents()
        d = docs[0]
        assert d["document_name"] == "doc"
        assert d["content_hash"] == "myhash"
        assert d["source_uri"] == "https://example.com"
        assert d["chunk_count"] == 9
        assert d["indexed_at"] is not None

    def test_delete_by_document_removes_registry(self, backend):
        backend.register_document("doc", "hash1", chunk_count=3)
        backend.delete_by_document("doc")
        assert backend.get_document_hash("doc") is None
        assert backend.list_documents() == []

    def test_clear_removes_all_registry(self, backend):
        backend.register_document("a", "h1", chunk_count=1)
        backend.register_document("b", "h2", chunk_count=2)
        backend.clear()
        assert backend.list_documents() == []


# ---------------------------------------------------------------------------
# sync_document
# ---------------------------------------------------------------------------


class TestSyncDocument:
    def test_new_document_returns_added(self, backend):
        raw = b"first version"
        result = sync_document(backend, "doc", raw)
        assert result.action == "added"
        assert result.document_name == "doc"
        assert result.content_hash == hashlib.sha256(raw).hexdigest()
        assert result.previous_chunk_count == 0

    def test_same_content_returns_skipped(self, backend):
        raw = b"some content"
        backend.register_document("doc", hashlib.sha256(raw).hexdigest(), chunk_count=2)
        result = sync_document(backend, "doc", raw)
        assert result.action == "skipped"
        assert result.chunk_count == 0

    def test_changed_content_returns_updated(self, backend):
        raw_v1 = b"version one"
        backend.register_document("doc", hashlib.sha256(raw_v1).hexdigest(), chunk_count=3)
        chunk = _make_chunk("doc")
        backend.add_chunks([chunk], _embeddings(1))

        raw_v2 = b"version two"
        result = sync_document(backend, "doc", raw_v2)
        assert result.action == "updated"
        assert result.previous_chunk_count == 1
        assert result.content_hash == hashlib.sha256(raw_v2).hexdigest()

    def test_changed_content_deletes_old_chunks(self, backend):
        raw_v1 = b"version one"
        backend.register_document("doc", hashlib.sha256(raw_v1).hexdigest(), chunk_count=1)
        backend.add_chunks([_make_chunk("doc")], _embeddings(1))
        assert backend.count() == 1

        sync_document(backend, "doc", b"version two")
        assert backend.count() == 0

    def test_changed_content_removes_registry_entry(self, backend):
        raw_v1 = b"v1"
        backend.register_document("doc", hashlib.sha256(raw_v1).hexdigest(), chunk_count=1)
        sync_document(backend, "doc", b"v2")
        # Old hash is gone; caller will register new hash after re-embedding
        assert backend.get_document_hash("doc") is None

    def test_content_hash_on_result_matches_sha256(self, backend):
        raw = b"hello"
        result = sync_document(backend, "doc", raw)
        assert result.content_hash == hashlib.sha256(raw).hexdigest()

    def test_full_cycle(self, backend):
        """added → skipped → updated → skipped"""
        raw_v1 = b"content v1"

        r1 = sync_document(backend, "doc", raw_v1)
        assert r1.action == "added"

        # Simulate post-embed register
        backend.register_document("doc", r1.content_hash, chunk_count=2)

        r2 = sync_document(backend, "doc", raw_v1)
        assert r2.action == "skipped"

        raw_v2 = b"content v2"
        r3 = sync_document(backend, "doc", raw_v2)
        assert r3.action == "updated"
        assert r3.previous_chunk_count == 0  # no actual chunks were added in this test

        backend.register_document("doc", r3.content_hash, chunk_count=3)

        r4 = sync_document(backend, "doc", raw_v2)
        assert r4.action == "skipped"

    def test_source_uri_does_not_affect_hash(self, backend):
        raw = b"content"
        r1 = sync_document(backend, "doc", raw, source_uri="https://v1.example.com/doc")
        r2 = sync_document(backend, "doc", raw, source_uri="https://v2.example.com/doc")
        # Both produce same content hash; second would be skipped if first registered
        assert r1.content_hash == r2.content_hash

    def test_independent_documents_tracked_separately(self, backend):
        raw_a = b"doc a content"
        raw_b = b"doc b content"
        backend.register_document("a", hashlib.sha256(raw_a).hexdigest(), chunk_count=1)

        ra = sync_document(backend, "a", raw_a)
        rb = sync_document(backend, "b", raw_b)
        assert ra.action == "skipped"
        assert rb.action == "added"

    # -- hash-first (no raw bytes) --

    def test_hash_only_skips_when_matching(self, backend):
        backend.register_document("doc", "etag-abc", chunk_count=3)
        result = sync_document(backend, "doc", content_hash="etag-abc")
        assert result.action == "skipped"
        assert result.content_hash == "etag-abc"

    def test_hash_only_added_when_new(self, backend):
        result = sync_document(backend, "doc", content_hash="etag-xyz")
        assert result.action == "added"
        assert result.content_hash == "etag-xyz"

    def test_hash_only_updated_when_changed(self, backend):
        backend.register_document("doc", "etag-v1", chunk_count=2)
        result = sync_document(backend, "doc", content_hash="etag-v2")
        assert result.action == "updated"
        assert result.previous_chunk_count == 0

    def test_neither_raw_nor_hash_raises(self, backend):
        with pytest.raises(ValueError):
            sync_document(backend, "doc")

    def test_hash_overrides_raw_bytes(self, backend):
        # explicit content_hash takes precedence over hashing raw
        result = sync_document(backend, "doc", b"ignored", content_hash="explicit-hash")
        assert result.content_hash == "explicit-hash"
