# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: f1e2d3c4-b5a6-9788-c1d2-e3f4a5b6c7d8
"""Integration smoke tests for WeaviateVectorBackend — requires Weaviate Cloud creds."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

weaviate_mod = pytest.importorskip("weaviate", reason="weaviate-client not installed")

from chonk.models import DocumentChunk  # noqa: E402
from chonk.storage._weaviate import WeaviateVectorBackend  # noqa: E402

DIM = 8
COLLECTION = "ChonkTestSmoke"


def _creds() -> tuple[str, str]:
    url = os.environ.get("WEAVIATE_URL", "")
    key = os.environ.get("WEAVIATE_API_KEY", "")
    if not url or not key:
        pytest.skip("WEAVIATE_URL / WEAVIATE_API_KEY not set")
    if not url.startswith("https://"):
        url = "https://" + url.removeprefix("grpc-")
    return url, key


def _chunks() -> tuple[list[DocumentChunk], list[list[float]]]:
    chunks = [
        DocumentChunk(
            document_name="doc_a",
            content="The quick brown fox",
            chunk_index=0,
            chunk_type="document",
        ),
        DocumentChunk(
            document_name="doc_a",
            content="Jumped over the lazy dog",
            chunk_index=1,
            chunk_type="document",
        ),
        DocumentChunk(
            document_name="doc_b",
            content="A completely different topic about weather",
            chunk_index=0,
            chunk_type="document",
        ),
    ]
    rng = np.random.default_rng(42)
    vecs = [rng.random(DIM, dtype="float32").tolist() for _ in chunks]
    return chunks, vecs


@pytest.fixture()
def backend() -> WeaviateVectorBackend:
    url, key = _creds()
    b = WeaviateVectorBackend(
        cluster_url=url,
        api_key=key,
        collection=COLLECTION,
        embedding_dim=DIM,
    )
    b.clear()
    yield b
    b.clear()
    b.close()


def test_add_and_count(backend: WeaviateVectorBackend) -> None:
    chunks, vecs = _chunks()
    backend.add_chunks(chunks, vecs)
    assert backend.count() == 3


def test_ann_search(backend: WeaviateVectorBackend) -> None:
    chunks, vecs = _chunks()
    backend.add_chunks(chunks, vecs)
    time.sleep(4)  # hfresh index initialization on free tier
    results = backend.search(vecs[0], limit=3)
    assert len(results) >= 1
    for chunk_id, score, chunk in results:
        assert isinstance(chunk_id, str)
        assert 0.0 <= score <= 1.0
        assert chunk.document_name in {"doc_a", "doc_b"}


def test_hybrid_search(backend: WeaviateVectorBackend) -> None:
    chunks, vecs = _chunks()
    backend.add_chunks(chunks, vecs)
    time.sleep(4)  # hfresh index initialization on free tier
    results = backend.search(vecs[0], limit=3, query_text="fox")
    assert len(results) >= 1


def test_filtered_search(backend: WeaviateVectorBackend) -> None:
    chunks, vecs = _chunks()
    backend.add_chunks(chunks, vecs, namespace="ns1")
    time.sleep(4)
    results = backend.search(vecs[0], limit=3, namespaces=["ns1"])
    assert len(results) >= 1
    results_ns2 = backend.search(vecs[0], limit=3, namespaces=["ns2"])
    assert len(results_ns2) == 0


def test_delete_by_document(backend: WeaviateVectorBackend) -> None:
    chunks, vecs = _chunks()
    backend.add_chunks(chunks, vecs)
    deleted = backend.delete_by_document("doc_a")
    assert deleted == 2
    assert backend.count() == 1


def test_get_all_chunks(backend: WeaviateVectorBackend) -> None:
    chunks, vecs = _chunks()
    backend.add_chunks(chunks, vecs)
    all_chunks = backend.get_all_chunks()
    assert len(all_chunks) == 3


def test_compact(backend: WeaviateVectorBackend) -> None:
    chunks, vecs = _chunks()
    backend.add_chunks(chunks, vecs)
    # Simulate orphan by deleting from catalog only
    backend._catalog.execute("DELETE FROM embeddings WHERE document_name = 'doc_b'")
    deleted = backend.compact()
    assert deleted == 1


def test_register_and_list_documents(backend: WeaviateVectorBackend) -> None:
    backend.register_document("doc_x", "abc123", source_uri="file://doc_x", chunk_count=5)
    docs = backend.list_documents()
    assert any(d["document_name"] == "doc_x" for d in docs)


def test_clear(backend: WeaviateVectorBackend) -> None:
    chunks, vecs = _chunks()
    backend.add_chunks(chunks, vecs)
    backend.clear()
    assert backend.count() == 0
