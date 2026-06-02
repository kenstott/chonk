# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 9f4d18db-0f85-4aa2-ae8a-360b5566e609
"""Unit tests for chonk.indexer — Indexer + IndexHandle."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from chonk.indexer import Indexer, IndexHandle, get_indexer, release_indexer
from chonk.storage._store import Store

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _StubModel:
    """SentenceTransformer stub that returns ones without loading any model."""

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        n = len(texts)
        return np.ones((n, 1024), dtype=np.float32)


@pytest.fixture()
def tmp_store(tmp_path):
    db = tmp_path / "test.duckdb"
    with Store(db, embedding_dim=1024) as store:
        yield store


@pytest.fixture()
def txt_dir(tmp_path):
    """Temp directory with a handful of .txt documents."""
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text(
            f"# Section {i}\n\n" + ("This is content for document number {i}. " * 40),
            encoding="utf-8",
        )
    return tmp_path


# ---------------------------------------------------------------------------
# 1. index_source with directory type — chunks land in store
# ---------------------------------------------------------------------------


def test_index_source_directory(tmp_store, txt_dir):
    indexer = Indexer(store=tmp_store, embed_model=_StubModel())
    source_config = {
        "type": "directory",
        "uri": str(txt_dir),
        "extensions": [".txt"],
    }
    chunks_stored = indexer.index_source(source_config)
    assert chunks_stored > 0
    assert tmp_store.count() == chunks_stored


# ---------------------------------------------------------------------------
# 2. abort() stops before store phase — no chunks written
# ---------------------------------------------------------------------------


def test_abort_before_store(tmp_store, txt_dir):
    store_called = threading.Event()

    original_add = tmp_store.add_document

    def intercepting_add(*args, **kwargs):
        store_called.set()
        return original_add(*args, **kwargs)

    tmp_store.add_document = intercepting_add

    indexer = Indexer(store=tmp_store, embed_model=_StubModel())
    # Set the abort flag before starting so it fires at the first abort check
    indexer.abort()

    aborted = threading.Event()
    indexer._on_abort = lambda _: aborted.set()

    source_config = {
        "type": "directory",
        "uri": str(txt_dir),
        "extensions": [".txt"],
    }
    result = indexer.index_source(source_config)

    assert result == 0
    assert not store_called.is_set(), "add_document must not be called after abort"
    assert aborted.is_set()
    assert tmp_store.count() == 0


# ---------------------------------------------------------------------------
# 3. on_progress callback is called with correct phases
# ---------------------------------------------------------------------------


def test_on_progress_phases(tmp_store, txt_dir):
    phases_seen: list[str] = []

    def on_progress(phase, done, total):
        phases_seen.append(phase)

    indexer = Indexer(
        store=tmp_store,
        embed_model=_StubModel(),
        on_progress=on_progress,
    )
    source_config = {
        "type": "directory",
        "uri": str(txt_dir),
        "extensions": [".txt"],
    }
    indexer.index_source(source_config)

    assert "crawl" in phases_seen
    assert "chunk" in phases_seen
    assert "embed" in phases_seen
    assert "store" in phases_seen


# ---------------------------------------------------------------------------
# 4. on_error is called on bad URI without crashing
# ---------------------------------------------------------------------------


def test_on_error_bad_uri(tmp_store, tmp_path):
    errors: list[tuple[str, Exception]] = []

    indexer = Indexer(
        store=tmp_store,
        embed_model=_StubModel(),
        on_error=lambda phase, exc: errors.append((phase, exc)),
    )
    source_config = {
        "type": "directory",
        "uri": str(tmp_path / "does_not_exist"),
        "extensions": [".txt"],
    }
    result = indexer.index_source(source_config)

    assert result == 0
    assert len(errors) == 1
    assert errors[0][0] == "crawl"
    assert isinstance(errors[0][1], Exception)


# ---------------------------------------------------------------------------
# 5. index_source_async returns running handle; stops after join
# ---------------------------------------------------------------------------


def test_index_source_async(tmp_store, txt_dir):
    indexer = Indexer(store=tmp_store, embed_model=_StubModel())
    source_config = {
        "type": "directory",
        "uri": str(txt_dir),
        "extensions": [".txt"],
    }
    handle = indexer.index_source_async(source_config)

    assert isinstance(handle, IndexHandle)
    # At some point the thread is running (or finishes very fast)
    handle.join(timeout=30)
    assert not handle.running
    assert tmp_store.count() > 0


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


import sys


def _reg():
    return sys.modules["chonk.indexer"]._registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate registry state between tests."""
    _reg().clear()
    yield
    _reg().clear()


def test_get_indexer_returns_same_instance(tmp_store):
    a = get_indexer("ns:alice", tmp_store, _StubModel())
    b = get_indexer("ns:alice", tmp_store, _StubModel())
    assert a is b


def test_get_indexer_different_namespaces_are_distinct(tmp_path):
    db_a = tmp_path / "a.duckdb"
    db_b = tmp_path / "b.duckdb"
    with Store(db_a) as store_a, Store(db_b) as store_b:
        a = get_indexer("ns:alice", store_a, _StubModel())
        b = get_indexer("ns:bob", store_b, _StubModel())
    assert a is not b


def test_release_indexer_removes_from_registry(tmp_store):
    get_indexer("ns:alice", tmp_store, _StubModel())
    assert "ns:alice" in _reg()
    release_indexer("ns:alice")
    assert "ns:alice" not in _reg()


def test_release_nonexistent_namespace_is_noop():
    release_indexer("ns:ghost")  # must not raise


def test_get_indexer_after_release_creates_new_instance(tmp_store):
    first = get_indexer("ns:alice", tmp_store, _StubModel())
    release_indexer("ns:alice")
    second = get_indexer("ns:alice", tmp_store, _StubModel())
    assert first is not second


def test_registry_thread_safety(tmp_store):
    results = []
    barrier = threading.Barrier(8)

    def _get():
        barrier.wait()
        results.append(get_indexer("ns:alice", tmp_store, _StubModel()))

    threads = [threading.Thread(target=_get) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(id(r) for r in results)) == 1  # all same instance
