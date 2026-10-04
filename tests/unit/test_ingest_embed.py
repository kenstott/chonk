# Copyright (c) 2025 Kenneth Stott. MIT License.

"""build() must embed the breadcrumb-enriched text, like every other embed path."""

from __future__ import annotations

import sys
import types

import numpy as np

from chonk._config import EmbedConfig
from chonk.ingest import _embed_chunks
from chonk.models import DocumentChunk


def test_embed_chunks_uses_embedding_content_when_set(monkeypatch):
    seen: list[str] = []

    class FakeModel:
        def __init__(self, name):
            pass

        def encode(self, texts, **_):
            seen.extend(texts)
            return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeModel),
    )
    chunks = [
        DocumentChunk(document_name="d", content="body", embedding_content="[d > Sec]\n\nbody"),
        DocumentChunk(document_name="d", content="plain"),
    ]

    _embed_chunks(chunks, EmbedConfig(model="fake", batch_size=8))

    assert seen == ["[d > Sec]\n\nbody", "plain"]


def test_add_source_embeds_enriched_text(monkeypatch):
    from chonk import ingest

    captured: list[list[str]] = []

    def fake_embed_texts(texts, model, batch_size=256):
        captured.append(list(texts))
        return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setattr(ingest, "_embed_texts", fake_embed_texts)
    chunks = [DocumentChunk(document_name="d", content="body", embedding_content="[d]\n\nbody")]
    monkeypatch.setattr(ingest, "_ingest_source", lambda src, loader: chunks)

    class FakeVector:
        def rebuild_fts_index(self):
            pass

    class FakeStore:
        vector = FakeVector()

        def add_document(self, *a, **k):
            pass

    idx = ingest.Index.__new__(ingest.Index)
    idx._store = FakeStore()
    idx._domain_map = {"ns": {}}
    idx._embed_model = "fake"
    idx._embed_cfg = EmbedConfig(model="fake")
    idx._loader_cfg = ingest.LoaderConfig()
    monkeypatch.setattr(idx, "_get_or_register_domain", lambda ns, d: "did", raising=False)
    monkeypatch.setattr(idx, "_invalidate_search", lambda: None, raising=False)

    idx.add_source({"name": "s", "type": "glob", "path": ".", "namespace": "ns"}, rebuild=False)

    assert captured == [["[d]\n\nbody"]]
