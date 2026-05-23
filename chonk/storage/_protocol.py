# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: fb0dc27c-4ed9-413d-ae7f-be8255a0d901
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""VectorBackend protocol — implemented by DuckDBVectorBackend and PgVectorBackend."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Contract for vector + FTS storage backends.

    Implementations must be safe for the following call sequence per document::

        backend.add_chunks(chunks, embeddings, namespace=..., ...)
        backend.register_document(document_name, content_hash, ...)

    ``chunk_id`` is the idempotency key — ``add_chunks`` must silently ignore
    duplicate chunk_ids (``ON CONFLICT DO NOTHING`` semantics).
    """

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list,
        embeddings,                          # np.ndarray shape (n, embedding_dim)
        namespace: str | None = None,
        source_id: str | None = None,
        domain_id: str | None = None,
        session_fingerprint: str | None = None,
    ) -> None: ...

    def register_document(
        self,
        document_name: str,
        content_hash: str,
        source_uri: str = "",
        chunk_count: int = 0,
    ) -> None: ...

    def delete_by_document(self, document_name: str) -> int: ...

    def clear(self) -> None: ...

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding,                     # np.ndarray shape (dim,) or (1, dim)
        limit: int = 5,
        query_text: str | None = None,       # None → pure vector; str → hybrid RRF
        include_breadcrumbs: bool = True,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_ids: list[str] | None = None,
        session_fingerprint: str | None = None,
    ) -> list[tuple[str, float, object]]: ...  # (chunk_id, score, DocumentChunk)

    def get_all_chunks(self) -> list: ...    # list[DocumentChunk] — used by graph builder

    # ------------------------------------------------------------------
    # Document registry
    # ------------------------------------------------------------------

    def get_document_hash(self, document_name: str) -> str | None: ...

    def list_documents(self) -> list[dict]: ...  # keys: document_name, content_hash, source_uri, indexed_at, chunk_count

    def count(self) -> int: ...

    # ------------------------------------------------------------------
    # Lifecycle / optimisation hints
    # ------------------------------------------------------------------

    def rebuild_fts_index(self) -> None: ...
    # Backends with live FTS indexes (PG tsvector) implement as no-op.

    def preload_embeddings(self) -> None: ...
    # Backends with index-backed ANN (pgvector HNSW) implement as no-op.
