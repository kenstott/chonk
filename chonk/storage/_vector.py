# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 3ce49853-1b78-46af-9a67-7fab26f10d28
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DuckDB VSS + FTS vector backend for chonk."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

try:
    import duckdb  # noqa: F401 — availability checked at runtime
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

if TYPE_CHECKING:
    from .._pool import ThreadLocalDuckDB  # only for type hints

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore

logger = logging.getLogger(__name__)


def _deserialize_section(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, TypeError):
        pass
    return [value]


_MISSING_DEPS_MSG = (
    "duckdb and numpy are required for storage. "
    "Install them with: pip install chonk[storage]"
)


def _require_deps() -> None:
    if not _DUCKDB_AVAILABLE or not _NUMPY_AVAILABLE:
        raise ImportError(_MISSING_DEPS_MSG)


class DuckDBVectorBackend:
    """Vector operations backed by DuckDB VSS (HNSW) + FTS extensions."""

    def __init__(self, db, embedding_dim: int = 1024):
        _require_deps()
        self._db = db
        self._embedding_dim = embedding_dim
        self._fts_dirty = True
        self._np_embeddings = None   # preloaded (n, dim) float32
        self._np_chunk_rows = None   # preloaded metadata rows
        self._init_schema()

    def preload_embeddings(self) -> None:
        """Load all embeddings into RAM for fast batched numpy search."""
        rows = self._conn.execute(
            """
            SELECT chunk_id, document_name, section, chunk_index,
                   content, breadcrumb, chunk_type, source_offset, source_length,
                   namespace, embedding
            FROM embeddings
            """
        ).fetchall()
        if not rows:
            return
        self._np_chunk_rows = rows
        self._np_embeddings = _np.array(
            [r[10] for r in rows], dtype="float32"
        )

    @property
    def _conn(self):
        return self._db.conn

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        from ._schema import get_ddl, VSS_INDEX_DDL

        # Load extensions (best-effort — may already be loaded)
        for ext in ("vss", "fts"):
            try:
                self._conn.execute(f"INSTALL {ext}").fetchall()
                self._conn.execute(f"LOAD {ext}").fetchall()
            except Exception as e:
                logger.debug(f"Extension {ext} load skipped: {e}")

        for ddl in get_ddl(self._embedding_dim):
            try:
                self._conn.execute(ddl).fetchall()
            except Exception as e:
                logger.debug(f"DDL skipped: {e}")

        # Always drop and recreate HNSW index to ensure cosine metric
        from ._schema import VSS_DROP_INDEX_DDL
        try:
            self._conn.execute(VSS_DROP_INDEX_DDL).fetchall()
            self._conn.execute(VSS_INDEX_DDL).fetchall()
            logger.debug("VSS HNSW cosine index ready")
        except Exception as e:
            logger.debug(f"VSS index creation skipped: {e}")

    # ------------------------------------------------------------------
    # Chunk ID
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_chunk_id(document_name: str, chunk_index: int, content: str) -> str:
        content_hash = hashlib.sha256(
            f"{document_name}:{chunk_index}:{content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{document_name}_{chunk_index}_{content_hash}"

    # ------------------------------------------------------------------
    # Add chunks
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list, embeddings, namespace: str | None = None) -> None:
        """Insert chunks with embeddings into the embeddings table.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: np.ndarray of shape (n, embedding_dim).
            namespace: Optional partition key. None means no namespace filter at search time.
        """
        if not chunks:
            return

        records = []
        for i, chunk in enumerate(chunks):
            embed_content = (
                chunk.embedding_content
                if hasattr(chunk, "embedding_content") and chunk.embedding_content
                else chunk.content
            )
            chunk_id = self._generate_chunk_id(
                chunk.document_name, chunk.chunk_index, embed_content
            )
            embedding = embeddings[i].tolist()
            chunk_type = (
                chunk.chunk_type.value
                if hasattr(chunk, "chunk_type") and hasattr(chunk.chunk_type, "value")
                else getattr(chunk, "chunk_type", "document") or "document"
            )
            raw_section = getattr(chunk, "section", []) or []
            section_str = json.dumps(raw_section) if isinstance(raw_section, list) else raw_section
            records.append((
                chunk_id,
                chunk.document_name,
                section_str,
                chunk.chunk_index,
                chunk.content,
                getattr(chunk, "breadcrumb", None),
                chunk_type,
                getattr(chunk, "source_offset", None),
                getattr(chunk, "source_length", None),
                namespace,
                embedding,
            ))

        self._conn.executemany(
            """
            INSERT INTO embeddings
                (chunk_id, document_name, section, chunk_index, content,
                 breadcrumb, chunk_type, source_offset, source_length, namespace, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            records,
        )
        self._fts_dirty = True

    # ------------------------------------------------------------------
    # FTS / BM25
    # ------------------------------------------------------------------

    def rebuild_fts_index(self) -> None:
        """Rebuild the BM25 full-text search index."""
        self._rebuild_fts_index()

    def _rebuild_fts_index(self) -> None:
        if not self._fts_dirty:
            return
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if count == 0:
                self._fts_dirty = False
                return
            self._conn.execute(
                "PRAGMA create_fts_index('embeddings', 'chunk_id', 'content', "
                "stemmer='porter', overwrite=1)"
            ).fetchall()
            self._fts_dirty = False
        except Exception as e:
            logger.debug(f"FTS index rebuild failed (vector-only mode): {e}")
            self._fts_dirty = False

    def _bm25_search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> list[tuple[str, float, object]]:
        from ..models import DocumentChunk
        try:
            self._rebuild_fts_index()
            rows = self._conn.execute(
                """
                SELECT e.chunk_id, e.document_name, e.section, e.chunk_index,
                       e.content, e.chunk_type, e.source_offset, e.source_length,
                       fts_main_embeddings.match_bm25(e.chunk_id, ?) AS bm25_score
                FROM embeddings e
                WHERE bm25_score IS NOT NULL
                ORDER BY bm25_score DESC
                LIMIT ?
                """,
                [query_text, limit],
            ).fetchall()

            results = []
            for row in rows:
                (chunk_id, doc_name, section, chunk_idx, content,
                 chunk_type_str, source_offset, source_length, score) = row
                chunk = DocumentChunk(
                    document_name=doc_name,
                    content=content,
                    section=_deserialize_section(section),
                    chunk_index=chunk_idx,
                    source_offset=source_offset,
                    source_length=source_length,
                )
                results.append((chunk_id, float(score), chunk))
            return results
        except Exception as e:
            logger.debug(f"BM25 search failed (vector-only mode): {e}")
            return []

    @staticmethod
    def _rrf_merge(
        vector_results: list[tuple],
        bm25_results: list[tuple],
        k: int = 60,
    ) -> list[tuple]:
        """Reciprocal Rank Fusion of vector and BM25 result lists."""
        max_rrf = 2.0 / (k + 1)
        scores: dict[str, float] = {}
        chunks: dict[str, tuple] = {}

        for rank, (chunk_id, _score, chunk) in enumerate(vector_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = (chunk_id, chunk)

        for rank, (chunk_id, _score, chunk) in enumerate(bm25_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunks:
                chunks[chunk_id] = (chunk_id, chunk)

        merged = []
        for chunk_id, rrf_score in scores.items():
            normalized = rrf_score / max_rrf
            cid, chunk = chunks[chunk_id]
            merged.append((cid, normalized, chunk))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding,
        limit: int = 5,
        query_text: str | None = None,
        include_breadcrumbs: bool = True,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[tuple[str, float, object]]:
        """Search by vector similarity, with optional BM25 hybrid re-ranking.

        Args:
            query_embedding: np.ndarray of shape (dim,) or (1, dim).
            limit: Maximum number of results to return.
            query_text: If provided, perform hybrid vector + BM25 RRF search.
            include_breadcrumbs: If True (default), prepend stored breadcrumb to
                returned chunk content. If False, return raw content only.
            namespaces: If provided, restrict search to rows whose namespace is in
                this list. None searches all namespaces (backwards-compatible default).
            chunk_types: If provided, restrict search to rows whose chunk_type is in
                this list. None searches all chunk types (backwards-compatible default).

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        from ..models import DocumentChunk

        query_vec = query_embedding.flatten().astype("float32")
        fetch_limit = limit * 3 if query_text else limit

        if self._np_embeddings is not None and _np is not None:
            # Fast numpy path: single matmul, no DuckDB round-trip
            sims = self._np_embeddings @ query_vec  # (n,)
            ns_set = set(namespaces) if namespaces is not None else None
            ct_set = set(chunk_types) if chunk_types is not None else None
            top_idx = _np.argpartition(sims, -fetch_limit)[-fetch_limit:]
            top_idx = top_idx[_np.argsort(sims[top_idx])[::-1]]
            if ns_set is not None:
                top_idx = [i for i in top_idx if self._np_chunk_rows[i][9] in ns_set]
            if ct_set is not None:
                top_idx = [i for i in top_idx if self._np_chunk_rows[i][6] in ct_set]
            rows = [(*self._np_chunk_rows[i][:9], float(sims[i])) for i in top_idx]
        else:
            query = query_vec.tolist()
            clauses: list[str] = []
            params: list = [query]
            if namespaces is not None:
                placeholders = ", ".join("?" * len(namespaces))
                clauses.append(f"e.namespace IN ({placeholders})")
                params.extend(namespaces)
            if chunk_types is not None:
                placeholders = ", ".join("?" * len(chunk_types))
                clauses.append(f"e.chunk_type IN ({placeholders})")
                params.extend(chunk_types)
            where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params += [query, fetch_limit]

            rows = self._conn.execute(
                f"""
                SELECT
                    e.chunk_id,
                    e.document_name,
                    e.section,
                    e.chunk_index,
                    e.content,
                    e.breadcrumb,
                    e.chunk_type,
                    e.source_offset,
                    e.source_length,
                    1.0 - array_cosine_distance(e.embedding, ?::FLOAT[{self._embedding_dim}]) AS similarity
                FROM embeddings e
                {where_clause}
                ORDER BY array_cosine_distance(e.embedding, ?::FLOAT[{self._embedding_dim}]) ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

        vector_results = []
        for row in rows:
            (chunk_id, doc_name, section, chunk_idx, content,
             breadcrumb, chunk_type_str, source_offset, source_length, similarity) = row
            displayed = (
                f"{breadcrumb}\n\n{content}"
                if include_breadcrumbs and breadcrumb
                else content
            )
            chunk = DocumentChunk(
                document_name=doc_name,
                content=displayed,
                section=_deserialize_section(section),
                chunk_index=chunk_idx,
                source_offset=source_offset,
                source_length=source_length,
                breadcrumb=breadcrumb,
                chunk_type=chunk_type_str or "document",
            )
            vector_results.append((chunk_id, float(similarity), chunk))

        if not query_text:
            return vector_results

        bm25_results = self._bm25_search(query_text, limit=fetch_limit)
        if not bm25_results:
            return vector_results[:limit]

        merged = self._rrf_merge(vector_results, bm25_results)
        return merged[:limit]

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_all_chunks(self) -> list:
        """Return all stored chunks as DocumentChunk objects."""
        from ..models import DocumentChunk

        rows = self._conn.execute(
            """
            SELECT document_name, content, section, chunk_index,
                   source_offset, source_length, breadcrumb
            FROM embeddings
            ORDER BY document_name, chunk_index
            """
        ).fetchall()
        return [
            DocumentChunk(
                document_name=row[0],
                content=row[1],
                section=_deserialize_section(row[2]),
                chunk_index=row[3],
                source_offset=row[4],
                source_length=row[5],
                breadcrumb=row[6],
                embedding_content=f"{row[6]}\n\n{row[1]}" if row[6] else row[1],
            )
            for row in rows
        ]

    def migrate_extract_breadcrumbs(self) -> int:
        """One-time migration: extract breadcrumb prefix from content into breadcrumb column.

        For databases created before the breadcrumb column was added, the breadcrumb
        was baked into content as ``[doc > section]\\n\\ncontent``. This method
        extracts it into the breadcrumb column and strips it from content.

        Returns:
            Number of rows updated.
        """
        from ._schema import EMBEDDINGS_MIGRATE_BREADCRUMB
        try:
            self._conn.execute(EMBEDDINGS_MIGRATE_BREADCRUMB).fetchall()
        except Exception:
            pass  # column already exists

        result = self._conn.execute("""
            UPDATE embeddings
            SET
                breadcrumb = regexp_extract(content, '^(\\[[^\\]]+\\])', 1),
                content    = regexp_replace(content, '^\\[[^\\]]+\\]\\n\\n', '')
            WHERE breadcrumb IS NULL
              AND content LIKE '[%'
        """).fetchall()

        updated = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE breadcrumb IS NOT NULL"
        ).fetchone()[0]
        return updated

    def count(self) -> int:
        """Return the total number of stored chunks."""
        result = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    def delete_by_document(self, document_name: str) -> int:
        """Delete all chunks for a document. Returns the number deleted."""
        count_before = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchone()[0]
        self._conn.execute(
            "DELETE FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchall()
        self._fts_dirty = True
        return count_before

    def clear(self) -> None:
        """Delete all chunks from the embeddings table."""
        self._conn.execute("DELETE FROM embeddings").fetchall()
        self._fts_dirty = True
