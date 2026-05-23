# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b9e1f2a3-4c5d-6e7f-8a9b-0c1d2e3f4a5b
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""PostgreSQL + pgvector backend for chonk."""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

_MISSING_DEPS_MSG = (
    "psycopg2 and pgvector are required for PgVectorBackend. "
    "Install them with: pip install chonk[pgvector]"
)


def _require_deps() -> None:
    try:
        import pgvector  # type: ignore[import-untyped]  # noqa: F401
        import psycopg2  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise ImportError(_MISSING_DEPS_MSG) from exc


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


class PgVectorBackend:
    """Vector operations backed by PostgreSQL + pgvector (HNSW cosine index).

    Args:
        dsn: PostgreSQL DSN string, e.g. ``"postgresql://user:pass@host/db"``.
        embedding_dim: Embedding vector dimension. Must match your model.
        table: Table name to use (default: ``"chonk_embeddings"``).
    """

    def __init__(
        self,
        dsn: str,
        embedding_dim: int = 1024,
        table: str = "chonk_embeddings",
    ) -> None:
        _require_deps()
        self._dsn = dsn
        self._embedding_dim = embedding_dim
        self._table = table
        self._conn = self._connect()
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        import psycopg2
        from pgvector.psycopg2 import register_vector  # type: ignore[import-untyped]

        conn = psycopg2.connect(self._dsn)
        conn.autocommit = False
        register_vector(conn)
        return conn

    def _ensure_connection(self) -> None:
        import psycopg2

        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            self._conn = self._connect()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        t = self._table
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    chunk_id            TEXT PRIMARY KEY,
                    document_name       TEXT NOT NULL,
                    section             TEXT,
                    chunk_index         INTEGER NOT NULL DEFAULT 0,
                    content             TEXT NOT NULL,
                    breadcrumb          TEXT,
                    chunk_type          TEXT NOT NULL DEFAULT 'document',
                    source_offset       INTEGER,
                    source_length       INTEGER,
                    namespace           TEXT,
                    source_detail       TEXT,
                    source_id           TEXT,
                    domain_id           TEXT,
                    session_fingerprint TEXT,
                    embedding           vector({self._embedding_dim}),
                    fts_vec             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                )
            """)
            for col in ("source_id", "domain_id", "session_fingerprint"):
                cur.execute(f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS {col} TEXT")
            cur.execute(f"""
                ALTER TABLE {t} ADD COLUMN IF NOT EXISTS
                    fts_vec tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            """)
            # HNSW cosine index — works on empty tables (unlike IVFFlat)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {t}_embedding_hnsw_idx
                ON {t} USING hnsw (embedding vector_cosine_ops)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {t}_fts_idx ON {t} USING gin(fts_vec)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chonk_documents (
                    document_name TEXT PRIMARY KEY,
                    content_hash  TEXT NOT NULL,
                    source_uri    TEXT NOT NULL DEFAULT '',
                    indexed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    chunk_count   INTEGER NOT NULL DEFAULT 0
                )
            """)
        self._conn.commit()

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

    def add_chunks(
        self,
        chunks: list,
        embeddings,
        namespace: str | None = None,
        source_id: str | None = None,
        domain_id: str | None = None,
        session_fingerprint: str | None = None,
    ) -> None:
        """Insert chunks with embeddings.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: np.ndarray of shape (n, embedding_dim).
            namespace: Optional partition key.
        """
        if not chunks:
            return

        import numpy as np
        from pgvector.psycopg2 import register_vector  # type: ignore[import-untyped]

        self._ensure_connection()
        register_vector(self._conn)

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
            chunk_type = (
                chunk.chunk_type.value
                if hasattr(chunk, "chunk_type") and hasattr(chunk.chunk_type, "value")
                else getattr(chunk, "chunk_type", "document") or "document"
            )
            raw_section = getattr(chunk, "section", []) or []
            section_str = json.dumps(raw_section) if isinstance(raw_section, list) else raw_section
            raw_detail = getattr(chunk, "source_detail", None)
            source_detail_str = json.dumps(raw_detail) if raw_detail is not None else None

            vec = np.array(embeddings[i], dtype="float32")
            records.append(
                (
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
                    source_detail_str,
                    source_id,
                    domain_id,
                    session_fingerprint,
                    vec,
                )
            )

        t = self._table
        with self._conn.cursor() as cur:
            for rec in records:
                cur.execute(
                    f"""
                    INSERT INTO {t}
                        (chunk_id, document_name, section, chunk_index, content,
                         breadcrumb, chunk_type, source_offset, source_length,
                         namespace, source_detail, source_id, domain_id,
                         session_fingerprint, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    rec,
                )
        self._conn.commit()

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
        domain_ids: list[str] | None = None,
        session_fingerprint: str | None = None,
    ) -> list[tuple[str, float, object]]:
        """Search by cosine similarity using pgvector HNSW index.

        BM25 hybrid is not supported on PgVectorBackend; ``query_text`` is
        accepted for API compatibility but triggers a warning and is ignored.

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        import numpy as np
        from pgvector.psycopg2 import register_vector  # type: ignore[import-untyped]

        from ..models import DocumentChunk

        if query_text is not None:
            logger.debug(
                "PgVectorBackend: query_text BM25 hybrid not supported; using pure vector search."
            )

        self._ensure_connection()
        register_vector(self._conn)

        query_vec = np.array(query_embedding, dtype="float32").flatten()

        t = self._table
        clauses: list[str] = []
        filter_params: list = []

        if namespaces is not None:
            clauses.append("namespace = ANY(%s)")
            filter_params.append(namespaces)
        if chunk_types is not None:
            clauses.append("chunk_type = ANY(%s)")
            filter_params.append(chunk_types)
        if domain_ids is not None:
            clauses.append("domain_id = ANY(%s)")
            filter_params.append(domain_ids)
        if session_fingerprint is not None:
            clauses.append("session_fingerprint = %s")
            filter_params.append(session_fingerprint)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        all_params = [query_vec] + filter_params + [query_vec, limit]

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    chunk_id, document_name, section, chunk_index, content,
                    breadcrumb, chunk_type, source_offset, source_length,
                    source_detail,
                    1.0 - (embedding <=> %s::vector) AS similarity
                FROM {t}
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                all_params,
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            (
                chunk_id,
                doc_name,
                section,
                chunk_idx,
                content,
                breadcrumb,
                chunk_type_str,
                source_offset,
                source_length,
                source_detail_str,
                similarity,
            ) = row
            displayed = (
                f"{breadcrumb}\n\n{content}" if include_breadcrumbs and breadcrumb else content
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
                source_detail=json.loads(source_detail_str) if source_detail_str else None,
            )
            results.append((chunk_id, float(similarity), chunk))

        return results

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    def delete_by_document(self, document_name: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        self._ensure_connection()
        t = self._table
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {t} WHERE document_name = %s",
                [document_name],
            )
            row = cur.fetchone()
            count = row[0] if row else 0
            cur.execute(
                f"DELETE FROM {t} WHERE document_name = %s",
                [document_name],
            )
        self._conn.commit()
        return count

    def clear(self) -> None:
        """Delete all chunks from the table."""
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table}")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of stored chunks."""
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            result = cur.fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # get_all_chunks
    # ------------------------------------------------------------------

    def get_all_chunks(self) -> list:
        """Return all stored chunks as DocumentChunk objects (server-side cursor)."""
        from ..models import DocumentChunk

        self._ensure_connection()
        chunks = []
        t = self._table
        with self._conn.cursor("get_all_chunks") as cur:
            cur.execute(
                f"""
                SELECT document_name, content, section, chunk_index,
                       source_offset, source_length, breadcrumb, source_detail, chunk_type
                FROM {t}
                ORDER BY document_name, chunk_index
                """
            )
            for row in cur:
                doc_name, content, section, chunk_idx, src_off, src_len, breadcrumb, src_detail, chunk_type = row
                chunks.append(DocumentChunk(
                    document_name=doc_name,
                    content=content,
                    section=_deserialize_section(section),
                    chunk_index=chunk_idx,
                    source_offset=src_off,
                    source_length=src_len,
                    breadcrumb=breadcrumb,
                    embedding_content=f"{breadcrumb}\n\n{content}" if breadcrumb else content,
                    source_detail=json.loads(src_detail) if src_detail else None,
                    chunk_type=chunk_type or "document",
                ))
        return chunks

    # ------------------------------------------------------------------
    # Document registry
    # ------------------------------------------------------------------

    def register_document(
        self,
        document_name: str,
        content_hash: str,
        source_uri: str = "",
        chunk_count: int = 0,
    ) -> None:
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chonk_documents (document_name, content_hash, source_uri, indexed_at, chunk_count)
                VALUES (%s, %s, %s, now(), %s)
                ON CONFLICT (document_name) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    source_uri   = EXCLUDED.source_uri,
                    indexed_at   = EXCLUDED.indexed_at,
                    chunk_count  = EXCLUDED.chunk_count
                """,
                [document_name, content_hash, source_uri, chunk_count],
            )
        self._conn.commit()

    def get_document_hash(self, document_name: str) -> str | None:
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT content_hash FROM chonk_documents WHERE document_name = %s",
                [document_name],
            )
            row = cur.fetchone()
        return row[0] if row else None

    def list_documents(self) -> list[dict]:
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT document_name, content_hash, source_uri, indexed_at, chunk_count "
                "FROM chonk_documents ORDER BY document_name"
            )
            rows = cur.fetchall()
        return [
            {"document_name": r[0], "content_hash": r[1], "source_uri": r[2],
             "indexed_at": r[3], "chunk_count": r[4]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Lifecycle hints (no-ops for PG)
    # ------------------------------------------------------------------

    def rebuild_fts_index(self) -> None:
        pass  # tsvector index is live — no manual rebuild needed

    def preload_embeddings(self) -> None:
        pass  # pgvector HNSW index handles ANN — no RAM preload needed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> PgVectorBackend:
        return self

    def __exit__(self, *_) -> None:
        self.close()
