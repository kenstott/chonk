# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7d268018-663f-42b4-ae26-dab0555ce04d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Store: composed facade over DuckDBVectorBackend and RelationalStore."""
from __future__ import annotations

from pathlib import Path

from ._pool import ThreadLocalDuckDB
from ._relational import RelationalStore
from ._vector import DuckDBVectorBackend

GLOBAL_NAMESPACE = "global"


class Store:
    """Composed storage facade backed by DuckDB.

    Provides a high-level interface to both vector search and
    relational entity storage via a single DuckDB file.

    Usage::

        with Store("index.duckdb") as store:
            store.add_document(chunks, embeddings)
            results = store.search(query_vec, limit=5)
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedding_dim: int = 1024,
        read_only: bool = False,
    ):
        """Create a Store backed by DuckDB.

        Args:
            db_path: Path to DuckDB file, or ":memory:" for an in-memory store.
            embedding_dim: Embedding vector dimension. Must match your model.
            read_only: Open in read-only mode (allows multiple concurrent readers).
        """
        db_path = str(db_path)
        self._db = ThreadLocalDuckDB(db_path, read_only=read_only)
        self.vector = DuckDBVectorBackend(self._db, embedding_dim=embedding_dim)
        if read_only:
            self.relational = None  # type: ignore
            return
        relational_url = (
            f"duckdb:///{db_path}" if db_path != ":memory:" else "duckdb://"
        )
        try:
            self.relational = RelationalStore(relational_url)
            self.relational.init_schema()
        except Exception as e:
            # duckdb-engine may not be installed; relational features are optional.
            # Install with: pip install duckdb-engine
            import logging
            logging.getLogger(__name__).warning(
                f"RelationalStore init failed (entity features unavailable): {e}. "
                "Install duckdb-engine for full SQLAlchemy+DuckDB support."
            )
            self.relational = None  # type: ignore

    def add_document(
        self,
        chunks: list,
        embeddings,
        namespace: str | None = None,
        source_id: str | None = None,
        domain_id: str | None = None,
    ) -> None:
        """Add chunks with embeddings. embeddings is np.ndarray shape (n, dim).

        Args:
            namespace: Optional partition key (e.g. "__base__" or a project ID).
                       None means no namespace — backwards-compatible default.
            source_id: Optional source registry ID for the originating source.
            domain_id: Optional domain registry ID (denormalization of source_id → domain).
        """
        self.vector.add_chunks(chunks, embeddings, namespace=namespace, source_id=source_id, domain_id=domain_id)

    def search(
        self,
        query_embedding,
        limit: int = 5,
        query_text: str | None = None,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_ids: list[str] | None = None,
    ) -> list:
        """Hybrid or pure vector search.

        Args:
            namespaces: If provided, restrict results to rows in these namespaces.
                        None searches all namespaces — backwards-compatible default.
            chunk_types: If provided, restrict results to rows with these chunk_types.
                         None searches all chunk types — backwards-compatible default.
            domain_ids: If provided, restrict results to rows in these domain_ids.
                        Independent of namespaces; when both set both apply (AND).

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        return self.vector.search(
            query_embedding, limit=limit, query_text=query_text,
            namespaces=namespaces, chunk_types=chunk_types, domain_ids=domain_ids,
        )

    # ------------------------------------------------------------------
    # Namespace / domain / source registry
    # ------------------------------------------------------------------

    def register_namespace(
        self,
        namespace_id: str,
        owner: str | None = None,
        description: str | None = None,
    ) -> None:
        """Upsert a namespace record."""
        self.vector._conn.execute(
            """
            INSERT INTO namespaces (namespace_id, owner, description)
            VALUES (?, ?, ?)
            ON CONFLICT (namespace_id) DO UPDATE SET
                owner       = excluded.owner,
                description = excluded.description,
                updated_at  = current_timestamp
            """,
            [namespace_id, owner, description],
        ).fetchall()

    def register_domain(
        self,
        domain_id: str,
        namespace_id: str,
        name: str,
        description: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        """Upsert a domain record."""
        self.vector._conn.execute(
            """
            INSERT INTO domains (domain_id, namespace_id, name, description, parent_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (domain_id) DO UPDATE SET
                namespace_id = excluded.namespace_id,
                name         = excluded.name,
                description  = excluded.description,
                parent_id    = excluded.parent_id,
                updated_at   = current_timestamp
            """,
            [domain_id, namespace_id, name, description, parent_id],
        ).fetchall()

    def register_source(
        self,
        source_id: str,
        domain_id: str,
        type: str,
        uri: str,
        config: dict | None = None,
    ) -> None:
        """Upsert a source record."""
        import json as _json
        config_json = _json.dumps(config) if config is not None else None
        self.vector._conn.execute(
            """
            INSERT INTO sources (source_id, domain_id, type, uri, config)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (source_id) DO UPDATE SET
                domain_id = excluded.domain_id,
                type      = excluded.type,
                uri       = excluded.uri,
                config    = excluded.config
            """,
            [source_id, domain_id, type, uri, config_json],
        ).fetchall()

    def resolve_domain_ids(
        self,
        namespace_domain_pairs: list[tuple[str, str]],
        include_global: bool = True,
    ) -> list[str]:
        """Resolve (namespace_id, domain_name) pairs to domain_ids, including all descendants.

        When include_global=True, always folds in all domains from namespace_id='global'.
        """
        pairs = list(namespace_domain_pairs)
        if include_global:
            global_rows = self.vector._conn.execute(
                "SELECT namespace_id, name FROM domains WHERE namespace_id = ?",
                [GLOBAL_NAMESPACE],
            ).fetchall()
            for row in global_rows:
                if (row[0], row[1]) not in pairs:
                    pairs.append((row[0], row[1]))

        if not pairs:
            return []

        values_placeholders = ", ".join("(?, ?)" for _ in pairs)
        params: list[str] = []
        for ns_id, dom_name in pairs:
            params.extend([ns_id, dom_name])

        sql = f"""
            WITH RECURSIVE domain_tree AS (
                SELECT domain_id FROM domains
                WHERE (namespace_id, name) IN (VALUES {values_placeholders})
                UNION ALL
                SELECT d.domain_id FROM domains d
                JOIN domain_tree dt ON d.parent_id = dt.domain_id
            )
            SELECT DISTINCT domain_id FROM domain_tree
        """
        rows = self.vector._conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]

    def resolve_session(
        self,
        namespace_id: str,
        active_domains: list[str],
        include_global: bool = True,
    ) -> list[str]:
        """Resolve a session to domain_ids.

        A session belongs to one namespace. active_domains are the domain names
        within that namespace the user has activated. Global namespace domains
        are always folded in when include_global=True.

        Example::

            domain_ids = store.resolve_session("user:alice", ["my_notes", "finance"])
            results = store.search(query_vec, domain_ids=domain_ids)
        """
        pairs = [(namespace_id, domain) for domain in active_domains]
        return self.resolve_domain_ids(pairs, include_global=include_global)

    def delete_domain(self, domain_id: str) -> int:
        """Delete all chunks for a domain_id.

        Also deletes associated chunk_entities and svo_triples rows.

        Returns:
            Number of chunks deleted from embeddings.
        """
        conn = self.vector._conn
        count_before = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE domain_id = ?",
            [domain_id],
        ).fetchone()[0]

        # Collect chunk_ids for cascade deletes
        chunk_ids = [
            r[0] for r in conn.execute(
                "SELECT chunk_id FROM embeddings WHERE domain_id = ?",
                [domain_id],
            ).fetchall()
        ]

        if chunk_ids:
            placeholders = ", ".join("?" * len(chunk_ids))
            try:
                conn.execute(
                    f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
            except Exception:
                pass
            try:
                conn.execute(
                    f"DELETE FROM svo_triples WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
            except Exception:
                pass

        conn.execute(
            "DELETE FROM embeddings WHERE domain_id = ?",
            [domain_id],
        ).fetchall()
        self.vector._fts_dirty = True
        return count_before

    def delete_document(self, document_name: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        return self.vector.delete_by_document(document_name)

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self.vector.count()

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_) -> None:
        self.close()
