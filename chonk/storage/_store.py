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
                Multiple search sessions from the same user should open the
                namespace DB with read_only=True to avoid the single-writer
                DuckDB limit. Only the background Indexer needs write access.
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
        session_fingerprint: str | None = None,
    ) -> None:
        """Add chunks with embeddings. embeddings is np.ndarray shape (n, dim).

        Args:
            namespace: Optional partition key (e.g. "__base__" or a project ID).
                       None means no namespace — backwards-compatible default.
            source_id: Optional source registry ID for the originating source.
            domain_id: Optional domain registry ID (denormalization of source_id → domain).
            session_fingerprint: Optional fingerprint tagging community summary chunks.
        """
        self.vector.add_chunks(chunks, embeddings, namespace=namespace, source_id=source_id, domain_id=domain_id, session_fingerprint=session_fingerprint)

    @staticmethod
    def session_fingerprint(domain_ids: list[str]) -> str:
        """Stable hex fingerprint of a sorted domain_ids set."""
        import hashlib
        key = ",".join(sorted(domain_ids))
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def community_cache_valid(self, fingerprint: str, domain_ids: list[str]) -> bool:
        """True if fingerprint exists in cache AND chunk_count matches current count."""
        row = self.vector._conn.execute(
            "SELECT chunk_count FROM community_cache WHERE fingerprint = ?",
            [fingerprint],
        ).fetchone()
        if row is None:
            return False
        current_count = self.vector._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE domain_id IN ({})".format(
                ", ".join("?" * len(domain_ids))
            ),
            domain_ids,
        ).fetchone()[0]
        return row[0] == current_count

    def write_community_cache(self, fingerprint: str, domain_ids: list[str]) -> None:
        """Record a community cache entry after building communities for domain_ids."""
        import json as _json
        chunk_count = self.vector._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE domain_id IN ({})".format(
                ", ".join("?" * len(domain_ids))
            ),
            domain_ids,
        ).fetchone()[0]
        self.vector._conn.execute(
            """
            INSERT INTO community_cache (fingerprint, domain_ids, chunk_count)
            VALUES (?, ?, ?)
            ON CONFLICT (fingerprint) DO UPDATE SET
                domain_ids  = excluded.domain_ids,
                chunk_count = excluded.chunk_count,
                created_at  = current_timestamp
            """,
            [fingerprint, _json.dumps(sorted(domain_ids)), chunk_count],
        ).fetchall()

    def invalidate_community_cache(self, domain_id: str) -> int:
        """Delete cache entries that include domain_id. Returns count deleted."""
        rows = self.vector._conn.execute(
            "SELECT fingerprint, domain_ids FROM community_cache"
        ).fetchall()
        import json as _json
        to_delete = [
            r[0] for r in rows
            if domain_id in _json.loads(r[1])
        ]
        if not to_delete:
            return 0
        placeholders = ", ".join("?" * len(to_delete))
        # Delete stale community summary chunks
        self.vector._conn.execute(
            f"DELETE FROM embeddings WHERE session_fingerprint IN ({placeholders})",
            to_delete,
        ).fetchall()
        self.vector._conn.execute(
            f"DELETE FROM community_cache WHERE fingerprint IN ({placeholders})",
            to_delete,
        ).fetchall()
        self.vector._fts_dirty = True
        return len(to_delete)

    def search(
        self,
        query_embedding,
        limit: int = 5,
        query_text: str | None = None,
        namespaces: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_ids: list[str] | None = None,
        session_fingerprint: str | None = None,
    ) -> list:
        """Hybrid or pure vector search.

        Args:
            namespaces: If provided, restrict results to rows in these namespaces.
                        None searches all namespaces — backwards-compatible default.
            chunk_types: If provided, restrict results to rows with these chunk_types.
                         None searches all chunk types — backwards-compatible default.
            domain_ids: If provided, restrict results to rows in these domain_ids.
                        Independent of namespaces; when both set both apply (AND).
            session_fingerprint: If provided, restrict results to rows with this
                        session_fingerprint. Used to filter community summary chunks.

        Returns:
            List of (chunk_id, score, DocumentChunk).
        """
        return self.vector.search(
            query_embedding, limit=limit, query_text=query_text,
            namespaces=namespaces, chunk_types=chunk_types, domain_ids=domain_ids,
            session_fingerprint=session_fingerprint,
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
                updated_at  = now()
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
                updated_at   = now()
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
        domains_table = "all_domains" if getattr(self.vector, "_global_attached", False) else "domains"
        pairs = list(namespace_domain_pairs)
        if include_global:
            global_rows = self.vector._conn.execute(
                f"SELECT namespace_id, name FROM {domains_table} WHERE namespace_id = ?",
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
                SELECT domain_id FROM {domains_table}
                WHERE (namespace_id, name) IN (VALUES {values_placeholders})
                UNION ALL
                SELECT d.domain_id FROM {domains_table} d
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

        Search sessions should open the store read-only so multiple concurrent
        sessions never conflict with each other or with a background Indexer::

            # Any number of these can run concurrently — read-only, no conflict.
            store = Store("user_alice.duckdb", read_only=True)
            store.attach_global("global.duckdb")
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

    # ------------------------------------------------------------------
    # Global attach / detach
    # ------------------------------------------------------------------

    def attach_global(self, global_db_path: str | Path) -> None:
        """Attach a global read-only DuckDB and create union views.

        After calling this, all read queries transparently span both this
        store's tables and the global store's tables. Write queries always
        target only this store's base tables.

        Creates views: all_embeddings, all_chunk_entities, all_svo_triples,
        all_domains, all_sources, all_namespaces.
        """
        from ._schema import CHUNK_ENTITIES_DDL, CHUNK_ENTITIES_MIGRATE_NAMESPACE

        conn = self.vector._conn

        # Ensure lazily-created local tables exist so views can reference them.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS svo_triples ("
            "  chunk_id   VARCHAR,"
            "  subject_id VARCHAR NOT NULL,"
            "  verb       VARCHAR NOT NULL,"
            "  object_id  VARCHAR NOT NULL,"
            "  confidence FLOAT   NOT NULL DEFAULT 1.0,"
            "  namespace  VARCHAR"
            ")"
        ).fetchall()
        conn.execute(CHUNK_ENTITIES_DDL).fetchall()
        conn.execute(CHUNK_ENTITIES_MIGRATE_NAMESPACE).fetchall()

        conn.execute(f"ATTACH '{global_db_path}' AS global_db (READ_ONLY)")

        def _global_has(table: str) -> bool:
            return conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_catalog = 'global_db' AND table_name = ?",
                [table],
            ).fetchone()[0] > 0

        # ── all_embeddings ──────────────────────────────────────────────────
        if _global_has("embeddings"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_embeddings AS
                SELECT chunk_id, document_name, section, chunk_index, content,
                       breadcrumb, chunk_type, source_offset, source_length,
                       namespace, source_detail, source_id, domain_id,
                       session_fingerprint, embedding
                FROM embeddings
                UNION ALL
                SELECT chunk_id, document_name, section, chunk_index, content,
                       breadcrumb, chunk_type, source_offset, source_length,
                       namespace, source_detail, source_id, domain_id,
                       session_fingerprint, embedding
                FROM global_db.embeddings
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_embeddings AS
                SELECT chunk_id, document_name, section, chunk_index, content,
                       breadcrumb, chunk_type, source_offset, source_length,
                       namespace, source_detail, source_id, domain_id,
                       session_fingerprint, embedding
                FROM embeddings
            """)

        # ── all_chunk_entities ──────────────────────────────────────────────
        if _global_has("chunk_entities"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_chunk_entities AS
                SELECT chunk_id, entity_id, frequency, positions_json, score, namespace
                FROM chunk_entities
                UNION ALL
                SELECT chunk_id, entity_id, frequency, positions_json, score, namespace
                FROM global_db.chunk_entities
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_chunk_entities AS
                SELECT chunk_id, entity_id, frequency, positions_json, score, namespace
                FROM chunk_entities
            """)

        # ── all_svo_triples ─────────────────────────────────────────────────
        if _global_has("svo_triples"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_svo_triples AS
                SELECT chunk_id, subject_id, verb, object_id, confidence, namespace
                FROM svo_triples
                UNION ALL
                SELECT chunk_id, subject_id, verb, object_id, confidence, namespace
                FROM global_db.svo_triples
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_svo_triples AS
                SELECT chunk_id, subject_id, verb, object_id, confidence, namespace
                FROM svo_triples
            """)

        # ── all_domains ─────────────────────────────────────────────────────
        if _global_has("domains"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_domains AS
                SELECT domain_id, namespace_id, name, description, parent_id,
                       created_at, updated_at
                FROM domains
                UNION ALL
                SELECT domain_id, namespace_id, name, description, parent_id,
                       created_at, updated_at
                FROM global_db.domains
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_domains AS
                SELECT domain_id, namespace_id, name, description, parent_id,
                       created_at, updated_at
                FROM domains
            """)

        # ── all_sources ─────────────────────────────────────────────────────
        if _global_has("sources"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_sources AS
                SELECT source_id, domain_id, type, uri, config, last_crawled
                FROM sources
                UNION ALL
                SELECT source_id, domain_id, type, uri, config, last_crawled
                FROM global_db.sources
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_sources AS
                SELECT source_id, domain_id, type, uri, config, last_crawled
                FROM sources
            """)

        # ── all_namespaces ──────────────────────────────────────────────────
        if _global_has("namespaces"):
            conn.execute("""
                CREATE OR REPLACE VIEW all_namespaces AS
                SELECT namespace_id, owner, description, created_at, updated_at
                FROM namespaces
                UNION ALL
                SELECT namespace_id, owner, description, created_at, updated_at
                FROM global_db.namespaces
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW all_namespaces AS
                SELECT namespace_id, owner, description, created_at, updated_at
                FROM namespaces
            """)

        self.vector._global_attached = True

    def detach_global(self) -> None:
        """Drop the union views and detach the global DB."""
        conn = self.vector._conn
        for view in ("all_embeddings", "all_chunk_entities", "all_svo_triples",
                     "all_domains", "all_sources", "all_namespaces"):
            conn.execute(f"DROP VIEW IF EXISTS {view}")
        conn.execute("DETACH global_db")
        self.vector._global_attached = False

    def delete_document(self, document_name: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        return self.vector.delete_by_document(document_name)

    # ── Entity descriptions ───────────────────────────────────────────────────

    # Priority order — lower number wins; never overwrite with higher number.
    _DESC_PRIORITY: dict[str, int] = {"user": 0, "schema": 1, "llm": 2}

    def update_entity_description(
        self,
        entity_id: str,
        description: str,
        namespace: str = GLOBAL_NAMESPACE,
    ) -> None:
        """Update an entity description unconditionally, marking source as 'user'.

        Use this for client curation — always wins over 'llm' and 'schema' sources.
        """
        conn = self.vector._conn
        conn.execute(
            """
            INSERT INTO entity_descriptions (entity_id, namespace, description, source, updated_at)
            VALUES (?, ?, ?, 'user', now())
            ON CONFLICT (entity_id, namespace) DO UPDATE SET
                description = excluded.description,
                source      = 'user',
                updated_at  = now()
            """,
            [entity_id, namespace, description],
        )

    def upsert_entity_description(
        self,
        entity_id: str,
        description: str,
        source: str = "llm",
        namespace: str = GLOBAL_NAMESPACE,
    ) -> None:
        """Upsert one entity description.

        Lower-priority sources never overwrite higher-priority ones:
        ``user`` > ``schema`` > ``llm``.
        """
        conn = self.vector._conn
        existing = conn.execute(
            "SELECT source FROM entity_descriptions WHERE entity_id = ? AND namespace = ?",
            [entity_id, namespace],
        ).fetchone()
        if existing:
            existing_pri = self._DESC_PRIORITY.get(existing[0], 99)
            new_pri = self._DESC_PRIORITY.get(source, 99)
            if new_pri >= existing_pri:
                return
        conn.execute(
            """
            INSERT INTO entity_descriptions (entity_id, namespace, description, source, updated_at)
            VALUES (?, ?, ?, ?, now())
            ON CONFLICT (entity_id, namespace) DO UPDATE SET
                description = excluded.description,
                source      = excluded.source,
                updated_at  = now()
            """,
            [entity_id, namespace, description, source],
        )

    def upsert_entity_descriptions_batch(
        self,
        descriptions: dict[str, str],
        source: str = "llm",
        namespace: str = GLOBAL_NAMESPACE,
    ) -> int:
        """Upsert multiple entity descriptions. Returns count written."""
        written = 0
        for entity_id, description in descriptions.items():
            before = self.vector._conn.execute(
                "SELECT source FROM entity_descriptions WHERE entity_id = ? AND namespace = ?",
                [entity_id, namespace],
            ).fetchone()
            if before:
                existing_pri = self._DESC_PRIORITY.get(before[0], 99)
                new_pri = self._DESC_PRIORITY.get(source, 99)
                if new_pri >= existing_pri:
                    continue
            self.vector._conn.execute(
                """
                INSERT INTO entity_descriptions (entity_id, namespace, description, source, updated_at)
                VALUES (?, ?, ?, ?, now())
                ON CONFLICT (entity_id, namespace) DO UPDATE SET
                    description = excluded.description,
                    source      = excluded.source,
                    updated_at  = now()
                """,
                [entity_id, namespace, description, source],
            )
            written += 1
        return written

    def get_entity_descriptions(
        self,
        entity_ids: list[str],
        namespace: str = GLOBAL_NAMESPACE,
    ) -> dict[str, str]:
        """Return ``{entity_id: description}`` for the given IDs."""
        if not entity_ids:
            return {}
        conn = self.vector._conn
        placeholders = ", ".join(["?" for _ in entity_ids])
        rows = conn.execute(
            f"SELECT entity_id, description FROM entity_descriptions "
            f"WHERE entity_id IN ({placeholders}) AND namespace = ?",
            entity_ids + [namespace],
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_chunk_entity_ids(self, chunk_id: str) -> list[str]:
        """Return all entity_ids associated with a chunk."""
        conn = self.vector._conn
        rows = conn.execute(
            "SELECT entity_id FROM chunk_entities WHERE chunk_id = ?",
            [chunk_id],
        ).fetchall()
        return [r[0] for r in rows]

    # ── Entity aliases ────────────────────────────────────────────────────────

    def add_entity_alias(
        self,
        alias: str,
        entity_id: str,
        source: str = "llm",
        namespace: str = GLOBAL_NAMESPACE,
    ) -> None:
        """Register *alias* as an alternate name for *entity_id*.

        An alias is unique per namespace — the first registration wins unless
        source is ``'user'``, which always overwrites.
        """
        conn = self.vector._conn
        if source == "user":
            conn.execute(
                """
                INSERT INTO entity_aliases (alias, entity_id, namespace, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (alias, namespace) DO UPDATE SET
                    entity_id = excluded.entity_id,
                    source    = excluded.source
                """,
                [alias, entity_id, namespace, source],
            )
        else:
            conn.execute(
                """
                INSERT INTO entity_aliases (alias, entity_id, namespace, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (alias, namespace) DO NOTHING
                """,
                [alias, entity_id, namespace, source],
            )

    def add_entity_aliases_batch(
        self,
        aliases: dict[str, str],
        source: str = "llm",
        namespace: str = GLOBAL_NAMESPACE,
    ) -> int:
        """Register multiple aliases. ``aliases`` maps alias → entity_id.

        Returns count inserted (conflicts silently ignored for non-user sources).
        """
        written = 0
        for alias, entity_id in aliases.items():
            before = self.vector._conn.execute(
                "SELECT source FROM entity_aliases WHERE alias = ? AND namespace = ?",
                [alias, namespace],
            ).fetchone()
            if before and source != "user":
                continue
            self.add_entity_alias(alias, entity_id, source=source, namespace=namespace)
            written += 1
        return written

    def resolve_entity_alias(
        self,
        alias: str,
        namespace: str = GLOBAL_NAMESPACE,
    ) -> str | None:
        """Return the canonical entity_id for *alias*, or None if unknown."""
        row = self.vector._conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias = ? AND namespace = ?",
            [alias, namespace],
        ).fetchone()
        return row[0] if row else None

    def get_entity_aliases(
        self,
        entity_id: str,
        namespace: str = GLOBAL_NAMESPACE,
    ) -> list[str]:
        """Return all aliases registered for *entity_id*."""
        rows = self.vector._conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? AND namespace = ?",
            [entity_id, namespace],
        ).fetchall()
        return [r[0] for r in rows]

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
