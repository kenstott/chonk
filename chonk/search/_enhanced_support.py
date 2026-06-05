# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 213e7f8a-5158-4869-8346-d12e39dc5fb1
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Module-level helpers and RetrievalTrace dataclass for EnhancedSearch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..community._index import CommunityIndex
    from ..graph._index import RelationshipIndex
    from ..ner._index import EntityIndex
    from ..storage._store import Store


def _load_entity_index_from_store(conn) -> EntityIndex | None:
    import duckdb

    try:
        count = conn.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()[0]
    except (duckdb.CatalogException, duckdb.BinderException):
        return None  # table absent — entity index not built yet (expected)
    if count == 0:
        return None
    from ..ner._index import EntityIndex as _EI
    return _EI.load_from_db(conn)


def _load_relationship_index_from_store(conn) -> RelationshipIndex | None:
    import duckdb

    try:
        count = conn.execute("SELECT COUNT(*) FROM svo_triples").fetchone()[0]
    except (duckdb.CatalogException, duckdb.BinderException):
        return None  # table absent — relationship index not built yet (expected)
    if count == 0:
        return None
    from ..graph._index import RelationshipIndex as _RI
    return _RI.load_from_db(conn)


def _load_community_index_from_store(store: Store) -> CommunityIndex | None:
    """Auto-load CommunityIndex from the store's DuckDB if community tables exist."""
    if store._db is None:
        return None  # non-DuckDB backend (e.g. pgvector) — no local community tables
    db_path = store._db._db_path
    if db_path == ":memory:":
        return None
    from ..community._index import CommunityIndex as _CI

    conn = store.vector._conn
    tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name IN ('chunk_communities', 'communities')"
    ).fetchall()}
    if "chunk_communities" not in tables or "communities" not in tables:
        return None
    count = conn.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()[0]
    if count == 0:
        return None
    return _CI.from_db(db_path)


def _tally_provenance(results: list) -> dict[str, int]:
    tally: dict[str, int] = {}
    for sc in results:
        tally[sc.provenance] = tally.get(sc.provenance, 0) + 1
    return tally


@dataclass
class RetrievalTrace:
    """Per-query record of which graph assets contributed to the retrieval cohort.

    Populated by :meth:`EnhancedSearch.search` when ``return_trace=True``.
    All list fields contain chunk IDs or entity IDs as strings.
    """

    mode: str = "vector_first"
    query_entities: list[str] = field(default_factory=list)

    # vector_first path
    seed_chunk_ids: list[str] = field(default_factory=list)
    structural_chunk_ids: list[str] = field(default_factory=list)
    entity_expanded_chunk_ids: list[str] = field(default_factory=list)
    cluster_expanded_chunk_ids: list[str] = field(default_factory=list)
    entity_embed_expanded_chunk_ids: list[str] = field(default_factory=list)

    # graph_first path
    graph_traversal_entities: list[str] = field(default_factory=list)
    svo_triples_count: int = 0

    # global path
    community_chunk_ids: list[str] = field(default_factory=list)

    # entity-ref expansion (all modes)
    ref_expansion_missing: list[str] = field(default_factory=list)
    ref_expansion_found: list[str] = field(default_factory=list)
    ref_expansion_chunks_added: int = 0
    context_graph_expansion_chunks_added: int = 0

    # final result summary
    pool_size: int = 0
    final_provenance: dict[str, int] = field(default_factory=dict)
