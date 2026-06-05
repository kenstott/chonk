# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6b3d8695-16ae-4239-b8b0-344b34fb0249
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""build_community — build and persist a CommunityIndex for a namespace."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _check_existing_count(db_path: Path) -> int:
    """Return the current chunk_communities row count, or 0 if the table is absent."""
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()
        return row[0] if row else 0
    except duckdb.Error:  # noqa: BLE001  # table absent == not yet built; 0 is correct
        return 0
    finally:
        con.close()


def _load_chunks(
    db_path: Path,
    namespace_id: str | None,
) -> tuple[list, Any]:
    """Open a read-only connection, load embedding rows, and return (rows, connection).

    Caller is responsible for closing the connection.
    """
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    if namespace_id:
        rows = con.execute(
            "SELECT chunk_id, content, breadcrumb, embedding FROM embeddings "
            "WHERE embedding IS NOT NULL AND namespace = ?",
            [namespace_id],
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT chunk_id, content, breadcrumb, embedding FROM embeddings "
            "WHERE embedding IS NOT NULL"
        ).fetchall()
    return rows, con


def _load_entity_edges(
    con,  # duckdb.DuckDBPyConnection
    rows: list,
) -> list[tuple[int, int, float]]:
    """Build cross-document entity bridge edges from chunk_entities.

    Args:
        con: Open read-only DuckDB connection.
        rows: Embedding rows as returned by _load_chunks (chunk_id at index 0).

    Returns:
        List of (index_a, index_b, weight) tuples for cross-document pairs.
    """
    from collections import defaultdict

    ce_rows = con.execute(
        "SELECT entity_id, chunk_id FROM chunk_entities WHERE frequency > 0"
    ).fetchall()

    if not ce_rows:
        return []

    chunk_ids_list = [r[0] for r in rows]
    id_to_idx = {cid: i for i, cid in enumerate(chunk_ids_list)}

    entity_to_chunks: dict[str, list[int]] = defaultdict(list)
    for eid, cid in ce_rows:
        if cid in id_to_idx:
            entity_to_chunks[eid].append(id_to_idx[cid])

    extra_edges: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for idxs in entity_to_chunks.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                doc_a = chunk_ids_list[ia].rsplit("_", 2)[0]
                doc_b = chunk_ids_list[ib].rsplit("_", 2)[0]
                if doc_a != doc_b:
                    key = (min(ia, ib), max(ia, ib))
                    if key not in seen:
                        seen.add(key)
                        extra_edges.append((ia, ib, 1.0))

    return extra_edges


def _embed_breadcrumbs(
    breadcrumbs: list[str],
    embed_model,  # str or SentenceTransformer instance
):
    """Embed breadcrumb strings and return a float32 ndarray, or None if all empty.

    Args:
        breadcrumbs: Per-chunk breadcrumb strings (may be empty).
        embed_model: SentenceTransformer model name or instance.

    Returns:
        numpy ndarray of shape (n, dim) or None when no non-empty breadcrumbs exist.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    non_empty = [bc for bc in breadcrumbs if bc.strip()]
    if not non_empty:
        return None

    model = SentenceTransformer(embed_model) if isinstance(embed_model, str) else embed_model
    bc_texts = [bc if bc.strip() else "" for bc in breadcrumbs]
    all_bc_vecs = np.array(
        model.encode(bc_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=256),
        dtype="float32",
    )
    for i, bc in enumerate(breadcrumbs):
        if not bc.strip():
            all_bc_vecs[i] = 0.0
    return all_bc_vecs


def build_community(
    db_path: str | Path,
    embed_model,  # str or SentenceTransformer instance
    *,
    namespace_id: str | None = None,
    alpha: float = 0.2,
    sim_threshold: float = 0.6,
    force: bool = False,
) -> int:
    """Build and persist a CommunityIndex.

    Reads chunk embeddings from *db_path*, builds the community graph,
    persists community tables back to the same DB.

    Args:
        db_path: Path to the namespace DuckDB file.
        embed_model: SentenceTransformer model name or instance (for breadcrumb embedding).
        namespace_id: Optional — if set, filters embeddings to this namespace only.
        alpha: Breadcrumb heading weight (0.0 = content only, 1.0 = heading only).
        sim_threshold: Cosine similarity threshold for community graph edges.
        force: If False and chunk_communities already populated, skip.

    Returns:
        Number of communities built.
    """
    import numpy as np

    from ._index import CommunityIndex

    db_path = Path(db_path)

    if not force:
        n = _check_existing_count(db_path)
        if n > 0:
            return n

    rows, con_ro = _load_chunks(db_path, namespace_id)

    extra_edges = _load_entity_edges(con_ro, rows)
    con_ro.close()

    if not rows:
        return 0

    chunk_ids = [r[0] for r in rows]
    chunk_texts = [r[1] or "" for r in rows]
    breadcrumbs = [r[2] or "" for r in rows]
    content_vecs = np.array([r[3] for r in rows], dtype="float32")

    heading_vecs = _embed_breadcrumbs(breadcrumbs, embed_model)

    idx = CommunityIndex.build(
        chunk_ids=chunk_ids,
        content_vecs=content_vecs,
        chunk_texts=chunk_texts,
        heading_vecs=heading_vecs,
        alpha=alpha,
        sim_threshold=sim_threshold,
        extra_edges=extra_edges or None,
    )

    idx.persist(db_path)
    return idx.community_count()
