# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6b3d8695-16ae-4239-b8b0-344b34fb0249
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""build_community — build and persist a CommunityIndex for a namespace."""
from __future__ import annotations

from pathlib import Path


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
    import duckdb
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from ._index import CommunityIndex

    db_path = Path(db_path)

    if not force:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            row = con.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()
            n = row[0] if row else 0
        except Exception:
            n = 0
        finally:
            con.close()
        if n > 0:
            return n

    # Load chunks
    con_ro = duckdb.connect(str(db_path), read_only=True)
    where = "WHERE embedding IS NOT NULL"
    if namespace_id:
        where += f" AND namespace = '{namespace_id}'"
    rows = con_ro.execute(
        f"SELECT chunk_id, content, breadcrumb, embedding FROM embeddings {where}"
    ).fetchall()

    # Load entity bridge edges for cross-document community links
    extra_edges: list[tuple[int, int, float]] = []
    try:
        ce_rows = con_ro.execute(
            "SELECT entity_id, chunk_id FROM chunk_entities WHERE frequency > 0"
        ).fetchall()
        if ce_rows:
            from collections import defaultdict
            chunk_ids_list = [r[0] for r in rows]
            id_to_idx = {cid: i for i, cid in enumerate(chunk_ids_list)}
            entity_to_chunks: dict[str, list[int]] = defaultdict(list)
            for eid, cid in ce_rows:
                if cid in id_to_idx:
                    entity_to_chunks[eid].append(id_to_idx[cid])
            seen: set[tuple[int, int]] = set()
            for eid, idxs in entity_to_chunks.items():
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
    except Exception:
        pass
    finally:
        con_ro.close()

    if not rows:
        return 0

    chunk_ids = [r[0] for r in rows]
    chunk_texts = [r[1] or "" for r in rows]
    breadcrumbs = [r[2] or "" for r in rows]
    content_vecs = np.array([r[3] for r in rows], dtype="float32")

    # Embed breadcrumbs
    heading_vecs = None
    non_empty_bc = [bc for bc in breadcrumbs if bc.strip()]
    if non_empty_bc:
        if isinstance(embed_model, str):
            model = SentenceTransformer(embed_model)
        else:
            model = embed_model
        bc_texts = [bc if bc.strip() else "" for bc in breadcrumbs]
        import numpy as _np
        all_bc_vecs = _np.array(
            model.encode(bc_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=256),
            dtype="float32",
        )
        for i, bc in enumerate(breadcrumbs):
            if not bc.strip():
                all_bc_vecs[i] = 0.0
        heading_vecs = all_bc_vecs

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
