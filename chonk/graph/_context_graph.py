# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: cd458f18-a532-4ffc-a977-9ae5ba6b047d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Chunk-level co-occurrence clustering and context graph edge computation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class ContextEdge:
    source_entity_id: str
    target_entity_id: str
    weight: float
    svo_signal: float
    cooccur_signal: float
    cluster_signal: float


@dataclass
class ContextGraphStats:
    entity_count: int = 0
    edge_count: int = 0
    chunk_count: int = 0


def _chunk_fingerprint(chunk_ids: list[str]) -> str:
    key = ",".join(sorted(chunk_ids))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_chunk_clusters(
    conn: Any,  # noqa: ANN401
    namespace: str = "global",
    algorithm: str = "agglomerative",
    min_chunks: int = 10,
    force: bool = False,
) -> dict[str, int]:
    from ..cluster._clusterer import cluster_entities

    rows = conn.execute(
        "SELECT DISTINCT chunk_id FROM chunk_entities WHERE COALESCE(namespace, 'global') = ?",
        [namespace],
    ).fetchall()
    chunk_ids = [r[0] for r in rows]

    if len(chunk_ids) < min_chunks:
        return {}

    fingerprint = _chunk_fingerprint(chunk_ids)

    if not force:
        cache_row = conn.execute(
            "SELECT chunk_fingerprint FROM context_graph_cache WHERE COALESCE(namespace, 'global') = ?",  # noqa: E501
            [namespace],
        ).fetchone()
        if cache_row and cache_row[0] == fingerprint:
            existing = conn.execute(
                "SELECT chunk_id, cluster_id FROM chunk_clusters WHERE COALESCE(namespace, 'global') = ?",  # noqa: E501
                [namespace],
            ).fetchall()
            if existing:
                return {r[0]: r[1] for r in existing}

    # Build {chunk_id: [entity_id, ...]}
    ce_rows = conn.execute(
        "SELECT chunk_id, entity_id FROM chunk_entities WHERE COALESCE(namespace, 'global') = ?",
        [namespace],
    ).fetchall()
    chunk_to_entities: dict[str, list[str]] = defaultdict(list)
    for chunk_id, entity_id in ce_rows:
        chunk_to_entities[chunk_id].append(entity_id)

    # Chunk co-occurrence: two chunks co-occur if they share at least one entity
    cooccurrence: dict[tuple[str, str], float] = defaultdict(float)
    entity_to_chunks: dict[str, list[str]] = defaultdict(list)
    for chunk_id, entity_ids in chunk_to_entities.items():
        for eid in entity_ids:
            entity_to_chunks[eid].append(chunk_id)

    for _, chunks in entity_to_chunks.items():
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                a, b = min(chunks[i], chunks[j]), max(chunks[i], chunks[j])
                cooccurrence[(a, b)] += 1.0

    cluster_map = cluster_entities(
        dict(cooccurrence),
        chunk_ids,
        algorithm=algorithm,
    )

    # Convert "cluster_NNNN" -> int
    result: dict[str, int] = {}
    for chunk_id, label in cluster_map.items():
        result[chunk_id] = int(label.replace("cluster_", ""))

    conn.execute(
        "DELETE FROM chunk_clusters WHERE namespace = ?",
        [namespace],
    ).fetchall()
    if result:
        conn.executemany(
            "INSERT INTO chunk_clusters (chunk_id, cluster_id, namespace) VALUES (?, ?, ?)",
            [(cid, cid_int, namespace) for cid, cid_int in result.items()],
        )

    return result


def build_context_graph_edges(
    conn: Any,  # noqa: ANN401
    namespace: str = "global",
    min_weight: float = 0.1,
    force: bool = False,
    algorithm: str = "agglomerative",
    min_chunks: int = 10,
) -> ContextGraphStats:
    # Fingerprint from chunk_ids + svo_triples count
    chunk_rows = conn.execute(
        "SELECT chunk_id FROM embeddings WHERE COALESCE(namespace, 'global') = ?",
        [namespace],
    ).fetchall()
    chunk_ids = [r[0] for r in chunk_rows]
    fingerprint = _chunk_fingerprint(chunk_ids)

    try:
        svo_count = conn.execute(
            "SELECT COUNT(*) FROM svo_triples WHERE COALESCE(namespace, 'global') = ?",
            [namespace],
        ).fetchone()[0]
    except Exception:
        svo_count = 0

    full_fingerprint = hashlib.sha256(f"{fingerprint}:{svo_count}".encode()).hexdigest()[:16]

    if not force:
        cache_row = conn.execute(
            "SELECT chunk_fingerprint, entity_count, edge_count "
            "FROM context_graph_cache WHERE namespace = ?",
            [namespace],
        ).fetchone()
        if cache_row and cache_row[0] == full_fingerprint:
            return ContextGraphStats(
                entity_count=cache_row[1],
                edge_count=cache_row[2],
                chunk_count=len(chunk_ids),
            )

    # Build chunk clusters
    chunk_cluster_map = build_chunk_clusters(
        conn, namespace=namespace, algorithm=algorithm, min_chunks=min_chunks, force=force
    )

    # entity -> set of chunks
    ce_rows = conn.execute(
        "SELECT chunk_id, entity_id FROM chunk_entities WHERE COALESCE(namespace, 'global') = ?",
        [namespace],
    ).fetchall()
    entity_to_chunks: dict[str, set[str]] = defaultdict(set)
    for chunk_id, entity_id in ce_rows:
        entity_to_chunks[entity_id].add(chunk_id)

    all_entities = sorted(entity_to_chunks.keys())

    # entity -> set of cluster_ids
    entity_to_clusters: dict[str, set[int]] = defaultdict(set)
    for chunk_id, cluster_id in chunk_cluster_map.items():
        for eid in [e for e, chunks in entity_to_chunks.items() if chunk_id in chunks]:
            entity_to_clusters[eid].add(cluster_id)

    # SVO pairs: (min(a,b), max(a,b)) -> True
    svo_pairs: set[tuple[str, str]] = set()
    try:
        svo_rows = conn.execute(
            "SELECT subject_id, object_id FROM svo_triples WHERE COALESCE(namespace, 'global') = ?",
            [namespace],
        ).fetchall()
        for subj, obj in svo_rows:
            svo_pairs.add((min(subj, obj), max(subj, obj)))
    except Exception:
        pass

    # Compute raw signals for each pair
    raw_edges: list[tuple[str, str, float, float, float, float]] = []

    for i in range(len(all_entities)):
        for j in range(i + 1, len(all_entities)):
            a, b = all_entities[i], all_entities[j]

            # SVO signal
            svo_signal = 1.0 if (min(a, b), max(a, b)) in svo_pairs else 0.0

            # Cooccur signal: +0.8 per shared chunk, capped at 0.8
            shared_chunks = entity_to_chunks[a] & entity_to_chunks[b]
            cooccur_signal = min(len(shared_chunks) * 0.8, 0.8)

            # Cluster signal: 0.4 * (|intersection| / |union|)
            clusters_a = entity_to_clusters[a]
            clusters_b = entity_to_clusters[b]
            if clusters_a or clusters_b:
                shared_clusters = len(clusters_a & clusters_b)
                union_clusters = len(clusters_a | clusters_b)
                cluster_signal = (
                    0.4 * (shared_clusters / union_clusters) if union_clusters > 0 else 0.0
                )
            else:
                cluster_signal = 0.0

            raw_weight = svo_signal + cooccur_signal + cluster_signal
            if raw_weight > 0 or svo_signal > 0 or cooccur_signal > 0 or cluster_signal > 0:
                raw_edges.append((a, b, raw_weight, svo_signal, cooccur_signal, cluster_signal))

    # Normalize weights to [0, 1]
    if raw_edges:
        max_raw = max(e[2] for e in raw_edges)
    else:
        max_raw = 0.0

    final_edges: list[tuple[str, str, float, float, float, float]] = []
    for a, b, raw_w, svo_s, cooc_s, clus_s in raw_edges:
        weight = raw_w / max_raw if max_raw > 0 else 0.0
        if weight >= min_weight:
            final_edges.append((a, b, weight, svo_s, cooc_s, clus_s))

    # Store symmetric edges
    conn.execute(
        "DELETE FROM context_graph_edges WHERE namespace = ?",
        [namespace],
    ).fetchall()
    rows_to_insert = []
    for a, b, weight, svo_s, cooc_s, clus_s in final_edges:
        rows_to_insert.append((a, b, namespace, weight, svo_s, cooc_s, clus_s))
        rows_to_insert.append((b, a, namespace, weight, svo_s, cooc_s, clus_s))

    if rows_to_insert:
        conn.executemany(
            "INSERT INTO context_graph_edges "
            "(source_entity_id, target_entity_id, namespace, weight, svo_signal, cooccur_signal, cluster_signal) "  # noqa: E501
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows_to_insert,
        )

    edge_count = len(final_edges) * 2
    entity_count = len(all_entities)

    # Upsert cache — store the full_fingerprint so future calls can detect svo changes
    conn.execute(
        """
        INSERT INTO context_graph_cache (namespace, chunk_fingerprint, entity_count, edge_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (namespace) DO UPDATE SET
            chunk_fingerprint = excluded.chunk_fingerprint,
            entity_count      = excluded.entity_count,
            edge_count        = excluded.edge_count,
            created_at        = now()
        """,
        [namespace, full_fingerprint, entity_count, edge_count],
    ).fetchall()

    return ContextGraphStats(
        entity_count=entity_count,
        edge_count=edge_count,
        chunk_count=len(chunk_ids),
    )


def build_context_graph_all_namespaces(
    conn: Any,  # noqa: ANN401
    min_weight: float = 0.1,
    force: bool = False,
    algorithm: str = "agglomerative",
    min_chunks: int = 10,
) -> dict[str, ContextGraphStats]:
    """Build context graph edges for every namespace present in chunk_entities.

    Returns a mapping of namespace -> ContextGraphStats.
    """
    rows = conn.execute(
        "SELECT DISTINCT COALESCE(namespace, 'global') FROM chunk_entities"
    ).fetchall()
    namespaces = [r[0] for r in rows]
    results: dict[str, ContextGraphStats] = {}
    for ns in namespaces:
        results[ns] = build_context_graph_edges(
            conn,
            namespace=ns,
            min_weight=min_weight,
            force=force,
            algorithm=algorithm,
            min_chunks=min_chunks,
        )
    return results


def get_context_graph_edges(
    conn: Any,  # noqa: ANN401
    entity_id: str,
    namespace: str = "global",
    min_weight: float = 0.1,
) -> list[ContextEdge]:
    import logging

    try:
        has_edges = conn.execute(
            "SELECT 1 FROM context_graph_edges WHERE namespace = ? LIMIT 1",
            [namespace],
        ).fetchone()
    except Exception:
        has_edges = None

    if not has_edges:
        logging.getLogger(__name__).debug(
            "context graph not built for namespace %r — skipping expansion; "
            "call build_context_graph() to enable",
            namespace,
        )
        return []

    rows = conn.execute(
        """
        SELECT target_entity_id, weight, svo_signal, cooccur_signal, cluster_signal
        FROM context_graph_edges
        WHERE source_entity_id = ? AND namespace = ? AND weight >= ?
        ORDER BY weight DESC
        """,
        [entity_id, namespace, min_weight],
    ).fetchall()
    return [
        ContextEdge(
            source_entity_id=entity_id,
            target_entity_id=r[0],
            weight=r[1],
            svo_signal=r[2],
            cooccur_signal=r[3],
            cluster_signal=r[4],
        )
        for r in rows
    ]
