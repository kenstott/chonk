# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Enhanced semantic similarity search with 4-dimensional cohort assembly.

Interface is identical to standard search::

    results: list[ScoredChunk] = search.search(query_embedding, k=5)

Internally assembles the cohort across four dimensions:
  1. Seed     — vector similarity (FAISS / DuckDB VSS)
  2. Structural — next/prev/parent chunk expansion (via chunk_index adjacency)
  3. Entity   — entity-adjacent chunks from EntityIndex
  4. Cluster  — cluster-neighbour chunks (budget-limited)

Each dimension can be independently enabled or disabled via constructor args,
supporting incremental ablation benchmarking.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

from ..models import DocumentChunk, ScoredChunk

if TYPE_CHECKING:
    from ..storage._store import Store
    from ..ner._index import EntityIndex
    from ..cluster._map import ClusterMap


class EnhancedSearch:
    """4-dimensional cohort assembler.

    Args:
        store: A chonk Store (DuckDBVectorBackend under the hood).
        entity_index: Optional populated EntityIndex for entity expansion.
        cluster_map: Optional populated ClusterMap for cluster expansion.
        seed_pool_multiplier: Seed pool = k * multiplier (default 3).
        entity_expansion_top_n: Max chunks per entity in entity expansion (default 3).
        cluster_budget: Max cluster-adjacent candidates (default 2 * k).
        lambda_diversity: MMR redundancy penalty weight (default 0.3).
        relevance_weight: Composite score weight for relevance (default 0.5).
        priority_weight: Composite score weight for source priority (default 0.2).
        coverage_weight: Composite score weight for marginal coverage (default 0.3).
        structural_expansion: Enable next/prev/parent expansion (default True).
        entity_expansion: Enable entity adjacency expansion (default True).
        cluster_expansion: Enable cluster adjacency expansion (default True).
    """

    # Source priority constants (from spec)
    _PRIORITY = {
        "seed": 1.0,
        "structural": 0.9,
        "entity_adjacent": 0.7,
        "cluster_adjacent": 0.5,
    }

    def __init__(
        self,
        store: "Store",
        entity_index: "EntityIndex | None" = None,
        cluster_map: "ClusterMap | None" = None,
        seed_pool_multiplier: int = 3,
        entity_expansion_top_n: int = 3,
        cluster_budget: int | None = None,
        lambda_diversity: float = 0.3,
        relevance_weight: float = 0.5,
        priority_weight: float = 0.2,
        coverage_weight: float = 0.3,
        structural_expansion: bool = True,
        entity_expansion: bool = True,
        cluster_expansion: bool = True,
        entity_embedding_expansion: bool = False,
        entity_embeddings=None,
        entity_embedding_ids: list[str] | None = None,
        ner_fn: Callable[[str], list[str]] | None = None,
        embed_fn: Callable[[list[str]], "np.ndarray"] | None = None,
        entity_embedding_top_k: int = 10,
        entity_ref_expansion: bool = False,
        entity_ref_expansion_k: int = 20,
        entity_ref_expansion_per_k: int | None = None,
        entity_ref_expansion_min_sim: float | None = None,
        query_ner_fn: Callable[[str], list[str]] | None = None,
        lane_entity_min_sim: float | None = None,
    ):
        self._store = store
        self._entity_index = entity_index
        self._cluster_map = cluster_map
        self._seed_multiplier = seed_pool_multiplier
        self._entity_top_n = entity_expansion_top_n
        self._cluster_budget = cluster_budget  # resolved to 2*k at call time if None
        self._lambda = lambda_diversity
        self._rw = relevance_weight
        self._pw = priority_weight
        self._cw = coverage_weight
        self._structural = structural_expansion
        self._entity = entity_expansion
        self._cluster = cluster_expansion
        self._entity_embed = entity_embedding_expansion
        self._entity_embeddings = entity_embeddings
        self._entity_embedding_ids = entity_embedding_ids
        self._ner_fn = ner_fn
        self._embed_fn = embed_fn
        self._entity_embed_top_k = entity_embedding_top_k
        self._entity_ref_expansion = entity_ref_expansion
        self._entity_ref_expansion_k = entity_ref_expansion_k
        self._entity_ref_expansion_per_k = entity_ref_expansion_per_k
        self._entity_ref_expansion_min_sim = entity_ref_expansion_min_sim
        self._query_ner_fn = query_ner_fn
        self._lane_entity_min_sim = lane_entity_min_sim
        self.last_expansion_stats: dict | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding,
        k: int = 5,
        query_text: str | None = None,
        query_entities: list[str] | None = None,
    ) -> list[ScoredChunk]:
        """Assemble a top-k cohort using all enabled expansion dimensions.

        Args:
            query_embedding: np.ndarray shape (dim,) or (1, dim).
            k: Target cohort size.
            query_text: Optional query text for BM25 hybrid seed search.

        Returns:
            Ranked list of up to k ScoredChunk objects.
        """
        seed_limit = k * self._seed_multiplier
        cluster_budget = self._cluster_budget if self._cluster_budget is not None else 2 * k

        # ------ Step 1: Seed -----------------------------------------------
        raw_seeds = self._store.search(query_embedding, limit=seed_limit, query_text=query_text)
        # raw_seeds: list of (chunk_id, score, DocumentChunk)

        # candidate pool: chunk_id -> ScoredChunk
        pool: dict[str, ScoredChunk] = {}
        seed_chunk_ids: list[str] = []

        for chunk_id, score, chunk in raw_seeds:
            if chunk_id not in pool:
                pool[chunk_id] = ScoredChunk(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=score,
                    provenance="seed",
                    embedding=None,
                )
            seed_chunk_ids.append(chunk_id)

        # ------ Step 2: Structural expansion --------------------------------
        if self._structural:
            structural_ids = self._structural_neighbors(seed_chunk_ids)
            for chunk_id, chunk in structural_ids:
                if chunk_id not in pool:
                    pool[chunk_id] = ScoredChunk(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        score=0.0,
                        provenance="structural",
                    )

        # ------ Step 3: Entity expansion ------------------------------------
        entity_source_ids = list(pool.keys())  # seed + structural
        expanded_entity_ids: set[str] = set()
        query_vec_flat = list(query_embedding.flatten()) if self._lane_entity_min_sim is not None else None

        if self._entity and self._entity_index is not None:
            for cid in entity_source_ids:
                for entity_id, _ in self._entity_index.get_entities_for_chunk(cid):
                    expanded_entity_ids.add(entity_id)
                    for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                        entity_id, top_n=self._entity_top_n
                    ):
                        if linked_chunk_id not in pool:
                            if query_vec_flat is not None:
                                emb = self._get_embedding(linked_chunk_id)
                                if emb and self._cosine_similarity(query_vec_flat, emb) < self._lane_entity_min_sim:
                                    continue
                            chunk = self._fetch_chunk(linked_chunk_id)
                            if chunk is not None:
                                pool[linked_chunk_id] = ScoredChunk(
                                    chunk_id=linked_chunk_id,
                                    chunk=chunk,
                                    score=0.0,
                                    provenance="entity_adjacent",
                                    linked_by=entity_id,
                                )

        # ------ Step 4: Cluster expansion (budget-limited) ------------------
        cluster_count = 0
        if self._cluster and self._cluster_map is not None and cluster_budget > 0:
            for entity_id in expanded_entity_ids:
                if cluster_count >= cluster_budget:
                    break
                for neighbor_entity_id in self._cluster_map.get_neighbors(entity_id):
                    if cluster_count >= cluster_budget:
                        break
                    cluster_chunks = self._entity_index.get_chunks_for_entity(
                        neighbor_entity_id, top_n=1
                    ) if self._entity_index else []
                    for linked_chunk_id, _ in cluster_chunks:
                        if linked_chunk_id not in pool:
                            chunk = self._fetch_chunk(linked_chunk_id)
                            if chunk is not None:
                                cluster_id = self._cluster_map.get_cluster(neighbor_entity_id)
                                pool[linked_chunk_id] = ScoredChunk(
                                    chunk_id=linked_chunk_id,
                                    chunk=chunk,
                                    score=0.0,
                                    provenance="cluster_adjacent",
                                    linked_by=neighbor_entity_id,
                                    cluster=cluster_id,
                                )
                                cluster_count += 1

        # ------ Step 5: Entity embedding ANN expansion ----------------------
        if (
            self._entity_embed
            and self._entity_embeddings is not None
            and self._entity_embedding_ids is not None
            and self._entity_index is not None
            and query_text
            and self._ner_fn is not None
            and self._embed_fn is not None
        ):
            import numpy as np
            query_ents = self._ner_fn(query_text)
            if query_ents:
                q_ent_vecs = self._embed_fn(query_ents)  # (n_query_ents, dim)
                scores = q_ent_vecs @ self._entity_embeddings.T  # (n_query_ents, n_entities)
                max_scores = scores.max(axis=0)  # (n_entities,)
                top_indices = np.argsort(-max_scores)[: self._entity_embed_top_k]
                for idx in top_indices:
                    eid = self._entity_embedding_ids[int(idx)]
                    for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                        eid, top_n=2
                    ):
                        if linked_chunk_id not in pool:
                            chunk = self._fetch_chunk(linked_chunk_id)
                            if chunk is not None:
                                pool[linked_chunk_id] = ScoredChunk(
                                    chunk_id=linked_chunk_id,
                                    chunk=chunk,
                                    score=0.0,
                                    provenance="entity_adjacent",
                                    linked_by=eid,
                                )

        # ------ Step 6: Score and select top-k ------------------------------
        candidates = list(pool.values())
        results = self._select_cohort(candidates, query_embedding, k)

        # ------ Step 7: Entity-ref expansion (adaptive post-selection) ------
        self.last_expansion_stats = None
        if self._entity_ref_expansion:
            ents = query_entities
            if ents is None and self._query_ner_fn is not None and query_text:
                ents = self._query_ner_fn(query_text)
            if ents:
                results = self._entity_ref_expand(results, pool, query_embedding, ents, k, query_text)

        return results

    # ------------------------------------------------------------------
    # Entity-ref expansion
    # ------------------------------------------------------------------

    def _entity_ref_expand(
        self,
        results: list[ScoredChunk],
        existing_pool: dict[str, ScoredChunk],
        query_embedding,
        query_entities: list[str],
        k: int,
        query_text: str | None,
    ) -> list[ScoredChunk]:
        """Post-selection: if query entities are absent from top-k, expand via semantic search."""
        result_text = " ".join((sc.chunk.content or "") for sc in results).lower()
        missing = [e for e in query_entities if e.lower() not in result_text]

        if not missing:
            self.last_expansion_stats = {"invoked": False, "missing_entities": []}
            return results

        new_pool = dict(existing_pool)
        found_entities: list[str] = []

        if self._embed_fn is not None:
            # Semantic: per-entity vector search
            if self._entity_ref_expansion_per_k is not None:
                per_k = self._entity_ref_expansion_per_k
            else:
                per_k = max(3, self._entity_ref_expansion_k // max(len(missing), 1))
            min_sim = self._entity_ref_expansion_min_sim
            entity_vecs = self._embed_fn(missing)
            for i, entity in enumerate(missing):
                hits = self._store.search(entity_vecs[i], limit=per_k, query_text=None)
                for chunk_id, score, chunk in hits:
                    if min_sim is not None and score < min_sim:
                        continue
                    if chunk_id not in new_pool:
                        new_pool[chunk_id] = ScoredChunk(
                            chunk_id=chunk_id, chunk=chunk, score=score,
                            provenance="entity_ref_expansion",
                        )
                    if entity not in found_entities:
                        found_entities.append(entity)
        else:
            # Literal fallback
            expanded_seeds = self._store.search(
                query_embedding, limit=self._entity_ref_expansion_k, query_text=query_text
            )
            for chunk_id, score, chunk in expanded_seeds:
                if chunk_id in new_pool:
                    continue
                chunk_text = (chunk.content or "").lower()
                matched = [e for e in missing if e.lower() in chunk_text]
                if matched:
                    new_pool[chunk_id] = ScoredChunk(
                        chunk_id=chunk_id, chunk=chunk, score=score,
                        provenance="entity_ref_expansion",
                    )
                    for e in matched:
                        if e not in found_entities:
                            found_entities.append(e)

        self.last_expansion_stats = {
            "invoked": True,
            "missing_entities": missing,
            "found_entities": found_entities,
            "unresolved_entities": [e for e in missing if e not in found_entities],
            "new_chunks_added": len(new_pool) - len(existing_pool),
        }

        if not found_entities:
            return results

        return self._select_cohort(list(new_pool.values()), query_embedding, k)

    # ------------------------------------------------------------------
    # Structural expansion
    # ------------------------------------------------------------------

    def _structural_neighbors(
        self, seed_chunk_ids: list[str]
    ) -> list[tuple[str, DocumentChunk]]:
        """Pull prev/next chunks by index adjacency from the vector store."""
        results: list[tuple[str, DocumentChunk]] = []
        all_chunks = self._store.vector.get_all_chunks()
        # Build lookup: (document_name, chunk_index) -> (chunk_id, chunk)
        lookup: dict[tuple[str, int], tuple[str, DocumentChunk]] = {}
        id_to_chunk: dict[str, DocumentChunk] = {}
        for chunk in all_chunks:
            # Re-derive chunk_id using the same hash as DuckDBVectorBackend
            from ..storage._vector import DuckDBVectorBackend
            cid = DuckDBVectorBackend._generate_chunk_id(
                chunk.document_name, chunk.chunk_index,
                chunk.embedding_content or chunk.content
            )
            lookup[(chunk.document_name, chunk.chunk_index)] = (cid, chunk)
            id_to_chunk[cid] = chunk

        for seed_id in seed_chunk_ids:
            if seed_id not in id_to_chunk:
                continue
            seed_chunk = id_to_chunk[seed_id]
            doc = seed_chunk.document_name
            idx = seed_chunk.chunk_index
            for neighbor_idx in (idx - 1, idx + 1):
                key = (doc, neighbor_idx)
                if key in lookup:
                    cid, chunk = lookup[key]
                    results.append((cid, chunk))
        return results

    def _fetch_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Fetch a single chunk by ID from the vector store."""
        # DuckDB direct lookup
        try:
            rows = self._store.vector._conn.execute(
                "SELECT document_name, content, section, chunk_index, "
                "source_offset, source_length FROM embeddings WHERE chunk_id = ?",
                [chunk_id],
            ).fetchall()
            if rows:
                row = rows[0]
                return DocumentChunk(
                    document_name=row[0],
                    content=row[1],
                    section=row[2],
                    chunk_index=row[3],
                    source_offset=row[4],
                    source_length=row[5],
                )
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Scoring and selection (greedy sequential MMR)
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _get_embedding(self, chunk_id: str) -> list[float] | None:
        """Fetch embedding vector for a chunk from DuckDB."""
        try:
            rows = self._store.vector._conn.execute(
                "SELECT embedding FROM embeddings WHERE chunk_id = ?",
                [chunk_id],
            ).fetchall()
            if rows and rows[0][0] is not None:
                return list(rows[0][0])
        except Exception:
            pass
        return None

    def _select_cohort(
        self,
        candidates: list[ScoredChunk],
        query_embedding,
        k: int,
    ) -> list[ScoredChunk]:
        """Greedy sequential MMR selection.

        Implements the algorithm from the spec:
          composite = rw * relevance + pw * priority + cw * coverage
        """
        if not candidates:
            return []

        query_vec = list(query_embedding.flatten())

        # Pre-fetch embeddings and compute relevance for all candidates
        cand_data: list[tuple[ScoredChunk, list[float], float]] = []
        for sc in candidates:
            emb = self._get_embedding(sc.chunk_id)
            if emb is None:
                emb = []
            relevance = self._cosine_similarity(query_vec, emb) if emb else 0.0
            sc.embedding = emb
            cand_data.append((sc, emb, relevance))

        selected: list[tuple[ScoredChunk, list[float]]] = []
        remaining = list(cand_data)

        for _ in range(min(k, len(remaining))):
            best_score = -float("inf")
            best_idx = 0

            for i, (sc, emb, relevance) in enumerate(remaining):
                priority = self._PRIORITY.get(sc.provenance, 0.5)

                if selected:
                    max_sim = max(
                        self._cosine_similarity(emb, sel_emb)
                        for _, sel_emb in selected
                        if emb and sel_emb
                    ) if emb else 0.0
                    coverage = relevance - self._lambda * max_sim
                else:
                    coverage = relevance

                composite = (
                    self._rw * relevance
                    + self._pw * priority
                    + self._cw * coverage
                )
                if composite > best_score:
                    best_score = composite
                    best_idx = i

            chosen_sc, chosen_emb, chosen_rel = remaining.pop(best_idx)
            chosen_sc.score = best_score
            chosen_sc.embedding = None  # don't leak internals in result
            selected.append((chosen_sc, chosen_emb))

        return [sc for sc, _ in selected]