# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 213e7f8a-5158-4869-8346-d12e39dc5fb1
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Scoring, cache, and structural-expansion mixin for EnhancedSearch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..models import DocumentChunk, ScoredChunk

if TYPE_CHECKING:
    from ..storage._store import Store

_log = logging.getLogger(__name__)


class _ScoringMixin:
    """Mixin providing chunk cache, embedding cache, structural neighbors, and MMR selection.

    Depends on ``self._store``, ``self._chunk_cache``, ``self._embedding_cache``,
    ``self._lambda``, ``self._rw``, ``self._pw``, ``self._cw``, ``self._PRIORITY``.
    """

    # Attributes provided by the host EnhancedSearch (declared for the type checker;
    # bare annotations create no runtime attribute).
    _store: Store
    _PRIORITY: dict[str, float]
    _lambda: float
    _rw: float
    _pw: float
    _cw: float

    # ------------------------------------------------------------------
    # Structural expansion
    # ------------------------------------------------------------------

    def _structural_neighbors(self, seed_chunk_ids: list[str]) -> list[tuple[str, DocumentChunk]]:
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
                chunk.document_name, chunk.chunk_index, chunk.embedding_content or chunk.content
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

    def preload_chunk_cache(self) -> None:
        """Preload all chunk metadata and embeddings into memory."""
        try:
            rows = self._store.vector._conn.execute(
                "SELECT chunk_id, document_name, content, section, chunk_index, "
                "source_offset, source_length, embedding FROM embeddings"
            ).fetchall()
        except Exception as exc:
            _log.debug("preload_chunk_cache failed: %s", exc)
            return
        self._chunk_cache = {}
        self._embedding_cache = {}
        for row in rows:
            cid, doc, content, section, idx, off, length, emb = row
            self._chunk_cache[cid] = DocumentChunk(
                document_name=doc,
                content=content,
                section=section,
                chunk_index=idx,
                source_offset=off,
                source_length=length,
            )
            if emb is not None:
                self._embedding_cache[cid] = np.array(emb, dtype="float32")

    def _fetch_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Fetch a single chunk by ID — from cache if available, else DuckDB."""
        if self._chunk_cache is not None:
            return self._chunk_cache.get(chunk_id)
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
        except Exception as exc:
            _log.warning("_fetch_chunk failed for %s: %s", chunk_id, exc)
        return None

    # ------------------------------------------------------------------
    # Scoring and selection (greedy sequential MMR)
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        a = np.asarray(a, dtype="float32")
        b = np.asarray(b, dtype="float32")
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _get_embedding(self, chunk_id: str) -> np.ndarray | list[float] | None:
        """Fetch embedding vector for a chunk — from cache if available, else DuckDB."""
        if self._embedding_cache is not None:
            return self._embedding_cache.get(chunk_id)
        try:
            rows = self._store.vector._conn.execute(
                "SELECT embedding FROM embeddings WHERE chunk_id = ?",
                [chunk_id],
            ).fetchall()
            if rows and rows[0][0] is not None:
                return list(rows[0][0])
        except Exception as exc:
            _log.warning("_get_embedding failed for %s: %s", chunk_id, exc)
        return None

    def _select_cohort(
        self,
        candidates: list[ScoredChunk],
        query_embedding: np.ndarray,
        k: int,
    ) -> list[ScoredChunk]:
        """Greedy sequential MMR selection (vectorized with numpy).

        Implements the algorithm from the spec:
          composite = rw * relevance + pw * priority + cw * coverage
        """
        if not candidates:
            return []

        q = np.array(query_embedding.flatten(), dtype=np.float32)
        qnorm = np.linalg.norm(q)
        q_unit = q / qnorm if qnorm > 0 else q

        # Pre-fetch embeddings, normalize to unit vectors
        scs: list[ScoredChunk] = []
        units: list[np.ndarray | None] = []
        relevances: list[float] = []
        dim = len(q_unit)

        for sc in candidates:
            raw = self._get_embedding(sc.chunk_id)
            # ScoredChunk.embedding is list | None; coerce ndarray to list
            if isinstance(raw, np.ndarray):
                sc.embedding = raw.tolist()
            else:
                sc.embedding = raw
            if raw is not None and len(raw) == dim:
                arr = np.asarray(raw, dtype=np.float32)
                norm = np.linalg.norm(arr)
                unit = arr / norm if norm > 0 else arr
                rel = float(np.dot(q_unit, unit))
            else:
                unit = None
                rel = 0.0
            scs.append(sc)
            units.append(unit)
            relevances.append(rel)

        priorities = np.array(
            [self._PRIORITY.get(sc.provenance, 0.5) for sc in scs], dtype=np.float32
        )

        remaining = list(range(len(scs)))
        sel_units: list[np.ndarray] = []
        result: list[ScoredChunk] = []

        sel_mat: np.ndarray = np.empty(
            (0, dim), dtype=np.float32
        )  # populated when sel_units is non-empty
        for _ in range(min(k, len(remaining))):
            # Build matrix of selected unit vectors for batch max-sim
            if sel_units:
                sel_mat = np.stack(sel_units)  # (n_sel, dim)

            best_score = -float("inf")
            best_j = 0

            for j, i in enumerate(remaining):
                rel = relevances[i]
                pri = float(priorities[i])
                unit = units[i]

                if sel_units:
                    if unit is not None:
                        sims = sel_mat @ unit  # (n_sel,) — all cosines in one BLAS call
                        max_sim = float(np.max(sims))
                    else:
                        max_sim = 0.0
                    coverage = rel - self._lambda * max_sim
                else:
                    coverage = rel

                composite = self._rw * rel + self._pw * pri + self._cw * coverage
                if composite > best_score:
                    best_score = composite
                    best_j = j

            chosen_i = remaining.pop(best_j)
            sc = scs[chosen_i]
            sc.score = best_score
            sc.embedding = None
            chosen_unit = units[chosen_i]
            if chosen_unit is not None:
                sel_units.append(chosen_unit)
            else:
                sel_units.append(np.zeros(dim, dtype=np.float32))
            result.append(sc)

        return result
