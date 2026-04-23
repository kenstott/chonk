# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CommunityIndex: chunk-level community detection for context injection.

Workflow:
  1. Build weighted-average vectors: α * heading_emb + (1-α) * content_emb
  2. Construct sparse cosine-similarity graph (edges above threshold)
  3. Run Louvain community detection
  4. Extract topic labels per community (top-K frequent non-stopword terms)

Usage::

    idx = CommunityIndex.build(
        chunk_ids, content_vecs,
        heading_vecs=heading_vecs,   # optional
        alpha=0.2,
        sim_threshold=0.6,
    )
    label = idx.topic_label(chunk_id)   # "cardiovascular disease, thrombosis"
    cid   = idx.community_id(chunk_id)
"""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict

import numpy as np


_STOPWORDS = frozenset(
    "a an the and or but if in on at to for of with by from as is was are were "
    "be been being have has had do does did will would could should may might "
    "shall can its it this that these those i we you he she they all any some "
    "no not so such than then there when where which who whom whose how what "
    "each other about above after before between into through during including "
    "also however therefore thus hence moreover furthermore although though "
    "context information provided based given using according "
    "your my our his her their his our them him her us me my yours mine ours "
    "said like very went made first more time just know need make good look come "
    "back down over well said told take got used even only still many much just "
    "upon away never always already something anything everything nothing both "
    "same way day year man men woman women people person life work new old "
    "said come came going goes went want wanted little great long rather quite "
    "table footnote cont figure chapter section page".split()
)


def _top_terms(texts: list[str], n: int = 5) -> str:
    counts: Counter = Counter()
    for text in texts:
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        for w in words:
            if len(w) > 3 and w not in _STOPWORDS:
                counts[w] += 1
    return ", ".join(w for w, _ in counts.most_common(n))


class CommunityIndex:
    """Chunk-to-community assignment with topic labels."""

    def __init__(self) -> None:
        self._chunk_to_community: dict[str, int] = {}
        self._community_to_label: dict[int, str] = {}
        self._community_to_chunks: dict[int, list[str]] = defaultdict(list)
        self._community_to_coherence: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        chunk_ids: list[str],
        content_vecs: np.ndarray,
        chunk_texts: list[str] | None = None,
        heading_vecs: np.ndarray | None = None,
        alpha: float = 0.2,
        sim_threshold: float = 0.6,
        top_label_terms: int = 5,
    ) -> "CommunityIndex":
        """Build a CommunityIndex from chunk embeddings.

        Args:
            chunk_ids: Parallel list of chunk IDs.
            content_vecs: (n, dim) float32 content embeddings (L2-normalised).
            chunk_texts: Optional parallel list of chunk text for label extraction.
            heading_vecs: Optional (n, dim) heading/breadcrumb embeddings.
            alpha: Weight for heading in weighted average (0 = content only).
            sim_threshold: Minimum cosine similarity for graph edges.
            top_label_terms: Number of terms per community label.
        """
        import networkx as nx

        n = len(chunk_ids)
        if n == 0:
            return cls()

        # ── 1. Compute working vectors ────────────────────────────────
        if heading_vecs is not None and alpha > 0:
            vecs = alpha * heading_vecs + (1.0 - alpha) * content_vecs
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
        else:
            vecs = content_vecs

        # ── 2. Build sparse similarity graph ─────────────────────────
        G = nx.Graph()
        G.add_nodes_from(range(n))

        batch = 256
        for i in range(0, n, batch):
            sims = vecs[i:i + batch] @ vecs.T  # (batch, n)
            for bi in range(sims.shape[0]):
                gi = i + bi
                js = np.where(sims[bi, gi + 1:] >= sim_threshold)[0] + gi + 1
                for j in js:
                    G.add_edge(gi, int(j), weight=float(sims[bi, int(j) - i]))

        # ── 3. Louvain community detection ────────────────────────────
        try:
            import community as louvain_mod
            partition: dict[int, int] = louvain_mod.best_partition(G, random_state=42)
        except ImportError:
            # Fallback: connected components as communities
            partition = {}
            for cid, component in enumerate(nx.connected_components(G)):
                for node in component:
                    partition[node] = cid

        # ── 4. Assign and extract labels ──────────────────────────────
        instance = cls()
        community_members: dict[int, list[int]] = defaultdict(list)
        for idx, cid in partition.items():
            instance._chunk_to_community[chunk_ids[idx]] = cid
            instance._community_to_chunks[cid].append(chunk_ids[idx])
            community_members[cid].append(idx)

        if chunk_texts:
            for cid, indices in community_members.items():
                texts = [chunk_texts[i] for i in indices if i < len(chunk_texts)]
                instance._community_to_label[cid] = _top_terms(texts, top_label_terms)

        # ── 5. Compute per-community coherence (mean intra-community cosine sim) ──
        for cid, indices in community_members.items():
            if len(indices) < 2:
                instance._community_to_coherence[cid] = 0.0
                continue
            member_vecs = vecs[indices]  # (m, dim)
            sims = member_vecs @ member_vecs.T  # (m, m)
            m = len(indices)
            # Mean of upper triangle (excluding diagonal)
            upper = sims[np.triu_indices(m, k=1)]
            instance._community_to_coherence[cid] = float(upper.mean()) if len(upper) > 0 else 0.0

        return instance

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def community_id(self, chunk_id: str) -> int | None:
        return self._chunk_to_community.get(chunk_id)

    def topic_label(self, chunk_id: str, min_coherence: float = 0.0) -> str | None:
        cid = self._chunk_to_community.get(chunk_id)
        if cid is None:
            return None
        if min_coherence > 0 and self._community_to_coherence.get(cid, 0.0) < min_coherence:
            return None
        return self._community_to_label.get(cid)

    def coherence(self, chunk_id: str) -> float:
        cid = self._chunk_to_community.get(chunk_id)
        if cid is None:
            return 0.0
        return self._community_to_coherence.get(cid, 0.0)

    def community_chunks(self, community_id: int) -> list[str]:
        return list(self._community_to_chunks.get(community_id, []))

    def community_count(self) -> int:
        return len(self._community_to_label)

    def chunk_count(self) -> int:
        return len(self._chunk_to_community)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, db_path) -> None:
        """Write chunk_communities and communities tables to DuckDB."""
        import duckdb
        con = duckdb.connect(str(db_path))
        con.execute(
            "CREATE TABLE IF NOT EXISTS chunk_communities "
            "(chunk_id TEXT PRIMARY KEY, community_id INTEGER)"
        )
        con.execute("DROP TABLE IF EXISTS communities")
        con.execute(
            "CREATE TABLE communities "
            "(community_id INTEGER PRIMARY KEY, topic_label TEXT, size INTEGER, coherence REAL)"
        )
        con.execute("DELETE FROM chunk_communities")
        con.execute("DELETE FROM communities")

        for chunk_id, cid in self._chunk_to_community.items():
            con.execute(
                "INSERT INTO chunk_communities VALUES (?, ?)",
                [chunk_id, cid],
            )
        for cid, label in self._community_to_label.items():
            size = len(self._community_to_chunks.get(cid, []))
            coherence = self._community_to_coherence.get(cid, 0.0)
            con.execute(
                "INSERT INTO communities VALUES (?, ?, ?, ?)",
                [cid, label, size, coherence],
            )
        con.close()

    @classmethod
    def from_db(cls, db_path) -> "CommunityIndex":
        """Load CommunityIndex from DuckDB tables."""
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                "SELECT chunk_id, community_id FROM chunk_communities"
            ).fetchall()
            community_rows = con.execute(
                "SELECT community_id, topic_label, coherence FROM communities"
            ).fetchall()
        except Exception:
            try:
                rows = con.execute(
                    "SELECT chunk_id, community_id FROM chunk_communities"
                ).fetchall()
                community_rows = [
                    (cid, lbl, 0.0)
                    for cid, lbl in con.execute(
                        "SELECT community_id, topic_label FROM communities"
                    ).fetchall()
                ]
            except Exception:
                con.close()
                return cls()
        con.close()

        instance = cls()
        for chunk_id, cid in rows:
            instance._chunk_to_community[chunk_id] = cid
            instance._community_to_chunks[cid].append(chunk_id)
        for cid, label, coherence in community_rows:
            instance._community_to_label[cid] = label or ""
            instance._community_to_coherence[cid] = coherence or 0.0
        return instance
