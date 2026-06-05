# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b5198bec-c7b1-4d45-8f98-8aa1d0a7d948
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CommunityIndex: chunk-level community detection for context injection.

Workflow:
  1. Build weighted-average vectors: α * heading_emb + (1-α) * content_emb
  2. Construct sparse cosine-similarity graph (edges above threshold)
  3. Run Leiden (default) or Louvain community detection (per resolution level)
  4. Extract topic labels per community (top-K frequent non-stopword terms)

Usage::

    idx = CommunityIndex.build(
        chunk_ids, content_vecs,
        heading_vecs=heading_vecs,   # optional
        alpha=0.2,
        sim_threshold=0.6,
        n_levels=3,
    )
    label = idx.topic_label(chunk_id)   # "cardiovascular disease, thrombosis"
    cid   = idx.community_id(chunk_id)

    # Multi-level access:
    label = idx.topic_label(chunk_id, level=0)   # coarsest
    label = idx.topic_label(chunk_id, level=-1)  # finest (same as None)
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)

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


def _ner_embedding_labels(
    chunk_ids: list[str],
    db_path,
    n: int = 5,
    synonym_threshold: float = 0.85,
) -> str:
    """Generate community label using NER entities and embedding-based synonym merging.

    Fetches entity surface forms for the given chunk_ids from the chunk_entities
    and entity_embeddings tables. Clusters by cosine similarity, picks most-frequent
    canonical form per cluster, returns top-n clusters by size.

    Falls back to empty string if tables are absent or have no data.
    """
    import duckdb
    import numpy as np

    if not chunk_ids:
        return ""

    try:
        con = duckdb.connect(str(db_path), read_only=True)
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = con.execute(
            f"SELECT ce.entity_id, ee.embedding "
            f"FROM chunk_entities ce "
            f"JOIN entity_embeddings ee ON ce.entity_id = ee.entity_id "
            f"WHERE ce.chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        con.close()
    except (duckdb.CatalogException, duckdb.IOException) as e:
        # Tables absent (not yet built) or file not accessible — expected; no labels.
        logger.debug(f"_ner_embedding_labels: expected DB absence: {e}")
        return ""
    except ImportError:
        return ""
    except Exception as e:
        logger.warning(f"_ner_embedding_labels: unexpected error: {e}")
        raise

    if not rows:
        return ""

    entity_ids = [r[0] for r in rows]
    emb_list = [r[1] for r in rows]
    if any(e is None for e in emb_list):
        return ""

    vecs = np.array(emb_list, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)

    assigned: list[int] = [-1] * len(entity_ids)
    clusters: list[list[int]] = []
    for i in range(len(entity_ids)):
        if assigned[i] != -1:
            continue
        cid = len(clusters)
        clusters.append([i])
        assigned[i] = cid
        for j in range(i + 1, len(entity_ids)):
            if assigned[j] != -1:
                continue
            sim = float(vecs[i] @ vecs[j])
            if sim >= synonym_threshold:
                clusters[cid].append(j)
                assigned[j] = cid

    cluster_info: list[tuple[int, str]] = []
    for members in clusters:
        freq: Counter = Counter()
        for idx in members:
            freq[entity_ids[idx]] += 1
        canonical = freq.most_common(1)[0][0]
        cluster_info.append((len(members), canonical))

    cluster_info.sort(key=lambda x: x[0], reverse=True)
    return ", ".join(label for _, label in cluster_info[:n])


class _LevelData:
    __slots__ = ("resolution", "chunk_to_community", "community_to_label",
                 "community_to_chunks", "community_to_coherence")

    def __init__(self, resolution: float) -> None:
        self.resolution = resolution
        self.chunk_to_community: dict[str, int] = {}
        self.community_to_label: dict[int, str] = {}
        self.community_to_chunks: dict[int, list[str]] = defaultdict(list)
        self.community_to_coherence: dict[int, float] = {}


# ------------------------------------------------------------------
# Build helpers (module-level to keep CommunityIndex.build thin)
# ------------------------------------------------------------------

def _resolve_levels(
    resolutions: list[float] | None,
    n_levels: int,
    resolution_min: float,
    resolution_max: float,
) -> list[float]:
    """Return the ordered list of resolution values for community detection."""
    if resolutions is not None:
        return list(resolutions)
    if n_levels == 1:
        return [(resolution_min + resolution_max) / 2]
    return list(np.logspace(
        np.log10(resolution_min),
        np.log10(resolution_max),
        num=n_levels,
    ).tolist())


def _compute_working_vecs(
    content_vecs: np.ndarray,
    heading_vecs: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """Blend heading and content embeddings; return L2-normalised working vectors."""
    if heading_vecs is not None and alpha > 0:
        vecs = alpha * heading_vecs + (1.0 - alpha) * content_vecs
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)
    return content_vecs


def _build_similarity_edges(
    vecs: np.ndarray,
    sim_threshold: float,
    extra_edges: list[tuple[int, int, float]] | None,
) -> list[tuple[int, int, float]]:
    """Return sparse cosine-similarity edge list above sim_threshold."""
    n = len(vecs)
    edges: list[tuple[int, int, float]] = []
    batch = 256
    for i in range(0, n, batch):
        sims = vecs[i:i + batch] @ vecs.T  # (batch, n)
        for bi in range(sims.shape[0]):
            gi = i + bi
            js = np.where(sims[bi, gi + 1:] >= sim_threshold)[0] + gi + 1
            for j in js:
                edges.append((gi, int(j), float(sims[bi, int(j) - i])))
    if extra_edges:
        edges.extend(extra_edges)
    return edges


def _run_leiden(
    n: int,
    edges: list[tuple[int, int, float]],
    resolution: float,
    n_iterations: int,
    seed: int | None,
) -> dict[int, int] | None:
    """Attempt Leiden partition; return None on ImportError."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return None

    g = ig.Graph(n=n, edges=[(e[0], e[1]) for e in edges])
    g.es["weight"] = [e[2] for e in edges]
    result = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if edges else None,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )
    partition: dict[int, int] = {}
    for cid, members in enumerate(result):
        for node in members:
            partition[node] = cid
    return partition


def _run_louvain(
    n: int,
    edges: list[tuple[int, int, float]],
    seed: int | None,
) -> dict[int, int]:
    """Run Louvain (or connected-components fallback) and return partition."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for gi, j, w in edges:
        G.add_edge(gi, j, weight=w)

    try:
        import community as louvain_mod
        return louvain_mod.best_partition(  # type: ignore[attr-defined]  # python-louvain stub gap
            G, random_state=seed if seed is not None else 42
        )
    except ImportError:
        partition: dict[int, int] = {}
        for cid, component in enumerate(nx.connected_components(G)):
            for node in component:
                partition[node] = cid
        return partition


def _run_partition(
    n: int,
    edges: list[tuple[int, int, float]],
    algorithm: str,
    resolution: float,
    n_iterations: int,
    seed: int | None,
) -> dict[int, int]:
    """Dispatch to leiden or louvain; fall back to louvain if leiden unavailable."""
    partition: dict[int, int] | None = None

    if algorithm == "leiden":
        partition = _run_leiden(n, edges, resolution, n_iterations, seed)

    if partition is None:
        partition = _run_louvain(n, edges, seed)

    if not partition:
        partition = {i: i for i in range(n)}

    return partition


def _assign_labels(
    level_data: _LevelData,
    community_members: dict[int, list[int]],
    chunk_ids: list[str],
    chunk_texts: list[str] | None,
    label_strategy: str,
    db_path,
    top_label_terms: int,
    label_synonym_threshold: float,
) -> None:
    """Populate level_data.community_to_label in-place."""
    if label_strategy == "ner_embedding" and db_path is not None:
        for cid, indices in community_members.items():
            cids = [chunk_ids[i] for i in indices if i < len(chunk_ids)]
            label = _ner_embedding_labels(
                cids, db_path,
                n=top_label_terms,
                synonym_threshold=label_synonym_threshold,
            )
            if not label and chunk_texts:
                texts = [chunk_texts[i] for i in indices if i < len(chunk_texts)]
                label = _top_terms(texts, top_label_terms)
            level_data.community_to_label[cid] = label
    elif chunk_texts:
        for cid, indices in community_members.items():
            texts = [chunk_texts[i] for i in indices if i < len(chunk_texts)]
            level_data.community_to_label[cid] = _top_terms(texts, top_label_terms)


def _compute_coherence(
    level_data: _LevelData,
    community_members: dict[int, list[int]],
    vecs: np.ndarray,
) -> None:
    """Populate level_data.community_to_coherence in-place."""
    for cid, indices in community_members.items():
        if len(indices) < 2:
            level_data.community_to_coherence[cid] = 0.0
            continue
        member_vecs = vecs[indices]
        sims = member_vecs @ member_vecs.T
        m = len(indices)
        upper = sims[np.triu_indices(m, k=1)]
        level_data.community_to_coherence[cid] = float(upper.mean()) if len(upper) > 0 else 0.0


def _build_level_data(
    resolution: float,
    partition: dict[int, int],
    chunk_ids: list[str],
    chunk_texts: list[str] | None,
    vecs: np.ndarray,
    label_strategy: str,
    db_path,
    top_label_terms: int,
    label_synonym_threshold: float,
) -> _LevelData:
    """Construct a _LevelData from a partition dict for one resolution level."""
    level_data = _LevelData(resolution)
    community_members: dict[int, list[int]] = defaultdict(list)

    for idx, cid in partition.items():
        level_data.chunk_to_community[chunk_ids[idx]] = cid
        level_data.community_to_chunks[cid].append(chunk_ids[idx])
        community_members[cid].append(idx)

    _assign_labels(
        level_data, community_members, chunk_ids, chunk_texts,
        label_strategy, db_path, top_label_terms, label_synonym_threshold,
    )
    _compute_coherence(level_data, community_members, vecs)
    return level_data


def _populate_flat_attrs(instance: CommunityIndex, finest: _LevelData) -> None:
    """Copy the finest level's data into the instance's flat attributes."""
    instance._chunk_to_community = dict(finest.chunk_to_community)
    instance._community_to_label = dict(finest.community_to_label)
    instance._community_to_chunks = defaultdict(list, {
        k: list(v) for k, v in finest.community_to_chunks.items()
    })
    instance._community_to_coherence = dict(finest.community_to_coherence)


# ------------------------------------------------------------------
# from_db helpers
# ------------------------------------------------------------------

def _load_multilevel_schema(
    con,
    instance: CommunityIndex,
) -> None:
    """Populate instance._levels from the multi-level chunk_communities schema."""
    chunk_rows = con.execute(
        "SELECT chunk_id, level, community_id FROM chunk_communities ORDER BY level"
    ).fetchall()
    community_rows = con.execute(
        "SELECT community_id, level, topic_label, coherence, resolution FROM communities ORDER BY level"
    ).fetchall()
    con.close()

    levels_by_idx: dict[int, _LevelData] = {}
    for chunk_id, level_idx, cid in chunk_rows:
        if level_idx not in levels_by_idx:
            levels_by_idx[level_idx] = _LevelData(1.0)
        ld = levels_by_idx[level_idx]
        ld.chunk_to_community[chunk_id] = cid
        ld.community_to_chunks[cid].append(chunk_id)

    for cid, level_idx, label, coherence, resolution in community_rows:
        if level_idx not in levels_by_idx:
            levels_by_idx[level_idx] = _LevelData(resolution or 1.0)
        ld = levels_by_idx[level_idx]
        ld.resolution = resolution or 1.0
        ld.community_to_label[cid] = label or ""
        ld.community_to_coherence[cid] = coherence or 0.0

    for level_idx in sorted(levels_by_idx.keys()):
        instance._levels.append(levels_by_idx[level_idx])


def _fetch_legacy_community_rows(con) -> list[tuple]:
    """Fetch community rows from legacy schema, with or without coherence column."""
    try:
        return con.execute(
            "SELECT community_id, topic_label, coherence FROM communities"
        ).fetchall()
    except Exception:
        return [
            (cid, lbl, 0.0)
            for cid, lbl in con.execute(
                "SELECT community_id, topic_label FROM communities"
            ).fetchall()
        ]


def _load_legacy_schema(
    con,
    instance: CommunityIndex,
) -> None:
    """Populate instance._levels from the legacy (no level column) schema."""
    chunk_rows = con.execute(
        "SELECT chunk_id, community_id FROM chunk_communities"
    ).fetchall()
    community_rows_old = _fetch_legacy_community_rows(con)
    con.close()

    ld = _LevelData(1.0)
    for chunk_id, cid in chunk_rows:
        ld.chunk_to_community[chunk_id] = cid
        ld.community_to_chunks[cid].append(chunk_id)
    for cid, label, coherence in community_rows_old:
        ld.community_to_label[cid] = label or ""
        ld.community_to_coherence[cid] = coherence or 0.0
    instance._levels.append(ld)


class CommunityIndex:
    """Chunk-to-community assignment with topic labels."""

    def __init__(self) -> None:
        self._chunk_to_community: dict[str, int] = {}
        self._community_to_label: dict[int, str] = {}
        self._community_to_chunks: dict[int, list[str]] = defaultdict(list)
        self._community_to_coherence: dict[int, float] = {}
        self._levels: list[_LevelData] = []

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
        label_strategy: str = "term_freq",
        db_path=None,
        label_synonym_threshold: float = 0.85,
        algorithm: str = "leiden",
        resolutions: list[float] | None = None,
        n_levels: int = 3,
        resolution_min: float = 0.25,
        resolution_max: float = 2.0,
        n_iterations: int = 10,
        seed: int | None = 42,
        extra_edges: list[tuple[int, int, float]] | None = None,
    ) -> CommunityIndex:
        """Build a CommunityIndex from chunk embeddings.

        Args:
            chunk_ids: Parallel list of chunk IDs.
            content_vecs: (n, dim) float32 content embeddings (L2-normalised).
            chunk_texts: Optional parallel list of chunk text for label extraction.
            heading_vecs: Optional (n, dim) heading/breadcrumb embeddings.
            alpha: Weight for heading in weighted average (0 = content only).
            sim_threshold: Minimum cosine similarity for graph edges.
            top_label_terms: Number of terms per community label.
            label_strategy: "term_freq" (default) or "ner_embedding".
            db_path: Path to DuckDB (required for "ner_embedding" strategy).
            label_synonym_threshold: Cosine similarity for merging synonyms (ner_embedding only).
            algorithm: "leiden" (default) or "louvain". Leiden requires chonk[leiden].
            resolutions: Explicit list of resolution values (takes priority over n_levels).
            n_levels: Number of hierarchy levels when resolutions is None.
            resolution_min: Coarsest resolution (level 0).
            resolution_max: Finest resolution (level n_levels-1).
            n_iterations: Leiden iterations per level.
            seed: Random seed for reproducibility.
        """
        n = len(chunk_ids)
        if n == 0:
            return cls()

        resolved = _resolve_levels(resolutions, n_levels, resolution_min, resolution_max)
        vecs = _compute_working_vecs(content_vecs, heading_vecs, alpha)
        edges = _build_similarity_edges(vecs, sim_threshold, extra_edges)

        instance = cls()
        for resolution in resolved:
            partition = _run_partition(n, edges, algorithm, resolution, n_iterations, seed)
            level_data = _build_level_data(
                resolution, partition, chunk_ids, chunk_texts, vecs,
                label_strategy, db_path, top_label_terms, label_synonym_threshold,
            )
            instance._levels.append(level_data)

        if instance._levels:
            _populate_flat_attrs(instance, instance._levels[-1])

        return instance

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def community_id(self, chunk_id: str, level: int | None = None) -> int | None:
        if level is None:
            return self._chunk_to_community.get(chunk_id)
        return self._levels[level].chunk_to_community.get(chunk_id)

    def topic_label(self, chunk_id: str, min_coherence: float = 0.0, level: int | None = None) -> str | None:
        if level is None:
            cid = self._chunk_to_community.get(chunk_id)
            if cid is None:
                return None
            if min_coherence > 0 and self._community_to_coherence.get(cid, 0.0) < min_coherence:
                return None
            return self._community_to_label.get(cid)
        ld = self._levels[level]
        cid = ld.chunk_to_community.get(chunk_id)
        if cid is None:
            return None
        if min_coherence > 0 and ld.community_to_coherence.get(cid, 0.0) < min_coherence:
            return None
        return ld.community_to_label.get(cid)

    def coherence(self, chunk_id: str, level: int | None = None) -> float:
        if level is None:
            cid = self._chunk_to_community.get(chunk_id)
            if cid is None:
                return 0.0
            return self._community_to_coherence.get(cid, 0.0)
        ld = self._levels[level]
        cid = ld.chunk_to_community.get(chunk_id)
        if cid is None:
            return 0.0
        return ld.community_to_coherence.get(cid, 0.0)

    def community_chunks(self, community_id: int, level: int | None = None) -> list[str]:
        if level is None:
            return list(self._community_to_chunks.get(community_id, []))
        return list(self._levels[level].community_to_chunks.get(community_id, []))

    def community_ids(self, level: int | None = None) -> list[int]:
        """Return all community IDs present in this index."""
        if level is None:
            return list(self._community_to_chunks.keys())
        return list(self._levels[level].community_to_chunks.keys())

    def topic_label_for_community(self, community_id: int, level: int | None = None) -> str:
        """Return the topic label for *community_id*, or empty string if unknown."""
        if level is None:
            return self._community_to_label.get(community_id, "")
        return self._levels[level].community_to_label.get(community_id, "")

    def level_count(self) -> int:
        """Return number of levels (minimum 1)."""
        return max(len(self._levels), 1)

    def resolutions_list(self) -> list[float]:
        """Return list of resolution values per level."""
        return [l.resolution for l in self._levels]

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

        con.execute("DROP TABLE IF EXISTS chunk_communities")
        con.execute("DROP TABLE IF EXISTS communities")
        con.execute(
            "CREATE TABLE chunk_communities "
            "(chunk_id TEXT, level INTEGER, community_id INTEGER, "
            "PRIMARY KEY (chunk_id, level))"
        )
        con.execute(
            "CREATE TABLE communities "
            "(community_id INTEGER, level INTEGER, topic_label TEXT, "
            "size INTEGER, coherence REAL, resolution REAL, "
            "PRIMARY KEY (community_id, level))"
        )

        if self._levels:
            for level_idx, ld in enumerate(self._levels):
                for chunk_id, cid in ld.chunk_to_community.items():
                    con.execute(
                        "INSERT INTO chunk_communities VALUES (?, ?, ?)",
                        [chunk_id, level_idx, cid],
                    )
                for cid, label in ld.community_to_label.items():
                    size = len(ld.community_to_chunks.get(cid, []))
                    coherence = ld.community_to_coherence.get(cid, 0.0)
                    con.execute(
                        "INSERT INTO communities VALUES (?, ?, ?, ?, ?, ?)",
                        [cid, level_idx, label, size, coherence, ld.resolution],
                    )
        else:
            # Persist flat attrs as level 0
            for chunk_id, cid in self._chunk_to_community.items():
                con.execute(
                    "INSERT INTO chunk_communities VALUES (?, ?, ?)",
                    [chunk_id, 0, cid],
                )
            for cid, label in self._community_to_label.items():
                size = len(self._community_to_chunks.get(cid, []))
                coherence = self._community_to_coherence.get(cid, 0.0)
                con.execute(
                    "INSERT INTO communities VALUES (?, ?, ?, ?, ?, ?)",
                    [cid, 0, label, size, coherence, 1.0],
                )

        con.close()

    @classmethod
    def from_db(cls, db_path) -> CommunityIndex:
        """Load CommunityIndex from DuckDB tables."""
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)

        instance = cls()

        try:
            _load_multilevel_schema(con, instance)
        except duckdb.CatalogException:
            # Multi-level schema absent — try legacy schema.
            try:
                _load_legacy_schema(con, instance)
            except duckdb.CatalogException:
                # Neither schema built yet; return empty index (valid not-yet-built state).
                try:
                    con.close()
                except Exception:
                    pass
                return instance
            except Exception as e:
                # Unexpected failure loading legacy schema.
                try:
                    con.close()
                except Exception:
                    pass
                logger.warning(f"from_db: unexpected error loading legacy schema: {e}")
                raise
        except Exception as e:
            # Unexpected failure loading multi-level schema.
            try:
                con.close()
            except Exception:
                pass
            logger.warning(f"from_db: unexpected error loading multi-level schema: {e}")
            raise

        # Populate flat attrs from finest level
        if instance._levels:
            _populate_flat_attrs(instance, instance._levels[-1])

        return instance
