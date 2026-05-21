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

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from ..models import DocumentChunk, ScoredChunk

if TYPE_CHECKING:
    from ..cluster._map import ClusterMap
    from ..community._index import CommunityIndex
    from ..generation._answer import Answer
    from ..graph._index import RelationshipIndex
    from ..ner._index import EntityIndex
    from ..storage._store import Store

_log = logging.getLogger(__name__)


def _load_entity_index_from_store(conn) -> EntityIndex | None:
    try:
        count = conn.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()[0]
        if count == 0:
            return None
        from ..ner._index import EntityIndex as _EI
        return _EI.load_from_db(conn)
    except Exception:
        return None


def _load_relationship_index_from_store(conn) -> RelationshipIndex | None:
    try:
        count = conn.execute("SELECT COUNT(*) FROM svo_triples").fetchone()[0]
        if count == 0:
            return None
        from ..graph._index import RelationshipIndex as _RI
        return _RI.load_from_db(conn)
    except Exception:
        return None


def _load_community_index_from_store(store) -> CommunityIndex | None:
    """Auto-load CommunityIndex from the store's DuckDB if community tables exist."""
    try:
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
    except Exception:
        return None


def _tally_provenance(results: list[ScoredChunk]) -> dict[str, int]:
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
        redaction_filter: Optional ``(Answer) -> Answer`` applied after generation in
            ``ask()``. Use to sanitize generated answer text before it leaves the
            perimeter. Not applied when ``search()`` is called directly.
        chunk_filter: Optional ``(list[ScoredChunk]) -> list[ScoredChunk]`` applied at
            the end of every ``search()`` call. Use to redact or drop sensitive chunks
            before they are returned to the caller. **Not applied** when ``search()``
            is called internally by ``ask()`` — use ``redaction_filter`` for that path.

            Example — drop chunks from a restricted source::

                def drop_restricted(chunks):
                    return [c for c in chunks if "restricted" not in c.chunk.source]

                search = EnhancedSearch(store, chunk_filter=drop_restricted)
    """

    # Source priority constants (from spec)
    _PRIORITY = {
        "seed": 1.0,
        "structural": 0.9,
        "entity_adjacent": 0.7,
        "cluster_adjacent": 0.5,
        "context_graph_expansion": 0.6,
    }

    def __init__(
        self,
        store: Store,
        entity_index: EntityIndex | None = None,
        cluster_map: ClusterMap | None = None,
        relationship_index: RelationshipIndex | None = None,
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
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        entity_embedding_top_k: int = 10,
        entity_ref_expansion: bool = False,
        entity_ref_expansion_k: int = 20,
        entity_ref_expansion_per_k: int | None = None,
        entity_ref_expansion_min_sim: float | None = None,
        query_ner_fn: Callable[[str], list[str]] | None = None,
        query_entity_id_fn: Callable[[str], list[str]] | None = None,
        lane_entity_min_sim: float | None = None,
        session_fingerprint: str | None = None,
        community_index: CommunityIndex | None = None,
        context_graph_expansion: bool = False,
        context_graph_min_weight: float = 0.1,
        context_graph_top_k: int = 5,
        redaction_filter: Callable[[Answer], Answer] | None = None,
        chunk_filter: Callable[[list[ScoredChunk]], list[ScoredChunk]] | None = None,
    ):
        self._store = store
        conn = store.vector._conn
        if entity_index is None:
            entity_index = _load_entity_index_from_store(conn)
        self._entity_index = entity_index
        self._cluster_map = cluster_map
        if relationship_index is None:
            relationship_index = _load_relationship_index_from_store(conn)
        self._relationship_index = relationship_index
        if community_index is None:
            community_index = _load_community_index_from_store(store)
        self._community_index = community_index
        self._chunk_cache: dict[str, DocumentChunk] | None = None
        self._embedding_cache: dict[str, list[float]] | None = None
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
        self._query_entity_id_fn = query_entity_id_fn
        self._lane_entity_min_sim = lane_entity_min_sim
        self._session_fingerprint = session_fingerprint
        self._context_graph_expansion = context_graph_expansion
        self._context_graph_min_weight = context_graph_min_weight
        self._context_graph_top_k = context_graph_top_k
        self._redaction_filter = redaction_filter
        self._chunk_filter = chunk_filter
        self.last_expansion_stats: dict | None = None

    # ------------------------------------------------------------------
    # Namespace / domain pre-filter
    # ------------------------------------------------------------------

    _NAMESPACE_FILTER_PROMPT = (
        "You are a retrieval routing assistant. Below are the available knowledge namespaces "
        "and their descriptions. Select the namespaces most likely to contain evidence for "
        "the query. Return ONLY a JSON array of namespace IDs, e.g. [\"cyber\", \"financial\"]. "
        "Include all namespaces that may be relevant; omit those that are clearly unrelated.\n\n"
        "Namespaces:\n{namespace_list}\n\n"
        "Query: {query}\n\n"
        "Return ONLY a JSON array of namespace IDs."
    )

    _DOMAIN_FILTER_PROMPT = (
        "You are a retrieval routing assistant. Below are the available knowledge domains "
        "and their descriptions. Select the domains most likely to contain evidence for "
        "the query. Return ONLY a JSON array of domain names exactly as listed, "
        "e.g. [\"sales/north-america\", \"finance/q1\"]. "
        "Include all domains that may be relevant; omit those that are clearly unrelated.\n\n"
        "Domains:\n{domain_list}\n\n"
        "Query: {query}\n\n"
        "Return ONLY a JSON array of domain names exactly as listed."
    )

    def _select_namespaces(
        self,
        query: str,
        llm_fn: Callable[[str], str],
    ) -> list[str] | None:
        """Call llm_fn to select relevant namespaces for query.

        Fetches (namespace_id, description) rows from the store. If fewer than
        two namespaces have descriptions, returns None (no filtering). Otherwise
        calls llm_fn with a routing prompt and parses the JSON array response.
        Falls back to None on any parse error so search degrades gracefully.
        """
        import json as _json

        rows = self._store.vector._conn.execute(
            "SELECT namespace_id, description FROM all_namespaces "
            "WHERE description IS NOT NULL AND description != '' "
            "ORDER BY namespace_id"
        ).fetchall()

        if len(rows) < 2:
            return None

        ns_lines = "\n".join(f"- {ns_id}: {desc}" for ns_id, desc in rows)
        prompt = self._NAMESPACE_FILTER_PROMPT.format(
            namespace_list=ns_lines,
            query=query,
        )
        try:
            raw = llm_fn(prompt)
            # Extract JSON array — strip markdown fences if present
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                return None
            selected: list[str] = _json.loads(match.group())
            known = {ns_id for ns_id, _ in rows}
            return [ns for ns in selected if ns in known] or None
        except Exception:
            return None

    def _select_domains(
        self,
        query: str,
        llm_fn: Callable[[str], str],
        namespaces: list[str] | None = None,
    ) -> list[str] | None:
        """Call llm_fn to select relevant domain_ids for query.

        Fetches domains with descriptions from the store (restricted to
        *namespaces* when supplied). Returns None when fewer than two domains
        have descriptions, or on any parse/LLM error.
        """
        import json as _json

        where = "WHERE d.description IS NOT NULL AND d.description != ''"
        params: list = []
        if namespaces:
            placeholders = ", ".join("?" * len(namespaces))
            where += f" AND d.namespace_id IN ({placeholders})"
            params.extend(namespaces)

        rows = self._store.vector._conn.execute(
            f"SELECT d.domain_id, d.namespace_id, d.name, d.description "
            f"FROM domains d {where} ORDER BY d.namespace_id, d.name",
            params,
        ).fetchall()

        if len(rows) < 2:
            return None

        # name_to_id for resolving LLM output back to domain_ids
        name_to_id = {name: domain_id for domain_id, _ns, name, _desc in rows}
        domain_lines = "\n".join(
            f"- {name} ({ns}): {desc}" for _did, ns, name, desc in rows
        )
        prompt = self._DOMAIN_FILTER_PROMPT.format(
            domain_list=domain_lines,
            query=query,
        )
        try:
            raw = llm_fn(prompt)
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                return None
            selected_names: list[str] = _json.loads(match.group())
            ids = [name_to_id[n] for n in selected_names if n in name_to_id]
            return ids or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @overload
    def search(
        self,
        query_embedding: np.ndarray | None = ...,
        k: int = ...,
        query_text: str | None = ...,
        query_entities: list[str] | None = ...,
        precomputed_entity_vecs: dict[str, np.ndarray] | None = ...,
        mode: str = ...,
        namespaces: list[str] | None = ...,
        domain_ids: list[str] | None = ...,
        namespace_filter_llm_fn: Callable[[str], str] | None = ...,
        domain_filter_llm_fn: Callable[[str], str] | None = ...,
        return_trace: Literal[False] = ...,
    ) -> list[ScoredChunk]: ...

    @overload
    def search(
        self,
        query_embedding: np.ndarray | None = ...,
        k: int = ...,
        query_text: str | None = ...,
        query_entities: list[str] | None = ...,
        precomputed_entity_vecs: dict[str, np.ndarray] | None = ...,
        mode: str = ...,
        namespaces: list[str] | None = ...,
        domain_ids: list[str] | None = ...,
        namespace_filter_llm_fn: Callable[[str], str] | None = ...,
        domain_filter_llm_fn: Callable[[str], str] | None = ...,
        return_trace: Literal[True] = ...,
    ) -> tuple[list[ScoredChunk], RetrievalTrace]: ...

    def search(
        self,
        query_embedding: np.ndarray | None = None,
        k: int = 30,
        query_text: str | None = None,
        query_entities: list[str] | None = None,
        precomputed_entity_vecs: dict[str, np.ndarray] | None = None,
        mode: str = "vector_first",
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        namespace_filter_llm_fn: Callable[[str], str] | None = None,
        domain_filter_llm_fn: Callable[[str], str] | None = None,
        return_trace: bool = False,
        _bypass_chunk_filter: bool = False,
    ) -> list[ScoredChunk] | tuple[list[ScoredChunk], RetrievalTrace]:
        """Assemble a top-k cohort using all enabled expansion dimensions.

        Args:
            query_embedding: np.ndarray shape (dim,) or (1, dim). If None,
                auto-generated from *query_text* using the ``embed_fn`` passed
                to the constructor (raises ValueError if neither is available).
            k: Target cohort size.
            query_text: Optional query text for BM25 hybrid seed search.
            mode: Retrieval mode — "vector_first" (default), "graph_first", or "global".
                  "graph_first": drives on RelationshipIndex traversal, vector reranks.
                  "global": searches community_summary chunks only.
            namespace_filter_llm_fn: Optional ``(prompt: str) -> str`` callable. When
                supplied and *namespaces* is None, fetches namespace descriptions from
                the store, calls the LLM to select relevant namespaces for *query_text*,
                and restricts the search to those namespaces.
            return_trace: If True, return ``(results, RetrievalTrace)`` instead of just results.

        Returns:
            Ranked list of up to k ScoredChunk objects.

        ## Agent guidance — framing queries for best retrieval

        **Issue one atomic sub-query per call.** Compound questions ("what is X and
        how does it relate to Y?") split the embedding across two intents and dilute
        both. Decompose before calling; recombine in reasoning.

        **Name the entity explicitly.** "What was Amazon's Total Net Sales for FY2025?"
        retrieves far better than "What were the e-commerce company's sales?" Named
        entities anchor the BM25 lane and entity-graph expansion. If the entity name
        is unknown, use search() to resolve it first, then re-query with the resolved
        name.

        **State the answer type in the query.** "What *date* did...", "What *dollar
        amount*...", "Which *CVE ID*..." biases the embedding toward passages that
        contain that answer type, not just passages that discuss the topic.

        **Prefer declarative over interrogative form for lookup queries.** Embedding
        "Amazon Total Net Sales FY2025 revenue" often outperforms "What was Amazon's
        revenue?" for precise fact retrieval because it matches the register of the
        source document.

        **Scope the namespace when you know the source type.** Pass
        ``namespaces=["financial"]`` for a 10-K question, ``namespaces=["cyber"]``
        for a CVE question. This eliminates off-domain noise and is the single
        highest-leverage accuracy lever for cross-domain corpora. Use
        ``namespace_filter_llm_fn`` to automate scoping when the source type is
        ambiguous.

        **Use ``return_trace=True`` to diagnose poor results.** If top chunks are
        off-topic, inspect ``RetrievalTrace.final_provenance`` to see which expansion
        dimension produced them. A high "entity_adjacent" share with low scores
        indicates the entity lane is over-expanding; tighten ``lane_entity_min_sim``
        or narrow the namespace.

        **Widen k before concluding evidence is absent.** A fact may rank outside
        the default k=5 window. Retry with k=20 before deciding the corpus does not
        contain the answer.
        """
        if query_embedding is None:
            if query_text is None:
                raise ValueError("Either query_embedding or query_text must be supplied.")
            if self._embed_fn is None:
                raise ValueError(
                    "query_embedding is None and no embed_fn was set on EnhancedSearch. "
                    "Pass embed_fn to the constructor or supply query_embedding directly."
                )
            query_embedding = self._embed_fn([query_text])[0]
        assert query_embedding is not None
        if namespace_filter_llm_fn is not None and namespaces is None and query_text:
            namespaces = self._select_namespaces(query_text, namespace_filter_llm_fn)
        if domain_filter_llm_fn is not None and domain_ids is None and query_text:
            domain_ids = self._select_domains(query_text, domain_filter_llm_fn, namespaces)
        trace = RetrievalTrace(mode=mode) if return_trace else None

        if mode == "global":
            results = self._global_search(query_embedding, k, query_text, namespaces, domain_ids, _trace=trace)
            if trace is not None:
                trace.pool_size = len(results)
                trace.final_provenance = _tally_provenance(results)
            if not _bypass_chunk_filter and self._chunk_filter is not None:
                results = self._chunk_filter(results)
            return (results, trace) if return_trace else results
        if mode == "graph_first":
            results = self._graph_first_search(
                query_embedding, k, query_text, query_entities, precomputed_entity_vecs, namespaces, domain_ids, _trace=trace
            )
            if trace is not None:
                trace.pool_size = len(results)
                trace.final_provenance = _tally_provenance(results)
            if not _bypass_chunk_filter and self._chunk_filter is not None:
                results = self._chunk_filter(results)
            return (results, trace) if return_trace else results
        if mode != "vector_first":
            raise ValueError(f"Unknown search mode {mode!r}. Use 'vector_first', 'graph_first', or 'global'.")

        ns_chunk_ids: set[str] | None = None
        if namespaces:
            placeholders = ", ".join("?" * len(namespaces))
            rows = self._store.vector._conn.execute(
                f"SELECT chunk_id FROM embeddings WHERE namespace IN ({placeholders})",
                list(namespaces),
            ).fetchall()
            ns_chunk_ids = {r[0] for r in rows}
        if domain_ids:
            placeholders = ", ".join("?" * len(domain_ids))
            rows = self._store.vector._conn.execute(
                f"SELECT chunk_id FROM embeddings WHERE domain_id IN ({placeholders})",
                list(domain_ids),
            ).fetchall()
            di_chunk_ids = {r[0] for r in rows}
            ns_chunk_ids = di_chunk_ids if ns_chunk_ids is None else ns_chunk_ids & di_chunk_ids

        seed_limit = k * self._seed_multiplier
        cluster_budget = self._cluster_budget if self._cluster_budget is not None else 2 * k

        if trace is not None:
            trace.query_entities = list(query_entities) if query_entities else []

        # ------ Step 1: Seed -----------------------------------------------
        raw_seeds = self._store.search(query_embedding, limit=seed_limit, query_text=query_text, namespaces=namespaces, domain_ids=domain_ids)
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

        if trace is not None:
            trace.seed_chunk_ids = list(seed_chunk_ids)

        # ------ Step 2: Structural expansion --------------------------------
        if self._structural:
            structural_ids = self._structural_neighbors(seed_chunk_ids)
            _structural_added: list[str] = []
            for chunk_id, chunk in structural_ids:
                if chunk_id not in pool:
                    pool[chunk_id] = ScoredChunk(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        score=0.0,
                        provenance="structural",
                    )
                    _structural_added.append(chunk_id)
            if trace is not None:
                trace.structural_chunk_ids = _structural_added

        # ------ Step 3: Entity expansion ------------------------------------
        entity_source_ids = list(pool.keys())  # seed + structural
        expanded_entity_ids: set[str] = set()
        query_vec_flat = query_embedding.flatten().astype("float32") if self._lane_entity_min_sim is not None else None
        _entity_added: list[str] = []

        if self._entity and self._entity_index is not None:
            for cid in entity_source_ids:
                for entity_id, _ in self._entity_index.get_entities_for_chunk(cid):
                    expanded_entity_ids.add(entity_id)
                    for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                        entity_id, top_n=self._entity_top_n
                    ):
                        if ns_chunk_ids is not None and linked_chunk_id not in ns_chunk_ids:
                            continue
                        if linked_chunk_id not in pool:
                            if query_vec_flat is not None:
                                emb = self._get_embedding(linked_chunk_id)
                                if emb is not None and self._cosine_similarity(query_vec_flat, emb) < self._lane_entity_min_sim:
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
                                _entity_added.append(linked_chunk_id)

        if trace is not None:
            trace.entity_expanded_chunk_ids = _entity_added

        # ------ Step 4: Cluster expansion (budget-limited) ------------------
        cluster_count = 0
        _cluster_added: list[str] = []
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
                        if ns_chunk_ids is not None and linked_chunk_id not in ns_chunk_ids:
                            continue
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
                                _cluster_added.append(linked_chunk_id)

        if trace is not None:
            trace.cluster_expanded_chunk_ids = _cluster_added

        # ------ Step 5: Entity embedding ANN expansion ----------------------
        _entity_embed_added: list[str] = []
        if (
            self._entity_embed
            and self._entity_embeddings is not None
            and self._entity_embedding_ids is not None
            and self._entity_index is not None
            and query_text
            and self._ner_fn is not None
            and (self._embed_fn is not None or precomputed_entity_vecs is not None)
        ):
            import numpy as np
            query_ents = query_entities if query_entities is not None else self._ner_fn(query_text)
            if trace is not None and query_ents and not trace.query_entities:
                trace.query_entities = list(query_ents)
            if query_ents:
                if precomputed_entity_vecs is not None:
                    vecs = [precomputed_entity_vecs[e] for e in query_ents if e in precomputed_entity_vecs]
                    q_ent_vecs = np.stack(vecs) if vecs else None
                    query_ents = [e for e in query_ents if e in precomputed_entity_vecs]
                else:
                    q_ent_vecs = self._embed_fn(query_ents)
                if q_ent_vecs is not None and len(query_ents) > 0:
                    scores = q_ent_vecs @ self._entity_embeddings.T  # (n_query_ents, n_entities)
                    max_scores = scores.max(axis=0)  # (n_entities,)
                    top_indices = np.argsort(-max_scores)[: self._entity_embed_top_k]
                    for idx in top_indices:
                        eid = self._entity_embedding_ids[int(idx)]
                        for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                            eid, top_n=2
                        ):
                            if ns_chunk_ids is not None and linked_chunk_id not in ns_chunk_ids:
                                continue
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
                                    _entity_embed_added.append(linked_chunk_id)

        if trace is not None:
            trace.entity_embed_expanded_chunk_ids = _entity_embed_added

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
                results = self._entity_ref_expand(results, pool, query_embedding, ents, k, query_text, precomputed_entity_vecs, namespaces, domain_ids, _trace=trace)

        if trace is not None:
            trace.pool_size = len(pool)
            trace.final_provenance = _tally_provenance(results)

        if not _bypass_chunk_filter and self._chunk_filter is not None:
            results = self._chunk_filter(results)

        return (results, trace) if return_trace else results

    def ask(
        self,
        query: str,
        embed_fn: Callable[[list[str]], np.ndarray],
        llm_fn: Callable[[str], str],
        k: int = 30,
        mode: str = "vector_first",
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        namespace_filter_llm_fn: Callable[[str], str] | None = None,
        domain_filter_llm_fn: Callable[[str], str] | None = None,
        token_budget: int = 4096,
        redaction_filter: Callable[[Answer], Answer] | None = None,
    ) -> Answer:
        """Embed query, search, generate and return an Answer.

        Args:
            query: Natural-language question.
            embed_fn: ``(texts: list[str]) -> np.ndarray`` — produces query embedding.
            llm_fn: ``(prompt: str) -> str`` — generates the answer.
            k: Retrieval cohort size.
            mode: Search mode passed to search().
            namespaces: Explicit namespace filter; overrides namespace_filter_llm_fn.
            domain_ids: Domain filter passed to search().
            namespace_filter_llm_fn: LLM callable for automatic namespace pre-filtering.
            token_budget: Max prompt tokens passed to AnswerGenerator.
            redaction_filter: Optional ``(Answer) -> Answer`` callable applied to the
                generated answer before it is returned. Overrides the instance-level
                filter set at construction time. Use this to sanitize sensitive data
                before the answer leaves a sovereign deployment perimeter.

                **Sovereign RAG pattern** — frontier model as planner, sovereign model
                as retriever/generator, redaction filter as the trust boundary:

                .. code-block:: python

                    import re
                    from chonk.generation import Answer

                    _SSN = re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b")
                    _EIN = re.compile(r"\\b\\d{2}-\\d{7}\\b")

                    def redact_pii(answer: Answer) -> Answer:
                        clean = _SSN.sub("[SSN REDACTED]", answer.text)
                        clean = _EIN.sub("[EIN REDACTED]", clean)
                        return Answer(text=clean, citations=answer.citations)

                    # Set once at construction — applies to every ask() call:
                    search = EnhancedSearch(store, redaction_filter=redact_pii)

                    # Or pass per-call to override:
                    result = search.ask(query, embed_fn, llm_fn, redaction_filter=redact_pii)

                The filter receives the fully-generated ``Answer`` (text + citations)
                and must return an ``Answer``. It may modify, replace, or raise. The
                frontier planner only ever sees the returned text — raw chunks and
                the original generated text never leave the perimeter.

        ## Agent guidance — when and how to use ask()

        **Use ask() only when the sub-query is fully formed.** ask() is a one-shot
        call: embed → retrieve → generate. There is no opportunity to inspect chunks,
        adjust scope, or retry between steps. If the agent needs to evaluate evidence
        quality or branch on what was found, use search() instead.

        **Frame the query as a complete, self-contained question.** The query string
        drives both the embedding (retrieval) and the prompt (generation). Ambiguous
        or truncated queries produce off-topic retrievals and underspecified answers.
        Include the entity name, the attribute sought, and any scoping constraint
        (fiscal year, version number, jurisdiction) in a single sentence.

        **Scope the namespace before calling.** Pass ``namespaces`` explicitly when
        the source type is known, or pass ``namespace_filter_llm_fn`` to let the
        LLM route automatically. An unscoped ask() on a heterogeneous corpus
        retrieves cross-domain noise that the generator may hallucinate from.

        **Treat Answer.citations as ground truth, not Answer.text.** The generator
        can hallucinate even with good retrieval. When the answer will be used as
        input to a subsequent reasoning step, verify the claim against
        ``Answer.citations`` before propagating it.

        **Use ask() for terminal steps, search() for intermediate steps.** A
        well-structured multi-step plan ends each branch with ask(); intermediate
        steps that feed into further reasoning use search() so the agent retains
        control over how retrieved evidence is interpreted and combined.
        """
        from ..generation import AnswerContext, AnswerGenerator

        query_embedding = embed_fn([query])[0]
        # Bypass chunk_filter — ask() applies redaction_filter on the generated Answer instead.
        chunks: list[ScoredChunk] = self.search(  # type: ignore[call-overload]
            query_embedding,
            k=k,
            query_text=query,
            mode=mode,
            namespaces=namespaces,
            domain_ids=domain_ids,
            namespace_filter_llm_fn=namespace_filter_llm_fn,
            domain_filter_llm_fn=domain_filter_llm_fn,
            _bypass_chunk_filter=True,
        )
        context = AnswerContext(chunks=chunks, query=query)
        answer = AnswerGenerator(llm_fn, token_budget=token_budget).generate(context)
        _filter = redaction_filter if redaction_filter is not None else self._redaction_filter
        if _filter is not None:
            answer = _filter(answer)
        return answer

    # ------------------------------------------------------------------
    # graph_first mode
    # ------------------------------------------------------------------

    def _graph_first_search(
        self,
        query_embedding,
        k: int,
        query_text: str | None,
        query_entities: list[str] | None,
        precomputed_entity_vecs,
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        _trace: RetrievalTrace | None = None,
    ) -> list[ScoredChunk]:
        """Driver: RelationshipIndex traversal. Assist: vector rerank via _select_cohort.

        Falls back to vector_first when prerequisites are absent:
          - no relationship_index
          - no query_text and no query_entities provided
          - NER produces no entity hits
        """
        fallback = (
            self._relationship_index is None
            or self._entity_index is None
            or (query_text is None and not query_entities)
        )
        if fallback:
            return self.search(
                query_embedding, k=k, query_text=query_text,
                query_entities=query_entities,
                precomputed_entity_vecs=precomputed_entity_vecs,
                mode="vector_first",
                namespaces=namespaces,
                domain_ids=domain_ids,
            )

        # Resolve query entity IDs (slugs) for graph traversal
        ents = None
        if self._query_entity_id_fn is not None and query_text:
            ents = self._query_entity_id_fn(query_text)
        if not ents:
            return self.search(
                query_embedding, k=k, query_text=query_text,
                query_entities=query_entities,
                precomputed_entity_vecs=precomputed_entity_vecs,
                mode="vector_first",
                namespaces=namespaces,
                domain_ids=domain_ids,
            )

        if _trace is not None:
            _trace.query_entities = list(ents)

        # Traverse RelationshipIndex: 2-hop traversal
        ents_set = set(ents)
        hop1: set[str] = set()
        svo_count = 0
        for entity_id in ents:
            for triple in self._relationship_index.get_objects(entity_id):
                hop1.add(triple.object_id)
                svo_count += 1
            for triple in self._relationship_index.get_subjects(entity_id):
                hop1.add(triple.subject_id)
                svo_count += 1
        hop1 -= ents_set

        hop2: set[str] = set()
        for entity_id in hop1:
            for triple in self._relationship_index.get_objects(entity_id):
                hop2.add(triple.object_id)
            for triple in self._relationship_index.get_subjects(entity_id):
                hop2.add(triple.subject_id)
        hop2 -= ents_set | hop1

        related = hop1 | hop2

        if _trace is not None:
            _trace.graph_traversal_entities = list(related)
            _trace.svo_triples_count = svo_count

        ns_chunk_ids: set[str] | None = None
        if namespaces:
            placeholders = ", ".join("?" * len(namespaces))
            rows = self._store.vector._conn.execute(
                f"SELECT chunk_id FROM embeddings WHERE namespace IN ({placeholders})",
                list(namespaces),
            ).fetchall()
            ns_chunk_ids = {r[0] for r in rows}
        if domain_ids:
            placeholders = ", ".join("?" * len(domain_ids))
            rows = self._store.vector._conn.execute(
                f"SELECT chunk_id FROM embeddings WHERE domain_id IN ({placeholders})",
                list(domain_ids),
            ).fetchall()
            di_chunk_ids = {r[0] for r in rows}
            ns_chunk_ids = di_chunk_ids if ns_chunk_ids is None else ns_chunk_ids & di_chunk_ids

        # Build pool from related-entity chunks
        pool: dict[str, ScoredChunk] = {}
        for related_entity_id in related:
            for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                related_entity_id, top_n=self._entity_top_n
            ):
                if ns_chunk_ids is not None and linked_chunk_id not in ns_chunk_ids:
                    continue
                if linked_chunk_id not in pool:
                    chunk = self._fetch_chunk(linked_chunk_id)
                    if chunk is not None:
                        pool[linked_chunk_id] = ScoredChunk(
                            chunk_id=linked_chunk_id,
                            chunk=chunk,
                            score=0.0,
                            provenance="entity_adjacent",
                            linked_by=related_entity_id,
                        )

        # Augment with vector seeds so reranker has enough candidates
        seed_limit = max(k * self._seed_multiplier, k - len(pool))
        for chunk_id, score, chunk in self._store.search(
            query_embedding, limit=seed_limit, query_text=query_text, namespaces=namespaces, domain_ids=domain_ids
        ):
            if chunk_id not in pool:
                pool[chunk_id] = ScoredChunk(
                    chunk_id=chunk_id, chunk=chunk, score=score, provenance="seed"
                )

        if not pool:
            return []

        return self._select_cohort(list(pool.values()), query_embedding, k)

    # ------------------------------------------------------------------
    # global mode
    # ------------------------------------------------------------------

    def _global_search(
        self,
        query_embedding,
        k: int,
        query_text: str | None,
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        _trace: RetrievalTrace | None = None,
    ) -> list[ScoredChunk]:
        """Driver: vector search over community_summary chunks only."""
        raw = self._store.search(
            query_embedding,
            limit=k,
            query_text=query_text,
            chunk_types=["community_summary"],
            namespaces=namespaces,
            domain_ids=domain_ids,
            session_fingerprint=self._session_fingerprint,
        )
        results: list[ScoredChunk] = []
        for chunk_id, score, chunk in raw:
            results.append(ScoredChunk(
                chunk_id=chunk_id,
                chunk=chunk,
                score=score,
                provenance="seed",
            ))
        if _trace is not None:
            _trace.community_chunk_ids = [sc.chunk_id for sc in results]
        return results

    # ------------------------------------------------------------------
    # MS-GraphRAG context assembly
    # ------------------------------------------------------------------

    def assemble_graph_context(
        self,
        hits: list,
        query_text: str | None = None,
        query_entities: list[str] | None = None,
        context_token_budget: int = 8000,
    ) -> str:
        """Assemble MS-GraphRAG-style structured context from retrieved chunks.

        Sections: Entities | Relationships | Community Reports | Source Text

        Each section is budget-trimmed so the total approximate token count
        (chars / 4) stays within *context_token_budget*.

        Args:
            hits: List of ``(chunk_id, score, DocumentChunk)`` tuples or ScoredChunk objects.
            query_text: Optional query text for NER-based entity resolution.
            query_entities: Pre-resolved entity IDs; used if query_text NER is absent.
            namespace: Namespace for entity description lookups.
        """
        # Normalise hits → (chunk_id, DocumentChunk)
        chunk_pairs: list[tuple[str, DocumentChunk]] = []
        for h in hits:
            if isinstance(h, ScoredChunk):
                chunk_pairs.append((h.chunk_id, h.chunk))
            else:
                chunk_pairs.append((h[0], h[2]))
        chunk_ids = [cid for cid, _ in chunk_pairs]

        # ── 1. Collect entity IDs from retrieved chunks ────────────────────
        entity_ids: set[str] = set()
        if self._entity_index is not None:
            for cid in chunk_ids:
                for eid, _ in self._entity_index.get_entities_for_chunk(cid):
                    entity_ids.add(eid)

        # Add query entity IDs via entity-ID resolver
        if self._query_entity_id_fn is not None and query_text:
            entity_ids.update(self._query_entity_id_fn(query_text))

        # ── 2. Fetch entity records (name, type, description) ──────────────
        entity_records: list[dict] = []
        if entity_ids:
            conn = self._store.vector._conn
            eid_list = list(entity_ids)
            placeholders = ", ".join("?" * len(eid_list))
            rows = conn.execute(
                f"SELECT e.id, e.name, e.entity_type, COALESCE(e.description, '') "
                f"FROM entities e "
                f"WHERE e.id IN ({placeholders})",
                eid_list,
            ).fetchall()
            entity_records = [
                {"id": r[0], "name": r[1], "type": r[2], "description": r[3]}
                for r in rows
            ]

        # ── 3. Collect all 1-hop SVO triples from matched entities ────────────
        # Include all relationships where a matched entity is subject or object,
        # regardless of whether the far endpoint was retrieved (MS-GraphRAG behaviour).
        rel_rows: list[tuple[str, str, str, str]] = []  # (subj_name, verb, obj_name, desc)
        if self._relationship_index is not None and entity_ids:
            id_to_name = {r["id"]: r["name"] for r in entity_records}
            seen: set[tuple[str, str, str]] = set()
            for eid in entity_ids:
                for t in self._relationship_index.get_objects(eid):
                    key = (t.subject_id, t.verb, t.object_id)
                    if key not in seen:
                        seen.add(key)
                        rel_rows.append((
                            id_to_name.get(t.subject_id, t.subject_id),
                            t.verb,
                            id_to_name.get(t.object_id, t.object_id),
                            t.description or "",
                        ))

        # ── 4. Community summaries for entity communities ──────────────────
        community_texts: list[str] = []
        if self._community_index is not None and chunk_ids:
            comm_ids: set[int] = set()
            for cid in chunk_ids:
                c = self._community_index.community_id(cid)
                if c is not None:
                    comm_ids.add(c)
            if comm_ids:
                conn = self._store.vector._conn
                names = [f"community:{cid}" for cid in comm_ids]
                placeholders = ", ".join("?" * len(names))
                rows = conn.execute(
                    f"SELECT content FROM embeddings "
                    f"WHERE document_name IN ({placeholders}) "
                    f"AND chunk_type = 'community_summary'",
                    names,
                ).fetchall()
                community_texts = [r[0] for r in rows if r[0]]

        # ── 5. Budget-aware assembly ───────────────────────────────────────
        # Approximate tokens = chars / 4. Fill sections in priority order:
        # Entities → Relationships → Community Reports → Source Text.
        budget_chars = context_token_budget * 4
        used = 0
        sections: list[str] = []

        def _chars(s: str) -> int:
            return len(s)

        def _add(block: str) -> bool:
            nonlocal used
            c = _chars(block)
            if used + c > budget_chars:
                return False
            sections.append(block)
            used += c
            return True

        if entity_records:
            header = "## Entities\n\n| Name | Type | Description |\n|------|------|-------------|"
            rows_str = "\n".join(
                f"| {r['name']} | {r['type']} | {r['description'] or '—'} |"
                for r in entity_records
            )
            _add(f"{header}\n{rows_str}")

        if rel_rows:
            header = "## Relationships\n\n| Subject | Relationship | Object | Description |\n|---------|-------------|--------|-------------|"
            rows_str = "\n".join(
                f"| {subj} | {verb} | {obj} | {desc or '—'} |"
                for subj, verb, obj, desc in rel_rows
            )
            _add(f"{header}\n{rows_str}")

        for text in community_texts:
            block = f"## Community Reports\n\n{text}" if not any(
                s.startswith("## Community Reports") for s in sections
            ) else text
            if not _add(block):
                break

        for _, chunk in chunk_pairs:
            block = f"## Source Text\n\n{chunk.content or ''}" if not any(
                s.startswith("## Source Text") for s in sections
            ) else (chunk.content or "")
            if not _add(block):
                break

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # MS-GraphRAG map-reduce global context
    # ------------------------------------------------------------------

    _MAP_PROMPT = (
        "You are an expert analyst. Using ONLY the community report below, "
        "provide a concise answer to the question and rate how relevant this "
        "community report is (0 = not relevant, 100 = fully answers the question).\n\n"
        "Return ONLY valid JSON: {{\"answer\": \"<answer>\", \"score\": <0-100>}}\n\n"
        "Question: {query}\n\n"
        "Community Report:\n{community_text}"
    )

    def map_reduce_global_context(
        self,
        hits: list,
        query_text: str,
        llm_fn: Callable[[str], str],
        concurrency: int = 4,
    ) -> str:
        """MS-GraphRAG map-reduce global context assembly.

        Map: call *llm_fn* once per community summary hit with a scoring prompt.
        Reduce: filter score > 0, sort by score desc, format top answers as context.

        Args:
            hits: Community summary chunks as ``(chunk_id, score, DocumentChunk)``
                tuples or ``ScoredChunk`` objects.
            query_text: The user's question.
            llm_fn: ``(prompt: str) -> str`` — LLM text completion callable.
            concurrency: Thread-pool size for parallel map calls.

        Returns:
            Formatted context string ready for final generation.
        """
        import concurrent.futures
        import json as _json

        chunk_pairs: list[tuple[str, DocumentChunk]] = []
        for h in hits:
            if isinstance(h, ScoredChunk):
                chunk_pairs.append((h.chunk_id, h.chunk))
            else:
                chunk_pairs.append((h[0], h[2]))

        def _map_one(pair: tuple[str, DocumentChunk]) -> tuple[str, int] | None:
            _, chunk = pair
            text = (chunk.content or "").strip()
            if not text:
                return None
            prompt = self._MAP_PROMPT.format(query=query_text, community_text=text)
            try:
                raw = llm_fn(prompt)
                # Extract JSON from response (may be wrapped in ```json ... ```)
                m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
                parsed = _json.loads(m.group() if m else raw)
                answer = str(parsed.get("answer", "")).strip()
                score = int(parsed.get("score", 0))
            except Exception:
                return None
            if not answer or score <= 0:
                return None
            return answer, score

        intermediate: list[tuple[str, int]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_map_one, p) for p in chunk_pairs]
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                if result is not None:
                    intermediate.append(result)

        if not intermediate:
            return ""

        intermediate.sort(key=lambda x: x[1], reverse=True)

        lines = ["## Intermediate Answers\n"]
        for answer, score in intermediate:
            lines.append(f"### Community Report (relevance: {score})\n{answer}\n")
        return "\n".join(lines)

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
        precomputed_entity_vecs: dict[str, np.ndarray] | None = None,
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
        _trace: RetrievalTrace | None = None,
    ) -> list[ScoredChunk]:
        """Post-selection: if query entities are absent from top-k, expand via semantic search."""
        result_text = " ".join((sc.chunk.content or "") for sc in results).lower()
        missing = [e for e in query_entities if e.lower() not in result_text]

        if not missing:
            self.last_expansion_stats = {"invoked": False, "missing_entities": []}
            if _trace is not None:
                _trace.ref_expansion_missing = []
                _trace.ref_expansion_found = []
                _trace.ref_expansion_chunks_added = 0
            return results

        new_pool = dict(existing_pool)
        found_entities: list[str] = []

        if self._context_graph_expansion and self._entity_index is not None:
            namespace = namespaces[0] if namespaces else "global"
            for entity in missing:
                edges = self._store.get_context_graph(
                    entity,
                    namespace=namespace,
                    min_weight=self._context_graph_min_weight,
                )
                for edge in edges[: self._context_graph_top_k]:
                    for linked_chunk_id, _ in self._entity_index.get_chunks_for_entity(
                        edge.target_entity_id, top_n=self._entity_top_n
                    ):
                        if linked_chunk_id not in new_pool:
                            chunk = self._fetch_chunk(linked_chunk_id)
                            if chunk is not None:
                                new_pool[linked_chunk_id] = ScoredChunk(
                                    chunk_id=linked_chunk_id,
                                    chunk=chunk,
                                    score=edge.weight,
                                    provenance="context_graph_expansion",
                                    linked_by=edge.target_entity_id,
                                )
                if entity not in found_entities:
                    found_entities.append(entity)

        if self._embed_fn is not None or precomputed_entity_vecs is not None:
            # Semantic: per-entity vector search
            if self._entity_ref_expansion_per_k is not None:
                per_k = self._entity_ref_expansion_per_k
            else:
                per_k = max(3, self._entity_ref_expansion_k // max(len(missing), 1))
            min_sim = self._entity_ref_expansion_min_sim
            if precomputed_entity_vecs is not None:
                available = [e for e in missing if e in precomputed_entity_vecs]
                entity_vecs = np.stack([precomputed_entity_vecs[e] for e in available]) if available else None
                missing = available
            else:
                entity_vecs = self._embed_fn(missing)
            _missing_iter = missing if (entity_vecs is not None and len(missing) > 0) else []
            for i, entity in enumerate(_missing_iter):
                hits = self._store.search(entity_vecs[i], limit=per_k, query_text=None, namespaces=namespaces, domain_ids=domain_ids)
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
                query_embedding, limit=self._entity_ref_expansion_k, query_text=query_text, namespaces=namespaces, domain_ids=domain_ids
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

        chunks_added = len(new_pool) - len(existing_pool)
        cg_chunks = sum(
            1 for sc in new_pool.values() if sc.provenance == "context_graph_expansion"
        )
        self.last_expansion_stats = {
            "invoked": True,
            "missing_entities": missing,
            "found_entities": found_entities,
            "unresolved_entities": [e for e in missing if e not in found_entities],
            "new_chunks_added": chunks_added,
            "context_graph_chunks_added": cg_chunks,
        }
        if _trace is not None:
            _trace.ref_expansion_missing = list(missing)
            _trace.ref_expansion_found = list(found_entities)
            _trace.ref_expansion_chunks_added = chunks_added
            _trace.context_graph_expansion_chunks_added = cg_chunks

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

    def _cosine_similarity(self, a, b) -> float:
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
        query_embedding,
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
            if units[chosen_i] is not None:
                sel_units.append(units[chosen_i])
            else:
                sel_units.append(np.zeros(dim, dtype=np.float32))
            result.append(sc)

        return result