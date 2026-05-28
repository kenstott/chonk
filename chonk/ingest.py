# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Config-driven build and search: build(config) -> Index."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from .indexer import IndexHandle
from .loader import DocumentLoader
from .models import ScoredChunk
from .storage import Store

if TYPE_CHECKING:
    from .search._enhanced import EnhancedSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_id(*parts: str) -> str:
    """Stable 16-char hex ID from string parts."""
    return hashlib.sha1(":".join(parts).encode()).hexdigest()[:16]


def _make_extractor(name: str):
    if name == "edgar":
        from .extractors._edgar import EdgarExtractor
        return EdgarExtractor(infer_bold_headings=True)
    raise ValueError(f"Unknown extractor: {name!r}")


def _make_loader(cfg: dict, enrich_override: bool | None = None) -> DocumentLoader:
    lc = cfg.get("loader", {})
    extractors = [_make_extractor(e) for e in lc.get("extra_extractors", [])]
    enrich = enrich_override if enrich_override is not None else lc.get("enrich_context", True)
    return DocumentLoader(
        min_chunk_size=lc.get("min_chunk_size", 1100),
        max_chunk_size=lc.get("max_chunk_size", 2200),
        enrich_context=enrich,
        extra_extractors=extractors or None,
    )


def _ingest_glob(loader: DocumentLoader, src: dict) -> list:
    base = Path(src["path"])
    pattern = src.get("pattern", "*")
    prefix = src.get("name_prefix", "")
    doc_type = src.get("doc_type")
    chunks = []
    for p in sorted(f for f in base.glob(pattern) if not f.name.startswith(".")):
        name = prefix + p.stem
        if doc_type:
            chunks.extend(loader.load_bytes(p.read_bytes(), name=name, doc_type=doc_type))
        else:
            chunks.extend(loader.load(str(p), name=name))
    return chunks


def _ingest_json_array(loader: DocumentLoader, src: dict) -> list:
    path = Path(src["path"])
    array_field = src["array_field"]
    id_path: list[str] = src.get("id_path", "").split(".") if src.get("id_path") else []
    obj = json.loads(path.read_text())
    items = obj.get(array_field, []) if isinstance(obj, dict) else obj
    chunks = []
    for item in items:
        name = item
        for key in id_path:
            name = name.get(key, {}) if isinstance(name, dict) else "unknown"
        name = str(name) if name else path.stem
        payload = json.dumps({array_field: [item]}).encode()
        chunks.extend(loader.load_bytes(payload, name=name, doc_type="json"))
    return chunks


def _ingest_sql(loader: DocumentLoader, src: dict) -> list:
    connection = src["connection"]
    query = src["query"].strip()
    name = src.get("name", "sql_source")
    name_col = src.get("name_col")
    content_col = src.get("content_col")
    if name_col and content_col:
        import duckdb
        db_path = connection.replace("duckdb:///", "")
        conn = duckdb.connect(db_path, read_only=True)
        rows = conn.execute(query).fetchall()
        col_names = [d[0] for d in conn.description]
        conn.close()
        name_idx = col_names.index(name_col)
        content_idx = col_names.index(content_col)
        chunks = []
        for row in rows:
            chunks.extend(
                loader.load_bytes(str(row[content_idx]).encode(), name=str(row[name_idx]), doc_type="text")
            )
        return chunks
    return loader.load_from_db(connection, queries={name: query})


_INGEST_FNS = {
    "glob": _ingest_glob,
    "json_array": _ingest_json_array,
    "sql": _ingest_sql,
}


def _embed_chunks(chunks: list, cfg: dict) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    ec = cfg.get("embed", {})
    model_name = ec.get("model", "BAAI/bge-large-en-v1.5")
    batch_size = ec.get("batch_size", 256)
    model = SentenceTransformer(model_name)
    texts = [c.content for c in chunks]
    vecs = []
    for i in range(0, len(texts), batch_size):
        v = model.encode(
            texts[i: i + batch_size],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vecs.append(np.array(v, dtype="float32"))
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0:
            print(f"  embedded {done:,}/{len(texts):,}", flush=True)
    return np.vstack(vecs)


def _embed_texts(texts: list[str], model_name: str, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vecs = []
    for i in range(0, len(texts), batch_size):
        v = model.encode(texts[i: i + batch_size], show_progress_bar=False, normalize_embeddings=True)
        vecs.append(np.array(v, dtype="float32"))
    return np.vstack(vecs)


def _ingest_source(src: dict, loader: DocumentLoader) -> list:
    stype = src.get("type") or ""
    fn = _INGEST_FNS.get(stype)
    if fn is None:
        raise ValueError(f"Unknown source type: {stype!r}")
    return fn(loader, src)


# ---------------------------------------------------------------------------
# Index — returned by build()
# ---------------------------------------------------------------------------

class Index:
    """Returned by :func:`build`. Wraps a fully-built Store with search and RT mutation.

    Domain names are fully-qualified strings (e.g. ``"sales/north-america/q1"``).
    The client owns the hierarchy and passes explicit FQ name lists when filtering.

    Args:
        store: Open Store instance.
        embed_model: SentenceTransformer model name — used for query embedding and rebuilds.
        search_defaults: Constructor kwargs forwarded to EnhancedSearch.
        index_cfg: ``index:`` config section — drives rebuild phases.
        loader_cfg: ``loader:`` config section — drives source ingestion.
        embed_cfg: ``embed:`` config section.
    """

    def __init__(
        self,
        store: Store,
        embed_model: str,
        search_defaults: dict,
        index_cfg: dict,
        loader_cfg: dict,
        embed_cfg: dict,
    ):
        self._store = store
        self._embed_model = embed_model
        self._search_defaults = search_defaults
        self._index_cfg = index_cfg
        self._loader_cfg = loader_cfg
        self._embed_cfg = embed_cfg
        self._enhanced: EnhancedSearch | None = None
        self._domain_map: dict[str, dict[str, str]] = self._load_domain_map()
        _routing_fn = self._make_routing_fn()
        self.namespace_filter_llm_fn: Callable[[str], str] | None = _routing_fn
        self.domain_filter_llm_fn: Callable[[str], str] | None = _routing_fn

    # -- internal --------------------------------------------------------------

    def _load_domain_map(self) -> dict[str, dict[str, str]]:
        """Reload {namespace_id: {fq_domain_name: domain_id}} from DB."""
        rows = self._store.vector._conn.execute(
            "SELECT namespace_id, name, domain_id FROM domains"
        ).fetchall()
        result: dict[str, dict[str, str]] = {}
        for ns, name, did in rows:
            result.setdefault(ns, {})[name] = did
        return result

    def _refresh_domain_map(self) -> None:
        self._domain_map = self._load_domain_map()

    def _resolve_domains(
        self,
        namespaces: list[str] | None,
        domains: list[str] | None,
    ) -> list[str] | None:
        if namespaces is None and domains is None:
            return None
        target_ns = namespaces or list(self._domain_map.keys())
        ids: list[str] = []
        if domains is not None:
            for ns in target_ns:
                ns_map = self._domain_map.get(ns, {})
                for name in domains:
                    if name in ns_map:
                        ids.append(ns_map[name])
        else:
            for ns in target_ns:
                ids.extend(self._domain_map.get(ns, {}).values())
        return ids or None

    def _get_or_register_domain(self, namespace_id: str, domain_name: str, description: str | None = None) -> str:
        ns_map = self._domain_map.get(namespace_id, {})
        if domain_name in ns_map:
            return ns_map[domain_name]
        did = _stable_id(namespace_id, domain_name)
        self._store.register_domain(did, namespace_id, domain_name, description=description)
        self._domain_map.setdefault(namespace_id, {})[domain_name] = did
        return did

    def _make_routing_fn(self) -> Callable[[str], str] | None:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        model = self._index_cfg.get("routing_model") or self._index_cfg.get("svo_model", "gpt-4o-mini")
        try:
            from openai import OpenAI as _OpenAI
            _client = _OpenAI()

            def _fn(prompt: str) -> str:
                resp = _client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return resp.choices[0].message.content or ""

            return _fn
        except ImportError:
            return None

    def _make_embed_fn(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self._embed_model)

        def _embed_fn(texts: list[str]) -> np.ndarray:
            return np.array(
                model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype="float32",
            )
        return _embed_fn

    def _get_search(self) -> EnhancedSearch:
        if self._enhanced is None:
            from .search._enhanced import EnhancedSearch
            self._enhanced = EnhancedSearch(
                self._store,
                embed_fn=self._make_embed_fn(),
                **self._search_defaults,
            )
        return self._enhanced

    def _invalidate_search(self) -> None:
        """Drop cached EnhancedSearch so it reloads indexes on next call."""
        self._enhanced = None

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._store.close()

    # -- properties ------------------------------------------------------------

    @property
    def store(self) -> Store:
        return self._store

    # -- domain / namespace inspection -----------------------------------------

    def domains(self, namespace: str | None = None) -> dict[str, list[str]]:
        """Return registered fully-qualified domain names.

        Args:
            namespace: If set, return only domains for this namespace.

        Returns:
            ``{namespace_id: [fq_domain_name, ...]}`` sorted dict.
        """
        if namespace is not None:
            return {namespace: sorted(self._domain_map.get(namespace, {}).keys())}
        return {ns: sorted(names.keys()) for ns, names in self._domain_map.items()}

    # -- RT mutations ----------------------------------------------------------

    def add_namespace(self, namespace_id: str, description: str | None = None) -> None:
        """Register a namespace. No-op if already registered."""
        self._store.register_namespace(namespace_id, description=description)
        self._domain_map.setdefault(namespace_id, {})

    def remove_namespace(self, namespace_id: str) -> None:
        """Delete a namespace and all its domains, chunks, and secondary index rows."""
        conn = self._store.vector._conn
        domain_ids = list(self._domain_map.get(namespace_id, {}).values())
        if domain_ids:
            placeholders = ", ".join("?" * len(domain_ids))
            conn.execute(f"DELETE FROM chunk_entities WHERE chunk_id IN (SELECT chunk_id FROM embeddings WHERE domain_id IN ({placeholders}))", domain_ids)
            conn.execute(f"DELETE FROM svo_triples WHERE chunk_id IN (SELECT chunk_id FROM embeddings WHERE domain_id IN ({placeholders}))", domain_ids)
            conn.execute(f"DELETE FROM embeddings WHERE domain_id IN ({placeholders})", domain_ids)
            conn.execute(f"DELETE FROM domains WHERE domain_id IN ({placeholders})", domain_ids)
        conn.execute("DELETE FROM namespaces WHERE namespace_id = ?", [namespace_id])
        self._domain_map.pop(namespace_id, None)
        self._invalidate_search()

    def add_domain(self, namespace_id: str, domain_name: str, description: str | None = None) -> str:
        """Register a domain. Returns the domain_id. No-op if already registered.

        Args:
            namespace_id: Parent namespace (must be registered first).
            domain_name: Fully-qualified domain name, e.g. ``"sales/north-america/q1"``.
            description: Optional human-readable description.

        Returns:
            Stable domain_id string.
        """
        return self._get_or_register_domain(namespace_id, domain_name, description)

    def remove_domain(self, namespace_id: str, domain_name: str) -> int:
        """Delete a domain and all its chunks. Returns chunk count deleted.

        Args:
            namespace_id: Namespace containing the domain.
            domain_name: Fully-qualified domain name.
        """
        ns_map = self._domain_map.get(namespace_id, {})
        domain_id = ns_map.get(domain_name)
        if domain_id is None:
            return 0
        n = self._store.delete_domain(domain_id)
        self._domain_map.get(namespace_id, {}).pop(domain_name, None)
        self._invalidate_search()
        return n

    def add_source(
        self,
        src: dict,
        *,
        rebuild: bool = True,
        on_progress: Any = None,
        on_complete: Any = None,
        on_error: Any = None,
    ) -> IndexHandle | None:
        """Ingest a new source, register its domain, and optionally trigger async rebuild.

        Args:
            src: Source dict — same schema as a ``sources:`` entry in the config.
                 Required keys: ``type``, and type-specific path/connection.
                 Optional keys: ``name``, ``namespace``, ``domain``, ``enrich_context``.
            rebuild: If True (default), trigger async secondary-index rebuild for the
                     namespace after ingest. Returns an :class:`IndexHandle` you can
                     ``.join()`` to wait for completion.
            on_progress: ``(phase, done, total) -> None`` callback for rebuild phases.
            on_complete: ``(total_chunks) -> None`` callback on rebuild success.
            on_error: ``(phase, exc) -> None`` callback on rebuild error.

        Returns:
            :class:`IndexHandle` if rebuild=True, else None.
        """
        namespace_id = src.get("namespace") or "global"
        domain_name = src.get("domain") or src.get("name", src.get("type", "unnamed"))
        enrich_override = src.get("enrich_context")
        loader_cfg = {"loader": self._loader_cfg}
        loader = _make_loader(loader_cfg, enrich_override=enrich_override)

        # Ensure namespace and domain exist
        if namespace_id not in self._domain_map:
            self._store.register_namespace(namespace_id)
            self._domain_map[namespace_id] = {}
        domain_id = self._get_or_register_domain(namespace_id, domain_name)

        # Ingest
        chunks = _ingest_source(src, loader)
        if not chunks:
            return None

        texts = [c.content for c in chunks]
        emb = _embed_texts(texts, self._embed_model, self._embed_cfg.get("batch_size", 256))
        self._store.add_document(chunks, emb, namespace=namespace_id, domain_id=domain_id)
        self._store.vector.rebuild_fts_index()
        self._invalidate_search()

        if rebuild:
            from .lifecycle import build_namespace_async
            return build_namespace_async(
                namespace_id,
                self._store.vector._conn.execute("PRAGMA database_list").fetchone()[2],
                self._embed_model,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error,
                force=True,
                run_ner=self._index_cfg.get("ner", True),
                run_community=self._index_cfg.get("community", True),
                spacy_model=self._index_cfg.get("spacy_model", "en_core_web_sm"),
                community_alpha=self._index_cfg.get("community_alpha", 0.2),
                community_sim_threshold=self._index_cfg.get("community_sim_threshold", 0.6),
            )
        return None

    def remove_source(
        self,
        namespace_id: str,
        domain_name: str,
        *,
        rebuild: bool = True,
        on_progress: Any = None,
        on_complete: Any = None,
        on_error: Any = None,
    ) -> IndexHandle | None:
        """Remove all chunks for a domain and optionally rebuild secondary indexes.

        Args:
            namespace_id: Namespace containing the domain.
            domain_name: Fully-qualified domain name to remove.
            rebuild: If True, trigger async secondary-index rebuild.

        Returns:
            :class:`IndexHandle` if rebuild=True and namespace has remaining chunks, else None.
        """
        self.remove_domain(namespace_id, domain_name)

        if rebuild and self._domain_map.get(namespace_id):
            from .lifecycle import build_namespace_async
            return build_namespace_async(
                namespace_id,
                self._store.vector._conn.execute("PRAGMA database_list").fetchone()[2],
                self._embed_model,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error,
                force=True,
                run_ner=self._index_cfg.get("ner", True),
                run_community=self._index_cfg.get("community", True),
                spacy_model=self._index_cfg.get("spacy_model", "en_core_web_sm"),
                community_alpha=self._index_cfg.get("community_alpha", 0.2),
                community_sim_threshold=self._index_cfg.get("community_sim_threshold", 0.6),
            )
        return None

    def rebuild(
        self,
        namespace_id: str | None = None,
        *,
        async_: bool = True,
        on_progress: Any = None,
        on_complete: Any = None,
        on_error: Any = None,
    ) -> list[IndexHandle]:
        """Trigger secondary-index rebuild for one or all namespaces.

        Args:
            namespace_id: If set, rebuild only this namespace. Otherwise rebuild all.
            async_: If True (default), run in background threads. If False, block.
            on_progress: ``(phase, done, total) -> None`` per-phase callback.
            on_complete: ``(total_chunks) -> None`` on success.
            on_error: ``(phase, exc) -> None`` on failure.

        Returns:
            List of :class:`IndexHandle` objects (one per namespace rebuilt).
        """
        from .lifecycle import build_namespace_async

        db_path = self._store.vector._conn.execute("PRAGMA database_list").fetchone()[2]
        namespaces = [namespace_id] if namespace_id else list(self._domain_map.keys())
        handles = []
        for ns in namespaces:
            h = build_namespace_async(
                ns,
                db_path,
                self._embed_model,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error,
                force=True,
                run_ner=self._index_cfg.get("ner", True),
                run_community=self._index_cfg.get("community", True),
                spacy_model=self._index_cfg.get("spacy_model", "en_core_web_sm"),
                community_alpha=self._index_cfg.get("community_alpha", 0.2),
                community_sim_threshold=self._index_cfg.get("community_sim_threshold", 0.6),
            )
            handles.append(h)
        if not async_:
            for h in handles:
                h.join()
            self._invalidate_search()
        return handles

    # -- search ----------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        mode: str | None = None,
        namespaces: list[str] | None = None,
        domains: list[str] | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Search the index.

        Args:
            query: Natural-language query string.
            k: Number of results (overrides config default).
            mode: ``"vector_first"`` | ``"graph_first"`` | ``"global"``.
            namespaces: Restrict to these namespace IDs. None = all.
            domains: Restrict to these fully-qualified domain names. None = all.
                     Combined with ``namespaces``: matches names within those namespaces only.
            **kwargs: Forwarded to :meth:`EnhancedSearch.search`.
        """
        es = self._get_search()
        call_k = k if k is not None else self._search_defaults.get("k", 10)
        call_mode = mode if mode is not None else self._search_defaults.get("mode", "vector_first")
        domain_ids = self._resolve_domains(namespaces, domains)
        kwargs.setdefault("namespace_filter_llm_fn", self.namespace_filter_llm_fn)
        kwargs.setdefault("domain_filter_llm_fn", self.domain_filter_llm_fn)
        return es.search(
            query_text=query,
            k=call_k,
            mode=call_mode,
            domain_ids=domain_ids,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

def build(config: str | Path | dict, *, force: bool = False) -> Index:
    """Build a fully-indexed store from a YAML config and return an :class:`Index`.

    Phases: ingest → embed → FTS → NER → community → SVO (opt-in).
    Each phase is skipped if already built (idempotent). Use ``force=True`` to rebuild.

    Config schema:

    .. code-block:: yaml

        store:
          path: my.duckdb
          embedding_dim: 1024          # default 1024

        loader:
          min_chunk_size: 1100
          max_chunk_size: 2200
          enrich_context: true

        embed:
          model: BAAI/bge-large-en-v1.5
          batch_size: 256

        index:
          ner: true
          community: true
          svo: false                   # requires LLM API access
          spacy_model: en_core_web_sm
          svo_model: gpt-4o-mini
          community_alpha: 0.2
          community_sim_threshold: 0.6

        search:
          k: 10
          mode: vector_first
          entity_ref_expansion: true
          lane_entity_min_sim: 0.60

        namespaces:
          global:
            description: "Shared company knowledge"
            domains:
              sales: "Sales data"
              sales/north-america: "North America sales"
              legal: "Legal and compliance"
          user:alice:
            domains:
              my-notes: "Alice's notes"

        sources:
          - name: na-sales
            type: glob
            path: ./data/na
            pattern: "*.pdf"
            namespace: global
            domain: sales/north-america  # fully-qualified; defaults to source name
    """
    if not isinstance(config, dict):
        cfg: dict[str, Any] = yaml.safe_load(Path(config).read_text())
    else:
        cfg = config

    sc = cfg.get("store", {})
    db_path = Path(sc["path"])
    embedding_dim = sc.get("embedding_dim", 1024)

    embed_cfg = cfg.get("embed", {})
    embed_model_name: str = embed_cfg.get("model", "BAAI/bge-large-en-v1.5")

    loader_cfg = cfg.get("loader", {})
    ic = cfg.get("index", {})
    run_ner = ic.get("ner", True)
    run_community = ic.get("community", True)
    run_svo = ic.get("svo", False)
    spacy_model = ic.get("spacy_model", "en_core_web_sm")
    svo_model = ic.get("svo_model", "gpt-4o-mini")
    community_alpha = ic.get("community_alpha", 0.2)
    community_sim_threshold = ic.get("community_sim_threshold", 0.6)

    search_defaults: dict[str, Any] = {
        "entity_ref_expansion": True,
        "lane_entity_min_sim": 0.60,
    }
    search_defaults.update(cfg.get("search", {}))

    # ── Phase: ingest ────────────────────────────────────────────────────────
    if db_path.exists() and force:
        db_path.unlink()
        print(f"Removed {db_path}")

    store = Store(db_path, embedding_dim=embedding_dim)

    if store.count() == 0 or force:
        default_loader = _make_loader({"loader": loader_cfg})

        # Register namespaces and their domain dictionaries
        ns_domains: dict[str, dict[str, str]] = {}  # ns → {fq_name: description}
        for ns_id, ns_cfg in (cfg.get("namespaces") or {}).items():
            if not isinstance(ns_cfg, dict):
                ns_cfg = {}
            store.register_namespace(ns_id, description=ns_cfg.get("description"))
            ns_domains[ns_id] = ns_cfg.get("domains") or {}

        seen_domain: dict[str, str] = {}  # "{ns}:{fq_name}" → domain_id

        def _get_domain_id(namespace_id: str, domain_name: str) -> str:
            key = f"{namespace_id}:{domain_name}"
            if key not in seen_domain:
                desc = ns_domains.get(namespace_id, {}).get(domain_name)
                did = _stable_id(namespace_id, domain_name)
                store.register_domain(did, namespace_id, domain_name, description=desc)
                seen_domain[key] = did
            return seen_domain[key]

        # (chunks, namespace_id, domain_id)
        source_chunks: list[tuple[list, str | None, str | None]] = []

        for src in cfg.get("sources", []):
            stype = src.get("type")
            fn = _INGEST_FNS.get(stype)
            if fn is None:
                raise ValueError(f"Unknown source type: {stype!r}")
            name = src.get("name", stype)
            namespace_id = src.get("namespace") or None
            domain_name = src.get("domain") or name
            domain_id: str | None = None

            if namespace_id:
                if namespace_id not in ns_domains:
                    store.register_namespace(namespace_id)
                    ns_domains[namespace_id] = {}
                domain_id = _get_domain_id(namespace_id, domain_name)

            enrich_override = src.get("enrich_context")
            loader = _make_loader({"loader": loader_cfg}, enrich_override=enrich_override) if enrich_override is not None else default_loader
            label = f"{name!r}" + (f" → {namespace_id}/{domain_name}" if namespace_id else "")
            print(f"Ingesting {label}...")
            chunks = fn(loader, src)
            source_chunks.append((chunks, namespace_id, domain_id))
            print(f"  {len(chunks):,} chunks")

        all_chunks = [c for chunks, _, _ in source_chunks for c in chunks]
        print(f"Total: {len(all_chunks):,} chunks — embedding...")
        emb = _embed_chunks(all_chunks, {"embed": embed_cfg})

        offset = 0
        for chunks, namespace_id, domain_id in source_chunks:
            n = len(chunks)
            store.add_document(chunks, emb[offset: offset + n], namespace=namespace_id, domain_id=domain_id)
            offset += n

        print("Building FTS index...")
        store.vector.rebuild_fts_index()
        print(f"Ingest complete: {store.count():,} chunks → {db_path}")
    else:
        print(f"Existing store: {store.count():,} chunks at {db_path}")

    # ── Phase: NER ───────────────────────────────────────────────────────────
    if run_ner:
        existing = store.vector._conn.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()[0]
        if existing == 0 or force:
            print("Building NER index...")
            from .ner import build_ner
            build_ner(store, spacy_model=spacy_model)
            print("  NER done.")
        else:
            print(f"NER index: {existing:,} associations (skipped)")

    # ── Phase: community ─────────────────────────────────────────────────────
    if run_community:
        existing = store.vector._conn.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()[0]
        if existing == 0 or force:
            print("Building community index...")
            from .community import build_community
            n = build_community(
                db_path,
                embed_model_name,
                alpha=community_alpha,
                sim_threshold=community_sim_threshold,
                force=force,
            )
            print(f"  {n} communities.")
        else:
            print(f"Community index: {existing:,} assignments (skipped)")

    # ── Phase: SVO ───────────────────────────────────────────────────────────
    if run_svo:
        existing = store.vector._conn.execute("SELECT COUNT(*) FROM svo_triples").fetchone()[0]
        if existing == 0 or force:
            print(f"Building SVO graph via {svo_model!r} (calls LLM API)...")
            from .graph import EntityGraphPipeline, SVOExtractor

            llm_client = ic.get("svo_llm_client")
            if llm_client is None:
                try:
                    from openai import OpenAI as _OpenAI

                    class _OpenAILLMClient:
                        def __init__(self, model: str) -> None:
                            self._client = _OpenAI()
                            self._model = model

                        def complete(self, prompt: str) -> str:
                            resp = self._client.chat.completions.create(
                                model=self._model,
                                messages=[{"role": "user", "content": prompt}],
                            )
                            return resp.choices[0].message.content or ""

                    llm_client = _OpenAILLMClient(svo_model)
                except ImportError:
                    raise RuntimeError(
                        "index.svo=true requires openai installed or an LLMClient "
                        "passed as index.svo_llm_client in the config dict."
                    )

            extractor = SVOExtractor(llm_client)
            pipeline = EntityGraphPipeline(extractor)
            stats = pipeline.build(store, force=force)
            print(f"  {stats.triples_written} triples.")
        else:
            print(f"SVO index: {existing:,} triples (skipped)")

    return Index(store, embed_model_name, search_defaults, ic, loader_cfg, embed_cfg)


# ---------------------------------------------------------------------------
# Horizontal scale: worker / coordinator
# ---------------------------------------------------------------------------

def _pg_connect(dsn: str):
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("psycopg2 required: pip install chonk[pgvector]") from exc
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def _process_queue_job(
    backend,
    source_uri: str,
    namespace: str,
    embed_model: str,
    batch_size: int,
    run_ner: bool,
    spacy_model: str,
) -> None:
    """Load source_uri, chunk, embed, optionally NER, write to backend."""
    import logging
    _log = logging.getLogger(__name__)

    loader = DocumentLoader()
    try:
        chunks = loader.load(source_uri)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {source_uri!r}: {exc}") from exc

    if not chunks:
        _log.warning("No chunks produced from %s", source_uri)
        return

    content_hash = hashlib.sha256(
        "\n".join(c.content for c in chunks).encode()
    ).hexdigest()[:32]

    emb = _embed_texts([c.content for c in chunks], embed_model, batch_size)
    backend.add_chunks(chunks, emb, namespace=namespace)
    backend.register_document(
        chunks[0].document_name,
        content_hash,
        source_uri=source_uri,
        chunk_count=len(chunks),
    )

    if run_ner:
        from .ner import build_ner_chunks
        build_ner_chunks(backend, chunks, spacy_model=spacy_model)


def run_worker(
    queue_dsn: str,
    backend_dsn: str,
    *,
    embed_model: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 256,
    run_ner: bool = False,
    spacy_model: str = "en_core_web_sm",
    idle_sleep: float = 2.0,
) -> None:
    """Pull items from ``ingest_queue`` and process them.

    Runs until interrupted. Workers check the ``control`` table's
    ``workers_paused`` flag before each item — coordinator sets this
    during graph builds to drain in-flight workers cleanly.

    Args:
        queue_dsn: PostgreSQL DSN for the queue/control tables.
        backend_dsn: PostgreSQL DSN for the vector store.
        embed_model: SentenceTransformer model name for embedding.
        batch_size: Embedding batch size.
        run_ner: Whether to run NER on each document.
        spacy_model: spaCy model for NER.
        idle_sleep: Seconds to sleep when the queue is empty.
    """
    import logging
    import socket
    import time
    import uuid

    log = logging.getLogger(__name__)
    worker_id = f"{socket.gethostname()}:{uuid.uuid4().hex[:8]}"
    log.info("Worker %s starting", worker_id)

    from .storage import PgVectorBackend
    backend = PgVectorBackend(backend_dsn)

    while True:
        # Check pause flag
        with _pg_connect(queue_dsn) as qconn:
            with qconn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM control WHERE key = 'workers_paused'"
                )
                row = cur.fetchone()
            qconn.commit()
        if row and row[0] == "1":
            time.sleep(idle_sleep)
            continue

        # Claim next pending item (SKIP LOCKED prevents double-assignment)
        with _pg_connect(queue_dsn) as qconn:
            with qconn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingest_queue
                    SET status = 'processing', worker_id = %s, leased_at = now()
                    WHERE id = (
                        SELECT id FROM ingest_queue
                        WHERE status = 'pending'
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, source_uri, namespace
                    """,
                    [worker_id],
                )
                job = cur.fetchone()
            qconn.commit()

        if job is None:
            time.sleep(idle_sleep)
            continue

        job_id, source_uri, namespace = job
        log.info("Worker %s processing job %s: %s", worker_id, job_id, source_uri)

        try:
            _process_queue_job(
                backend, source_uri, namespace,
                embed_model=embed_model,
                batch_size=batch_size,
                run_ner=run_ner,
                spacy_model=spacy_model,
            )
            with _pg_connect(queue_dsn) as qconn:
                with qconn.cursor() as cur:
                    cur.execute(
                        "UPDATE ingest_queue SET status = 'done' WHERE id = %s",
                        [job_id],
                    )
                qconn.commit()
            log.info("Job %s done", job_id)
        except Exception as exc:
            log.error("Job %s failed: %s", job_id, exc, exc_info=True)
            with _pg_connect(queue_dsn) as qconn:
                with qconn.cursor() as cur:
                    cur.execute(
                        "UPDATE ingest_queue SET status = 'failed' WHERE id = %s",
                        [job_id],
                    )
                qconn.commit()


def run_coordinator(
    queue_dsn: str,
    backend_dsn: str,
    *,
    graph_interval: int = 300,
    embed_model: str = "BAAI/bge-large-en-v1.5",
    spacy_model: str = "en_core_web_sm",
    community_alpha: float = 0.2,
    community_sim_threshold: float = 0.6,
    lease_timeout_minutes: int = 10,
    poll_sleep: float = 5.0,
) -> None:
    """Coordinator: graph build, stale-lease requeue, and pause/resume signal.

    State machine::

        DISPATCHING → (interval elapsed or queue drained)
            → DRAINING   (stop dispatching, wait for in-flight workers)
            → BUILDING   (pause workers, run graph build, unpause)
            → DISPATCHING

    Args:
        queue_dsn: PostgreSQL DSN for the queue/control tables.
        backend_dsn: PostgreSQL DSN for the vector store.
        graph_interval: Seconds between graph builds (default 300).
        embed_model: SentenceTransformer model used for community embeddings.
        spacy_model: spaCy model used by NER (for community build context).
        community_alpha: Alpha for community detection weighting.
        community_sim_threshold: Minimum cosine sim for community edges.
        lease_timeout_minutes: Minutes before a 'processing' job is re-queued.
        poll_sleep: Seconds between coordinator loop ticks.
    """
    import logging
    import time

    log = logging.getLogger(__name__)
    log.info("Coordinator starting (graph_interval=%ds)", graph_interval)

    state = "DISPATCHING"
    last_graph_build = 0.0

    while True:
        now = time.time()

        if state == "DISPATCHING":
            _requeue_stale_leases(queue_dsn, lease_timeout_minutes)

            queue_empty = _queue_pending_count(queue_dsn) == 0
            interval_elapsed = (now - last_graph_build) >= graph_interval
            if queue_empty or interval_elapsed:
                log.info(
                    "Transitioning DISPATCHING → DRAINING "
                    "(queue_empty=%s, interval_elapsed=%s)",
                    queue_empty,
                    interval_elapsed,
                )
                state = "DRAINING"

        elif state == "DRAINING":
            if _queue_processing_count(queue_dsn) == 0:
                log.info("All workers drained. Transitioning → BUILDING")
                state = "BUILDING"

        elif state == "BUILDING":
            _set_control(queue_dsn, "workers_paused", "1")
            log.info("Workers paused. Starting graph build.")
            try:
                _run_graph_build(
                    backend_dsn,
                    embed_model=embed_model,
                    alpha=community_alpha,
                    sim_threshold=community_sim_threshold,
                )
                log.info("Graph build complete.")
            except Exception as exc:
                log.error("Graph build failed: %s", exc, exc_info=True)
            finally:
                _set_control(queue_dsn, "workers_paused", "0")
                log.info("Workers unpaused.")
            last_graph_build = time.time()
            state = "DISPATCHING"

        time.sleep(poll_sleep)


# ---------------------------------------------------------------------------
# Coordinator helpers
# ---------------------------------------------------------------------------

def _set_control(dsn: str, key: str, value: str) -> None:
    with _pg_connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO control (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                [key, value],
            )
        conn.commit()


def _queue_pending_count(dsn: str) -> int:
    with _pg_connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ingest_queue WHERE status = 'pending'")
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else 0


def _queue_processing_count(dsn: str) -> int:
    with _pg_connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ingest_queue WHERE status = 'processing'")
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else 0


def _requeue_stale_leases(dsn: str, timeout_minutes: int = 10) -> None:
    """Reset 'processing' jobs whose lease has expired back to 'pending'."""
    with _pg_connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingest_queue
                SET status = 'pending', worker_id = NULL, leased_at = NULL
                WHERE status = 'processing'
                  AND leased_at < now() - interval '%s minutes'
                """,
                [timeout_minutes],
            )
            count = cur.rowcount
        conn.commit()
    if count:
        import logging
        logging.getLogger(__name__).warning("Requeued %d stale lease(s)", count)


def _run_graph_build(
    backend_dsn: str,
    *,
    embed_model: str,
    alpha: float,
    sim_threshold: float,
) -> None:
    """Fetch all namespaces from PG and run community detection per namespace.

    Reads chunks from the PG backend via get_all_chunks(), builds a temporary
    in-memory DuckDB store for graph computation, then writes results back.
    """
    import logging
    import tempfile

    import numpy as np

    log = logging.getLogger(__name__)

    from .storage import PgVectorBackend, Store

    pg = PgVectorBackend(backend_dsn)

    # Enumerate namespaces with chunks
    namespaces_row = pg._conn.execute(
        "SELECT DISTINCT namespace FROM embeddings WHERE namespace IS NOT NULL"
    ).fetchall()
    namespaces = [r[0] for r in namespaces_row]

    if not namespaces:
        log.info("No namespaces found; skipping graph build.")
        pg.close()
        return

    for ns in namespaces:
        log.info("Building community graph for namespace %s", ns)
        # Pull chunks for this namespace into a temp DuckDB for graph building
        chunks_rows = pg._conn.execute(
            "SELECT chunk_id, document_name, content, section, chunk_index, "
            "       breadcrumb, chunk_type, source_offset, source_length, "
            "       source_detail, source_id, domain_id, embedding "
            "FROM embeddings WHERE namespace = %s",
            [ns],
        ).fetchall()

        if not chunks_rows:
            continue

        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
            tmp_path = f.name

        with Store(tmp_path, embedding_dim=pg._embedding_dim) as tmp_store:
            from .models import DocumentChunk

            tmp_chunks = []
            tmp_embeddings = []
            for row in chunks_rows:
                (
                    _chunk_id, doc_name, content, section_str, chunk_idx,
                    breadcrumb, chunk_type, src_off, src_len,
                    src_det_str, src_id, dom_id, embedding_vec,
                ) = row
                import json as _json
                sec = _json.loads(section_str) if section_str else []
                src_det = _json.loads(src_det_str) if src_det_str else None
                tmp_chunks.append(DocumentChunk(
                    document_name=doc_name,
                    content=content,
                    section=sec,
                    chunk_index=chunk_idx,
                    breadcrumb=breadcrumb,
                    chunk_type=chunk_type or "document",
                    source_offset=src_off,
                    source_length=src_len,
                    source_detail=src_det,
                ))
                tmp_embeddings.append(np.array(embedding_vec, dtype="float32"))

            emb_array = np.stack(tmp_embeddings)
            tmp_store.add_document(tmp_chunks, emb_array, namespace=ns)

            from .community import build_community
            n = build_community(
                tmp_path,
                embed_model,
                alpha=alpha,
                sim_threshold=sim_threshold,
                force=True,
            )
            log.info("Namespace %s: %d communities built", ns, n)

        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    pg.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="chonk ingest — worker / coordinator modes for horizontal scale",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--worker",
        action="store_true",
        help="Run as a worker: pull jobs from queue and process them.",
    )
    mode.add_argument(
        "--coordinator",
        action="store_true",
        help="Run as coordinator: dispatch leases, build graphs, pause/resume.",
    )
    parser.add_argument(
        "--queue",
        required=True,
        metavar="DSN",
        help="PostgreSQL DSN for the ingest_queue / control tables.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        metavar="DSN",
        help="PostgreSQL DSN for the vector store.",
    )
    parser.add_argument(
        "--graph-interval",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Coordinator: seconds between graph builds (default 300).",
    )
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-large-en-v1.5",
        metavar="MODEL",
        help="SentenceTransformer model name for embedding (default BAAI/bge-large-en-v1.5).",
    )
    parser.add_argument(
        "--ner",
        action="store_true",
        help="Worker: run NER on each ingested document.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        metavar="MODEL",
        help="spaCy model for NER (default en_core_web_sm).",
    )
    args = parser.parse_args()

    if args.worker:
        run_worker(
            args.queue,
            args.backend,
            embed_model=args.embed_model,
            run_ner=args.ner,
            spacy_model=args.spacy_model,
        )
    else:
        run_coordinator(
            args.queue,
            args.backend,
            graph_interval=args.graph_interval,
            embed_model=args.embed_model,
            spacy_model=args.spacy_model,
        )
        sys.exit(0)
