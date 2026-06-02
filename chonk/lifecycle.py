# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 43e91c0f-6376-43c8-ac24-6085a1badbe0
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Namespace lifecycle: async build pipeline and background freshness refresh."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .community import build_community
from .indexer import IndexHandle
from .storage._store import Store


def build_namespace_async(
    namespace_id: str,
    db_path: str | Path,
    embed_model: str | Any,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_complete: Callable[[int], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
    force: bool = False,
    run_ner: bool = True,
    run_community: bool = True,
    spacy_model: str = "en_core_web_sm",
    community_alpha: float = 0.2,
    community_sim_threshold: float = 0.6,
    embed_batch_size: int = 256,
) -> IndexHandle:
    """Build the full index pipeline for *namespace_id* in a background thread.

    Phases: crawl → chunk → embed → NER → community → FTS.
    Returns an IndexHandle; call .join() to wait for completion.

    Idempotent when force=False: returns immediately if namespace_cache_valid().
    Safe to call concurrently for different namespaces (separate DB files).

    Args:
        namespace_id: Namespace to build.
        db_path: Path to this namespace's DuckDB file.
        embed_model: SentenceTransformer model name or instance.
        on_progress: Called with (phase, done, total).
        on_complete: Called with total chunk count on success.
        on_error: Called with (phase, exception) on failure.
        force: Rebuild even if namespace_cache_valid() is True.
        run_ner: Run NER after chunking.
        run_community: Build community index after NER.
        spacy_model: spaCy model for NER.
        community_alpha: Breadcrumb weight for community graph.
        community_sim_threshold: Cosine similarity threshold for community edges.
        embed_batch_size: Embedding batch size.
    """
    _on_progress = on_progress or (lambda *_: None)
    _on_complete = on_complete or (lambda *_: None)
    _on_error = on_error or (lambda *_: None)

    def _run() -> None:
        db_path_ = Path(db_path)
        store = Store(db_path_, read_only=False)

        try:
            if not force and store.namespace_cache_valid(namespace_id):
                _on_complete(
                    store.vector._conn.execute(
                        "SELECT COUNT(*) FROM embeddings WHERE namespace = ?",
                        [namespace_id],
                    ).fetchone()[0]
                )
                return

            # ── Phase: crawl / chunk / embed ──────────────────────────────────
            sources = store.vector._conn.execute(
                """
                SELECT s.source_id, s.type, s.uri, s.config, d.domain_id
                FROM sources s
                JOIN domains d ON s.domain_id = d.domain_id
                WHERE d.namespace_id = ?
                """,
                [namespace_id],
            ).fetchall()

            import json as _json

            from .indexer import Indexer

            indexer = Indexer(
                store,
                embed_model,
                on_progress=_on_progress,
                on_error=lambda phase, exc: _on_error(phase, exc),
                embed_batch_size=embed_batch_size,
            )
            total_chunks = 0
            for source_id, stype, uri, config_json, domain_id in sources:
                config = _json.loads(config_json) if config_json else {}
                source_config = {
                    "source_id": source_id,
                    "type": stype,
                    "uri": uri,
                    "domain_id": domain_id,
                    "namespace": namespace_id,
                    **config,
                }
                total_chunks += indexer.index_source(source_config)
                store.vector._conn.execute(
                    "UPDATE sources SET last_crawled = now() WHERE source_id = ?",
                    [source_id],
                ).fetchall()

            store._mark_namespace_built(namespace_id, "chunks")
            _on_progress("embed", total_chunks, total_chunks)

            # ── Phase: NER ────────────────────────────────────────────────────
            if run_ner:
                try:
                    from .ner import build_ner

                    _on_progress("ner", 0, 1)
                    build_ner(store, spacy_model=spacy_model, namespace=namespace_id)
                    store._mark_namespace_built(namespace_id, "ner")
                    _on_progress("ner", 1, 1)
                except Exception as exc:
                    _on_error("ner", exc)

            # ── Phase: community ──────────────────────────────────────────────
            if run_community:
                try:
                    _on_progress("community", 0, 1)
                    build_community(
                        db_path_,
                        embed_model,
                        namespace_id=namespace_id,
                        alpha=community_alpha,
                        sim_threshold=community_sim_threshold,
                        force=force,
                    )
                    store._mark_namespace_built(namespace_id, "community")
                    _on_progress("community", 1, 1)
                except Exception as exc:
                    _on_error("community", exc)

            # ── Phase: FTS ────────────────────────────────────────────────────
            try:
                _on_progress("fts", 0, 1)
                store.vector.rebuild_fts_index()
                _on_progress("fts", 1, 1)
            except Exception as exc:
                _on_error("fts", exc)

            _on_complete(total_chunks)

        finally:
            store.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return IndexHandle(t)


class NamespaceRefresher:
    """Periodic background job that re-indexes stale namespaces.

    Each interval, checks all registered namespaces via namespace_cache_valid().
    Stale namespaces are queued for rebuild via build_namespace_async().
    Concurrent builds across different namespaces run in parallel (separate DBs).

    Args:
        db_path_fn: Callable mapping namespace_id to its DB file path.
        embed_model: SentenceTransformer model name or instance.
        interval_seconds: How often to check all namespaces (default 3600).
        on_rebuild: Called with namespace_id when a rebuild is triggered.
        build_kwargs: Extra kwargs forwarded to build_namespace_async.
    """

    def __init__(
        self,
        db_path_fn: Callable[[str], str | Path],
        embed_model: str | Any,
        interval_seconds: int = 3600,
        on_rebuild: Callable[[str], None] | None = None,
        **build_kwargs: Any,
    ) -> None:
        self._db_path_fn = db_path_fn
        self._embed_model = embed_model
        self._interval = interval_seconds
        self._on_rebuild = on_rebuild or (lambda _: None)
        self._build_kwargs = build_kwargs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active: dict[str, IndexHandle] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background refresh loop (non-blocking)."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the refresh loop and wait for the loop thread to exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._check_all()

    def _check_all(self) -> None:
        try:
            global_db_path = self._db_path_fn("global")
        except Exception:
            return

        try:
            import duckdb

            con = duckdb.connect(str(global_db_path), read_only=True)
            namespace_ids = [
                r[0] for r in con.execute("SELECT namespace_id FROM namespaces").fetchall()
            ]
            con.close()
        except Exception:
            return

        for ns_id in namespace_ids:
            with self._lock:
                handle = self._active.get(ns_id)
                if handle is not None and handle.running:
                    continue

            try:
                db_path = self._db_path_fn(ns_id)
                store = Store(db_path, read_only=True)
                valid = store.namespace_cache_valid(ns_id)
                store.close()
            except Exception:
                continue

            if not valid:
                self._on_rebuild(ns_id)
                handle = build_namespace_async(
                    ns_id,
                    self._db_path_fn(ns_id),
                    self._embed_model,
                    **self._build_kwargs,
                )
                with self._lock:
                    self._active[ns_id] = handle
