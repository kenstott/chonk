# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""EntityGraphPipeline — on-demand entity graph build with progress feedback."""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._extractor import SVOExtractor
from ._index import RelationshipIndex


@dataclass
class EntityGraphStats:
    """Summary returned by :meth:`EntityGraphPipeline.build`."""

    chunks_processed: int = 0
    chunks_skipped: int = 0
    triples_written: int = 0
    descriptions_written: int = 0
    aliases_written: int = 0
    entity_embeddings_written: int = 0


# Phase name constants used in progress callbacks
PHASE_LOAD = "load"
PHASE_EXTRACT = "extract"
PHASE_PERSIST_TRIPLES = "persist_triples"
PHASE_PERSIST_DESCRIPTIONS = "persist_descriptions"
PHASE_PERSIST_ALIASES = "persist_aliases"
PHASE_EMBED_ENTITIES = "embed_entities"

ProgressFn = Callable[[str, int, int], None]
"""``(phase, done, total) -> None``"""


@runtime_checkable
class EmbedModel(Protocol):
    def encode(self, texts: list[str], **kwargs) -> object: ...


class EntityGraphPipeline:
    """Build entity descriptions, aliases, SVO triples, and entity embeddings on demand.

    Args:
        extractor: Configured :class:`SVOExtractor` for triple + description + alias extraction.
        embed_model: Optional sentence-transformer-compatible model for entity embeddings.
            If ``None``, the entity embedding step is skipped.
        concurrency: Thread-pool size for parallel LLM extraction calls.
        namespace: Namespace written to ``entity_descriptions`` and ``entity_aliases``.
            Defaults to ``"global"``.
        embed_dim: Embedding dimension. Must match the store's dimension.

    Usage::

        from chonk.graph import EntityGraphPipeline, SVOExtractor

        extractor = SVOExtractor(my_llm)
        pipeline = EntityGraphPipeline(extractor, embed_model=st_model)

        def on_progress(phase, done, total):
            print(f"{phase}: {done}/{total}")

        with Store("index.duckdb") as store:
            stats = pipeline.build(store, progress=on_progress)

        print(stats)
    """

    def __init__(
        self,
        extractor: SVOExtractor,
        *,
        embed_model: EmbedModel | None = None,
        concurrency: int = 4,
        namespace: str = "global",
        embed_dim: int = 1024,
    ) -> None:
        self._extractor = extractor
        self._embed_model = embed_model
        self._concurrency = concurrency
        self._namespace = namespace
        self._embed_dim = embed_dim

    def build(
        self,
        store,
        *,
        progress: ProgressFn | None = None,
        force: bool = False,
    ) -> EntityGraphStats:
        """Run the full entity graph build pipeline against *store*.

        Args:
            store: A writable :class:`~chonk.storage.Store` instance.
            progress: Optional callback ``(phase, done, total) -> None`` fired at each step.
            force: If ``False`` (default), skip triple extraction when ``svo_triples``
                already has rows. Set ``True`` to rebuild unconditionally.

        Returns:
            :class:`EntityGraphStats` with counts for each write phase.
        """
        def _prog(phase: str, done: int, total: int) -> None:
            if progress is not None:
                progress(phase, done, total)

        stats = EntityGraphStats()

        # ── Phase 1: load ─────────────────────────────────────────────
        _prog(PHASE_LOAD, 0, 1)
        conn = store._db.conn

        if not force:
            try:
                n = conn.execute("SELECT COUNT(*) FROM svo_triples").fetchone()[0]
                if n > 0:
                    _prog(PHASE_LOAD, 1, 1)
                    return stats
            except Exception:
                pass

        rows = conn.execute(
            "SELECT chunk_id, content FROM embeddings"
        ).fetchall()

        try:
            chunk_entity_rows = conn.execute("""
                SELECT ce.chunk_id, ce.entity_id,
                       COALESCE(e.entity_type, 'concept') AS entity_type
                FROM chunk_entities ce
                LEFT JOIN entities e ON e.id = ce.entity_id
            """).fetchall()
        except Exception:
            chunk_entity_rows = []

        chunk_entities_map: dict[str, list[dict]] = defaultdict(list)
        for chunk_id, entity_id, entity_type in chunk_entity_rows:
            chunk_entities_map[chunk_id].append({
                "id": entity_id,
                "type": entity_type,
                "description": "",
            })

        eligible = [(cid, content) for cid, content in rows
                    if len(chunk_entities_map.get(cid, [])) >= 2]
        stats.chunks_skipped = len(rows) - len(eligible)

        _prog(PHASE_LOAD, 1, 1)

        # ── Phase 2: extract ──────────────────────────────────────────
        relationship_index = RelationshipIndex()
        new_descriptions: dict[str, str] = {}
        new_aliases: dict[str, list[str]] = defaultdict(list)

        def _extract_one(row: tuple) -> tuple:
            chunk_id, content = row
            entities = chunk_entities_map.get(chunk_id, [])
            if len(entities) >= 2:
                triples, descs, aliases, _rel = self._extractor.extract_entity_anchored(
                    content or "", chunk_id, entities
                )
                return triples, descs, aliases
            return [], {}, {}

        total = len(eligible)
        done = 0
        _prog(PHASE_EXTRACT, 0, total)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            futures = {pool.submit(_extract_one, row): row for row in eligible}
            for fut in concurrent.futures.as_completed(futures):
                triples, descs, aliases = fut.result()
                for t in triples:
                    relationship_index.add(t)
                new_descriptions.update(descs)
                for eid, alias_list in aliases.items():
                    new_aliases[eid].extend(alias_list)
                done += 1
                stats.chunks_processed += 1
                _prog(PHASE_EXTRACT, done, total)

        # ── Phase 3: persist triples ──────────────────────────────────
        _prog(PHASE_PERSIST_TRIPLES, 0, 1)
        stats.triples_written = relationship_index.save_to_db(conn)
        _prog(PHASE_PERSIST_TRIPLES, 1, 1)

        # ── Phase 4: persist descriptions ─────────────────────────────
        if new_descriptions:
            _prog(PHASE_PERSIST_DESCRIPTIONS, 0, len(new_descriptions))
            stats.descriptions_written = store.set_entity_descriptions_batch(
                new_descriptions
            )
            _prog(PHASE_PERSIST_DESCRIPTIONS, stats.descriptions_written,
                  len(new_descriptions))

        # ── Phase 5: persist aliases ──────────────────────────────────
        if new_aliases:
            flat = {alias: eid for eid, alias_list in new_aliases.items()
                    for alias in alias_list}
            _prog(PHASE_PERSIST_ALIASES, 0, len(flat))
            stats.aliases_written = store.add_entity_aliases_batch(
                flat, source="llm", namespace=self._namespace
            )
            _prog(PHASE_PERSIST_ALIASES, stats.aliases_written, len(flat))

        # ── Phase 6: embed entities ───────────────────────────────────
        if self._embed_model is not None:
            stats.entity_embeddings_written = self._embed_entities(store, _prog)

        return stats

    # ------------------------------------------------------------------

    def _embed_entities(self, store, prog: ProgressFn) -> int:
        import numpy as np

        from ..models import DocumentChunk

        conn = store._db.conn
        entity_rows = conn.execute("""
            SELECT e.id, e.name,
                   COALESCE(e.description, '') AS description,
                   e.entity_type
            FROM entities e
        """).fetchall()

        if not entity_rows:
            return 0

        try:
            alias_rows = conn.execute(
                "SELECT entity_id, alias FROM entity_aliases WHERE namespace = ?",
                [self._namespace],
            ).fetchall()
        except Exception:
            alias_rows = []

        aliases_map: dict[str, list[str]] = defaultdict(list)
        for eid, alias in alias_rows:
            aliases_map[eid].append(alias)

        texts: list[str] = []
        chunks: list[DocumentChunk] = []
        for entity_id, name, description, entity_type in entity_rows:
            alias_str = ", ".join(aliases_map.get(entity_id, []))
            parts = [name]
            if alias_str:
                parts.append(alias_str)
            if description:
                parts.append(description)
            text = ". ".join(parts)
            texts.append(text)
            chunks.append(DocumentChunk(
                document_name=f"__entity__{entity_id}",
                content=text,
                chunk_index=0,
                chunk_type="entity",
            ))

        prog(PHASE_EMBED_ENTITIES, 0, len(chunks))
        vecs = self._embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=512
        ).astype("float32")

        db.execute("DELETE FROM embeddings WHERE chunk_type = 'entity'")
        store.add_document(chunks, np.array(vecs))
        prog(PHASE_EMBED_ENTITIES, len(chunks), len(chunks))
        return len(chunks)
