# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""build_ner — run NER on a Store and persist results to chunk_entities."""

from __future__ import annotations

import hashlib
import json

from ._index import EntityIndex
from ._spacy import SpacyMatcher
from ._spacy_labels import SpacyLabel

_SPACY_MODEL = "en_core_web_sm"
_NUMERIC_TYPES = {
    SpacyLabel.ORDINAL, SpacyLabel.MONEY,
    SpacyLabel.PERCENT, SpacyLabel.QUANTITY,
}
_ID_SUFFIXES = ("_identifier", "_reference", "_number", "_num", "_code", "_key", "_id", "_ref", "_no")


def _strip_id_alias(entity_id: str) -> str | None:
    """Return the suffix-stripped alias for an entity ID, or None if no suffix matches."""
    for suffix in _ID_SUFFIXES:
        if entity_id.endswith(suffix) and len(entity_id) > len(suffix):
            return entity_id[: -len(suffix)]
    return None


_NER_CACHE_DDL = (
    "CREATE TABLE IF NOT EXISTS ner_cache "
    "(config_fingerprint VARCHAR PRIMARY KEY, chunk_count INTEGER NOT NULL, "
    "created_at TIMESTAMP DEFAULT current_timestamp)"
)


def _ner_config_fingerprint(
    spacy_model: str,
    use_schema_vocab: bool,
    vocab_entities: list[dict] | None,
) -> str:
    payload = json.dumps({
        "spacy_model": spacy_model,
        "use_schema_vocab": use_schema_vocab,
        "vocab_entities": sorted(
            [json.dumps(e, sort_keys=True) for e in (vocab_entities or [])]
        ),
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def build_ner(
    store,
    *,
    spacy_model: str = _SPACY_MODEL,
    use_schema_vocab: bool = False,
    vocab_entities: list[dict] | None = None,
    force: bool = False,
    build_context_graph: bool = False,
    namespace: str = "global",
) -> int:
    """Run NER on all chunks in *store* and persist results to ``chunk_entities``.

    Incremental by default: skips chunks already processed under the same config
    fingerprint.  Pass ``force=True`` to rebuild from scratch regardless.

    Args:
        store: A ``Store`` instance (open context manager).
        spacy_model: spaCy model name.
        use_schema_vocab: Build schema vocabulary from db_schema/api chunks.
        vocab_entities: Extra entity vocab entries — each dict has keys
            ``type`` ("static" or "db_query"), ``entity_type``, and either
            ``names`` (static) or ``connection`` + ``sql`` (db_query).
        force: If True, ignore cache and rebuild all chunks.

    Returns:
        Number of ``chunk_entities`` associations written.
    """
    from chonk.storage._vector import DuckDBVectorBackend

    con = store.vector._conn
    con.execute(_NER_CACHE_DDL)

    fingerprint = _ner_config_fingerprint(spacy_model, use_schema_vocab, vocab_entities)

    all_chunk_ids: set[str] = {
        row[0] for row in con.execute("SELECT chunk_id FROM embeddings").fetchall()
    }
    processed_ids: set[str] = {
        row[0] for row in con.execute("SELECT DISTINCT chunk_id FROM chunk_entities").fetchall()
    }
    cached = con.execute(
        "SELECT chunk_count FROM ner_cache WHERE config_fingerprint = ?", [fingerprint]
    ).fetchone()

    config_match = cached is not None
    new_chunk_ids = all_chunk_ids - processed_ids

    if force:
        skip_ids: set[str] | None = None
        incremental = False
    elif config_match and not new_chunk_ids:
        return con.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()[0]
    elif config_match and new_chunk_ids:
        skip_ids = processed_ids
        incremental = True
    else:
        skip_ids = None
        incremental = False

    label_types = [t for t in SpacyLabel if t not in _NUMERIC_TYPES]
    matcher = SpacyMatcher(model=spacy_model, strip_numeric=True, entity_types=label_types)
    entity_index = EntityIndex()

    all_chunks = store.vector.get_all_chunks()

    schema_matcher = None
    data_matcher = None
    if use_schema_vocab or vocab_entities:
        from ._schema_vocab import SchemaVocabBuilder
        builder = SchemaVocabBuilder()
        if use_schema_vocab:
            builder.add_chunks(all_chunks)
        for entry in (vocab_entities or []):
            etype = entry.get("entity_type", "term")
            if entry.get("type") == "static":
                builder.add_entities(entry.get("names", []), entity_type=etype)
            elif entry.get("type") == "db_query":
                builder.add_from_db(entry["connection"], {etype: entry["sql"]})
        schema_matcher = builder.build()
        data_matcher = builder.build_data_matcher() if builder.data_term_count() > 0 else None

    skip = skip_ids or set()
    chunks_to_process = []
    for c in all_chunks:
        embed_content = c.embedding_content if c.embedding_content else c.content
        cid = DuckDBVectorBackend._generate_chunk_id(c.document_name, c.chunk_index, embed_content)
        if cid not in skip:
            chunks_to_process.append((cid, c))

    # entity_id -> (name, display_name, entity_type) — populated from matches
    entity_meta: dict[str, tuple[str, str, str]] = {}

    from ._merge import merge_matches

    for chunk_id, chunk in chunks_to_process:
        ner_text = chunk.content

        if schema_matcher is not None or data_matcher is not None:
            vocab_hits: list = []
            if schema_matcher is not None:
                vocab_hits = merge_matches(
                    schema_matcher.match(chunk.content), vocab_hits,
                    source_text=chunk.content,
                )
            if data_matcher is not None:
                vocab_hits = merge_matches(
                    data_matcher.match(chunk.content), vocab_hits,
                    source_text=chunk.content,
                )
            combined = merge_matches(vocab_hits, matcher.match(ner_text), source_text=ner_text)
            for m in combined:
                if m.entity_id not in entity_meta:
                    entity_meta[m.entity_id] = (m.name, m.display_name, m.entity_type or "concept")
            entity_index.index_chunk(chunk_id, chunk.content, combined)
        else:
            matches = matcher.match(ner_text)
            for m in matches:
                if m.entity_id not in entity_meta:
                    entity_meta[m.entity_id] = (m.name, m.display_name, m.entity_type or "concept")
            entity_index.index_chunk(chunk_id, chunk.content, matches)

    entity_index.recompute_scores()

    data = entity_index.to_dict()
    if not incremental:
        con.execute("DELETE FROM chunk_entities")
        con.execute("DELETE FROM entities")
    for a in data["associations"]:
        ns_row = con.execute(
            "SELECT namespace FROM embeddings WHERE chunk_id = ?", [a["chunk_id"]]
        ).fetchone()
        chunk_namespace = ns_row[0] if ns_row else None
        con.execute(
            "INSERT OR REPLACE INTO chunk_entities"
            "(chunk_id, entity_id, frequency, positions_json, score, namespace)"
            " VALUES (?,?,?,?,?,?)",
            [a["chunk_id"], a["entity_id"], a["frequency"],
             json.dumps(a["positions"]), a["score"], chunk_namespace],
        )
        name, display_name, entity_type = entity_meta.get(
            a["entity_id"], (a["entity_id"], a["entity_id"], "concept")
        )
        con.execute(
            "INSERT OR IGNORE INTO entities(id, name, display_name, entity_type) VALUES (?,?,?,?)",
            [a["entity_id"], name, display_name, entity_type],
        )
        alias = _strip_id_alias(a["entity_id"])
        if alias:
            con.execute(
                "INSERT OR IGNORE INTO entity_aliases(alias, entity_id, source) VALUES (?,?,?)",
                [alias, a["entity_id"], "strip_suffix"],
            )

    con.execute(
        "INSERT OR REPLACE INTO ner_cache(config_fingerprint, chunk_count) VALUES (?, ?)",
        [fingerprint, len(all_chunk_ids)],
    )

    if build_context_graph:
        from ..graph._context_graph import build_context_graph_edges
        build_context_graph_edges(con, namespace=namespace, force=True)

    return len(data["associations"])
