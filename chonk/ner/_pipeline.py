# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""NerPipeline — unified two-pass NER with options.

Handles the common scenario (schema vocab + spaCy) behind one object::

    from chonk.ner import NerPipeline

    pipeline = NerPipeline(db_enrich=True, spacy_entities=True)

    # Feed schema sources (any combination)
    pipeline.add_tables(table_meta_list)       # from DB introspection
    pipeline.add_sql(open("schema.sql").read()) # from DDL files
    pipeline.add_chunks(schema_chunks)          # from loader.load_schema()

    # Run on document chunks (single call, everything merged internally)
    for chunk_id, chunk in indexed_chunks:
        matches = pipeline.match(chunk.content)
        entity_index.index_chunk(chunk_id, chunk.content, matches)

    # Or: run on all chunks at once and index in one shot
    pipeline.run_on_chunks(chunks, entity_index, chunk_ids)

Options
-------
db_enrich       Use schema/column/API terms as a custom vocabulary matcher.
                All identifiers are normalized (camelCase / snake_case /
                SCREAMING_SNAKE → "first name" style) so they match prose.
spacy_entities  Run spaCy NER.  Where both matchers fire on the same span,
                the schema vocab wins (schema entities suppress overlapping
                spaCy hits via merge_matches).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._merge import merge_matches
from ._schema_vocab import SchemaVocabBuilder
from ._vocabulary import EntityMatch

if TYPE_CHECKING:
    from ._index import EntityIndex
    from ._schema import SchemaMatcher
    from ._spacy import SpacyMatcher
    from ._spacy_labels import SpacyLabel


class NerPipeline:
    """Unified NER pipeline: schema vocab + spaCy, merged.

    Args:
        db_enrich: Match schema/column/API terms against document text.
            Feed terms via ``add_tables()``, ``add_sql()``, ``add_chunks()``.
        spacy_entities: Run spaCy NER on document text.
        spacy_model: spaCy model name (default ``"en_core_web_sm"``).
        spacy_entity_types: Whitelist of spaCy label strings
            (e.g. ``["ORG", "PERSON", "GPE"]``). ``None`` keeps all labels.
        min_schema_term_length: Ignore schema identifiers shorter than this
            (default 2 — filters noise like ``id`` column abbreviations).
    """

    def __init__(
        self,
        db_enrich: bool = False,
        spacy_entities: bool = False,
        spacy_model: str = "en_core_web_sm",
        spacy_entity_types: list[str] | list[SpacyLabel] | None = None,
        min_schema_term_length: int = 2,
    ):
        self._db_enrich = db_enrich
        self._spacy_entities = spacy_entities
        self._spacy_model = spacy_model
        self._spacy_entity_types = spacy_entity_types
        self._builder = SchemaVocabBuilder(min_term_length=min_schema_term_length)
        self._schema_matcher: SchemaMatcher | None = None  # lazily built
        self._spacy_matcher: SpacyMatcher | None = None  # lazily built
        self._dirty = False  # True when new terms added after first match()

    # ------------------------------------------------------------------
    # Schema vocabulary population
    # ------------------------------------------------------------------

    def add_tables(self, tables: list) -> NerPipeline:
        """Add table/column names from a list of TableMeta objects."""
        self._builder.add_tables(tables)
        self._dirty = True
        return self

    def add_endpoints(self, endpoints: list) -> NerPipeline:
        """Add path/field names from a list of EndpointMeta objects."""
        self._builder.add_endpoints(endpoints)
        self._dirty = True
        return self

    def add_sql(self, ddl: str) -> NerPipeline:
        """Extract table and column names from raw SQL DDL text."""
        self._builder.add_sql(ddl)
        self._dirty = True
        return self

    def add_chunks(self, chunks: list) -> NerPipeline:
        """Extract terms from DocumentChunk objects from load_schema()/load_api()."""
        self._builder.add_chunks(chunks)
        self._dirty = True
        return self

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def match(self, text: str) -> list[EntityMatch]:
        """Run enabled matchers against *text* and return merged results.

        Schema vocab wins on any span overlap with spaCy (schema entities
        suppress overlapping spaCy hits; spaCy hits on non-overlapping spans
        are kept).

        Returns an empty list if neither ``db_enrich`` nor ``spacy_entities``
        is enabled.
        """
        schema_hits: list[EntityMatch] = []
        spacy_hits: list[EntityMatch] = []

        if self._db_enrich:
            schema_hits = self._get_schema_matcher().match(text)

        if self._spacy_entities:
            spacy_hits = self._get_spacy_matcher().match(text)

        if schema_hits and spacy_hits:
            return merge_matches(schema_hits, spacy_hits, source_text=text)
        return schema_hits or spacy_hits

    def run_on_chunks(
        self,
        chunks: list,
        entity_index: EntityIndex,
        chunk_ids: list[str] | None = None,
    ) -> dict[str, list[EntityMatch]]:
        """Run NER on every chunk and index results into *entity_index*.

        Args:
            chunks: List of DocumentChunk objects.
            entity_index: EntityIndex to record associations in.
            chunk_ids: Stable IDs parallel to *chunks*.  If ``None``,
                uses ``chunk.document_name + ":" + str(chunk.chunk_index)``.

        Returns:
            Mapping of ``chunk_id -> list[EntityMatch]`` for every chunk
            that produced at least one match.
        """
        results: dict[str, list[EntityMatch]] = {}
        for i, chunk in enumerate(chunks):
            if chunk_ids is not None:
                cid = chunk_ids[i]
            else:
                cid = f"{chunk.document_name}:{chunk.chunk_index}"
            content = chunk.content or ""
            matches = self.match(content)
            if matches:
                entity_index.index_chunk(cid, content, matches)
                results[cid] = matches
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def schema_term_counts(self) -> dict[str, int]:
        """Return counts of accumulated schema terms by category."""
        return {
            "tables": self._builder.table_count(),
            "columns": self._builder.column_count(),
            "api_terms": self._builder.api_term_count(),
        }

    # ------------------------------------------------------------------
    # Internal lazy initialisation
    # ------------------------------------------------------------------

    def _get_schema_matcher(self) -> SchemaMatcher:
        if self._schema_matcher is None or self._dirty:
            self._schema_matcher = self._builder.build()
            self._dirty = False
        return self._schema_matcher

    def _get_spacy_matcher(self) -> SpacyMatcher:
        if self._spacy_matcher is None:
            from ._spacy import SpacyMatcher as _SpacyMatcher

            self._spacy_matcher = _SpacyMatcher(
                model=self._spacy_model,
                entity_types=self._spacy_entity_types,
            )
        return self._spacy_matcher
