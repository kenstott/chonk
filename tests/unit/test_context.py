# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 75e44fdf-178e-4130-b329-f5e639aa0819
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for chonk.context — enrich_chunk and enrich_chunks."""

import pytest

from chonk.context import enrich_chunk, enrich_chunks
from chonk.models import DocumentChunk


def _make_chunk(section=None, content="Some chunk content.", doc_name="test_doc", breadcrumb=None):
    return DocumentChunk(
        document_name=doc_name,
        content=content,
        section=section,
        chunk_index=0,
        breadcrumb=breadcrumb,
    )


class TestEnrichChunk:
    def test_doc_and_section_both_present(self):
        chunk = _make_chunk(breadcrumb="[my_report > Methods > Table 1]")
        result = enrich_chunk(chunk)
        assert result.embedding_content == "[my_report > Methods > Table 1]\n\nSome chunk content."

    def test_breadcrumb_doc_only(self):
        chunk = _make_chunk(breadcrumb="[my_report]", content="Plain content.")
        result = enrich_chunk(chunk)
        assert result.embedding_content == "[my_report]\n\nPlain content."

    def test_no_breadcrumb_falls_back_to_fields(self):
        chunk = _make_chunk(section=["Results"], doc_name="my_report", content="Text.")
        result = enrich_chunk(chunk)
        assert result.embedding_content == "[my_report > Results]\n\nText."

    def test_no_breadcrumb_no_section_uses_doc_only(self):
        chunk = _make_chunk(section=None, doc_name="my_report", content="Text.")
        result = enrich_chunk(chunk)
        assert result.embedding_content == "[my_report]\n\nText."

    def test_no_breadcrumb_no_doc_no_section_passthrough(self):
        chunk = DocumentChunk(document_name="", content="Plain.", section=None, chunk_index=0)
        result = enrich_chunk(chunk)
        assert result.embedding_content == "Plain."

    def test_original_content_not_mutated(self):
        chunk = _make_chunk(breadcrumb="[doc > Intro]", content="Original text.")
        enrich_chunk(chunk)
        assert chunk.content == "Original text."

    def test_returns_new_instance(self):
        chunk = _make_chunk(breadcrumb="[doc > Intro]")
        result = enrich_chunk(chunk)
        assert result is not chunk

    def test_unknown_strategy_raises(self):
        chunk = _make_chunk(breadcrumb="[doc > Intro]")
        with pytest.raises(ValueError, match="magic"):
            enrich_chunk(chunk, strategy="magic")

    def test_all_other_fields_preserved(self):
        chunk = DocumentChunk(
            document_name="my_doc", content="Content here.",
            section=["Sec A"], chunk_index=5,
            source_offset=100, source_length=50, chunk_type="schema",
            breadcrumb="[my_doc > Sec A]",
        )
        result = enrich_chunk(chunk)
        assert result.document_name == "my_doc"
        assert result.chunk_index == 5
        assert result.source_offset == 100
        assert result.source_length == 50
        assert result.chunk_type == "schema"

    def test_content_appears_after_breadcrumb(self):
        chunk = _make_chunk(breadcrumb="[doc > Results]", content="The p-value was 0.03.")
        result = enrich_chunk(chunk)
        assert result.embedding_content.endswith("The p-value was 0.03.")
        assert "\n\n" in result.embedding_content

    def test_document_name_included_with_section(self):
        chunk = _make_chunk(breadcrumb="[techcorp_msa > Indemnification]")
        result = enrich_chunk(chunk)
        assert "techcorp_msa" in result.embedding_content
        assert "Indemnification" in result.embedding_content

    def test_document_name_disambiguates_identical_sections(self):
        chunk_a = DocumentChunk(
            document_name="techcorp_msa", content="…cap is 12 months fees…",
            section=["Limitation of Liability"], chunk_index=0,
            breadcrumb="[techcorp_msa > Limitation of Liability]",
        )
        chunk_b = DocumentChunk(
            document_name="cloudsolutions_agreement", content="…cap is 12 months fees…",
            section=["Limitation of Liability"], chunk_index=0,
            breadcrumb="[cloudsolutions_agreement > Limitation of Liability]",
        )
        result_a = enrich_chunk(chunk_a)
        result_b = enrich_chunk(chunk_b)
        assert result_a.embedding_content != result_b.embedding_content
        assert "techcorp_msa" in result_a.embedding_content
        assert "cloudsolutions_agreement" in result_b.embedding_content

    def test_include_doc_name_false_no_doc_in_breadcrumb(self):
        chunk = _make_chunk(breadcrumb="[Signs and Symptoms]", content="Text.")
        result = enrich_chunk(chunk)
        assert result.embedding_content == "[Signs and Symptoms]\n\nText."
        assert "test_doc" not in result.embedding_content


class TestEnrichChunks:
    def test_batch_all_get_embedding_content(self):
        chunks = [_make_chunk(breadcrumb=f"[doc > Sec {i}]") for i in range(5)]
        results = enrich_chunks(chunks)
        assert all(r.embedding_content is not None for r in results)

    def test_empty_list(self):
        assert enrich_chunks([]) == []

    def test_mixed_sections(self):
        chunks = [
            _make_chunk(breadcrumb="[doc_a > Intro]", content="Has a section.", doc_name="doc_a"),
            _make_chunk(breadcrumb="[doc_a]", content="No section here.", doc_name="doc_a"),
            _make_chunk(breadcrumb="[doc_a > Methods]", content="Another section.", doc_name="doc_a"),
        ]
        results = enrich_chunks(chunks)
        assert "doc_a" in results[0].embedding_content
        assert "Intro" in results[0].embedding_content
        assert "doc_a" in results[1].embedding_content
        assert "Methods" in results[2].embedding_content

    def test_returns_new_list(self):
        chunks = [_make_chunk(breadcrumb="[doc > A]"), _make_chunk(breadcrumb="[doc > B]")]
        results = enrich_chunks(chunks)
        assert results is not chunks

    def test_originals_not_mutated(self):
        chunks = [_make_chunk(breadcrumb="[doc > Sec]", content="Text.")]
        enrich_chunks(chunks)
        assert chunks[0].embedding_content is None

    def test_length_preserved(self):
        chunks = [_make_chunk(breadcrumb="[doc]") for _ in range(7)]
        assert len(enrich_chunks(chunks)) == 7

    def test_all_chunks_include_document_name(self):
        chunks = [
            DocumentChunk(
                document_name="techcorp_msa", content=f"Clause {i} text.",
                section=[f"Section {i}"], chunk_index=i,
                breadcrumb=f"[techcorp_msa > Section {i}]",
            )
            for i in range(5)
        ]
        results = enrich_chunks(chunks)
        for r in results:
            assert "techcorp_msa" in r.embedding_content


class TestDocumentChunkSectionNormalization:
    def test_section_none_becomes_empty_list(self):
        chunk = DocumentChunk(document_name="d", content="c", section=None, chunk_index=0)
        assert chunk.section == []

    def test_section_str_becomes_single_element_list(self):
        chunk = DocumentChunk(document_name="d", content="c", section="Introduction", chunk_index=0)
        assert chunk.section == ["Introduction"]

    def test_section_empty_str_becomes_empty_list(self):
        chunk = DocumentChunk(document_name="d", content="c", section="", chunk_index=0)
        assert chunk.section == []

    def test_section_list_unchanged(self):
        chunk = DocumentChunk(document_name="d", content="c", section=["A", "B"], chunk_index=0)
        assert chunk.section == ["A", "B"]

    def test_section_default_is_empty_list(self):
        chunk = DocumentChunk(document_name="d", content="c", chunk_index=0)
        assert chunk.section == []


class TestDocumentChunkSourceDerivation:
    def test_default_chunk_type_is_document(self):
        assert DocumentChunk(document_name="d", content="c", chunk_index=0).source == "document"

    def test_db_table_yields_schema(self):
        assert DocumentChunk(document_name="d", content="c", chunk_index=0, chunk_type="db_table").source == "schema"

    def test_db_column_yields_schema(self):
        assert DocumentChunk(document_name="d", content="c", chunk_index=0, chunk_type="db_column").source == "schema"

    def test_db_schema_yields_schema(self):
        assert DocumentChunk(document_name="d", content="c", chunk_index=0, chunk_type="db_schema").source == "schema"

    def test_api_endpoint_yields_api(self):
        assert DocumentChunk(document_name="d", content="c", chunk_index=0, chunk_type="api_endpoint").source == "api"

    def test_api_graphql_query_yields_api(self):
        assert DocumentChunk(document_name="d", content="c", chunk_index=0, chunk_type="api_graphql_query").source == "api"

    def test_legacy_graphql_query_yields_api(self):
        assert DocumentChunk(document_name="d", content="c", chunk_index=0, chunk_type="graphql_query").source == "api"

    def test_legacy_graphql_mutation_yields_api(self):
        assert DocumentChunk(document_name="d", content="c", chunk_index=0, chunk_type="graphql_mutation").source == "api"

    def test_explicit_source_not_overridden(self):
        chunk = DocumentChunk(document_name="d", content="c", chunk_index=0, chunk_type="db_table", source="custom")
        assert chunk.source == "custom"
