# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Unit tests for CommunitySummarizer and related CommunityIndex additions (Phase 4.2)."""

import pytest

from chonk.community import CommunityIndex, CommunitySummarizer
from chonk.models import DocumentChunk
from chonk.graph import LLMClient


# ---------------------------------------------------------------------------
# Stub LLM helpers
# ---------------------------------------------------------------------------

class StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        return self._response


class EchoLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


def _make_index(n_communities: int = 2, chunks_per: int = 3) -> CommunityIndex:
    """Build a minimal CommunityIndex without running Louvain."""
    idx = CommunityIndex()
    for cid in range(n_communities):
        for i in range(chunks_per):
            chunk_id = f"chunk_{cid}_{i}"
            idx._chunk_to_community[chunk_id] = cid
            idx._community_to_chunks[cid].append(chunk_id)
        idx._community_to_label[cid] = f"label_{cid}"
        idx._community_to_coherence[cid] = 0.8
    return idx


# ---------------------------------------------------------------------------
# CommunitySummarizer — constructor
# ---------------------------------------------------------------------------

class TestCommunitySummarizerConstructor:
    def test_requires_llm_client(self):
        with pytest.raises(TypeError):
            CommunitySummarizer("not_a_client")  # type: ignore[arg-type]

    def test_stub_satisfies_protocol(self):
        CommunitySummarizer(StubLLM("ok"))

    def test_non_client_class_rejected(self):
        class Bad:
            pass
        with pytest.raises(TypeError):
            CommunitySummarizer(Bad())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CommunitySummarizer — summarize
# ---------------------------------------------------------------------------

class TestCommunitySummarizerSummarize:
    def test_returns_document_chunk(self):
        chunk = CommunitySummarizer(StubLLM("summary text")).summarize(0, ["text a", "text b"])
        assert isinstance(chunk, DocumentChunk)

    def test_chunk_type_is_community_summary(self):
        chunk = CommunitySummarizer(StubLLM("s")).summarize(1, ["x"])
        assert chunk.chunk_type == "community_summary"

    def test_document_name_encodes_community_id(self):
        chunk = CommunitySummarizer(StubLLM("s")).summarize(42, ["x"])
        assert chunk.document_name == "community:42"

    def test_document_name_string_community_id(self):
        chunk = CommunitySummarizer(StubLLM("s")).summarize("cluster_7", ["x"])
        assert chunk.document_name == "community:cluster_7"

    def test_content_is_llm_output(self):
        expected = "Clusters around billing and payment concepts."
        chunk = CommunitySummarizer(StubLLM(expected)).summarize(0, ["billing"])
        assert chunk.content == expected.strip()

    def test_source_is_community(self):
        chunk = CommunitySummarizer(StubLLM("s")).summarize(0, ["text"])
        assert chunk.source == "community"

    def test_empty_texts_returns_none(self):
        result = CommunitySummarizer(StubLLM("s")).summarize(0, [])
        assert result is None

    def test_whitespace_stripped_from_content(self):
        chunk = CommunitySummarizer(StubLLM("  padded  \n")).summarize(0, ["t"])
        assert chunk.content == "padded"

    def test_prompt_contains_chunk_texts(self):
        llm = EchoLLM("s")
        CommunitySummarizer(llm).summarize(0, ["unique_abc_xyz_sentinel"])
        assert "unique_abc_xyz_sentinel" in llm.prompts[0]

    def test_prompt_contains_topic_label(self):
        llm = EchoLLM("s")
        CommunitySummarizer(llm).summarize(0, ["text"], topic_label="billing, invoices")
        assert "billing, invoices" in llm.prompts[0]

    def test_section_contains_topic_label(self):
        chunk = CommunitySummarizer(StubLLM("s")).summarize(0, ["t"], topic_label="payments")
        assert chunk.section == ["payments"]

    def test_section_empty_when_no_label(self):
        chunk = CommunitySummarizer(StubLLM("s")).summarize(0, ["t"])
        assert chunk.section == []

    def test_multiple_chunks_all_in_prompt(self):
        llm = EchoLLM("s")
        CommunitySummarizer(llm).summarize(0, ["alpha", "beta", "gamma"])
        prompt = llm.prompts[0]
        assert "alpha" in prompt
        assert "beta" in prompt
        assert "gamma" in prompt


# ---------------------------------------------------------------------------
# CommunitySummarizer — summarize_all
# ---------------------------------------------------------------------------

class TestCommunitySummarizerSummarizeAll:
    def test_returns_one_chunk_per_community(self):
        idx = _make_index(n_communities=3, chunks_per=2)
        chunks = CommunitySummarizer(StubLLM("s")).summarize_all(
            idx, lambda cid: f"text for {cid}"
        )
        assert len(chunks) == 3

    def test_min_chunks_filter_skips_small_communities(self):
        idx = _make_index(n_communities=2, chunks_per=1)
        chunks = CommunitySummarizer(StubLLM("s")).summarize_all(
            idx, lambda cid: "text", min_chunks=2
        )
        assert chunks == []

    def test_default_min_chunks_is_two(self):
        idx = _make_index(n_communities=2, chunks_per=1)
        chunks = CommunitySummarizer(StubLLM("s")).summarize_all(
            idx, lambda cid: "text"
        )
        assert chunks == []

    def test_missing_chunk_text_is_skipped(self):
        idx = _make_index(n_communities=1, chunks_per=3)
        chunks = CommunitySummarizer(StubLLM("s")).summarize_all(
            idx, lambda cid: None
        )
        assert chunks == []

    def test_partial_chunk_text_uses_available(self):
        idx = _make_index(n_communities=1, chunks_per=3)
        # Only return text for chunk_0_0, not the others
        def get_text(cid: str) -> str | None:
            return "text" if cid == "chunk_0_0" else None
        # With min_chunks=1 this should succeed
        chunks = CommunitySummarizer(StubLLM("s")).summarize_all(
            idx, get_text, min_chunks=1
        )
        assert len(chunks) == 1

    def test_empty_index_returns_empty(self):
        idx = CommunityIndex()
        chunks = CommunitySummarizer(StubLLM("s")).summarize_all(
            idx, lambda cid: "text"
        )
        assert chunks == []

    def test_all_chunks_have_community_summary_type(self):
        idx = _make_index(n_communities=2, chunks_per=2)
        chunks = CommunitySummarizer(StubLLM("s")).summarize_all(
            idx, lambda cid: "text"
        )
        assert all(c.chunk_type == "community_summary" for c in chunks)

    def test_all_chunks_have_community_source(self):
        idx = _make_index(n_communities=2, chunks_per=2)
        chunks = CommunitySummarizer(StubLLM("s")).summarize_all(
            idx, lambda cid: "text"
        )
        assert all(c.source == "community" for c in chunks)


# ---------------------------------------------------------------------------
# CommunitySummarizer — custom prompt overrides
# ---------------------------------------------------------------------------

class TestCommunitySummarizerOverrides:
    def test_custom_system_prompt_used(self):
        llm = EchoLLM("s")
        CommunitySummarizer(llm, system_prompt="CUSTOM_SYS ").summarize(0, ["t"])
        assert llm.prompts[0].startswith("CUSTOM_SYS")

    def test_custom_user_template_used(self):
        llm = EchoLLM("s")
        tmpl = "TOPIC={topic_label} N={n_chunks} DATA={chunks}"
        CommunitySummarizer(llm, user_template=tmpl).summarize(0, ["mytext"], topic_label="x")
        assert "TOPIC=x" in llm.prompts[0]
        assert "DATA=mytext" in llm.prompts[0]


# ---------------------------------------------------------------------------
# CommunityIndex — community_ids
# ---------------------------------------------------------------------------

class TestCommunityIndexCommunityIds:
    def test_empty_index_returns_empty(self):
        assert CommunityIndex().community_ids() == []

    def test_returns_all_community_ids(self):
        idx = _make_index(n_communities=4)
        assert set(idx.community_ids()) == {0, 1, 2, 3}

    def test_count_matches_community_ids_length(self):
        idx = _make_index(n_communities=5)
        assert len(idx.community_ids()) == idx.community_count()


# ---------------------------------------------------------------------------
# CommunityIndex — topic_label_for_community
# ---------------------------------------------------------------------------

class TestCommunityIndexTopicLabelForCommunity:
    def test_returns_label(self):
        idx = _make_index(n_communities=2)
        assert idx.topic_label_for_community(0) == "label_0"
        assert idx.topic_label_for_community(1) == "label_1"

    def test_unknown_community_returns_empty_string(self):
        idx = _make_index(n_communities=1)
        assert idx.topic_label_for_community(99) == ""


# ---------------------------------------------------------------------------
# DocumentChunk — community_summary source derivation
# ---------------------------------------------------------------------------

class TestDocumentChunkCommunitySummarySource:
    def test_source_is_community(self):
        chunk = DocumentChunk(
            document_name="community:0",
            content="test",
            chunk_type="community_summary",
        )
        assert chunk.source == "community"

    def test_other_chunk_types_unaffected(self):
        assert DocumentChunk(document_name="x", content="y", chunk_type="document").source == "document"
        assert DocumentChunk(document_name="x", content="y", chunk_type="db_table").source == "schema"
        assert DocumentChunk(document_name="x", content="y", chunk_type="api_endpoint").source == "api"


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

class TestTopLevelImport:
    def test_community_summarizer_importable(self):
        import chonk
        assert chonk.CommunitySummarizer is CommunitySummarizer

    def test_community_index_importable(self):
        import chonk
        assert chonk.CommunityIndex is CommunityIndex

    def test_llm_client_still_importable(self):
        import chonk
        assert chonk.LLMClient is LLMClient
