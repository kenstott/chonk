# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 0de403a8-1b30-435c-999f-33da743835a5
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for chonk.generation — AnswerContext, PromptBuilder, Answer, AnswerGenerator."""

import pytest

from chonk.generation import AnswerContext, PromptBuilder, Answer, AnswerGenerator
from chonk.models import DocumentChunk, ScoredChunk


def _make_scored_chunk(
    doc_name: str,
    content: str,
    provenance: str = "seed",
    chunk_index: int = 0,
) -> ScoredChunk:
    chunk = DocumentChunk(
        document_name=doc_name,
        content=content,
        chunk_index=chunk_index,
    )
    return ScoredChunk(
        chunk_id=f"{doc_name}_{chunk_index}",
        chunk=chunk,
        score=1.0,
        provenance=provenance,
    )


class TestAnswerContext:
    def test_construction_minimal(self):
        ctx = AnswerContext(chunks=[], query="what is X?")
        assert ctx.query == "what is X?"
        assert ctx.chunks == []
        assert ctx.community_context is None
        assert ctx.active_entities == []

    def test_construction_full(self):
        sc = _make_scored_chunk("doc1", "content")
        ctx = AnswerContext(
            chunks=[sc],
            query="tell me about Y",
            community_context="domain framing",
            active_entities=["Y", "Z"],
        )
        assert len(ctx.chunks) == 1
        assert ctx.community_context == "domain framing"
        assert ctx.active_entities == ["Y", "Z"]


class TestPromptBuilder:
    def test_build_includes_query(self):
        ctx = AnswerContext(chunks=[], query="what is X?")
        builder = PromptBuilder()
        prompt = builder.build(ctx, token_budget=4096)
        assert "what is X?" in prompt

    def test_build_includes_community_context(self):
        ctx = AnswerContext(chunks=[], query="Q", community_context="domain framing")
        prompt = PromptBuilder().build(ctx, token_budget=4096)
        assert "domain framing" in prompt

    def test_build_no_community_context(self):
        ctx = AnswerContext(chunks=[], query="Q")
        prompt = PromptBuilder().build(ctx, token_budget=4096)
        assert "Context:" not in prompt

    def test_build_includes_chunk_content(self):
        sc = _make_scored_chunk("doc1", "The answer is 42.")
        ctx = AnswerContext(chunks=[sc], query="Q")
        prompt = PromptBuilder().build(ctx, token_budget=4096)
        assert "The answer is 42." in prompt

    def test_build_includes_provenance_label(self):
        sc = _make_scored_chunk("doc1", "content", provenance="seed")
        ctx = AnswerContext(chunks=[sc], query="Q")
        prompt = PromptBuilder().build(ctx, token_budget=4096)
        assert "[seed]" in prompt

    def test_provenance_order_seed_before_cluster(self):
        cluster_sc = _make_scored_chunk("doc1", "cluster content", provenance="cluster_adjacent")
        seed_sc = _make_scored_chunk("doc2", "seed content", provenance="seed")
        ctx = AnswerContext(chunks=[cluster_sc, seed_sc], query="Q")
        prompt = PromptBuilder().build(ctx, token_budget=4096)
        assert prompt.index("seed content") < prompt.index("cluster content")

    def test_provenance_order_all_tiers(self):
        chunks = [
            _make_scored_chunk("d1", "cluster", provenance="cluster_adjacent"),
            _make_scored_chunk("d2", "entity", provenance="entity_adjacent"),
            _make_scored_chunk("d3", "structural", provenance="structural"),
            _make_scored_chunk("d4", "seed", provenance="seed"),
        ]
        ctx = AnswerContext(chunks=chunks, query="Q")
        prompt = PromptBuilder().build(ctx, token_budget=4096)
        positions = {
            "seed": prompt.index("seed"),
            "structural": prompt.index("structural"),
            "entity": prompt.index("entity"),
            "cluster": prompt.index("cluster"),
        }
        assert positions["seed"] < positions["structural"] < positions["entity"] < positions["cluster"]

    def test_token_budget_excludes_oversized_chunks(self):
        # 400-char content ≈ 100 tokens each
        big_content = "x" * 400
        chunks = [_make_scored_chunk(f"doc{i}", big_content, provenance="seed", chunk_index=i) for i in range(10)]
        ctx = AnswerContext(chunks=chunks, query="Q")
        # Budget of 150 tokens: header ~2 tokens + 1 chunk (100 tokens) fits, 2nd would exceed
        selected = PromptBuilder().select_chunks(ctx, token_budget=150)
        assert len(selected) == 1

    def test_select_chunks_returns_empty_when_budget_exhausted_by_header(self):
        sc = _make_scored_chunk("doc1", "x" * 400)
        ctx = AnswerContext(chunks=[sc], query="Q")
        # Budget of 1 token — header alone exceeds it
        selected = PromptBuilder().select_chunks(ctx, token_budget=1)
        assert len(selected) == 0

    def test_select_chunks_all_fit(self):
        chunks = [_make_scored_chunk(f"doc{i}", "short", chunk_index=i) for i in range(5)]
        ctx = AnswerContext(chunks=chunks, query="Q")
        selected = PromptBuilder().select_chunks(ctx, token_budget=4096)
        assert len(selected) == 5

    def test_build_returns_string(self):
        ctx = AnswerContext(chunks=[], query="Q")
        result = PromptBuilder().build(ctx, token_budget=4096)
        assert isinstance(result, str)


class TestAnswer:
    def test_construction(self):
        sc = _make_scored_chunk("doc1", "content")
        answer = Answer(text="The answer.", citations=[sc])
        assert answer.text == "The answer."
        assert len(answer.citations) == 1

    def test_default_citations_empty(self):
        answer = Answer(text="answer")
        assert answer.citations == []


class TestAnswerGenerator:
    def test_calls_llm_fn(self):
        calls = []

        def fake_llm(prompt: str) -> str:
            calls.append(prompt)
            return "generated answer"

        sc = _make_scored_chunk("doc1", "content about X")
        ctx = AnswerContext(chunks=[sc], query="what is X?")
        gen = AnswerGenerator(llm_fn=fake_llm)
        answer = gen.generate(ctx)

        assert len(calls) == 1
        assert "what is X?" in calls[0]

    def test_returns_answer_dataclass(self):
        gen = AnswerGenerator(llm_fn=lambda p: "response")
        ctx = AnswerContext(chunks=[], query="Q")
        answer = gen.generate(ctx)
        assert isinstance(answer, Answer)
        assert answer.text == "response"

    def test_citations_are_selected_chunks(self):
        sc1 = _make_scored_chunk("doc1", "x" * 400, provenance="seed", chunk_index=0)
        sc2 = _make_scored_chunk("doc2", "x" * 400, provenance="structural", chunk_index=1)
        sc3 = _make_scored_chunk("doc3", "x" * 400, provenance="cluster_adjacent", chunk_index=2)
        ctx = AnswerContext(chunks=[sc1, sc2, sc3], query="Q")
        # Budget of 120 tokens: header + 1 chunk fits, 2nd might not
        gen = AnswerGenerator(llm_fn=lambda p: "answer", token_budget=120)
        answer = gen.generate(ctx)
        # seed chunk should be cited; cluster might not fit
        assert sc1 in answer.citations

    def test_llm_fn_receives_chunk_content(self):
        received = []
        sc = _make_scored_chunk("doc1", "special content here")
        ctx = AnswerContext(chunks=[sc], query="Q")
        gen = AnswerGenerator(llm_fn=lambda p: received.append(p) or "ok")
        gen.generate(ctx)
        assert "special content here" in received[0]

    def test_custom_token_budget(self):
        # Very small budget — no chunks should be cited
        big_content = "w" * 4000
        sc = _make_scored_chunk("doc1", big_content)
        ctx = AnswerContext(chunks=[sc], query="Q")
        gen = AnswerGenerator(llm_fn=lambda p: "ok", token_budget=10)
        answer = gen.generate(ctx)
        assert answer.citations == []
