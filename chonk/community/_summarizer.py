# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CommunitySummarizer — LLM-driven summary chunk generation for each community."""

from __future__ import annotations

from collections.abc import Callable

from ..graph._llm import LLMClient
from ..models import DocumentChunk
from ._index import CommunityIndex

_SYSTEM_PROMPT = """\
You are a knowledge-graph analyst. Given a set of text chunks that belong to the same \
semantic community, write a concise thematic summary (2-4 sentences) capturing the main \
topics, entities, and relationships present. The summary will be used for retrieval — \
make it dense with key terms.

Return only the summary text. No headings, no bullets, no preamble.
"""

_USER_TEMPLATE = """\
Community topic: {topic_label}
Number of chunks: {n_chunks}

Chunks:
{chunks}

Write a thematic summary of this community.
"""


class CommunitySummarizer:
    """Generates a DocumentChunk summary for each community via an injected LLM.

    Follows the same injection pattern as SVOExtractor: chonk owns the default
    prompt; the LLM backend is provided by the caller.

    Args:
        llm: Concrete LLMClient implementation (required).
        system_prompt: Override the system prompt. Defaults to the chonk standard.
        user_template: Override the user message. Must contain ``{topic_label}``,
            ``{n_chunks}``, and ``{chunks}`` placeholders.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        system_prompt: str | None = None,
        user_template: str | None = None,
    ) -> None:
        if not isinstance(llm, LLMClient):
            raise TypeError(f"llm must implement LLMClient protocol, got {type(llm)}")
        self._llm = llm
        self._system = system_prompt if system_prompt is not None else _SYSTEM_PROMPT
        self._user_template = user_template if user_template is not None else _USER_TEMPLATE

    def summarize(
        self,
        community_id: int | str,
        chunk_texts: list[str],
        topic_label: str = "",
        max_chars: int = 400_000,
    ) -> DocumentChunk | None:
        """Summarize one community and return a DocumentChunk.

        Args:
            community_id: Identifier stored in the chunk's document_name.
            chunk_texts: Text content of all member chunks.
            topic_label: Optional topic label for the community.
            max_chars: Truncate combined chunk text to this many characters
                before building the prompt (~1 char ≈ 0.25 tokens).

        Returns:
            A DocumentChunk with chunk_type="community_summary", or None if
            *chunk_texts* is empty.
        """
        if not chunk_texts:
            return None
        combined = "\n---\n".join(chunk_texts)
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        prompt = (
            self._system
            + "\n\n"
            + self._user_template.format(
                topic_label=topic_label or "(unlabeled)",
                n_chunks=len(chunk_texts),
                chunks=combined,
            )
        )
        summary = self._llm.complete(prompt)
        return DocumentChunk(
            document_name=f"community:{community_id}",
            content=summary.strip(),
            chunk_type="community_summary",
            section=[topic_label] if topic_label else [],
        )

    def summarize_all(
        self,
        community_index: CommunityIndex,
        get_chunk_text: Callable[[str], str | None],
        min_chunks: int = 2,
        level: int | None = None,
    ) -> list[DocumentChunk]:
        """Summarize all communities in *community_index*.

        Args:
            community_index: Populated CommunityIndex.
            get_chunk_text: Callable mapping chunk_id → text (or None to skip).
            min_chunks: Skip communities with fewer than this many chunks.
            level: Level to summarize. None = finest (default); 0 = coarsest.

        Returns:
            Flat list of community_summary DocumentChunks (one per eligible community).
        """
        results: list[DocumentChunk] = []
        for cid in community_index.community_ids(level=level):
            chunk_ids = community_index.community_chunks(cid, level=level)
            if len(chunk_ids) < min_chunks:
                continue
            texts = [t for cid_ in chunk_ids if (t := get_chunk_text(cid_)) is not None]
            if not texts:
                continue
            label = community_index.topic_label_for_community(cid, level=level)
            doc_community_id = f"{level}:{cid}" if level is not None else cid
            chunk = self.summarize(doc_community_id, texts, topic_label=label)
            if chunk is not None:
                results.append(chunk)
        return results
