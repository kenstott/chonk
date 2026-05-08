# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""PromptBuilder — assemble a ranked, budget-bounded prompt from AnswerContext."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ScoredChunk
    from ._context import AnswerContext

_TIER: dict[str, int] = {
    "seed": 0,
    "structural": 1,
    "entity_adjacent": 2,
    "cluster_adjacent": 3,
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class PromptBuilder:
    """Assembles a prompt string from AnswerContext within a token budget.

    Chunks are ordered by provenance tier (seed > structural > entity_adjacent >
    cluster_adjacent). The community context and query header are always included;
    chunks are added greedily until the budget is exhausted.
    """

    def _build_header(self, context: "AnswerContext") -> str:
        parts = []
        if context.community_context:
            parts.append(f"Context: {context.community_context}")
        parts.append(f"Query: {context.query}")
        return "\n".join(parts)

    def select_chunks(self, context: "AnswerContext", token_budget: int) -> list["ScoredChunk"]:
        """Return the subset of chunks that fit within token_budget."""
        sorted_chunks = sorted(
            context.chunks, key=lambda sc: _TIER.get(sc.provenance, 4)
        )
        used = _estimate_tokens(self._build_header(context))
        selected: list["ScoredChunk"] = []
        for sc in sorted_chunks:
            cost = _estimate_tokens(sc.chunk.content)
            if used + cost > token_budget:
                break
            used += cost
            selected.append(sc)
        return selected

    def build(self, context: "AnswerContext", token_budget: int) -> str:
        """Build the prompt string, respecting token_budget."""
        selected = self.select_chunks(context, token_budget)
        parts = [self._build_header(context)]
        for sc in selected:
            parts.append(
                f"[{sc.provenance}] {sc.chunk.document_name}:\n{sc.chunk.content}"
            )
        return "\n\n".join(parts)
