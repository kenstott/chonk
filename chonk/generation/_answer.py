# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: c61fb669-8bf4-4f08-9d05-8afab68fb6bd
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Answer and AnswerGenerator — thin wrapper around a user-supplied LLM function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from ._prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from ..models import ScoredChunk
    from ._context import AnswerContext


@dataclass
class Answer:
    """Generated answer with source citations.

    Attributes:
        text: LLM-generated answer text.
        citations: Chunks that were included in the prompt (provenance-ordered).
    """
    text: str
    citations: list["ScoredChunk"] = field(default_factory=list)


class AnswerGenerator:
    """Generates answers by delegating to a user-supplied LLM function.

    Chonk owns prompt assembly (PromptBuilder); the caller owns the LLM call.

    Args:
        llm_fn: Callable that accepts a prompt string and returns answer text.
        token_budget: Max tokens to pass to PromptBuilder (default 4096).
    """

    def __init__(self, llm_fn: Callable[[str], str], token_budget: int = 4096):
        self._llm_fn = llm_fn
        self._token_budget = token_budget
        self._builder = PromptBuilder()

    def generate(self, context: "AnswerContext") -> Answer:
        """Build prompt, call llm_fn, return Answer with citations."""
        citations = self._builder.select_chunks(context, self._token_budget)
        prompt = self._builder.build(context, self._token_budget)
        text = self._llm_fn(prompt)
        return Answer(text=text, citations=citations)
