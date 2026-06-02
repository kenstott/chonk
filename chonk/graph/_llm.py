# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 5f232ee3-6908-4781-b3f2-66f84f6df5da
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""LLMClient protocol — injected into SVOExtractor by the caller."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract for an LLM completion backend.

    Callers implement this for their provider (Anthropic, OpenAI, etc.).
    chonk never imports a concrete provider SDK.
    """

    def complete(self, prompt: str) -> str:
        """Send prompt, return the raw text response."""
        ...
