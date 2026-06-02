# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 79eb3577-585f-44ab-86a7-d2a07072bc13
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""AnswerContext — retrieval output ready for prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ScoredChunk


@dataclass
class AnswerContext:
    """Retrieval output packaged for prompt assembly.

    Attributes:
        chunks: Ranked ScoredChunk list from EnhancedSearch (includes provenance).
        community_context: Optional community-level framing string (Phase 4.2).
        query: The original user query.
        active_entities: Entity names extracted from the query or retrieval.
    """
    chunks: list["ScoredChunk"]
    query: str
    community_context: str | None = None
    active_entities: list[str] = field(default_factory=list)
