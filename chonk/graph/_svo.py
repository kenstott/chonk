# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SVOTriple — typed directed edge between two entities."""

from __future__ import annotations

from dataclasses import dataclass

VERB_SET: frozenset[str] = frozenset({
    "type_of",
    "references",
    "contains",
    "part_of",
    "governs",
    "requires",
    "defined_by",
    "equivalent_to",
    "created_by",
})


@dataclass(frozen=True)
class SVOTriple:
    """Typed directed edge: subject --verb--> object.

    verb must be in VERB_SET.  references and part_of are deterministically
    derived from FK constraints; all others require LLM classification.
    LLM extraction is the caller's responsibility — this dataclass is a
    pure storage primitive.
    """
    subject_id: str
    verb: str
    object_id: str
    confidence: float
    source_chunk_id: str | None = None

    def __post_init__(self) -> None:
        if self.verb not in VERB_SET:
            raise ValueError(
                f"verb {self.verb!r} not in closed vocabulary. "
                f"Allowed: {sorted(VERB_SET)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
