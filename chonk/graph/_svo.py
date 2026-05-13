# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SVOTriple — typed directed edge between two entities."""

from __future__ import annotations

from dataclasses import dataclass

VERB_SET: frozenset[str] = frozenset(
    {
        # Taxonomy / Classification
        "type_of",
        "instance_of",
        "classified_as",
        # Structure / Schema
        "has",
        "contains",
        "part_of",
        "composed_of",
        "extends",
        "implements",
        # Database / Lineage
        "references",
        "indexed_by",
        "partitioned_by",
        "derived_from",
        "calculated_by",
        "aggregates",
        "sourced_from",
        "populates",
        "transforms",
        # Governance / Compliance
        "governs",
        "requires",
        "defined_by",
        "complies_with",
        "enforced_by",
        "exempt_from",
        "audited_by",
        "validates",
        # Ownership / Responsibility
        "created_by",
        "owned_by",
        "maintained_by",
        "manages",
        # Equivalence / Mapping
        "equivalent_to",
        "maps_to",
        "supersedes",
        "version_of",
        "inverse_of",
        # Membership / Location
        "member_of",
        "located_in",
        "used_for",
        # Causation / Dependency
        "depends_on",
        "triggers",
        "enables",
        "causes",
        "blocks",
        "precedes",
        # Data Flow / Integration
        "produces",
        "consumes",
        "exposes",
        "masks",
        # Business Actions
        "creates",
        "places",
        "approves",
        "assigns",
        "fulfills",
        "authorizes",
        "issues",
    }
)


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
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
