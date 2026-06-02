# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: aa608ab8-c43c-43cc-9402-fa4270caaa5e
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Graph primitives for Phase 4 GraphRAG extensions."""

from ._builder import RelationshipIndexBuilder
from ._context_graph import ContextEdge, ContextGraphStats
from ._entity_pipeline import (
    PHASE_EMBED_ENTITIES,
    PHASE_EXTRACT,
    PHASE_LOAD,
    PHASE_PERSIST_ALIASES,
    PHASE_PERSIST_DESCRIPTIONS,
    PHASE_PERSIST_TRIPLES,
    EntityGraphPipeline,
    EntityGraphStats,
)
from ._extractor import SVOExtractor
from ._index import RelationshipIndex
from ._llm import LLMClient
from ._svo import VERB_SET, SVOTriple

__all__ = ["SVOTriple", "VERB_SET", "RelationshipIndex", "LLMClient", "SVOExtractor", "RelationshipIndexBuilder", "EntityGraphPipeline", "EntityGraphStats", "PHASE_LOAD", "PHASE_EXTRACT", "PHASE_PERSIST_TRIPLES", "PHASE_PERSIST_DESCRIPTIONS", "PHASE_PERSIST_ALIASES", "PHASE_EMBED_ENTITIES", "ContextEdge", "ContextGraphStats"]
