# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d433c31c-035d-4fc5-a7da-9e6596502656
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Chunky Monkey — a dairy-free RAG pipeline for delicious semantic similarity, clustering and NER."""
from ._versioning import VersionedRef
from .chunking import (
    NOVEL_STRUCTURAL_LEVELS,
    chunk_document,
    extract_markdown_sections,
    is_list_line,
    is_table_line,
    merge_blocks,
    promote_plain_text_headers,
)
from .cluster import ClusterMap, CooccurrenceMatrix, cluster_entities
from .community import CommunityIndex, CommunityIndexBuilder, CommunitySummarizer
from .context import enrich_chunk, enrich_chunks
from .generation import Answer, AnswerContext, AnswerGenerator, PromptBuilder
from .graph import (
    VERB_SET,
    LLMClient,
    RelationshipIndex,
    RelationshipIndexBuilder,
    SVOExtractor,
    SVOTriple,
)
from .loader import DocumentLoader
from .models import (
    ClusterRecord,
    DocumentChunk,
    Entity,
    EntityAssociation,
    LoadedDocument,
    ScoredChunk,
)
from .ner import (
    ALL_SPACY_LABELS,
    EntityIndex,
    EntityMatch,
    NerPipeline,
    SchemaMatcher,
    SchemaVocabBuilder,
    SpacyLabel,
    SpacyMatcher,
    VocabularyMatcher,
    merge_matches,
    normalize_schema_term,
)
from .schema import ColumnMeta, EndpointMeta, FieldMeta, TableMeta
from .search import EnhancedSearch
from .transports import (
    Crawler,
    DirectoryCrawler,
    FetchResult,
    FtpTransport,
    HttpTransport,
    ImapTransport,
    ImportCrawler,
    LocalTransport,
    S3Transport,
    SftpTransport,
    SqlAlchemyTransport,
    Transport,
    WebCrawler,
)

__all__ = [
    "DocumentChunk",
    "LoadedDocument",
    "EntityAssociation",
    "Entity",
    "ClusterRecord",
    "ScoredChunk",
    "chunk_document",
    "extract_markdown_sections",
    "is_list_line",
    "is_table_line",
    "merge_blocks",
    "promote_plain_text_headers",
    "NOVEL_STRUCTURAL_LEVELS",
    "enrich_chunk",
    "enrich_chunks",
    "DocumentLoader",
    # Transports & Crawlers
    "Transport",
    "FetchResult",
    "Crawler",
    "WebCrawler",
    "DirectoryCrawler",
    "ImportCrawler",
    "LocalTransport",
    "HttpTransport",
    "S3Transport",
    "FtpTransport",
    "SftpTransport",
    "SqlAlchemyTransport",
    "ImapTransport",
    # NER
    "VocabularyMatcher",
    "EntityMatch",
    "EntityIndex",
    "SpacyMatcher",
    "SpacyLabel",
    "ALL_SPACY_LABELS",
    "merge_matches",
    "SchemaMatcher",
    "normalize_schema_term",
    "SchemaVocabBuilder",
    "NerPipeline",
    # Cluster
    "CooccurrenceMatrix",
    "cluster_entities",
    "ClusterMap",
    # Search
    "EnhancedSearch",
    # Schema metadata
    "ColumnMeta",
    "TableMeta",
    "FieldMeta",
    "EndpointMeta",
    # Generation
    "AnswerContext",
    "PromptBuilder",
    "Answer",
    "AnswerGenerator",
    # Graph
    "SVOTriple",
    "VERB_SET",
    "RelationshipIndex",
    "LLMClient",
    "SVOExtractor",
    "RelationshipIndexBuilder",
    # Community
    "CommunityIndex",
    "CommunitySummarizer",
    "CommunityIndexBuilder",
    # Versioning
    "VersionedRef",
]
