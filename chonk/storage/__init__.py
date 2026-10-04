# Copyright (c) 2025 Kenneth Stott. MIT License.

"""chonk storage — DuckDB vector store + SQLAlchemy relational store."""

from ._pg import PgVectorBackend
from ._pinecone import PineconeVectorBackend
from ._protocol import VectorBackend
from ._qdrant import QdrantVectorBackend
from ._relational import RelationalStore
from ._schema import (
    CHUNK_ENTITIES_DDL,
    EMBEDDINGS_DDL,
    ENTITIES_DDL,
    SCHEMA_VERSION,
    SchemaVersionError,
    get_ddl,
)
from ._store import EntityLookup, NamespaceEvidence, Store
from ._vector import DuckDBVectorBackend, SyncResult, prune_documents, sync_document
from ._weaviate import WeaviateVectorBackend

__all__ = [
    "Store",
    "EntityLookup",
    "NamespaceEvidence",
    "DuckDBVectorBackend",
    "PgVectorBackend",
    "PineconeVectorBackend",
    "QdrantVectorBackend",
    "RelationalStore",
    "VectorBackend",
    "WeaviateVectorBackend",
    "SyncResult",
    "sync_document",
    "prune_documents",
    "get_ddl",
    "SCHEMA_VERSION",
    "SchemaVersionError",
    "EMBEDDINGS_DDL",
    "ENTITIES_DDL",
    "CHUNK_ENTITIES_DDL",
]
