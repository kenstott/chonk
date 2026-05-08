# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: c9f6fe35-26e5-4e0a-bb6c-77278a38c5ed
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""chonk storage — DuckDB vector store + SQLAlchemy relational store."""

from ._pg import PgVectorBackend
from ._protocol import VectorBackend
from ._relational import RelationalStore
from ._schema import CHUNK_ENTITIES_DDL, EMBEDDINGS_DDL, ENTITIES_DDL, get_ddl
from ._store import Store
from ._vector import DuckDBVectorBackend, SyncResult, sync_document

__all__ = [
    "Store",
    "DuckDBVectorBackend",
    "PgVectorBackend",
    "RelationalStore",
    "VectorBackend",
    "SyncResult",
    "sync_document",
    "get_ddl",
    "EMBEDDINGS_DDL",
    "ENTITIES_DDL",
    "CHUNK_ENTITIES_DDL",
]
