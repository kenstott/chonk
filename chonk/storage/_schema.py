# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 75483088-32a9-4c97-bbaa-288624d47278
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DDL for chonk's minimal storage schema."""

EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id     TEXT PRIMARY KEY,
    document_name TEXT NOT NULL,
    section      TEXT,
    chunk_index  INTEGER NOT NULL DEFAULT 0,
    content      TEXT NOT NULL,
    breadcrumb   TEXT,
    chunk_type   TEXT NOT NULL DEFAULT 'document',
    source_offset INTEGER,
    source_length INTEGER,
    namespace    TEXT,
    embedding    FLOAT[{dim}]
)
""".strip()

EMBEDDINGS_MIGRATE_BREADCRUMB = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS breadcrumb TEXT
""".strip()

EMBEDDINGS_MIGRATE_NAMESPACE = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS namespace TEXT
""".strip()

EMBEDDINGS_MIGRATE_SOURCE_DETAIL = """
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS source_detail TEXT
""".strip()

ENTITIES_DDL = """
CREATE TABLE IF NOT EXISTS entities (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    display_name TEXT NOT NULL,
    entity_type  TEXT NOT NULL DEFAULT 'concept',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""".strip()

CHUNK_ENTITIES_DDL = """
CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id        TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    frequency       INTEGER NOT NULL DEFAULT 1,
    positions_json  TEXT NOT NULL DEFAULT '[]',
    score           REAL NOT NULL DEFAULT 0.0,
    namespace       TEXT,
    PRIMARY KEY (chunk_id, entity_id)
)
""".strip()

CHUNK_ENTITIES_MIGRATE_NAMESPACE = """
ALTER TABLE chunk_entities ADD COLUMN IF NOT EXISTS namespace TEXT
""".strip()

DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    document_name TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    source_uri    TEXT NOT NULL DEFAULT '',
    indexed_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    chunk_count   INTEGER NOT NULL DEFAULT 0
)
""".strip()

VSS_INDEX_DDL = "CREATE INDEX IF NOT EXISTS embeddings_vss ON embeddings USING HNSW (embedding) WITH (metric = 'cosine')"
VSS_DROP_INDEX_DDL = "DROP INDEX IF EXISTS embeddings_vss"

FTS_DDL = "PRAGMA create_fts_index('embeddings', 'chunk_id', 'content', overwrite=1)"


def get_ddl(embedding_dim: int = 1024) -> list[str]:
    return [
        EMBEDDINGS_DDL.format(dim=embedding_dim),
        EMBEDDINGS_MIGRATE_BREADCRUMB,
        EMBEDDINGS_MIGRATE_NAMESPACE,
        EMBEDDINGS_MIGRATE_SOURCE_DETAIL,
        ENTITIES_DDL,
        CHUNK_ENTITIES_DDL,
        CHUNK_ENTITIES_MIGRATE_NAMESPACE,
        DOCUMENTS_DDL,
    ]
