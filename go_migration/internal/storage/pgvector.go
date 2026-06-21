package storage

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/kennethstott/chonk/internal/models"
)

// PgVectorBackend is the PostgreSQL + pgvector storage backend.
// Python source: chonk/storage/_pg.py  PgVectorBackend
//
// MIGRATION STUB — G4 sprint.
// Wire-up to pgx/v5 + pgvector-go is pending.
// All methods return ErrNotImplemented until the sprint is complete.
//
// Key design decisions vs. Python:
//   - Python psycopg2 → pgx/v5 (pgxpool for connection pooling)
//   - Python pgvector extension wrapper → pgvector-go
//   - Python RebuildFTSIndex() no-op (PG has live tsvector) → same here
//   - Python PreloadEmbeddings() no-op (PG HNSW auto-loaded) → same here
type PgVectorBackend struct {
	pool         *pgxpool.Pool
	embeddingDim int
}

// NewPgVectorBackend connects to the PostgreSQL database at dsn.
//
// MIGRATION STUB: pgxpool.New is correct; schema migration (CREATE TABLE …,
// CREATE INDEX USING hnsw …) pending G4.
func NewPgVectorBackend(ctx context.Context, dsn string, embeddingDim int) (*PgVectorBackend, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("pgvector backend: connect: %w", err)
	}
	return &PgVectorBackend{pool: pool, embeddingDim: embeddingDim}, nil
}

func (b *PgVectorBackend) AddChunks(
	chunks []models.DocumentChunk,
	embeddings [][]float64,
	namespace, sourceID, domainID, sessionFingerprint string,
) error {
	// TODO(G4): INSERT INTO chunks … ON CONFLICT (chunk_id) DO NOTHING
	return ErrNotImplemented
}

func (b *PgVectorBackend) RegisterDocument(documentName, contentHash, sourceURI string, chunkCount int) error {
	// TODO(G4): INSERT INTO documents … ON CONFLICT (document_name) DO UPDATE
	return ErrNotImplemented
}

func (b *PgVectorBackend) DeleteByDocument(documentName string) (int, error) {
	return 0, ErrNotImplemented
}

func (b *PgVectorBackend) Clear() error {
	return ErrNotImplemented
}

func (b *PgVectorBackend) Search(
	queryEmbedding []float64,
	limit int,
	queryText string,
	includeBreadcrumbs bool,
	namespaces, chunkTypes, domainIDs []string,
	sessionFingerprint string,
) ([]SearchResult, error) {
	// TODO(G4): pgvector cosine_distance ORDER BY + ts_rank hybrid RRF
	return nil, ErrNotImplemented
}

func (b *PgVectorBackend) GetAllChunks() ([]models.DocumentChunk, error) {
	return nil, ErrNotImplemented
}

func (b *PgVectorBackend) GetDocumentHash(documentName string) (string, error) {
	return "", ErrNotImplemented
}

func (b *PgVectorBackend) ListDocuments() ([]DocumentRecord, error) {
	return nil, ErrNotImplemented
}

func (b *PgVectorBackend) Count() (int, error) {
	return 0, ErrNotImplemented
}

// RebuildFTSIndex is a no-op for PostgreSQL (tsvector stays live).
func (b *PgVectorBackend) RebuildFTSIndex() error {
	return nil
}

// PreloadEmbeddings is a no-op for PostgreSQL (HNSW index auto-loaded).
func (b *PgVectorBackend) PreloadEmbeddings() error {
	return nil
}

func (b *PgVectorBackend) Close() error {
	if b.pool != nil {
		b.pool.Close()
	}
	return nil
}

// Ensure PgVectorBackend satisfies VectorBackend at compile time.
var _ VectorBackend = (*PgVectorBackend)(nil)
