package storage

import (
	"fmt"

	"github.com/kennethstott/chonk/internal/models"
)

// Store is the composed storage facade over VectorBackend.
// Python source: chonk/storage/_store.py  Store
//
// Usage (DuckDB — local dev):
//
//	store, err := NewStore(StoreConfig{DBPath: "index.duckdb", EmbeddingDim: 1024})
//	defer store.Close()
//	store.AddDocument(chunks, embeddings, ...)
//
// Usage (PostgreSQL — horizontal scale):
//
//	store, err := NewStore(StoreConfig{DSN: "postgresql://user:pass@host/db", EmbeddingDim: 1024})
type Store struct {
	Vector VectorBackend
}

// StoreConfig holds constructor parameters for NewStore.
// Python source: Store.__init__ parameters.
type StoreConfig struct {
	DBPath       string // DuckDB file path; ":memory:" for in-memory; ignored when DSN set
	EmbeddingDim int    // Embedding vector dimension; must match your model
	ReadOnly     bool   // DuckDB read-only mode (ignored for PG)
	DSN          string // PostgreSQL DSN; when set, uses PgVectorBackend
}

// NewStore creates a Store backed by DuckDB (default) or PostgreSQL.
func NewStore(cfg StoreConfig) (*Store, error) {
	if cfg.EmbeddingDim == 0 {
		cfg.EmbeddingDim = 1024
	}
	if cfg.DSN != "" {
		// TODO(G4): pass context properly
		// pg, err := NewPgVectorBackend(context.Background(), cfg.DSN, cfg.EmbeddingDim)
		// For now return stub error — postgres backend pending G4
		return nil, fmt.Errorf("postgres backend: pending G4 sprint")
	}
	dbPath := cfg.DBPath
	if dbPath == "" {
		dbPath = ":memory:"
	}
	duck, err := NewDuckDBVectorBackend(dbPath, cfg.EmbeddingDim)
	if err != nil {
		return nil, fmt.Errorf("store: duckdb init: %w", err)
	}
	return &Store{Vector: duck}, nil
}

// AddDocument stores chunks+embeddings and registers the document.
// Python source: Store.add_document() (convenience wrapper).
func (s *Store) AddDocument(
	documentName, contentHash, sourceURI string,
	chunks []models.DocumentChunk,
	embeddings [][]float64,
	namespace string,
) error {
	if err := s.Vector.AddChunks(chunks, embeddings, namespace, "", "", ""); err != nil {
		return fmt.Errorf("store.AddDocument AddChunks: %w", err)
	}
	if err := s.Vector.RegisterDocument(documentName, contentHash, sourceURI, len(chunks)); err != nil {
		return fmt.Errorf("store.AddDocument RegisterDocument: %w", err)
	}
	return nil
}

// Search is a convenience wrapper around Vector.Search.
func (s *Store) Search(
	queryEmbedding []float64,
	limit int,
	queryText string,
) ([]SearchResult, error) {
	return s.Vector.Search(queryEmbedding, limit, queryText, true, nil, nil, nil, "")
}

// Close releases all backend resources.
func (s *Store) Close() error {
	return s.Vector.Close()
}
