// Package storage defines the VectorBackend interface and the Store facade.
//
// MIGRATION MARKER: Ported from chonk/storage/ sub-package.
// Python sources:
//   - chonk/storage/_protocol.py  (107 LOC) — VectorBackend Protocol
//   - chonk/storage/_store.py     (1056 LOC) — Store facade
//   - chonk/storage/_vector.py               — DuckDBVectorBackend
//   - chonk/storage/_pg.py                   — PgVectorBackend
//   - chonk/storage/_pool.py                 — ThreadLocalDuckDB → sync.RWMutex
//
// Status: Interface COMPLETE; DuckDB and PG backend implementations are STUBS
// (wire-up to go-duckdb and pgx/v5 is the G4 sprint work).
//
// Translation notes:
//   - Python VectorBackend Protocol → Go VectorBackend interface
//   - Python numpy ndarray embeddings → Go [][]float64
//   - Python _conn Any + _fts_dirty bool → embedded in DuckDBVectorBackend struct
//   - Python ThreadLocalDuckDB → sync.RWMutex single-writer guard
//   - Python Optional args → Go functional options or zero-value strings
package storage

import (
	"github.com/kennethstott/chonk/internal/models"
)

// VectorBackend is the Go equivalent of the Python VectorBackend Protocol.
// Implementations: DuckDBVectorBackend, PgVectorBackend.
//
// Call sequence per document:
//
//	backend.AddChunks(chunks, embeddings, ...)
//	backend.RegisterDocument(name, hash, ...)
//
// AddChunks must silently ignore duplicate chunk_ids (ON CONFLICT DO NOTHING).
type VectorBackend interface {
	// ── Ingestion ────────────────────────────────────────────────────────────

	// AddChunks stores chunks with their embeddings.
	// embeddings[i] corresponds to chunks[i].
	AddChunks(
		chunks []models.DocumentChunk,
		embeddings [][]float64,
		namespace string,
		sourceID string,
		domainID string,
		sessionFingerprint string,
	) error

	// RegisterDocument records document metadata in the document registry.
	RegisterDocument(documentName, contentHash, sourceURI string, chunkCount int) error

	// DeleteByDocument removes all chunks for the given document.
	// Returns the number of chunks deleted.
	DeleteByDocument(documentName string) (int, error)

	// Clear removes all chunks and documents.
	Clear() error

	// ── Retrieval ────────────────────────────────────────────────────────────

	// Search performs vector + optional BM25 hybrid search.
	// queryText == "" → pure vector; queryText != "" → hybrid RRF.
	// Returns (chunkID, score, chunk) triples ordered by score descending.
	Search(
		queryEmbedding []float64,
		limit int,
		queryText string,
		includeBreadcrumbs bool,
		namespaces []string,
		chunkTypes []string,
		domainIDs []string,
		sessionFingerprint string,
	) ([]SearchResult, error)

	// GetAllChunks returns every chunk (used by graph builder).
	GetAllChunks() ([]models.DocumentChunk, error)

	// ── Document registry ─────────────────────────────────────────────────────

	// GetDocumentHash returns the stored content hash for documentName, or "".
	GetDocumentHash(documentName string) (string, error)

	// ListDocuments returns metadata for all registered documents.
	ListDocuments() ([]DocumentRecord, error)

	// Count returns the total number of chunks stored.
	Count() (int, error)

	// ── Lifecycle ─────────────────────────────────────────────────────────────

	// RebuildFTSIndex refreshes the full-text-search index.
	// No-op for backends with live FTS (PG tsvector).
	RebuildFTSIndex() error

	// PreloadEmbeddings warms the ANN index into memory.
	// No-op for backends with index-backed ANN (pgvector HNSW).
	PreloadEmbeddings() error

	// Close releases backend resources.
	Close() error
}

// SearchResult is one entry returned by VectorBackend.Search.
// Python source: search result tuple (chunk_id, score, DocumentChunk)
type SearchResult struct {
	ChunkID string
	Score   float64
	Chunk   models.DocumentChunk
}

// DocumentRecord is one row from VectorBackend.ListDocuments.
// Python source: dict keys: document_name, content_hash, source_uri, indexed_at, chunk_count
type DocumentRecord struct {
	DocumentName string
	ContentHash  string
	SourceURI    string
	IndexedAt    string // ISO-8601 timestamp
	ChunkCount   int
}

// SyncResult is returned by sync operations (add / update / delete).
// Python source: chonk/storage/_store.py  SyncResult
type SyncResult struct {
	Added   int
	Updated int
	Deleted int
	Skipped int // already current (hash match)
}
