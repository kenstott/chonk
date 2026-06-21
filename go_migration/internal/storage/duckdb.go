package storage

import (
	"database/sql"
	"fmt"
	"sync"

	"github.com/kennethstott/chonk/internal/models"
)

// DuckDBVectorBackend is the primary local storage backend.
// Python source: chonk/storage/_vector.py  DuckDBVectorBackend
//
// MIGRATION STUB — G4 sprint.
// Wire-up to go-duckdb (github.com/marcboeker/go-duckdb) is pending.
// All methods return ErrNotImplemented until the sprint is complete.
//
// Key design decisions vs. Python:
//   - Python ThreadLocalDuckDB (thread-local connections) → sync.RWMutex
//     single-writer / multiple-reader guard on one shared *sql.DB.
//   - Python _fts_dirty bool flag → ftsDirty field; callers must call
//     RebuildFTSIndex() before a Search() after any AddChunks().
//   - Python numpy ndarray embeddings → [][]float64 (serialised to BLOB).
type DuckDBVectorBackend struct {
	db           *sql.DB
	mu           sync.RWMutex
	embeddingDim int
	ftsDirty     bool
}

// ErrNotImplemented is returned by stub methods pending G4 implementation.
var ErrNotImplemented = fmt.Errorf("not yet implemented: pending G4 sprint")

// NewDuckDBVectorBackend opens (or creates) a DuckDB file at dbPath.
// Use ":memory:" for an ephemeral in-memory store.
//
// MIGRATION STUB: replace the sql.Open call with go-duckdb once the dependency
// is pinned in go.mod.
func NewDuckDBVectorBackend(dbPath string, embeddingDim int) (*DuckDBVectorBackend, error) {
	// TODO(G4): db, err := sql.Open("duckdb", dbPath)
	// For now, return the struct with a nil db so interface is satisfiable.
	return &DuckDBVectorBackend{
		db:           nil,
		embeddingDim: embeddingDim,
	}, nil
}

func (b *DuckDBVectorBackend) AddChunks(
	chunks []models.DocumentChunk,
	embeddings [][]float64,
	namespace, sourceID, domainID, sessionFingerprint string,
) error {
	// TODO(G4): implement INSERT OR IGNORE INTO chunks …
	return ErrNotImplemented
}

func (b *DuckDBVectorBackend) RegisterDocument(documentName, contentHash, sourceURI string, chunkCount int) error {
	// TODO(G4): implement INSERT OR REPLACE INTO documents …
	return ErrNotImplemented
}

func (b *DuckDBVectorBackend) DeleteByDocument(documentName string) (int, error) {
	// TODO(G4): DELETE FROM chunks WHERE document_name = ?
	return 0, ErrNotImplemented
}

func (b *DuckDBVectorBackend) Clear() error {
	// TODO(G4): TRUNCATE chunks; TRUNCATE documents
	return ErrNotImplemented
}

func (b *DuckDBVectorBackend) Search(
	queryEmbedding []float64,
	limit int,
	queryText string,
	includeBreadcrumbs bool,
	namespaces, chunkTypes, domainIDs []string,
	sessionFingerprint string,
) ([]SearchResult, error) {
	// TODO(G4): implement VSS + optional BM25 RRF hybrid search
	return nil, ErrNotImplemented
}

func (b *DuckDBVectorBackend) GetAllChunks() ([]models.DocumentChunk, error) {
	// TODO(G4): SELECT * FROM chunks
	return nil, ErrNotImplemented
}

func (b *DuckDBVectorBackend) GetDocumentHash(documentName string) (string, error) {
	// TODO(G4): SELECT content_hash FROM documents WHERE document_name = ?
	return "", ErrNotImplemented
}

func (b *DuckDBVectorBackend) ListDocuments() ([]DocumentRecord, error) {
	// TODO(G4): SELECT * FROM documents
	return nil, ErrNotImplemented
}

func (b *DuckDBVectorBackend) Count() (int, error) {
	// TODO(G4): SELECT COUNT(*) FROM chunks
	return 0, ErrNotImplemented
}

func (b *DuckDBVectorBackend) RebuildFTSIndex() error {
	// TODO(G4): PRAGMA fts_rebuild or equivalent DuckDB FTS refresh
	b.mu.Lock()
	defer b.mu.Unlock()
	b.ftsDirty = false
	return ErrNotImplemented
}

func (b *DuckDBVectorBackend) PreloadEmbeddings() error {
	// TODO(G4): load FAISS index or DuckDB VSS extension
	return ErrNotImplemented
}

func (b *DuckDBVectorBackend) Close() error {
	if b.db != nil {
		return b.db.Close()
	}
	return nil
}

// Ensure DuckDBVectorBackend satisfies VectorBackend at compile time.
var _ VectorBackend = (*DuckDBVectorBackend)(nil)
