//go:build integration

// Package integration contains tests requiring live I/O (DuckDB files, Postgres).
// Run with: go test -tags integration ./tests/integration/...
//
// MIGRATION MARKER: Integration test suite mirrors chonk/tests/integration/
// Python integration tests. Requires CHONK_TEST_PG_DSN for PostgreSQL tests.
package integration

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/kennethstott/chonk/internal/models"
	"github.com/kennethstott/chonk/internal/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// DuckDB — file-backed (not :memory:)
// ---------------------------------------------------------------------------

func TestDuckDB_FileBackedStore(t *testing.T) {
	// G4 STUB: this test will pass once DuckDB backend is fully implemented.
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.duckdb")

	b, err := storage.NewDuckDBVectorBackend(dbPath, 1024)
	require.NoError(t, err)
	defer b.Close()

	// Verify the stub reports ErrNotImplemented (will change to nil in G4).
	assert.ErrorIs(t, b.Clear(), storage.ErrNotImplemented, "G4 stub expected")
}

// ---------------------------------------------------------------------------
// PostgreSQL + pgvector (requires live Postgres with pgvector extension)
// ---------------------------------------------------------------------------

func TestPgVector_FullRoundTrip(t *testing.T) {
	dsn := os.Getenv("CHONK_TEST_PG_DSN")
	if dsn == "" {
		t.Skip("CHONK_TEST_PG_DSN not set")
	}

	// G4 STUB: Wire up once PgVectorBackend is fully implemented.
	ctx := context.Background()
	b, err := storage.NewPgVectorBackend(ctx, dsn, 1024)
	require.NoError(t, err)
	defer b.Close()

	chunk := models.NewDocumentChunk("test-doc", "This is a test chunk for pgvector.")
	embeddings := [][]float64{make([]float64, 1024)}

	// G4 STUB: These will pass once AddChunks is implemented.
	err = b.AddChunks([]models.DocumentChunk{chunk}, embeddings, "test", "", "", "")
	assert.ErrorIs(t, err, storage.ErrNotImplemented, "G4 stub expected")
}
