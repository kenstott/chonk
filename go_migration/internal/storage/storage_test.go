package storage_test

import (
	"testing"

	"github.com/kennethstott/chonk/internal/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Interface compliance — compile-time
// ---------------------------------------------------------------------------
// These vars prove both backends satisfy VectorBackend at compile time.
// They live in _test.go so they don't pollute the production binary.

var _ storage.VectorBackend = (*storage.DuckDBVectorBackend)(nil)
var _ storage.VectorBackend = (*storage.PgVectorBackend)(nil)

// ---------------------------------------------------------------------------
// NewStore
// ---------------------------------------------------------------------------

func TestNewStore_InMemory(t *testing.T) {
	store, err := storage.NewStore(storage.StoreConfig{
		DBPath:       ":memory:",
		EmbeddingDim: 1024,
	})
	require.NoError(t, err)
	require.NotNil(t, store)
	require.NotNil(t, store.Vector)
	defer store.Close()
}

func TestNewStore_DefaultEmbeddingDim(t *testing.T) {
	// EmbeddingDim 0 → defaults to 1024
	store, err := storage.NewStore(storage.StoreConfig{DBPath: ":memory:"})
	require.NoError(t, err)
	require.NotNil(t, store)
	defer store.Close()
}

func TestNewStore_PostgresDSN_ReturnsError(t *testing.T) {
	// PG backend is a G4 stub; NewStore returns error for DSN
	_, err := storage.NewStore(storage.StoreConfig{
		DSN:          "postgresql://user:pass@localhost/db",
		EmbeddingDim: 1024,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "G4")
}

// ---------------------------------------------------------------------------
// DuckDBVectorBackend stub behaviour
// ---------------------------------------------------------------------------

func TestDuckDB_StubMethods_ReturnNotImplemented(t *testing.T) {
	b, err := storage.NewDuckDBVectorBackend(":memory:", 1024)
	require.NoError(t, err)
	defer b.Close()

	assert.ErrorIs(t, b.AddChunks(nil, nil, "", "", "", ""), storage.ErrNotImplemented)
	assert.ErrorIs(t, b.RegisterDocument("doc", "hash", "", 0), storage.ErrNotImplemented)
	assert.ErrorIs(t, b.Clear(), storage.ErrNotImplemented)
	assert.ErrorIs(t, b.RebuildFTSIndex(), storage.ErrNotImplemented)
	assert.ErrorIs(t, b.PreloadEmbeddings(), storage.ErrNotImplemented)

	_, err = b.Search(nil, 5, "", true, nil, nil, nil, "")
	assert.ErrorIs(t, err, storage.ErrNotImplemented)

	_, err = b.Count()
	assert.ErrorIs(t, err, storage.ErrNotImplemented)

	hash, err := b.GetDocumentHash("doc")
	assert.ErrorIs(t, err, storage.ErrNotImplemented)
	assert.Empty(t, hash)
}

// ---------------------------------------------------------------------------
// SyncResult zero value
// ---------------------------------------------------------------------------

func TestSyncResult_ZeroValue(t *testing.T) {
	var r storage.SyncResult
	assert.Equal(t, 0, r.Added)
	assert.Equal(t, 0, r.Skipped)
}
