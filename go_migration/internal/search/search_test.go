package search_test

import (
	"testing"

	"github.com/kennethstott/chonk/internal/models"
	"github.com/kennethstott/chonk/internal/search"
	"github.com/kennethstott/chonk/internal/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// EnhancedSearchOptions defaults
// ---------------------------------------------------------------------------

func TestDefaultEnhancedSearchOptions(t *testing.T) {
	opts := search.DefaultEnhancedSearchOptions()
	assert.Equal(t, 3, opts.SeedPoolMultiplier)
	assert.Equal(t, 3, opts.EntityExpansionTopN)
	assert.InDelta(t, 0.3, opts.LambdaDiversity, 1e-9)
	assert.InDelta(t, 0.5, opts.RelevanceWeight, 1e-9)
	assert.InDelta(t, 0.2, opts.PriorityWeight, 1e-9)
	assert.InDelta(t, 0.3, opts.CoverageWeight, 1e-9)
	assert.True(t, opts.StructuralExpansion)
	assert.True(t, opts.EntityExpansion)
	assert.True(t, opts.ClusterExpansion)
}

// ---------------------------------------------------------------------------
// EnhancedSearch with stub backend
// ---------------------------------------------------------------------------

// stubBackend satisfies storage.VectorBackend and returns preset results.
type stubBackend struct {
	results []storage.SearchResult
	err     error
}

func (b *stubBackend) AddChunks(_ []models.DocumentChunk, _ [][]float64, _, _, _, _ string) error {
	return nil
}
func (b *stubBackend) RegisterDocument(_, _, _ string, _ int) error { return nil }
func (b *stubBackend) DeleteByDocument(_ string) (int, error)       { return 0, nil }
func (b *stubBackend) Clear() error                                  { return nil }
func (b *stubBackend) Search(_ []float64, limit int, _ string, _ bool, _, _, _ []string, _ string) ([]storage.SearchResult, error) {
	if b.err != nil {
		return nil, b.err
	}
	if len(b.results) > limit {
		return b.results[:limit], nil
	}
	return b.results, nil
}
func (b *stubBackend) GetAllChunks() ([]models.DocumentChunk, error)   { return nil, nil }
func (b *stubBackend) GetDocumentHash(_ string) (string, error)        { return "", nil }
func (b *stubBackend) ListDocuments() ([]storage.DocumentRecord, error) { return nil, nil }
func (b *stubBackend) Count() (int, error)                             { return 0, nil }
func (b *stubBackend) RebuildFTSIndex() error                          { return nil }
func (b *stubBackend) PreloadEmbeddings() error                        { return nil }
func (b *stubBackend) Close() error                                    { return nil }

func makeResults(n int) []storage.SearchResult {
	results := make([]storage.SearchResult, n)
	for i := range results {
		results[i] = storage.SearchResult{
			ChunkID: "c" + string(rune('0'+i)),
			Score:   1.0 - float64(i)*0.1,
			Chunk:   models.NewDocumentChunk("doc", "content"),
		}
	}
	return results
}

func TestSearch_ReturnsKResults(t *testing.T) {
	backend := &stubBackend{results: makeResults(10)}
	s := search.NewEnhancedSearch(backend, nil, search.DefaultEnhancedSearchOptions())

	resp, err := s.Search(search.SearchRequest{QueryEmbedding: []float64{0.1, 0.2}, K: 3})
	require.NoError(t, err)
	assert.Len(t, resp.Chunks, 3)
}

func TestSearch_DefaultK(t *testing.T) {
	// K=0 → defaults to 5
	backend := &stubBackend{results: makeResults(10)}
	s := search.NewEnhancedSearch(backend, nil, search.DefaultEnhancedSearchOptions())

	resp, err := s.Search(search.SearchRequest{QueryEmbedding: []float64{0.1}})
	require.NoError(t, err)
	assert.LessOrEqual(t, len(resp.Chunks), 5)
}

func TestSearch_ProvenanceIsSeed(t *testing.T) {
	backend := &stubBackend{results: makeResults(3)}
	s := search.NewEnhancedSearch(backend, nil, search.DefaultEnhancedSearchOptions())

	resp, err := s.Search(search.SearchRequest{QueryEmbedding: []float64{0.1}, K: 3})
	require.NoError(t, err)
	for _, sc := range resp.Chunks {
		assert.Equal(t, "seed", sc.Provenance)
	}
}

func TestSearch_TraceMatchesResultCount(t *testing.T) {
	backend := &stubBackend{results: makeResults(5)}
	s := search.NewEnhancedSearch(backend, nil, search.DefaultEnhancedSearchOptions())

	resp, err := s.Search(search.SearchRequest{QueryEmbedding: []float64{0.1}, K: 5})
	require.NoError(t, err)
	assert.Equal(t, resp.Trace.Total, len(resp.Chunks))
}

func TestSearch_ChunkFilter(t *testing.T) {
	backend := &stubBackend{results: makeResults(5)}
	s := search.NewEnhancedSearch(backend, nil, search.DefaultEnhancedSearchOptions())
	// Filter: drop everything
	s.WithChunkFilter(func(_ []models.ScoredChunk) []models.ScoredChunk {
		return nil
	})

	resp, err := s.Search(search.SearchRequest{QueryEmbedding: []float64{0.1}, K: 5})
	require.NoError(t, err)
	assert.Empty(t, resp.Chunks)
}

func TestSearch_BackendError(t *testing.T) {
	backend := &stubBackend{err: storage.ErrNotImplemented}
	s := search.NewEnhancedSearch(backend, nil, search.DefaultEnhancedSearchOptions())

	_, err := s.Search(search.SearchRequest{QueryEmbedding: []float64{0.1}, K: 3})
	require.Error(t, err)
}
