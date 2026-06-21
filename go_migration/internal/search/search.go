// Package search provides the 4-dimensional cohort assembler (EnhancedSearch).
//
// MIGRATION MARKER: Ported from chonk/search/ sub-package.
// Python sources:
//   - chonk/search/_enhanced.py         (893 LOC) — EnhancedSearch
//   - chonk/search/_enhanced_scoring.py           — _ScoringMixin (MMR, composite)
//   - chonk/search/_enhanced_graph.py             — _GraphMixin (graph expansion)
//   - chonk/search/_enhanced_support.py           — RetrievalTrace, helpers
//
// Status: Interface + RetrievalTrace COMPLETE; full 4-lane implementation is G6 sprint.
//
// Translation notes:
//   - Python multiple inheritance (_GraphMixin, _ScoringMixin) → Go struct composition
//   - Python lambda_diversity, relevance_weight etc. → EnhancedSearchOptions
//   - Python Optional[Callable] redaction_filter → Go func or nil
//   - Python overload search() variants → single Search() with full params
package search

import (
	"fmt"

	"github.com/kennethstott/chonk/internal/models"
	"github.com/kennethstott/chonk/internal/ner"
	"github.com/kennethstott/chonk/internal/storage"
)

// RetrievalTrace records provenance tallies for a single search call.
// Python source: search._enhanced_support.RetrievalTrace
type RetrievalTrace struct {
	Seed            int
	Structural      int
	EntityAdjacent  int
	ClusterAdjacent int
	Total           int
}

// EnhancedSearchOptions configures the 4-lane cohort assembler.
// Python source: EnhancedSearch.__init__ kwargs
type EnhancedSearchOptions struct {
	// SeedPoolMultiplier: seed pool = k * multiplier (Python default 3)
	SeedPoolMultiplier int
	// EntityExpansionTopN: max chunks per entity (Python default 3)
	EntityExpansionTopN int
	// ClusterBudget: max cluster-adjacent candidates (Python default 2*k)
	ClusterBudget int
	// LambdaDiversity: MMR redundancy penalty weight (Python default 0.3)
	LambdaDiversity float64
	// RelevanceWeight: composite score relevance weight (Python default 0.5)
	RelevanceWeight float64
	// PriorityWeight: composite score source priority weight (Python default 0.2)
	PriorityWeight float64
	// CoverageWeight: composite marginal coverage weight (Python default 0.3)
	CoverageWeight float64
	// StructuralExpansion: enable next/prev/parent expansion (default true)
	StructuralExpansion bool
	// EntityExpansion: enable entity adjacency expansion (default true)
	EntityExpansion bool
	// ClusterExpansion: enable cluster adjacency expansion (default true)
	ClusterExpansion bool
}

// DefaultEnhancedSearchOptions returns the defaults matching Python's EnhancedSearch.
func DefaultEnhancedSearchOptions() EnhancedSearchOptions {
	return EnhancedSearchOptions{
		SeedPoolMultiplier:  3,
		EntityExpansionTopN: 3,
		LambdaDiversity:     0.3,
		RelevanceWeight:     0.5,
		PriorityWeight:      0.2,
		CoverageWeight:      0.3,
		StructuralExpansion: true,
		EntityExpansion:     true,
		ClusterExpansion:    true,
	}
}

// SearchRequest bundles parameters for a single Search call.
type SearchRequest struct {
	QueryEmbedding []float64
	K              int     // number of results to return
	QueryText      string  // optional — triggers hybrid BM25+vector search
	Namespaces     []string
	ChunkTypes     []string
}

// SearchResponse bundles results + trace.
type SearchResponse struct {
	Chunks []models.ScoredChunk
	Trace  RetrievalTrace
}

// EnhancedSearch is the 4-dimensional cohort assembler.
// Python source: search._enhanced.EnhancedSearch
type EnhancedSearch struct {
	store       storage.VectorBackend
	entityIndex *ner.EntityIndex // nil → entity expansion disabled
	opts        EnhancedSearchOptions
	// chunkFilter: optional post-search filter applied to results
	chunkFilter func([]models.ScoredChunk) []models.ScoredChunk
}

// NewEnhancedSearch creates an EnhancedSearch.
// entityIndex may be nil (disables entity + cluster expansion).
func NewEnhancedSearch(
	store storage.VectorBackend,
	entityIndex *ner.EntityIndex,
	opts EnhancedSearchOptions,
) *EnhancedSearch {
	if opts.SeedPoolMultiplier == 0 {
		opts = DefaultEnhancedSearchOptions()
	}
	return &EnhancedSearch{
		store:       store,
		entityIndex: entityIndex,
		opts:        opts,
	}
}

// WithChunkFilter attaches an optional post-search chunk filter.
// Python source: EnhancedSearch(chunk_filter=…)
func (s *EnhancedSearch) WithChunkFilter(f func([]models.ScoredChunk) []models.ScoredChunk) {
	s.chunkFilter = f
}

// Search performs the 4-lane retrieval and returns scored chunks.
// Python source: search._enhanced.EnhancedSearch.search()
//
// MIGRATION STUB — G6 sprint.
// Current implementation: pure vector seed only (no structural/entity/cluster).
// Full 4-lane assembly (structural, entity, cluster) + MMR scoring is TODO.
func (s *EnhancedSearch) Search(req SearchRequest) (SearchResponse, error) {
	if req.K <= 0 {
		req.K = 5
	}
	seedPool := req.K * s.opts.SeedPoolMultiplier
	if seedPool == 0 {
		seedPool = req.K * 3
	}

	// ── Lane 1: Seed (vector similarity) ─────────────────────────────────────
	rawResults, err := s.store.Search(
		req.QueryEmbedding,
		seedPool,
		req.QueryText,
		true,
		req.Namespaces,
		req.ChunkTypes,
		nil,
		"",
	)
	if err != nil {
		return SearchResponse{}, fmt.Errorf("enhanced search: seed lane: %w", err)
	}

	scored := make([]models.ScoredChunk, 0, len(rawResults))
	for _, r := range rawResults {
		scored = append(scored, models.ScoredChunk{
			ChunkID:    r.ChunkID,
			Chunk:      r.Chunk,
			Score:      r.Score,
			Provenance: "seed",
		})
	}

	// ── Lane 2: Structural expansion ─────────────────────────────────────────
	// TODO(G6): adjacency expansion via chunk_index ± 1 lookups

	// ── Lane 3: Entity adjacency ─────────────────────────────────────────────
	// TODO(G6): entity index lookup + expansion

	// ── Lane 4: Cluster adjacency ────────────────────────────────────────────
	// TODO(G6): cluster map lookup + expansion

	// ── MMR scoring + final truncation ───────────────────────────────────────
	// TODO(G6): apply MMR with lambda_diversity, relevance/priority/coverage weights
	if len(scored) > req.K {
		scored = scored[:req.K]
	}

	// ── Optional chunk filter ─────────────────────────────────────────────────
	if s.chunkFilter != nil {
		scored = s.chunkFilter(scored)
	}

	trace := RetrievalTrace{
		Seed:  len(scored),
		Total: len(scored),
	}
	return SearchResponse{Chunks: scored, Trace: trace}, nil
}
