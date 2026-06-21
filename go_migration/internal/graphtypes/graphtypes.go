// Package graphtypes is the dependency-cycle-breaking leaf package.
//
// MIGRATION MARKER: NEW PACKAGE — does not exist in Python source.
// ADR: aipa_test_mcp_server/future_state_architecture/code-conversion.md
// Purpose: Breaks the import cycle  graph → cluster → ner → graph
//          that exists in the Python codebase (resolved via lazy function-body
//          imports in Python; Go rejects circular imports at compile time).
//
// Layer 2 (G2) in the 8-layer build order. All of graph, cluster, ner, and
// community import graphtypes instead of each other.
//
// Contents: Only struct definitions — zero logic, zero external imports.
package graphtypes

// ContextEdge represents a directed edge in the context graph.
// Python source: chonk/graph/_context_graph.py  ContextEdge
type ContextEdge struct {
	FromChunkID string
	ToChunkID   string
	Relation    string  // "next" | "prev" | "parent" | "entity_co" | "cluster_co"
	Weight      float64
}

// ContextGraphStats holds statistics for a built context graph.
// Python source: chonk/graph/_context_graph.py  ContextGraphStats
type ContextGraphStats struct {
	NodeCount int
	EdgeCount int
	Density   float64 // edges / (nodes * (nodes-1))
}

// ContextGraph is a simple adjacency-list graph for chunk context relationships.
// Python source: chonk/graph/_context_graph.py  ContextGraph
type ContextGraph struct {
	Edges []ContextEdge
	// adjacency index built lazily; nil until first lookup
	fromIndex map[string][]ContextEdge
	toIndex   map[string][]ContextEdge
}

// EdgesFrom returns all edges leaving chunkID.
func (g *ContextGraph) EdgesFrom(chunkID string) []ContextEdge {
	if g.fromIndex == nil {
		g.buildIndex()
	}
	return g.fromIndex[chunkID]
}

// EdgesTo returns all edges arriving at chunkID.
func (g *ContextGraph) EdgesTo(chunkID string) []ContextEdge {
	if g.toIndex == nil {
		g.buildIndex()
	}
	return g.toIndex[chunkID]
}

// Stats returns basic graph statistics.
func (g *ContextGraph) Stats() ContextGraphStats {
	n := g.nodeCount()
	e := len(g.Edges)
	var density float64
	if n > 1 {
		density = float64(e) / float64(n*(n-1))
	}
	return ContextGraphStats{NodeCount: n, EdgeCount: e, Density: density}
}

func (g *ContextGraph) buildIndex() {
	g.fromIndex = make(map[string][]ContextEdge, len(g.Edges))
	g.toIndex = make(map[string][]ContextEdge, len(g.Edges))
	for _, e := range g.Edges {
		g.fromIndex[e.FromChunkID] = append(g.fromIndex[e.FromChunkID], e)
		g.toIndex[e.ToChunkID] = append(g.toIndex[e.ToChunkID], e)
	}
}

func (g *ContextGraph) nodeCount() int {
	seen := make(map[string]struct{}, len(g.Edges)*2)
	for _, e := range g.Edges {
		seen[e.FromChunkID] = struct{}{}
		seen[e.ToChunkID] = struct{}{}
	}
	return len(seen)
}
