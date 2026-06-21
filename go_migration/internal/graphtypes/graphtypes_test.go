package graphtypes_test

import (
	"testing"

	"github.com/kennethstott/chonk/internal/graphtypes"
	"github.com/stretchr/testify/assert"
)

func TestContextGraph_EdgesFrom(t *testing.T) {
	g := &graphtypes.ContextGraph{
		Edges: []graphtypes.ContextEdge{
			{FromChunkID: "a", ToChunkID: "b", Relation: "next", Weight: 1.0},
			{FromChunkID: "a", ToChunkID: "c", Relation: "entity_co", Weight: 0.5},
			{FromChunkID: "b", ToChunkID: "c", Relation: "next", Weight: 1.0},
		},
	}
	edges := g.EdgesFrom("a")
	assert.Len(t, edges, 2)
}

func TestContextGraph_EdgesTo(t *testing.T) {
	g := &graphtypes.ContextGraph{
		Edges: []graphtypes.ContextEdge{
			{FromChunkID: "a", ToChunkID: "b", Relation: "next", Weight: 1.0},
			{FromChunkID: "c", ToChunkID: "b", Relation: "prev", Weight: 1.0},
		},
	}
	edges := g.EdgesTo("b")
	assert.Len(t, edges, 2)
}

func TestContextGraph_Stats(t *testing.T) {
	g := &graphtypes.ContextGraph{
		Edges: []graphtypes.ContextEdge{
			{FromChunkID: "a", ToChunkID: "b", Relation: "next", Weight: 1.0},
			{FromChunkID: "b", ToChunkID: "c", Relation: "next", Weight: 1.0},
		},
	}
	stats := g.Stats()
	assert.Equal(t, 3, stats.NodeCount)
	assert.Equal(t, 2, stats.EdgeCount)
	assert.InDelta(t, 0.333, stats.Density, 0.01)
}

func TestContextGraph_EmptyStats(t *testing.T) {
	g := &graphtypes.ContextGraph{}
	stats := g.Stats()
	assert.Equal(t, 0, stats.NodeCount)
	assert.Equal(t, 0, stats.EdgeCount)
	assert.Equal(t, 0.0, stats.Density)
}
