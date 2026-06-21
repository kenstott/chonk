package models_test

import (
	"testing"

	"github.com/kennethstott/chonk/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// DocumentChunk — source derivation (mirrors Python __post_init__ tests)
// ---------------------------------------------------------------------------

func TestDocumentChunk_DefaultSource(t *testing.T) {
	c := models.NewDocumentChunk("doc", "hello world")
	assert.Equal(t, "document", c.Source)
	assert.Equal(t, "document", c.ChunkType)
}

func TestDocumentChunk_SchemaTypes(t *testing.T) {
	for _, ct := range []string{"db_table", "db_column", "db_schema"} {
		c := models.NewDocumentChunk("doc", "text").WithChunkType(ct)
		assert.Equal(t, "schema", c.Source, "ChunkType=%s", ct)
	}
}

func TestDocumentChunk_APITypes(t *testing.T) {
	for _, ct := range []string{
		"api_endpoint", "api_operation", "api_parameter",
		"graphql_query", "graphql_mutation", "graphql_type", "graphql_field",
	} {
		c := models.NewDocumentChunk("doc", "text").WithChunkType(ct)
		assert.Equal(t, "api", c.Source, "ChunkType=%s", ct)
	}
}

func TestDocumentChunk_CommunityType(t *testing.T) {
	c := models.NewDocumentChunk("doc", "text").WithChunkType("community_summary")
	assert.Equal(t, "community", c.Source)
}

func TestDocumentChunk_SectionDefaultsToEmpty(t *testing.T) {
	c := models.NewDocumentChunk("doc", "text")
	require.NotNil(t, c.Section)
	assert.Empty(t, c.Section)
}

func TestDocumentChunk_NilOptionalFields(t *testing.T) {
	c := models.NewDocumentChunk("doc", "text")
	assert.Nil(t, c.SourceOffset)
	assert.Nil(t, c.SourceLength)
	assert.Nil(t, c.EmbeddingContent)
	assert.Nil(t, c.Breadcrumb)
	assert.Nil(t, c.RenderedSource)
}

// ---------------------------------------------------------------------------
// Entity
// ---------------------------------------------------------------------------

func TestEntity_DefaultType(t *testing.T) {
	e := models.NewEntity("e1", "OpenAI", "OpenAI")
	assert.Equal(t, "concept", e.EntityType)
	assert.Empty(t, e.Aliases)
}

// ---------------------------------------------------------------------------
// EntityAssociation
// ---------------------------------------------------------------------------

func TestEntityAssociation_DefaultChunkLength(t *testing.T) {
	ea := models.NewEntityAssociation("e1", "c1", 3, []int{0, 10}, 0.75)
	assert.Equal(t, 1, ea.ChunkLength)
}

// ---------------------------------------------------------------------------
// ScoredChunk
// ---------------------------------------------------------------------------

func TestScoredChunk_NilOptionals(t *testing.T) {
	sc := models.ScoredChunk{
		ChunkID:    "c1",
		Score:      0.9,
		Provenance: "seed",
	}
	assert.Nil(t, sc.LinkedBy)
	assert.Nil(t, sc.Cluster)
	assert.Nil(t, sc.Embedding)
}

// ---------------------------------------------------------------------------
// LoadedDocument
// ---------------------------------------------------------------------------

func TestLoadedDocument_Defaults(t *testing.T) {
	d := models.NewLoadedDocument("README", "# Hello", "markdown")
	assert.Equal(t, "markdown", d.DocFormat)
	assert.Empty(t, d.SourceURI)
	require.NotNil(t, d.Sections)
	assert.Empty(t, d.Sections)
}
