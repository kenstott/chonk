// Package models defines the core data types for chonk.
//
// MIGRATION MARKER: Ported from chonk/models.py
// Python source: chonk/models.py (129 LOC)
// Status: COMPLETE — all Python dataclasses translated to Go structs.
//
// Translation notes:
//   - Python @dataclass → Go struct with constructor function (New*)
//   - Python field(default_factory=list) → Go nil slice (allocate on first use)
//   - Python __post_init__ logic for DocumentChunk.source → replicated in NewDocumentChunk
//   - Python Any fields (source_detail) → map[string]interface{}
package models

// EntityAssociation links a named entity to a document chunk with scoring metadata.
// Python source: models.EntityAssociation
type EntityAssociation struct {
	EntityID    string
	ChunkID     string
	Frequency   int
	Positions   []int
	Score       float64
	ChunkLength int // character length of the chunk; defaults to 1
}

// NewEntityAssociation creates an EntityAssociation with safe defaults.
func NewEntityAssociation(entityID, chunkID string, frequency int, positions []int, score float64) EntityAssociation {
	return EntityAssociation{
		EntityID:    entityID,
		ChunkID:     chunkID,
		Frequency:   frequency,
		Positions:   positions,
		Score:       score,
		ChunkLength: 1,
	}
}

// Entity is a named entity from the vocabulary.
// Python source: models.Entity
type Entity struct {
	EntityID    string
	Name        string
	DisplayName string
	EntityType  string // default: "concept"
	Aliases     []string
}

// NewEntity creates an Entity with default type "concept".
func NewEntity(entityID, name, displayName string) Entity {
	return Entity{
		EntityID:    entityID,
		Name:        name,
		DisplayName: displayName,
		EntityType:  "concept",
		Aliases:     []string{},
	}
}

// ClusterRecord is a cluster of entities with a cohesion score.
// Python source: models.ClusterRecord
type ClusterRecord struct {
	ClusterID    string
	Entities     []string
	CohesionScore float64
}

// ScoredChunk is a chunk returned from enhanced search with composite score and provenance.
// Python source: models.ScoredChunk
//
// Provenance values: "seed" | "structural" | "entity_adjacent" | "cluster_adjacent"
type ScoredChunk struct {
	ChunkID   string
	Chunk     DocumentChunk
	Score     float64
	Provenance string
	LinkedBy  *string  // entity_id that linked this chunk (nil if not entity-linked)
	Cluster   *string  // cluster_id for cluster-adjacent chunks (nil otherwise)
	Embedding []float64 // cached embedding for scoring; nil unless populated
}

// DocumentChunk is a unit of document content ready for embedding and search.
// Python source: models.DocumentChunk
//
// ChunkType values: "document" | "db_table" | "db_column" | "api_endpoint" |
// "db_schema" | "community_summary" | "graphql_query" | "graphql_mutation" |
// "graphql_type" | "graphql_field"
type DocumentChunk struct {
	DocumentName          string
	Content               string
	Section               []string          // ordered list of section headings
	ChunkIndex            int
	SourceOffset          *int              // nil if not set
	SourceLength          *int              // nil if not set
	SourceDetail          map[string]interface{} // arbitrary format-specific metadata
	EmbeddingContent      *string           // nil until enrich_chunks sets it
	ChunkType             string            // default: "document"
	Breadcrumb            *string           // nil until context enrichment
	ParagraphContinuation bool
	Source                string            // "document" | "schema" | "api" | "community"
	RenderedSource        *string           // per-record rendered markdown; nil unless set
}

// NewDocumentChunk creates a DocumentChunk and derives Source from ChunkType,
// replicating the Python __post_init__ logic.
func NewDocumentChunk(documentName, content string) DocumentChunk {
	c := DocumentChunk{
		DocumentName: documentName,
		Content:      content,
		Section:      []string{},
		ChunkType:    "document",
	}
	c.Source = deriveSource(c.ChunkType)
	return c
}

// WithChunkType sets ChunkType and re-derives Source.
func (c DocumentChunk) WithChunkType(ct string) DocumentChunk {
	c.ChunkType = ct
	c.Source = deriveSource(ct)
	return c
}

// deriveSource mirrors DocumentChunk.__post_init__ source derivation in Python.
func deriveSource(chunkType string) string {
	switch chunkType {
	case "db_table", "db_column", "db_schema":
		return "schema"
	case "api_endpoint", "api_operation", "api_parameter",
		"graphql_query", "graphql_mutation", "graphql_type", "graphql_field":
		return "api"
	case "community_summary":
		return "community"
	default:
		return "document"
	}
}

// LoadedDocument is a document fetched from a source and ready for chunking.
// Python source: models.LoadedDocument
type LoadedDocument struct {
	Name      string
	Content   string
	DocFormat string   // "pdf" | "docx" | "markdown" | "text" | "html" | "csv" | etc.
	SourceURI string
	Sections  []string // ordered section headings found in the document
}

// NewLoadedDocument creates a LoadedDocument with safe defaults.
func NewLoadedDocument(name, content, docFormat string) LoadedDocument {
	return LoadedDocument{
		Name:      name,
		Content:   content,
		DocFormat: docFormat,
		Sections:  []string{},
	}
}
