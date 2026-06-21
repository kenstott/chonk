// Package ingest orchestrates the full pipeline: config → chunks → embeddings → store.
//
// MIGRATION MARKER: Ported from chonk/ingest.py (900 LOC) and chonk/_ingest_worker.py.
// Python surface:
//   - build(config) → Index  →  Build(cfg) → *Index
//   - Index.search(…)        →  Index.Search(…)
//   - Index.ask(…)           →  Index.Ask(…)
//
// Status: Interface + Config loading COMPLETE; full pipeline wiring is G7 sprint.
//
// Translation notes:
//   - Python config dict / YAML → Go ChonkConfig loaded from YAML via gopkg.in/yaml.v3
//   - Python sentence-transformers encode → together.Client.EmbedTexts
//   - Python Store context manager → Go Store.Close() deferred
package ingest

import (
	"context"
	"fmt"
	"os"

	"github.com/kennethstott/chonk/internal/chunking"
	"github.com/kennethstott/chonk/internal/extractors"
	"github.com/kennethstott/chonk/internal/generation"
	"github.com/kennethstott/chonk/internal/models"
	"github.com/kennethstott/chonk/internal/search"
	"github.com/kennethstott/chonk/internal/storage"
	"github.com/kennethstott/chonk/internal/together"
	"gopkg.in/yaml.v3"
)

// ── Config types ─────────────────────────────────────────────────────────────

// ChonkConfig is the root configuration object.
// Python source: chonk/_config.py  ChonkConfig
type ChonkConfig struct {
	Store      StoreConfig      `yaml:"store"`
	Embed      EmbedConfig      `yaml:"embed"`
	Loader     LoaderConfig     `yaml:"loader"`
	Index      IndexCfg         `yaml:"index"`
	Sources    []SourceConfig   `yaml:"sources"`
	Namespaces map[string]interface{} `yaml:"namespaces"`
	Search     map[string]interface{} `yaml:"search"`
}

// StoreConfig configures the vector backend.
type StoreConfig struct {
	Path         string `yaml:"path"`          // DuckDB file path or ":memory:"
	DSN          string `yaml:"dsn"`           // PostgreSQL DSN; overrides Path when set
	EmbeddingDim int    `yaml:"embedding_dim"` // default 1024
}

// EmbedConfig configures the embedding model.
// Python source: chonk/_config.py  EmbedConfig
type EmbedConfig struct {
	Model     string `yaml:"model"`      // default "BAAI/bge-large-en-v1.5"
	BatchSize int    `yaml:"batch_size"` // default 256
	APIKey    string `yaml:"api_key"`    // Together.ai API key (or TOGETHER_API_KEY env var)
}

// LoaderConfig configures chunking parameters.
// Python source: chonk/_config.py  LoaderConfig
type LoaderConfig struct {
	MinChunkSize    int      `yaml:"min_chunk_size"`   // default 1100
	MaxChunkSize    int      `yaml:"max_chunk_size"`   // default 2200
	EnrichContext   bool     `yaml:"enrich_context"`   // default true
	ExtraExtractors []string `yaml:"extra_extractors"` // e.g. ["edgar"]
}

// IndexCfg configures the NER / graph / community indexing.
// Python source: chonk/_config.py  IndexConfig
type IndexCfg struct {
	NER                  bool    `yaml:"ner"`
	Community            bool    `yaml:"community"`
	SVO                  bool    `yaml:"svo"`
	SpacyModel           string  `yaml:"spacy_model"`           // ignored — spaCy dropped
	SVOModel             string  `yaml:"svo_model"`             // default "gpt-4o-mini"
	CommunityAlpha       float64 `yaml:"community_alpha"`       // default 0.2
	CommunitySIMThreshold float64 `yaml:"community_sim_threshold"` // default 0.6
}

// SourceConfig describes a data source to ingest.
type SourceConfig struct {
	Type    string                 `yaml:"type"`    // "glob" | "json_array" | "transport"
	Path    string                 `yaml:"path"`
	Pattern string                 `yaml:"pattern"`
	Options map[string]interface{} `yaml:"options"`
}

// LoadConfig reads and parses a YAML config file.
// Python source: ChonkConfig.from_dict()
func LoadConfig(path string) (ChonkConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ChonkConfig{}, fmt.Errorf("load config: read %s: %w", path, err)
	}
	var cfg ChonkConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return ChonkConfig{}, fmt.Errorf("load config: parse YAML: %w", err)
	}
	// Apply defaults
	if cfg.Embed.Model == "" {
		cfg.Embed.Model = together.DefaultEmbedModel
	}
	if cfg.Embed.BatchSize == 0 {
		cfg.Embed.BatchSize = 256
	}
	if cfg.Loader.MinChunkSize == 0 {
		cfg.Loader.MinChunkSize = 1100
	}
	if cfg.Loader.MaxChunkSize == 0 {
		cfg.Loader.MaxChunkSize = 2200
	}
	if cfg.Store.EmbeddingDim == 0 {
		cfg.Store.EmbeddingDim = 1024
	}
	return cfg, nil
}

// ── Index ─────────────────────────────────────────────────────────────────────

// Index is the built search index returned by Build().
// Python source: chonk/ingest.py  Index
type Index struct {
	store     *storage.Store
	search    *search.EnhancedSearch
	generator *generation.AnswerGenerator
	client    *together.Client
}

// Build constructs a fully configured Index from the given config.
// Python source: ingest.build(config)
//
// MIGRATION STUB — G7 sprint.
// Current implementation: wires store, together client, and search.
// Full pipeline (NER, community, graph, lifecycle) is G5–G7.
func Build(cfg ChonkConfig) (*Index, error) {
	// API key: config → env var fallback
	apiKey := cfg.Embed.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("TOGETHER_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("build: together API key required (set embed.api_key or TOGETHER_API_KEY)")
	}

	dbPath := cfg.Store.Path
	if dbPath == "" {
		dbPath = ":memory:"
	}
	store, err := storage.NewStore(storage.StoreConfig{
		DBPath:       dbPath,
		DSN:          cfg.Store.DSN,
		EmbeddingDim: cfg.Store.EmbeddingDim,
	})
	if err != nil {
		return nil, fmt.Errorf("build: storage: %w", err)
	}

	client := together.New(apiKey, together.WithEmbedModel(cfg.Embed.Model))
	es := search.NewEnhancedSearch(store.Vector, nil, search.DefaultEnhancedSearchOptions())
	gen := generation.NewAnswerGenerator(func(prompt string) (string, error) {
		return client.Chat(context.Background(), []together.Message{
			{Role: "user", Content: prompt},
		}, nil)
	}, 4096)

	return &Index{
		store:     store,
		search:    es,
		generator: gen,
		client:    client,
	}, nil
}

// IngestBytes embeds and stores raw bytes as a named document.
// Python equivalent: loader.load_bytes() + store.add_document()
//
// MIGRATION STUB: full extractor detection + chunking + enrichment + NER is G7.
func (idx *Index) IngestBytes(ctx context.Context, name string, data []byte, docType string) error {
	reg := extractors.NewRegistry()
	ext := reg.Detect(docType)
	if ext == nil {
		ext = extractors.TextExtractor{}
	}
	text, err := ext.Extract(data, name)
	if err != nil {
		return fmt.Errorf("ingest bytes: extract: %w", err)
	}
	opts := chunking.ChunkOptions{
		MinSize: 1100, MaxSize: 2200, OverflowMargin: 0.15,
	}
	chunks := chunking.ChunkDocument(name, text, opts)
	if len(chunks) == 0 {
		return nil
	}
	contents := make([]string, len(chunks))
	for i, c := range chunks {
		contents[i] = c.Content
	}
	embeddings, err := idx.client.EmbedTexts(ctx, contents)
	if err != nil {
		return fmt.Errorf("ingest bytes: embed: %w", err)
	}
	return idx.store.AddDocument(name, "", "", chunks, embeddings, "")
}

// Search queries the index and returns scored chunks.
func (idx *Index) Search(ctx context.Context, query string, k int) ([]models.ScoredChunk, error) {
	embeddings, err := idx.client.EmbedTexts(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	resp, err := idx.search.Search(search.SearchRequest{
		QueryEmbedding: embeddings[0],
		K:              k,
		QueryText:      query,
	})
	if err != nil {
		return nil, err
	}
	return resp.Chunks, nil
}

// Ask retrieves relevant context and generates an answer.
// Python source: Index.ask()
func (idx *Index) Ask(ctx context.Context, question string, k int) (generation.Answer, error) {
	chunks, err := idx.Search(ctx, question, k)
	if err != nil {
		return generation.Answer{}, fmt.Errorf("ask: search: %w", err)
	}
	ac := generation.AnswerContext{Query: question, Chunks: chunks}
	return idx.generator.Generate(ac)
}

// Close releases all index resources.
func (idx *Index) Close() error {
	return idx.store.Close()
}
