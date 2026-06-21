package ingest_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/kennethstott/chonk/internal/ingest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// ---------------------------------------------------------------------------
// LoadConfig
// ---------------------------------------------------------------------------

func TestLoadConfig_Defaults(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	// Minimal config — all optional fields use defaults
	require.NoError(t, os.WriteFile(cfgPath, []byte("store:\n  path: \":memory:\"\n"), 0600))

	cfg, err := ingest.LoadConfig(cfgPath)
	require.NoError(t, err)
	assert.Equal(t, "BAAI/bge-large-en-v1.5", cfg.Embed.Model)
	assert.Equal(t, 256, cfg.Embed.BatchSize)
	assert.Equal(t, 1100, cfg.Loader.MinChunkSize)
	assert.Equal(t, 2200, cfg.Loader.MaxChunkSize)
	assert.Equal(t, 1024, cfg.Store.EmbeddingDim)
}

func TestLoadConfig_FullConfig(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	cfg := map[string]interface{}{
		"store": map[string]interface{}{
			"path":          ":memory:",
			"embedding_dim": 768,
		},
		"embed": map[string]interface{}{
			"model":      "BAAI/bge-base-en-v1.5",
			"batch_size": 128,
			"api_key":    "sk-test",
		},
		"loader": map[string]interface{}{
			"min_chunk_size": 600,
			"max_chunk_size": 1500,
			"enrich_context": true,
		},
	}
	data, err := yaml.Marshal(cfg)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(cfgPath, data, 0600))

	loaded, err := ingest.LoadConfig(cfgPath)
	require.NoError(t, err)
	assert.Equal(t, "BAAI/bge-base-en-v1.5", loaded.Embed.Model)
	assert.Equal(t, 128, loaded.Embed.BatchSize)
	assert.Equal(t, 768, loaded.Store.EmbeddingDim)
	assert.Equal(t, 600, loaded.Loader.MinChunkSize)
	assert.Equal(t, "sk-test", loaded.Embed.APIKey)
}

func TestLoadConfig_MissingFile(t *testing.T) {
	_, err := ingest.LoadConfig("/nonexistent/config.yaml")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "config.yaml")
}

func TestLoadConfig_InvalidYAML(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "bad.yaml")
	require.NoError(t, os.WriteFile(cfgPath, []byte("invalid: yaml: [unclosed"), 0600))
	_, err := ingest.LoadConfig(cfgPath)
	require.Error(t, err)
}

// ---------------------------------------------------------------------------
// Build — no API key → error
// ---------------------------------------------------------------------------

func TestBuild_NoAPIKey_ReturnsError(t *testing.T) {
	t.Setenv("TOGETHER_API_KEY", "") // ensure env var is unset for this test
	cfg := ingest.ChonkConfig{
		Store: ingest.StoreConfig{Path: ":memory:"},
	}
	_, err := ingest.Build(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "API key")
}
