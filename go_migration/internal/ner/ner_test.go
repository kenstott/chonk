package ner_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kennethstott/chonk/internal/ner"
	"github.com/kennethstott/chonk/internal/together"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// EntityIndex
// ---------------------------------------------------------------------------

func TestEntityIndex_IndexAndRetrieve(t *testing.T) {
	idx := ner.NewEntityIndex()
	matches := []ner.EntityMatch{
		{EntityID: "org_openai", Name: "OpenAI", EntityType: "organization", Start: 0, End: 6},
		{EntityID: "person_sam_altman", Name: "Sam Altman", EntityType: "person", Start: 20, End: 30},
	}
	idx.IndexChunk("chunk-1", matches)

	assert.Len(t, idx.EntitiesForChunk("chunk-1"), 2)
	assert.Contains(t, idx.ChunksForEntity("org_openai"), "chunk-1")
	assert.Contains(t, idx.ChunksForEntity("person_sam_altman"), "chunk-1")
}

func TestEntityIndex_MultiChunkReverseIndex(t *testing.T) {
	idx := ner.NewEntityIndex()
	m := ner.EntityMatch{EntityID: "org_google", Name: "Google", EntityType: "organization"}
	idx.IndexChunk("c1", []ner.EntityMatch{m})
	idx.IndexChunk("c2", []ner.EntityMatch{m})
	chunks := idx.ChunksForEntity("org_google")
	assert.Len(t, chunks, 2)
}

func TestEntityIndex_MissingChunk(t *testing.T) {
	idx := ner.NewEntityIndex()
	assert.Nil(t, idx.EntitiesForChunk("nonexistent"))
}

// ---------------------------------------------------------------------------
// MergeMatches
// ---------------------------------------------------------------------------

func TestMergeMatches_NoOverlap(t *testing.T) {
	matches := []ner.EntityMatch{
		{EntityID: "a", Start: 0, End: 5, Score: 0.9},
		{EntityID: "b", Start: 10, End: 15, Score: 0.8},
	}
	merged := ner.MergeMatches(matches)
	assert.Len(t, merged, 2)
}

func TestMergeMatches_Overlap_KeepsHigherScore(t *testing.T) {
	matches := []ner.EntityMatch{
		{EntityID: "a", Start: 0, End: 10, Score: 0.7},
		{EntityID: "b", Start: 5, End: 15, Score: 0.9},
	}
	merged := ner.MergeMatches(matches)
	assert.Len(t, merged, 1)
	assert.Equal(t, "b", merged[0].EntityID)
}

func TestMergeMatches_Empty(t *testing.T) {
	assert.Nil(t, ner.MergeMatches(nil))
	assert.Nil(t, ner.MergeMatches([]ner.EntityMatch{}))
}

// ---------------------------------------------------------------------------
// NormalizeSchemaTermm
// ---------------------------------------------------------------------------

func TestNormalizeSchemaTermm(t *testing.T) {
	cases := []struct {
		input, want string
	}{
		{"firstName", "first name"},
		{"ACCOUNT_ID", "account id"},
		{"user_name", "user name"},
		{"productSKU", "product s k u"},
		{"id", "id"},
	}
	for _, tc := range cases {
		assert.Equal(t, tc.want, ner.NormalizeSchemaTermm(tc.input), "input=%s", tc.input)
	}
}

// ---------------------------------------------------------------------------
// NerPipeline (with mock Together.ai server)
// ---------------------------------------------------------------------------

func mockTogetherServer(t *testing.T, entities []map[string]interface{}) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		payload := map[string]interface{}{
			"entities": entities,
		}
		body, _ := json.Marshal(payload)
		resp := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"content": string(body)}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

func TestNerPipeline_Extract_Success(t *testing.T) {
	srv := mockTogetherServer(t, []map[string]interface{}{
		{"name": "OpenAI", "entity_type": "organization", "start": 0, "end": 6},
	})
	defer srv.Close()

	client := together.New("test-key",
		together.WithBaseURL(srv.URL),
		together.WithHTTPClient(srv.Client()),
	)
	pipeline := ner.NewNerPipeline(client, nil)
	matches, err := pipeline.Extract(context.Background(), "OpenAI released GPT-4.")
	require.NoError(t, err)
	require.Len(t, matches, 1)
	assert.Equal(t, "OpenAI", matches[0].Name)
	assert.Equal(t, "organization", matches[0].EntityType)
}

func TestNerPipeline_Extract_EmptyText(t *testing.T) {
	client := together.New("test-key")
	pipeline := ner.NewNerPipeline(client, nil)
	matches, err := pipeline.Extract(context.Background(), "")
	require.NoError(t, err)
	assert.Nil(t, matches)
}

func TestNerPipeline_Extract_InvalidOffsets_Fallback(t *testing.T) {
	// API returns bad offsets — pipeline should fallback to string search
	srv := mockTogetherServer(t, []map[string]interface{}{
		{"name": "Google", "entity_type": "organization", "start": 999, "end": 1000},
	})
	defer srv.Close()

	client := together.New("test-key",
		together.WithBaseURL(srv.URL),
		together.WithHTTPClient(srv.Client()),
	)
	pipeline := ner.NewNerPipeline(client, nil)
	matches, err := pipeline.Extract(context.Background(), "Google is a tech company.")
	require.NoError(t, err)
	require.Len(t, matches, 1)
	assert.Equal(t, 0, matches[0].Start)
	assert.Equal(t, 6, matches[0].End)
}
