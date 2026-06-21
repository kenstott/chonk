package together_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kennethstott/chonk/internal/together"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockServer creates a test server returning the given JSON payload.
func mockServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

// ---------------------------------------------------------------------------
// EmbedTexts
// ---------------------------------------------------------------------------

func TestEmbedTexts_Success(t *testing.T) {
	srv := mockServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/embeddings", r.URL.Path)
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		resp := map[string]interface{}{
			"data": []map[string]interface{}{
				{"index": 0, "embedding": []float64{0.1, 0.2, 0.3}},
				{"index": 1, "embedding": []float64{0.4, 0.5, 0.6}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	c := together.New("test-key",
		together.WithBaseURL(srv.URL),
		together.WithHTTPClient(srv.Client()),
	)
	embeddings, err := c.EmbedTexts(context.Background(), []string{"hello", "world"})
	require.NoError(t, err)
	require.Len(t, embeddings, 2)
	assert.InDeltaSlice(t, []float64{0.1, 0.2, 0.3}, embeddings[0], 1e-9)
	assert.InDeltaSlice(t, []float64{0.4, 0.5, 0.6}, embeddings[1], 1e-9)
}

func TestEmbedTexts_Empty(t *testing.T) {
	c := together.New("test-key")
	embeddings, err := c.EmbedTexts(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, embeddings)
}

func TestEmbedTexts_HTTP4xx(t *testing.T) {
	srv := mockServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": "invalid api key"}`))
	})
	c := together.New("bad-key",
		together.WithBaseURL(srv.URL),
		together.WithHTTPClient(srv.Client()),
	)
	_, err := c.EmbedTexts(context.Background(), []string{"hello"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "401")
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

func TestChat_Success(t *testing.T) {
	srv := mockServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		resp := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"content": "The answer is 42."}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	c := together.New("test-key",
		together.WithBaseURL(srv.URL),
		together.WithHTTPClient(srv.Client()),
	)
	msgs := []together.Message{
		{Role: "user", Content: "What is the meaning of life?"},
	}
	reply, err := c.Chat(context.Background(), msgs, nil)
	require.NoError(t, err)
	assert.Equal(t, "The answer is 42.", reply)
}

func TestChat_EmptyChoices(t *testing.T) {
	srv := mockServer(t, func(w http.ResponseWriter, _ *http.Request) {
		resp := map[string]interface{}{"choices": []interface{}{}}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	c := together.New("test-key",
		together.WithBaseURL(srv.URL),
		together.WithHTTPClient(srv.Client()),
	)
	_, err := c.Chat(context.Background(), []together.Message{{Role: "user", Content: "hi"}}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty choices")
}

func TestChat_WithOptions(t *testing.T) {
	var gotBody map[string]interface{}
	srv := mockServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		resp := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"content": "{}"}},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	})

	c := together.New("key",
		together.WithBaseURL(srv.URL),
		together.WithHTTPClient(srv.Client()),
	)
	opts := &together.ChatOptions{Model: "gpt-4o", MaxTokens: 512, JSONMode: true}
	_, err := c.Chat(context.Background(), []together.Message{{Role: "user", Content: "q"}}, opts)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", gotBody["model"])
	assert.Equal(t, float64(512), gotBody["max_tokens"])
	assert.NotNil(t, gotBody["response_format"])
}
