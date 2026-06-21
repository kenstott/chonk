// Package together provides an HTTP client for the Together.ai API.
//
// MIGRATION MARKER: NEW PACKAGE — replaces three Python in-process ML libraries:
//   - sentence-transformers + torch  → EmbedTexts()
//   - spacy + thinc                  → ExtractEntities()
//   - OpenAI-compatible LLM client   → Chat()
//
// ADR: aipa_test_mcp_server/future_state_architecture/ml-services.md
// Status: COMPLETE (interface + HTTP implementation)
//
// All ML inference (embedding, NER, LLM generation, SVO extraction,
// community summarisation) goes through this package.
package together

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	// DefaultBaseURL is the Together.ai API base URL.
	DefaultBaseURL = "https://api.together.xyz/v1"
	// DefaultEmbedModel is the default text embedding model.
	// Python equivalent: ChonkConfig.embed.model default "BAAI/bge-large-en-v1.5"
	DefaultEmbedModel = "BAAI/bge-large-en-v1.5"
	// DefaultChatModel is the default chat completion model.
	DefaultChatModel = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

// Client is the Together.ai API client.
// Inject via constructor into NER pipeline, graph builder, community summariser,
// indexer, and generation answer generator.
type Client struct {
	apiKey    string
	baseURL   string
	embedModel string
	chatModel  string
	http      *http.Client
}

// Option is a functional option for Client.
type Option func(*Client)

// WithBaseURL sets a custom base URL (useful for testing with a mock server).
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = url }
}

// WithEmbedModel sets the embedding model name.
func WithEmbedModel(model string) Option {
	return func(c *Client) { c.embedModel = model }
}

// WithChatModel sets the chat completion model name.
func WithChatModel(model string) Option {
	return func(c *Client) { c.chatModel = model }
}

// WithHTTPClient injects a custom http.Client (for testing).
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) { c.http = hc }
}

// New creates a Together.ai client with the given API key and options.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		apiKey:     apiKey,
		baseURL:    DefaultBaseURL,
		embedModel: DefaultEmbedModel,
		chatModel:  DefaultChatModel,
		http:       &http.Client{Timeout: 60 * time.Second},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// ── Embeddings ────────────────────────────────────────────────────────────────

// EmbedTexts encodes a batch of sentences and returns one float64 slice per input.
// Python equivalent: SentenceTransformer.encode(sentences, normalize_embeddings=True)
func (c *Client) EmbedTexts(ctx context.Context, sentences []string) ([][]float64, error) {
	if len(sentences) == 0 {
		return nil, nil
	}
	reqBody := map[string]interface{}{
		"model": c.embedModel,
		"input": sentences,
	}
	var resp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := c.post(ctx, "/embeddings", reqBody, &resp); err != nil {
		return nil, fmt.Errorf("together embed: %w", err)
	}
	// Re-order by index (API may return out of order)
	out := make([][]float64, len(sentences))
	for _, d := range resp.Data {
		if d.Index < len(out) {
			out[d.Index] = d.Embedding
		}
	}
	return out, nil
}

// ── Chat completion ───────────────────────────────────────────────────────────

// Message is a single entry in a chat conversation.
type Message struct {
	Role    string `json:"role"`    // "system" | "user" | "assistant"
	Content string `json:"content"`
}

// ChatOptions holds optional parameters for Chat.
type ChatOptions struct {
	Model       string  // overrides client default when set
	MaxTokens   int     // default 1024
	Temperature float64 // default 0.7
	JSONMode    bool    // request JSON object output
}

// Chat sends a chat completion request and returns the assistant response text.
// Python equivalent: LLMClient.__call__(prompt) in generation._answer
func (c *Client) Chat(ctx context.Context, messages []Message, opts *ChatOptions) (string, error) {
	model := c.chatModel
	maxTokens := 1024
	temperature := 0.7
	if opts != nil {
		if opts.Model != "" {
			model = opts.Model
		}
		if opts.MaxTokens > 0 {
			maxTokens = opts.MaxTokens
		}
		if opts.Temperature > 0 {
			temperature = opts.Temperature
		}
	}
	reqBody := map[string]interface{}{
		"model":       model,
		"messages":    messages,
		"max_tokens":  maxTokens,
		"temperature": temperature,
	}
	if opts != nil && opts.JSONMode {
		reqBody["response_format"] = map[string]string{"type": "json_object"}
	}
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := c.post(ctx, "/chat/completions", reqBody, &resp); err != nil {
		return "", fmt.Errorf("together chat: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("together chat: empty choices")
	}
	return resp.Choices[0].Message.Content, nil
}

// ── HTTP helper ───────────────────────────────────────────────────────────────

func (c *Client) post(ctx context.Context, path string, body interface{}, out interface{}) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("http: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read body: %w", err)
	}
	if resp.StatusCode >= 400 {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(string(data), 200))
	}
	return json.Unmarshal(data, out)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
