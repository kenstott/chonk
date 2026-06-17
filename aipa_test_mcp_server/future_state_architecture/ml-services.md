# ML Services — Together.ai Integration

## Principle

All machine-learning inference that the Python codebase runs in-process is replaced
by HTTP calls to the Together.ai API. No model weights are downloaded to the host.
No GPU or ML runtime is required.

## Libraries Replaced

| Python library | Used for | Replacement |
|---|---|---|
| `sentence-transformers` | Text embedding | Together.ai Embeddings API |
| `torch` | Tensor ops (transitive dep of sentence-transformers) | Eliminated |
| `spacy` | NER tagging | Together.ai Chat API (structured output) |
| `scikit-learn` | Agglomerative clustering, DBSCAN | Pure Go implementation |
| `igraph` | Graph construction for Leiden/Louvain | Pure Go (`yourbasic/graph` or custom) |
| `leidenalg` | Community detection partition | Pure Go Louvain implementation |

`scikit-learn`, `igraph`, and `leidenalg` are replaced with pure Go because their
algorithms (cosine similarity matrix, graph adjacency, Louvain) are simple enough
to implement without a large dependency.

## Together.ai Client (Go)

A single thin client wraps the Together.ai REST API. All service calls go through it.

```go
package together

type Client struct {
    APIKey  string
    BaseURL string        // default: https://api.together.xyz/v1
    HTTP    *http.Client
}

func NewClient(apiKey string) *Client
func (c *Client) Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error)
func (c *Client) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
```

The API key is read from `TOGETHER_API_KEY` environment variable. The client is
created once at process start and injected into each service that needs it.

## Embedding Service

**Python location:** `sentence_transformers.SentenceTransformer.encode(...)` called
in `indexer.py`, `ingest.py`, `community/_build.py`, `search/_enhanced.py`.

**Go replacement:** `together.Client.Embed()`

```go
type EmbedRequest struct {
    Model string   // e.g. "togethercomputer/m2-bert-80M-8k-retrieval"
    Input []string // batch of texts
}

type EmbedResponse struct {
    Data []struct {
        Embedding []float32
        Index     int
    }
}
```

**Embedding model:** `togethercomputer/m2-bert-80M-8k-retrieval` (1024-dim) as
default, matching the current Python default of `BAAI/bge-large-en-v1.5` (also
1024-dim). Override via `CHONK_EMBED_MODEL` env var.

**Batching:** Together's embedding endpoint accepts up to 512 inputs per request.
The Go `Indexer` sends chunks in batches of 256 (matching the Python default),
collecting results and assembling the `[][]float32` matrix in order.

**Call site in Go indexer:**

```go
func (idx *Indexer) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
    resp, err := idx.together.Embed(ctx, together.EmbedRequest{
        Model: idx.embedModel,
        Input: texts,
    })
    if err != nil {
        return nil, fmt.Errorf("embed batch: %w", err)
    }
    // sort by Index to preserve order
    sort.Slice(resp.Data, func(i, j int) bool {
        return resp.Data[i].Index < resp.Data[j].Index
    })
    out := make([][]float32, len(resp.Data))
    for i, d := range resp.Data {
        out[i] = d.Embedding
    }
    return out, nil
}
```

## NER Inference

**Python location:** `ner/_spacy.py` — `SpacyMatcher` uses a spacy model to tag
entities; `ner/_pipeline.py` orchestrates extraction per chunk.

**Go replacement:** Together.ai Chat API with a structured JSON prompt.

```go
func (p *NERPipeline) ExtractEntities(ctx context.Context, text string) ([]Entity, error) {
    prompt := buildNERPrompt(text, p.labelSet)
    resp, err := p.together.Chat(ctx, together.ChatRequest{
        Model: p.model,   // e.g. "meta-llama/Llama-3-8b-chat-hf"
        Messages: []Message{
            {Role: "system", Content: nerSystemPrompt},
            {Role: "user",   Content: prompt},
        },
        ResponseFormat: together.JSONSchema(entityListSchema),
    })
    // parse resp.Choices[0].Message.Content as []Entity JSON
}
```

**Label set:** The same label taxonomy defined in Python's `_schema_vocab.py` is
embedded as a Go constant block or a `//go:embed` JSON file.

**NER model default:** `meta-llama/Llama-3-8b-chat-hf` (fast, cheap, good for
structured extraction). Override via `CHONK_NER_MODEL` env var.

## Community Summarisation

**Python location:** `community/_summarizer.py` — calls an OpenAI-compatible LLM
to generate a short label and description for each detected community.

**Go replacement:** Together.ai Chat API, same prompt template.

```go
func (s *Summarizer) Summarize(ctx context.Context, topTerms []string) (CommunityLabel, error) {
    resp, err := s.together.Chat(ctx, together.ChatRequest{
        Model:    s.model,
        Messages: buildSummarizeMessages(topTerms),
    })
    // parse label + description from resp
}
```

## SVO / Graph Triple Extraction

**Python location:** `graph/_llm.py`, `graph/_svo.py` — LLM-backed subject-verb-object
extraction, previously using an OpenAI-compatible endpoint.

**Go replacement:** Together.ai Chat API. No change to prompt structure; only the
client changes.

```go
func (e *SVOExtractor) Extract(ctx context.Context, text string) ([]Triple, error) {
    resp, err := e.together.Chat(ctx, together.ChatRequest{
        Model:          e.model,
        Messages:       buildSVOMessages(text),
        ResponseFormat: together.JSONSchema(tripleListSchema),
    })
    // parse []Triple from structured output
}
```

## Answer Generation

**Python location:** `generation/_answer.py` — `AnswerGenerator` calls an LLM with
retrieved context chunks to produce a final answer.

**Go replacement:** Together.ai Chat API.

```go
func (g *AnswerGenerator) Generate(
    ctx context.Context,
    question string,
    chunks []models.DocumentChunk,
) (string, error) {
    prompt := g.promptBuilder.Build(question, chunks)
    resp, err := g.together.Chat(ctx, together.ChatRequest{
        Model:    g.model,
        Messages: []Message{{Role: "user", Content: prompt}},
    })
    return resp.Choices[0].Message.Content, nil
}
```

## Configuration

All Together.ai settings are read from environment variables, consistent with the
Python fine-tuning script.

| Env var | Default | Purpose |
|---|---|---|
| `TOGETHER_API_KEY` | (required) | API key |
| `CHONK_EMBED_MODEL` | `togethercomputer/m2-bert-80M-8k-retrieval` | Embedding model |
| `CHONK_NER_MODEL` | `meta-llama/Llama-3-8b-chat-hf` | NER extraction model |
| `CHONK_CHAT_MODEL` | `meta-llama/Llama-3-70b-chat-hf` | SVO / summarise / answer model |
| `CHONK_EMBED_BATCH_SIZE` | `256` | Texts per embedding request |

## Error Handling

- All Together.ai calls are wrapped in retry logic with exponential back-off
  (max 3 retries, initial delay 500ms).
- Rate-limit responses (HTTP 429) are retried after the `Retry-After` header
  delay if present.
- A failed embedding batch surfaces as an error on the `Indexer`; the caller
  decides whether to abort or skip the batch (same as Python's `on_error` callback).
- NER and SVO failures are non-fatal by default: the chunk is stored without
  entity/graph annotations rather than blocking ingestion.
