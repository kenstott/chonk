# Unit Testing — Go Conversion

## Structure Convention

Each package gets two `_test.go` files in the same directory:

```
chonk-go/
├── models/
│   └── models_test.go
├── chunking/
│   ├── chunking_test.go            ← unit tests (always run)
│   └── chunking_integration_test.go ← integration tests (build tag guarded)
...
```

Integration test files carry a build tag on line 1:

```go
//go:build integration
```

Test functions use the standard `testing.T` receiver. Group related cases with
`t.Run`. No test framework beyond the standard library and
`github.com/stretchr/testify/assert` (optional).

**Run unit tests** (every commit, no network, no API key):

```
go test ./...
```

**Run integration tests** (merge / nightly, may need `TOGETHER_API_KEY`):

```
go test -tags integration ./...
```

**Run a single layer unit tests:**

```
go test ./chunking/... ./context/...
```

## Integration Checkpoints

Four natural checkpoints fall out of the layer structure. Each one wires real
implementations together — no stubs — and gates the next layer of work.

### Checkpoint 1 — Layer 1: Extract + Chunk

No ML, no storage, no network. First meaningful pipeline test.

```go
//go:build integration

func TestExtractAndChunk(t *testing.T) {
    data, _ := os.ReadFile("testdata/sample.md")
    ext := &extractors.MarkdownExtractor{}
    chunks, err := ext.Extract("sample.md", data)
    require.NoError(t, err)
    require.NotEmpty(t, chunks)
    // context enrichment
    enriched := context.EnrichChunks(chunks)
    for _, c := range enriched {
        assert.NotEmpty(t, c.EmbeddingContent)
    }
}
```

Pass criteria: all five extractor types produce non-empty chunks; `EnrichChunks`
sets `EmbeddingContent` on every chunk.

### Checkpoint 2 — Layer 3: DocumentLoader

First multi-package wiring test. No ML.

```go
//go:build integration

func TestDocumentLoaderMixedFormats(t *testing.T) {
    loader := loader.New(extractors.DefaultRegistry, loader.Options{EnrichContext: true})
    for _, name := range []string{"sample.txt", "sample.csv", "sample.md", "sample.html"} {
        chunks, err := loader.LoadFile("testdata/" + name)
        require.NoError(t, err, name)
        assert.NotEmpty(t, chunks, name)
    }
}
```

Pass criteria: all four formats load without error; chunks have correct
`SectionPath` for structured formats (markdown headings, CSV header row).

### Checkpoint 3 — Layer 5: Index a Directory

First full ingest pipeline. Uses stub embedder — no API key required.

```go
//go:build integration

func TestIndexDirectory(t *testing.T) {
    store := storage.NewStore(":memory:", 1024)
    idx := indexer.New(store, &stubEmbedder{dim: 1024})
    n, err := idx.IndexSource(context.Background(), map[string]any{
        "type": "directory", "uri": "testdata/docs", "extensions": []string{".txt", ".md"},
    })
    require.NoError(t, err)
    assert.Equal(t, n, store.Count())
    assert.Greater(t, store.Count(), 0)
    // FTS search should return a match
    results, err := store.Search("sample keyword", nil, 5)
    require.NoError(t, err)
    assert.NotEmpty(t, results)
}
```

Pass criteria: `store.Count()` matches returned chunk count; keyword present in
fixture file appears in FTS results.

### Checkpoint 4 — Layer 7: Full Pipeline

End-to-end with real Together.ai calls (or recorded HTTP fixtures). Requires
`TOGETHER_API_KEY` or a fixture server.

```go
//go:build integration

func TestFullPipeline(t *testing.T) {
    apiKey := os.Getenv("TOGETHER_API_KEY")
    if apiKey == "" {
        t.Skip("TOGETHER_API_KEY not set")
    }
    idx, err := ingest.Build("testdata/config.yaml")
    require.NoError(t, err)
    assert.Greater(t, idx.Count(), 0)
    results, err := idx.Search(context.Background(), "what is chunking?", 3)
    require.NoError(t, err)
    assert.NotEmpty(t, results)
}
```

Pass criteria: pipeline completes without error; search returns at least one
result from an ingested document.

To avoid requiring a live key in CI, record HTTP interactions with
`github.com/dnaeon/go-vcr` and commit the cassette under `testdata/fixtures/`.

---

## Stub Pattern

Python tests use `StubLLM`, `_StubModel`, and `FakeStore` to avoid real ML calls.
Go equivalents implement the package interface inline in `_test.go`.

```go
// together_test.go in any package that needs it
type stubTogetherClient struct {
    embedResult [][]float32
    chatResult  string
}

func (s *stubTogetherClient) Embed(_ context.Context, req together.EmbedRequest) (together.EmbedResponse, error) {
    resp := together.EmbedResponse{}
    for i := range req.Input {
        resp.Data = append(resp.Data, struct {
            Embedding []float32
            Index     int
        }{s.embedResult[i%len(s.embedResult)], i})
    }
    return resp, nil
}

func (s *stubTogetherClient) Chat(_ context.Context, _ together.ChatRequest) (together.ChatResponse, error) {
    return together.ChatResponse{Choices: []together.Choice{{Message: together.Message{Content: s.chatResult}}}}, nil
}
```

Inject stubs through constructor arguments — same pattern as Python's `SVOExtractor(StubLLM(...))`.

---

## Layer 0 — models, schema, transports

**Python template:** `tests/unit/test_context.py` (struct field preservation), `tests/integration/test_storage.py` (struct round-trip).

### `models/models_test.go`

| Python test pattern | Go equivalent |
|---|---|
| Field defaults are zero/nil | `assert chunk.ChunkIndex == 0` |
| All fields preserved through assignment | Create struct, read back each field |
| Immutability — functions return new values | Assign, modify copy, check original unchanged |

Key cases:
- `DocumentChunk` zero value is valid (empty strings, zero int)
- `ScoredChunk.Score` accepts 0.0 and 1.0 boundary values
- `Entity` with empty `Aliases` slice (not nil) — test `len(e.Aliases) == 0`

### `transports/transports_test.go`

| Python test pattern | Go equivalent |
|---|---|
| Local transport lists files by extension | Create temp dir with `t.TempDir()`, call `List()` |
| Fetch returns raw bytes | Write known content to temp file, `Fetch()` returns same bytes |
| Transport interface satisfied by all implementations | Compile-time: `var _ Transport = &LocalTransport{}` |

Key cases (port from `tests/integration/test_loader.py`):
- `LocalTransport.List` with extension filter returns only matching files
- `LocalTransport.Fetch` returns exact file bytes
- `LocalTransport.List` on empty directory returns empty slice (not error)

---

## Layer 1 — chunking, context, extractors, generation, structinfer

**Python template:** `tests/unit/test_chunking.py`, `tests/unit/test_context.py`, `tests/unit/test_extractors.py`, `tests/unit/test_generation.py`

### `chunking/chunking_test.go`

Port directly from `tests/unit/test_chunking.py`. All cases map 1:1.

| Python case | Go case |
|---|---|
| `is_table_line("| col1 | col2 |")` → true | `IsTableLine("| col1 | col2 |")` → true |
| `is_list_line("- item")` → true | `IsListLine("- item")` → true |
| `chunk_document` splits long text | `ChunkDocument(name, longText, opts)` → len > 1 |
| Overlap: last N tokens of prev chunk appear at start of next | Compare suffix/prefix of adjacent chunks |
| Heading promotion detects `# H1` | `ChunkDocument` with markdown headings → chunk with non-empty `SectionPath` |
| Empty string returns empty slice | `ChunkDocument("doc", "", opts)` → `[]DocumentChunk{}` |

### `context/context_test.go`

Port directly from `tests/unit/test_context.py`.

| Python case | Go case |
|---|---|
| Breadcrumb present → prepended to embedding content | `EnrichChunk(chunk)` returns chunk with `EmbeddingContent == breadcrumb + "\n\n" + content` |
| No breadcrumb → built from doc+section | `EnrichChunk` without breadcrumb falls back to `[doc > section]\n\ncontent` |
| No breadcrumb, no section → `[doc]\n\ncontent` | Same |
| Empty doc, empty section → content unchanged | `EnrichChunk` returns `content` when no context available |
| Original struct not mutated | Compare `original.Content` before and after call |
| Returns new struct, not same pointer | `enriched != &original` |
| All non-embedding fields preserved | Check `ChunkIndex`, `SourceOffset`, `ChunkType` unchanged |

### `extractors/extractors_test.go`

Port directly from `tests/unit/test_extractors.py`.

| Python case | Go case |
|---|---|
| `HtmlExtractor`: `<h1>` → `# heading` | `(&HTMLExtractor{}).Extract("f.html", []byte("<h1>T</h1>"))` chunks contain `# T` |
| `HtmlExtractor`: strips `<script>`, `<style>`, `<nav>` | Content of those tags absent from chunks |
| `HtmlExtractor`: `<table>` → pipe rows | Chunk content contains `|` |
| `MarkdownExtractor`: ATX headings build section path | Chunk `SectionPath` populated from `## heading` |
| `MarkdownExtractor`: fenced code block is single chunk | ` ```go ` block → exactly one chunk |
| `TextExtractor`: blank-line split | Multi-paragraph text → multiple chunks |
| `CSVExtractor`: first row is header | Chunk `Section` contains joined header |
| `Registry.For` returns correct extractor | `DefaultRegistry.For("file.html")` returns `*HTMLExtractor` |
| `Registry.For` returns nil for unsupported | `DefaultRegistry.For("file.pdf")` returns nil, ok==false |
| Interface conformance | `var _ Extractor = &HTMLExtractor{}` etc. for all five |

### `generation/generation_test.go`

Port directly from `tests/unit/test_generation.py`.

| Python case | Go case |
|---|---|
| `AnswerContext` minimal construction | `AnswerContext{Chunks: nil, Query: "Q"}` — no panic |
| `PromptBuilder.Build` includes query | `strings.Contains(prompt, "Q")` |
| `PromptBuilder.Build` includes community context when present | Same check |
| `PromptBuilder.Build` omits community context section when nil | Length/content check |
| `AnswerGenerator.Generate` returns stub LLM output | Inject `stubTogetherClient{chatResult: "answer"}`, assert result == "answer" |
| Empty chunks list does not panic | `Generate(ctx, "Q", nil)` — no panic |

The `AnswerGenerator` takes a `TogetherClient` interface — use `stubTogetherClient` in all unit tests. No live API calls.

---

## Layer 2 — graphtypes

**Python template:** `tests/unit/test_context_graph.py` (schema setup), `tests/unit/test_graph.py` (struct construction).

### `graphtypes/graphtypes_test.go`

| Python case | Go case |
|---|---|
| `ContextEdge` fields readable after construction | Build struct, read each field |
| `ContextGraph` zero value has empty node/edge maps | `g := ContextGraph{}; len(g.Nodes) == 0` |
| `ContextGraphStats` zero value safe | No panic constructing zero value |
| Package has no imports beyond `models` | `go list -deps ./graphtypes/...` — verify |

---

## Layer 3 — graph, loader

**Python template:** `tests/unit/test_graph.py`, `tests/unit/test_extractor.py`

### `graph/graph_test.go`

Port from `tests/unit/test_graph.py` and `tests/unit/test_extractor.py`.

| Python case | Go case |
|---|---|
| `SVOTriple` construction — all fields stored | Build `SVOTriple`, read back subject/verb/object/confidence |
| `SVOTriple` confidence < 0 raises | Return error or panic — test for err != nil |
| `SVOTriple` confidence > 1 raises | Same |
| `SVOTriple` confidence 0.0 and 1.0 boundary values accepted | No error |
| All verbs in `VERB_SET` accepted | Loop over constants, none errors |
| `SVOExtractor.Extract` with stub client returns parsed triple | Inject `stubTogetherClient{chatResult: jsonTriple}` |
| `SVOExtractor.Extract` with empty LLM response returns empty slice | `chatResult: "[]"` → len 0 |
| `RelationshipIndex.Add` then `GetEdges` returns added triple | Build index, add triple, retrieve by subject |
| `LLMClient` interface satisfied by stub | Compile-time assertion |

### `loader/loader_test.go`

| Python case | Go case |
|---|---|
| `DocumentLoader.Load` with HTML bytes returns chunks | `loader.Load("f.html", htmlBytes)` → len > 0 |
| `DocumentLoader.Load` with unknown extension returns error | `loader.Load("f.xyz", data)` → err != nil |
| `DocumentLoader.LoadFile` reads from disk | Write temp file, `LoadFile(path)` returns chunks |
| `EnrichContext: true` → chunks have `EmbeddingContent` set | Check field non-empty |
| `EnrichContext: false` → `EmbeddingContent` empty | Check field empty |

---

## Layer 4 — storage, community

**Python template:** `tests/integration/test_storage.py`, `tests/unit/test_community_summarizer.py`, `tests/unit/test_context_graph.py`

### `storage/storage_test.go`

Use in-memory DuckDB for all tests: `Store(":memory:", embeddingDim)`.

| Python case | Go case |
|---|---|
| `Store.Count()` == 0 after construction | `store.Count() == 0` |
| `AddDocument` increments count | Add 3 chunks, `Count() == 3` |
| `Search` returns results | Add chunks with known embeddings, query with similar vector |
| `RegisterNamespace` then `RegisterDomain` succeeds | No error, name readable back |
| Duplicate `AddDocument` for same chunk_id upserts, not duplicates | Add same chunks twice, `Count()` unchanged |
| `RebuildFTSIndex` after add — FTS search returns match | Add chunk with keyword, rebuild, FTS search finds it |
| Thread safety: concurrent writes do not corrupt count | Spawn 10 goroutines each adding 5 chunks, check final count == 50 |

### `community/community_test.go`

| Python case | Go case |
|---|---|
| `CommunitySummarizer` requires a client (not nil) | `NewCommunitySummarizer(nil)` → non-nil error |
| `CommunitySummarizer.Summarize` returns stub LLM output | Inject stub, call with top terms |
| `CommunityIndex` zero value — `GetCommunity` returns -1/not-found | `idx.GetCommunity("unknown_chunk")` |
| `CommunityBuilder.Build` with synthetic embeddings produces ≥1 community | 5 identical vectors → 1 community |
| Louvain with fully disconnected graph → each node is own community | Similarity matrix all zeros → N communities for N nodes |

Community LLM calls use `stubTogetherClient`. No real Together.ai calls in unit tests.

---

## Layer 5 — cluster, ner, indexer, lifecycle

**Python template:** `tests/unit/test_cluster.py`, `tests/unit/test_ner.py`, `tests/unit/test_ner_pipeline.py`, `tests/unit/test_indexer.py`

### `cluster/cluster_test.go`

Port directly from `tests/unit/test_cluster.py`.

| Python case | Go case |
|---|---|
| Co-occurrence raw counts: A+B co-occur 3× → matrix[A][B] == 3.0 | Build `CooccurrenceMatrix` with same synthetic entity index |
| `min_cooccurrence` filter removes A+C (count == 1) | `CooccurrenceMatrix{MinCooccurrence: 3}` → A+C absent |
| Jaccard normalisation → all scores in [0, 1] | Loop over matrix values |
| `ClusterMap` groups A+B and C+D into separate clusters | `ClusterMap.Clusters()` → 2 groups with expected members |
| `ClusterMap` with single entity → 1 cluster of size 1 | No panic |
| Cosine similarity: identical vectors → score == 1.0 | `CosineSimilarity(v, v) == 1.0` |
| Cosine similarity: orthogonal vectors → score == 0.0 | Construct orthogonal pair |

### `ner/ner_test.go`

Port directly from `tests/unit/test_ner.py`.

| Python case | Go case |
|---|---|
| `VocabularyMatcher` — case-insensitive entity match | `matcher.Match("HCA Healthcare ...")` contains entity id |
| Alias match: "hospital corporation of america" → `ent_hca` | Same |
| Frequency counted: entity appears twice → `Frequency == 2` | Check returned `EntityMatch.Frequency` |
| Positions recorded: offset matches text | `match.Positions[0] == strings.Index(lowerText, "wire transfer")` |
| Spans recorded | `match.Spans[0].Start` and `.End` enclose the matched term |
| `EntityIndex.AddMatch` then `GetChunkEntities` round-trip | Add match, retrieve by chunk id |
| Merge: overlapping spans deduped | Two matches with same span → one match returned |
| `NERPipeline.ExtractEntities` with stub client returns parsed entities | Inject stub returning JSON entity list |
| Vocabulary embedded in binary (compile check) | `go build ./ner/...` — no missing embed files |

### `indexer/indexer_test.go`

Port directly from `tests/unit/test_indexer.py`.

| Python case | Go case |
|---|---|
| `Indexer.IndexSource` with local dir → chunks stored | Use `t.TempDir()` with `.txt` files; inject stub embedder; check `store.Count() > 0` |
| Returned count matches `store.Count()` | `n == store.Count()` |
| `IndexSourceAsync` returns handle; `Wait()` blocks until done | Start async, `Wait()`, then check count |
| Abort via context cancel → stops mid-batch | Cancel context after first batch; count < total chunks |

Stub embedder returns `ones(n, dim)` — port of Python's `_StubModel`.

```go
type stubEmbedder struct{ dim int }

func (s *stubEmbedder) Embed(_ context.Context, req together.EmbedRequest) (together.EmbedResponse, error) {
    resp := together.EmbedResponse{}
    for i := range req.Input {
        vec := make([]float32, s.dim)
        for j := range vec { vec[j] = 1.0 }
        resp.Data = append(resp.Data, struct{Embedding []float32; Index int}{vec, i})
    }
    return resp, nil
}
```

---

## Layer 6 — search

**Python template:** `tests/unit/test_enhanced_search.py`

### `search/search_test.go`

| Python case | Go case |
|---|---|
| Seed lane returns nearest-neighbour chunks | Add 5 chunks with known embeddings; query with same vector as chunk 0 → chunk 0 in top result |
| Entity lane: chunks matching active entity returned | Add entity annotation to chunk, search with same entity id |
| Cluster lane: chunks in same cluster as seed | Add cluster annotation; seed returns co-cluster members |
| Structural lane: adjacent chunks included | Check `chunk_index` neighbours returned |
| `top_k` respected | `EnhancedSearch{TopK: 2}.Search(...)` returns ≤ 2 results |
| All four lanes disabled → empty results | `EnhancedSearch{Seed: false, Entity: false, Cluster: false, Structural: false}` |
| Combined lanes deduplicate: same chunk in two lanes appears once | Check no duplicate `ChunkID` in results |
| `ScoredChunk.Provenance` identifies which lane sourced each result | Check field value matches expected lane name |

Use in-memory DuckDB store with synthetic `float32` embeddings (dim 8, same as Python's `DIM = 8`). Stub `TogetherClient` for embedding the query vector.

---

## Layer 7 — ingest

**Python template:** `tests/integration/test_loader.py`, `tests/unit/test_indexer.py`

### `ingest/ingest_test.go`

End-to-end integration test. Only layer that connects all packages.

| Test | What it verifies |
|---|---|
| `Build("index_config.yaml")` returns non-nil `*Index` | Config loads, no panic |
| `Index.Search("query")` after ingesting `.txt` returns results | Full pipeline: ingest → embed → store → search |
| `Index.AddNamespace` + `AddDomain` accessible after `Build` | Namespace/domain API wired through |
| `Index.Count()` matches number of ingested chunks | Count consistent after pipeline |
| Unsupported format (`.pdf`) skipped gracefully | No error; supported files still ingested |
| Config with no sources → empty index, no error | `Index.Count() == 0` |

Use stub embedder — do **not** require a live `TOGETHER_API_KEY`. The stub is injected via a `BuildOptions` struct:

```go
opts := ingest.BuildOptions{EmbedClient: &stubEmbedder{dim: 1024}}
idx, err := ingest.BuildWithOptions("testdata/config.yaml", opts)
```

Place test YAML and fixture files under `ingest/testdata/`.

---

## Common Patterns Summary

| Pattern | Implementation |
|---|---|
| No real API calls | Inject `stubTogetherClient` via constructor |
| No real ML calls | Inject `stubEmbedder` returning `ones(n, dim)` |
| In-memory storage | `Store(":memory:", dim)` |
| Temporary files | `t.TempDir()` — cleaned up automatically |
| Interface conformance | `var _ InterfaceName = (*ConcreteType)(nil)` compile-time assertion |
| Boundary values | Confidence 0.0/1.0, empty slice, zero-length text |
| No global state | Each test constructs its own instances |
| Parallel-safe | Avoid `t.Parallel()` for DuckDB tests sharing `:memory:` |
