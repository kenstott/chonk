package generation_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/kennethstott/chonk/internal/generation"
	"github.com/kennethstott/chonk/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func makeChunk(content string, score float64) models.ScoredChunk {
	c := models.NewDocumentChunk("doc", content)
	return models.ScoredChunk{
		ChunkID:    "c1",
		Chunk:      c,
		Score:      score,
		Provenance: "seed",
	}
}

func makeLLMFunc(reply string) generation.LLMFunc {
	return func(_ string) (string, error) { return reply, nil }
}

func failingLLMFunc() generation.LLMFunc {
	return func(_ string) (string, error) {
		return "", fmt.Errorf("API error: 503")
	}
}

// ---------------------------------------------------------------------------
// PromptBuilder
// ---------------------------------------------------------------------------

func TestPromptBuilder_SelectChunks_FitsAll(t *testing.T) {
	pb := generation.PromptBuilder{}
	ctx := generation.AnswerContext{
		Query: "What is X?",
		Chunks: []models.ScoredChunk{
			makeChunk("Short A", 0.9),
			makeChunk("Short B", 0.8),
		},
	}
	selected := pb.SelectChunks(ctx, 4096)
	assert.Len(t, selected, 2)
}

func TestPromptBuilder_SelectChunks_RespectsBudget(t *testing.T) {
	pb := generation.PromptBuilder{}
	big := strings.Repeat("x", 2000)
	ctx := generation.AnswerContext{
		Query: "Q",
		Chunks: []models.ScoredChunk{
			makeChunk(big, 0.9),
			makeChunk(big, 0.8),
			makeChunk(big, 0.7),
		},
	}
	// tokenBudget=500 → charBudget = 500*4 * 3/4 = 1500; each big chunk is 2000 chars
	selected := pb.SelectChunks(ctx, 500)
	assert.Len(t, selected, 0, "no chunk should fit in a 1500-char budget with 2000-char chunks")
}

func TestPromptBuilder_Build_ContainsQuery(t *testing.T) {
	pb := generation.PromptBuilder{}
	ctx := generation.AnswerContext{
		Query:  "What is RAG?",
		Chunks: []models.ScoredChunk{makeChunk("RAG stands for Retrieval Augmented Generation.", 0.9)},
	}
	prompt := pb.Build(ctx, 4096)
	assert.Contains(t, prompt, "What is RAG?")
	assert.Contains(t, prompt, "RAG stands for")
	assert.Contains(t, prompt, "## Question")
	assert.Contains(t, prompt, "## Context")
}

func TestPromptBuilder_Build_IncludesBreadcrumb(t *testing.T) {
	pb := generation.PromptBuilder{}
	breadcrumb := "Introduction > Overview"
	c := models.NewDocumentChunk("doc", "Some content")
	c.Breadcrumb = &breadcrumb
	ctx := generation.AnswerContext{
		Query:  "Q",
		Chunks: []models.ScoredChunk{{ChunkID: "c1", Chunk: c, Score: 0.8, Provenance: "seed"}},
	}
	prompt := pb.Build(ctx, 4096)
	assert.Contains(t, prompt, "Introduction > Overview")
}

// ---------------------------------------------------------------------------
// AnswerGenerator
// ---------------------------------------------------------------------------

func TestAnswerGenerator_Generate_Success(t *testing.T) {
	gen := generation.NewAnswerGenerator(makeLLMFunc("42"), 4096)
	ctx := generation.AnswerContext{
		Query:  "What is the answer?",
		Chunks: []models.ScoredChunk{makeChunk("The answer is forty-two.", 0.95)},
	}
	answer, err := gen.Generate(ctx)
	require.NoError(t, err)
	assert.Equal(t, "42", answer.Text)
	assert.NotEmpty(t, answer.Citations)
}

func TestAnswerGenerator_Generate_LLMError(t *testing.T) {
	gen := generation.NewAnswerGenerator(failingLLMFunc(), 4096)
	ctx := generation.AnswerContext{
		Query:  "Q",
		Chunks: []models.ScoredChunk{makeChunk("ctx", 0.9)},
	}
	_, err := gen.Generate(ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "503")
}

func TestAnswerGenerator_DefaultTokenBudget(t *testing.T) {
	// tokenBudget <= 0 → defaults to 4096, no panic
	gen := generation.NewAnswerGenerator(makeLLMFunc("ok"), 0)
	ctx := generation.AnswerContext{Query: "Q", Chunks: nil}
	answer, err := gen.Generate(ctx)
	require.NoError(t, err)
	assert.Equal(t, "ok", answer.Text)
}

// ---------------------------------------------------------------------------
// Answer zero value
// ---------------------------------------------------------------------------

func TestAnswer_ZeroValue(t *testing.T) {
	var a generation.Answer
	assert.Empty(t, a.Text)
	assert.Nil(t, a.Citations)
}
