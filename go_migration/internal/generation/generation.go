// Package generation provides answer generation and prompt building.
//
// MIGRATION MARKER: Ported from chonk/generation/ sub-package.
// Python sources:
//   - chonk/generation/_answer.py          (57 LOC)  — Answer, AnswerGenerator
//   - chonk/generation/_context.py                   — AnswerContext
//   - chonk/generation/_prompt_builder.py            — PromptBuilder
//
// Status: COMPLETE
//
// Translation notes:
//   - Python Callable[[str], str] llm_fn → Go func(prompt string) (string, error)
//   - Python dataclass Answer → Go Answer struct
//   - Python AnswerContext → Go AnswerContext struct
//   - Python PromptBuilder → Go PromptBuilder struct
//   - Token budget estimated via len(text)/4 (same heuristic as Python)
package generation

import (
	"fmt"
	"strings"

	"github.com/kennethstott/chonk/internal/models"
)

// Answer holds the generated text and the chunks cited in the prompt.
// Python source: generation._answer.Answer
type Answer struct {
	Text      string
	Citations []models.ScoredChunk
}

// AnswerContext carries the query and retrieved chunks passed to generation.
// Python source: generation._context.AnswerContext
type AnswerContext struct {
	Query  string
	Chunks []models.ScoredChunk
}

// LLMFunc is a function that takes a prompt string and returns the model reply.
// Python source: generation._answer.AnswerGenerator llm_fn parameter.
// When using Together.ai: wrap together.Client.Chat() to satisfy this signature.
type LLMFunc func(prompt string) (string, error)

// PromptBuilder assembles RAG prompts from query + retrieved chunks.
// Python source: generation._prompt_builder.PromptBuilder
type PromptBuilder struct{}

// SelectChunks greedily picks as many ScoredChunks as fit within tokenBudget.
// Chunks are taken in order (highest score first, as supplied by EnhancedSearch).
// Token budget is estimated at 4 characters per token (same heuristic as Python).
// Python source: generation._prompt_builder.PromptBuilder.select_chunks()
func (p PromptBuilder) SelectChunks(ctx AnswerContext, tokenBudget int) []models.ScoredChunk {
	charBudget := tokenBudget * 4
	// Reserve ~25% for query + prompt scaffolding
	charBudget = charBudget * 3 / 4
	var selected []models.ScoredChunk
	used := 0
	for _, sc := range ctx.Chunks {
		l := len(sc.Chunk.Content)
		if used+l > charBudget {
			break
		}
		selected = append(selected, sc)
		used += l
	}
	return selected
}

// Build assembles the final prompt string.
// Python source: generation._prompt_builder.PromptBuilder.build()
func (p PromptBuilder) Build(ctx AnswerContext, tokenBudget int) string {
	selected := p.SelectChunks(ctx, tokenBudget)
	var sb strings.Builder
	sb.WriteString("Answer the following question using only the provided context.\n\n")
	sb.WriteString("## Context\n\n")
	for i, sc := range selected {
		breadcrumb := ""
		if sc.Chunk.Breadcrumb != nil {
			breadcrumb = "[" + *sc.Chunk.Breadcrumb + "] "
		}
		fmt.Fprintf(&sb, "### Source %d %s\n%s\n\n", i+1, breadcrumb, sc.Chunk.Content)
	}
	sb.WriteString("## Question\n\n")
	sb.WriteString(ctx.Query)
	sb.WriteString("\n\n## Answer\n\n")
	return sb.String()
}

// AnswerGenerator generates answers by delegating to a user-supplied LLMFunc.
// Python source: generation._answer.AnswerGenerator
type AnswerGenerator struct {
	llmFn       LLMFunc
	tokenBudget int
	builder     PromptBuilder
}

// NewAnswerGenerator creates an AnswerGenerator.
// tokenBudget defaults to 4096 when <= 0 (matching Python default).
func NewAnswerGenerator(llmFn LLMFunc, tokenBudget int) *AnswerGenerator {
	if tokenBudget <= 0 {
		tokenBudget = 4096
	}
	return &AnswerGenerator{
		llmFn:       llmFn,
		tokenBudget: tokenBudget,
	}
}

// Generate builds a prompt, calls llmFn, and returns an Answer with citations.
// Python source: generation._answer.AnswerGenerator.generate()
func (g *AnswerGenerator) Generate(ctx AnswerContext) (Answer, error) {
	citations := g.builder.SelectChunks(ctx, g.tokenBudget)
	prompt := g.builder.Build(ctx, g.tokenBudget)
	text, err := g.llmFn(prompt)
	if err != nil {
		return Answer{}, fmt.Errorf("generate: llm call failed: %w", err)
	}
	return Answer{Text: text, Citations: citations}, nil
}
