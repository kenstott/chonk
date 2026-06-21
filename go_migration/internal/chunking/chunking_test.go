package chunking_test

import (
	"strings"
	"testing"

	"github.com/kennethstott/chonk/internal/chunking"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// ChunkDocument
// ---------------------------------------------------------------------------

func TestChunkDocument_BasicSplit(t *testing.T) {
	content := strings.Repeat("word ", 200) + "\n\n" + strings.Repeat("more ", 200)
	chunks := chunking.ChunkDocument("doc", content, chunking.DefaultChunkOptions())
	require.NotEmpty(t, chunks)
	for _, c := range chunks {
		assert.Equal(t, "doc", c.DocumentName)
		assert.NotEmpty(t, c.Content)
	}
}

func TestChunkDocument_Empty(t *testing.T) {
	chunks := chunking.ChunkDocument("doc", "", chunking.DefaultChunkOptions())
	assert.Nil(t, chunks)
}

func TestChunkDocument_WhitespaceOnly(t *testing.T) {
	chunks := chunking.ChunkDocument("doc", "   \n\n  ", chunking.DefaultChunkOptions())
	assert.Nil(t, chunks)
}

func TestChunkDocument_ChunkIndexSequential(t *testing.T) {
	// Build content that will produce at least 2 chunks
	content := strings.Repeat("a ", 400) + "\n\n" + strings.Repeat("b ", 400) + "\n\n" + strings.Repeat("c ", 400)
	opts := chunking.ChunkOptions{MinSize: 300, MaxSize: 600, OverflowMargin: 0.15}
	chunks := chunking.ChunkDocument("doc", content, opts)
	for i, c := range chunks {
		assert.Equal(t, i, c.ChunkIndex, "chunk %d has wrong index", i)
	}
}

// ---------------------------------------------------------------------------
// ExtractMarkdownSections
// ---------------------------------------------------------------------------

func TestExtractMarkdownSections_BasicHeadings(t *testing.T) {
	md := "# Title\nsome text\n## Section 1\nmore text\n### Subsection\nend"
	sections := chunking.ExtractMarkdownSections(md)
	assert.Equal(t, []string{"Title", "Section 1", "Subsection"}, sections)
}

func TestExtractMarkdownSections_NoHeadings(t *testing.T) {
	sections := chunking.ExtractMarkdownSections("just prose, no headings")
	assert.Empty(t, sections)
}

func TestExtractMarkdownSections_Empty(t *testing.T) {
	assert.Empty(t, chunking.ExtractMarkdownSections(""))
}

// ---------------------------------------------------------------------------
// IsListLine
// ---------------------------------------------------------------------------

func TestIsListLine(t *testing.T) {
	cases := []struct {
		line string
		want bool
	}{
		{"- item", true},
		{"* item", true},
		{"+ item", true},
		{"  - indented", true},
		{"regular text", false},
		{"-no space", false},
		{"", false},
	}
	for _, tc := range cases {
		assert.Equal(t, tc.want, chunking.IsListLine(tc.line), "line=%q", tc.line)
	}
}

// ---------------------------------------------------------------------------
// IsTableLine
// ---------------------------------------------------------------------------

func TestIsTableLine(t *testing.T) {
	cases := []struct {
		line string
		want bool
	}{
		{"| col1 | col2 |", true},
		{"| --- | --- |", true},
		{"  | col1 | col2 |  ", true},
		{"regular text", false},
		{"|no trailing pipe", false},
		{"", false},
	}
	for _, tc := range cases {
		assert.Equal(t, tc.want, chunking.IsTableLine(tc.line), "line=%q", tc.line)
	}
}

// ---------------------------------------------------------------------------
// MergeBlocks
// ---------------------------------------------------------------------------

func TestMergeBlocks_MergesSmallBlocks(t *testing.T) {
	blocks := []string{"short", "also short", "another"}
	merged := chunking.MergeBlocks(blocks, 1000)
	assert.Len(t, merged, 1)
	assert.Contains(t, merged[0], "short")
	assert.Contains(t, merged[0], "another")
}

func TestMergeBlocks_SplitsAtMaxSize(t *testing.T) {
	big := strings.Repeat("x", 600)
	blocks := []string{big, big}
	merged := chunking.MergeBlocks(blocks, 700)
	assert.Len(t, merged, 2)
}

func TestMergeBlocks_Empty(t *testing.T) {
	assert.Nil(t, chunking.MergeBlocks(nil, 1000))
	assert.Nil(t, chunking.MergeBlocks([]string{}, 1000))
}

// ---------------------------------------------------------------------------
// PromotePlainTextHeaders
// ---------------------------------------------------------------------------

func TestPromotePlainTextHeaders_ChapterPromotion(t *testing.T) {
	content := "CHAPTER 1\nSome text here."
	result := chunking.PromotePlainTextHeaders(content, nil) // nil → default levels
	assert.Contains(t, result, "## CHAPTER 1")
}

func TestPromotePlainTextHeaders_NoPromotion(t *testing.T) {
	content := "Just regular prose with no structural markers."
	result := chunking.PromotePlainTextHeaders(content, nil)
	assert.Equal(t, content, result)
}

// ---------------------------------------------------------------------------
// NovelStructuralLevels
// ---------------------------------------------------------------------------

func TestNovelStructuralLevels_MatchesPART(t *testing.T) {
	for _, sl := range chunking.NovelStructuralLevels {
		if sl.Level == 1 {
			assert.True(t, sl.Pattern.MatchString("PART IV"), "PART IV should match level 1")
			assert.True(t, sl.Pattern.MatchString("PART 1"))
		}
	}
}
