// Package chunking provides pure chunking functions.
//
// MIGRATION MARKER: Ported from chonk/chunking.py (936 LOC)
// Status: STUB — core data types and interface complete; full algorithm port TODO.
//
// Python surface ported:
//   - chunk_document()             → ChunkDocument()
//   - extract_markdown_sections()  → ExtractMarkdownSections()
//   - is_list_line()               → IsListLine()
//   - is_table_line()              → IsTableLine()
//   - merge_blocks()               → MergeBlocks()
//   - promote_plain_text_headers() → PromotePlainTextHeaders()
//   - NOVEL_STRUCTURAL_LEVELS      → NovelStructuralLevels
//
// Translation notes:
//   - Python tuple[str, int] structural levels → Go StructuralLevel struct
//   - Python Optional[list] → Go nil slice
//   - Python dataclass DocumentChunk → Go models.DocumentChunk
package chunking

import (
	"regexp"
	"strings"

	"github.com/kennethstott/chonk/internal/models"
)

// StructuralLevel pairs a regex pattern with a heading level (1 = #, 2 = ##).
// Python source: NOVEL_STRUCTURAL_LEVELS list[tuple[str, int]]
type StructuralLevel struct {
	Pattern *regexp.Regexp
	Level   int
}

// NovelStructuralLevels is the default plain-text header promotion config
// for prose corpora (novels, non-fiction).
// Python source: NOVEL_STRUCTURAL_LEVELS constant in chunking.py
var NovelStructuralLevels = []StructuralLevel{
	{
		Pattern: regexp.MustCompile(`(?:PART|BOOK|SCENE)\s+(?:[IVXLCDM]+|THE\s+[A-Z]+|\d+)`),
		Level:   1,
	},
	{
		Pattern: regexp.MustCompile(`(?i:CHAPTER)\s+(?:[IVXLCDM]+|\d+)`),
		Level:   2,
	},
}

// ChunkOptions configures chunk_document behaviour.
// Python source: chunk_document kwargs (min_size, max_size, overflow_margin, etc.)
type ChunkOptions struct {
	MinSize         int     // default 600
	MaxSize         int     // default 1500
	OverflowMargin  float64 // default 0.15
	StructuralLevels []StructuralLevel // nil → use default Markdown heading detection
}

// DefaultChunkOptions returns the defaults matching Python's chunk_document defaults.
func DefaultChunkOptions() ChunkOptions {
	return ChunkOptions{
		MinSize:        600,
		MaxSize:        1500,
		OverflowMargin: 0.15,
	}
}

// ChunkDocument splits document content into DocumentChunks.
// Python source: chunking.chunk_document()
//
// MIGRATION STUB: Full markdown-aware block merging and section extraction
// algorithm to be ported from chunking.py (lines 300–936).
// Current implementation: naive size-based splitting with section tracking.
func ChunkDocument(documentName, content string, opts ChunkOptions) []models.DocumentChunk {
	if opts.MinSize == 0 {
		opts = DefaultChunkOptions()
	}
	if strings.TrimSpace(content) == "" {
		return nil
	}
	sections := ExtractMarkdownSections(content)
	_ = sections // will be used in full implementation

	// Naive implementation: split at paragraph boundaries.
	paragraphs := splitParagraphs(content)
	var chunks []models.DocumentChunk
	var buf strings.Builder
	chunkIdx := 0

	flush := func() {
		text := strings.TrimSpace(buf.String())
		if text == "" {
			return
		}
		c := models.NewDocumentChunk(documentName, text)
		c.ChunkIndex = chunkIdx
		chunks = append(chunks, c)
		chunkIdx++
		buf.Reset()
	}

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		if buf.Len() > 0 && buf.Len()+len(para) > opts.MaxSize {
			flush()
		}
		if buf.Len() > 0 {
			buf.WriteString("\n\n")
		}
		buf.WriteString(para)
		if buf.Len() >= opts.MinSize {
			flush()
		}
	}
	flush()
	return chunks
}

// ExtractMarkdownSections returns the ordered list of section headings found
// in a Markdown document.
// Python source: chunking.extract_markdown_sections()
func ExtractMarkdownSections(content string) []string {
	var sections []string
	for _, line := range strings.Split(content, "\n") {
		if !strings.HasPrefix(line, "#") {
			continue
		}
		heading := strings.TrimLeft(line, "#")
		heading = strings.TrimSpace(heading)
		if heading != "" {
			sections = append(sections, heading)
		}
	}
	return sections
}

// IsListLine reports whether a line is a Markdown list item.
// Python source: chunking.is_list_line()
func IsListLine(line string) bool {
	trimmed := strings.TrimLeft(line, " \t")
	if len(trimmed) < 2 {
		return false
	}
	return (trimmed[0] == '-' || trimmed[0] == '*' || trimmed[0] == '+') && trimmed[1] == ' '
}

// IsTableLine reports whether a line is a Markdown table row.
// Python source: chunking.is_table_line()
func IsTableLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, "|") && strings.HasSuffix(trimmed, "|")
}

// MergeBlocks merges a list of text blocks up to max_size.
// Python source: chunking.merge_blocks()
//
// MIGRATION STUB: Full implementation (table/list awareness, section tracking)
// to be ported from chunking.py.
func MergeBlocks(blocks []string, maxSize int) []string {
	if len(blocks) == 0 {
		return nil
	}
	var result []string
	var buf strings.Builder
	for _, block := range blocks {
		if buf.Len() > 0 && buf.Len()+len(block)+2 > maxSize {
			result = append(result, buf.String())
			buf.Reset()
		}
		if buf.Len() > 0 {
			buf.WriteString("\n\n")
		}
		buf.WriteString(block)
	}
	if buf.Len() > 0 {
		result = append(result, buf.String())
	}
	return result
}

// PromotePlainTextHeaders promotes structural markers (CHAPTER, PART, etc.)
// to Markdown headings in prose documents.
// Python source: chunking.promote_plain_text_headers()
//
// MIGRATION STUB: Full algorithm (chapter title extraction, roman numeral
// detection, TOC entry detection) to be ported from chunking.py.
func PromotePlainTextHeaders(content string, levels []StructuralLevel) string {
	if len(levels) == 0 {
		levels = NovelStructuralLevels
	}
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		for _, sl := range levels {
			if sl.Pattern.MatchString(trimmed) {
				prefix := strings.Repeat("#", sl.Level) + " "
				lines[i] = prefix + trimmed
				break
			}
		}
	}
	return strings.Join(lines, "\n")
}

// ── helpers ───────────────────────────────────────────────────────────────────

// splitParagraphs splits content at blank lines (double newline).
func splitParagraphs(content string) []string {
	// Normalise Windows line endings
	content = strings.ReplaceAll(content, "\r\n", "\n")
	parts := strings.Split(content, "\n\n")
	return parts
}
