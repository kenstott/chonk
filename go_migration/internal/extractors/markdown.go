package extractors

import (
	"regexp"
	"unicode/utf8"

	"github.com/kennethstott/chonk/internal/models"
)

// frontmatterRe strips YAML frontmatter from Markdown documents.
// Matches "---\n…\n---\n" at the start of the document (with optional \r).
var frontmatterRe = regexp.MustCompile(`(?s)\A---\r?\n.*?\n---\r?\n?`)

// MarkdownExtractor handles Markdown documents.
// Python source: chonk/extractors/_markdown.py
// MIGRATION MARKER: Status COMPLETE — frontmatter stripping ported exactly.
type MarkdownExtractor struct{}

// CanHandle returns true for markdown and md.
func (e MarkdownExtractor) CanHandle(docType string) bool {
	switch docType {
	case "markdown", "md", "mdown", "mkd", "text/markdown":
		return true
	}
	return false
}

// Extract strips YAML frontmatter and returns the Markdown body.
func (e MarkdownExtractor) Extract(data []byte, _ string) (string, error) {
	if len(data) == 0 {
		return "", nil
	}
	var text string
	if utf8.Valid(data) {
		text = string(data)
	} else {
		text = decodeLatin1(data)
	}
	text = frontmatterRe.ReplaceAllString(text, "")
	// Strip leading newlines (Python lstripNewlines equivalent)
	for len(text) > 0 && (text[0] == '\n' || text[0] == '\r') {
		text = text[1:]
	}
	return text, nil
}

// Annotate is a no-op for Markdown.
func (e MarkdownExtractor) Annotate(chunks []models.DocumentChunk, _ []byte, _ string) []models.DocumentChunk {
	return chunks
}
