package extractors

import (
	"regexp"
	"strings"
)

// MarkdownExtractor extracts markdown files, stripping YAML frontmatter if present.
type MarkdownExtractor struct{}

// HANDLED contains the document types this extractor can handle.
var HANDLED = map[string]bool{
	"markdown": true,
}

// canHandle checks if the extractor can handle the given document type.
func (e MarkdownExtractor) canHandle(docType string) bool {
	return HANDLED[docType]
}

// extract processes the input data, removing YAML frontmatter if present.
func (e MarkdownExtractor) extract(data []byte, sourcePath string) (string, error) {
	text := string(data)
	// Remove YAML frontmatter
	text = frontmatterRE.ReplaceAllString(text, "")
	// Remove leading newlines
	text = strings.TrimLeft(text, "\n")
	return text, nil
}

// annotate processes chunks with additional metadata from the source data.
// Currently returns the chunks unmodified.
func (e MarkdownExtractor) annotate(chunks []DocumentChunk, data []byte, sourcePath string) []DocumentChunk {
	return chunks
}

// Regex to match YAML frontmatter at the start of the document
var frontmatterRE = regexp.MustCompile(`^---\r?\n.*?\n---\r?\n?`)
