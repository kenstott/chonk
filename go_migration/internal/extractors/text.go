package extractors

import (
	"unicode/utf8"

	"github.com/kennethstott/chonk/internal/models"
)

// TextExtractor handles plain text formats.
// Python source: chonk/extractors/_text.py
// MIGRATION MARKER: Partial port — Python version also handles encoding detection
// via chardet; Go uses stdlib utf8.Valid + Latin-1 fallback.
type TextExtractor struct{}

// CanHandle returns true for text, txt, and plain-text MIME types.
func (e TextExtractor) CanHandle(docType string) bool {
	switch docType {
	case "text", "txt", "text/plain":
		return true
	}
	return false
}

// Extract decodes bytes as UTF-8, falling back to Latin-1.
func (e TextExtractor) Extract(data []byte, _ string) (string, error) {
	if len(data) == 0 {
		return "", nil
	}
	if utf8.Valid(data) {
		return string(data), nil
	}
	return decodeLatin1(data), nil
}

// Annotate is a no-op for plain text.
func (e TextExtractor) Annotate(chunks []models.DocumentChunk, _ []byte, _ string) []models.DocumentChunk {
	return chunks
}
