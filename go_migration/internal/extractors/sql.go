package extractors

import (
	"strings"

	"github.com/kennethstott/chonk/internal/models"
)

// SQLExtractor handles raw SQL DDL / schema files.
// Python source: chonk/extractors/_nosql.py, transports/_db_schema.py
// MIGRATION MARKER: Status COMPLETE — SQL DDL is plain text; extractor
// normalises whitespace and returns the DDL body for chunking.
type SQLExtractor struct{}

// CanHandle returns true for sql, ddl, and schema aliases.
func (e SQLExtractor) CanHandle(docType string) bool {
	switch docType {
	case "sql", "ddl", "schema":
		return true
	}
	return false
}

// Extract normalises SQL DDL text.
// Decodes bytes, strips byte-order marks, returns the SQL body.
func (e SQLExtractor) Extract(data []byte, _ string) (string, error) {
	if len(data) == 0 {
		return "", nil
	}
	text := decodeText(data)
	// Strip UTF-8 BOM if present
	text = strings.TrimPrefix(text, "\xef\xbb\xbf")
	return strings.TrimSpace(text), nil
}

// Annotate is a no-op for SQL.
func (e SQLExtractor) Annotate(chunks []models.DocumentChunk, _ []byte, _ string) []models.DocumentChunk {
	return chunks
}
