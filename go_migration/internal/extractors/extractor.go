// Package extractors defines the Extractor protocol and the registry.
//
// MIGRATION MARKER: Ported from chonk/extractors/_protocol.py and __init__.py
// Python source:
//   - chonk/extractors/_protocol.py  (36 LOC) — Extractor Protocol
//   - chonk/extractors/__init__.py            — detect_extractor, normalize_type
//
// Status: COMPLETE (protocol + registry)
// Dropped extractors (per ADR extraction-scope.md):
//   PDF, DOCX, XLSX, PPTX, ODF, JSON, XML, YAML, Parquet, Python AST,
//   TypeScript, Java, FHIR, EDGAR, email, MIME, ClinicalTrials, FDA,
//   ATT&CK, CVE, CWE, NIST, NoSQL, lookup_table, renderer → all 21 dropped.
//
// Kept extractors (5):
//   text, csv, markdown, html, sql/db-schema
//
// Translation notes:
//   - Python Protocol + runtime_checkable → Go interface
//   - Python can_handle(doc_type str) → Go CanHandle(docType string) bool
//   - Python annotate() returning list → Go Annotate() returning new slice
//   - Registry: first-match semantics (same as Python detect_extractor)
package extractors

import "github.com/kennethstott/chonk/internal/models"

// Extractor is the Go equivalent of the Python Extractor Protocol.
// Implementations: TextExtractor, CSVExtractor, MarkdownExtractor,
// HTMLExtractor, SQLExtractor.
type Extractor interface {
	// Extract converts raw bytes to plain UTF-8 text.
	// sourcePath is optional (empty string if unavailable).
	Extract(data []byte, sourcePath string) (string, error)

	// CanHandle returns true if this extractor handles the given docType alias.
	// docType examples: "text", "csv", "markdown", "html", "sql"
	CanHandle(docType string) bool

	// Annotate stamps format-specific navigation metadata onto chunks.
	// Implementations that have no metadata to add must return chunks unchanged.
	Annotate(chunks []models.DocumentChunk, data []byte, sourcePath string) []models.DocumentChunk
}

// Registry holds an ordered list of extractors.
// First-match semantics: CanHandle is tested in registration order.
type Registry struct {
	extractors []Extractor
}

// NewRegistry returns a Registry pre-loaded with the 5 production extractors
// in canonical priority order.
func NewRegistry() *Registry {
	r := &Registry{}
	r.Register(TextExtractor{})
	r.Register(CSVExtractor{})
	r.Register(MarkdownExtractor{})
	r.Register(HTMLExtractor{})
	r.Register(SQLExtractor{})
	return r
}

// Register appends an extractor to the registry (lowest priority).
// To insert at higher priority, build the Registry manually.
func (r *Registry) Register(e Extractor) {
	r.extractors = append(r.extractors, e)
}

// Detect returns the first extractor whose CanHandle(docType) is true,
// or nil if none match.
// Python equivalent: detect_extractor(doc_type) in extractors/__init__.py
func (r *Registry) Detect(docType string) Extractor {
	normalized := NormalizeType(docType)
	for _, e := range r.extractors {
		if e.CanHandle(normalized) {
			return e
		}
	}
	return nil
}

// NormalizeType maps file extensions and aliases to canonical docType strings.
// Python equivalent: normalize_type() in extractors/__init__.py
func NormalizeType(raw string) string {
	switch raw {
	case "txt", "text/plain":
		return "text"
	case "md", "mdown", "mkd", "text/markdown":
		return "markdown"
	case "htm", "text/html":
		return "html"
	case "tsv":
		return "csv"
	case "ddl", "schema":
		return "sql"
	default:
		return raw
	}
}

// DetectTypeFromSource infers docType from a file path's extension.
// Python equivalent: detect_type_from_source() in extractors/__init__.py
func DetectTypeFromSource(sourcePath string) string {
	if sourcePath == "" {
		return "text"
	}
	// Walk from end to find last '.'
	for i := len(sourcePath) - 1; i >= 0; i-- {
		if sourcePath[i] == '.' {
			ext := sourcePath[i+1:]
			return NormalizeType(ext)
		}
		if sourcePath[i] == '/' || sourcePath[i] == '\\' {
			break
		}
	}
	return "text"
}
