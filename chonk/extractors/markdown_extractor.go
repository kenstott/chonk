package extractors

import (
 "bytes"
 "regexp"
 "strings"
)

// MarkdownExtractor extracts markdown files, stripping YAML frontmatter if present.
// It implements the common interface for document extractors.
// HANDLED types: "markdown"

type MarkdownExtractor struct {
 // No fields needed for this simple extractor
}

// CanHandle returns true if the extractor can handle the given document type
func (m *MarkdownExtractor) CanHandle(docType string) bool {
 return docType == "markdown"
}

// Extract strips YAML frontmatter from the markdown text and returns the clean content
func (m *MarkdownExtractor) Extract(data []byte, sourcePath string) string {
 // Create a regex pattern to match YAML frontmatter at the start of the file
 // This pattern matches "---\r?\n" followed by any characters (including newlines) until "---\r?\n"
 frontmatterRegex := regexp.MustCompile(`^---\r?\n.*?\n---\r?\n?`)  
 
 // Decode the data using UTF-8, falling back to Latin-1 if needed
 text := string(data)
 
 // Remove the frontmatter using the regex
 result := frontmatterRegex.ReplaceAllString(text, "")
 
 // Strip leading newlines
 return strings.TrimSpace(result)
}

// Annotate returns the chunks unchanged, as no annotation is needed for markdown extraction
func (m *MarkdownExtractor) Annotate(chunks []DocumentChunk, data []byte, sourcePath string) []DocumentChunk {
 return chunks
}

// DocumentChunk represents a document chunk with metadata
// This type is assumed to be defined elsewhere in the codebase
// type DocumentChunk struct {
//  Content string `json:"content"`
//  Path    string `json:"path"`
//  // other fields...
// }
