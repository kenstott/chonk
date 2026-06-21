package extractors

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/kennethstott/chonk/internal/models"
)

// CSVExtractor renders CSV/TSV data as a Markdown table.
// Python source: chonk/extractors/_csv.py
// MIGRATION MARKER: Status COMPLETE — dialect sniffing, markdown table render,
// and row-range annotation ported. pandas dependency replaced by encoding/csv.
type CSVExtractor struct{}

// CanHandle returns true for csv and tsv.
func (e CSVExtractor) CanHandle(docType string) bool {
	return docType == "csv" || docType == "tsv"
}

// Extract converts CSV/TSV bytes to a Markdown table string.
func (e CSVExtractor) Extract(data []byte, _ string) (string, error) {
	if len(data) == 0 {
		return "", nil
	}
	text := decodeText(data)
	rows, err := parseCSV(text)
	if err != nil {
		return "", fmt.Errorf("csv parse: %w", err)
	}
	rows = filterEmpty(rows)
	if len(rows) == 0 {
		return "", nil
	}
	return renderMarkdownTable(rows), nil
}

// Annotate adds row_start / row_end metadata to each chunk based on which
// rendered table rows appear in its content.
func (e CSVExtractor) Annotate(chunks []models.DocumentChunk, data []byte, _ string) []models.DocumentChunk {
	if len(data) == 0 || len(chunks) == 0 {
		return chunks
	}
	text := decodeText(data)
	rows, err := parseCSV(text)
	if err != nil || len(rows) < 2 {
		return chunks
	}
	rows = filterEmpty(rows)
	if len(rows) < 2 {
		return chunks
	}
	// Build (1-based index → rendered row) pairs for data rows only.
	type rowEntry struct {
		idx      int
		rendered string
	}
	var entries []rowEntry
	for i, row := range rows[1:] {
		entries = append(entries, rowEntry{idx: i + 1, rendered: renderRow(row)})
	}
	for ci := range chunks {
		content := chunks[ci].Content
		var matched []int
		for _, entry := range entries {
			if strings.Contains(content, entry.rendered) {
				matched = append(matched, entry.idx)
			}
		}
		if len(matched) > 0 {
			if chunks[ci].SourceDetail == nil {
				chunks[ci].SourceDetail = make(map[string]interface{})
			}
			mn, mx := minMax(matched)
			chunks[ci].SourceDetail["row_start"] = mn
			chunks[ci].SourceDetail["row_end"] = mx
		}
	}
	return chunks
}

// ── helpers ───────────────────────────────────────────────────────────────────

func decodeText(data []byte) string {
	if utf8.Valid(data) {
		return string(data)
	}
	return decodeLatin1(data)
}

// decodeLatin1 converts raw bytes to a string treating each byte as a Latin-1
// (ISO-8859-1) code point — safe fallback for non-UTF-8 files.
func decodeLatin1(data []byte) string {
	var sb strings.Builder
	sb.Grow(len(data))
	for _, b := range data {
		sb.WriteRune(rune(b))
	}
	return sb.String()
}

func parseCSV(text string) ([][]string, error) {
	// Detect delimiter: count tabs vs commas in the first 4 KB.
	sample := text
	if len(sample) > 4096 {
		sample = sample[:4096]
	}
	delim := ','
	if strings.Count(sample, "\t") > strings.Count(sample, ",") {
		delim = '\t'
	}
	r := csv.NewReader(strings.NewReader(text))
	r.Comma = rune(delim)
	r.LazyQuotes = true
	r.TrimLeadingSpace = true
	return r.ReadAll()
}

func filterEmpty(rows [][]string) [][]string {
	out := rows[:0]
	for _, row := range rows {
		for _, cell := range row {
			if strings.TrimSpace(cell) != "" {
				out = append(out, row)
				break
			}
		}
	}
	return out
}

func renderMarkdownTable(rows [][]string) string {
	if len(rows) == 0 {
		return ""
	}
	var buf bytes.Buffer
	buf.WriteString(renderRow(rows[0]))
	buf.WriteByte('\n')
	// separator
	seps := make([]string, len(rows[0]))
	for i := range seps {
		seps[i] = "---"
	}
	buf.WriteString("| " + strings.Join(seps, " | ") + " |")
	for _, row := range rows[1:] {
		buf.WriteByte('\n')
		buf.WriteString(renderRow(row))
	}
	return buf.String()
}

func renderRow(row []string) string {
	cells := make([]string, len(row))
	for i, cell := range row {
		cell = strings.ReplaceAll(cell, "|", `\|`)
		cell = strings.ReplaceAll(cell, "\n", " ")
		cells[i] = strings.TrimSpace(cell)
	}
	return "| " + strings.Join(cells, " | ") + " |"
}

func minMax(vals []int) (int, int) {
	mn, mx := vals[0], vals[0]
	for _, v := range vals[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}
