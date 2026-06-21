package extractors_test

import (
	"strings"
	"testing"

	"github.com/kennethstott/chonk/internal/extractors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

func TestRegistry_Detect_KnownTypes(t *testing.T) {
	r := extractors.NewRegistry()
	cases := []struct {
		docType string
		wantNil bool
	}{
		{"text", false},
		{"txt", false},
		{"csv", false},
		{"tsv", false},
		{"markdown", false},
		{"md", false},
		{"html", false},
		{"htm", false},
		{"sql", false},
		{"ddl", false},
		{"pdf", true},   // dropped
		{"docx", true},  // dropped
		{"xlsx", true},  // dropped
		{"parquet", true}, // dropped
	}
	for _, tc := range cases {
		e := r.Detect(tc.docType)
		if tc.wantNil {
			assert.Nil(t, e, "expected nil for docType=%s", tc.docType)
		} else {
			assert.NotNil(t, e, "expected extractor for docType=%s", tc.docType)
		}
	}
}

func TestNormalizeType(t *testing.T) {
	assert.Equal(t, "text", extractors.NormalizeType("txt"))
	assert.Equal(t, "text", extractors.NormalizeType("text/plain"))
	assert.Equal(t, "markdown", extractors.NormalizeType("md"))
	assert.Equal(t, "html", extractors.NormalizeType("htm"))
	assert.Equal(t, "csv", extractors.NormalizeType("tsv"))
	assert.Equal(t, "sql", extractors.NormalizeType("ddl"))
}

func TestDetectTypeFromSource(t *testing.T) {
	assert.Equal(t, "markdown", extractors.DetectTypeFromSource("README.md"))
	assert.Equal(t, "csv", extractors.DetectTypeFromSource("/data/file.csv"))
	assert.Equal(t, "html", extractors.DetectTypeFromSource("page.html"))
	assert.Equal(t, "text", extractors.DetectTypeFromSource("noextension"))
	assert.Equal(t, "text", extractors.DetectTypeFromSource(""))
}

// ---------------------------------------------------------------------------
// TextExtractor
// ---------------------------------------------------------------------------

func TestTextExtractor_Extract(t *testing.T) {
	e := extractors.TextExtractor{}
	out, err := e.Extract([]byte("hello world"), "")
	require.NoError(t, err)
	assert.Equal(t, "hello world", out)
}

func TestTextExtractor_Empty(t *testing.T) {
	e := extractors.TextExtractor{}
	out, err := e.Extract(nil, "")
	require.NoError(t, err)
	assert.Equal(t, "", out)
}

func TestTextExtractor_Latin1Fallback(t *testing.T) {
	// 0xFF is not valid UTF-8
	data := []byte{0xFF, 0xFE}
	e := extractors.TextExtractor{}
	out, err := e.Extract(data, "")
	require.NoError(t, err)
	assert.NotEmpty(t, out)
}

// ---------------------------------------------------------------------------
// MarkdownExtractor
// ---------------------------------------------------------------------------

func TestMarkdownExtractor_StripsFrontmatter(t *testing.T) {
	md := "---\ntitle: Test\n---\n# Hello\nBody text"
	e := extractors.MarkdownExtractor{}
	out, err := e.Extract([]byte(md), "")
	require.NoError(t, err)
	assert.False(t, strings.HasPrefix(out, "---"))
	assert.Contains(t, out, "# Hello")
}

func TestMarkdownExtractor_NoFrontmatter(t *testing.T) {
	md := "# Hello\nBody text"
	e := extractors.MarkdownExtractor{}
	out, err := e.Extract([]byte(md), "")
	require.NoError(t, err)
	assert.Equal(t, md, out)
}

func TestMarkdownExtractor_Empty(t *testing.T) {
	e := extractors.MarkdownExtractor{}
	out, err := e.Extract(nil, "")
	require.NoError(t, err)
	assert.Equal(t, "", out)
}

// ---------------------------------------------------------------------------
// CSVExtractor
// ---------------------------------------------------------------------------

func TestCSVExtractor_BasicTable(t *testing.T) {
	data := "name,age\nAlice,30\nBob,25"
	e := extractors.CSVExtractor{}
	out, err := e.Extract([]byte(data), "")
	require.NoError(t, err)
	assert.Contains(t, out, "| name | age |")
	assert.Contains(t, out, "| --- | --- |")
	assert.Contains(t, out, "| Alice | 30 |")
	assert.Contains(t, out, "| Bob | 25 |")
}

func TestCSVExtractor_TSVDialect(t *testing.T) {
	data := "name\tage\nAlice\t30"
	e := extractors.CSVExtractor{}
	out, err := e.Extract([]byte(data), "")
	require.NoError(t, err)
	assert.Contains(t, out, "| name | age |")
}

func TestCSVExtractor_Empty(t *testing.T) {
	e := extractors.CSVExtractor{}
	out, err := e.Extract(nil, "")
	require.NoError(t, err)
	assert.Equal(t, "", out)
}

// ---------------------------------------------------------------------------
// HTMLExtractor
// ---------------------------------------------------------------------------

func TestHTMLExtractor_HeadingsAndText(t *testing.T) {
	html := "<html><body><h1>Title</h1><p>Some text</p></body></html>"
	e := extractors.HTMLExtractor{}
	out, err := e.Extract([]byte(html), "")
	require.NoError(t, err)
	assert.Contains(t, out, "# Title")
	assert.Contains(t, out, "Some text")
}

func TestHTMLExtractor_SkipsScript(t *testing.T) {
	html := "<html><head><script>var x = 1;</script></head><body>visible</body></html>"
	e := extractors.HTMLExtractor{}
	out, err := e.Extract([]byte(html), "")
	require.NoError(t, err)
	assert.NotContains(t, out, "var x")
	assert.Contains(t, out, "visible")
}

func TestHTMLExtractor_Links(t *testing.T) {
	html := `<a href="https://example.com">click here</a>`
	e := extractors.HTMLExtractor{}
	out, err := e.Extract([]byte(html), "")
	require.NoError(t, err)
	assert.Contains(t, out, "[click here](https://example.com)")
}

// ---------------------------------------------------------------------------
// SQLExtractor
// ---------------------------------------------------------------------------

func TestSQLExtractor_StripsBOM(t *testing.T) {
	ddl := "\xef\xbb\xbfCREATE TABLE users (id INT);"
	e := extractors.SQLExtractor{}
	out, err := e.Extract([]byte(ddl), "")
	require.NoError(t, err)
	assert.False(t, strings.HasPrefix(out, "\xef\xbb\xbf"))
	assert.Contains(t, out, "CREATE TABLE")
}

func TestSQLExtractor_Empty(t *testing.T) {
	e := extractors.SQLExtractor{}
	out, err := e.Extract(nil, "")
	require.NoError(t, err)
	assert.Equal(t, "", out)
}
