# Extraction Scope

## Supported Formats (Go)

Five formats are supported in the Go port. All others are dropped.

| Format | Extension(s) | Python source | Go file |
|---|---|---|---|
| Plain text | `.txt`, `.log`, `.text` | `extractors/_text.py` | `extractors/text.go` |
| CSV | `.csv`, `.tsv` | `extractors/_csv.py` | `extractors/csv.go` |
| Markdown | `.md`, `.markdown` | `extractors/_markdown.py` | `extractors/markdown.go` |
| Database schema | `.sql`, `.db`, `.sqlite`, `.duckdb` | `transports/_db_schema.py` | `extractors/db.go` |
| HTML | `.html`, `.htm` | `extractors/_html.py` | `extractors/html.go` |

## Dropped Formats

The following Python extractors have no Go equivalent in the initial port.
Code referencing them must be removed or stubbed.

| Python extractor | Format |
|---|---|
| `_pdf.py` | PDF (pypdf) |
| `_docx.py` | Microsoft Word (python-docx) |
| `_xlsx.py` | Microsoft Excel (openpyxl) |
| `_pptx.py` | Microsoft PowerPoint (python-pptx) |
| `_yaml.py` | YAML (pyyaml) |
| `_odf.py` | OpenDocument (odfpy) |
| `_json.py` | JSON |
| `_xml.py` | XML |
| `_parquet.py` | Parquet (pyarrow) |
| `_python.py` | Python source (AST) |
| `_typescript.py` | TypeScript/JavaScript |
| `_java.py` | Java source |
| `_fhir.py` | HL7 FHIR |
| `_edgar.py` | SEC EDGAR |
| `_email.py` | MIME email |
| `_mime.py` | MIME |
| `_clinical_trial.py` | ClinicalTrials.gov |
| `_fda_label.py` | FDA drug labels |
| `_attack.py` | MITRE ATT&CK |
| `_cve.py` | CVE |
| `_cwe.py` | CWE |
| `_nist.py` | NIST NVD |
| `_protocol.py` | Protocol buffers |
| `_nosql.py` | NoSQL schema |
| `_lookup_table.py` | Lookup tables |
| `_renderer.py` | Render utilities |

## Extractor Interface (Go)

All extractors implement a single interface. The registry maps file extension to
extractor; the `DocumentLoader` calls `CanHandle` then `Extract`.

```go
// Extractor converts raw file bytes into DocumentChunks.
type Extractor interface {
    // CanHandle returns true if this extractor owns the given filename.
    CanHandle(filename string) bool
    // Extract parses content and returns chunks. Never returns nil slice on success.
    Extract(filename string, content []byte) ([]models.DocumentChunk, error)
}
```

## Format Details

### Plain Text (`text.go`)

- Split on blank lines to form paragraphs.
- Apply `chunking.ChunkDocument` with configurable min/max sizes.
- No heading detection; section path stays empty.
- Mirrors `extractors/_text.py` — the simplest extractor.

### CSV (`csv.go`)

- Use `encoding/csv` from the standard library (replaces `pandas`).
- Each row becomes a chunk with `section` = `[header_row_joined]`.
- Schema/column header detection: first row treated as header unless all values
  are numeric.
- Mirrors `extractors/_csv.py` row-per-chunk strategy.

### Markdown (`markdown.go`)

- Parse ATX headings (`# h1`, `## h2`, …) to build section paths.
- Fenced code blocks kept intact as single chunks regardless of size.
- Tables kept intact as single chunks.
- Mirrors `extractors/_markdown.py` / `chunking.extract_markdown_sections`.
- Standard library only: `bufio`, `strings`, `regexp`.

### Database Schema (`db.go`)

- Accept `.sql` (DDL text) and `.db`/`.sqlite`/`.duckdb` (live connection via
  `database/sql` or `go-duckdb`).
- For DDL text: parse `CREATE TABLE` statements; each table becomes a chunk with
  its column definitions.
- For live databases: introspect `information_schema` or SQLite `sqlite_master`.
- Mirrors `transports/_db_schema.py` schema-extraction logic.

### HTML (`html.go`)

- Use `golang.org/x/net/html` for parsing (replaces BeautifulSoup-style logic).
- Extract visible text, strip scripts and styles.
- Map `<h1>`–`<h6>` to section path.
- Each logical section becomes a chunk.
- Mirrors `extractors/_html.py`.

## Registry

```go
var DefaultRegistry = NewRegistry(
    &TextExtractor{},
    &CSVExtractor{},
    &MarkdownExtractor{},
    &DBExtractor{},
    &HTMLExtractor{},
)

type Registry struct {
    extractors []Extractor
}

func (r *Registry) For(filename string) (Extractor, bool) {
    for _, e := range r.extractors {
        if e.CanHandle(filename) {
            return e, true
        }
    }
    return nil, false
}
```

## DocumentLoader

`DocumentLoader` in Go mirrors the Python `DocumentLoader` class. It accepts a
`Registry`, calls `For(filename)` to pick an extractor, then passes the result
through `context.EnrichChunks` if `EnrichContext` is true.

```go
type LoaderOptions struct {
    MinChunkSize int
    MaxChunkSize int
    EnrichContext bool
}

type DocumentLoader struct {
    Registry *Registry
    Options  LoaderOptions
}

func (l *DocumentLoader) Load(filename string, content []byte) ([]models.DocumentChunk, error)
func (l *DocumentLoader) LoadFile(path string) ([]models.DocumentChunk, error)
```
