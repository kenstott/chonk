// Package schema defines metadata types for relational and API schemas.
//
// MIGRATION MARKER: Ported from chonk/schema.py
// Python source: chonk/schema.py (~50 LOC)
// Status: COMPLETE
//
// Translation notes:
//   - Python @dataclass fields with default None → Go pointer types or empty slices
//   - All Python Optional[T] → *T in Go
package schema

// ColumnMeta describes a single column in a relational table.
// Python source: schema.ColumnMeta
type ColumnMeta struct {
	Name        string
	DataType    string
	Nullable    bool
	IsPrimary   bool
	IsForeign   bool
	Description *string
}

// TableMeta describes a database table with its columns.
// Python source: schema.TableMeta
type TableMeta struct {
	Name        string
	Schema      *string // nil means default schema
	Columns     []ColumnMeta
	Description *string
	RowCount    *int64
}

// FieldMeta describes a field in a JSON/API schema.
// Python source: schema.FieldMeta
type FieldMeta struct {
	Name        string
	FieldType   string
	Required    bool
	Description *string
	Enum        []string
}

// EndpointMeta describes an HTTP API endpoint.
// Python source: schema.EndpointMeta
type EndpointMeta struct {
	Path        string
	Method      string // "GET" | "POST" | "PUT" | "PATCH" | "DELETE"
	Summary     *string
	Parameters  []FieldMeta
	RequestBody []FieldMeta
	Responses   map[string]string // status_code → description
}
