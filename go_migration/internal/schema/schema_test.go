package schema_test

import (
	"testing"

	"github.com/kennethstott/chonk/internal/schema"
	"github.com/stretchr/testify/assert"
)

func TestTableMeta_NilSchema(t *testing.T) {
	tm := schema.TableMeta{
		Name:    "users",
		Columns: []schema.ColumnMeta{{Name: "id", DataType: "bigint", IsPrimary: true}},
	}
	assert.Nil(t, tm.Schema)
	assert.Nil(t, tm.RowCount)
}

func TestEndpointMeta_ZeroValue(t *testing.T) {
	ep := schema.EndpointMeta{
		Path:   "/api/v1/users",
		Method: "GET",
	}
	assert.Empty(t, ep.Parameters)
	assert.Nil(t, ep.Summary)
}
