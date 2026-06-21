package transports

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// LocalTransport reads files from the local filesystem.
// Python source: chonk/transports/_local.py
// MIGRATION MARKER: Status COMPLETE
type LocalTransport struct{}

// CanHandle returns true for paths starting with "file://" or absolute/relative
// paths that do not match a network scheme.
func (t LocalTransport) CanHandle(uri string) bool {
	if strings.HasPrefix(uri, "file://") {
		return true
	}
	// Accept absolute paths and relative paths (no "://" scheme separator).
	return !strings.Contains(uri, "://")
}

// Fetch reads the file at uri from the local filesystem.
func (t LocalTransport) Fetch(uri string, _ *FetchOptions) (FetchResult, error) {
	path := strings.TrimPrefix(uri, "file://")
	data, err := os.ReadFile(path)
	if err != nil {
		return FetchResult{}, fmt.Errorf("local transport: %w", err)
	}
	return FetchResult{
		Data:       data,
		SourcePath: filepath.Clean(path),
	}, nil
}
