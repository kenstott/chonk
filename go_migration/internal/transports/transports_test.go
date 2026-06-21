package transports_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kennethstott/chonk/internal/transports"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

func TestRegistry_DetectLocal(t *testing.T) {
	r := transports.NewRegistry()
	tr := r.Detect("/path/to/file.txt")
	assert.NotNil(t, tr)
}

func TestRegistry_DetectHTTP(t *testing.T) {
	r := transports.NewRegistry()
	tr := r.Detect("https://example.com/doc.html")
	assert.NotNil(t, tr)
}

func TestRegistry_DetectS3(t *testing.T) {
	r := transports.NewRegistry()
	tr := r.Detect("s3://my-bucket/my-key")
	assert.NotNil(t, tr)
}

func TestRegistry_NoMatch(t *testing.T) {
	r := transports.NewRegistry()
	tr := r.Detect("mongodb://localhost/db")
	assert.Nil(t, tr)
}

// ---------------------------------------------------------------------------
// LocalTransport
// ---------------------------------------------------------------------------

func TestLocalTransport_CanHandle(t *testing.T) {
	tr := transports.LocalTransport{}
	assert.True(t, tr.CanHandle("/tmp/file.txt"))
	assert.True(t, tr.CanHandle("file:///tmp/file.txt"))
	assert.True(t, tr.CanHandle("relative/path.txt"))
	assert.False(t, tr.CanHandle("s3://bucket/key"))
	assert.False(t, tr.CanHandle("https://example.com"))
}

func TestLocalTransport_FetchFile(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "test.txt")
	require.NoError(t, os.WriteFile(fpath, []byte("hello chonk"), 0600))

	tr := transports.LocalTransport{}
	result, err := tr.Fetch(fpath, nil)
	require.NoError(t, err)
	assert.Equal(t, []byte("hello chonk"), result.Data)
	assert.Equal(t, filepath.Clean(fpath), result.SourcePath)
}

func TestLocalTransport_FetchFileURI(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "test.txt")
	require.NoError(t, os.WriteFile(fpath, []byte("content"), 0600))

	tr := transports.LocalTransport{}
	result, err := tr.Fetch("file://"+fpath, nil)
	require.NoError(t, err)
	assert.Equal(t, []byte("content"), result.Data)
}

func TestLocalTransport_FetchMissingFile(t *testing.T) {
	tr := transports.LocalTransport{}
	_, err := tr.Fetch("/nonexistent/path.txt", nil)
	require.Error(t, err)
}

// ---------------------------------------------------------------------------
// HttpTransport
// ---------------------------------------------------------------------------

func TestHttpTransport_CanHandle(t *testing.T) {
	tr := transports.HttpTransport{}
	assert.True(t, tr.CanHandle("http://example.com"))
	assert.True(t, tr.CanHandle("https://example.com"))
	assert.False(t, tr.CanHandle("s3://bucket/key"))
	assert.False(t, tr.CanHandle("/local/path"))
}

func TestHttpTransport_FetchOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, "test content")
	}))
	defer srv.Close()

	tr := transports.HttpTransport{Client: srv.Client()}
	result, err := tr.Fetch(srv.URL+"/doc.txt", nil)
	require.NoError(t, err)
	assert.Equal(t, []byte("test content"), result.Data)
	assert.Contains(t, result.DetectedMIME, "text/plain")
}

func TestHttpTransport_Fetch404(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(404)
	}))
	defer srv.Close()

	tr := transports.HttpTransport{Client: srv.Client()}
	_, err := tr.Fetch(srv.URL+"/missing", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "404")
}

func TestHttpTransport_CustomHeaders(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		fmt.Fprint(w, "ok")
	}))
	defer srv.Close()

	tr := transports.HttpTransport{Client: srv.Client()}
	opts := &transports.FetchOptions{
		Headers: map[string]string{"Authorization": "Bearer token123"},
	}
	_, err := tr.Fetch(srv.URL+"/", opts)
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(gotAuth, "Bearer"))
}
