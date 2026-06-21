package transports

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HttpTransport fetches resources over HTTP / HTTPS.
// Python source: chonk/transports/_http.py
// MIGRATION MARKER: Status COMPLETE — requests library replaced by net/http stdlib.
type HttpTransport struct {
	// Client allows injection of a custom HTTP client for testing.
	// Defaults to a client with 30s timeout.
	Client *http.Client
}

// CanHandle returns true for http:// and https:// URIs.
func (t HttpTransport) CanHandle(uri string) bool {
	return strings.HasPrefix(uri, "http://") || strings.HasPrefix(uri, "https://")
}

// Fetch downloads the resource at uri using HTTP GET.
func (t HttpTransport) Fetch(uri string, opts *FetchOptions) (FetchResult, error) {
	client := t.client(opts)
	req, err := http.NewRequest(http.MethodGet, uri, nil)
	if err != nil {
		return FetchResult{}, fmt.Errorf("http transport: build request: %w", err)
	}
	if opts != nil {
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
	}
	resp, err := client.Do(req)
	if err != nil {
		return FetchResult{}, fmt.Errorf("http transport: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return FetchResult{}, fmt.Errorf("http transport: status %d for %s", resp.StatusCode, uri)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return FetchResult{}, fmt.Errorf("http transport: read body: %w", err)
	}
	return FetchResult{
		Data:         data,
		DetectedMIME: resp.Header.Get("Content-Type"),
		SourcePath:   uri,
	}, nil
}

func (t HttpTransport) client(opts *FetchOptions) *http.Client {
	if t.Client != nil {
		return t.Client
	}
	timeout := 30 * time.Second
	if opts != nil && opts.TimeoutSecs > 0 {
		timeout = time.Duration(opts.TimeoutSecs) * time.Second
	}
	return &http.Client{Timeout: timeout}
}
