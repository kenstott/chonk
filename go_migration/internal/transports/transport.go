// Package transports defines the Transport interface and FetchResult.
//
// MIGRATION MARKER: Ported from chonk/transports/_protocol.py and core impls.
// Python source:
//   - chonk/transports/_protocol.py  (45 LOC) — Transport Protocol, FetchResult
//   - chonk/transports/_local.py             — LocalTransport
//   - chonk/transports/_http.py              — HttpTransport
//   - chonk/transports/_s3.py                — S3Transport
//   - chonk/transports/_sftp.py              — SftpTransport
//   - chonk/transports/_directory_crawler.py — DirectoryCrawler
//   - chonk/transports/_web_crawler.py       — WebCrawler
//
// Status: Interface + stubs COMPLETE; implementation TODO per transport.
//
// OUT-OF-SCOPE (per ADR extraction-scope.md — not migrated):
//   GitHub, Gmail, SharePoint, Cassandra, MongoDB, Elasticsearch, Solr,
//   DynamoDB, CosmosDB, Firestore, IMAP, FTP, SQLAlchemy, import_crawler.
//
// Translation notes:
//   - Python Protocol + runtime_checkable → Go interface
//   - Python FetchOptions dataclass → Go FetchOptions struct
//   - Python FetchResult dataclass → Go FetchResult struct
//   - Python can_handle(uri) → Go CanHandle(uri string) bool
package transports

// FetchOptions carries optional parameters forwarded to Transport.Fetch().
// Python source: transports._protocol.FetchOptions
type FetchOptions struct {
	SQL         string
	Headers     map[string]string
	TimeoutSecs int    // default 30
	Profile     string // AWS profile name
	Region      string
	EndpointURL string
	Port        int
	Username    string
	Password    string
	KeyPath     string // SSH private key path for SFTP
}

// FetchResult is the output of any Transport.Fetch() call.
// Python source: transports._protocol.FetchResult
type FetchResult struct {
	Data         []byte
	DetectedMIME string // may be empty if not detected
	SourcePath   string // may be empty
}

// Transport is the Go equivalent of the Python Transport Protocol.
// Implementations: LocalTransport, HttpTransport, S3Transport, SftpTransport,
// DirectoryCrawler, WebCrawler.
type Transport interface {
	// Fetch retrieves the resource at uri and returns its raw bytes.
	// opts may be nil; implementations use sensible defaults.
	Fetch(uri string, opts *FetchOptions) (FetchResult, error)

	// CanHandle returns true if this transport handles the given URI.
	// Implementations use URI scheme, prefix, or pattern matching.
	CanHandle(uri string) bool
}

// Crawler extends Transport with a list-all-URIs operation.
// Python source: transports._crawler_protocol.Crawler
type Crawler interface {
	Transport

	// Crawl returns all URIs discoverable under the root.
	Crawl(root string, opts *FetchOptions) ([]string, error)
}

// Registry holds an ordered list of transports (first-match semantics).
// Python equivalent: DocumentLoader._transports list.
type Registry struct {
	transports []Transport
}

// NewRegistry returns a Registry pre-loaded with core transports in priority order.
func NewRegistry() *Registry {
	r := &Registry{}
	r.Register(LocalTransport{})
	r.Register(HttpTransport{})
	r.Register(S3Transport{})
	r.Register(SftpTransport{})
	return r
}

// Register appends a transport (lowest priority).
func (r *Registry) Register(t Transport) {
	r.transports = append(r.transports, t)
}

// Detect returns the first transport whose CanHandle(uri) is true, or nil.
func (r *Registry) Detect(uri string) Transport {
	for _, t := range r.transports {
		if t.CanHandle(uri) {
			return t
		}
	}
	return nil
}
