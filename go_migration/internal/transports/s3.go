package transports

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awscfg "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// S3Transport fetches objects from Amazon S3.
// Python source: chonk/transports/_s3.py
// MIGRATION MARKER: Status COMPLETE — boto3 replaced by aws-sdk-go-v2.
//
// URI format: s3://bucket/key
type S3Transport struct{}

// CanHandle returns true for s3:// URIs.
func (t S3Transport) CanHandle(uri string) bool {
	return strings.HasPrefix(uri, "s3://")
}

// Fetch downloads the S3 object at uri.
// FetchOptions.Region and FetchOptions.Profile are honoured when set.
func (t S3Transport) Fetch(uri string, opts *FetchOptions) (FetchResult, error) {
	bucket, key, err := parseS3URI(uri)
	if err != nil {
		return FetchResult{}, err
	}
	client, err := t.newClient(opts)
	if err != nil {
		return FetchResult{}, fmt.Errorf("s3 transport: build client: %w", err)
	}
	out, err := client.GetObject(context.Background(), &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return FetchResult{}, fmt.Errorf("s3 transport: GetObject %s: %w", uri, err)
	}
	defer out.Body.Close()
	data, err := io.ReadAll(out.Body)
	if err != nil {
		return FetchResult{}, fmt.Errorf("s3 transport: read body: %w", err)
	}
	var mime string
	if out.ContentType != nil {
		mime = *out.ContentType
	}
	return FetchResult{Data: data, DetectedMIME: mime, SourcePath: uri}, nil
}

func parseS3URI(uri string) (bucket, key string, err error) {
	// Strip "s3://"
	rest := strings.TrimPrefix(uri, "s3://")
	idx := strings.IndexByte(rest, '/')
	if idx < 0 {
		return "", "", fmt.Errorf("s3 transport: invalid URI (no key): %s", uri)
	}
	return rest[:idx], rest[idx+1:], nil
}

func (t S3Transport) newClient(opts *FetchOptions) (*s3.Client, error) {
	var cfgOpts []func(*awscfg.LoadOptions) error
	if opts != nil {
		if opts.Region != "" {
			cfgOpts = append(cfgOpts, awscfg.WithRegion(opts.Region))
		}
		if opts.Profile != "" {
			cfgOpts = append(cfgOpts, awscfg.WithSharedConfigProfile(opts.Profile))
		}
		if opts.EndpointURL != "" {
			// Custom endpoint (e.g. LocalStack)
			ep := opts.EndpointURL
			cfgOpts = append(cfgOpts, awscfg.WithEndpointResolverWithOptions(
				aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...interface{}) (aws.Endpoint, error) {
					return aws.Endpoint{URL: ep, HostnameImmutable: true}, nil
				}),
			))
		}
	}
	cfg, err := awscfg.LoadDefaultConfig(context.Background(), cfgOpts...)
	if err != nil {
		return nil, err
	}
	return s3.NewFromConfig(cfg), nil
}
