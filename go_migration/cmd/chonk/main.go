// Command chonk is the CLI entry point for the Go RAG pipeline.
//
// MIGRATION MARKER: New entry point — replaces the Python library-only API.
// The Go binary is self-contained (no Python runtime, no venv, no model download).
//
// Usage:
//
//	chonk ingest   --config config.yaml [files...]
//	chonk search   --config config.yaml --query "your question" [--k 5]
//	chonk ask      --config config.yaml --query "your question" [--k 5]
//	chonk version
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/kennethstott/chonk/internal/ingest"
)

const version = "0.1.0-migration"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}
	cmd := os.Args[1]
	args := os.Args[2:]
	var err error
	switch cmd {
	case "ingest":
		err = runIngest(args)
	case "search":
		err = runSearch(args)
	case "ask":
		err = runAsk(args)
	case "version":
		fmt.Println("chonk", version)
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		printUsage()
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

// ── ingest ────────────────────────────────────────────────────────────────────

func runIngest(args []string) error {
	fs := flag.NewFlagSet("ingest", flag.ContinueOnError)
	cfgPath := fs.String("config", "", "path to YAML config file (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cfgPath == "" {
		return fmt.Errorf("--config is required")
	}
	cfg, err := ingest.LoadConfig(*cfgPath)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	idx, err := ingest.Build(cfg)
	if err != nil {
		return fmt.Errorf("build index: %w", err)
	}
	defer idx.Close()

	ctx := context.Background()
	files := fs.Args()
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "no files specified; nothing to ingest")
		return nil
	}
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", f, err)
			continue
		}
		docType := strings.TrimPrefix(filepath.Ext(f), ".")
		name := strings.TrimSuffix(filepath.Base(f), filepath.Ext(f))
		if err := idx.IngestBytes(ctx, name, data, docType); err != nil {
			fmt.Fprintf(os.Stderr, "ingest %s: %v\n", f, err)
			continue
		}
		fmt.Printf("ingested: %s\n", f)
	}
	return nil
}

// ── search ────────────────────────────────────────────────────────────────────

func runSearch(args []string) error {
	fs := flag.NewFlagSet("search", flag.ContinueOnError)
	cfgPath := fs.String("config", "", "path to YAML config file (required)")
	query := fs.String("query", "", "search query (required)")
	k := fs.Int("k", 5, "number of results")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cfgPath == "" {
		return fmt.Errorf("--config is required")
	}
	if *query == "" {
		return fmt.Errorf("--query is required")
	}
	cfg, err := ingest.LoadConfig(*cfgPath)
	if err != nil {
		return err
	}
	idx, err := ingest.Build(cfg)
	if err != nil {
		return err
	}
	defer idx.Close()

	chunks, err := idx.Search(context.Background(), *query, *k)
	if err != nil {
		return err
	}
	for i, sc := range chunks {
		fmt.Printf("[%d] %.4f  %s\n", i+1, sc.Score, sc.Chunk.DocumentName)
		preview := sc.Chunk.Content
		if len(preview) > 200 {
			preview = preview[:200] + "…"
		}
		fmt.Printf("    %s\n\n", strings.ReplaceAll(preview, "\n", " "))
	}
	return nil
}

// ── ask ───────────────────────────────────────────────────────────────────────

func runAsk(args []string) error {
	fs := flag.NewFlagSet("ask", flag.ContinueOnError)
	cfgPath := fs.String("config", "", "path to YAML config file (required)")
	query := fs.String("query", "", "question to ask (required)")
	k := fs.Int("k", 5, "context chunks")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cfgPath == "" {
		return fmt.Errorf("--config is required")
	}
	if *query == "" {
		return fmt.Errorf("--query is required")
	}
	cfg, err := ingest.LoadConfig(*cfgPath)
	if err != nil {
		return err
	}
	idx, err := ingest.Build(cfg)
	if err != nil {
		return err
	}
	defer idx.Close()

	answer, err := idx.Ask(context.Background(), *query, *k)
	if err != nil {
		return err
	}
	fmt.Println(answer.Text)
	fmt.Printf("\n--- %d source(s) cited ---\n", len(answer.Citations))
	for _, sc := range answer.Citations {
		fmt.Printf("  [%s] %s\n", sc.Provenance, sc.Chunk.DocumentName)
	}
	return nil
}

// ── usage ─────────────────────────────────────────────────────────────────────

func printUsage() {
	fmt.Println(`chonk - a dairy-free RAG pipeline (Go port)

Usage:
  chonk ingest  --config <file.yaml> [file1 file2 ...]
  chonk search  --config <file.yaml> --query "text" [--k N]
  chonk ask     --config <file.yaml> --query "question" [--k N]
  chonk version

Flags:
  --config   path to YAML configuration file
  --query    search query or question
  --k        number of results / context chunks (default 5)

Environment:
  TOGETHER_API_KEY   Together.ai API key (alternative to embed.api_key in config)`)
}
