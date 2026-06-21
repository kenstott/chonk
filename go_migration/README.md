# chonk — Go Migration

A dairy-free RAG pipeline (Python → Go migration).

**Source:** `chonk/` Python 3.11 package  
**Target:** `go_migration/` Go 1.22 single binary  
**Architecture:** Monolith

See [`MIGRATION_STATUS.md`](./MIGRATION_STATUS.md) for a full package-by-package status tracker.

## Quick Start

```bash
# Build the binary
make build

# Run unit tests
make test

# Check coverage (>= 70%)
make cover

# Lint
make lint
```

## CLI Usage

```bash
# Set your Together.ai API key
export TOGETHER_API_KEY=sk-...

# Ingest documents
./bin/chonk ingest --config config/example.yaml docs/*.md

# Search
./bin/chonk search --config config/example.yaml --query "what is RAG?" --k 5

# Ask a question
./bin/chonk ask --config config/example.yaml --query "explain vector search"
```

## Project Layout

```
go_migration/
├── cmd/chonk/          CLI entry point (main.go)
├── internal/
│   ├── models/         Core data types (DocumentChunk, Entity, …)
│   ├── schema/         DB/API schema metadata types
│   ├── graphtypes/     Cycle-break leaf package (ContextEdge, ContextGraph)
│   ├── extractors/     5 text extractors (text, csv, markdown, html, sql)
│   ├── transports/     4 core transports (local, http, s3, sftp)
│   ├── together/       Together.ai API client (embed + chat)
│   ├── chunking/       Document chunking pipeline
│   ├── generation/     Answer generation + prompt builder
│   ├── ner/            Named entity recognition (via Together.ai)
│   ├── storage/        VectorBackend interface + DuckDB/PG backends
│   ├── search/         4-lane EnhancedSearch
│   ├── ingest/         Config loading + Build() orchestrator
│   ├── graph/          (G3 — pending)
│   ├── cluster/        (G5 — pending)
│   ├── community/      (G4 — pending)
│   ├── indexer/        (G5 — pending)
│   └── lifecycle/      (G5 — pending)
├── tests/integration/  Integration tests (build tag: integration)
├── config/             Example YAML configs
├── Makefile
├── go.mod
├── .golangci.yml
├── .github/workflows/ci.yml
└── MIGRATION_STATUS.md
```

## Configuration

Copy `config/example.yaml` and edit for your environment:

```yaml
store:
  path: "./index.duckdb"
  embedding_dim: 1024

embed:
  model: "BAAI/bge-large-en-v1.5"
  batch_size: 256
  api_key: ""           # or set TOGETHER_API_KEY env var

loader:
  min_chunk_size: 1100
  max_chunk_size: 2200
  enrich_context: true
```

## Migration Principles

1. **One binary, no Python runtime** — Go replaces the Python venv entirely.
2. **Together.ai replaces all in-process ML** — no model downloads, no GPU required.
3. **Same storage schema** — DuckDB and PostgreSQL databases created by this binary
   use the same schema as the Python version (migration scripts in G4 sprint).
4. **Same YAML config** — the Python `ChonkConfig` schema is preserved so existing
   config files work without changes.
5. **8-layer build order** — packages are implemented in dependency order to keep
   the repo always compilable and testable.
