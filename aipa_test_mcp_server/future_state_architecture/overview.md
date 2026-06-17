# Future State Architecture — Overview

## Goal

Rewrite the Chonk RAG pipeline in Go 1.22+, preserving the existing Python architecture's
module boundaries and abstractions. The Go version targets production deployments: lower
memory footprint, no Python interpreter, no native ML dependencies on the host.

## Guiding Principles

- **Same architecture, different runtime.** Go packages mirror the Python sub-packages
  (`chunking`, `context`, `storage`, `search`, `extractors`, `transports`, `ner`,
  `graph`, `community`, `cluster`, `generation`, `ingest`). Public interfaces stay
  conceptually identical; implementation adapts to Go idioms.
- **Extraction scope is reduced.** Only five formats are supported: plain text, CSV,
  Markdown, SQL/database schemas, and HTML. All other extractors (PDF, DOCX, XLSX,
  PPTX, YAML, ODF, FHIR, EDGAR, and the domain-specific ones) are dropped.
- **No local ML.** `sentence-transformers`, `torch`, `spacy`, `scikit-learn`,
  `igraph`, and `leidenalg` are replaced entirely by the Together.ai API. Embedding,
  community summarisation, and NER inference are remote calls, not in-process models.
- **Single binary.** The Go build produces one self-contained executable with no pip
  install, no venv, and no model download on first run.

## Document Index

| Document | Topic |
|---|---|
| [code-conversion.md](code-conversion.md) | Module-by-module conversion guide and Go package layout |
| [extraction-scope.md](extraction-scope.md) | Supported formats, dropped formats, extractor interface |
| [ml-services.md](ml-services.md) | Together.ai integration replacing all local ML libraries |
| [conversion-plan.md](conversion-plan.md) | Layer-by-layer build order derived from dependency DAG |
| [unit-testing.md](unit-testing.md) | Go unit test patterns per layer, mapped from Python test templates |

## High-Level Module Map

```
chonk-go/
├── models/          ← DocumentChunk, Entity, ScoredChunk, … (pure structs)
├── chunking/        ← chunk_document(), section/heading logic
├── context/         ← enrich_chunk(), breadcrumb injection
├── extractors/      ← text, csv, md, db, html only
├── transports/      ← local, s3, sftp, http, directory crawler, db
├── storage/         ← DuckDB (go-duckdb) + pgvector backends
├── indexer/         ← background goroutine indexer
├── ner/             ← entity extraction via Together.ai (replaces spacy)
├── graph/           ← SVO extraction via Together.ai LLM
├── cluster/         ← agglomerative clustering via cosine similarity
├── community/       ← Leiden/Louvain via together embed + local graph
├── search/          ← enhanced 4-lane search (vector, structural, entity, cluster)
├── generation/      ← answer synthesis via Together.ai chat
└── ingest/          ← config-driven build pipeline, Index façade
```
