# Changelog

All notable changes to `chonk-rag` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version bump may
include breaking API changes.

## [Unreleased]

### Added
- `chonk build <config.yaml> [--store PATH] [--force]` command (also
  `python -m chonk build`).
- HARE-Bench reproduction pipeline under `scripts/hare/`: publish the corpus
  (stage 1) and the GPU-built index (stage 2) to the public bucket, then
  `reproduce.sh` downloads the index, verifies it against the bucket's
  `manifest.json`, and regenerates the results. The bucket is configured
  through environment variables.
- HARE-Bench ingest configs (`work/configs/ingest/`).
- `scripts/hare/fetch_entity_records.py`: builds the legal-entity records from
  the GLEIF golden copy and SEC EDGAR.
- `CONTRIBUTING.md`, `GOVERNANCE.md`, `SUPPORT.md`, and this changelog.
- JOSS paper draft under `paper/`.
- `docs/reproducing-benchmarks.md`.
- `scripts/hare/reproduce.sh --gpu gcp|vultr`: runs the reproduction on a GPU
  instance and copies the results back. On GCP the instance has a provider-side
  deadline and is deleted even if the launching machine dies.

### Fixed
- HARE-Bench run configs pointed at the no-breadcrumb store without the GLEIF
  entity records, so the main runs lacked them and the `*_no_gleif` ablation
  compared identical data. `fang_base.toml` now uses the store that includes them.
- A static domain filter (the `*_no_gleif` runs) only worked if an ADF run had
  tagged the store's chunks with domains first; runs that filter by domain now
  tag the store themselves.
- Benchmark Auto Domain Filter: the classifier always called OpenAI with the
  generator's model name, so runs on Together or Anthropic models failed. It now
  runs on the generator's provider.
- `scripts/hare/reproduce.sh` did not set up the scorer: it now copies the typed
  scorer into the data directory and fetches the GraphRAG-Bench repository the
  evaluator imports from. The evaluator now fails when either is missing; it
  used to print a message and write no scores.
- `scripts/hare/compare_results.py` failed on the score files `run-all` writes;
  it now reads that layout as well as the typed scorer's.
- Benchmark rerank checkpoint: runs that differed only in BM25 or the Auto
  Domain Filter shared one checkpoint, so the second run reused the first run's
  ranking and dropped its own extra candidates. The checkpoint key now includes
  both settings, and a checkpointed ranking is reused only when the question's
  candidate set is unchanged.
- `build()` and `Index.add_source()` embedded raw chunk content and ignored
  `enrich_context`; they now embed the breadcrumb-enriched text, like every
  other embed path.

### Removed
- Local working material (training data, migration plans, scratch notes) is no
  longer tracked in the repository.

## [0.5.5] - 2026-08-15
### Fixed
- `run_worker` had no way to stop, which raced the coordinator test (#21).

## [0.5.4] - 2026-08-15
### Fixed
- Worker-coordinator tests could not run in isolation (#20).

## [0.5.3] - 2026-08-15
### Changed
- The backend parity suite runs against embedded Weaviate, with no service (#19).

## [0.5.2] - 2026-08-15
### Changed
- The backend parity suite runs against Qdrant and PostgreSQL (#18).

## [0.5.1] - 2026-08-15
### Fixed
- `EntityLookup` and `NamespaceEvidence` are exported; near-match works in both
  directions (#17).

## [0.5.0] - 2026-08-14
### Added
- Explanations for why an entity lookup came back empty (#16).

## [0.4.2] - 2026-08-14
### Fixed
- An unrecognised vocabulary entry was dropped silently; it now raises (#15).

## [0.4.1] - 2026-08-14
### Fixed
- `build_ner` could not write to PostgreSQL (#14).

## [0.4.0] - 2026-08-14
### Added
- Glossary vocabulary (#13).
### Fixed
- `clear()` cascade and backend-only `rebuild()` (#13).

## [0.3.1] - 2026-08-14
### Fixed
- `SchemaMatcher` registered entities it could never match (#12).

## [0.3.0] - 2026-08-14
### Added
- Typed entity IDs, namespace scoping, and entity-type chunk filtering (#11).

## [0.2.4] - 2026-08-13
### Changed
- Published on PyPI as `chonk-rag`.

## [0.2.3] - 2026-08-13
### Changed
- Complete PyPI package metadata; publish authenticates with `PYPI_API_TOKEN`.

## [0.2.2] - 2026-08-13
### Fixed
- Release CI downloads the spaCy model before the test gate.

## [0.2.1] - 2026-08-13
### Changed
- PyPI release workflow; NER `row_limit` default.

## [0.2.0] - 2026-08-13
- First tagged release.

[Unreleased]: https://github.com/kenstott/chonk/compare/v0.5.5...HEAD
[0.5.5]: https://github.com/kenstott/chonk/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/kenstott/chonk/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/kenstott/chonk/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/kenstott/chonk/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/kenstott/chonk/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/kenstott/chonk/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/kenstott/chonk/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/kenstott/chonk/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/kenstott/chonk/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/kenstott/chonk/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/kenstott/chonk/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/kenstott/chonk/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/kenstott/chonk/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/kenstott/chonk/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/kenstott/chonk/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kenstott/chonk/releases/tag/v0.2.0
