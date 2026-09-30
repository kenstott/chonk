# Reproducing the benchmark results

Chonk was evaluated on two benchmarks:

- **HARE-Bench** (`fang2026`): 500 questions over a mixed corpus of SEC 10-K
  filings, CVE records, Federal Register notices, USPTO patents, and GLEIF/SEC
  legal-entity records. Scored with the type-aware deterministic scorer in
  `work/score_typed.py`.
- **GraphRAG-Bench** (Medical + Novel): the published benchmark, scored with its
  native `answer_correctness` metric using an LLM judge.

Everything runs through one driver, `demo/graphrag_bench.py`. Every run is fully
described by a TOML config; the TOML is the source of truth for retrieval mode,
`k`, rerank, SRR, ADF, domain filters, and generator model.

## What "reproducible" means here

HARE-Bench is reproducible **from a previously generated corpus and index**.
The corpus and the index stores are published artifacts in a bucket, pinned by
SHA-256 in the bucket's `manifest.json`. Reproducing the benchmark means regenerating the results (answer
generation and scoring) from those artifacts with `scripts/hare/reproduce.sh`.

The index can also be rebuilt from the published corpus (stage 2 below), but
the rebuilt stores will not be byte-identical: embedding and NER models produce
slightly different floating-point results on CPU, CUDA, and Apple Metal, which
can shift retrieval rankings and so make benchmark scores drift. The published
index, with its build environment in `index_build_info.json`, is the reference.
The corpus is not regenerated at all: its source APIs change over time.

## Reproduce HARE-Bench

```bash
uv sync --all-extras --group dev
uv run python -m spacy download en_core_web_sm
cp .env.example .env              # set OPENAI_API_KEY (generator and ADF classifier);
                                  # HARE_PUBLIC_URL already points at the published artifacts

scripts/hare/reproduce.sh         # every run in work/configs/fang
```

`reproduce.sh` is the reproduction method. It downloads `manifest.json` and the
index stores and embedding caches from the bucket, checks every file against the
manifest's SHA-256 digests, generates and scores answers for every
run config into `work/reproduce/`, and prints each run's reported and
reproduced scores side by side. It does no chunking or indexing; the index is a
published artifact.

| Option | Effect |
|---|---|
| `--config-dir DIR` | Run only the configs in `DIR` |
| `--out-dir DIR` | Working directory (default `work/reproduce`) |

Each 500-question run with gpt-4o-mini costs roughly US$5–10 in API fees, and
`work/configs/fang` holds 50 runs. To reproduce one run, put its TOML alone in a
directory and pass `--config-dir`. The TOMLs use
`extends = "../fang_base.toml"`, so the directory must sit next to
`work/configs/fang_base.toml`:

```bash
mkdir -p work/configs/one
cp work/configs/fang/fang_ner_ref_bc_laned60_community_k30_srr_bm25_mini_adf.toml work/configs/one/
scripts/hare/reproduce.sh --config-dir work/configs/one
```

A GPU is recommended for embedding and reranking.

## How the artifacts are produced

Initialization and benchmarking are independent activities. The bucket is the
only thing they share: initialization writes the corpus, the index, and
`manifest.json` to it; the benchmark reads them from it. Nothing passes through
git.

| Activity | Script | Reads | Writes to the bucket |
|---|---|---|---|
| Initialization, stage 1 | `scripts/hare/01_publish_sources.sh` | bucket (`corpus/`), GLEIF, SEC EDGAR | `corpus/`, `manifest.json` |
| Initialization, stage 2 | `scripts/hare/02_publish_index.sh` | bucket (`corpus/`) | `index/`, `manifest.json` |
| Benchmark | `scripts/hare/reproduce.sh` | bucket (`index/`, `manifest.json`) | — |

The bucket and its credentials come from the environment (or `.env`):

| Variable | Used by | Purpose |
|---|---|---|
| `HARE_BUCKET_URL` | initialization | `s3://<bucket>/<prefix>` to publish to |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | initialization | S3 credentials |
| `AWS_ENDPOINT_OVERRIDE`, `AWS_REGION` | initialization | S3-compatible endpoint, e.g. Cloudflare R2 (`auto` region) |
| `HARE_PUBLIC_URL` | benchmark | HTTPS URL serving the same prefix; no credentials needed |
| `SEC_USER_AGENT` | stage 1 | `"Name email@example.com"` (SEC fair-access policy) |

Without `HARE_PUBLIC_URL`, downloads use the S3 API and credentials. The
published artifacts are at `s3://chonk/benchmark/fang2026` (Cloudflare R2),
served publicly at `https://chonk.simpleishard.io/benchmark/fang2026/`.

Stage 2 builds the index on the local machine and records the build environment
(platform, accelerator, library versions, git commit, uncommitted paths) in
`index_build_info.json`, published with the stores. `--gpu` builds on a Vultr
GPU instance instead (needs `VULTR_API_KEY`). Each stage uploads its files
first and `manifest.json` last, so the manifest never names a missing file.

### Corpus (`work/corpus/`)

| Directory | Source | Contents |
|---|---|---|
| `10k/` | SEC EDGAR | 10-K filings of Apple, Amazon, Alphabet, Meta, Netflix |
| `cve/` | NIST NVD | 872 CVE records naming those vendors |
| `fedreg/` | Federal Register | 33 notices and rules |
| `patents/` | USPTO (PatentsView) | 3,917 patent grants, as a DuckDB table |
| `gleif/` | GLEIF golden copy + SEC EDGAR | 173 legal-entity records |

The first four were captured once and are published as-is. `gleif/` is
generated by `scripts/hare/fetch_entity_records.py`: the five parent companies
plus every entity the GLEIF Level 2 relationship file reports as consolidated
under them, one Markdown document per entity with its LEI, names, jurisdiction,
addresses, parents, subsidiaries, and, for the parents, the SEC EDGAR record
(CIK, tickers, former names). `gleif/raw/` holds the selected golden-copy rows,
the raw SEC responses, and `provenance.json` with the SHA-256 of the golden-copy
files used. GLEIF republishes the golden copy three times a day and keeps no
public archive, so the published copy is the reference.

### Index (`work/fang2026/data/`)

`scripts/hare/build_index.sh OUT_DIR` builds these from the corpus:

| Store | Ingest config | Used by |
|---|---|---|
| `chonk_bc_1100_2200.duckdb` | `work/configs/ingest/fang2026_bc.yaml` | `*_bc_*` configs |
| `chonk_nobc_1100_2200.duckdb` | `work/configs/ingest/fang2026_nobc.yaml` | configs without `db_name` |
| `chonk_nobc_1100_2200_gleif.duckdb` | `work/configs/ingest/fang2026_nobc_gleif.yaml` | source of the vanilla store |
| `vanilla_rag.duckdb` | `graphrag_bench.py index-vanilla --from-store` | `fang_vanilla_*` configs |

For each of the first three it runs `chonk build` (ingest, embed, full-text
index), then `graphrag_bench.py build-ner --with-embeddings` and
`build-community`. It then builds the vanilla store and the question and entity
embedding caches (`prime-cache`).

## Relationship to the reported numbers

The results in `work/fang2026/results/` were produced before this pipeline
existed, from index stores and entity records that were later lost. The ingest
configs and entity records were rebuilt:

- The ingest configs were reconstructed from the corpus layout and the
  document-name patterns the benchmark uses to assign domains. The patent SQL
  and the per-source breadcrumb settings are inferred.
- The original entity records covered an unknown set of entities in an unknown
  layout. The rebuilt set is defined by the rule above.
- `chonk.build()` previously embedded raw chunk text and ignored breadcrumbs, so
  the original breadcrumb and no-breadcrumb stores may have had identical
  embeddings. The rebuilt stores embed the breadcrumb text.

Scores from `reproduce.sh` are therefore expected to be close to, not identical
with, the reported ones. From now on, the published artifacts are the
reference: every run of `reproduce.sh` uses byte-identical index stores, so
differences come only from generation (LLM sampling and API model updates).

## Inspect the reported numbers

Each `bench_eval_<run>_rp.json` has an `overall` score, a `by_type` breakdown
over the five question types, and a `per_question` list with the gold answer
and per-question score. The run name encodes the configuration; the matching
config is `work/configs/fang/<run name without the _rp suffix>.toml`.

## Scoring

HARE-Bench scores are deterministic for the `boolean`, `number`, `date`, and
`entity` check types. The `text` check type uses cosine similarity between
sentence embeddings. A correct answer scores 1.0, an explicit abstention 0.3,
and a wrong or hallucinated answer 0.0. The reported metric is the mean of the
five per-type means, each over 100 questions.

To re-score a generation file with the typed scorer alone:

```bash
uv run python work/score_typed.py \
  --schemas work/fang2026/data/fang2026_gold_schemas.jsonl \
  --results work/reproduce/results/<run>_rp.jsonl \
  --out work/reproduce/results/<run>_typed_scores.json
```

## GraphRAG-Bench

```bash
uv run python demo/graphrag_bench.py download    --out-dir work
uv run python demo/graphrag_bench.py index       --out-dir work
uv run python demo/graphrag_bench.py prime-cache --out-dir work
uv run python demo/graphrag_bench.py run-all --config-dir work/configs/runs --out-dir work
uv run python demo/graphrag_bench.py report      --out-dir work
```

## Known differences from published baselines

The benchmark specification fixes only the generator and judge models
(gpt-4o-mini). Published GraphRAG systems do not disclose their chunking, entity
extraction, or community-detection settings, so scores are comparable but not
identical conditions. See the "Replication notes" table in the README.
