---
title: 'Chonk: structure-aware chunking and entity-graph retrieval for heterogeneous document corpora'
tags:
  - Python
  - retrieval-augmented generation
  - information retrieval
  - named entity recognition
  - knowledge graphs
  - embeddings
authors:
  - name: Kenneth Stott
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Logick, United States
    index: 1
date: 30 September 2026
bibliography: paper.bib
---

# Summary

Retrieval-augmented generation (RAG) answers a question by retrieving passages
from a document collection and passing them to a language model
[@lewis2020rag]. How well this works depends mostly on the retrieval step: which
passages are cut from the source documents, and which of them are returned for
a given question. Chonk is a Python library for that retrieval step. It loads
documents from files, web pages, databases, and SaaS systems; splits them into
chunks at structural boundaries (headings, table rows, list items, sentences);
extracts named entities; builds an entity co-occurrence graph and communities
over it; and serves hybrid dense, lexical, and graph-guided search over the
result. It stores vectors in DuckDB [@raasveldt2019duckdb] by default, with
PostgreSQL/pgvector, Qdrant, Pinecone, and Weaviate backends behind one
interface. Building the graph uses no LLM calls.

# Statement of need

Two problems motivate Chonk. First, most RAG pipelines cut documents into
fixed-size token windows with overlap. Chunk boundaries then fall mid-table or
mid-paragraph, and the overlap adds duplicate embeddings that must be removed
downstream. Second, enterprise and research corpora are usually heterogeneous:
the same entity appears in financial filings, security advisories, regulations,
and patents under different names ("Apple Inc.", "AAPL", "Apple Computer").
Dense retrieval alone does not resolve these aliases and often returns passages
from the wrong document type.

LLM-based graph RAG systems such as Microsoft GraphRAG [@edge2024graphrag],
HippoRAG [@gutierrez2024hipporag], and LightRAG [@guo2024lightrag] address the
second problem by extracting typed relations with a language model. This costs
LLM calls proportional to corpus size at index time, and the graph must be
re-extracted when documents change. Chonk takes the other route: entities come
from spaCy [@honnibal2020spacy] plus user-supplied vocabularies (database schema
identifiers, glossaries, structured records), edges are untyped co-occurrence
within a chunk, and communities come from Louvain [@blondel2008louvain] or
Leiden [@traag2019leiden] clustering. Chonk is intended for researchers
comparing retrieval strategies and for practitioners who need a retrieval layer
that can be rebuilt incrementally, audited, and run without an LLM at index
time.

# State of the field

General-purpose frameworks such as LlamaIndex and LangChain provide chunkers,
loaders, and retriever abstractions, and delegate graph construction to LLM
calls or external graph databases. The LLM-based graph RAG systems above ship as
research code tied to a specific pipeline. Chonk differs in three ways: its
chunker follows document structure for every supported format rather than
splitting on token counts; its entity graph is built from NLP primitives and
curated vocabularies instead of LLM extraction; and its search returns chunks
with full provenance (source path, section breadcrumb, page, sheet, row range,
or database record) so each answer can be traced to its origin.

# Software design

Chonk is organised around three extension points, each a small protocol with a
first-match registry: `Transport` (fetch bytes from a location), `Extractor`
(turn bytes of a given MIME type into structured sections), and `VectorBackend`
(store and search embeddings). New sources, formats, and stores are added by
registering an implementation; the chunker, NER pipeline, graph builders, and
search do not change. A backend parity test suite runs the same contract tests
against DuckDB, PostgreSQL, Qdrant, and embedded Weaviate.

Search is provided by `EnhancedSearch`, with three modes: vector-first (dense
seed retrieval expanded through structural neighbours, shared entities,
clusters, and communities), graph-first (entity-graph traversal reranked by
vector similarity), and global (search over community summaries). Dense and
BM25 [@robertson2009bm25] results can be merged with reciprocal rank fusion
[@cormack2009rrf]. A completeness gate checks whether the entities named in the
query appear in the retrieved set and expands retrieval until they do or a
budget runs out. Embeddings use Sentence Transformers [@reimers2019sbert]. A
document registry supports incremental sync and pruning of deleted sources. An
optional Model Context Protocol server exposes the index to LLM agents.

# Research impact statement

The author used Chonk to build and evaluate HARE-Bench, a 500-question
cross-domain retrieval benchmark over SEC filings, CVE records, Federal Register
notices, and USPTO patents, and to run ablations on GraphRAG-Bench
[@xiang2025graphragbench]. The benchmark driver, every run configuration, the
questions, the typed gold schemas, the deterministic scorer, and the scored
output of each reported run are included in the repository;
`docs/reproducing-benchmarks.md` explains how to check and re-run them. A paper
describing these results is in preparation.

# AI usage disclosure

Generative AI tools were used in developing this software and its
documentation. The tools were Claude Code (Anthropic) with the models Claude
Sonnet 4.6, Claude Haiku 4.5, Claude Opus 4.8, Claude Opus 5, and Claude Opus
5.5. They were used to write and refactor code, write tests, diagnose failures,
draft documentation, and draft this paper. Commits with AI assistance carry a
`Co-Authored-By` trailer naming the model. Project instructions given to the
tools are in `CLAUDE.md` and `AGENTS.md` in the repository.

The author made the architectural and design decisions: the structure-aware
chunking approach, the LLM-free entity co-occurrence graph, the extension-point
protocols, the retrieval modes and completeness gate, and the benchmark design
and scoring rules. The author reviewed all AI-generated code, tests,
documentation, and paper text, and is responsible for its correctness.

# Acknowledgements

The author thanks the maintainers of the open-source libraries Chonk builds on.

# References
