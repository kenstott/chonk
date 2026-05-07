# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7a4e2b91-3c88-4f02-b5d1-e920c7f84a3d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
Chunky Monkey — GraphRAG-Bench evaluation.

Evaluates contextual chunking against published GraphRAG-Bench leaderboard.
Uses BGE-large-en-v1.5 embeddings + benchmark's native answer_correctness metric.

Usage:
    python demo/graphrag_bench.py download      --out-dir /tmp/grb
    python demo/graphrag_bench.py inspect       --out-dir /tmp/grb
    python demo/graphrag_bench.py index         --out-dir /tmp/grb [--force]
    python demo/graphrag_bench.py index-vanilla --out-dir /tmp/grb [--force]
    python demo/graphrag_bench.py run           --out-dir /tmp/grb [--rerank] [--enhanced] [--vanilla] [--run-name NAME]
    python demo/graphrag_bench.py eval          --out-dir /tmp/grb --run-name <name>
    python demo/graphrag_bench.py report        --out-dir /tmp/grb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Load .env from project root before anything else
_PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(_PROJECT_ROOT))
from chonk import NOVEL_STRUCTURAL_LEVELS, chunk_document, promote_plain_text_headers
from chonk.context import enrich_chunks
from chonk.storage._store import Store

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EMBED_MODEL        = "BAAI/bge-large-en-v1.5"
EMBED_DIM          = 1024
GEN_MODEL          = "gpt-4o-mini"
GEN_MODEL_TOGETHER = "Qwen/Qwen2.5-72B-Instruct-Turbo"   # closest serverless Qwen2.5 (14B not available serverless on Together)
TOGETHER_BASE_URL  = "https://api.together.xyz/v1"
K             = 5
K_FETCH       = 20    # candidates to retrieve before reranking (ignored when --rerank is off)
RERANK_MODEL         = "BAAI/bge-reranker-large"       # local cross-encoder
RERANK_MODEL_TOGETHER = "Salesforce/Llama-Rank-V1.1"   # Together AI reranker API
RERANK_MODEL_COHERE  = "rerank-english-v3.0"           # Cohere reranker API
SPACY_MODEL   = "en_core_web_sm"
MIN_CHUNK     = 400
MAX_CHUNK     = 1200
DATASET_NAME  = "GraphRAG-Bench/GraphRAG-Bench"
SUBSETS       = ["medical", "novel"]
DB_FILENAME         = "chonk.duckdb"
VANILLA_DB_FILENAME = "vanilla_rag.duckdb"
VANILLA_K             = 5     # paper Appendix H.2: retrieval_topk=5
VANILLA_CHUNK_TOKENS  = 256   # benchmark uses 256-token chunks
VANILLA_CHUNK_OVERLAP = 0
VANILLA_TEMPERATURE   = 0.7   # paper: "generation temperature of 0.7"

def _model_rpm_limit(model: str) -> int:
    """Return 80%-of-tier-4 RPM limit for a given OpenAI model, loaded from openai_rpm_limits.json."""
    _limits_file = Path(__file__).parent / "openai_rpm_limits.json"
    try:
        _data = json.loads(_limits_file.read_text())
        return _data.get(model, _data.get("_default", 2400))
    except Exception:
        return 2400

# Published leaderboard results scraped from graphrag-bench.github.io, April 2026.
# Avg = mean(Fact ACC, Reason ACC, Summ ACC, Creative ACC) for each subset.
# Original scale is 0–100%; stored here as 0–1.
# Published leaderboard uses gpt-4o-mini generator + gpt-4o-mini judge (answer_correctness metric).
PUBLISHED_BASELINES = {
    # ── Top methods ───────────────────────────────────────────────────────────
    "G-reasoner":               {"med_acc": 0.7330, "nov_acc": 0.5894, "overall": 0.6612},
    "AutoPrunedRetriever-llm":  {"med_acc": 0.6700, "nov_acc": 0.6372, "overall": 0.6536},
    "HippoRAG2":                {"med_acc": 0.6485, "nov_acc": 0.5648, "overall": 0.6067},
    "Fast-GraphRAG":            {"med_acc": 0.6412, "nov_acc": 0.5202, "overall": 0.5807},
    "LightRAG":                 {"med_acc": 0.6259, "nov_acc": 0.4509, "overall": 0.5384},
    # ── Vanilla RAG baselines ─────────────────────────────────────────────────
    "RAG (w/ rerank)":          {"med_acc": 0.6243, "nov_acc": 0.4835, "overall": 0.5539},
    "RAG (w/o rerank)":         {"med_acc": 0.6100, "nov_acc": 0.4793, "overall": 0.5447},
    # ── Other methods ─────────────────────────────────────────────────────────
    "RAPTOR":                   {"med_acc": 0.5710, "nov_acc": 0.4324, "overall": 0.5017},
    "MS-GraphRAG (local)":      {"med_acc": 0.4516, "nov_acc": 0.5093, "overall": 0.4805},
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Download
# ─────────────────────────────────────────────────────────────────────────────

def cmd_download(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    from datasets import load_dataset

    for subset in SUBSETS:
        out_file = data_dir / f"{subset}_questions.jsonl"
        if out_file.exists():
            print(f"  {subset}: already downloaded ({out_file})")
            continue
        print(f"  Downloading {subset} subset...")
        ds = load_dataset(DATASET_NAME, subset, split="train", trust_remote_code=True)
        with open(out_file, "w") as f:
            for record in ds:
                f.write(json.dumps(record) + "\n")
        print(f"  {subset}: {len(ds)} questions → {out_file}")

    repo_dir = out_dir / "GraphRAG-Benchmark"
    if not repo_dir.exists():
        print("\nCloning GraphRAG-Benchmark repo...")
        ret = os.system(f"git clone --depth=1 https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git {repo_dir} 2>&1")
        if ret != 0:
            print("  WARNING: git clone failed — evaluation scripts unavailable")
    else:
        print(f"\nRepo already cloned at {repo_dir}")

    print("\nDownload complete. Run 'inspect' to check corpus availability.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.5: Inspect
# ─────────────────────────────────────────────────────────────────────────────

def _load_questions(data_dir: Path) -> list[dict]:
    questions = []
    for subset in SUBSETS:
        f = data_dir / f"{subset}_questions.jsonl"
        if f.exists():
            with open(f) as fh:
                for line in fh:
                    questions.append(json.loads(line))
    return questions


def cmd_inspect(args: argparse.Namespace) -> None:
    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    repo_dir = out_dir / "GraphRAG-Benchmark"

    questions = _load_questions(data_dir)
    if not questions:
        print("No questions found. Run 'download' first.")
        return

    print(f"Total questions: {len(questions)}")
    by_subset = defaultdict(list)
    by_type   = defaultdict(list)
    for q in questions:
        by_subset[q.get("source", q.get("subset", "?"))].append(q)
        by_type[q.get("question_type", "?")].append(q)

    print("\nBy subset:")
    for k, v in sorted(by_subset.items()):
        print(f"  {k}: {len(v)}")
    print("\nBy question type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {len(v)}")

    q = questions[0]
    print(f"\nRecord keys: {list(q.keys())}")
    print(f"\nSample question ({q.get('source','?')} / {q.get('question_type','?')}):")
    print(f"  Q: {q['question'][:120]}")
    print(f"  A: {str(q['answer'])[:120]}")
    ev = q.get("evidence", [])
    print(f"  Evidence passages: {len(ev)}")
    if ev:
        print(f"  Evidence[0]: {str(ev[0])[:200]}")

    print("\n── Source corpus check ──")
    corpus_dir = repo_dir / "Datasets"
    corpus_files = []
    if corpus_dir.exists():
        for root, _, files in os.walk(corpus_dir):
            for fname in files:
                p = Path(root) / fname
                corpus_files.append(p)
        print(f"  Files in Datasets/: {len(corpus_files)}")
        for p in corpus_files[:20]:
            print(f"    {p.relative_to(repo_dir)}  ({p.stat().st_size:,} bytes)")
        if len(corpus_files) > 20:
            print(f"    ... and {len(corpus_files)-20} more")
    else:
        print("  Datasets/ directory not found in cloned repo.")

    ev_chars = sum(len(str(p)) for q in questions for p in q.get("evidence", []))
    print("\n  Evidence reconstruction fallback:")
    print(f"    Total evidence chars across all questions: {ev_chars:,}")
    seen: set[str] = set()
    for q in questions:
        for p in q.get("evidence", []):
            seen.add(str(p).strip())
    print(f"    {len(seen):,} unique passages, ~{sum(len(s) for s in seen)//1000:,}K chars total")

    if not corpus_files:
        print("\n  GATE: No raw corpus found. Will reconstruct from evidence passages.")
    else:
        print("\n  GATE: Source corpus available. Proceeding with full chunking.")

    info = {
        "n_questions":   len(questions),
        "by_subset":     {k: len(v) for k, v in by_subset.items()},
        "by_type":       {k: len(v) for k, v in by_type.items()},
        "corpus_files":  [str(p) for p in corpus_files],
        "n_evidence":    len(seen),
        "corpus_source": "repo" if corpus_files else "evidence_reconstruction",
    }
    (out_dir / "corpus_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nSaved corpus_info.json")


# ─────────────────────────────────────────────────────────────────────────────
# Corpus builder (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _build_corpus(out_dir: Path) -> list[tuple[str, str]]:
    """Returns list of (doc_id, text)."""
    info_file = out_dir / "corpus_info.json"
    if not info_file.exists():
        raise RuntimeError("Run 'inspect' first.")

    info     = json.loads(info_file.read_text())
    data_dir = out_dir / "data"

    if info["corpus_source"] == "repo" and info["corpus_files"]:
        corpus_files = [Path(p) for p in info["corpus_files"]
                        if "Corpus" in p and p.endswith(".json")]
        docs = []
        for path in corpus_files:
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            records = raw if isinstance(raw, list) else [raw]
            for rec in records:
                name = rec.get("corpus_name", path.stem)
                text = rec.get("context", "")
                if isinstance(text, str) and text.strip():
                    docs.append((name, text))
        if docs:
            return docs

    print("  Using evidence reconstruction (no raw corpus available)")
    questions = _load_questions(data_dir)
    seen: dict[str, str] = {}
    for q in questions:
        src = q.get("source", q.get("subset", "unknown"))
        for passage in q.get("evidence", []):
            text = str(passage).strip()
            if text and text not in seen:
                doc_id = f"{src}_ev_{len(seen)}"
                seen[text] = doc_id
    return [(doc_id, text) for text, doc_id in seen.items()]


# ─────────────────────────────────────────────────────────────────────────────
# Naive chunker (vanilla RAG baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _naive_chunks(text: str,
                  chunk_tokens: int = VANILLA_CHUNK_TOKENS,
                  overlap_tokens: int = VANILLA_CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size token chunks with overlap using BGE BERT tokenizer."""
    from transformers import AutoTokenizer
    enc = AutoTokenizer.from_pretrained(EMBED_MODEL)
    tokens = enc.encode(text, add_special_tokens=False)
    step = max(1, chunk_tokens - overlap_tokens)
    result = []
    for start in range(0, len(tokens), step):
        chunk = enc.decode(tokens[start:start + chunk_tokens])
        if chunk.strip():
            result.append(chunk)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Index (chunk + embed + store via chonk)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build_ner(args: argparse.Namespace) -> None:
    """Run NER on an existing index and persist chunk_entities to DB."""
    import duckdb
    data_dir = Path(args.out_dir) / "data"
    db_path  = data_dir / args.db_name
    force    = getattr(args, "force", False)
    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    n = con.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()[0]
    con.close()
    if n > 0 and not force:
        print(f"chunk_entities already populated ({n:,} rows) — skipping. Use --force to rebuild.")
    else:
        with Store(db_path, embedding_dim=EMBED_DIM) as store:
            entity_index = _build_entity_index_from_store(store)
        _persist_entity_index(entity_index, db_path)

    # Build entity embeddings if requested or not yet present
    with_embeddings = getattr(args, "with_embeddings", False)
    if with_embeddings:
        import duckdb as _ddb
        _con = _ddb.connect(str(db_path), read_only=True)
        try:
            ne = _con.execute("SELECT COUNT(*) FROM entity_embeddings").fetchone()[0]
        except Exception:
            ne = 0
        _con.close()
        if ne > 0 and not force:
            print(f"entity_embeddings already populated ({ne:,} rows) — skipping.")
        else:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(EMBED_MODEL)
            entity_index = _load_entity_index_from_db(db_path)
            _build_and_persist_entity_embeddings(entity_index, embed_model, db_path)


def cmd_index(args: argparse.Namespace) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    embed_content_only = getattr(args, "embed_content_only", False)
    min_chunk = getattr(args, 'min_chunk', MIN_CHUNK) or MIN_CHUNK
    max_chunk = getattr(args, 'max_chunk', MAX_CHUNK) or MAX_CHUNK
    size_suffix = f"_{min_chunk}_{max_chunk}" if (min_chunk != MIN_CHUNK or max_chunk != MAX_CHUNK) else ""
    base = "chonk_nobc" if embed_content_only else "chonk"
    db_path = data_dir / f"{base}{size_suffix}.duckdb"

    if db_path.exists() and not args.force:
        with Store(db_path, embedding_dim=EMBED_DIM) as store:
            n = store.count()
        print(f"Index already exists: {n:,} chunks at {db_path}")
        print("Use --force to reindex.")
        return

    if db_path.exists() and args.force:
        db_path.unlink()
        print(f"Removed existing index: {db_path}")

    corpus = _build_corpus(out_dir)
    print(f"Corpus: {len(corpus)} documents")

    print(f"Chunking with header promotion (min={min_chunk}, max={max_chunk})...")
    all_chunks = []
    for doc_id, text in corpus:
        is_novel = doc_id.lower().startswith("novel")
        if is_novel:
            promoted = promote_plain_text_headers(
                text,
                promote_questions=False,
                promote_short_phrases=False,
                structural_levels=NOVEL_STRUCTURAL_LEVELS,
                toc_proximity=300,
            )
        else:
            promoted = promote_plain_text_headers(
                text,
                promote_questions=True,
                promote_short_phrases=True,
                strip_isolated_letters=True,
            )
        chunks = chunk_document(
            doc_id, promoted,
            min_chunk_size=min_chunk,
            max_chunk_size=max_chunk,
            include_breadcrumb=True,
            include_doc_name=False,
            promote_headings=False,  # already promoted above
        )
        chunks = enrich_chunks(chunks, strategy="prefix")
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks):,}")
    avg = sum(len(c.content) for c in all_chunks) / max(1, len(all_chunks))
    print(f"Avg chunk size: {avg:.0f} chars")

    print(f"Embedding {len(all_chunks):,} chunks with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [
        c.content if embed_content_only else (c.embedding_content if c.embedding_content else c.content)
        for c in all_chunks
    ]

    batch_size = 256
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs  = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(vecs)
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0:
            print(f"  {done:,}/{len(texts):,}")

    emb = np.vstack(embeddings).astype("float32")
    print(f"Embeddings: {emb.shape}")

    print(f"Storing in {db_path}...")
    with Store(db_path, embedding_dim=EMBED_DIM) as store:
        store.add_document(all_chunks, emb)
        n = store.count()

    print(f"Index complete: {n:,} chunks → {db_path}")

    if getattr(args, "with_ner", False):
        with Store(db_path, embedding_dim=EMBED_DIM) as store:
            entity_index = _build_entity_index_from_store(store)
        _persist_entity_index(entity_index, db_path)


def cmd_index_vanilla(args: argparse.Namespace) -> None:
    """Build vanilla RAG index: naive 256-token fixed chunks, no breadcrumbs."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from chonk.models import DocumentChunk

    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    chunk_tokens = getattr(args, 'chunk_tokens', VANILLA_CHUNK_TOKENS) or VANILLA_CHUNK_TOKENS
    db_suffix = f"_{chunk_tokens}" if chunk_tokens != VANILLA_CHUNK_TOKENS else ""
    db_path = data_dir / f"vanilla_rag{db_suffix}.duckdb"

    if db_path.exists() and not args.force:
        with Store(db_path, embedding_dim=EMBED_DIM) as store:
            n = store.count()
        print(f"Vanilla index exists: {n:,} chunks at {db_path}")
        print("Use --force to reindex.")
        return

    if db_path.exists() and args.force:
        db_path.unlink()
        print(f"Removed existing index: {db_path}")

    corpus = _build_corpus(out_dir)
    print(f"Corpus: {len(corpus)} documents")
    print(f"Naive chunking: {chunk_tokens}-token chunks, {VANILLA_CHUNK_OVERLAP}-token overlap...")

    all_chunks: list[DocumentChunk] = []
    for doc_id, text in corpus:
        for i, chunk_text in enumerate(_naive_chunks(text, chunk_tokens=chunk_tokens)):
            all_chunks.append(DocumentChunk(
                document_name=doc_id,
                chunk_index=i,
                content=chunk_text,
                breadcrumb="",
                embedding_content=chunk_text,
            ))

    print(f"Total vanilla chunks: {len(all_chunks):,}")
    avg = sum(len(c.content) for c in all_chunks) / max(1, len(all_chunks))
    print(f"Avg chunk size: {avg:.0f} chars")

    print(f"Embedding {len(all_chunks):,} chunks with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c.content for c in all_chunks]

    batch_size = 256
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs  = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(vecs)
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0:
            print(f"  {done:,}/{len(texts):,}")

    emb = np.vstack(embeddings).astype("float32")
    print(f"Embeddings: {emb.shape}")

    print(f"Storing in {db_path}...")
    with Store(db_path, embedding_dim=EMBED_DIM) as store:
        store.add_document(all_chunks, emb)
        n = store.count()
    print(f"Vanilla index complete: {n:,} chunks → {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Retrieve + generate
# ─────────────────────────────────────────────────────────────────────────────

_REFUSAL_PHRASES = (
    "does not contain", "does not provide", "does not mention",
    "cannot determine", "cannot answer", "not enough information",
    "insufficient information", "not mentioned", "no information",
    "context does not", "not specified", "not stated",
)

def _format_breadcrumb(crumb: str, style: str = "markdown") -> str:
    """Convert [doc > sec > subsec] to various heading formats.

    style='markdown': ## doc\\n### sec\\n#### subsec  (default)
    style='literal':  Document: doc. Section: sec. Subsection: subsec.
    style='symbol':   original [doc > sec > subsec] unchanged
    """
    if style == "symbol":
        return crumb
    inner = crumb.strip("[]")
    parts = [p.strip() for p in inner.split(">") if p.strip()]
    if not parts:
        return crumb
    if style == "literal":
        labels = ["Document", "Section", "Subsection", "Topic"]
        return ". ".join(
            f"{labels[min(i, len(labels)-1)]}: {p}" for i, p in enumerate(parts)
        ) + "."
    # markdown (default)
    levels = ["##", "###", "####", "#####"]
    return "\n".join(f"{levels[min(i, len(levels)-1)]} {p}" for i, p in enumerate(parts))


def _is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in _REFUSAL_PHRASES)


_STRUCTURED_GEN_SYSTEM = (
    "Answer the question based only on the provided context. "
    "If the context does not contain enough information, say so. "
    "You MUST format your response exactly as:\nANSWER: <your answer here>\n"
    "Do not include any text before or after this line."
)
_STRUCTURED_GEN_RETRY_HINT = (
    "Your response did not follow the required format. "
    "You MUST respond with exactly one line starting with 'ANSWER: ' followed by your answer. "
    "Example: ANSWER: The capital of France is Paris."
)
_UNSTRUCTURED_GEN_SYSTEM = (
    "Answer the question based only on the provided context. "
    "If the context does not contain enough information, "
    "say so rather than making up an answer."
)

def _extract_structured_answer(text: str) -> str | None:
    """Extract content after 'ANSWER:' marker. Returns None if marker absent."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("ANSWER:"):
            return stripped[len("ANSWER:"):].strip()
    return None

def _generate(question: str, context: str, client, model: str = GEN_MODEL,
              temperature: float = 0.0, retry_hint: str | None = None,
              structured: bool = False) -> str:
    system = _STRUCTURED_GEN_SYSTEM if structured else _UNSTRUCTURED_GEN_SYSTEM
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    if retry_hint:
        user_content += f"\n\n{retry_hint}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content.strip()
    if not structured:
        return raw
    answer = _extract_structured_answer(raw)
    if answer is not None:
        return answer
    # Format non-compliant — retry once with explicit format hint
    retry_content = user_content + f"\n\n{_STRUCTURED_GEN_RETRY_HINT}"
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": retry_content},
        ],
        temperature=temperature,
        max_tokens=500,
    )
    raw2 = resp2.choices[0].message.content.strip()
    answer2 = _extract_structured_answer(raw2)
    return answer2 if answer2 is not None else raw2


def _build_entity_index_from_store(store) -> EntityIndex:
    """Run NER on all chunks in store and return a populated EntityIndex."""
    from chonk.ner import EntityIndex, SpacyLabel, SpacyMatcher
    from chonk.storage._vector import DuckDBVectorBackend
    _NUMERIC_TYPES = {SpacyLabel.CARDINAL, SpacyLabel.ORDINAL, SpacyLabel.MONEY,
                      SpacyLabel.PERCENT, SpacyLabel.QUANTITY}
    _label_types = [t for t in SpacyLabel if t not in _NUMERIC_TYPES]
    print(f"Building EntityIndex with SpacyMatcher({SPACY_MODEL})...")
    matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True, entity_types=_label_types)
    entity_index = EntityIndex()

    all_chunks = store.vector.get_all_chunks()
    print(f"  Running NER on {len(all_chunks):,} chunks...")
    for chunk in all_chunks:
        embed_content = chunk.embedding_content if chunk.embedding_content else chunk.content
        chunk_id = DuckDBVectorBackend._generate_chunk_id(
            chunk.document_name, chunk.chunk_index, embed_content
        )
        entity_index.run_ner(chunk_id, chunk.content, matcher)

    entity_index.recompute_scores()
    print(f"  {entity_index.total_chunks():,} chunks, {len(entity_index.entity_ids()):,} entities")
    return entity_index


def _persist_entity_index(entity_index, db_path: Path) -> None:
    """Write entity_index associations to chunk_entities/entities tables."""

    print("  Persisting to chunk_entities table...")
    data = entity_index.to_dict()
    con = _connect_with_retry(db_path)
    con.execute("DELETE FROM chunk_entities")
    con.execute("DELETE FROM entities")
    for a in data["associations"]:
        con.execute(
            "INSERT OR REPLACE INTO chunk_entities(chunk_id, entity_id, frequency, positions_json, score) VALUES (?,?,?,?,?)",
            [a["chunk_id"], a["entity_id"], a["frequency"], json.dumps(a["positions"]), a["score"]],
        )
        con.execute(
            "INSERT OR IGNORE INTO entities(id, name, display_name) VALUES (?,?,?)",
            [a["entity_id"], a["entity_id"], a["entity_id"]],
        )
    con.close()
    print(f"  Persisted {len(data['associations']):,} associations → {db_path}")


def _build_and_persist_entity_embeddings(entity_index, embed_model, db_path: Path) -> None:
    """Embed all unique entity name strings and store in entity_embeddings table."""

    entity_ids = list(entity_index.entity_ids())
    if not entity_ids:
        return
    print(f"  Embedding {len(entity_ids):,} unique entity strings...")
    vecs = embed_model.encode(
        entity_ids, normalize_embeddings=True, show_progress_bar=False, batch_size=512
    ).astype("float32")
    con = _connect_with_retry(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS entity_embeddings "
        "(entity_id TEXT PRIMARY KEY, embedding FLOAT[])"
    )
    con.execute("DELETE FROM entity_embeddings")
    for eid, vec in zip(entity_ids, vecs):
        con.execute(
            "INSERT INTO entity_embeddings(entity_id, embedding) VALUES (?, ?)",
            [eid, vec.tolist()],
        )
    con.close()
    print(f"  Persisted {len(entity_ids):,} entity embeddings → {db_path}")


def cmd_build_community(args: argparse.Namespace) -> None:
    """Build community index: heading vectors + weighted average + Louvain detection."""
    import duckdb
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from chonk.community import CommunityIndex

    data_dir = Path(args.out_dir) / "data"
    db_path  = data_dir / args.db_name
    alpha    = getattr(args, "alpha", 0.2)
    sim_threshold = getattr(args, "sim_threshold", 0.6)
    force    = getattr(args, "force", False)
    label_strategy = getattr(args, "community_label_strategy", "ner_embedding")

    if not db_path.exists():
        raise FileNotFoundError(f"Index DB not found: {db_path}")

    # Check if already built
    if not force:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()[0]
            con.close()
            if n > 0:
                print(f"chunk_communities already populated ({n:,} rows) — skipping. Use --force to rebuild.")
                return
        except Exception:
            con.close()

    print(f"Loading chunks and embeddings from {db_path}...")
    con_ro = duckdb.connect(str(db_path), read_only=True)
    rows = con_ro.execute(
        "SELECT chunk_id, content, breadcrumb, embedding FROM embeddings WHERE embedding IS NOT NULL"
    ).fetchall()
    con_ro.close()

    chunk_ids: list[str] = [r[0] for r in rows]
    chunk_texts: list[str] = [r[1] or "" for r in rows]
    breadcrumbs: list[str] = [r[2] or "" for r in rows]
    content_vecs = np.array([r[3] for r in rows], dtype="float32")
    print(f"  {len(chunk_ids):,} chunks with embeddings")

    # Embed breadcrumbs (heading vectors)
    heading_vecs = None
    non_empty_bc = [bc for bc in breadcrumbs if bc.strip()]
    if non_empty_bc:
        print(f"  Embedding {len(non_empty_bc):,} non-empty breadcrumbs (α={alpha})...")
        model = SentenceTransformer(EMBED_MODEL)
        bc_texts = [bc if bc.strip() else "" for bc in breadcrumbs]
        all_bc_vecs = model.encode(
            bc_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=256
        ).astype("float32")
        # Zero out empty breadcrumbs so they don't contribute
        for i, bc in enumerate(breadcrumbs):
            if not bc.strip():
                all_bc_vecs[i] = 0.0
        heading_vecs = all_bc_vecs
        del model
    else:
        print("  No breadcrumbs found — using content vectors only.")

    print(f"  Building community index (sim_threshold={sim_threshold}, label_strategy={label_strategy})...")
    idx = CommunityIndex.build(
        chunk_ids=chunk_ids,
        content_vecs=content_vecs,
        chunk_texts=chunk_texts,
        heading_vecs=heading_vecs,
        alpha=alpha,
        sim_threshold=sim_threshold,
        label_strategy=label_strategy,
        db_path=db_path if label_strategy == "ner_embedding" else None,
    )
    print(f"  {idx.community_count():,} communities across {idx.chunk_count():,} chunks")

    # Persist — retry with backoff if DB is locked by another process
    import time as _time
    for _attempt in range(60):
        try:
            idx.persist(db_path)
            print(f"  Persisted → {db_path}")
            break
        except Exception as _e:
            if "lock" in str(_e).lower() or "conflict" in str(_e).lower():
                print(f"  DB locked, waiting 10s... (attempt {_attempt+1}/60)", flush=True)
                _time.sleep(10)
            else:
                raise
    else:
        raise RuntimeError("Could not acquire write lock on DB after 10 minutes.")


# ─────────────────────────────────────────────────────────────────────────────
# Per-run DuckDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connect_with_retry(db_path, max_attempts: int = 30, delay: float = 2.0):
    """Open a DuckDB write connection, retrying on lock conflicts."""
    import time as _time

    import duckdb
    for attempt in range(max_attempts):
        try:
            return duckdb.connect(str(db_path))
        except Exception as e:
            if ("lock" in str(e).lower() or "conflict" in str(e).lower()) and attempt < max_attempts - 1:
                _time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Could not acquire DB write lock after {max_attempts * delay:.0f}s")


def _run_db_path(data_dir: Path, run_name: str) -> Path:
    runs_dir = data_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    return runs_dir / f"{run_name}.duckdb"


def _init_run_db(db_path: Path) -> None:
    con = _connect_with_retry(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            question TEXT,
            source TEXT,
            question_type TEXT,
            context TEXT,
            evidence TEXT,
            generated_answer TEXT,
            gold_answer TEXT,
            retrieved_chunks TEXT,
            retrieved_scores TEXT,
            expansion_stats TEXT,
            entity_ref_retry TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval_scores (
            id TEXT PRIMARY KEY,
            question_type TEXT,
            answer_correctness REAL,
            rouge_score REAL,
            coverage_score REAL,
            faithfulness REAL
        )
    """)
    con.close()


def _upsert_results_to_db(db_path: Path, results: list[dict]) -> None:
    """Insert or replace a batch of results without clearing the table."""
    import json as _json
    con = _connect_with_retry(db_path)
    for r in results:
        con.execute(
            "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                r.get("id"), r.get("question"), r.get("source"),
                r.get("question_type"), r.get("context"),
                _json.dumps(r.get("evidence")),
                r.get("generated_answer"), r.get("gold_answer"),
                _json.dumps(r.get("retrieved_chunks")),
                _json.dumps(r.get("retrieved_scores")),
                _json.dumps(r.get("entity_ref_expansion")) if r.get("entity_ref_expansion") else None,
                _json.dumps(r.get("entity_ref_retry")) if r.get("entity_ref_retry") else None,
            ],
        )
    con.close()


def _write_results_to_db(db_path: Path, results: list[dict]) -> None:
    import json as _json
    # Deduplicate by id (last write wins — handles checkpoint resume duplicates)
    seen: dict[str, dict] = {}
    for r in results:
        seen[r.get("id")] = r
    results = list(seen.values())
    con = _connect_with_retry(db_path)
    con.execute("DELETE FROM results")
    for r in results:
        con.execute(
            "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                r.get("id"), r.get("question"), r.get("source"),
                r.get("question_type"), r.get("context"),
                _json.dumps(r.get("evidence")),
                r.get("generated_answer"), r.get("gold_answer"),
                _json.dumps(r.get("retrieved_chunks")),
                _json.dumps(r.get("retrieved_scores")),
                _json.dumps(r.get("entity_ref_expansion")) if r.get("entity_ref_expansion") else None,
                _json.dumps(r.get("entity_ref_retry")) if r.get("entity_ref_retry") else None,
            ],
        )
    con.close()


def _read_results_from_db(db_path: Path) -> list[dict]:
    import json as _json

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute("SELECT * FROM results").fetchall()
    cols = ["id","question","source","question_type","context","evidence",
            "generated_answer","gold_answer","retrieved_chunks","retrieved_scores",
            "expansion_stats","entity_ref_retry"]
    con.close()
    records = []
    for row in rows:
        r = dict(zip(cols, row))
        for key in ("evidence","retrieved_chunks","retrieved_scores","expansion_stats","entity_ref_retry"):
            if r[key]:
                try: r[key] = _json.loads(r[key])
                except Exception: pass
        records.append(r)
    return records


def _load_community_index(db_path: Path):
    """Load CommunityIndex from DB, return None if not built."""
    from chonk.community import CommunityIndex
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM chunk_communities").fetchone()[0]
        con.close()
        if n == 0:
            return None
        print(f"Loading CommunityIndex from DB ({n:,} assignments)...")
        return CommunityIndex.from_db(db_path)
    except Exception:
        return None


def _load_entity_embeddings(db_path: Path):
    """Load entity embeddings matrix and ID list from DB. Returns (matrix, ids) or (None, None)."""
    import duckdb
    import numpy as np

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("SELECT entity_id, embedding FROM entity_embeddings").fetchall()
    except Exception:
        con.close()
        return None, None
    con.close()
    if not rows:
        return None, None
    ids = [r[0] for r in rows]
    mat = np.array([r[1] for r in rows], dtype="float32")
    return mat, ids


def _load_entity_index_from_db(db_path: Path) -> EntityIndex:
    """Reconstruct EntityIndex from persisted chunk_entities table."""
    import duckdb

    from chonk.ner import EntityIndex

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT chunk_id, entity_id, frequency, positions_json, score FROM chunk_entities"
    ).fetchall()
    total_chunks = con.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunk_entities").fetchone()[0]
    con.close()

    associations = [
        {
            "entity_id": r[1],
            "chunk_id": r[0],
            "frequency": r[2],
            "positions": json.loads(r[3]) if r[3] else [],
            "score": r[4],
            "chunk_length": 1,
        }
        for r in rows
    ]
    return EntityIndex.from_dict({
        "total_chunks": total_chunks,
        "score_weights": [0.4, 0.3, 0.3],
        "associations": associations,
    })


def _prune_redundant(hits, db_conn, threshold):
    """Remove near-duplicate chunks from reranked hits (greedy, post-rerank).

    Fetches embeddings in one batch query. Keeps the first occurrence when
    two chunks have cosine similarity >= threshold.
    """
    import math
    if threshold is None or len(hits) <= 1:
        return hits
    chunk_ids = [cid for cid, _, _ in hits]
    placeholders = ", ".join(["?" for _ in chunk_ids])
    rows = db_conn.execute(
        f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    embs = {row[0]: list(row[1]) for row in rows if row[1] is not None}

    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    selected, selected_embs = [], []
    for hit in hits:
        cid = hit[0]
        emb = embs.get(cid, [])
        if emb and selected_embs:
            if max(_cosine(emb, se) for se in selected_embs if se) >= threshold:
                continue
        selected.append(hit)
        selected_embs.append(emb)
    return selected


def _build_enhanced_search(store, db_path: Path | None = None, use_ner_x: bool = False, embed_model=None, entity_ref_expansion: bool = False, entity_ref_expansion_k: int = 20, entity_ref_expansion_per_k: int | None = None, entity_ref_expansion_min_sim: float | None = None, use_cluster: bool = False, lane_entity_min_sim: float | None = None):
    """Load EnhancedSearch: from pre-built DB tables if available, else rebuild in memory."""
    import duckdb

    from chonk.ner import EntityIndex, SpacyMatcher
    from chonk.search import EnhancedSearch
    from chonk.storage._vector import DuckDBVectorBackend

    if db_path is not None and db_path.exists():
        con = duckdb.connect(str(db_path), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM chunk_entities").fetchone()[0]
        con.close()
        if n > 0:
            print(f"Loading EntityIndex from DB ({n:,} associations)...")
            entity_index = _load_entity_index_from_db(db_path)
            print(f"  {entity_index.total_chunks():,} chunks, {len(entity_index.entity_ids()):,} entities")

            cluster_map = None
            if use_cluster:
                from chonk.cluster import ClusterMap
                print("  Building ClusterMap...")
                cluster_map = ClusterMap.build(entity_index)
                print(f"  {cluster_map.cluster_count():,} clusters across {cluster_map.entity_count():,} entities")

            matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True)
            query_ner_fn = lambda text: [m.display_name for m in matcher.match(text)]

            ner_x_kwargs: dict = {}
            if use_ner_x and embed_model is not None:
                mat, ids = _load_entity_embeddings(db_path)
                if mat is not None:
                    print(f"  Loaded {len(ids):,} entity embeddings for ner-x expansion.")
                    ner_x_kwargs = dict(
                        entity_embedding_expansion=True,
                        entity_embeddings=mat,
                        entity_embedding_ids=ids,
                        ner_fn=lambda text: [m.entity_id for m in matcher.match(text)],
                        entity_embedding_top_k=10,
                    )
                else:
                    print("  entity_embeddings table empty — ner-x disabled. Run build-ner --with-embeddings first.")

            embed_fn_kwargs: dict = {}
            if embed_model is not None:
                embed_fn_kwargs["embed_fn"] = lambda texts: embed_model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )

            return EnhancedSearch(
                store,
                entity_index=entity_index,
                cluster_map=cluster_map,
                structural_expansion=False,
                cluster_expansion=use_cluster,
                query_ner_fn=query_ner_fn,
                entity_ref_expansion=entity_ref_expansion,
                entity_ref_expansion_k=entity_ref_expansion_k,
                entity_ref_expansion_per_k=entity_ref_expansion_per_k,
                entity_ref_expansion_min_sim=entity_ref_expansion_min_sim,
                lane_entity_min_sim=lane_entity_min_sim,
                **ner_x_kwargs,
                **embed_fn_kwargs,
            )

    print(f"Building EntityIndex with SpacyMatcher({SPACY_MODEL}) (not pre-built, rebuild in memory)...")
    matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True)
    entity_index = EntityIndex()
    all_chunks = store.vector.get_all_chunks()
    print(f"  Running NER on {len(all_chunks):,} chunks...")
    for chunk in all_chunks:
        embed_content = chunk.embedding_content if chunk.embedding_content else chunk.content
        chunk_id = DuckDBVectorBackend._generate_chunk_id(
            chunk.document_name, chunk.chunk_index, embed_content
        )
        entity_index.run_ner(chunk_id, chunk.content, matcher)
    entity_index.recompute_scores()
    print(f"  {entity_index.total_chunks():,} chunks, {len(entity_index.entity_ids()):,} entities")
    cluster_map = None
    if use_cluster:
        from chonk.cluster import ClusterMap
        print("  Building ClusterMap...")
        cluster_map = ClusterMap.build(entity_index)
        print(f"  {cluster_map.cluster_count():,} clusters across {cluster_map.entity_count():,} entities")
    query_ner_fn = lambda text: [m.display_name for m in matcher.match(text)]
    return EnhancedSearch(
        store,
        entity_index=entity_index,
        cluster_map=cluster_map,
        structural_expansion=False,
        cluster_expansion=use_cluster,
        query_ner_fn=query_ner_fn,
        entity_ref_expansion=entity_ref_expansion,
        entity_ref_expansion_k=entity_ref_expansion_k,
        entity_ref_expansion_per_k=entity_ref_expansion_per_k,
        entity_ref_expansion_min_sim=entity_ref_expansion_min_sim,
        lane_entity_min_sim=lane_entity_min_sim,
    )


def cmd_run(args: argparse.Namespace) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np
    import openai
    from sentence_transformers import SentenceTransformer

    out_dir     = Path(args.out_dir)
    data_dir    = out_dir / "data"
    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)
    run_name     = getattr(args, "run_name", "contextual")

    # Kill any stale processes already running the same run-name
    import signal
    import subprocess as _sp
    _my_pid = os.getpid()
    try:
        _procs = _sp.check_output(
            ["pgrep", "-f", f"graphrag_bench.py.*--run-name {run_name}( |$)"],
            text=True,
        ).split()
        for _pid in _procs:
            _pid = int(_pid)
            if _pid != _my_pid:
                try:
                    os.kill(_pid, signal.SIGKILL)
                    print(f"[preflight] killed stale pid {_pid} ({run_name})", flush=True)
                except ProcessLookupError:
                    pass
    except _sp.CalledProcessError:
        pass  # no matching processes
    results_f    = results_dir / f"{run_name}.jsonl"
    ckpt_f       = results_dir / f"{run_name}_checkpoint.jsonl"
    use_vanilla  = getattr(args, "vanilla", False)
    use_rerank        = getattr(args, "rerank", False)
    rerank_provider   = getattr(args, "rerank_provider", "local")
    use_enhanced      = getattr(args, "enhanced", False)
    use_ner_x         = getattr(args, "ner_x", False)
    use_entity_ref_expansion = getattr(args, "entity_ref_expansion", False)
    entity_ref_expansion_per_k  = getattr(args, "entity_ref_expansion_per_k", None)
    entity_ref_expansion_min_sim = getattr(args, "entity_ref_expansion_min_sim", None)
    use_cluster = getattr(args, "cluster", False)
    use_entity_ref_retry     = getattr(args, "entity_ref_retry", False)
    use_structured_gen       = getattr(args, "structured_gen", False)
    use_breadcrumb_context = getattr(args, "breadcrumb_context", False)
    breadcrumb_style       = getattr(args, "breadcrumb_style", "markdown")
    use_community_context  = getattr(args, "community_context", False)
    community_min_coherence = getattr(args, "community_min_coherence", 0.0)
    breadcrumb_embed       = getattr(args, "breadcrumb_embed", False)
    no_breadcrumb_embed    = not breadcrumb_embed  # legacy alias for internal logic
    redundancy_threshold   = getattr(args, "redundancy_threshold", None)
    lane_entity_min_sim    = getattr(args, "lane_entity_min_sim", None)
    concentration_threshold = getattr(args, "concentration_threshold", None)
    query_complexity_threshold = getattr(args, "query_complexity_threshold", 2)
    db_name_override = getattr(args, 'db_name', None)
    question_ids_file = getattr(args, 'question_ids', None)
    if db_name_override:
        db_path = data_dir / db_name_override
    else:
        _ctx_db = DB_FILENAME.replace(".duckdb", "_nobc.duckdb") if no_breadcrumb_embed else DB_FILENAME
        db_path = data_dir / (VANILLA_DB_FILENAME if use_vanilla else _ctx_db)
    top_k        = VANILLA_K if use_vanilla else (getattr(args, "top_k", None) or K)
    gen_temperature = VANILLA_TEMPERATURE  # paper: 0.7 for all systems

    if not db_path.exists():
        print("No index found. Run 'index' first.")
        return

    # Write sidecar flags file for reproducibility
    _flags = {
        "run_name": run_name,
        "db": str(db_path.name),
        "vanilla": use_vanilla,
        "rerank": use_rerank,
        "rerank_provider": rerank_provider,
        "enhanced": use_enhanced,
        "entity_ref_expansion": use_entity_ref_expansion,
        "entity_ref_expansion_min_sim": entity_ref_expansion_min_sim,
        "entity_ref_expansion_per_k": entity_ref_expansion_per_k,
        "lane_entity_min_sim": lane_entity_min_sim,
        "community_context": use_community_context,
        "community_min_coherence": community_min_coherence,
        "redundancy_threshold": redundancy_threshold,
        "breadcrumb_embed": breadcrumb_embed,
        "breadcrumb_context": use_breadcrumb_context,
        "breadcrumb_style": breadcrumb_style,
        "cluster": use_cluster,
        "ner_x": use_ner_x,
        "concentration_threshold": concentration_threshold,
        "query_complexity_threshold": query_complexity_threshold,
        "top_k": top_k,
    }
    _flags_path = results_dir / f"{run_name}_flags.json"
    _flags_path.write_text(json.dumps(_flags, indent=2), encoding="utf-8")

    import atexit as _atexit
    import signal as _signal
    _run_completed = [False]
    def _cleanup_flags():
        if not _run_completed[0] and _flags_path.exists():
            _flags_path.unlink(missing_ok=True)
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    _atexit.register(_cleanup_flags)
    def _sigterm_handler(signum, frame):
        raise SystemExit(0)
    _signal.signal(_signal.SIGTERM, _sigterm_handler)

    questions = _load_questions(data_dir)
    if question_ids_file:
        with open(question_ids_file) as _f:
            _order = json.load(_f)
        _id_to_q = {q.get("id"): q for q in questions}
        questions = [_id_to_q[qid] for qid in _order if qid in _id_to_q]
    if args.limit:
        questions = questions[:args.limit]
    print(f"Questions: {len(questions)}")

    done_ids: set[str] = set()
    if ckpt_f.exists():
        with open(ckpt_f) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["id"])
        print(f"Resuming from checkpoint: {len(done_ids)} already done")

    run_db = _run_db_path(data_dir, run_name)
    _init_run_db(run_db)

    # Resume from DuckDB only for fully-generated results (generated_answer IS NOT NULL).
    # Rows with context but NULL generated_answer are retrieval-phase stubs — they must
    # be regenerated, not skipped.
    if run_db.exists():
        import duckdb as _duckdb
        try:
            _con = _duckdb.connect(str(run_db), read_only=True)
            _db_done = set(
                row[0] for row in _con.execute(
                    "SELECT id FROM results WHERE generated_answer IS NOT NULL"
                ).fetchall()
            )
            _con.close()
            if _db_done - done_ids:
                print(f"Resuming from DB: {len(_db_done)} already generated")
                done_ids.update(_db_done)
        except Exception:
            pass

    pending = [(i, q) for i, q in enumerate(questions)
               if q.get("id", f"q{i}") not in done_ids]
    print(f"Pending: {len(pending)}")

    # ── 1. Embed all questions (full-corpus cache — order-independent across runs)
    # Cache key is the full unfiltered corpus so any --question-ids subset or
    # reordering reuses the same cache file via ID lookup.
    q_vecs_cache = data_dir / "question_embeddings.npy"
    q_ids_cache  = data_dir / "question_ids.json"

    _corpus_qs  = _load_questions(data_dir)
    _corpus_ids = [q.get("id", f"q{i}") for i, q in enumerate(_corpus_qs)]

    _corpus_vecs = None
    if q_vecs_cache.exists() and q_ids_cache.exists():
        if json.loads(q_ids_cache.read_text()) == _corpus_ids:
            print(f"Loading cached question embeddings from {q_vecs_cache}")
            _corpus_vecs = np.load(str(q_vecs_cache))
        else:
            q_vecs_cache.unlink()  # stale — corpus changed

    import hashlib as _hl_pre
    _ent_cache_key_pre = _hl_pre.md5(
        (EMBED_MODEL + SPACY_MODEL + json.dumps(_corpus_ids)).encode()
    ).hexdigest()[:16]
    _ent_vecs_cache_path = data_dir / f"entity_vecs_{_ent_cache_key_pre}.npz"
    _ent_ents_cache_path = data_dir / f"entity_ents_{_ent_cache_key_pre}.json"
    _ent_cache_exists    = _ent_vecs_cache_path.exists() and _ent_ents_cache_path.exists()

    _needs_entity_embed_at_run = use_ner_x or use_entity_ref_expansion or concentration_threshold is not None
    if _needs_entity_embed_at_run and not _ent_cache_exists:
        raise RuntimeError(
            f"Entity embedding cache missing for model '{EMBED_MODEL}' / spaCy '{SPACY_MODEL}'.\n"
            f"Run: python demo/graphrag_bench.py prime-cache --out-dir {args.out_dir}"
        )
    if _corpus_vecs is None:
        raise RuntimeError(
            f"Question embedding cache missing.\n"
            f"Run: python demo/graphrag_bench.py prime-cache --out-dir {args.out_dir}"
        )

    embed_model = None
    if use_entity_ref_retry:
        _embed_device = os.environ.get("EMBED_DEVICE") or None
        if not _embed_device:
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
                    _embed_device = "cpu"
            except ImportError:
                pass
        embed_model = SentenceTransformer(EMBED_MODEL, device=_embed_device) if _embed_device else SentenceTransformer(EMBED_MODEL)

    # Build lookup and align to the (possibly filtered/reordered) questions list
    _id_to_vec = {qid: _corpus_vecs[i] for i, qid in enumerate(_corpus_ids)}
    all_vecs = np.stack([_id_to_vec[q.get("id", f"q{i}")] for i, q in enumerate(questions)])

    # slice to pending indices only
    pending_indices = [i for i, _ in pending]
    q_vecs = all_vecs[pending_indices]

    # ── 2. Retrieve context for each question (sequential; DuckDB conn is serialized)
    fetch_k = K_FETCH if use_rerank else top_k
    print(f"Retrieving context from index (fetch_k={fetch_k}, k={top_k}, rerank={use_rerank}, enhanced={use_enhanced}, vanilla={use_vanilla})...")

    reranker = None
    together_rerank_client = None
    cohere_rerank_client = None
    if use_rerank:
        if rerank_provider == "together":
            from together import Together
            together_rerank_client = Together(api_key=os.environ["TOGETHER_API_KEY"])
            print(f"Using Together reranker: {RERANK_MODEL_TOGETHER}")
        elif rerank_provider == "cohere":
            import cohere
            cohere_rerank_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
            print(f"Using Cohere reranker: {RERANK_MODEL_COHERE}")
        else:
            import os as _os

            from sentence_transformers import CrossEncoder
            _rerank_device = _os.environ.get("RERANKER_DEVICE") or None
            if not _rerank_device:
                try:
                    import torch
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
                        _rerank_device = "cpu"
                except ImportError:
                    pass
            print(f"Loading reranker: {RERANK_MODEL}{'  [device='+_rerank_device+']' if _rerank_device else ''}...")
            reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=_rerank_device) if _rerank_device else CrossEncoder(RERANK_MODEL, max_length=512)

    community_index = _load_community_index(db_path) if use_community_context else None
    if use_community_context and community_index is None:
        print("WARNING: --community-context set but no community index found. Run 'build-community' first.")

    # ── Complexity / concentration helpers ────────────────────────────────────
    _complexity_matcher = None
    _CONJUNCTION_WORDS = frozenset(["and", "or", "vs", "versus", "between"])

    def _query_complexity(question: str) -> int:
        """Return complexity score = entity_count + clause_signal_count."""
        entity_count = 0
        if _complexity_matcher is not None:
            entity_count = len(_complexity_matcher.match(question))
        tokens = re.sub(r"[^\w\s?]", " ", question.lower()).split()
        clause_signals = (
            question.count("?")
            + question.count(",")
            + sum(1 for t in tokens if t in _CONJUNCTION_WORDS)
        )
        return entity_count + clause_signals

    needs_ner_for_complexity = use_community_context and query_complexity_threshold > 0
    needs_ner_for_concentration = concentration_threshold is not None and not use_enhanced
    if needs_ner_for_complexity or needs_ner_for_concentration:
        from chonk.ner import SpacyMatcher
        _complexity_matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True)
        print("Loaded SpacyMatcher for query complexity / concentration gating.")

    work_items: list[dict] = []
    with Store(db_path, embedding_dim=EMBED_DIM, read_only=True) as store:
        enhanced_search = _build_enhanced_search(
            store, db_path, use_ner_x=use_ner_x, embed_model=embed_model,
            entity_ref_expansion=use_entity_ref_expansion,
            entity_ref_expansion_per_k=entity_ref_expansion_per_k,
            entity_ref_expansion_min_sim=entity_ref_expansion_min_sim,
            use_cluster=use_cluster,
            lane_entity_min_sim=lane_entity_min_sim,
        ) if use_enhanced else None

        # Build concentration fallback search (entity ref expansion) if gating is enabled
        # and entity_ref_expansion is not already always-on.
        _conc_search = None
        if concentration_threshold is not None and not use_entity_ref_expansion:
            _conc_search = _build_enhanced_search(
                store, db_path, use_ner_x=False, embed_model=embed_model,
                entity_ref_expansion=True,
                entity_ref_expansion_per_k=entity_ref_expansion_per_k,
                entity_ref_expansion_min_sim=entity_ref_expansion_min_sim,
                use_cluster=False,
                lane_entity_min_sim=lane_entity_min_sim,
            )

        # Preload embeddings + chunk metadata into RAM (eliminates per-query DuckDB round-trips)
        store.vector.preload_embeddings()
        print(f"  Embeddings preloaded ({store.vector.count()} chunks).", flush=True)
        if enhanced_search is not None:
            enhanced_search.preload_chunk_cache()
            print("  Chunk cache preloaded.", flush=True)
        if _conc_search is not None:
            _conc_search.preload_chunk_cache()

        # Load pre-computed NER entity embeddings from cache (built by prime-cache)
        _precomputed_entity_vecs: dict[str, np.ndarray] | None = None
        _precomputed_question_entities: list[list[str]] | None = None
        _needs_entity_embed = (
            use_enhanced
            and enhanced_search is not None
            and (use_ner_x or use_entity_ref_expansion)
        )
        if _needs_entity_embed:
            print(f"  Loading cached entity embeddings from {_ent_vecs_cache_path.name}", flush=True)
            _npz = np.load(str(_ent_vecs_cache_path))
            _ent_strings = list(_npz["strings"])
            _ent_matrix  = _npz["vecs"]
            _precomputed_entity_vecs = {s: _ent_matrix[i] for i, s in enumerate(_ent_strings)}
            _q_entities_by_id: dict[str, list[str]] = json.loads(_ent_ents_cache_path.read_text())
            _precomputed_question_entities = [
                _q_entities_by_id.get(q.get("id", f"q{i}"), []) for i, q in pending
            ]

        # Phase 1: vector search for all questions (sequential, DuckDB serialized)
        _all_hits: list[tuple] = []
        _all_expansion_stats: list = []
        for j, (i, q) in enumerate(pending):
            _q_entities = _precomputed_question_entities[j] if _precomputed_question_entities is not None else None
            if use_enhanced and enhanced_search is not None:
                scored = enhanced_search.search(
                    q_vecs[j], k=fetch_k, query_text=q["question"],
                    query_entities=_q_entities,
                    precomputed_entity_vecs=_precomputed_entity_vecs,
                )
                hits   = [(sc.chunk_id, sc.score, sc.chunk) for sc in scored]
                expansion_stats = enhanced_search.last_expansion_stats
            else:
                hits = store.vector.search(
                    q_vecs[j], limit=fetch_k,
                    query_text=q["question"],
                    include_breadcrumbs=False,
                )
                expansion_stats = None

            if _conc_search is not None:
                src_counts: dict[str, int] = defaultdict(int)
                for _, _, _chunk in hits:
                    src_counts[_chunk.document_name] += 1
                _k_hits = len(hits)
                if _k_hits > 0:
                    _max_frac = max(src_counts.values()) / _k_hits
                    if _max_frac >= concentration_threshold:
                        scored_conc = _conc_search.search(
                            q_vecs[j], k=fetch_k, query_text=q["question"],
                            query_entities=_q_entities,
                            precomputed_entity_vecs=_precomputed_entity_vecs,
                        )
                        hits = [(sc.chunk_id, sc.score, sc.chunk) for sc in scored_conc]
                        expansion_stats = _conc_search.last_expansion_stats

            _all_hits.append(hits)
            _all_expansion_stats.append(expansion_stats)
            if (j + 1) % 100 == 0 or (j + 1) == len(pending):
                print(f"  Vector search {j+1}/{len(pending)}", flush=True)

        # Phase 2: batch reranking across all questions (GPU fully utilized)
        if use_rerank and reranker is not None:
            print(f"  Batch reranking {len(_all_hits)} questions...", flush=True)
            # Move embedder off GPU/MPS so reranker has full VRAM for large batches
            try:
                import gc

                import torch
                if embed_model is not None:
                    del embed_model
                    embed_model = None
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # FFN intermediate dominates peak activation: B × seq_len × 4096 × 4 bytes
                    # Use 75% of free VRAM, accounting for other in-flight tensors
                    _free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                    # Empirical: ~20 MiB/sample peak activation (model weights + FFN + attention)
                    # at seq_len=512 on XLM-RoBERTa-large. Scale by actual seq_len.
                    _seq_len = getattr(reranker, "max_length", 512)
                    _mib_per_sample = 20.0 * _seq_len / 512
                    _rerank_batch_size = max(16, min(512, int(_free_mib * 0.55 / _mib_per_sample) // 16 * 16))
                    print(f"  CUDA: {_free_mib} MiB free, computed batch_size={_rerank_batch_size}.", flush=True)
                else:
                    _rerank_batch_size = 64
                    print("  CPU reranking, batch_size=64.", flush=True)
            except Exception:
                _rerank_batch_size = 64
            _RERANK_CHUNK = args.rerank_chunk
            _rerank_ckpt_path = results_dir / f"{run_name}_rerank_ckpt.json"
            # Load existing checkpoint: qid -> [chunk_id, ...]
            _rerank_ckpt: dict[str, list[str]] = {}
            if _rerank_ckpt_path.exists():
                try:
                    _rerank_ckpt = json.loads(_rerank_ckpt_path.read_text())
                    print(f"  Rerank checkpoint: {len(_rerank_ckpt)} questions already done.", flush=True)
                except Exception:
                    _rerank_ckpt = {}
            _reranked_hits: list = [None] * len(_all_hits)
            # Build per-question chunk_id->hit lookup for checkpoint restoration
            _hit_by_cid: list[dict] = [
                {cid: (cid, sc, chunk) for cid, sc, chunk in _all_hits[j]}
                for j in range(len(_all_hits))
            ]
            # Restore already-checkpointed questions
            _pending_rerank_indices = []
            for j, (i, q) in enumerate(pending):
                qid = q.get("id", f"q{i}")
                if qid in _rerank_ckpt:
                    ordered = [_hit_by_cid[j][cid] for cid in _rerank_ckpt[qid] if cid in _hit_by_cid[j]]
                    _reranked_hits[j] = ordered
                else:
                    _pending_rerank_indices.append(j)
            if len(_pending_rerank_indices) < len(_all_hits):
                print(f"  Restored {len(_all_hits) - len(_pending_rerank_indices)} from rerank checkpoint.", flush=True)
            for _ci, _chunk_start in enumerate(range(0, len(_pending_rerank_indices), _RERANK_CHUNK)):
                _chunk_idx = _pending_rerank_indices[_chunk_start:_chunk_start + _RERANK_CHUNK]
                _chunk_end_display = _chunk_start + len(_chunk_idx)
                _pair_offsets = [0]
                _chunk_pairs = []
                for j in _chunk_idx:
                    q = pending[j][1]
                    pairs = [(q["question"], chunk.content) for _, _, chunk in _all_hits[j]]
                    _chunk_pairs.extend(pairs)
                    _pair_offsets.append(len(_chunk_pairs))
                _chunk_scores = reranker.predict(_chunk_pairs, batch_size=_rerank_batch_size, show_progress_bar=False)
                for _ji, j in enumerate(_chunk_idx):
                    scores = _chunk_scores[_pair_offsets[_ji]:_pair_offsets[_ji + 1]]
                    ranked = sorted(zip(scores, _all_hits[j]), key=lambda x: x[0], reverse=True)[:top_k]
                    _reranked_hits[j] = [h for _, h in ranked]
                    qid = pending[j][1].get("id", f"q{pending[j][0]}")
                    _rerank_ckpt[qid] = [h[0] for h in _reranked_hits[j]]
                _rerank_ckpt_path.write_text(json.dumps(_rerank_ckpt))
                print(f"  Reranked {_chunk_end_display}/{len(_pending_rerank_indices)} pending", flush=True)
            _all_hits = _reranked_hits
            print("  Reranking complete.", flush=True)

        for j, (i, q) in enumerate(pending):
            qid  = q.get("id", f"q{i}")
            hits = _all_hits[j]
            expansion_stats = _all_expansion_stats[j]

            if use_rerank and together_rerank_client is not None:
                docs   = [chunk.content for _, _, chunk in hits]
                resp   = together_rerank_client.rerank.create(
                    model=RERANK_MODEL_TOGETHER,
                    query=q["question"],
                    documents=docs,
                    top_n=top_k,
                )
                hits   = [hits[r.index] for r in resp.results]
            elif use_rerank and cohere_rerank_client is not None:
                docs   = [chunk.content for _, _, chunk in hits]
                resp   = cohere_rerank_client.rerank(
                    model=RERANK_MODEL_COHERE,
                    query=q["question"],
                    documents=docs,
                    top_n=top_k,
                )
                hits   = [hits[r.index] for r in resp.results]
            elif not use_rerank:
                hits = hits[:top_k]
            if redundancy_threshold is not None:
                hits = _prune_redundant(hits, store.vector._conn, redundancy_threshold)
            chunk_texts = [chunk.content or "" for _, _, chunk in hits]

            # ── Enhancement #3: Query Complexity Routing ──────────────────────
            _inject_community = community_index is not None
            if _inject_community and query_complexity_threshold > 0:
                _score = _query_complexity(q["question"])
                if _score < query_complexity_threshold:
                    _inject_community = False

            # Collect unique community topic labels for retrieved chunks
            community_header = ""
            if _inject_community:
                labels = []
                seen_cids: set = set()
                for cid, _, _ in hits:
                    comm_id = community_index.community_id(cid)
                    if comm_id is not None and comm_id not in seen_cids:
                        seen_cids.add(comm_id)
                        lbl = community_index.topic_label(cid, min_coherence=community_min_coherence)
                        if lbl:
                            labels.append(lbl)
                if labels:
                    community_header = "Topic context: " + "; ".join(dict.fromkeys(labels)) + "\n\n"

            def _fmt_chunk(cid, chunk):
                text = chunk.content or ""
                if use_breadcrumb_context and chunk.breadcrumb:
                    text = f"{_format_breadcrumb(chunk.breadcrumb, style=breadcrumb_style)}\n\n{text}"
                return text

            wi = {
                "_slot":          len(work_items),
                "qid":            qid,
                "question":       q["question"],
                "source":         q.get("source", q.get("subset", "?")),
                "qtype":          q.get("question_type", "?"),
                "context":        community_header + "\n\n".join(
                    _fmt_chunk(cid, chunk) for cid, _, chunk in hits
                ),
                "chunk_ids":      [cid for cid, _, _ in hits],
                "scores":         [float(sc) for _, sc, _ in hits],
                "chunk_texts":    chunk_texts,
                "expansion_stats": expansion_stats,
                "evidence":       q.get("evidence", []),
                "gold":           str(q.get("answer", "")),
            }
            work_items.append(wi)

            _upsert_results_to_db(run_db, [{
                "id": wi["qid"], "question": wi["question"], "source": wi["source"],
                "question_type": wi["qtype"], "context": wi["context"],
                "evidence": wi["evidence"], "generated_answer": None,
                "gold_answer": wi["gold"], "retrieved_chunks": wi["chunk_ids"],
                "retrieved_scores": wi["scores"],
                "entity_ref_expansion": wi.get("expansion_stats"),
                "entity_ref_retry": None,
            }])
            if (j + 1) % 100 == 0 or (j + 1) == len(pending):
                pct = 100 * (j + 1) // len(pending)
                print(f"  Generated {j+1}/{len(pending)} ({pct}%)", flush=True)

    # Build one client per endpoint (round-robin for parallelism across multiple dedicated endpoints)
    endpoint_ids: list[str] = getattr(args, "endpoint_ids", None) or [args.gen_model]
    if args.gen_provider == "together":
        clients = [
            openai.OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url=TOGETHER_BASE_URL, timeout=120.0)
            for _ in endpoint_ids
        ]
    else:
        clients = [openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)]
        endpoint_ids = [args.gen_model]

    n_endpoints = len(clients)
    print(f"Generating answers with {args.concurrency} parallel workers across {n_endpoints} endpoint(s)...")

    retry_ner_fn = None
    if use_entity_ref_retry:
        from chonk.ner import SpacyMatcher
        _retry_matcher = SpacyMatcher(model=SPACY_MODEL, strip_numeric=True)
        retry_ner_fn = lambda text: [m.display_name for m in _retry_matcher.match(text)]

    new_results: list[dict] = []
    ckpt_lock   = threading.Lock()
    done_count  = [len(done_ids)]

    def _process(item: dict) -> dict:
        slot   = item["_slot"] % n_endpoints
        client = clients[slot]
        model  = endpoint_ids[slot]
        for attempt in range(3):
            try:
                answer = _generate(item["question"], item["context"], client, model,
                                   temperature=gen_temperature, structured=use_structured_gen)
                break
            except Exception as exc:
                if attempt == 2:
                    answer = f"[ERROR: {exc}]"
                else:
                    time.sleep(2 ** attempt)

        retry_stats: dict | None = None
        if use_entity_ref_retry and retry_ner_fn is not None:
            q_entities = retry_ner_fn(item["question"])
            uncovered: list[str] = []
            if q_entities:
                ent_vecs = embed_model.encode(q_entities, normalize_embeddings=True, show_progress_bar=False)
                ans_vec  = embed_model.encode([answer], normalize_embeddings=True, show_progress_bar=False)[0]
                sims     = ent_vecs @ ans_vec  # (n_entities,)
                uncovered = [e for e, s in zip(q_entities, sims) if s < 0.3]
            if uncovered:
                hint = (
                    f"Your answer does not appear to address the following key concepts from the question: "
                    f"{', '.join(uncovered)}. "
                    f"Please re-read the context and provide an answer that directly relates to these concepts."
                )
                retry_answer = answer
                for attempt in range(3):
                    try:
                        retry_answer = _generate(item["question"], item["context"], client, model,
                                                 temperature=gen_temperature, retry_hint=hint,
                                                 structured=use_structured_gen)
                        break
                    except Exception:
                        if attempt == 2:
                            retry_answer = answer
                        else:
                            time.sleep(2 ** attempt)
                retry_stats = {
                    "invoked": True,
                    "uncovered_entities": uncovered,
                }
                answer = retry_answer

        result = {
            "id":               item["qid"],
            "question":         item["question"],
            "source":           item["source"],
            "question_type":    item["qtype"],
            "context":          item["context"],
            "evidence":         item["evidence"],
            "generated_answer": answer,
            "gold_answer":      item["gold"],
            "retrieved_chunks": item["chunk_ids"],
            "retrieved_scores": item["scores"],
        }
        if item.get("expansion_stats") is not None:
            result["entity_ref_expansion"] = item["expansion_stats"]
        if retry_stats is not None:
            result["entity_ref_retry"] = retry_stats
        return result

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_process, item): item for item in work_items}
        for fut in as_completed(futures):
            result = fut.result()
            with ckpt_lock:
                new_results.append(result)
                done_count[0] += 1
                total = done_count[0]
                if total % 50 == 0:
                    print(f"  {total}/{len(questions)}", flush=True)
                # Checkpoint every 100 completions
                if len(new_results) % 100 == 0:
                    batch = new_results[-100:]
                    with open(ckpt_f, "a") as f:
                        for r in batch:
                            f.write(json.dumps(r) + "\n")
                    _upsert_results_to_db(run_db, batch)
                    print(f"  {total}/{len(questions)}  (checkpoint saved)", flush=True)

    # Final checkpoint flush + merge
    with open(ckpt_f, "a") as f:
        checkpointed = set()
        if ckpt_f.exists():
            pass  # already flushed incrementally
        remainder = len(new_results) % 100
        if remainder:
            for r in new_results[-remainder:]:
                f.write(json.dumps(r) + "\n")

    all_results: list[dict] = []
    if ckpt_f.exists():
        with open(ckpt_f) as f:
            for line in f:
                all_results.append(json.loads(line))
    ckpt_ids = {r["id"] for r in all_results}
    for r in new_results:
        if r["id"] not in ckpt_ids:
            all_results.append(r)

    with open(results_f, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    _rerank_ckpt_path.unlink(missing_ok=True)
    _write_results_to_db(run_db, all_results)
    _run_completed[0] = True
    print(f"\nComplete: {len(all_results)} results → {results_f} + {run_db}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Evaluate (benchmark's native metric)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Together dedicated endpoint lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def cmd_bench_eval(args: argparse.Namespace) -> None:
    """Run benchmark's native generation_eval.py on our output with checkpointing."""
    import asyncio
    import sys

    import numpy as np

    out_dir     = Path(args.out_dir)
    data_dir    = out_dir / "data"
    results_dir = out_dir / "results"
    repo_dir    = out_dir / "GraphRAG-Benchmark"
    run_name    = getattr(args, "run_name", "contextual")
    results_f   = results_dir / f"{run_name}.jsonl"
    ckpt_f      = results_dir / f"bench_eval_ckpt_{run_name}.jsonl"
    out_f       = results_dir / f"bench_eval_{run_name}.json"

    run_db = _run_db_path(data_dir, run_name)
    if run_db.exists():
        records = _read_results_from_db(run_db)
    elif results_f.exists():
        records = [json.loads(line) for line in open(results_f)]
        _init_run_db(run_db)
        _write_results_to_db(run_db, records)
    else:
        print(f"No results found: {results_f}. Run 'run' first.")
        return
    if not repo_dir.exists():
        print(f"Benchmark repo not found at {repo_dir}. Run 'download' first.")
        return
    question_ids_file = getattr(args, 'question_ids', None)
    if question_ids_file:
        with open(question_ids_file) as _f:
            allowed_ids = set(json.load(_f))
        records = [r for r in records if r["id"] in allowed_ids]
    if args.limit:
        records = records[:args.limit]

    # Convert to benchmark format
    bench_records = [{
        "id":            r["id"],
        "question":      r["question"],
        "question_type": r["question_type"],
        "generated_answer": r["generated_answer"],
        "ground_truth":  r.get("gold_answer") or r.get("ground_truth", ""),
        "context":       [r["context"]],
    } for r in records]

    # Load checkpoint
    done: dict[str, dict] = {}
    if ckpt_f.exists():
        for line in open(ckpt_f):
            item = json.loads(line)
            done[item["id"]] = item
        print(f"Resuming bench-eval: {len(done)} already done")
        # Flush checkpoint items to DB immediately so they're visible in report
        if done and run_db.exists():
            con = _connect_with_retry(run_db)
            for item in done.values():
                con.execute("DELETE FROM eval_scores WHERE id = ?", [item.get("id")])
                con.execute(
                    "INSERT INTO eval_scores VALUES (?,?,?,?,?,?)",
                    [item.get("id"), item.get("question_type"),
                     item.get("answer_correctness"), item.get("rouge_score"),
                     item.get("coverage_score"), item.get("faithfulness")],
                )
            con.close()
            print(f"  Flushed {len(done)} checkpoint scores to DB")

    pending = [r for r in bench_records if r["id"] not in done]
    print(f"Pending: {len(pending)} samples")

    if not pending:
        print("All samples already evaluated.")
    else:
        # Add benchmark repo to path
        sys.path.insert(0, str(repo_dir))
        import httpx
        from Evaluation.metrics import (
            compute_answer_correctness,
            compute_coverage_score,
            compute_faithfulness_score,
            compute_rouge_score,
        )
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        judge_provider = getattr(args, "judge_provider", "openai")
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )
        _judge_kwargs = dict(
            model=args.judge,
            temperature=0.0,
            top_p=1,
            seed=42,
            presence_penalty=0,
            frequency_penalty=0,
            max_retries=0,
            timeout=90,
            http_async_client=_http_client,
        )
        if judge_provider == "together":
            llm = ChatOpenAI(
                **_judge_kwargs,
                base_url="https://api.together.xyz/v1",
                api_key=SecretStr(os.environ["TOGETHER_API_KEY"]),
            )
        else:
            llm = ChatOpenAI(
                **_judge_kwargs,
                base_url="https://api.openai.com/v1",
                api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
            )

        # ── Pre-compute embeddings in one batched pass ──────────────────────
        # Ground truth embeddings are shared across runs; cache to disk.
        from langchain_core.embeddings import Embeddings as LCEmbeddings
        from sentence_transformers import SentenceTransformer

        gt_cache_f = out_dir / "data" / "gt_embeddings.npy"
        gt_id_f    = out_dir / "data" / "gt_embedding_ids.json"

        embed_model = SentenceTransformer(EMBED_MODEL)

        # Ground truth embeddings — cached by sorted ID set so any run subset
        # or reordering reuses the same cache file via ID lookup.
        _gt_by_id = {r["id"]: r["ground_truth"] for r in bench_records}
        _gt_ids_sorted = sorted(_gt_by_id.keys())

        _gt_cache_vecs = None
        if gt_cache_f.exists() and gt_id_f.exists():
            _cached_gt_ids = json.loads(gt_id_f.read_text())
            # Hit if cached IDs are a superset of what we need (common in full→grid direction)
            if set(_gt_ids_sorted).issubset(set(_cached_gt_ids)):
                print("Loading cached ground-truth embeddings...")
                _all_gt_vecs = np.load(str(gt_cache_f))
                _cached_id_idx = {qid: i for i, qid in enumerate(_cached_gt_ids)}
                _gt_cache_vecs = np.stack([_all_gt_vecs[_cached_id_idx[qid]] for qid in _gt_ids_sorted])

        if _gt_cache_vecs is None:
            _gt_texts_sorted = [_gt_by_id[qid] for qid in _gt_ids_sorted]
            print(f"Encoding {len(_gt_texts_sorted)} ground-truth texts (batched)...")
            _gt_cache_vecs = embed_model.encode(
                _gt_texts_sorted, batch_size=32, normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            # Only expand cache if this run has more questions than what's cached
            if not gt_cache_f.exists() or len(_gt_ids_sorted) > len(json.loads(gt_id_f.read_text()) if gt_id_f.exists() else []):
                np.save(str(gt_cache_f), _gt_cache_vecs)
                gt_id_f.write_text(json.dumps(_gt_ids_sorted), encoding="utf-8")
                print(f"  Cached → {gt_cache_f}")

        # Align to bench_records order
        _gt_sorted_idx = {qid: i for i, qid in enumerate(_gt_ids_sorted)}
        gt_ids  = [r["id"] for r in bench_records]
        gt_vecs = np.stack([_gt_cache_vecs[_gt_sorted_idx[rid]] for rid in gt_ids])

        # Answer embeddings (per run — cached)
        ans_cache_f = out_dir / "data" / f"ans_embeddings_{run_name}.npy"
        ans_id_f    = out_dir / "data" / f"ans_embedding_ids_{run_name}.json"
        all_ans_ids   = [r["id"] for r in bench_records]
        all_ans_texts = [r["generated_answer"] for r in bench_records]
        if ans_cache_f.exists() and ans_id_f.exists() and json.loads(ans_id_f.read_text()) == all_ans_ids:
            print("Loading cached answer embeddings...")
            all_ans_vecs = np.load(str(ans_cache_f))
        else:
            print(f"Encoding {len(all_ans_texts)} answer texts (batched)...")
            all_ans_vecs = embed_model.encode(
                all_ans_texts, batch_size=32, normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            np.save(str(ans_cache_f), all_ans_vecs)
            ans_id_f.write_text(json.dumps(all_ans_ids), encoding="utf-8")
            print(f"  Cached → {ans_cache_f}")
        del embed_model  # free memory

        # Build lookup: text → embedding vector
        emb_lookup: dict[str, np.ndarray] = {}
        for r, vec in zip(bench_records, gt_vecs):
            emb_lookup[r["ground_truth"]] = vec
        for r, vec in zip(bench_records, all_ans_vecs):
            emb_lookup[r["generated_answer"]] = vec

        class CachedEmbeddings(LCEmbeddings):
            """Returns pre-computed embeddings by text lookup; never calls the model."""
            def embed_documents(self, texts):
                return [emb_lookup[t].tolist() for t in texts]
            def embed_query(self, text):
                return emb_lookup[text].tolist()
            async def aembed_query(self, text):
                return emb_lookup[text].tolist()
            async def aembed_documents(self, texts):
                return [emb_lookup[t].tolist() for t in texts]

        embedding = CachedEmbeddings()

        METRIC_CONFIG = {
            "Fact Retrieval":       ["rouge_score", "answer_correctness"],
            "Complex Reasoning":    ["rouge_score", "answer_correctness"],
            "Contextual Summarize": ["answer_correctness", "coverage_score"],
            "Creative Generation":  ["answer_correctness", "coverage_score", "faithfulness"],
        }

        semaphore = asyncio.Semaphore(args.concurrency)
        _judge_model = getattr(args, "judge", GEN_MODEL)
        eval_rpm: int = getattr(args, "eval_rpm", None) or _model_rpm_limit(_judge_model)
        nan_limit: int | None = getattr(args, "nan_limit", None)
        _nan_final = [0]  # count of items finalized as NaN (shared across coroutines)

        # Token-bucket throttle: at most eval_rpm tokens per 60s window.
        _rpm_tokens   = [float(eval_rpm)]
        _rpm_last_ts  = [time.monotonic()]
        _rpm_lock     = asyncio.Lock()
        print(f"[eval] RPM limit: {eval_rpm} (judge={_judge_model})", flush=True)

        async def _acquire_rpm_token():
            async with _rpm_lock:
                import asyncio as _aio
                now = time.monotonic()
                elapsed = now - _rpm_last_ts[0]
                _rpm_tokens[0] = min(float(eval_rpm), _rpm_tokens[0] + elapsed * (eval_rpm / 60.0))
                _rpm_last_ts[0] = now
                if _rpm_tokens[0] < 1.0:
                    wait = (1.0 - _rpm_tokens[0]) / (eval_rpm / 60.0)
                    print(f"[eval] RPM throttle: sleeping {wait:.1f}s", flush=True)
                    await _aio.sleep(wait)
                    _rpm_tokens[0] = 0.0
                    _rpm_last_ts[0] = time.monotonic()
                else:
                    _rpm_tokens[0] -= 1.0

        async def _eval_one(r: dict) -> dict:
            import asyncio as _aio
            qtype   = r["question_type"]
            metrics = METRIC_CONFIG.get(qtype, ["answer_correctness"])
            result  = {"id": r["id"], "question_type": qtype}
            for attempt in range(5):
                # Hold semaphore only during the API call — release it before any retry sleep
                # so other items can proceed while this one waits.
                async with semaphore:
                    _t0 = time.monotonic()
                    try:
                        tasks = {}
                        if "rouge_score" in metrics:
                            tasks["rouge_score"] = compute_rouge_score(r["generated_answer"], r["ground_truth"])
                        if "answer_correctness" in metrics:
                            tasks["answer_correctness"] = compute_answer_correctness(
                                r["question"], r["generated_answer"], r["ground_truth"], llm, embedding
                            )
                        if "coverage_score" in metrics:
                            tasks["coverage_score"] = compute_coverage_score(
                                r["question"], r["ground_truth"], r["generated_answer"], llm
                            )
                        if "faithfulness" in metrics:
                            tasks["faithfulness"] = compute_faithfulness_score(
                                r["question"], r["generated_answer"], r["context"], llm
                            )
                        # answer_correctness = 3 LLM calls; others = 1 each; rouge_score = 0
                        _rpm_weights = {"answer_correctness": 3, "coverage_score": 1, "faithfulness": 1}
                        api_count = sum(_rpm_weights.get(k, 0) for k in tasks)
                        for _ in range(api_count):
                            await _acquire_rpm_token()
                        vals = await asyncio.gather(*tasks.values(), return_exceptions=True)
                        for key, val in zip(tasks.keys(), vals):
                            if isinstance(val, BaseException):
                                print(f"[eval] {r['id']} {key} exception: {type(val).__name__}: {val}", flush=True)
                            result[key] = float(val) if isinstance(val, (int, float)) else float("nan")
                        nan_metrics = [k for k in tasks if k != "rouge_score" and result.get(k) != result.get(k)]
                        if nan_metrics:
                            # Parse failure (API responded but non-compliant) — do not retry, deterministic
                            _nan_final[0] += 1
                        print(f"[eval] {r['id']} done in {time.monotonic()-_t0:.1f}s metrics={list(tasks.keys())}", flush=True)
                        return result
                    except Exception as e:
                        is_rate = "429" in str(e) or "rate_limit" in str(e).lower() or "RateLimit" in type(e).__name__
                        print(f"[eval] {r['id']} attempt {attempt+1}/5 {'rate-limit' if is_rate else 'error'}: {e}", flush=True)
                        if is_rate:
                            delay = 15 * (2 ** attempt)
                            print(f"[eval] backing off {delay}s", flush=True)
                        else:
                            for key in METRIC_CONFIG.get(qtype, ["answer_correctness"]):
                                result.setdefault(key, float("nan"))
                            _nan_final[0] += 1
                            return result
                # Sleep outside the semaphore so other items can run
                await _aio.sleep(5 * (attempt + 1))
            _nan_final[0] += 1
            return result

        async def _eval_one_safe(r: dict) -> dict:
            import asyncio as _aio
            try:
                return await _aio.wait_for(_eval_one(r), timeout=300)
            except TimeoutError:
                print(f"[eval] {r['id']} timed out after 300s — skipping", flush=True)
                qtype = r.get("question_type", "?")
                return {"id": r["id"], "question_type": qtype,
                        **{k: float("nan") for k in METRIC_CONFIG.get(qtype, ["answer_correctness"])}}

        async def _run_all():
            import asyncio as _aio
            _run_all_start = time.monotonic()
            _completed = [0]
            _ckpt_lock = _aio.Lock()

            async def _eval_and_save(r: dict):
                result = await _eval_one_safe(r)
                if not isinstance(result, dict):
                    return
                async with _ckpt_lock:
                    done[result["id"]] = result
                    with open(ckpt_f, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    if run_db.exists():
                        con = _connect_with_retry(run_db)
                        con.execute("DELETE FROM eval_scores WHERE id = ?", [result.get("id")])
                        con.execute(
                            "INSERT INTO eval_scores VALUES (?,?,?,?,?,?)",
                            [
                                result.get("id"), result.get("question_type"),
                                result.get("answer_correctness"), result.get("rouge_score"),
                                result.get("coverage_score"), result.get("faithfulness"),
                            ],
                        )
                        con.close()
                    _completed[0] += 1
                    _total_elapsed = time.monotonic() - _run_all_start
                    _qpm = _completed[0] / (_total_elapsed / 60.0) if _total_elapsed > 0 else 0
                    if _completed[0] % 20 == 0:
                        print(f"  {_completed[0]}/{len(pending)} evaluated | elapsed={_total_elapsed:.0f}s | avg={_qpm:.1f}q/min", flush=True)
                        await _check_early_stop()

            _es_target = getattr(args, "early_stop_target", None)
            _es_min_n  = getattr(args, "early_stop_min_n", 500)

            async def _check_early_stop():
                import math as _math
                if _es_target is None or len(done) < _es_min_n:
                    return
                _scores = [
                    v["answer_correctness"] for v in done.values()
                    if "answer_correctness" in v
                    and isinstance(v["answer_correctness"], float)
                    and not _math.isnan(v["answer_correctness"])
                ]
                if len(_scores) < _es_min_n:
                    return
                completed = _completed[0]
                _n    = len(_scores)
                _mean = sum(_scores) / _n
                _var  = sum((s - _mean) ** 2 for s in _scores) / _n
                _se   = _math.sqrt(_var / _n) if _var > 0 else 0.0
                _uci  = _mean + 1.645 * _se
                _n_remaining = len(pending) - completed
                _max_possible = (_n * _mean + _n_remaining) / (_n + _n_remaining) if (_n + _n_remaining) > 0 else 0.0
                print(f"  [early-stop] n={_n} mean={_mean:.4f} upper_95={_uci:.4f} max_possible={_max_possible:.4f} target={_es_target:.4f}", flush=True)
                if _max_possible < _es_target:
                    print(f"\n=== EARLY STOP: max_possible {_max_possible:.4f} < target {_es_target:.4f} — mathematically impossible to win ===", flush=True)
                    sys.exit(2)
                if _uci < _es_target:
                    print(f"\n=== EARLY STOP: upper_95 CI {_uci:.4f} < target {_es_target:.4f} — aborting ===", flush=True)
                    sys.exit(2)

            await _aio.gather(*[_eval_and_save(r) for r in pending], return_exceptions=True)

        asyncio.run(_run_all())

    # Aggregate results by question type
    by_type: dict[str, list] = defaultdict(list)
    for item in done.values():
        by_type[item["question_type"]].append(item)

    aggregated: dict[str, dict] = {}
    for qtype, items in by_type.items():
        agg: dict[str, float] = {}
        for key in ["rouge_score", "answer_correctness", "coverage_score", "faithfulness"]:
            vals = [i[key] for i in items if key in i and not (isinstance(i[key], float) and i[key] != i[key])]
            if vals:
                agg[key] = float(np.nanmean(vals))
        aggregated[qtype] = agg

    params = {
        "judge": args.judge,
        "judge_provider": getattr(args, "judge_provider", "openai"),
        "run_name": run_name,
        "n_evaluated": len(done),
    }
    with open(out_f, "w") as f:
        json.dump({"_params": params, **aggregated}, f, indent=2)

    # Write per-question eval scores to run DB
    if run_db.exists():
        con = _connect_with_retry(run_db)
        con.execute("DELETE FROM eval_scores")
        for item in done.values():
            con.execute(
                "INSERT INTO eval_scores VALUES (?,?,?,?,?,?)",
                [
                    item.get("id"), item.get("question_type"),
                    item.get("answer_correctness"), item.get("rouge_score"),
                    item.get("coverage_score"), item.get("faithfulness"),
                ],
            )
        con.close()

    print(f"\nBench eval complete → {out_f}")
    for qtype, scores in aggregated.items():
        print(f"  {qtype}: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))


def cmd_score(args: argparse.Namespace) -> None:
    """Print the All score (matches report: mean-of-means per subset×type) for a run.

    Outputs a single float (e.g. '0.6850') or 'nan' if the run is incomplete.
    Intended for shell scripts: SCORE=$(python ... score --run-name foo)
    """
    import math as _math
    from collections import defaultdict as _dd
    data_dir = Path(args.out_dir) / "data"
    run_db   = data_dir / "runs" / f"{args.run_name}.duckdb"
    if not run_db.exists():
        print("nan")
        return
    import duckdb as _ddb
    con = _ddb.connect(str(run_db), read_only=True)
    try:
        rows = con.execute(
            "SELECT e.question_type, r.source, e.answer_correctness "
            "FROM eval_scores e JOIN results r ON e.id = r.id"
        ).fetchall()
    except Exception:
        rows = []
    finally:
        con.close()
    # Group by (subset, question_type) — same logic as cmd_bench_report
    by_st: dict[tuple, list] = _dd(list)
    for qtype, source, ac in rows:
        if ac is None or (isinstance(ac, float) and _math.isnan(ac)):
            continue
        subset = "Med" if source and "medical" in str(source).lower() else "Nov"
        by_st[(subset, qtype)].append(float(ac))
    def _m(lst): return sum(lst) / len(lst) if lst else float("nan")  # noqa
    qtypes = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
    med_vals = [_m(by_st[(s, qt)]) for s in ["Med"] for qt in qtypes]
    nov_vals = [_m(by_st[(s, qt)]) for s in ["Nov"] for qt in qtypes]
    med = _m([v for v in med_vals if not _math.isnan(v)])
    nov = _m([v for v in nov_vals if not _math.isnan(v)])
    if _math.isnan(med) or _math.isnan(nov):
        print("nan")
    else:
        print(f"{(med + nov) / 2:.4f}")


def cmd_make_order(args: argparse.Namespace) -> None:
    """Generate a stratified question order file for representative full-corpus evaluation.

    Interleaves questions by (subset × question_type) so any prefix of N questions
    has the same stratum distribution as the full set.
    """
    out_dir  = Path(args.out_dir)
    data_dir = out_dir / "data"
    out_file = Path(args.output) if args.output else data_dir / "full_corpus_stratified_order.json"

    # Load questions tagged with their subset
    questions = []
    for subset in SUBSETS:
        f = data_dir / f"{subset}_questions.jsonl"
        if f.exists():
            with open(f) as fh:
                for line in fh:
                    q = json.loads(line)
                    q["_subset"] = subset
                    questions.append(q)

    # Bucket by (subset, question_type)
    buckets: dict[tuple, list] = defaultdict(list)
    for q in questions:
        key = (q.get("_subset", "unknown"), q.get("question_type", "unknown"))
        buckets[key].append(q.get("id", ""))

    # Assign fractional rank = position_within_bucket / bucket_size so that
    # sorting by this rank interleaves buckets proportionally (like a zipper).
    ranked: list[tuple] = []
    for key, ids in sorted(buckets.items()):
        n = len(ids)
        for pos, qid in enumerate(ids):
            ranked.append((pos / n, key, qid))
    ranked.sort(key=lambda x: (x[0], x[1]))
    ordered_ids = [x[2] for x in ranked]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(ordered_ids, f)

    total = len(ordered_ids)
    print(f"Stratified order: {total} questions → {out_file}")
    for key, ids in sorted(buckets.items()):
        print(f"  {key[0]}/{key[1]}: {len(ids)} ({100 * len(ids) / total:.1f}%)")


def cmd_bench_report(args: argparse.Namespace) -> None:
    """Combined matrix: our runs + full leaderboard, scored with benchmark's native metric."""
    out_dir     = Path(args.out_dir)
    data_dir    = out_dir / "data"
    results_dir = out_dir / "results"

    # Discover run DBs first, fall back to legacy bench_eval_*.json
    run_filter:  str | None = getattr(args, "filter", None)
    run_exclude: str | None = getattr(args, "exclude", None)
    run_results: dict[str, dict] = {}
    runs_dir = data_dir / "runs"
    if runs_dir.exists():
        import duckdb as _ddb
        import numpy as _np
        for run_db in sorted(runs_dir.glob("*.duckdb")):
            rn = run_db.stem
            if run_filter and run_filter not in rn:
                continue
            if run_exclude and run_exclude in rn:
                continue
            try:
                con = _ddb.connect(str(run_db), read_only=True)
                rows = con.execute(
                    "SELECT e.question_type, r.source, e.answer_correctness "
                    "FROM eval_scores e JOIN results r ON e.id = r.id"
                ).fetchall()
                total_results = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
                con.close()
            except Exception:
                continue
            if not rows:
                continue
            n_eval = len(rows)
            n_nan = sum(1 for _, _, ac in rows if ac is None or ac != ac)
            # by_subset_type[(source, qtype)] -> [ac, ...]
            by_st: dict[tuple, list] = {}
            for qtype, source, ac in rows:
                subset = "Med" if source and "medical" in source.lower() else "Nov"
                for key in [(subset, qtype), ("All", qtype)]:
                    if key not in by_st:
                        by_st[key] = []
                    if ac is not None and ac == ac:
                        by_st[key].append(ac)
            def _mean(lst): return float(_np.mean(lst)) if lst else float("nan")  # noqa
            scores_st = {k: _mean(v) for k, v in by_st.items()}
            run_results[rn] = {"scores_st": scores_st, "n_eval": n_eval, "n_total": total_results, "n_nan": n_nan}

    # Legacy bench_eval_*.json — no Med/Nov split, but load All-level scores for comparison
    import json as _json
    QTYPE_MAP = {
        "Fact Retrieval": "Fact", "Complex Reasoning": "Rsn",
        "Contextual Summarize": "Summ", "Creative Generation": "Crea",
    }
    for jf in sorted((results_dir).glob("bench_eval_*.json")):
        rn = jf.stem[len("bench_eval_"):]
        if rn in run_results:
            continue  # already loaded from DuckDB
        if run_filter and run_filter not in rn:
            continue
        if run_exclude and run_exclude in rn:
            continue
        try:
            data = _json.loads(jf.read_text())
        except Exception:
            continue
        params = data.get("_params", {})
        n_eval = params.get("n_evaluated", 0)
        scores_st: dict = {}
        type_scores = []
        for long, short in QTYPE_MAP.items():
            ac = data.get(long, {}).get("answer_correctness")
            if ac is not None:
                scores_st[("All", long)] = ac
                type_scores.append(ac)
        if type_scores:
            import numpy as _np2
            scores_st[("All", "All")] = float(_np2.mean(type_scores))
        if scores_st:
            run_results[rn] = {"scores_st": scores_st, "n_eval": n_eval, "n_total": n_eval, "legacy": True}

    QTYPES = [
        ("Fact Retrieval",       "Fact"),
        ("Complex Reasoning",    "Rsn"),
        ("Contextual Summarize", "Summ"),
        ("Creative Generation",  "Crea"),
    ]
    SUBSETS = ["Med", "Nov", "All"]

    print("\n── Benchmark Eval Results (answer_correctness) ──\n")
    # Header: Run | Med-Fact Med-Rsn Med-Summ Med-Crea Med | Nov-Fact ... Nov | All-Fact ... All
    col_w = 6
    run_w = 52
    hdr_parts = []
    for subset in SUBSETS:
        for _, short in QTYPES:
            hdr_parts.append(f"{subset[:1]}-{short:>4}")
        hdr_parts.append(f"  {subset:>3}")
    header = f"  {'Run':<{run_w}}  " + "  ".join(hdr_parts)
    print(header)
    print("  " + "-" * (len(header) - 2))

    fmt = lambda v: f"{v:.3f}" if v == v else "  — "  # noqa: E731

    def composite(scores_st, subset):
        if subset == "All":
            med = composite(scores_st, "Med")
            nov = composite(scores_st, "Nov")
            valid = [v for v in [med, nov] if v == v]
            return sum(valid) / len(valid) if valid else float("nan")
        vals = [scores_st.get((subset, qt), float("nan")) for qt, _ in QTYPES]
        valid = [v for v in vals if v == v]
        return sum(valid) / len(valid) if valid else float("nan")

    for run_name, info in sorted(run_results.items()):
        st = info.get("scores_st", {})
        n_eval = info.get("n_eval", 0)
        n_total = info.get("n_total", 0)
        n_nan = info.get("n_nan", 0)
        nan_str = f", {n_nan} NaN" if n_nan else ""
        partial = f" ({n_eval}/{n_total}{nan_str})" if n_total and (n_eval < n_total or n_nan) else (f" ({n_nan} NaN)" if n_nan else "")
        parts = []
        for subset in SUBSETS:
            for qt, _ in QTYPES:
                parts.append(f"{fmt(st.get((subset, qt), float('nan'))):>6}")
            parts.append(f"  {fmt(composite(st, subset)):>5}")
        print(f"  {run_name:<{run_w}}  " + "  ".join(parts) + partial)

    # Leaderboard (gpt-4o-mini judge and generator per benchmark paper arXiv:2506.05690)
    # Qwen2.5-14B appendix numbers from paper (Table 7, RAG w/ rerank)
    # Combined med+nov avg per question type is not directly available — showing overall acc
    print("\n── Published Leaderboard (gpt-4o-mini generator + judge) ──\n")
    print(f"  {'Method':<28}  {'Med ACC':>8}  {'Nov ACC':>8}  {'Overall':>8}")
    print("  " + "-" * 58)
    for name, scores in PUBLISHED_BASELINES.items():
        med = f"{scores['med_acc']:.3f}" if scores["med_acc"] is not None else "   —"
        nov = f"{scores['nov_acc']:.3f}" if scores["nov_acc"] is not None else "   —"
        ov  = f"{scores['overall']:.3f}" if scores["overall"] is not None else "   —"
        print(f"  {name:<28}  {med:>8}  {nov:>8}  {ov:>8}")

    print("\n  * Leaderboard: gpt-4o-mini generator + gpt-4o-mini judge (arXiv:2506.05690).")
    print("  * Our bench-eval: gpt-4o-mini generator + gpt-4o-mini judge, answer_correctness metric.")
    print("  * Overall = mean(Med, Nov); Med/Nov = mean of 4 question-type ACC scores.")


# ─────────────────────────────────────────────────────────────────────────────
# Together dedicated endpoint lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def cmd_use_endpoints(args: argparse.Namespace) -> None:
    """Run benchmark using pre-existing Together dedicated endpoints."""
    import types
    run_name = getattr(args, "run_name", "contextual")
    rerank   = getattr(args, "rerank", False)
    enhanced = getattr(args, "enhanced", False)
    run_args = types.SimpleNamespace(
        out_dir=args.out_dir,
        gen_provider="together",
        gen_model=args.endpoint_ids[0],
        endpoint_ids=args.endpoint_ids,
        concurrency=args.concurrency * len(args.endpoint_ids),
        limit=None,
        run_name=run_name,
        rerank=rerank,
        enhanced=enhanced,
    )
    cmd_run(run_args)

    eval_args = types.SimpleNamespace(
        out_dir=args.out_dir,
        judge=args.judge,
        limit=None,
        concurrency=20,
        run_name=run_name,
    )
    cmd_bench_eval(eval_args)


def cmd_provision(args: argparse.Namespace) -> None:
    """Create N Together dedicated endpoints in parallel, run benchmark, stop all."""
    import time
    import types
    from concurrent.futures import ThreadPoolExecutor

    from together import Together
    from together.types.autoscaling_param import AutoscalingParam

    together_client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    n = args.num_endpoints

    # Store (id, name) tuples — API calls need id, generation needs name
    def _create_one(i: int) -> tuple[str, str]:
        ep = together_client.endpoints.create(
            model=args.model,
            display_name=f"{args.model.replace('/', '_')}_{i}",
            hardware=args.hardware,
            autoscaling=AutoscalingParam(min_replicas=1, max_replicas=1),
        )
        return (ep.id, ep.name)

    print(f"Creating {n} dedicated endpoint(s) for {args.model} on {args.hardware}...")
    with ThreadPoolExecutor(max_workers=n) as ex:
        ep_tuples = list(ex.map(_create_one, range(n)))
    endpoint_ids   = [t[0] for t in ep_tuples]   # internal IDs for lifecycle ops
    endpoint_names = [t[1] for t in ep_tuples]   # names for API model param
    print(f"Endpoints created (ids): {endpoint_ids}")
    print(f"Endpoint names: {endpoint_names}")

    def _wait_one(eid: str) -> bool:
        for _ in range(120):
            time.sleep(10)
            state = getattr(together_client.endpoints.retrieve(eid), "state", "unknown")
            print(f"  {eid[:20]}… [{state}]", flush=True)
            if state == "STARTED":
                return True
        return False

    print("Waiting for all endpoints to be ready...")
    with ThreadPoolExecutor(max_workers=n) as ex:
        ready = list(ex.map(_wait_one, endpoint_ids))

    failed = [eid for eid, ok in zip(endpoint_ids, ready) if not ok]
    for eid in failed:
        together_client.endpoints.update(eid, state="STOPPED")
        endpoint_ids.remove(eid)

    if not endpoint_ids:
        print("No endpoints became ready. Aborting.")
        return

    print(f"\n{len(endpoint_ids)} endpoint(s) ready. Starting benchmark...")

    try:
        args.gen_provider  = "together"
        args.gen_model     = endpoint_names[0]   # fallback for single-endpoint path
        args.endpoint_ids  = endpoint_names       # names are used as model param
        args.concurrency   = args.concurrency * len(endpoint_names)
        args.limit         = None
        cmd_run(args)

        eval_args = types.SimpleNamespace(
            out_dir=args.out_dir,
            judge="gpt-4.1",
            limit=None,
            concurrency=20,
        )
        cmd_bench_eval(eval_args)
    finally:
        print("\nStopping all endpoints...")
        for eid in endpoint_ids:
            together_client.endpoints.update(eid, state="STOPPED")
            print(f"  Stopped {eid}")


def cmd_prime_cache(args: argparse.Namespace) -> None:
    """Pre-compute and persist question + entity embedding caches."""
    import hashlib
    from pathlib import Path

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from chonk.ner import SpacyMatcher

    out_dir   = Path(args.out_dir)
    data_dir  = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    embed_model_name = args.embed_model or EMBED_MODEL
    spacy_model_name = args.spacy_model or SPACY_MODEL

    _embed_device = os.environ.get("EMBED_DEVICE") or None
    if not _embed_device:
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available():
                _embed_device = "cpu"
        except ImportError:
            pass

    print(f"Loading embed model: {embed_model_name}" + (f" [{_embed_device}]" if _embed_device else ""))
    embed_model = SentenceTransformer(embed_model_name, device=_embed_device) if _embed_device else SentenceTransformer(embed_model_name)

    # ── Question embeddings ───────────────────────────────────────────────────
    corpus_qs  = _load_questions(data_dir)
    corpus_ids = [q.get("id", f"q{i}") for i, q in enumerate(corpus_qs)]
    q_vecs_cache = data_dir / "question_embeddings.npy"
    q_ids_cache  = data_dir / "question_ids.json"

    if q_vecs_cache.exists() and q_ids_cache.exists():
        if json.loads(q_ids_cache.read_text()) == corpus_ids:
            print(f"Question embeddings already cached ({len(corpus_ids)} questions). Skipping.")
        else:
            q_vecs_cache.unlink()
            q_ids_cache.unlink()
            print("Question ID mismatch — rebuilding question embeddings.")

    if not q_vecs_cache.exists():
        print(f"Embedding {len(corpus_qs)} questions...")
        q_vecs = embed_model.encode(
            [q["question"] for q in corpus_qs],
            normalize_embeddings=True, show_progress_bar=True, batch_size=256,
        ).astype("float32")
        np.save(str(q_vecs_cache), q_vecs)
        q_ids_cache.write_text(json.dumps(corpus_ids), encoding="utf-8")
        print(f"Saved → {q_vecs_cache.name}")

    # ── Entity embeddings ─────────────────────────────────────────────────────
    cache_key    = hashlib.md5(
        (embed_model_name + spacy_model_name + json.dumps(corpus_ids)).encode()
    ).hexdigest()[:16]
    ent_vecs_cache = data_dir / f"entity_vecs_{cache_key}.npz"
    ent_ents_cache = data_dir / f"entity_ents_{cache_key}.json"

    if ent_vecs_cache.exists() and ent_ents_cache.exists():
        print(f"Entity caches already exist ({ent_vecs_cache.name}). Skipping.")
    else:
        print(f"Running NER on {len(corpus_qs)} questions (spaCy: {spacy_model_name})...")
        ner = SpacyMatcher(model=spacy_model_name, strip_numeric=True)
        q_entities_by_id: dict[str, list[str]] = {}
        for i, q in enumerate(corpus_qs):
            qid = corpus_ids[i]
            q_entities_by_id[qid] = [m.name for m in ner.match(q["question"])]
            if (i + 1) % 500 == 0 or (i + 1) == len(corpus_qs):
                print(f"  NER {i+1}/{len(corpus_qs)}", flush=True)

        all_unique_ents = list({e for ents in q_entities_by_id.values() for e in ents})
        print(f"Batch-embedding {len(all_unique_ents)} unique entities...")
        ent_vecs = embed_model.encode(
            all_unique_ents, normalize_embeddings=True,
            show_progress_bar=True, batch_size=256,
        ).astype("float32")
        np.savez(str(ent_vecs_cache), strings=np.array(all_unique_ents), vecs=ent_vecs)
        ent_ents_cache.write_text(json.dumps(q_entities_by_id), encoding="utf-8")
        print(f"Saved → {ent_vecs_cache.name}, {ent_ents_cache.name}")

    print("prime-cache complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Chunky Monkey GraphRAG-Bench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # download
    p = sub.add_parser("download", help="Download dataset and clone benchmark repo")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.set_defaults(func=cmd_download)

    # inspect
    p = sub.add_parser("inspect", help="Inspect dataset structure and check corpus availability")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.set_defaults(func=cmd_inspect)

    # index (replaces chunk + embed)
    p = sub.add_parser("index", help="Chunk + embed + store corpus via chonk")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--force", action="store_true", help="Delete existing index and reindex")
    p.add_argument("--embed-content-only", action="store_true",
                   help="Embed content only (no breadcrumb); stores to chonk_nobc.duckdb")
    p.add_argument("--min-chunk", type=int, default=None)
    p.add_argument("--max-chunk", type=int, default=None)
    p.add_argument("--with-ner", action="store_true",
                   help="Run NER + cluster after indexing and persist to DB")
    p.set_defaults(func=cmd_index)

    # build-ner
    p = sub.add_parser("build-ner", help="Run NER + persist chunk_entities to an existing index DB")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--db-name", required=True, help="DB filename inside {out_dir}/data/")
    p.add_argument("--with-embeddings", action="store_true", dest="with_embeddings",
                   help="Also embed entity strings and store in entity_embeddings table (required for --ner-x)")
    p.add_argument("--force", action="store_true", help="Rebuild even if tables already populated")
    p.set_defaults(func=cmd_build_ner)

    # index-vanilla
    p = sub.add_parser("index-vanilla", help="Build vanilla RAG index: naive 256-token chunks")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--force", action="store_true", help="Delete existing index and reindex")
    p.add_argument("--chunk-tokens", type=int, default=None)
    p.set_defaults(func=cmd_index_vanilla)

    # run
    p = sub.add_parser("run", help="Retrieve and generate answers for all questions")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--limit",       type=int, default=None, metavar="N",
                   help="Limit to first N questions (for testing)")
    p.add_argument("--gen-model",     default=GEN_MODEL,
                   help=f"Generation model (default: {GEN_MODEL})")
    p.add_argument("--gen-provider",  default="openai", choices=["openai", "together"],
                   help="API provider for generation: openai or together (default: openai)")
    p.add_argument("--concurrency",   type=int, default=20,
                   help="Parallel workers (default: 20)")
    p.add_argument("--run-name",      default="contextual",
                   help="Output file prefix (default: contextual)")
    # ── Base ──────────────────────────────────────────────────────────────────
    g_base = p.add_argument_group("base", "Core retrieval mode")
    g_base.add_argument("--vanilla", action="store_true",
                        help=f"Use vanilla RAG index ({VANILLA_CHUNK_TOKENS}-token chunks, k={VANILLA_K})")
    g_base.add_argument("--rerank", action="store_true",
                        help=f"Rerank top-{K_FETCH} candidates to top-{K}")
    g_base.add_argument("--rerank-provider", default="local", choices=["local", "together", "cohere"],
                        help=f"Reranker: local={RERANK_MODEL}, together={RERANK_MODEL_TOGETHER} (default: local)")
    g_base.add_argument("--rerank-chunk", type=int, default=200, metavar="N",
                        help="Outer loop batch size for reranking + checkpoint interval (default: 200)")

    # ── Expansion ─────────────────────────────────────────────────────────────
    g_exp = p.add_argument_group(
        "expansion",
        "Grow the candidate pool beyond vector seed. Dependencies: --enhanced required for all expansion flags.")
    g_exp.add_argument("--enhanced", action="store_true",
                       help=f"Enable entity/structural/cluster expansion (SpacyMatcher/{SPACY_MODEL})")
    g_exp.add_argument("--ner-x", action="store_true", dest="ner_x",
                       help="Add entity embedding ANN expansion; requires --enhanced + build-ner --with-embeddings")
    g_exp.add_argument("--cluster", action="store_true",
                       help="Enable cluster-neighbor expansion (off by default); requires --enhanced")
    g_exp.add_argument("--entity-ref-expansion", action="store_true", dest="entity_ref_expansion",
                       help="Post-selection: if query entities absent from top-k, re-search and insert covering chunks")
    g_exp.add_argument("--entity-ref-expansion-per-k", type=int, default=None, dest="entity_ref_expansion_per_k",
                       help="Per-entity retrieval k for --entity-ref-expansion (default: k / n_missing)")
    g_exp.add_argument("--entity-ref-expansion-min-sim", type=float, default=None, dest="entity_ref_expansion_min_sim",
                       help="Min cosine sim for --entity-ref-expansion hits (default: no filter)")
    g_exp.add_argument("--breadcrumb-embed", action="store_true",
                       help="Use bc-in-embedding index (chunkymonkey_1100_2200.duckdb); improves seed quality for structured docs")

    # ── Pool Management ───────────────────────────────────────────────────────
    g_pool = p.add_argument_group(
        "pool_management",
        "Control which expanded candidates survive to the final top-k. Meaningful only when --enhanced is set.")
    g_pool.add_argument("--lane-entity-min-sim", type=float, default=None, dest="lane_entity_min_sim",
                        help="Quality gate: drop entity-adjacent chunks with query-sim < threshold before pool merge (e.g. 0.45)")
    g_pool.add_argument("--redundancy-threshold", type=float, default=None, dest="redundancy_threshold",
                        help="Dedup: drop near-duplicate chunks from merged pool before top-k selection (e.g. 0.92); fires after lane filter")
    g_pool.add_argument("--concentration-threshold", type=float, default=None, dest="concentration_threshold",
                        help="Auto-trigger --entity-ref-expansion if ≥ this fraction of top-k is from one doc (e.g. 0.6)")

    # ── Context Enrichment ────────────────────────────────────────────────────
    g_ctx = p.add_argument_group(
        "context_enrichment",
        "Modify what the generator sees. Independent of retrieval quality.")
    g_ctx.add_argument("--community-context", action="store_true", dest="community_context",
                       help="Inject community topic labels into generation prompt (requires build-community)")
    g_ctx.add_argument("--community-min-coherence", type=float, default=0.0, dest="community_min_coherence",
                       help="Suppress community labels with coherence < threshold (default: 0.0)")
    g_ctx.add_argument("--query-complexity-threshold", type=int, default=2, dest="query_complexity_threshold",
                       help="Skip community context for low-complexity queries (default: 2)")
    g_ctx.add_argument("--breadcrumb-context", action="store_true",
                       help="Prepend breadcrumb heading to each chunk in the generator context window")
    g_ctx.add_argument("--breadcrumb-style", default="markdown",
                       choices=["markdown", "literal", "symbol"], dest="breadcrumb_style",
                       help="Breadcrumb format in context: markdown (## headings), literal (Section: X.), symbol ([X > Y])")

    # ── Misc ──────────────────────────────────────────────────────────────────
    g_misc = p.add_argument_group("misc")
    g_misc.add_argument("--entity-ref-retry", action="store_true", dest="entity_ref_retry",
                        help="On refusal answer, retry with partial-answer hint (max 1 retry)")
    g_misc.add_argument("--structured-gen", action="store_true", dest="structured_gen",
                        help="Require ANSWER: <text> format from generator; retry once if non-compliant; strip marker before judge")
    g_misc.add_argument("--top-k", type=int, default=None, dest="top_k", metavar="K",
                        help=f"Override retrieval top-k (default: {K})")
    g_misc.add_argument("--db-name", default=None, dest="db_name",
                        help="Override DB filename inside {out_dir}/data/ (default: auto from --breadcrumb-embed)")
    g_misc.add_argument("--question-ids", default=None, dest="question_ids", metavar="PATH",
                        help="JSON file with list of question IDs to run (default: all)")

    p.set_defaults(func=cmd_run)

    # build-community
    p = sub.add_parser("build-community", help="Build community index: heading vectors + weighted avg + Louvain detection")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--db-name", required=True, help="DB filename inside {out_dir}/data/")
    p.add_argument("--alpha", type=float, default=0.2,
                   help="Heading weight in weighted average (default: 0.2)")
    p.add_argument("--sim-threshold", type=float, default=0.6, dest="sim_threshold",
                   help="Min cosine sim for graph edges (default: 0.6)")
    p.add_argument("--community-label-strategy", default="ner_embedding",
                   choices=["term_freq", "ner_embedding"], dest="community_label_strategy",
                   help="Community label method: ner_embedding (default) uses entity embeddings; term_freq uses word frequency")
    p.add_argument("--force", action="store_true", help="Rebuild even if already populated")
    p.set_defaults(func=cmd_build_community)

    # use-endpoints — run benchmark against existing Together dedicated endpoints
    p = sub.add_parser("use-endpoints", help="Run benchmark using existing Together dedicated endpoints")
    p.add_argument("endpoint_ids", nargs="+", metavar="ENDPOINT_ID",
                   help="One or more Together dedicated endpoint IDs")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Workers per endpoint (default: 20, total = N * 20)")
    p.add_argument("--judge",       default="gpt-4o-mini")
    p.add_argument("--run-name",    default="contextual",
                   help="Output file prefix (default: contextual)")
    p.add_argument("--rerank",      action="store_true",
                   help=f"Rerank top-{K_FETCH} candidates to top-{K} with {RERANK_MODEL}")
    p.add_argument("--enhanced",    action="store_true",
                   help=f"Use NER+cluster EnhancedSearch ({SPACY_MODEL} + agglomerative clustering)")
    p.set_defaults(func=cmd_use_endpoints)

    # provision — create Together dedicated endpoint, run benchmark, stop endpoint
    p = sub.add_parser("provision", help="Create Together dedicated endpoint, run benchmark, stop endpoint")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--model",       default="Qwen/Qwen2.5-14B-Instruct",
                   help="Model to deploy (default: Qwen/Qwen2.5-14B-Instruct)")
    p.add_argument("--hardware",       default="2x_nvidia_h100_80gb_sxm",
                   help="Together hardware tier (default: 2x_nvidia_h100_80gb_sxm)")
    p.add_argument("--num-endpoints",  type=int, default=1,
                   help="Number of parallel dedicated endpoints (default: 1)")
    p.add_argument("--concurrency",    type=int, default=20,
                   help="Workers per endpoint (default: 20, total = N * 20)")
    p.set_defaults(func=cmd_provision)

    # eval — score a run with benchmark's native answer_correctness metric
    p = sub.add_parser("eval", help="Score a run with benchmark's native answer_correctness metric")
    p.add_argument("--out-dir",     default="/tmp/grb", metavar="DIR")
    p.add_argument("--run-name",    default="contextual",
                   help="Results file prefix to evaluate (default: contextual)")
    p.add_argument("--judge",          default="gpt-4o-mini",
                   help="Judge model (default: gpt-4o-mini)")
    p.add_argument("--judge-provider", default="openai", choices=["openai", "together"],
                   help="API provider for judge (default: openai)")
    p.add_argument("--limit",          type=int, default=None, metavar="N")
    p.add_argument("--question-ids",   default=None, metavar="PATH",
                   help="JSON file with list of question IDs to evaluate")
    p.add_argument("--concurrency",    type=int, default=5,
                   help="Parallel workers (default: 5)")
    p.add_argument("--eval-rpm",       type=int, default=None, dest="eval_rpm",
                   help="Max judge API requests per minute (token-bucket throttle, default: no limit)")
    p.add_argument("--eval-batch-size", type=int, default=10, dest="eval_batch_size",
                   help="Items per async batch (default: 10)")
    p.add_argument("--nan-limit",      type=int, default=None, dest="nan_limit",
                   help="Stop retrying NaN items once total finalized-NaN count is ≤ this threshold (default: always retry)")
    p.add_argument("--early-stop-target", type=float, default=None, dest="early_stop_target",
                   help="Abort eval (exit 2) when upper 95%% CI of answer_correctness falls below this score")
    p.add_argument("--early-stop-min-n",  type=int,   default=500,  dest="early_stop_min_n",
                   help="Min evaluated questions before early-stop check fires (default: 500)")
    p.set_defaults(func=cmd_bench_eval)

    # score — print All score for a completed run (for shell script use)
    p = sub.add_parser("score", help="Print All score (mean of Med+Nov answer_correctness) for a run")
    p.add_argument("--out-dir",  default="/tmp/grb", metavar="DIR")
    p.add_argument("--run-name", required=True, help="Run name to score")
    p.set_defaults(func=cmd_score)

    # make-order — generate stratified question order for full-corpus runs
    p = sub.add_parser("make-order", help="Generate stratified question order file for full-corpus runs")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Output JSON path (default: {out_dir}/data/full_corpus_stratified_order.json)")
    p.set_defaults(func=cmd_make_order)

    # prime-cache — pre-compute and persist all question + entity embeddings
    p = sub.add_parser("prime-cache", help="Pre-compute question and entity embedding caches (run before 'run')")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--embed-model", default=None, metavar="MODEL",
                   help=f"Embedding model (default: {EMBED_MODEL})")
    p.add_argument("--spacy-model", default=None, metavar="MODEL",
                   help=f"spaCy NER model (default: {SPACY_MODEL})")
    p.set_defaults(func=cmd_prime_cache)

    # report — combined matrix of all eval runs + leaderboard
    p = sub.add_parser("report", help="Combined matrix: our eval runs + leaderboard")
    p.add_argument("--out-dir", default="/tmp/grb", metavar="DIR")
    p.add_argument("--filter", default=None, metavar="STR",
                   help="Only show runs whose name contains STR (e.g. 'grid' or 'full')")
    p.add_argument("--exclude", default=None, metavar="STR",
                   help="Exclude runs whose name contains STR (e.g. 'full' to show only grid runs)")
    p.set_defaults(func=cmd_bench_report)

    return ap


def main() -> None:
    ap   = _make_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
