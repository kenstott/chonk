"""
AIPA test MCP server — chonk source + docs search.

Build the index first (once):
    python -c "from chonk.ingest import build; build('index_config.yaml')"

Then start the server:
    python server.py

Transport (CHONK_TRANSPORT env var):
    stdio   (default) — local subprocess; MCP host manages the process
    http    — Starlette/uvicorn; set CHONK_HOST / CHONK_PORT / CHONK_API_KEY

Claude Desktop config (stdio):
    {
      "mcpServers": {
        "aipa-test": {
          "command": "python",
          "args": ["/path/to/aipa_test_mcp_server/server.py"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

# Load ../.env
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    with _ENV_FILE.open() as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np  # noqa: E402
from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402

from chonk.storage import Store  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent / "index.duckdb"
_EMBEDDING_DIM = int(os.environ.get("CHONK_EMBEDDING_DIM", "1024"))
_EMBED_MODEL = os.environ.get("CHONK_EMBED_MODEL", "BAAI/bge-large-en-v1.5")

# LLM config for the `ask` (RAG synthesis) tool.
# Set CHONK_LLM_BACKEND to: together | ollama | anthropic
_LLM_BACKEND = os.environ.get(
    "CHONK_LLM_BACKEND", "together"
)  # "together" | "ollama" | "anthropic"
_CHAT_MODEL = os.environ.get("CHONK_CHAT_MODEL", "Qwen/Qwen3.5-9B")
_TOGETHER_BASE_URL = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_ANTHROPIC_API_VERSION = os.environ.get("ANTHROPIC_API_VERSION", "2023-06-01")
_ANSWER_TOKEN_BUDGET = int(os.environ.get("CHONK_ANSWER_TOKEN_BUDGET", "4096"))
_ANSWER_MAX_TOKENS = int(os.environ.get("CHONK_ANSWER_MAX_TOKENS", "1024"))

if not _DB_PATH.exists():
    raise RuntimeError(
        f"Index not found: {_DB_PATH}\n"
        "Build it first:\n"
        "    python -c \"from chonk.ingest import build; build('index_config.yaml')\""
    )

STORE = Store(str(_DB_PATH), embedding_dim=_EMBEDDING_DIM, read_only=True)

# Domain registry, resolved once from the index. The tools expose domain *names*
# (e.g. 'legacy', 'future_state'); Store.search filters by the opaque domain_id, so
# the names are translated here. Descriptions drive the self-documenting tool/server
# guidance below so a planner knows which domain to filter to.
_DOMAIN_NAME_TO_ID: dict[str, str] = {}
_DOMAIN_DESCRIPTIONS: dict[str, str] = {}
for _did, _name, _desc in STORE.vector._conn.execute(
    "SELECT domain_id, name, description FROM domains"
).fetchall():
    _DOMAIN_NAME_TO_ID[_name] = _did
    _DOMAIN_DESCRIPTIONS[_name] = _desc or ""
_DOMAIN_NAMES = sorted(_DOMAIN_NAME_TO_ID)


def _domain_guidance() -> str:
    """Bullet list of 'name — description' for every indexed domain."""
    return "\n".join(f"  - '{n}': {_DOMAIN_DESCRIPTIONS[n]}" for n in _DOMAIN_NAMES)


_SERVER_INSTRUCTIONS = (
    "Retrieval over the chonk project. Content is partitioned into domains; "
    "pass the `domains` argument to scope a query when the question is clearly "
    "about one of them, and omit it to search everything.\n"
    f"{_domain_guidance()}\n"
    "Tools: `search` returns ranked raw chunks (you synthesize); `ask` returns "
    "a synthesized answer with citations; `get_chunk` fetches one chunk by id with "
    "optional neighbors. Filter to 'future_state' for the Go-rewrite / target "
    "architecture, and to 'legacy' for the current Python codebase and its docs."
)

SERVER = Server("aipa-test-chonk", instructions=_SERVER_INSTRUCTIONS)

# Enhanced retrieval front-end for the ask tool. Returns ScoredChunk directly
# (with provenance + MMR reranking). Entity/cluster/graph expansion is inactive on
# this index (built with ner/community/svo disabled), but base reranking applies.
from chonk.search import EnhancedSearch  # noqa: E402

SEARCHER = EnhancedSearch(STORE)

# Lazy-loaded embedding model (loaded on first search call)
_MODEL: Any = None


def _get_model() -> Any:  # noqa: ANN401
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(_EMBED_MODEL)
    return _MODEL


def _embed(text: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype="float32")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_domain_ids(domains: list[str] | None) -> list[str] | None:
    if domains is None:
        return None
    unknown = [d for d in domains if d not in _DOMAIN_NAME_TO_ID]
    if unknown:
        raise ValueError(f"Unknown domain(s): {unknown}. Available: {_DOMAIN_NAMES}")
    return [_DOMAIN_NAME_TO_ID[d] for d in domains]


_SYSTEM_PROMPT = (
    "Answer concisely. Lead with the direct answer, then only the "
    "supporting detail the question needs. Prefer tight prose or a "
    "short list over long exposition. Do not pad, restate the "
    "question, or add a preamble. Ground every claim in the "
    "provided context, but synthesize in your own words — the "
    "sources are returned to the caller as citations, so do not "
    "reproduce source text verbatim or quote long passages."
)


def _together_chat(prompt: str) -> str:
    """Synchronous Together.ai chat completion."""
    import httpx

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set; the ask tool requires it.")

    resp = httpx.post(
        f"{_TOGETHER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": _CHAT_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": _ANSWER_MAX_TOKENS,
            "temperature": 0,
            # Qwen3.5 is a thinking model; disable CoT so the answer lands in
            # message.content rather than a separate reasoning field.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices")
    if not choices:
        raise RuntimeError(f"Together response had no choices: {data}")
    return choices[0]["message"]["content"]


def _ollama_chat(prompt: str) -> str:
    """Synchronous Ollama chat completion via /api/chat."""
    import httpx

    resp = httpx.post(
        f"{_OLLAMA_BASE_URL}/api/chat",
        json={
            "model": _CHAT_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": _ANSWER_MAX_TOKENS},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content")
    if not content:
        raise RuntimeError(f"Ollama response had no content: {data}")
    return content


def _anthropic_chat(prompt: str) -> str:
    """Synchronous Anthropic Messages API completion."""
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; the ask tool requires it.")

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        },
        json={
            "model": _CHAT_MODEL,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _ANSWER_MAX_TOKENS,
            "temperature": 0,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("content")
    if not content:
        raise RuntimeError(f"Anthropic response had no content: {data}")
    return content[0]["text"]


def _llm_fn(prompt: str) -> str:
    if _LLM_BACKEND == "ollama":
        return _ollama_chat(prompt)
    if _LLM_BACKEND == "anthropic":
        return _anthropic_chat(prompt)
    return _together_chat(prompt)


def _serialize(chunk_id: str, score: float, chunk: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "chunk_id": chunk_id,
        "score": float(score),
        "document_name": getattr(chunk, "document_name", ""),
        "section": getattr(chunk, "section", []),
        "namespace": getattr(chunk, "namespace", ""),
        "chunk_type": getattr(chunk, "chunk_type", "document"),
        "breadcrumb": getattr(chunk, "breadcrumb", None),
        "content": getattr(chunk, "content", ""),
        "source_detail": getattr(chunk, "source_detail", None),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@SERVER.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description=(
                "Semantic search over the chonk project source code and documentation. "
                "Searches across Python source (chonk/), docs/, examples/, and training manifests. "
                "Pass a plain-text query; the server embeds it internally. "
                "Returns ranked raw chunks with content, breadcrumb, domain, and score "
                "for you to read and synthesize. Optionally scope to a domain (see the "
                "`domains` argument); omit to search everything."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or code search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                    "domains": {
                        "type": "array",
                        "items": {"type": "string", "enum": _DOMAIN_NAMES},
                        "description": (
                            "Restrict the search to one or more domains; omit to search "
                            "all. Available domains:\n" + _domain_guidance()
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask",
            description=(
                "RAG answer synthesis. Retrieves the most relevant chunks for the query, "
                "then generates a grounded natural-language answer with source citations "
                f"via the Together.ai chat model ({_CHAT_MODEL}). Use this when you want a "
                "ready-made answer rather than raw chunks. Optionally scope retrieval to a "
                "domain (see the `domains` argument); omit to search everything."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question to answer from the indexed corpus.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 8,
                        "description": "Number of chunks to retrieve as context.",
                    },
                    "domains": {
                        "type": "array",
                        "items": {"type": "string", "enum": _DOMAIN_NAMES},
                        "description": (
                            "Restrict retrieval to one or more domains; omit to search "
                            "all. Available domains:\n" + _domain_guidance()
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_chunk",
            description="Fetch a specific chunk by chunk_id with optional surrounding context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "include_neighbors": {"type": "boolean", "default": False},
                    "neighbor_radius": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 1,
                    },
                },
                "required": ["chunk_id"],
            },
        ),
    ]


@SERVER.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    args = arguments or {}
    if name == "search":
        return await _search(args)
    if name == "ask":
        return await _ask(args)
    if name == "get_chunk":
        return await _get_chunk(args)
    raise ValueError(f"Unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _search(args: dict[str, Any]) -> list[TextContent]:
    query = args.get("query", "").strip()
    if not query:
        raise ValueError("query is required")

    limit = int(args.get("limit", 5))
    domains: list[str] | None = args.get("domains") or None
    domain_ids = _resolve_domain_ids(domains)

    embedding = await asyncio.get_event_loop().run_in_executor(None, _embed, query)

    results = []
    for chunk_id, score, chunk in STORE.search(
        query_embedding=embedding,
        limit=limit,
        query_text=query,
        domain_ids=domain_ids,
    ):
        results.append(_serialize(chunk_id, score, chunk))

    payload = {
        "results": results,
        "meta": {
            "query": query,
            "domains_filter": domains,
            "instructions": (
                "Each result has 'content' (the source text), 'document_name', "
                "'section', 'namespace', and 'score' (higher = more relevant). "
                "Base answers strictly on the content fields. "
                "Cite document_name/section when helpful. "
                "If no content clearly answers the question, say so."
            ),
        },
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def _ask(args: dict[str, Any]) -> list[TextContent]:
    from chonk.generation import AnswerContext, AnswerGenerator

    query = args.get("query", "").strip()
    if not query:
        raise ValueError("query is required")

    limit = int(args.get("limit", 8))
    domains: list[str] | None = args.get("domains") or None
    domain_ids = _resolve_domain_ids(domains)

    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, _embed, query)

    scored = await loop.run_in_executor(
        None,
        lambda: SEARCHER.search(
            query_embedding=embedding,
            k=limit,
            query_text=query,
            domain_ids=domain_ids,
        ),
    )

    context = AnswerContext(chunks=scored, query=query)
    generator = AnswerGenerator(_llm_fn, token_budget=_ANSWER_TOKEN_BUDGET)
    # generate() builds the prompt and makes a blocking HTTP call; run off-loop.
    answer = await asyncio.get_event_loop().run_in_executor(None, generator.generate, context)

    payload = {
        "answer": answer.text,
        "citations": [_serialize(c.chunk_id, c.score, c.chunk) for c in answer.citations],
        "meta": {
            "query": query,
            "domains_filter": domains,
            "model": _CHAT_MODEL,
            "chunks_retrieved": len(scored),
            "chunks_cited": len(answer.citations),
        },
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def _get_chunk(args: dict[str, Any]) -> list[TextContent]:
    import duckdb

    from chonk.models import DocumentChunk
    from chonk.storage._vector import _deserialize_section  # type: ignore[attr-defined]

    chunk_id = args.get("chunk_id", "").strip()
    if not chunk_id:
        raise ValueError("chunk_id is required")

    conn = duckdb.connect(str(_DB_PATH), read_only=True)
    row = conn.execute(
        """
        SELECT chunk_id, document_name, section, chunk_index, content,
               breadcrumb, chunk_type, source_offset, source_length, source_detail
        FROM embeddings WHERE chunk_id = ?
        """,
        [chunk_id],
    ).fetchone()

    if row is None:
        conn.close()
        raise KeyError(f"chunk_id not found: {chunk_id}")

    (
        cid,
        doc_name,
        section_raw,
        chunk_index,
        content,
        breadcrumb,
        chunk_type,
        src_offset,
        src_length,
        src_detail_str,
    ) = row

    base_chunk = DocumentChunk(
        document_name=doc_name,
        content=content,
        section=_deserialize_section(section_raw),
        chunk_index=chunk_index,
        source_offset=src_offset,
        source_length=src_length,
        breadcrumb=breadcrumb,
        chunk_type=chunk_type or "document",
        source_detail=json.loads(src_detail_str) if src_detail_str else None,
    )
    out: dict[str, Any] = {"chunk": _serialize(cid, 1.0, base_chunk)}

    include_neighbors = bool(args.get("include_neighbors", False))
    radius = int(args.get("neighbor_radius", 1))
    if include_neighbors and radius > 0:
        nb_rows = conn.execute(
            """
            SELECT chunk_id, document_name, section, chunk_index, content,
                   breadcrumb, chunk_type, source_offset, source_length, source_detail
            FROM embeddings
            WHERE document_name = ? AND chunk_index BETWEEN ? AND ? AND chunk_id <> ?
            ORDER BY chunk_index
            """,
            [doc_name, chunk_index - radius, chunk_index + radius, chunk_id],
        ).fetchall()
        neighbors = []
        for ncid, ndoc, nsec, nidx, ncontent, nbcrumb, nctype, noff, nlen, ndet in nb_rows:
            nc = DocumentChunk(
                document_name=ndoc,
                content=ncontent,
                section=_deserialize_section(nsec),
                chunk_index=nidx,
                source_offset=noff,
                source_length=nlen,
                breadcrumb=nbcrumb,
                chunk_type=nctype or "document",
                source_detail=json.loads(ndet) if ndet else None,
            )
            neighbors.append(_serialize(ncid, 1.0, nc))
        out["neighbors"] = neighbors

    conn.close()
    return [TextContent(type="text", text=json.dumps(out))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_TRANSPORT = os.environ.get("CHONK_TRANSPORT", "stdio").lower()


async def _run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await SERVER.run(
            read_stream,
            write_stream,
            SERVER.create_initialization_options(),
        )


async def _run_http() -> None:
    from contextlib import asynccontextmanager

    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    _API_KEY = os.environ.get("CHONK_API_KEY")
    _manager = StreamableHTTPSessionManager(app=SERVER, stateless=False)

    async def handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        if _API_KEY and scope.get("type") == "http":
            request = Request(scope, receive)
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {_API_KEY}":
                response = JSONResponse({"error": "Unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await _manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(_app):  # noqa: ANN001, ANN202
        async with _manager.run():
            yield

    app = Starlette(
        lifespan=lifespan,
        routes=[Mount("/mcp", app=handle_mcp)],
    )
    host = os.environ.get("CHONK_HOST", "0.0.0.0")
    port = int(os.environ.get("CHONK_PORT", "8000"))
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


def _log(msg: str) -> None:
    """Write a timestamped diagnostic line to stderr."""
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[aipa-mcp {ts}] {msg}", file=__import__("sys").stderr, flush=True)


async def main() -> None:
    _log(f"transport      : {_TRANSPORT}")
    _log(f"llm backend    : {_LLM_BACKEND}")
    _log(f"chat model     : {_CHAT_MODEL}")
    _log(f"embed model    : {_EMBED_MODEL}")
    _log(f"db path        : {_DB_PATH}")
    _log(f"domains        : {_DOMAIN_NAMES}")
    if _TRANSPORT == "http":
        host = os.environ.get("CHONK_HOST", "0.0.0.0")
        port = int(os.environ.get("CHONK_PORT", "8000"))
        _log(f"listening on   : http://{host}:{port}/mcp")
        await _run_http()
    else:
        _log("waiting for MCP host on stdin (stdio transport)")
        await _run_stdio()
    _log("server exited")


if __name__ == "__main__":
    asyncio.run(main())
