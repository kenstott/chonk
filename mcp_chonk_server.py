#!/usr/bin/env python
"""
MCP server exposing Chonk search over one or more DuckDB indexes.

Requirements:
    pip install "chonk[storage]" mcp

Single DB (simple):
    CHONK_DB_PATH         – path to DuckDB file
    CHONK_EMBEDDING_DIM   – embedding dimension (int, default 1024)

Multiple DBs (named):
    CHONK_DB_CONFIG       – JSON object mapping name → {"path": "...", "embedding_dim": N}
                            e.g. '{"main": {"path": "/data/main.duckdb"},
                                   "archive": {"path": "/data/archive.duckdb", "embedding_dim": 768}}'

    When CHONK_DB_CONFIG is set it takes precedence over CHONK_DB_PATH.
    search_chunks accepts an optional "db" parameter to target a specific store;
    omitting it searches all stores and merges results by score.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from chonk.storage import Store

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_DIM = 1024


def _load_stores() -> dict[str, Store]:
    config_json = os.environ.get("CHONK_DB_CONFIG")
    if config_json:
        config = json.loads(config_json)
        return {
            name: Store(
                entry["path"],
                embedding_dim=int(entry.get("embedding_dim", _DEFAULT_DIM)),
                read_only=True,
            )
            for name, entry in config.items()
        }

    db_path = os.environ.get("CHONK_DB_PATH")
    if not db_path:
        raise RuntimeError(
            "Either CHONK_DB_PATH or CHONK_DB_CONFIG env var is required"
        )
    dim = int(os.environ.get("CHONK_EMBEDDING_DIM", str(_DEFAULT_DIM)))
    return {"default": Store(db_path, embedding_dim=dim, read_only=True)}


STORES: dict[str, Store] = _load_stores()

SERVER = Server("chonk-search")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_chunk(chunk_id: str, score: float, chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "score": float(score),
        "document_name": getattr(chunk, "document_name", ""),
        "section": getattr(chunk, "section", []),
        "chunk_type": getattr(chunk, "chunk_type", "document"),
        "source": getattr(chunk, "source", ""),
        "breadcrumb": getattr(chunk, "breadcrumb", None),
        "content": getattr(chunk, "content", ""),
        "source_offset": getattr(chunk, "source_offset", None),
        "source_length": getattr(chunk, "source_length", None),
        "source_detail": getattr(chunk, "source_detail", None),
    }


def _db_path_for(store_name: str) -> str:
    config_json = os.environ.get("CHONK_DB_CONFIG")
    if config_json:
        return json.loads(config_json)[store_name]["path"]
    return os.environ["CHONK_DB_PATH"]


def _fetch_chunk_by_id(store_name: str, chunk_id: str) -> dict[str, Any]:
    import duckdb

    from chonk.models import DocumentChunk
    from chonk.storage._vector import _deserialize_section  # type: ignore[attr-defined]

    conn = duckdb.connect(_db_path_for(store_name), read_only=True)
    row = conn.execute(
        """
        SELECT chunk_id, document_name, section, chunk_index, content,
               breadcrumb, chunk_type, source_offset, source_length, source_detail
        FROM embeddings WHERE chunk_id = ?
        """,
        [chunk_id],
    ).fetchone()
    conn.close()
    if row is None:
        raise KeyError(f"chunk_id not found: {chunk_id}")

    (
        cid,
        doc_name,
        section_raw,
        chunk_index,
        content,
        breadcrumb,
        chunk_type,
        source_offset,
        source_length,
        source_detail_str,
    ) = row

    chunk = DocumentChunk(
        document_name=doc_name,
        content=content,
        section=_deserialize_section(section_raw),
        chunk_index=chunk_index,
        source_offset=source_offset,
        source_length=source_length,
        breadcrumb=breadcrumb,
        chunk_type=chunk_type or "document",
        source_detail=json.loads(source_detail_str) if source_detail_str else None,
    )
    return _serialize_chunk(cid, 1.0, chunk)


def _fetch_neighbors(
    store_name: str,
    base_chunk: dict[str, Any],
    radius: int,
) -> list[dict[str, Any]]:
    import duckdb

    from chonk.models import DocumentChunk
    from chonk.storage._vector import _deserialize_section  # type: ignore[attr-defined]

    conn = duckdb.connect(_db_path_for(store_name), read_only=True)
    row = conn.execute(
        "SELECT chunk_index FROM embeddings WHERE chunk_id = ?",
        [base_chunk["chunk_id"]],
    ).fetchone()
    if row is None:
        conn.close()
        return []

    base_idx = int(row[0])
    rows = conn.execute(
        """
        SELECT chunk_id, document_name, section, chunk_index, content,
               breadcrumb, chunk_type, source_offset, source_length, source_detail
        FROM embeddings
        WHERE document_name = ? AND chunk_index BETWEEN ? AND ? AND chunk_id <> ?
        ORDER BY chunk_index
        """,
        [
            base_chunk["document_name"],
            base_idx - radius,
            base_idx + radius,
            base_chunk["chunk_id"],
        ],
    ).fetchall()
    conn.close()

    neighbors: list[dict[str, Any]] = []
    for (
        cid,
        dname,
        section_raw,
        cidx,
        content,
        breadcrumb,
        chunk_type,
        source_offset,
        source_length,
        source_detail_str,
    ) in rows:
        chunk = DocumentChunk(
            document_name=dname,
            content=content,
            section=_deserialize_section(section_raw),
            chunk_index=cidx,
            source_offset=source_offset,
            source_length=source_length,
            breadcrumb=breadcrumb,
            chunk_type=chunk_type or "document",
            source_detail=json.loads(source_detail_str) if source_detail_str else None,
        )
        neighbors.append(_serialize_chunk(cid, 1.0, chunk))
    return neighbors


# ---------------------------------------------------------------------------
# MCP: tool definitions
# ---------------------------------------------------------------------------


@SERVER.list_tools()
async def list_tools() -> list[Tool]:
    db_names = list(STORES.keys())
    return [
        Tool(
            name="search_chunks",
            description=(
                "Hybrid vector + BM25 search over the Chonk index. "
                "Supply a query_embedding (float array) computed from the user's question. "
                "Optionally include query_text for BM25 hybrid. "
                f"Available DBs: {db_names}. Omit 'db' to search all and merge by score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Embedding vector for the query, shape (dim,).",
                    },
                    "query_text": {
                        "type": "string",
                        "description": "Raw query text for BM25 hybrid search.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 5,
                    },
                    "db": {
                        "type": "string",
                        "description": (
                            f"Target a specific DB by name. One of: {db_names}. "
                            "Omit to search all."
                        ),
                    },
                    "namespaces": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "chunk_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            'e.g. ["document"], ["db_table","db_column"], ["api_endpoint"]'
                        ),
                    },
                },
                "required": ["query_embedding"],
            },
        ),
        Tool(
            name="get_chunk",
            description="Fetch a specific chunk by chunk_id, optionally with neighbors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "db": {
                        "type": "string",
                        "description": (
                            f"Which DB to query. One of: {db_names}. Defaults to first."
                        ),
                    },
                    "include_neighbors": {"type": "boolean", "default": False},
                    "neighbor_radius": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 1,
                    },
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="expand_chunk_graph",
            description=(
                "Expand a chunk into entity/relation/community overlays. "
                "Use after retrieving a chunk via search_chunks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "db": {
                        "type": "string",
                        "description": f"Which DB to query. One of: {db_names}.",
                    },
                },
                "required": ["chunk_id"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# MCP: single dispatch handler (correct API)
# ---------------------------------------------------------------------------


@SERVER.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    args = arguments or {}

    if name == "search_chunks":
        return await _search_chunks(args)
    if name == "get_chunk":
        return await _get_chunk(args)
    if name == "expand_chunk_graph":
        return await _expand_chunk_graph(args)

    raise ValueError(f"Unknown tool: {name!r}")


async def _search_chunks(args: dict[str, Any]) -> list[TextContent]:
    raw = args.get("query_embedding")
    if raw is None:
        raise ValueError("query_embedding is required")

    query_embedding = np.asarray(raw, dtype="float32")
    if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding[0]
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be shape (dim,) or (1, dim)")

    limit = int(args.get("limit", 5))
    query_text: str | None = args.get("query_text") or None
    namespaces: list[str] | None = args.get("namespaces")
    chunk_types: list[str] | None = args.get("chunk_types")
    target_db: str | None = args.get("db")

    if target_db and target_db not in STORES:
        raise ValueError(f"Unknown db {target_db!r}. Available: {list(STORES)}")
    target_stores = {target_db: STORES[target_db]} if target_db else STORES

    all_results: list[dict[str, Any]] = []
    for db_name, store in target_stores.items():
        for cid, score, chunk in store.search(
            query_embedding=query_embedding,
            limit=limit,
            query_text=query_text,
            namespaces=namespaces,
            chunk_types=chunk_types,
        ):
            row = _serialize_chunk(cid, score, chunk)
            row["db"] = db_name
            all_results.append(row)

    all_results.sort(key=lambda r: r["score"], reverse=True)
    all_results = all_results[:limit]

    wrapper = {
        "results": all_results,
        "usage": {
            "instructions": (
                "Each result has 'content' (the text to rely on), 'document_name', "
                "'section', 'breadcrumb' describing origin, 'score' (higher = more similar), "
                "and 'db' indicating which store it came from. "
                "Base your answer strictly on the 'content' fields. "
                "Cite document_name/section when helpful. "
                "If no content clearly answers the question, say you don't know."
            )
        },
    }
    return [TextContent(type="text", text=json.dumps(wrapper))]


async def _get_chunk(args: dict[str, Any]) -> list[TextContent]:
    chunk_id = args.get("chunk_id")
    if not chunk_id:
        raise ValueError("chunk_id is required")

    db_name = args.get("db") or next(iter(STORES))
    if db_name not in STORES:
        raise ValueError(f"Unknown db {db_name!r}. Available: {list(STORES)}")

    include_neighbors = bool(args.get("include_neighbors", False))
    neighbor_radius = int(args.get("neighbor_radius", 1))

    base = _fetch_chunk_by_id(db_name, chunk_id)
    out: dict[str, Any] = {"chunk": base, "db": db_name}
    if include_neighbors and neighbor_radius > 0:
        out["neighbors"] = _fetch_neighbors(db_name, base, neighbor_radius)

    return [TextContent(type="text", text=json.dumps(out))]


async def _expand_chunk_graph(args: dict[str, Any]) -> list[TextContent]:
    chunk_id = args.get("chunk_id", "")
    raise NotImplementedError(
        "expand_chunk_graph is not yet wired to an EntityIndex/RelationshipIndex. "
        "Implement using your graph/NER indices for chunk_id=" + chunk_id
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await SERVER.run(
            read_stream,
            write_stream,
            SERVER.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
