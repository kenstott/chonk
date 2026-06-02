# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 043d5c95-ea19-4084-9218-abba10f0a83e
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""NoSQLRenderer — annotates DocumentChunks from NoSQL crawler JSON payloads.

NoSQL crawlers (MongoDB, Elasticsearch, Solr, DynamoDB, Firestore, Cosmos)
embed a ``_source_meta`` key in each document's JSON payload.  This renderer
strips that key from the displayed content and copies it to
``chunk.source_detail`` so callers can trace a chunk back to its origin.

``chunk.source_detail`` for a MongoDB document looks like::

    {
        "type": "mongodb",
        "uri": "mongodb+srv://cluster.mongodb.net",
        "database": "mydb",
        "collection": "articles",
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import DocumentChunk


def _walk(obj: Any, lines: list[str], depth: int, path: str) -> None:
    heading = "#" * min(depth, 6)
    if isinstance(obj, dict):
        for key, val in obj.items():
            child_path = f"{path} > {key}" if path else key
            if isinstance(val, (dict, list)):
                lines.append(f"{heading} {child_path}")
                _walk(val, lines, depth + 1, child_path)
            else:
                lines.append(f"{heading} {child_path}")
                if val is not None and str(val).strip():
                    lines.append(str(val))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            if isinstance(item, (dict, list)):
                lines.append(f"{heading} {child_path}")
                _walk(item, lines, depth + 1, child_path)
            else:
                if item is not None and str(item).strip():
                    lines.append(str(item))
    else:
        if obj is not None and str(obj).strip():
            lines.append(str(obj))


class NoSQLRenderer:
    """Renderer for JSON documents emitted by NoSQL crawlers.

    Fires when the parsed JSON object contains a ``_source_meta`` key,
    strips it from the rendered output, and stamps ``chunk.source_detail``
    with the connection metadata.
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:
        return isinstance(obj, dict) and "_source_meta" in obj

    def render(self, obj: object) -> str:
        assert isinstance(obj, dict)
        clean = {k: v for k, v in obj.items() if k != "_source_meta"}
        lines: list[str] = []
        _walk(clean, lines, depth=1, path="")
        return "\n\n".join(lines)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        meta: dict = obj.get("_source_meta", {}) if isinstance(obj, dict) else {}  # type: ignore[union-attr]
        for chunk in chunks:
            chunk.source_detail = meta
        return chunks
