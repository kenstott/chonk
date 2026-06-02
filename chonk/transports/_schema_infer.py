# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 841a33cf-26a5-4248-a593-a2c90e748ba8
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Shared schema inference utilities for NoSQL crawlers."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _walk(doc: dict, counts: dict, prefix: str = "") -> None:
    """Recursively record field paths and their value types."""
    for key, value in doc.items():
        path = f"{prefix}.{key}" if prefix else key
        counts[path]["present"] += 1
        counts[path]["types"][_type_name(value)] += 1
        if isinstance(value, dict):
            _walk(value, counts, prefix=path)
        elif isinstance(value, list) and value:
            item_types: dict[str, int] = defaultdict(int)
            for item in value:
                item_types[_type_name(item)] += 1
                if isinstance(item, dict):
                    _walk(item, counts, prefix=f"{path}[]")
            counts[path]["item_types"] = item_types


def infer_schema_text(
    docs: list[dict],
    label: str,
    total_docs: int | None = None,
    sample_size: int | None = None,
) -> str:
    """Infer a human-readable union schema from a list of sampled documents.

    Args:
        docs:         Sample of documents (dicts).
        label:        Display label, e.g. ``"db/collection"`` or ``"index_name"``.
        total_docs:   Total document count if known (for the header).
        sample_size:  Number of docs sampled (defaults to len(docs)).

    Returns:
        Multi-line plain-text schema description suitable for chunking.
    """
    n = len(docs)
    if n == 0:
        return f"Source: {label}\nNo documents sampled — schema unavailable.\n"

    sample_size = sample_size or n
    counts: dict[str, Any] = defaultdict(lambda: {"present": 0, "types": defaultdict(int), "item_types": {}})

    for doc in docs:
        _walk(doc, counts)

    lines = [f"Source: {label}"]
    if total_docs is not None:
        lines.append(f"Sampled: {sample_size:,} / {total_docs:,} documents")
    else:
        lines.append(f"Sampled: {sample_size:,} documents")
    lines.append("")
    lines.append(f"{'Field':<40}  {'Type':<20}  Present")
    lines.append("-" * 72)

    for path in sorted(counts):
        info = counts[path]
        present_pct = int(100 * info["present"] / n)
        type_str = " | ".join(
            f"{t}×{c}" if c > 1 else t
            for t, c in sorted(info["types"].items(), key=lambda x: -x[1])
        )
        if info.get("item_types"):
            item_str = " | ".join(
                f"{t}×{c}" if c > 1 else t
                for t, c in sorted(info["item_types"].items(), key=lambda x: -x[1])
            )
            type_str += f"  → items: {item_str}"
        indent = "  " * (path.count(".") + path.count("[]"))
        lines.append(f"{indent}{path:<40}  {type_str:<20}  {present_pct}%")

    return "\n".join(lines) + "\n"


def collect_field_paths(docs: list[dict]) -> set[str]:
    """Return all dot-notation field paths observed across *docs*."""
    from collections import defaultdict
    counts: dict = defaultdict(lambda: {"present": 0, "types": defaultdict(int), "item_types": {}})
    for doc in docs:
        _walk(doc, counts)
    return set(counts.keys())
