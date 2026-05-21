# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 3a7e2f1d-b849-4c8e-9d01-e5f6a2b3c4d7
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""JSON / JSONL extractor — emits markdown with key-path breadcrumb headings."""

from __future__ import annotations

import json
from typing import Any

from ._renderer import Renderer


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


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


class JsonExtractor:
    """Extract JSON / JSONL files into markdown.

    When a ``Renderer`` matches the parsed object it takes over rendering and
    annotation entirely.  Otherwise falls back to the generic key-path walk.

    Args:
        renderers: Optional list of :class:`Renderer` instances to try before
                   the generic walk.  First match wins.
        chunk_by_row: When True, each JSONL line becomes its own ``## heading``
            chunk (one chunk per row).  ``row_key`` names the field used as the
            heading label (defaults to the first string field).  No effect on
            single-object JSON files.
        row_key: Field name to use as the ``## heading`` label when
            ``chunk_by_row=True``.  If the field is absent, falls back to the
            first string field, then the row index.
    """

    HANDLED = {"json", "jsonl"}

    def __init__(
        self,
        renderers: list[Renderer] | None = None,
        chunk_by_row: bool = False,
        row_key: str | None = None,
    ) -> None:
        self._renderers: list[Renderer] = renderers or []
        self.chunk_by_row = chunk_by_row
        self.row_key = row_key

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def _find_renderer(self, source_path: str | None, obj: object) -> Renderer | None:
        for r in self._renderers:
            if r.can_render(source_path, obj):
                return r
        return None

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        text = _decode(data)

        # JSONL: one object per line — renderers not applied (no single root obj)
        if source_path and source_path.lower().endswith(".jsonl"):
            sections: list[str] = []
            for row_idx, raw_line in enumerate(text.splitlines()):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    sections.append(raw_line)
                    continue
                obj_lines: list[str] = []
                if self.chunk_by_row and isinstance(obj, dict):
                    label = None
                    if self.row_key:
                        label = str(obj.get(self.row_key, "")).strip() or None
                    if label is None:
                        for v in obj.values():
                            if isinstance(v, str) and v.strip():
                                label = v.strip()
                                break
                    if label is None:
                        label = str(row_idx)
                    obj_lines.append(f"## {label}")
                    for k, v in obj.items():
                        if k == self.row_key:
                            continue
                        if isinstance(v, list):
                            flat = ", ".join(str(i) for i in v if str(i).strip())
                            if flat:
                                obj_lines.append(f"{k}: {flat}")
                        elif v is not None and str(v).strip():
                            obj_lines.append(f"{k}: {v}")
                else:
                    _walk(obj, obj_lines, depth=1, path="")
                sections.append("\n".join(obj_lines))
            return "\n\n".join(sections)

        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return text

        renderer = self._find_renderer(source_path, obj)
        if renderer:
            return renderer.render(obj)

        lines = []
        _walk(obj, lines, depth=1, path="")
        return "\n\n".join(lines)

    def annotate(self, chunks: list, data: bytes, source_path: str | None = None) -> list:
        text = _decode(data)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return chunks

        renderer = self._find_renderer(source_path, obj)
        if renderer:
            return renderer.annotate(chunks, obj)
        return chunks
