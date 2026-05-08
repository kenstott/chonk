# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7c4b1e9f-d23a-4f7e-8b05-c1d2e3f4a5b6
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""XML extractor — emits markdown with element-path breadcrumb headings."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ._renderer import Renderer


def _tag(element: ET.Element) -> str:
    tag = element.tag
    if tag.startswith("{"):
        tag = tag[tag.index("}") + 1 :]
    return tag


def _walk(element: ET.Element, lines: list[str], depth: int, path: str) -> None:
    tag = _tag(element)
    current_path = f"{path} > {tag}" if path else tag
    heading = "#" * min(depth, 6)
    lines.append(f"{heading} {current_path}")

    text = (element.text or "").strip()
    if text:
        lines.append(text)

    for child in element:
        _walk(child, lines, depth + 1, current_path)

    tail = (element.tail or "").strip()
    if tail:
        lines.append(tail)


def _to_dict(element: ET.Element) -> dict:
    """Convert an ElementTree element to a plain dict for renderer dispatch."""
    result: dict = {"_tag": _tag(element), **element.attrib}
    children = list(element)
    if children:
        child_dicts: dict[str, list] = {}
        for child in children:
            key = _tag(child)
            child_dicts.setdefault(key, []).append(_to_dict(child))
        for key, vals in child_dicts.items():
            result[key] = vals if len(vals) > 1 else vals[0]
    else:
        text = (element.text or "").strip()
        if text:
            result["_text"] = text
    return result


class XmlExtractor:
    """Extract XML files into markdown.

    When a ``Renderer`` matches the parsed root object it takes over rendering
    and annotation entirely.  Otherwise falls back to the generic element-path
    walk.

    Args:
        renderers: Optional list of :class:`Renderer` instances to try before
                   the generic walk.  First match wins.
    """

    HANDLED = {"xml"}

    def __init__(self, renderers: list[Renderer] | None = None) -> None:
        self._renderers: list[Renderer] = renderers or []

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def _parse(self, data: bytes) -> ET.Element | None:
        try:
            return ET.fromstring(data)
        except ET.ParseError:
            return None

    def _find_renderer(self, source_path: str | None, obj: object) -> Renderer | None:
        for r in self._renderers:
            if r.can_render(source_path, obj):
                return r
        return None

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        root = self._parse(data)
        if root is None:
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1")

        renderer = self._find_renderer(source_path, _to_dict(root))
        if renderer:
            return renderer.render(_to_dict(root))

        lines: list[str] = []
        _walk(root, lines, depth=1, path="")
        return "\n\n".join(lines)

    def annotate(self, chunks: list, data: bytes, source_path: str | None = None) -> list:
        root = self._parse(data)
        if root is None:
            return chunks

        obj = _to_dict(root)
        renderer = self._find_renderer(source_path, obj)
        if renderer:
            return renderer.annotate(chunks, obj)
        return chunks
