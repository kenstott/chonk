# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 1ac793cd-2c5a-4faa-967a-741b147e26bb
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""TypeScript/JavaScript source code extractor using regex + brace-depth tracking."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chonk.models import DocumentChunk

_IMPORT_RE = re.compile(r"^import\b")
_CLASS_RE = re.compile(r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)")
_INTERFACE_RE = re.compile(r"^(?:export\s+)?interface\s+(\w+)")
_FUNCTION_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(")
_CONST_ARROW_RE = re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(")
_METHOD_RE = re.compile(
    r"^\s{2,}(?:(?:public|private|protected|static|async|readonly|override)\s+)*(\w+)\s*[=(]\s*"
)


def _strip_jsdoc(lines: list[str]) -> str:
    """Extract text from JSDoc comment lines."""
    text_lines: list[str] = []
    for line in lines:
        stripped = line.strip().lstrip("/*").strip()
        if stripped:
            text_lines.append(stripped)
    return " ".join(text_lines)


_KEYWORD_NAMES = {"if", "for", "while", "switch", "return", "else", "try", "catch"}


def _collect_imports(lines: list[str]) -> tuple[list[str], list[int]]:
    """Return (import_lines, import_indices) for all import statements."""
    indices = [idx for idx, ln in enumerate(lines) if _IMPORT_RE.match(ln.strip())]
    return [lines[idx] for idx in indices], indices


def _emit_imports(
    lines: list[str],
    parts: list[str],
    section_map: dict[tuple[str, ...], dict[str, object]],
) -> None:
    """Append import section to parts and section_map in-place."""
    import_lines, indices = _collect_imports(lines)
    if not indices:
        return
    parts.append("## Imports\n\n```typescript\n" + "\n".join(import_lines) + "\n```")
    section_map[("Imports",)] = {
        "line_start": indices[0] + 1,
        "line_end": indices[-1] + 1,
        "symbol": "Imports",
    }


def _advance_jsdoc(
    stripped: str, line: str, in_jsdoc: bool, pending_jsdoc: list[str]
) -> tuple[bool, list[str]]:
    """Update JSDoc tracking state for the current line. Returns (in_jsdoc, pending_jsdoc)."""
    if stripped.startswith("/**"):
        pending_jsdoc = [line]
        in_jsdoc = "*/" not in stripped
        return in_jsdoc, pending_jsdoc
    # already inside a JSDoc block
    pending_jsdoc.append(line)
    if "*/" in stripped:
        in_jsdoc = False
    return in_jsdoc, pending_jsdoc


def _consume_jsdoc(pending_jsdoc: list[str]) -> str:
    """Return stripped JSDoc text and clear the list."""
    return _strip_jsdoc(pending_jsdoc) if pending_jsdoc else ""


def _handle_class_or_interface(
    stripped: str,
    line: str,
    i: int,
    brace_depth: int,
    class_stack: list[tuple[str, int]],
    pending_jsdoc: list[str],
    parts: list[str],
) -> tuple[bool, int, int, list[str]]:
    """Handle a class or interface declaration line.

    Returns (matched, next_i, new_brace_depth, cleared_pending_jsdoc).
    """
    m = _CLASS_RE.match(stripped)
    if m:
        name = m.group(1)
        jsdoc = _consume_jsdoc(pending_jsdoc)
        parts.append(f"# {name}")
        if jsdoc:
            parts.append(jsdoc)
        depth_before = brace_depth
        brace_depth += line.count("{") - line.count("}")
        class_stack.append((name, depth_before))
        return True, i + 1, brace_depth, []

    m = _INTERFACE_RE.match(stripped)
    if m:
        name = m.group(1)
        jsdoc = _consume_jsdoc(pending_jsdoc)
        parts.append(f"# {name}")
        if jsdoc:
            parts.append(jsdoc)
        brace_depth += line.count("{") - line.count("}")
        return True, i + 1, brace_depth, []

    return False, i, brace_depth, pending_jsdoc


def _handle_top_level_callable(
    stripped: str,
    i: int,
    brace_depth: int,
    class_stack: list[tuple[str, int]],
    pending_jsdoc: list[str],
    parts: list[str],
    section_map: dict[tuple[str, ...], dict[str, object]],
    lines: list[str],
) -> tuple[bool, int, int, list[str]]:
    """Handle a top-level function or const-arrow declaration.

    Returns (matched, next_i, new_brace_depth, cleared_pending_jsdoc).
    """
    if class_stack:
        return False, i, brace_depth, pending_jsdoc

    m = _FUNCTION_RE.match(stripped) or _CONST_ARROW_RE.match(stripped)
    if not m:
        return False, i, brace_depth, pending_jsdoc

    name = m.group(1)
    jsdoc = _consume_jsdoc(pending_jsdoc)
    parts.append(f"# {name}")
    if jsdoc:
        parts.append(jsdoc)
    line_start = i + 1
    body, i, brace_depth = _collect_body(lines, i, brace_depth)
    section_map[(name,)] = {
        "line_start": line_start,
        "line_end": i,
        "symbol": name,
    }
    parts.append(f"```typescript\n{body}\n```")
    return True, i, brace_depth, []


def _handle_class_method(
    line: str,
    i: int,
    brace_depth: int,
    class_stack: list[tuple[str, int]],
    pending_jsdoc: list[str],
    parts: list[str],
    section_map: dict[tuple[str, ...], dict[str, object]],
    lines: list[str],
) -> tuple[bool, int, int, list[str]]:
    """Handle a method declaration inside the current class body.

    Returns (matched, next_i, new_brace_depth, cleared_pending_jsdoc).
    """
    if not class_stack:
        return False, i, brace_depth, pending_jsdoc

    class_name, class_depth = class_stack[-1]
    if brace_depth != class_depth + 1:
        return False, i, brace_depth, pending_jsdoc

    m = _METHOD_RE.match(line)
    if not m:
        return False, i, brace_depth, pending_jsdoc

    mname = m.group(1)
    if mname in _KEYWORD_NAMES:
        return False, i, brace_depth, pending_jsdoc

    jsdoc = _consume_jsdoc(pending_jsdoc)
    parts.append(f"## {mname}")
    if jsdoc:
        parts.append(jsdoc)
    line_start = i + 1
    body, i, brace_depth = _collect_body(lines, i, brace_depth)
    section_map[(class_name, mname)] = {
        "line_start": line_start,
        "line_end": i,
        "symbol": f"{class_name}.{mname}",
    }
    parts.append(f"```typescript\n{body}\n```")
    if class_stack and brace_depth <= class_stack[-1][1]:
        class_stack.pop()
    return True, i, brace_depth, []


class TypeScriptExtractor:
    def can_handle(self, doc_type: str) -> bool:
        return doc_type in {"typescript", "javascript"}

    def _scan(self, source: str) -> tuple[str, dict[tuple[str, ...], dict[str, object]]]:
        """Return (markdown, section_map) where section_map maps section key → line metadata."""
        lines = source.splitlines()
        parts: list[str] = []
        section_map: dict[tuple[str, ...], dict[str, object]] = {}

        _emit_imports(lines, parts, section_map)

        brace_depth = 0
        class_stack: list[tuple[str, int]] = []
        pending_jsdoc: list[str] = []
        in_jsdoc = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith("/**") or in_jsdoc:
                in_jsdoc, pending_jsdoc = _advance_jsdoc(stripped, line, in_jsdoc, pending_jsdoc)
                i += 1
                continue

            matched, i, brace_depth, pending_jsdoc = _handle_class_or_interface(
                stripped, line, i, brace_depth, class_stack, pending_jsdoc, parts
            )
            if matched:
                continue

            matched, i, brace_depth, pending_jsdoc = _handle_top_level_callable(
                stripped, i, brace_depth, class_stack, pending_jsdoc, parts, section_map, lines
            )
            if matched:
                continue

            matched, i, brace_depth, pending_jsdoc = _handle_class_method(
                line, i, brace_depth, class_stack, pending_jsdoc, parts, section_map, lines
            )
            if matched:
                continue

            pending_jsdoc = []
            delta = line.count("{") - line.count("}")
            brace_depth += delta
            while class_stack and brace_depth <= class_stack[-1][1]:
                class_stack.pop()

            i += 1

        return "\n\n".join(parts), section_map

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        if not data:
            return ""

        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"UTF-8 decode failed in {source_path or '<unknown>'}: {exc}") from exc

        markdown, _ = self._scan(source)
        return markdown

    def annotate(
        self,
        chunks: list[DocumentChunk],
        data: bytes,
        source_path: str | None = None,
    ) -> list[DocumentChunk]:
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"UTF-8 decode failed in {source_path or '<unknown>'}: {exc}") from exc

        _, section_map = self._scan(source)

        for chunk in chunks:
            key = tuple(chunk.section)
            if key in section_map:
                chunk.source_detail = section_map[key]

        return chunks


def _collect_body(lines: list[str], start: int, current_depth: int) -> tuple[str, int, int]:
    """Collect lines from start until the opening brace's matching close brace."""
    collected = []
    depth = current_depth
    found_open = False
    i = start

    while i < len(lines):
        line = lines[i]
        collected.append(line)
        opens = line.count("{")
        closes = line.count("}")
        if opens > 0:
            found_open = True
        depth += opens - closes
        i += 1
        if found_open and depth <= current_depth:
            break

    return "\n".join(collected), i, depth
