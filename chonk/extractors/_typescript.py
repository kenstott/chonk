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
_METHOD_RE = re.compile(r"^\s{2,}(?:(?:public|private|protected|static|async|readonly|override)\s+)*(\w+)\s*[=(]\s*")


def _strip_jsdoc(lines: list[str]) -> str:
    """Extract text from JSDoc comment lines."""
    text_lines: list[str] = []
    for line in lines:
        stripped = line.strip().lstrip("/*").strip()
        if stripped:
            text_lines.append(stripped)
    return " ".join(text_lines)


class TypeScriptExtractor:
    def can_handle(self, doc_type: str) -> bool:
        return doc_type in {"typescript", "javascript"}

    def _scan(self, source: str) -> tuple[str, dict[tuple[str, ...], dict]]:
        """Return (markdown, section_map) where section_map maps section key → line metadata."""
        lines = source.splitlines()
        parts: list[str] = []
        section_map: dict[tuple[str, ...], dict] = {}

        import_line_indices = [idx for idx, ln in enumerate(lines) if _IMPORT_RE.match(ln.strip())]
        if import_line_indices:
            import_lines = [lines[idx] for idx in import_line_indices]
            parts.append("## Imports\n\n```typescript\n" + "\n".join(import_lines) + "\n```")
            section_map[("Imports",)] = {
                "line_start": import_line_indices[0] + 1,
                "line_end": import_line_indices[-1] + 1,
                "symbol": "Imports",
            }

        brace_depth = 0
        class_stack: list[tuple[str, int]] = []
        pending_jsdoc: list[str] = []
        in_jsdoc = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith("/**"):
                in_jsdoc = True
                pending_jsdoc = [line]
                if "*/" in stripped:
                    in_jsdoc = False
                i += 1
                continue
            if in_jsdoc:
                pending_jsdoc.append(line)
                if "*/" in stripped:
                    in_jsdoc = False
                i += 1
                continue

            m = _CLASS_RE.match(stripped)
            if m:
                name = m.group(1)
                jsdoc = _strip_jsdoc(pending_jsdoc) if pending_jsdoc else ""
                pending_jsdoc = []
                parts.append(f"# {name}")
                if jsdoc:
                    parts.append(jsdoc)
                depth_before = brace_depth
                brace_depth += line.count("{") - line.count("}")
                class_stack.append((name, depth_before))
                i += 1
                continue

            m = _INTERFACE_RE.match(stripped)
            if m:
                name = m.group(1)
                jsdoc = _strip_jsdoc(pending_jsdoc) if pending_jsdoc else ""
                pending_jsdoc = []
                parts.append(f"# {name}")
                if jsdoc:
                    parts.append(jsdoc)
                brace_depth += line.count("{") - line.count("}")
                i += 1
                continue

            m = _FUNCTION_RE.match(stripped)
            if m and not class_stack:
                name = m.group(1)
                jsdoc = _strip_jsdoc(pending_jsdoc) if pending_jsdoc else ""
                pending_jsdoc = []
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
                continue

            m = _CONST_ARROW_RE.match(stripped)
            if m and not class_stack:
                name = m.group(1)
                jsdoc = _strip_jsdoc(pending_jsdoc) if pending_jsdoc else ""
                pending_jsdoc = []
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
                continue

            if class_stack:
                class_name = class_stack[-1][0]
                class_depth = class_stack[-1][1]
                if brace_depth == class_depth + 1:
                    m = _METHOD_RE.match(line)
                    if m:
                        mname = m.group(1)
                        if mname not in {"if", "for", "while", "switch", "return", "else", "try", "catch"}:
                            jsdoc = _strip_jsdoc(pending_jsdoc) if pending_jsdoc else ""
                            pending_jsdoc = []
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
