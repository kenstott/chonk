# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d73803cc-9c25-46b4-92e2-6325b9e52ecd
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Java source code extractor using regex + brace-depth tracking."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chonk.models import DocumentChunk

_IMPORT_RE = re.compile(r"^\s*import\s+")
_TYPE_DECL_RE = re.compile(
    r"(?:^|\s)(?:public|protected|private|abstract|final|static|strictfp|\s)*"
    r"(?:class|interface|enum|record)\s+(\w+)"
)
_METHOD_RE = re.compile(
    r"^\s+(?:(?:public|protected|private|static|final|abstract|synchronized|default|native|transient|volatile)\s+)*"
    r"(?:<[\w,\s<>?]+>\s+)?"  # optional generic return type
    r"(?:\w[\w.<>\[\],\s]*)\s+"  # return type
    r"(\w+)\s*\("  # method name + (
)


def _strip_javadoc(lines: list[str]) -> str:
    text_lines: list[str] = []
    for line in lines:
        stripped = line.strip().lstrip("/*").strip()
        if stripped:
            text_lines.append(stripped)
    return " ".join(text_lines)


_SKIP_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "else",
    "try",
    "catch",
    "finally",
    "new",
    "throw",
    "assert",
    "synchronized",
    "super",
    "this",
}


class JavaExtractor:
    def can_handle(self, doc_type: str) -> bool:
        return doc_type in {"java"}

    def _scan(self, source: str) -> tuple[str, dict[tuple[str, ...], dict]]:
        """Return (markdown, section_map) where section_map maps section key → line metadata."""
        lines = source.splitlines()
        parts: list[str] = []
        section_map: dict[tuple[str, ...], dict] = {}

        import_line_indices = [idx for idx, ln in enumerate(lines) if _IMPORT_RE.match(ln)]
        if import_line_indices:
            import_lines = [lines[idx] for idx in import_line_indices]
            parts.append("## Imports\n\n```java\n" + "\n".join(import_lines) + "\n```")
            section_map[("Imports",)] = {
                "line_start": import_line_indices[0] + 1,
                "line_end": import_line_indices[-1] + 1,
                "symbol": "Imports",
            }

        brace_depth = 0
        type_stack: list[tuple[str, int]] = []
        pending_javadoc: list[str] = []
        in_javadoc = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith("/**"):
                in_javadoc = True
                pending_javadoc = [line]
                if "*/" in stripped:
                    in_javadoc = False
                i += 1
                continue
            if in_javadoc:
                pending_javadoc.append(line)
                if "*/" in stripped:
                    in_javadoc = False
                i += 1
                continue

            if _IMPORT_RE.match(line):
                pending_javadoc = []
                i += 1
                continue

            if brace_depth <= 1:
                m = _TYPE_DECL_RE.search(line)
                if m:
                    name = m.group(1)
                    javadoc = _strip_javadoc(pending_javadoc) if pending_javadoc else ""
                    pending_javadoc = []
                    parts.append(f"# {name}")
                    if javadoc:
                        parts.append(javadoc)
                    depth_before = brace_depth
                    brace_depth += line.count("{") - line.count("}")
                    type_stack.append((name, depth_before))
                    i += 1
                    continue

            if type_stack:
                class_name = type_stack[-1][0]
                class_depth = type_stack[-1][1]
                if brace_depth == class_depth + 1:
                    m = _METHOD_RE.match(line)
                    if m:
                        mname = m.group(1)
                        if mname not in _SKIP_KEYWORDS:
                            javadoc = _strip_javadoc(pending_javadoc) if pending_javadoc else ""
                            pending_javadoc = []
                            parts.append(f"## {mname}")
                            if javadoc:
                                parts.append(javadoc)
                            line_start = i + 1
                            body, i, brace_depth = _collect_java_body(lines, i, brace_depth)
                            section_map[(class_name, mname)] = {
                                "line_start": line_start,
                                "line_end": i,
                                "symbol": f"{class_name}.{mname}",
                            }
                            parts.append(f"```java\n{body}\n```")
                            while type_stack and brace_depth <= type_stack[-1][1]:
                                type_stack.pop()
                            continue

            pending_javadoc = []
            brace_depth += line.count("{") - line.count("}")
            while type_stack and brace_depth <= type_stack[-1][1]:
                type_stack.pop()
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


def _collect_java_body(lines: list[str], start: int, current_depth: int) -> tuple[str, int, int]:
    """Collect lines from start until the method's matching close brace."""
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
