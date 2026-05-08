# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Java source code extractor using regex + brace-depth tracking."""

from __future__ import annotations

import re

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


class JavaExtractor:
    def can_handle(self, doc_type: str) -> bool:
        return doc_type in {"java"}

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        if not data:
            return ""

        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"UTF-8 decode failed in {source_path or '<unknown>'}: {exc}") from exc

        lines = source.splitlines()
        parts: list[str] = []

        # Collect import lines
        import_lines = [ln for ln in lines if _IMPORT_RE.match(ln)]
        if import_lines:
            parts.append("## Imports\n\n```java\n" + "\n".join(import_lines) + "\n```")

        brace_depth = 0
        # Stack of (type_name, depth_before_open)
        type_stack: list[tuple[str, int]] = []
        pending_javadoc: list[str] = []
        in_javadoc = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Javadoc accumulation
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

            # Skip import lines (already collected)
            if _IMPORT_RE.match(line):
                pending_javadoc = []
                i += 1
                continue

            # Top-level type declaration (class, interface, enum, record)
            # Only at depth 0 or depth 1 (inner classes)
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

            # Method detection inside a type body
            if type_stack:
                class_depth = type_stack[-1][1]
                if brace_depth == class_depth + 1:
                    m = _METHOD_RE.match(line)
                    if m:
                        mname = m.group(1)
                        skip_keywords = {
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
                        if mname not in skip_keywords:
                            javadoc = _strip_javadoc(pending_javadoc) if pending_javadoc else ""
                            pending_javadoc = []
                            parts.append(f"## {mname}")
                            if javadoc:
                                parts.append(javadoc)
                            body, i, brace_depth = _collect_java_body(lines, i, brace_depth)
                            parts.append(f"```java\n{body}\n```")
                            while type_stack and brace_depth <= type_stack[-1][1]:
                                type_stack.pop()
                            continue

            pending_javadoc = []
            brace_depth += line.count("{") - line.count("}")
            while type_stack and brace_depth <= type_stack[-1][1]:
                type_stack.pop()
            i += 1

        return "\n\n".join(parts)


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
