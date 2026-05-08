# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""TypeScript/JavaScript source code extractor using regex + brace-depth tracking."""

from __future__ import annotations

import re

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
        import_lines = [ln for ln in lines if _IMPORT_RE.match(ln.strip())]
        if import_lines:
            parts.append("## Imports\n\n```typescript\n" + "\n".join(import_lines) + "\n```")

        brace_depth = 0
        # Track open class contexts: list of (class_name, brace_depth_at_open)
        class_stack: list[tuple[str, int]] = []
        pending_jsdoc: list[str] = []
        in_jsdoc = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # JSDoc accumulation
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

            # Class declaration
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

            # Interface declaration
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

            # Top-level function
            m = _FUNCTION_RE.match(stripped)
            if m and not class_stack:
                name = m.group(1)
                jsdoc = _strip_jsdoc(pending_jsdoc) if pending_jsdoc else ""
                pending_jsdoc = []
                parts.append(f"# {name}")
                if jsdoc:
                    parts.append(jsdoc)
                # Collect body
                body, i, brace_depth = _collect_body(lines, i, brace_depth)
                parts.append(f"```typescript\n{body}\n```")
                continue

            # Arrow function (const foo = ...)
            m = _CONST_ARROW_RE.match(stripped)
            if m and not class_stack:
                name = m.group(1)
                jsdoc = _strip_jsdoc(pending_jsdoc) if pending_jsdoc else ""
                pending_jsdoc = []
                parts.append(f"# {name}")
                if jsdoc:
                    parts.append(jsdoc)
                body, i, brace_depth = _collect_body(lines, i, brace_depth)
                parts.append(f"```typescript\n{body}\n```")
                continue

            # Method inside class (brace_depth == class depth + 1)
            if class_stack:
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
                            body, i, brace_depth = _collect_body(lines, i, brace_depth)
                            parts.append(f"```typescript\n{body}\n```")
                            # Check if class closed
                            if class_stack and brace_depth <= class_stack[-1][1]:
                                class_stack.pop()
                            continue

            # Update brace depth for other lines
            pending_jsdoc = []
            delta = line.count("{") - line.count("}")
            brace_depth += delta
            # Pop class stack if we closed back to its depth
            while class_stack and brace_depth <= class_stack[-1][1]:
                class_stack.pop()

            i += 1

        return "\n\n".join(parts)


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
