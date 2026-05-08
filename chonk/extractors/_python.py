# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Python source code extractor using stdlib ast."""

from __future__ import annotations

import ast


class PythonExtractor:
    def can_handle(self, doc_type: str) -> bool:
        return doc_type in {"python"}

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        if not data:
            return ""

        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"UTF-8 decode failed in {source_path or '<unknown>'}: {exc}") from exc

        try:
            tree = ast.parse(source, filename=source_path or "<unknown>")
        except SyntaxError as exc:
            raise ValueError(f"Python syntax error in {source_path or '<unknown>'}: {exc}") from exc

        parts: list[str] = []

        # Collect top-level imports
        import_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any(node is top for top in ast.iter_child_nodes(tree))
        ]
        if import_nodes:
            segments: list[str] = []
            for node in import_nodes:
                seg = ast.get_source_segment(source, node)
                if seg:
                    segments.append(seg)
            if segments:
                parts.append("## Imports\n\n```python\n" + "\n".join(segments) + "\n```")

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                parts.append(f"# {node.name}")
                docstring = ast.get_docstring(node)
                if docstring:
                    parts.append(docstring)
                methods = [
                    item
                    for item in ast.iter_child_nodes(node)
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if methods:
                    for item in methods:
                        parts.append(f"## {item.name}")
                        method_doc = ast.get_docstring(item)
                        if method_doc:
                            parts.append(method_doc)
                        seg = ast.get_source_segment(source, item)
                        if seg:
                            parts.append(f"```python\n{seg}\n```")
                else:
                    seg = ast.get_source_segment(source, node)
                    if seg:
                        parts.append(f"```python\n{seg}\n```")

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts.append(f"# {node.name}")
                docstring = ast.get_docstring(node)
                if docstring:
                    parts.append(docstring)
                seg = ast.get_source_segment(source, node)
                if seg:
                    parts.append(f"```python\n{seg}\n```")

        return "\n\n".join(parts)
