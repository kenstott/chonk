# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 3b533f58-6d20-4608-a2a3-0cd1957b1d38
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""YAML extractor — emits markdown with key-path breadcrumb headings.

Supports single and multi-document YAML files (documents separated by ---).
Requires: pyyaml>=6.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._json import _walk

if TYPE_CHECKING:
    from chonk.models import DocumentChunk


class YamlExtractor:
    """Extract YAML / multi-document YAML into markdown with hierarchical headings."""

    HANDLED = {"yaml"}

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for YAML extraction. Install it with: pip install pyyaml"
            ) from exc

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        try:
            docs = list(yaml.safe_load_all(text))
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML parse error in {source_path or '<unknown>'}: {exc}") from exc

        doc_blocks: list[str] = []
        for doc in docs:
            if doc is None:
                continue
            lines: list[str] = []
            _walk(doc, lines, depth=1, path="")
            if lines:
                doc_blocks.append("\n\n".join(lines))

        return "\n\n---\n\n".join(doc_blocks)

    def annotate(
        self, chunks: list[DocumentChunk], data: bytes, source_path: str | None = None
    ) -> list[DocumentChunk]:
        return chunks
