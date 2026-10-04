# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Markdown extractor — strips YAML frontmatter and returns clean markdown text."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chonk.models import DocumentChunk

# Matches a YAML frontmatter block at the very start of the file
_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\n---\r?\n?", re.DOTALL)


class MarkdownExtractor:
    """Extract markdown files, stripping YAML frontmatter if present."""

    HANDLED = {"markdown"}

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        return _FRONTMATTER_RE.sub("", text).lstrip("\n")

    def annotate(
        self, chunks: list[DocumentChunk], data: bytes, source_path: str | None = None
    ) -> list[DocumentChunk]:
        return chunks
