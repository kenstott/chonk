# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 51839c24-977c-4bc0-af66-411d9c701d3e
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Plain-text and text-like format extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chonk.models import DocumentChunk


class TextExtractor:
    """Extract content from text-based formats by decoding bytes."""

    HANDLED = {"text", "csv", "txt"}

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")

    def annotate(
        self, chunks: list[DocumentChunk], data: bytes, source_path: str | None = None
    ) -> list[DocumentChunk]:
        return chunks
