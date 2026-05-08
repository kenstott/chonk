# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: a1b2c3d4-e5f6-7890-abcd-ef0123456789
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Renderer protocol — domain-specific object-to-markdown converters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..models import DocumentChunk


@runtime_checkable
class Renderer(Protocol):
    """Converts a parsed object (dict or list) to markdown and annotates chunks.

    Plugs into ``JsonExtractor`` and ``XmlExtractor`` to replace the generic
    key-path walk with domain-aware rendering for known formats (CVE, STIX,
    FHIR, CycloneDX, etc.).

    Implement all three methods::

        class MyRenderer:
            def can_render(self, source_path, obj):
                return isinstance(obj, dict) and "my_key" in obj

            def render(self, obj):
                return "# Title\\n\\n..."

            def annotate(self, chunks, obj):
                for chunk in chunks:
                    chunk.source_detail = {"format": "myformat"}
                return chunks
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:
        """Return True if this renderer handles the given parsed object."""
        ...

    def render(self, obj: object) -> str:
        """Convert the parsed object to a markdown string for chunking."""
        ...

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        """Stamp ``source_detail`` onto chunks after chunking."""
        ...
