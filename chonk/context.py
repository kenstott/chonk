# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 79a01dcf-aad6-4086-a01a-4b3465b18e71
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Contextual enrichment — prepend document name and section path into embedding_content.

The thesis: encoding document name and section breadcrumbs into embedding_content
before embedding improves retrieval relevance, clustering, and NER vs embedding
raw content alone.

WHY DOCUMENT NAME:
  Section headings are often generic even in well-structured documents:
  "1. Definitions", "8. Limitation of Liability", "9. Indemnification" — these
  appear identically across every SaaS contract. The filename ("techcorp_msa",
  "cloudsolutions_agreement") is the primary disambiguator.

  Similarly, spreadsheet sheets named "Data" or "Q1", slides titled "Overview",
  and wiki pages with repeated section structures all depend on the document
  name for context that the heading alone cannot provide.

WHY SECTION PATH:
  Within a single document, repeated leaf headings ("Parameters", "Returns",
  "Notes", "Headcount") need the parent path to be meaningful. A chunk from
  "APAC > Engineering > Headcount" is indistinguishable from "EMEA > Engineering
  > Headcount" on content alone if the region name only appears in the ancestor
  heading.

TOGETHER:
  embedding_content = "[doc_name > Ancestor > Section]\n\n{content}"
  gives every chunk a unique, human-readable address that survives any split
  boundary.
"""

from __future__ import annotations

import dataclasses

from .models import DocumentChunk


def enrich_chunk(chunk: DocumentChunk) -> DocumentChunk:
    """Return a new DocumentChunk with embedding_content set.

    Never mutates the input chunk.

    The generated embedding_content is::

        [doc_name > Ancestor > Section]

        <content>

    The breadcrumb is taken from ``chunk.breadcrumb`` when present (set by
    ``chunk_document`` and respecting ``include_doc_name`` and
    ``max_breadcrumb_chars``).  When ``breadcrumb`` is absent the function
    rebuilds it from ``chunk.document_name`` and ``chunk.section``.

    If no breadcrumb can be constructed (no document name, no section path),
    ``embedding_content`` is set to ``chunk.content`` unchanged.
    """
    crumb = chunk.breadcrumb
    if not crumb:
        section_parts = (
            chunk.section
            if isinstance(chunk.section, list)
            else ([chunk.section] if chunk.section else [])
        )
        parts = [chunk.document_name] + section_parts if chunk.document_name else section_parts
        crumb = f"[{' > '.join(parts)}]" if parts else None

    if crumb:
        embedding_content = f"{crumb}\n\n{chunk.content}"
    else:
        embedding_content = chunk.content

    return dataclasses.replace(chunk, embedding_content=embedding_content)


def enrich_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Return a new list of DocumentChunks with embedding_content set on each."""
    return [enrich_chunk(chunk) for chunk in chunks]
