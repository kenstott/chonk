# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 977eefa1-3279-4be9-b5b5-c606a823ecf7
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Lookup table extractor — one markdown section per row for dense entity retrieval."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentChunk


class LookupTableExtractor:
    """Extract tabular data as one ``##``-headed section per row.

    Each data row becomes::

        ## {key_value}
        col_a: val_a
        col_b: val_b
        ...

    Rows are separated by double newlines so ``chunk_document`` flushes at each
    heading, producing one chunk per entity.  Ideal for entity resolution lookup
    tables (CIK registry, GLEIF, ticker → aliases) where per-entity chunks
    maximise retrieval recall without runtime entity expansion.

    Args:
        key_column: Column name (str) or 0-based index (int) to use as the
            ``##`` heading.  Defaults to 0 (first column).
        skip_empty_key: Drop rows where the key column is blank (default True).
    """

    HANDLED = {"lookup_table", "csv_rows"}

    def __init__(
        self,
        key_column: str | int = 0,
        skip_empty_key: bool = True,
    ) -> None:
        self.key_column = key_column
        self.skip_empty_key = skip_empty_key

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t|;")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(io.StringIO(text), dialect)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if len(rows) < 2:
            return ""

        headers = [h.strip() for h in rows[0]]

        if isinstance(self.key_column, int):
            key_idx = self.key_column
        else:
            key_name = self.key_column.lower()
            key_idx = next(
                (i for i, h in enumerate(headers) if h.lower() == key_name),
                0,
            )

        sections: list[str] = []
        for row in rows[1:]:
            while len(row) < len(headers):
                row.append("")

            key_val = row[key_idx].strip()
            if not key_val and self.skip_empty_key:
                continue

            lines = [f"## {key_val}"]
            for i, (header, value) in enumerate(zip(headers, row)):
                if i == key_idx:
                    continue
                v = value.strip()
                if v:
                    lines.append(f"{header}: {v}")

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        data: bytes,
        source_path: str | None = None,
    ) -> list[DocumentChunk]:
        return chunks
