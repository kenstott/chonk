# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: fca21083-b8ee-47ea-9eb2-b18865b66266
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CSV extractor — renders tabular data as a markdown table."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentChunk


class CsvExtractor:
    """Extract CSV / TSV files into a markdown table.

    Each CSV becomes one markdown table.  Column headers are used as the
    header row; if the file has no header the columns are numbered (col_1, …).
    """

    HANDLED = {"csv", "tsv"}

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        # Sniff dialect; fall back to excel (standard CSV)
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t|;")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(io.StringIO(text), dialect)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            return ""

        headers = rows[0]
        body = rows[1:]

        def _cell(value: str) -> str:
            return value.replace("|", "\\|").replace("\n", " ").strip()

        header_row = "| " + " | ".join(_cell(h) for h in headers) + " |"
        sep_row = "| " + " | ".join("---" for _ in headers) + " |"
        data_rows = ["| " + " | ".join(_cell(c) for c in row) + " |" for row in body]

        return "\n".join([header_row, sep_row] + data_rows)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        data: bytes,
        source_path: str | None = None,
    ) -> list[DocumentChunk]:
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
            return chunks

        def _cell(value: str) -> str:
            return value.replace("|", "\\|").replace("\n", " ").strip()

        # Map 1-based data row index → rendered markdown row string
        row_map: list[tuple[int, str]] = []
        for data_row_idx, row in enumerate(rows[1:], start=1):
            row_map.append((data_row_idx, "| " + " | ".join(_cell(c) for c in row) + " |"))

        for chunk in chunks:
            content = chunk.content
            matched: list[int] = [idx for idx, rendered in row_map if rendered in content]
            if matched:
                chunk.source_detail = {"row_start": min(matched), "row_end": max(matched)}

        return chunks
