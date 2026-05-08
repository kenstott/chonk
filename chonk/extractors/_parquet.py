# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Parquet / Arrow / Feather extractor — schema summary or data table."""
from __future__ import annotations

import os


class ParquetExtractor:
    """Extract Parquet, Arrow IPC, and Feather files.

    mode="schema" (default): column names, types, row count, sample values.
    mode="data": full content as a markdown table (small files only).
    """

    HANDLED = {"parquet", "arrow", "feather"}

    def __init__(self, mode: str = "schema"):
        self._mode = mode

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in self.HANDLED

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        try:
            import pyarrow as pa
            import pyarrow.feather as feather
            import pyarrow.ipc as ipc
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for parquet/arrow/feather extraction. "
                "Install with: pip install chonk[parquet]"
            ) from exc

        ext = os.path.splitext(source_path or "")[1].lower()
        buf = pa.BufferReader(data)

        if ext == ".arrow":
            reader = ipc.open_stream(buf)
            table = reader.read_all()
        elif ext == ".feather":
            table = feather.read_table(buf)
        else:
            table = pq.read_table(buf)

        if self._mode == "data":
            return self._render_data(table)
        return self._render_schema(table)

    @staticmethod
    def _render_schema(table) -> str:
        lines = [f"Rows: {table.num_rows}", "Columns:"]
        for field in table.schema:
            lines.append(f"  {field.name}: {field.type}")
        sample_rows = min(3, table.num_rows)
        if sample_rows:
            lines.append("\nSample:")
            for i in range(sample_rows):
                row = {col: str(table.column(col)[i].as_py()) for col in table.schema.names}
                lines.append(str(row))
        return "\n".join(lines)

    @staticmethod
    def _render_data(table) -> str:
        headers = table.schema.names
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join("---" for _ in headers) + " |"
        data_rows = [
            "| " + " | ".join(
                str(table.column(col)[i].as_py()).replace("|", "\\|")
                for col in headers
            ) + " |"
            for i in range(table.num_rows)
        ]
        return "\n".join([header_row, sep_row] + data_rows)


    def annotate(self, chunks: list, data: bytes, source_path: str | None = None) -> list:
        return chunks