# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Code indexing example — Python AST chunking + ImportCrawler.

Demonstrates:
- Loading Python source files with AST-based chunking (class/method granularity)
- Using ImportCrawler to follow imports transitively from an entry point
- source_detail: line numbers and symbol names on every chunk
"""

import pathlib

from chonk import DocumentLoader
from chonk.transports import ImportCrawler

if __name__ == "__main__":
    # --- Single file ---------------------------------------------------------
    loader = DocumentLoader()

    # Load one Python file directly — chunks are class/method boundaries
    sample = pathlib.Path(__file__).parent.parent / "chonk" / "models.py"
    chunks = loader.load(str(sample))

    print(f"\n=== {sample.name} — {len(chunks)} chunk(s) ===")
    for chunk in chunks:
        detail = chunk.source_detail or {}
        symbol = detail.get("symbol", "")
        line_start = detail.get("line_start", "?")
        line_end = detail.get("line_end", "?")
        print(f"  [{line_start}–{line_end}] {symbol or chunk.section}")

    # --- ImportCrawler — transitive import graph -----------------------------
    # Crawl all Python modules reachable from chonk/loader.py, up to depth 2.
    root = pathlib.Path(__file__).parent.parent
    crawler = ImportCrawler(root_path=str(root), max_depth=2)

    loader2 = DocumentLoader(extra_transports=[crawler])
    entry = str(root / "chonk" / "loader.py")

    all_chunks = loader2.load(entry)
    files_seen = {c.document_name for c in all_chunks}

    print("\n=== ImportCrawler from chonk/loader.py (depth 2) ===")
    print(f"Files crawled : {len(files_seen)}")
    print(f"Total chunks  : {len(all_chunks)}")

    # Show first 5 chunks with line-number detail
    for chunk in all_chunks[:5]:
        detail = chunk.source_detail or {}
        print(
            f"  {chunk.document_name}:{detail.get('line_start', '?')}–"
            f"{detail.get('line_end', '?')}  {detail.get('symbol', '')}"
        )
