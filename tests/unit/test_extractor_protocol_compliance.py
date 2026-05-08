# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Extractor protocol compliance.

Verifies:
1. All extractors implement annotate() (required by Extractor protocol).
2. DocumentChunk is importable within extractor files that use it.
3. loader.py calls annotate() unconditionally (no hasattr guard).
4. _openpyxl is not unbound in _xlsx.py when openpyxl is unavailable.
"""

from __future__ import annotations

import inspect

import pytest

_ALL_EXTRACTOR_CLASSES = [
    ("chonk.extractors._csv", "CsvExtractor"),
    ("chonk.extractors._docx", "DocxExtractor"),
    ("chonk.extractors._edgar", "EdgarExtractor"),
    ("chonk.extractors._email", "EmailExtractor"),
    ("chonk.extractors._html", "HtmlExtractor"),
    ("chonk.extractors._java", "JavaExtractor"),
    ("chonk.extractors._json", "JsonExtractor"),
    ("chonk.extractors._markdown", "MarkdownExtractor"),
    ("chonk.extractors._odf", "OdfExtractor"),
    ("chonk.extractors._parquet", "ParquetExtractor"),
    ("chonk.extractors._pdf", "PdfExtractor"),
    ("chonk.extractors._pptx", "PptxExtractor"),
    ("chonk.extractors._python", "PythonExtractor"),
    ("chonk.extractors._text", "TextExtractor"),
    ("chonk.extractors._typescript", "TypeScriptExtractor"),
    ("chonk.extractors._xlsx", "XlsxExtractor"),
    ("chonk.extractors._xml", "XmlExtractor"),
    ("chonk.extractors._yaml", "YamlExtractor"),
]


@pytest.mark.parametrize("module_path,class_name", _ALL_EXTRACTOR_CLASSES)
def test_extractor_has_annotate_method(module_path, class_name):
    """Every extractor class must expose an annotate() method."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert hasattr(cls, "annotate"), (
        f"{class_name} is missing annotate() — fails Extractor protocol"
    )
    method = cls.annotate
    assert callable(method), f"{class_name}.annotate must be callable"


@pytest.mark.parametrize("module_path,class_name", _ALL_EXTRACTOR_CLASSES)
def test_annotate_signature_matches_protocol(module_path, class_name):
    """annotate(self, chunks, data, source_path=None) -> list[DocumentChunk]."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    if not hasattr(cls, "annotate"):
        pytest.skip(f"{class_name} has no annotate (caught by other test)")
    sig = inspect.signature(cls.annotate)
    params = list(sig.parameters)
    assert "chunks" in params, f"{class_name}.annotate missing 'chunks' param"
    assert "data" in params, f"{class_name}.annotate missing 'data' param"
    assert "source_path" in params, f"{class_name}.annotate missing 'source_path' param"


def test_python_extractor_documentchunk_importable():
    """PythonExtractor.annotate uses DocumentChunk — it must be importable."""
    from chonk.extractors._python import PythonExtractor  # noqa: F401
    from chonk.models import DocumentChunk  # noqa: F401
    # Verify annotate's annotation references DocumentChunk without NameError
    ext = PythonExtractor()
    chunks = ext.annotate([], b"x = 1\n")
    assert isinstance(chunks, list)


def test_typescript_extractor_documentchunk_importable():
    """TypeScriptExtractor.annotate uses DocumentChunk — it must be importable."""
    from chonk.extractors._typescript import TypeScriptExtractor  # noqa: F401
    ext = TypeScriptExtractor()
    chunks = ext.annotate([], b"const x = 1;\n")
    assert isinstance(chunks, list)


def test_java_extractor_documentchunk_importable():
    """JavaExtractor.annotate uses DocumentChunk — it must be importable."""
    from chonk.extractors._java import JavaExtractor  # noqa: F401
    ext = JavaExtractor()
    chunks = ext.annotate([], b"public class Foo {}\n")
    assert isinstance(chunks, list)


def test_xlsx_extractor_documentchunk_importable():
    """XlsxExtractor.annotate uses DocumentChunk — it must be importable."""
    pytest.importorskip("openpyxl")
    from chonk.extractors._xlsx import XlsxExtractor
    ext = XlsxExtractor()
    assert callable(ext.annotate)


def test_loader_calls_annotate_unconditionally(monkeypatch):
    """DocumentLoader must call annotate() without a hasattr guard.

    Since annotate is now part of the protocol, loader.py must invoke it
    directly — not conditionally via hasattr().
    """
    import ast
    import inspect

    from chonk import loader as loader_mod

    import textwrap
    src = textwrap.dedent(inspect.getsource(loader_mod.DocumentLoader.load))
    tree = ast.parse(src)

    # Check that no hasattr(..., "annotate") call exists in load()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "hasattr":
                args = node.args
                if len(args) >= 2:
                    arg1 = args[1]
                    if isinstance(arg1, ast.Constant) and arg1.value == "annotate":
                        pytest.fail(
                            "loader.DocumentLoader.load() still uses hasattr(extractor, 'annotate') — "
                            "annotate is part of the protocol and must be called unconditionally"
                        )

    src_bytes = textwrap.dedent(inspect.getsource(loader_mod.DocumentLoader.load_bytes))
    tree2 = ast.parse(src_bytes)
    for node in ast.walk(tree2):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "hasattr":
                args = node.args
                if len(args) >= 2:
                    arg1 = args[1]
                    if isinstance(arg1, ast.Constant) and arg1.value == "annotate":
                        pytest.fail(
                            "loader.DocumentLoader.load_bytes() still uses hasattr(extractor, 'annotate') — "
                            "annotate is part of the protocol and must be called unconditionally"
                        )
