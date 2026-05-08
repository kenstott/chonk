# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 9907fc1c-ada1-42a0-9fee-a1712bbb8d9e
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Document format extractors."""

from ._csv import CsvExtractor
from ._docx import DocxExtractor
from ._edgar import EdgarExtractor
from ._email import EmailExtractor
from ._html import HtmlExtractor
from ._java import JavaExtractor
from ._json import JsonExtractor
from ._markdown import MarkdownExtractor
from ._mime import detect_type_from_source, is_binary_type, normalize_type
from ._odf import OdfExtractor
from ._parquet import ParquetExtractor
from ._pdf import PdfExtractor
from ._pptx import PptxExtractor
from ._protocol import Extractor
from ._python import PythonExtractor
from ._text import TextExtractor
from ._typescript import TypeScriptExtractor
from ._xlsx import XlsxExtractor
from ._xml import XmlExtractor
from ._yaml import YamlExtractor

_REGISTRY: list[Extractor] | None = None


def _build_registry() -> list[Extractor]:
    return [
        EdgarExtractor(),
        PdfExtractor(),
        DocxExtractor(),
        XlsxExtractor(),
        PptxExtractor(),
        HtmlExtractor(),
        JsonExtractor(),
        YamlExtractor(),
        MarkdownExtractor(),
        CsvExtractor(),
        ParquetExtractor(),
        OdfExtractor(),
        EmailExtractor(),
        XmlExtractor(),
        PythonExtractor(),
        TypeScriptExtractor(),
        JavaExtractor(),
        TextExtractor(),
    ]


def register_extractor(extractor: Extractor, *, prepend: bool = True) -> None:
    """Add a custom extractor to the global registry.

    Args:
        extractor: An object satisfying the Extractor protocol.
        prepend:   If True (default), insert before built-in extractors so the
                   custom extractor takes priority over the defaults.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    if prepend:
        _REGISTRY.insert(0, extractor)
    else:
        _REGISTRY.append(extractor)


def detect_extractor(doc_type: str) -> Extractor:
    """Return the first registered extractor that can handle doc_type."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    for ext in _REGISTRY:
        if ext.can_handle(doc_type):
            return ext
    raise ValueError(f"No extractor found for doc_type={doc_type!r}")


__all__ = [
    "Extractor",
    "TextExtractor",
    "HtmlExtractor",
    "PdfExtractor",
    "DocxExtractor",
    "XlsxExtractor",
    "PptxExtractor",
    "EdgarExtractor",
    "JsonExtractor",
    "YamlExtractor",
    "MarkdownExtractor",
    "CsvExtractor",
    "ParquetExtractor",
    "OdfExtractor",
    "EmailExtractor",
    "XmlExtractor",
    "PythonExtractor",
    "TypeScriptExtractor",
    "JavaExtractor",
    "detect_extractor",
    "register_extractor",
    "normalize_type",
    "detect_type_from_source",
    "is_binary_type",
]
