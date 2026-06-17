# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4ad05fa2-5f30-41e3-b3ed-9ca4ec3e6a84
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""HTML extractor — converts HTML to Markdown using stdlib html.parser."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chonk.models import DocumentChunk


def _strip_html_chrome(html: str) -> str:
    """Pre-strip navigation chrome tags from raw HTML.

    Removes <nav>, <aside>, <noscript>, <script>, <style> and elements
    with navigation-related CSS classes/IDs. Uses regex on raw HTML
    since html.parser can't handle unclosed tags reliably.
    """
    for tag in ("nav", "aside", "header", "footer", "noscript", "script", "style"):
        html = re.sub(
            rf"<{tag}[\s>].*?</{tag}>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
    _NAV_ATTR_RE = re.compile(
        r"<(div|section|ul|table)[^>]*?"
        r"(?:class|id|role)\s*=\s*[\"'][^\"']*?"
        r"(?:sidebar|navbox|navbar|navigation|toc\b|catlinks"
        r"|mw-panel|mw-head|mw-editsection"
        r"|menu|breadcrumb|noprint"
        r"|portal|sister-?project|interlanguage|authority-control"
        r"|reflist|references|footnotes|mw-references-wrap|citation)"
        r"[^\"']*?[\"'][^>]*>.*?</\1>",
        re.DOTALL | re.IGNORECASE,
    )
    html = _NAV_ATTR_RE.sub("", html)
    return re.sub(
        r"<sup[^>]*class=[\"'][^\"']*reference[^\"']*[\"'][^>]*>.*?</sup>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )


class _MarkdownConverter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._output: list[str] = []
        self._tag_stack: list[str] = []
        self._list_stack: list[str] = []  # "ul" or "ol"
        self._ol_counters: list[int] = []
        self._in_pre = False
        self._href: str | None = None
        self._link_text: list[str] = []
        self._in_link = False
        self._in_cell = False
        self._cell_buf: list[str] = []
        self._current_row_cells: list[str] = []
        self._last_row_col_count: int = 0

    def _append(self, text: str) -> None:
        if self._in_cell:
            self._cell_buf.append(text)
        else:
            self._output.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._output.append("\n\n")
        elif tag == "p":
            if not self._in_cell:
                self._output.append("\n\n")
        elif tag == "br":
            if self._in_cell:
                self._cell_buf.append(" ")
            else:
                self._output.append("\n")
        elif tag == "pre":
            self._in_pre = True
            self._output.append("\n\n```\n")
        elif tag == "ul":
            self._list_stack.append("ul")
        elif tag == "ol":
            self._list_stack.append("ol")
            self._ol_counters.append(0)
        elif tag == "li":
            indent = "  " * (len(self._list_stack) - 1)
            if self._list_stack and self._list_stack[-1] == "ol":
                self._ol_counters[-1] += 1
                self._output.append(f"\n{indent}{self._ol_counters[-1]}. ")
            else:
                self._output.append(f"\n{indent}- ")
        elif tag in ("strong", "b"):
            self._append("**")
        elif tag in ("em", "i"):
            self._append("*")
        elif tag == "a":
            attr_dict = dict(attrs)
            self._href = attr_dict.get("href")
            self._in_link = True
            self._link_text = []
        elif tag == "tr":
            self._current_row_cells = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_buf = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            prefix = "#" * level + " "
            text_parts: list[str] = []
            while self._output and self._output[-1] != "\n\n":
                text_parts.append(self._output.pop())
            text = "".join(reversed(text_parts)).strip()
            self._output.append(f"{prefix}{text}\n\n")
        elif tag == "p":
            if not self._in_cell:
                self._output.append("\n")
        elif tag == "pre":
            self._in_pre = False
            self._output.append("\n```\n\n")
        elif tag == "ul":
            if self._list_stack:
                self._list_stack.pop()
            self._output.append("\n")
        elif tag == "ol":
            if self._list_stack:
                self._list_stack.pop()
            if self._ol_counters:
                self._ol_counters.pop()
            self._output.append("\n")
        elif tag in ("strong", "b"):
            self._append("**")
        elif tag in ("em", "i"):
            self._append("*")
        elif tag == "a":
            link_text = "".join(self._link_text).strip()
            if self._href and link_text:
                formatted = f"[{link_text}]({self._href})"
            else:
                formatted = link_text
            self._append(formatted)
            self._in_link = False
            self._href = None
            self._link_text = []
        elif tag in ("td", "th"):
            cell_text = " ".join("".join(self._cell_buf).split())
            self._current_row_cells.append(cell_text)
            self._cell_buf = []
            self._in_cell = False
        elif tag == "tr":
            if self._current_row_cells:
                row = "| " + " | ".join(self._current_row_cells) + " |"
                self._output.append(f"\n{row}")
                self._last_row_col_count = len(self._current_row_cells)
        elif tag == "thead":
            if self._last_row_col_count:
                sep = "| " + " | ".join(["---"] * self._last_row_col_count) + " |"
                self._output.append(f"\n{sep}")

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._link_text.append(data)
            return
        if not self._in_pre and "\n" in data and not data.strip():
            return
        self._append(data)

    def get_markdown(self) -> str:
        text = "".join(self._output)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _convert_html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown, preserving heading structure.

    Uses stdlib html.parser — no external dependencies.
    Handles: headings, paragraphs, lists (ul/ol/li), <br>, <pre>/<code>,
    bold, italic, links, and tables.
    Strips navigation chrome (nav, aside, sidebar, navbox, etc.).
    """
    html = _strip_html_chrome(html)
    converter = _MarkdownConverter()
    converter.feed(html)
    return converter.get_markdown()


class _HeadingScanner(HTMLParser):
    """Collects heading level, id-attribute, and text from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[int, str | None, str]] = []
        self._level: int | None = None
        self._anchor: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self._level = int(tag[1])
            self._anchor = dict(attrs).get("id")
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit() and self._level is not None:
            text = "".join(self._buf).strip()
            if text:
                self.records.append((self._level, self._anchor, text))
            self._level = None
            self._anchor = None
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._level is not None:
            self._buf.append(data)


class HtmlExtractor:
    """Extract plain text (as Markdown) from HTML documents."""

    def can_handle(self, doc_type: str) -> bool:
        return doc_type in ("html", "htm")

    def extract(self, data: bytes, source_path: str | None = None) -> str:
        text = data.decode("utf-8", errors="replace")
        return _convert_html_to_markdown(text)

    def annotate(
        self, chunks: list[DocumentChunk], data: bytes, source_path: str | None = None
    ) -> list[DocumentChunk]:
        html = data.decode("utf-8", errors="replace")

        scanner = _HeadingScanner()
        scanner.feed(html)
        if not scanner.records:
            return chunks

        # Build (anchor, heading_path) for each heading in document order
        heading_stack: list[tuple[int, str | None, str]] = []
        section_anchors: list[tuple[str | None, list[str]]] = []
        for level, anchor, text in scanner.records:
            heading_stack = [(line, a, t) for line, a, t in heading_stack if line < level]
            heading_stack.append((level, anchor, text))
            path = [t for _, _, t in heading_stack]
            top_anchor = next((a for _, a, _ in reversed(heading_stack) if a), None)
            section_anchors.append((top_anchor, path))

        # Convert to markdown and split into segments keyed by heading
        markdown = _convert_html_to_markdown(html)
        heading_idx = 0
        segments: list[tuple[str | None, list[str], str]] = []
        current_anchor: str | None = None
        current_path: list[str] = []
        current_buf: list[str] = []

        for line in markdown.split("\n"):
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m and heading_idx < len(section_anchors):
                if current_buf:
                    segments.append((current_anchor, current_path, "\n".join(current_buf)))
                current_anchor, current_path = section_anchors[heading_idx]
                heading_idx += 1
                current_buf = [line]
            else:
                current_buf.append(line)
        if current_buf:
            segments.append((current_anchor, current_path, "\n".join(current_buf)))

        for chunk in chunks:
            content = chunk.content
            for anchor, path, seg_text in segments:
                if any(frag in seg_text for frag in content.split("\n") if len(frag.strip()) > 20):
                    detail: dict[str, object] = {"heading_path": path}
                    if anchor:
                        detail["anchor"] = anchor
                    chunk.source_detail = detail
                    break

        return chunks
