# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 6bdda530-9e8a-4fc9-9c12-941e3197beca
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DocumentLoader — orchestrates Transport → Extractor → chunk_document → enrich_chunks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .models import DocumentChunk
from .schema import ColumnMeta, TableMeta, FieldMeta, EndpointMeta
from ._struct_inference import infer_csv, infer_json, infer_jsonl, infer_parquet

_STRUCTURED_EXTENSIONS = frozenset({".parquet", ".arrow", ".feather", ".csv", ".jsonl", ".ndjson"})
from .chunking import chunk_document
from .extractors import detect_extractor, detect_type_from_source, normalize_type
from .transports import (
    detect_transport,
    FetchResult,
    LocalTransport,
    HttpTransport,
    S3Transport,
    FtpTransport,
    SftpTransport,
    WebCrawler,
    DirectoryCrawler,
    Crawler,
)

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Full pipeline: fetch bytes → extract text → chunk → contextual enrichment.

    Usage::

        loader = DocumentLoader(chunk_size=1500)
        chunks = loader.load("/path/to/document.pdf")
        chunks = loader.load("https://example.com/doc.html", name="example")
        chunks = loader.load_text("raw text here", name="my-doc")
    """

    def __init__(
        self,
        min_chunk_size: int = 600,
        max_chunk_size: int = 1500,
        overflow_margin: float = 0.15,
        context_strategy: str | None = "prefix",
        include_doc_name: bool = True,
        extra_transports: list | None = None,
        extra_extractors: list | None = None,
    ):
        """Args:
            min_chunk_size: Accumulation floor — accumulate across sections until
                this size is reached (default 600).
            max_chunk_size: Hard ceiling — blocks exceeding this + overflow_margin
                are split at natural boundaries (default 1500).
            overflow_margin: Fractional slack above max_chunk_size before a split
                is forced (default 0.15 = 15%).
            context_strategy: "prefix" (default) or "inline" embeds an LCA
                breadcrumb at the start of every chunk's content and sets
                embedding_content = content.  None produces naive chunks with no
                breadcrumb and embedding_content = None.
            include_doc_name: Include the document name as the first breadcrumb
                element (default True).  Set False when all chunks share a single
                corpus document name that adds no signal (e.g. one "Medical" doc
                containing many unrelated articles).
            extra_transports: Additional transport backends checked before defaults.
            extra_extractors: Additional extractor backends checked before defaults.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overflow_margin = overflow_margin
        self.context_strategy = context_strategy
        self.include_doc_name = include_doc_name
        self._extra_transports = extra_transports or []
        self._extra_extractors = extra_extractors or []

        self._transport_registry = self._extra_transports + [
            LocalTransport(),
            HttpTransport(),
            S3Transport(),
            FtpTransport(),
            SftpTransport(),
        ]

    def _find_transport(self, uri: str):
        for transport in self._transport_registry:
            if transport.can_handle(uri):
                return transport
        raise ValueError(f"No transport found for URI: {uri!r}")

    def _find_extractor(self, doc_type: str):
        for extractor in self._extra_extractors:
            if extractor.can_handle(doc_type):
                return extractor
        return detect_extractor(doc_type)

    def _find_extractor_raw(self, raw_doc_type: str):
        """Try extra extractors first with the raw (un-normalised) type string.

        Falls back to normalising via normalize_type and then detect_extractor.
        This allows custom extractors to register arbitrary doc_type strings
        (e.g. "csv-summary", "confluence-wiki") without having to add them to
        the MIME registry.
        """
        for extractor in self._extra_extractors:
            if extractor.can_handle(raw_doc_type):
                return extractor
        resolved = normalize_type(raw_doc_type)
        return self._find_extractor(resolved)

    def _enrich(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        if self.context_strategy is None:
            return chunks
        from .context import enrich_chunks
        return enrich_chunks(chunks, strategy=self.context_strategy)

    def load(self, uri: str, name: str | None = None) -> list[DocumentChunk]:
        """Fetch + extract + chunk + enrich.

        For .parquet, .arrow, .feather, .csv, .jsonl, and .ndjson files this
        delegates to load_structured_file() and returns N+1 schema chunks.

        Args:
            uri: File path, file:// URL, http(s):// URL, s3://, ftp://, or sftp://.
            name: Document name for chunk metadata. Defaults to the last path segment.

        Returns:
            List of DocumentChunk objects.
        """
        import os
        if os.path.splitext(uri)[1].lower() in _STRUCTURED_EXTENSIONS:
            return self.load_structured_file(uri, name=name)

        transport = self._find_transport(uri)
        result: FetchResult = transport.fetch(uri)

        doc_type = detect_type_from_source(result.source_path or uri, result.detected_mime)
        extractor = self._find_extractor(doc_type)
        text = extractor.extract(result.data, result.source_path)

        doc_name = name or Path(result.source_path or uri).stem

        chunks = chunk_document(
            doc_name, text,
            self.min_chunk_size, self.max_chunk_size, self.overflow_margin,
            include_breadcrumb=(self.context_strategy is not None),
            include_doc_name=self.include_doc_name,
        )
        return self._enrich(chunks)

    def load_bytes(
        self,
        data: bytes,
        name: str,
        doc_type: str = "auto",
        source_path: str | None = None,
    ) -> list[DocumentChunk]:
        """Skip fetch; extract from raw bytes.

        Args:
            data: Raw document bytes.
            name: Document name for chunk metadata.
            doc_type: Explicit document type (e.g. "pdf", "docx") or "auto" to detect
                      from source_path.
            source_path: Optional path hint for type detection.

        Returns:
            List of DocumentChunk objects.
        """
        if doc_type == "auto":
            resolved_type = detect_type_from_source(source_path or name, None)
            extractor = self._find_extractor(resolved_type)
        else:
            extractor = self._find_extractor_raw(doc_type)
        text = extractor.extract(data, source_path)

        chunks = chunk_document(
            name, text,
            self.min_chunk_size, self.max_chunk_size, self.overflow_margin,
            include_breadcrumb=(self.context_strategy is not None),
            include_doc_name=self.include_doc_name,
        )
        return self._enrich(chunks)

    def load_query(
        self,
        connection_url: str,
        query: str,
        name: str,
        params: dict | None = None,
    ) -> list[DocumentChunk]:
        """Execute a SQL query via SQLAlchemy and load the result as a document.

        Results are rendered as a markdown table by the CsvExtractor and then
        chunked like any other document.

        Args:
            connection_url: SQLAlchemy connection URL (e.g. ``postgresql://...``).
            query:          SQL query to execute.
            name:           Document name for chunk metadata.
            params:         Optional dict of named bind parameter values.

        Returns:
            List of DocumentChunk objects.

        Example::

            chunks = loader.load_query(
                "sqlite:///data.db",
                "SELECT title, body FROM articles WHERE published = 1",
                name="articles",
            )
        """
        from .transports._sqlalchemy import SqlAlchemyTransport
        from .extractors._csv import CsvExtractor

        transport = SqlAlchemyTransport(connection_url, query=query, params=params)
        result = transport.fetch(f"sqlalchemy://{name}")
        text = CsvExtractor().extract(result.data)

        chunks = chunk_document(
            name, text,
            self.min_chunk_size, self.max_chunk_size, self.overflow_margin,
            include_breadcrumb=(self.context_strategy is not None),
            include_doc_name=self.include_doc_name,
        )
        return self._enrich(chunks)

    def load_imap(
        self,
        uri: str,
        *,
        include_attachments: bool = False,
        limit: int | None = None,
    ) -> list[DocumentChunk]:
        """Fetch and index email messages from an IMAP mailbox.

        Each message becomes a separate indexed document.  Attachments are
        optionally extracted and appended to the message text using the
        extractor registry.

        Args:
            uri:                 ``imap[s]://user:pass@host[:port]/mailbox``
                                 Append ``?search=CRITERIA&limit=N`` for
                                 server-side filtering (IMAP RFC 3501 syntax).
            include_attachments: When True, extract and index attachment content
                                 inline with the message body (default False).
            limit:               Maximum number of messages to load, most-recent
                                 first.  Overrides any ``limit`` in the URI.

        Returns:
            Combined list of DocumentChunk objects from all messages.

        Example::

            chunks = loader.load_imap(
                "imaps://me@gmail.com:app-password@imap.gmail.com/INBOX",
                include_attachments=True,
                limit=100,
            )
        """
        from .transports._imap import ImapTransport
        from .extractors._email import EmailExtractor

        transport = ImapTransport()
        extractor = EmailExtractor(include_attachments=include_attachments)

        all_chunks: list[DocumentChunk] = []
        for result in transport.fetch_messages(uri, limit=limit):
            try:
                text = extractor.extract(result.data, source_path=result.source_path)
                if not text.strip():
                    continue
                name = result.source_path or "email"
                chunks = chunk_document(
                    name, text,
                    self.min_chunk_size, self.max_chunk_size, self.overflow_margin,
                    include_breadcrumb=(self.context_strategy is not None),
                )
                all_chunks.extend(self._enrich(chunks))
            except Exception as exc:
                logger.warning("load_imap: skipping message %s: %s", result.source_path, exc)
        return all_chunks

    def load_text(self, text: str, name: str) -> list[DocumentChunk]:
        """Skip fetch and extract; chunk and enrich pre-extracted text.

        Args:
            text: Pre-extracted plain text.
            name: Document name for chunk metadata.

        Returns:
            List of DocumentChunk objects.
        """
        chunks = chunk_document(
            name, text,
            self.min_chunk_size, self.max_chunk_size, self.overflow_margin,
            include_breadcrumb=(self.context_strategy is not None),
            include_doc_name=self.include_doc_name,
        )
        return self._enrich(chunks)

    # ── Structured metadata loaders ──────────────────────────────────────────

    def load_schema(self, tables: list[TableMeta]) -> list[DocumentChunk]:
        """Build N+1 DocumentChunks per table (one table chunk + one per column).

        Each chunk is independently embeddable: a query about a specific column
        retrieves that column's chunk directly rather than the whole table block.

        Args:
            tables: List of TableMeta describing each table and its columns.

        Returns:
            Enriched DocumentChunk list, ready for embedding.
        """
        chunks: list[DocumentChunk] = []
        for table in tables:
            db_prefix = table.source_db or "db"
            doc_name = f"schema:{db_prefix}.{table.name}"

            lines = [f"Table: {table.name}"]
            if table.schema_name:
                lines.append(f"Schema: {table.schema_name}")
            if table.source_db:
                lines.append(f"Source DB: {table.source_db}")
            if table.row_count is not None:
                lines.append(f"Row count: {table.row_count}")
            if table.description:
                lines.append(f"Description: {table.description}")
            if table.columns:
                lines.append(f"Columns: {', '.join(c.name for c in table.columns)}")

            chunks.append(DocumentChunk(
                document_name=doc_name,
                content="\n".join(lines),
                section=["table_description"],
                chunk_index=0,
                chunk_type="db_table",
            ))

            for col_idx, col in enumerate(table.columns, start=1):
                col_doc_name = f"schema:{db_prefix}.{table.name}.{col.name}"
                col_lines = [
                    f"Column: {col.name}",
                    f"Table: {table.name}",
                    f"Type: {col.data_type}",
                    f"Nullable: {'yes' if col.nullable else 'no'}",
                ]
                if col.is_primary_key:
                    col_lines.append("Primary key: yes")
                if col.is_foreign_key and col.foreign_key_ref:
                    col_lines.append(f"Foreign key: {col.foreign_key_ref}")
                if col.description:
                    col_lines.append(f"Description: {col.description}")

                chunks.append(DocumentChunk(
                    document_name=col_doc_name,
                    content="\n".join(col_lines),
                    section=["column_description"],
                    chunk_index=col_idx,
                    chunk_type="db_column",
                ))

        return self._enrich(chunks)

    def load_api(self, endpoints: list[EndpointMeta]) -> list[DocumentChunk]:
        """Build N+1 DocumentChunks per endpoint (one endpoint chunk + one per field).

        Args:
            endpoints: List of EndpointMeta describing each endpoint and its fields.

        Returns:
            Enriched DocumentChunk list, ready for embedding.
        """
        _GRAPHQL_CHUNK_TYPES = {
            "graphql_query": "api_graphql_query",
            "graphql_mutation": "api_graphql_mutation",
            "graphql_type": "api_graphql_type",
        }

        chunks: list[DocumentChunk] = []
        for endpoint in endpoints:
            api_prefix = endpoint.source_api or "api"
            doc_name = f"api:{api_prefix}.{endpoint.path}"
            chunk_type = _GRAPHQL_CHUNK_TYPES.get(endpoint.endpoint_type, "api_endpoint")

            lines = [f"Endpoint: {endpoint.path}"]
            if endpoint.method:
                lines.append(f"Method: {endpoint.method}")
            lines.append(f"Type: {endpoint.endpoint_type}")
            if endpoint.source_api:
                lines.append(f"Source API: {endpoint.source_api}")
            if endpoint.description:
                lines.append(f"Description: {endpoint.description}")
            if endpoint.fields:
                lines.append(f"Fields: {', '.join(f.name for f in endpoint.fields)}")

            chunks.append(DocumentChunk(
                document_name=doc_name,
                content="\n".join(lines),
                section=[],
                chunk_index=0,
                chunk_type=chunk_type,
            ))

            for field_idx, fld in enumerate(endpoint.fields, start=1):
                field_doc_name = f"api:{api_prefix}.{endpoint.path}.{fld.name}"
                field_lines = [
                    f"Field: {fld.name}",
                    f"Endpoint: {endpoint.path}",
                    f"Type: {fld.field_type}",
                    f"Required: {'yes' if fld.required else 'no'}",
                ]
                if fld.description:
                    field_lines.append(f"Description: {fld.description}")

                chunks.append(DocumentChunk(
                    document_name=field_doc_name,
                    content="\n".join(field_lines),
                    section=[],
                    chunk_index=field_idx,
                    chunk_type="api_field",
                ))

        return self._enrich(chunks)

    def load_structured_file(
        self, path_or_uri: str, name: str | None = None
    ) -> list[DocumentChunk]:
        """Infer schema from a structured file and return N+1 DocumentChunks.

        Supports .csv, .json, .jsonl / .ndjson, .parquet, .arrow, .feather.
        Internally calls load_schema(), so output is identical to passing a
        TableMeta directly — regardless of the source file type.

        Args:
            path_or_uri: Local path or URI of the structured file.
            name: Table name used in document_name. Defaults to the file stem.

        Returns:
            Enriched DocumentChunk list (one table chunk + one per column).
        """
        import os
        ext = os.path.splitext(path_or_uri)[1].lower()
        doc_name = name or os.path.splitext(os.path.basename(path_or_uri))[0]

        _supported = {".parquet", ".arrow", ".feather", ".csv", ".json", ".jsonl", ".ndjson"}
        if ext not in _supported:
            raise ValueError(
                f"Unsupported structured file extension: {ext!r}. "
                "Supported: .parquet, .arrow, .feather, .csv, .json, .jsonl, .ndjson"
            )

        transport = self._find_transport(path_or_uri)
        result = transport.fetch(path_or_uri)
        data = result.data

        if ext in (".parquet", ".arrow", ".feather"):
            table_meta = infer_parquet(data, ext, doc_name)
        elif ext == ".csv":
            table_meta = infer_csv(data, doc_name)
        elif ext == ".json":
            table_meta = infer_json(data, doc_name)
        else:  # .jsonl / .ndjson
            table_meta = infer_jsonl(data, doc_name)

        return self.load_schema([table_meta])

    # ── Multi-document crawl methods ─────────────────────────────────────────

    def load_crawl(
        self,
        uri: str,
        crawler: "Crawler | None" = None,
        **crawler_kwargs,
    ) -> list[DocumentChunk]:
        """Crawl *uri* with *crawler*, then load each discovered document.

        This is the generic entry point.  ``load_site`` and ``load_directory``
        are convenience wrappers that auto-select an appropriate crawler.

        Args:
            uri:            Root URI to crawl (URL, path, ``s3://`` prefix, …).
            crawler:        A ``Crawler`` implementation.  If None, uses
                            ``WebCrawler`` for http(s) and ``DirectoryCrawler``
                            for everything else.
            **crawler_kwargs: Passed to ``crawler.crawl()``.

        Returns:
            Combined list of chunks from all discovered documents.
        """
        if crawler is None:
            if uri.startswith("http://") or uri.startswith("https://"):
                crawler = WebCrawler()
            else:
                crawler = DirectoryCrawler()

        uris = crawler.crawl(uri, **crawler_kwargs)
        logger.info("load_crawl: %d URI(s) discovered from %s", len(uris), uri)

        all_chunks: list[DocumentChunk] = []
        for doc_uri in uris:
            try:
                chunks = self.load(doc_uri)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.warning("load_crawl: skipping %s: %s", doc_uri, exc)
        return all_chunks

    def load_site(
        self,
        url: str,
        max_pages: int = 50,
        max_depth: int = 3,
        same_domain: bool = True,
        exclude_patterns: list[str] | None = None,
        include_pattern: str | None = None,
        crawler: "Crawler | None" = None,
    ) -> list[DocumentChunk]:
        """Crawl a website and load all discovered HTML pages.

        Extend for authenticated services (SharePoint, Confluence, …) by
        passing a custom ``crawler`` that implements the ``Crawler`` protocol::

            class SharePointCrawler:
                def can_handle(self, uri): return "sharepoint.com" in uri
                def crawl(self, uri, **kw): ...  # return list of URIs

            chunks = loader.load_site(url, crawler=SharePointCrawler())

        Args:
            url:              Root URL to start from.
            max_pages:        Maximum pages to fetch (default 50).
            max_depth:        Maximum link-follow depth (default 3).
            same_domain:      Stay on the same hostname (default True).
            exclude_patterns: Regex patterns for URLs to skip.
            include_pattern:  If set, only follow URLs matching this regex.
            crawler:          Custom crawler; overrides all other params.

        Returns:
            Combined list of DocumentChunk objects from all pages.
        """
        if crawler is None:
            crawler = WebCrawler(
                max_pages=max_pages,
                max_depth=max_depth,
                same_domain=same_domain,
                exclude_patterns=exclude_patterns,
                include_pattern=include_pattern,
            )
        return self.load_crawl(url, crawler=crawler)

    def load_directory(
        self,
        path: str,
        extensions: list[str] | None = None,
        recursive: bool = True,
        max_files: int = 1000,
        crawler: "Crawler | None" = None,
    ) -> list[DocumentChunk]:
        """Load all documents in a local directory or S3 prefix.

        Extend for cloud storage (Azure Blob, GCS, …) by passing a custom
        ``crawler`` that implements the ``Crawler`` protocol.

        Args:
            path:       Directory path (local or ``s3://bucket/prefix``).
            extensions: File extensions to include (default: broad document set).
            recursive:  Recurse into subdirectories (default True).
            max_files:  Maximum files to process (default 1000).
            crawler:    Custom crawler; overrides all other params.

        Returns:
            Combined list of DocumentChunk objects from all files.
        """
        if crawler is None:
            crawler = DirectoryCrawler(
                extensions=extensions,
                recursive=recursive,
                max_files=max_files,
            )
        return self.load_crawl(path, crawler=crawler)
