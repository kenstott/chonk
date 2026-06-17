# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: c3f5a8b2-4d7e-4c9f-b6d3-2e1a7f0b5c8d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SolrCrawler — index documents from Apache Solr via the JSON query API.

Implements both the ``Crawler`` and ``Transport`` protocols. Pagination uses
Solr's cursor mark (``cursorMark``) API for efficient deep pagination over
arbitrarily large collections.

Schema is retrieved via ``GET /solr/{collection}/schema`` and emitted as a
single text chunk alongside document chunks.

Usage::

    from chonk.transports import SolrCrawler
    from chonk.loader import DocumentLoader

    # Index all documents in a collection
    crawler = SolrCrawler("http://localhost:8983/solr", collection="articles")
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl("http://localhost:8983/solr/articles", crawler=crawler)

    # Filter and project fields
    crawler = SolrCrawler(
        "http://localhost:8983/solr",
        collection="kb",
        query="published:true",
        fields=["id", "title", "body", "author"],
        page_size=500,
        username="solr",
        password="SolrRocks",
        field_aliases={"_version_": "version"},
    )

    # NER vocabulary
    vocab = crawler.get_field_names()  # after crawl()

Requires: requests>=2.28  (``pip install requests``)
"""

from __future__ import annotations

import json
import logging

from ._protocol import FetchResult

_log = logging.getLogger(__name__)


class SolrCrawler:
    """Crawl an Apache Solr collection via the select handler.

    Implements both ``Crawler`` and ``Transport`` — pass the same instance to both::

        crawler = SolrCrawler("http://localhost:8983/solr", collection="articles")
        loader = DocumentLoader(extra_transports=[crawler])
        chunks = loader.load_crawl("http://localhost:8983/solr/articles", crawler=crawler)

    Args:
        base_url:     Solr base URL including ``/solr`` (e.g. ``http://localhost:8983/solr``).
        collection:   Solr collection (or core) name.
        query:        Solr query string (default ``*:*``).
        fields:       List of field names to retrieve.  ``None`` = all fields (``fl=*``).
        page_size:    Documents per cursor page (default 200).
        username:     HTTP Basic auth username.
        password:     HTTP Basic auth password.
        verify_ssl:   Verify TLS certificates (default True).
        field_aliases: Map raw field names to normalized names for NER vocabulary.
    """

    def __init__(
        self,
        base_url: str,
        collection: str,
        query: str = "*:*",
        fields: list[str] | None = None,
        page_size: int = 200,
        username: str | None = None,
        password: str | None = None,
        verify_ssl: bool = True,
        field_aliases: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._collection = collection
        self._query = query
        self._fields = ",".join(fields) if fields else "*"
        self._page_size = page_size
        self._username = username
        self._password = password
        self._verify_ssl = verify_ssl
        self._field_aliases: dict[str, str] = field_aliases or {}
        self._cache: dict[str, FetchResult] = {}
        self._known_fields: set[str] = set()

    # ── Transport protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(f"{self._base_url}/{self._collection}")

    def fetch(self, uri: str, **__: object) -> FetchResult:
        if uri not in self._cache:
            raise KeyError(f"SolrCrawler: unknown URI {uri!r} — call crawl() first")
        return self._cache[uri]

    # ── Crawler protocol ──────────────────────────────────────────────────────

    def crawl(self, uri: str = "", **__: object) -> list[str]:
        """Paginate through the collection using cursorMark, cache all documents, and emit schema.

        Args:
            uri:  Ignored — collection is fixed at construction time.

        Returns:
            List of ``{base_url}/{collection}/select/{id}`` URIs followed by one schema URI.
        """
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "requests is required for SolrCrawler. Install with: pip install requests"
            ) from exc

        session = requests.Session()
        session.verify = self._verify_ssl
        if self._username:
            session.auth = (self._username, self._password or "")

        select_url = f"{self._base_url}/{self._collection}/select"
        cursor_mark = "*"
        self._cache.clear()
        self._known_fields.clear()
        total = 0

        while True:
            params = {
                "q": self._query,
                "fl": self._fields,
                "rows": self._page_size,
                "sort": "id asc",
                "cursorMark": cursor_mark,
                "wt": "json",
            }
            resp = session.get(select_url, params=params)
            resp.raise_for_status()
            data = resp.json()

            docs = data.get("response", {}).get("docs", [])
            if not docs:
                break

            for doc in docs:
                doc_id = str(doc.get("id", ""))
                doc_uri = f"{self._base_url}/{self._collection}/select/{doc_id}"
                payload = {
                    "_source_meta": {
                        "type": "solr",
                        "base_url": self._base_url,
                        "collection": self._collection,
                        "doc_id": doc_id,
                    },
                    **doc,
                }
                self._cache[doc_uri] = FetchResult(
                    data=json.dumps(payload, indent=2, default=str).encode("utf-8"),
                    detected_mime="application/json",
                    source_path=f"{self._collection}/{doc_id}",
                )
                total += 1

            next_cursor = data.get("nextCursorMark", cursor_mark)
            if next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor

        # ── Schema chunk via /schema API ──────────────────────────────────────
        schema_uri = f"{self._base_url}/{self._collection}/_schema"
        schema_text = self._fetch_schema(session)
        self._cache[schema_uri] = FetchResult(
            data=schema_text.encode("utf-8"),
            detected_mime="text/plain",
            source_path=f"{self._collection}/_schema",
        )

        _log.info("SolrCrawler: indexed %d document(s) from %r", total, self._collection)
        return list(self._cache.keys())

    # ── NER vocabulary ────────────────────────────────────────────────────────

    def get_field_names(self) -> list[str]:
        """Return normalized field names and collection name.

        Call after ``crawl()``.  Aliases defined in ``field_aliases`` are applied.
        """
        normalized = {self._field_aliases.get(f, f) for f in self._known_fields}
        normalized.add(self._collection)
        return sorted(normalized)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch_schema(self, session: object) -> str:
        try:
            resp = session.get(  # type: ignore[union-attr]
                f"{self._base_url}/{self._collection}/schema",
                params={"wt": "json"},
            )
            resp.raise_for_status()
            schema = resp.json().get("schema", {})
        except Exception as exc:
            return f"Source: {self._collection}\nSchema unavailable: {exc}\n"

        schema_name = schema.get("name", self._collection)
        fields = schema.get("fields", [])

        lines = [
            f"Source: {self._collection}",
            f"Schema (Solr): {schema_name}",
            "",
            f"{'Field':<40}  {'Type':<24}  Flags",
            "-" * 80,
        ]

        for field in sorted(fields, key=lambda f: f["name"]):
            name = field["name"]
            ftype = field.get("type", "")
            self._known_fields.add(name)
            flags = []
            if field.get("required"):
                flags.append("required")
            if field.get("multiValued"):
                flags.append("multiValued")
            if not field.get("indexed", True):
                flags.append("not-indexed")
            if not field.get("stored", True):
                flags.append("not-stored")
            lines.append(f"  {name:<40}  {ftype:<24}  {', '.join(flags)}")

        dyn = schema.get("dynamicFields", [])
        if dyn:
            lines.append("")
            lines.append("Dynamic fields:")
            for field in sorted(dyn, key=lambda f: f["name"]):
                lines.append(f"  {field['name']:<40}  {field.get('type', '')}")

        copy = schema.get("copyFields", [])
        if copy:
            lines.append("")
            lines.append("Copy fields:")
            for cf in copy:
                lines.append(f"  {cf.get('source', '')} → {cf.get('dest', '')}")

        return "\n".join(lines) + "\n"
