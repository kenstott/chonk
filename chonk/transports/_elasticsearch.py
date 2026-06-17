# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b2e4f7a1-3c9d-4b8e-a5f2-1d0c6e9b3a7f
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ElasticsearchCrawler — index documents from Elasticsearch or OpenSearch via REST API.

Implements both the ``Crawler`` and ``Transport`` protocols. The crawler uses the
``search_after`` pagination API so it works with arbitrarily large indices without
requiring scroll context management.

Both Elasticsearch (≥7.x) and OpenSearch (≥1.x) are supported — they share the
same REST API surface for the operations used here.

Schema is retrieved via ``GET /{index}/_mapping`` and emitted as a single text
chunk alongside document chunks.

Usage::

    from chonk.transports import ElasticsearchCrawler
    from chonk.loader import DocumentLoader

    # Index all documents in an index
    crawler = ElasticsearchCrawler("https://localhost:9200", index="articles")
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl("https://localhost:9200/articles", crawler=crawler)

    # With authentication and field filtering
    crawler = ElasticsearchCrawler(
        "https://my-cluster.es.io:9243",
        index="kb-docs",
        api_key="base64encodedkey==",
        source_fields=["title", "body", "author", "tags"],
        query={"term": {"published": True}},
        page_size=500,
        field_aliases={"@timestamp": "timestamp"},
    )

    # NER vocabulary
    vocab = crawler.get_field_names()  # after crawl()

Requires: requests>=2.28  (``pip install requests``)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ._protocol import FetchOptions, FetchResult

_log = logging.getLogger(__name__)


class ElasticsearchCrawler:
    """Crawl an Elasticsearch / OpenSearch index via the REST search API.

    Implements both ``Crawler`` and ``Transport`` — pass the same instance to both::

        crawler = ElasticsearchCrawler("https://localhost:9200", index="articles")
        loader = DocumentLoader(extra_transports=[crawler])
        chunks = loader.load_crawl("https://localhost:9200/articles", crawler=crawler)

    Args:
        base_url:       Elasticsearch/OpenSearch base URL (e.g. ``https://localhost:9200``).
        index:          Index name (or index pattern, e.g. ``logs-*``).
        api_key:        Elasticsearch API key (base-64 encoded ``id:key`` string).
        username:       HTTP Basic auth username (alternative to api_key).
        password:       HTTP Basic auth password.
        query:          Elasticsearch query DSL dict (default ``{"match_all": {}})``).
        source_fields:  Fields to include in ``_source``.  ``None`` = all fields.
        page_size:      Documents per page (default 200).
        verify_ssl:     Verify TLS certificates (default True).
        field_aliases:  Map raw field names to normalized names for NER vocabulary.
    """

    def __init__(
        self,
        base_url: str,
        index: str,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        query: dict[str, object] | None = None,
        source_fields: list[str] | None = None,
        page_size: int = 200,
        verify_ssl: bool = True,
        field_aliases: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._index = index
        self._api_key = api_key
        self._username = username
        self._password = password
        self._query: dict[str, object] = query or {"match_all": {}}
        self._source_fields = source_fields
        self._page_size = page_size
        self._verify_ssl = verify_ssl
        self._field_aliases: dict[str, str] = field_aliases or {}
        self._cache: dict[str, FetchResult] = {}
        self._known_fields: set[str] = set()

    # ── Transport protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(self._base_url)

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        if uri not in self._cache:
            raise KeyError(f"ElasticsearchCrawler: unknown URI {uri!r} — call crawl() first")
        return self._cache[uri]

    # ── Crawler protocol ──────────────────────────────────────────────────────

    def crawl(self, uri: str = "", **__: object) -> list[str]:
        """Paginate through the index using search_after, cache all documents, and emit schema.

        Args:
            uri:  Ignored — index is fixed at construction time.

        Returns:
            List of ``{base_url}/{index}/_doc/{id}`` URIs followed by one schema URI.
        """
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "requests is required for ElasticsearchCrawler. Install with: pip install requests"
            ) from exc

        session = requests.Session()
        session.verify = self._verify_ssl
        if self._api_key:
            session.headers["Authorization"] = f"ApiKey {self._api_key}"
        elif self._username:
            session.auth = (self._username, self._password or "")
        session.headers["Content-Type"] = "application/json"

        self._cache.clear()
        self._known_fields.clear()
        search_after: list[Any] | None = None
        total = 0

        while True:
            body: dict[str, object] = {
                "size": self._page_size,
                "query": self._query,
                "sort": [{"_id": "asc"}],
            }
            if self._source_fields is not None:
                body["_source"] = self._source_fields
            if search_after is not None:
                body["search_after"] = search_after

            url = f"{self._base_url}/{self._index}/_search"
            resp = session.post(url, json=body)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                doc_id = hit["_id"]
                source = hit.get("_source", {})
                doc_uri = f"{self._base_url}/{self._index}/_doc/{doc_id}"
                payload = {
                    "_source_meta": {
                        "type": "elasticsearch",
                        "base_url": self._base_url,
                        "index": self._index,
                        "doc_id": doc_id,
                    },
                    "_id": doc_id,
                    "_index": hit.get("_index", self._index),
                    **source,
                }
                self._cache[doc_uri] = FetchResult(
                    data=json.dumps(payload, indent=2, default=str).encode("utf-8"),
                    detected_mime="application/json",
                    source_path=f"{self._index}/{doc_id}",
                )
                total += 1

            search_after = hits[-1]["sort"]
            if len(hits) < self._page_size:
                break

        # ── Schema chunk via _mapping API ─────────────────────────────────────
        schema_uri = f"{self._base_url}/{self._index}/_schema"
        schema_text = self._fetch_mapping_schema(session)
        self._cache[schema_uri] = FetchResult(
            data=schema_text.encode("utf-8"),
            detected_mime="text/plain",
            source_path=f"{self._index}/_schema",
        )

        _log.info("ElasticsearchCrawler: indexed %d document(s) from %r", total, self._index)
        return list(self._cache.keys())

    # ── NER vocabulary ────────────────────────────────────────────────────────

    def get_field_names(self) -> list[str]:
        """Return normalized field names and index name.

        Call after ``crawl()``.  Aliases defined in ``field_aliases`` are applied.
        """
        normalized = {self._field_aliases.get(f, f) for f in self._known_fields}
        normalized.add(self._index)
        return sorted(normalized)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch_mapping_schema(self, session: Any) -> str:  # noqa: ANN401
        try:
            resp = session.get(f"{self._base_url}/{self._index}/_mapping")
            resp.raise_for_status()
            mapping = resp.json()
        except Exception as exc:
            return f"Source: {self._index}\nMapping unavailable: {exc}\n"

        lines = [f"Source: {self._index}", "Schema (Elasticsearch/_mapping):", ""]
        lines.append(f"{'Field':<50}  Type")
        lines.append("-" * 72)

        for idx_data in mapping.values():
            props = idx_data.get("mappings", {}).get("properties", {})
            self._walk_mapping(props, lines, prefix="")

        return "\n".join(lines) + "\n"

    def _walk_mapping(self, props: dict[str, Any], lines: list[str], prefix: str) -> None:
        for name, meta in sorted(props.items()):
            path = f"{prefix}.{name}" if prefix else name
            ftype = meta.get("type", "object")
            self._known_fields.add(path)
            indent = "  " * path.count(".")
            lines.append(f"  {indent}{path:<48}  {ftype}")
            if "properties" in meta:
                self._walk_mapping(meta["properties"], lines, prefix=path)
            for subname, submeta in meta.get("fields", {}).items():
                subpath = f"{path}.{subname}"
                self._known_fields.add(subpath)
                lines.append(
                    f"  {indent}  {subpath:<46}  {submeta.get('type', 'unknown')}  (multi-field)"
                )
