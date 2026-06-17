# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: f6c8d1e5-7a0b-4f2c-e9a6-5b4d0c3f8a1e
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CosmosCrawler — index documents from Azure Cosmos DB (NoSQL API) via the SDK.

Implements both the ``Crawler`` and ``Transport`` protocols. Uses the
``azure-cosmos`` Python SDK to iterate over items in one or more containers.

Schema is inferred by sampling up to ``schema_sample_size`` items per container.
One schema chunk is emitted per container alongside the document chunks.

Usage::

    from chonk.transports import CosmosCrawler
    from chonk.loader import DocumentLoader

    crawler = CosmosCrawler(
        url="https://myaccount.documents.azure.com:443/",
        key="base64key==",
        database="mydb",
        containers=["articles"],
    )
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl("cosmos://mydb/articles", crawler=crawler)

    # Multiple containers with a custom query
    crawler = CosmosCrawler(
        url="https://myaccount.documents.azure.com:443/",
        key="base64key==",
        database="mydb",
        containers=["articles", "reports"],
        query="SELECT c.id, c.title, c.body FROM c WHERE c.published = true",
        max_item_count=500,
        field_aliases={"_ts": "timestamp", "_etag": "etag"},
    )

    # NER vocabulary
    vocab = crawler.get_field_names()  # after crawl()

Requires: azure-cosmos>=4.5  (``pip install azure-cosmos``)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ._protocol import FetchOptions, FetchResult
from ._schema_infer import DEFAULT_SCHEMA_SAMPLE_SIZE, collect_field_paths, infer_schema_text

_log = logging.getLogger(__name__)

_SCHEMA_SAMPLE_SIZE = DEFAULT_SCHEMA_SAMPLE_SIZE


def _default(obj: Any) -> Any:  # noqa: ANN401
    return str(obj)


class CosmosCrawler:
    """Crawl Azure Cosmos DB containers and index each item as a JSON FetchResult.

    Args:
        url:                Cosmos DB account endpoint URL.
        key:                Account key or resource token.
        database:           Database name.
        containers:         Container names to crawl.  ``None`` = all containers.
        query:              Cosmos DB SQL query (default ``SELECT * FROM c``).
        max_item_count:     Max items per request page (default 200).
        connection_mode:    ``"Gateway"`` (default) or ``"Direct"``.
        schema_sample_size: Max items sampled per container for schema inference (default 500).
        field_aliases:      Map raw field names to normalized names for NER vocabulary.
    """

    SCHEME = "cosmos"

    def __init__(
        self,
        url: str,
        key: str,
        database: str,
        containers: list[str] | None = None,
        query: str = "SELECT * FROM c",
        max_item_count: int = 200,
        connection_mode: str = "Gateway",
        schema_sample_size: int = _SCHEMA_SAMPLE_SIZE,
        field_aliases: dict[str, str] | None = None,
    ) -> None:
        self._url = url
        self._key = key
        self._database = database
        self._containers = containers
        self._query = query
        self._max_item_count = max_item_count
        self._connection_mode = connection_mode
        self._schema_sample_size = schema_sample_size
        self._field_aliases: dict[str, str] = field_aliases or {}
        self._cache: dict[str, FetchResult] = {}
        self._known_fields: set[str] = set()
        self._known_containers: list[str] = []

    # ── Transport protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(f"{self.SCHEME}://")

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        if uri not in self._cache:
            raise KeyError(f"CosmosCrawler: unknown URI {uri!r} — call crawl() first")
        return self._cache[uri]

    # ── Crawler protocol ──────────────────────────────────────────────────────

    def crawl(self, uri: str = "", **__: object) -> list[str]:
        """Query all configured containers, cache items, and emit schema chunks.

        Returns:
            List of ``cosmos://{database}/{container}/{id}`` URIs followed by
            one schema URI per container.
        """
        try:
            from azure.cosmos import CosmosClient, PartitionKey  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "azure-cosmos is required for CosmosCrawler. Install with: pip install azure-cosmos"
            ) from exc

        client = CosmosClient(self._url, credential=self._key)
        db = client.get_database_client(self._database)

        container_names: list[str]
        if self._containers:
            container_names = self._containers
        else:
            container_names = [c["id"] for c in db.list_containers()]

        self._cache.clear()
        self._known_fields.clear()
        self._known_containers = list(container_names)
        total = 0

        for cname in container_names:
            container = db.get_container_client(cname)
            items = container.query_items(
                query=self._query,
                max_item_count=self._max_item_count,
                enable_cross_partition_query=True,
            )
            sample: list[dict[str, object]] = []
            cname_total = 0

            for item in items:
                item_id = str(item.get("id", total))
                if len(sample) < self._schema_sample_size:
                    sample.append(json.loads(json.dumps(item, default=_default)))
                doc_uri = f"{self.SCHEME}://{self._database}/{cname}/{item_id}"
                payload = {
                    "_source_meta": {
                        "type": "cosmos",
                        "url": self._url,
                        "database": self._database,
                        "container": cname,
                        "item_id": item_id,
                    },
                    **json.loads(json.dumps(item, default=_default)),
                }
                self._cache[doc_uri] = FetchResult(
                    data=json.dumps(payload, indent=2, default=_default).encode("utf-8"),
                    detected_mime="application/json",
                    source_path=f"{self._database}/{cname}/{item_id}",
                )
                total += 1
                cname_total += 1

            # ── Schema chunk per container ────────────────────────────────────
            self._known_fields.update(collect_field_paths(sample))
            schema_text = infer_schema_text(
                sample,
                f"{self._database}/{cname}",
                total_docs=cname_total,
                sample_size=len(sample),
            )
            schema_uri = f"{self.SCHEME}://{self._database}/{cname}/_schema"
            self._cache[schema_uri] = FetchResult(
                data=schema_text.encode("utf-8"),
                detected_mime="text/plain",
                source_path=f"{self._database}/{cname}/_schema",
            )

        _log.info("CosmosCrawler: indexed %d item(s)", total)
        return list(self._cache.keys())

    # ── NER vocabulary ────────────────────────────────────────────────────────

    def get_field_names(self) -> list[str]:
        """Return normalized field names, container names, and database name.

        Call after ``crawl()``.  Aliases defined in ``field_aliases`` are applied.
        """
        normalized = {self._field_aliases.get(f, f) for f in self._known_fields}
        normalized.update(self._known_containers)
        normalized.add(self._database)
        return sorted(normalized)
