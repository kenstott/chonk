# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: a1f3b7c2-4e8d-4a9f-b5c3-2d0e6f8a1b4c
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""MongoCrawler — index documents from MongoDB collections.

Implements both the ``Crawler`` and ``Transport`` protocols. Pass the same
instance as both ``crawler=`` and in ``extra_transports=``.

Each MongoDB document is serialised to pretty-printed JSON and returned as a
separate ``FetchResult``.  The ``_id`` field is cast to string so it is always
serialisable.

Schema inference tries the collection's ``$jsonSchema`` validator first; if
none is defined, it falls back to a ``$sample`` aggregation.  One schema chunk
is emitted per collection alongside the document chunks.

Usage::

    from chonk.transports import MongoCrawler
    from chonk.loader import DocumentLoader

    # Index an entire database
    crawler = MongoCrawler("mongodb://localhost:27017", database="mydb")
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl("mongodb://localhost:27017/mydb", crawler=crawler)

    # Index specific collections with a filter
    crawler = MongoCrawler(
        "mongodb+srv://user:pass@cluster.mongodb.net",
        database="mydb",
        collections=["articles", "reports"],
        query={"published": True},
        projection={"title": 1, "body": 1, "author": 1},
        batch_size=500,
        field_aliases={"_id": "id", "ts": "timestamp"},
    )

    # NER vocabulary
    vocab = crawler.get_field_names()  # after crawl()

Requires: pymongo>=4.0  (``pip install pymongo``)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ._protocol import FetchResult
from ._schema_infer import DEFAULT_SCHEMA_SAMPLE_SIZE, collect_field_paths, infer_schema_text

_log = logging.getLogger(__name__)

_SCHEMA_SAMPLE_SIZE = DEFAULT_SCHEMA_SAMPLE_SIZE


def _to_json(doc: dict) -> bytes:
    def _default(obj: Any) -> Any:
        return str(obj)
    return json.dumps(doc, indent=2, default=_default).encode("utf-8")


class MongoCrawler:
    """Crawl MongoDB collections and index each document as a JSON FetchResult.

    Implements both ``Crawler`` and ``Transport`` — pass the same instance to both::

        crawler = MongoCrawler("mongodb://localhost:27017", database="mydb")
        loader = DocumentLoader(extra_transports=[crawler])
        chunks = loader.load_crawl("mongodb://localhost:27017/mydb", crawler=crawler)

    Args:
        connection_uri:  MongoDB connection string (``mongodb://`` or ``mongodb+srv://``).
        database:        Database name to index.  May also be embedded in the URI.
        collections:     Restrict to these collection names.  ``None`` = all
                         non-system collections.
        query:           PyMongo filter document applied to every ``find()`` call.
                         Defaults to ``{}`` (all documents).
        projection:      PyMongo projection document.  ``None`` = all fields.
        batch_size:      Cursor batch size (default 200).
        field_aliases:   Map raw field names to normalized names for NER vocabulary
                         (e.g. ``{"_id": "id", "ts": "timestamp"}``).
    """

    SCHEME = "mongodb"

    def __init__(
        self,
        connection_uri: str,
        database: str | None = None,
        collections: list[str] | None = None,
        query: dict | None = None,
        projection: dict | None = None,
        batch_size: int = 200,
        field_aliases: dict[str, str] | None = None,
    ):
        self._uri = connection_uri
        self._database = database or self._db_from_uri(connection_uri)
        self._collections = collections
        self._query: dict = query or {}
        self._projection: dict | None = projection
        self._batch_size = batch_size
        self._field_aliases: dict[str, str] = field_aliases or {}
        self._cache: dict[str, FetchResult] = {}
        self._known_fields: set[str] = set()
        self._known_collections: list[str] = []
        self._known_database: str = ""

    # ── Transport protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("mongodb://") or uri.startswith("mongodb+srv://")

    def fetch(self, uri: str, **__) -> FetchResult:
        if uri not in self._cache:
            raise KeyError(
                f"MongoCrawler: unknown URI {uri!r} — call crawl() first"
            )
        return self._cache[uri]

    # ── Crawler protocol ──────────────────────────────────────────────────────

    def crawl(self, uri: str = "", **__) -> list[str]:
        """Connect to MongoDB, index all matching documents, and emit schema chunks.

        Args:
            uri:  Connection URI (overrides constructor value if non-empty).

        Returns:
            List of ``mongodb://`` URIs — document URIs followed by schema URIs.
        """
        try:
            import pymongo
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoCrawler. "
                "Install with: pip install pymongo"
            )

        connection_uri = uri or self._uri
        db_name = self._db_from_uri(connection_uri) or self._database
        if not db_name:
            raise ValueError(
                "MongoCrawler: database name required — pass database= or embed it in the URI."
            )

        client = pymongo.MongoClient(connection_uri)
        db = client[db_name]

        collection_names: list[str]
        if self._collections:
            collection_names = self._collections
        else:
            collection_names = [
                name
                for name in db.list_collection_names()
                if not name.startswith("system.")
            ]

        self._cache.clear()
        self._known_fields.clear()
        self._known_collections = list(collection_names)
        self._known_database = db_name

        total = 0
        for coll_name in collection_names:
            coll = db[coll_name]
            cursor = coll.find(
                self._query,
                self._projection,
                batch_size=self._batch_size,
            )
            for doc in cursor:
                doc_id = str(doc.get("_id", ""))
                doc_uri = f"mongodb://{db_name}/{coll_name}/{doc_id}"
                payload = {
                    "_source_meta": {
                        "type": "mongodb",
                        "database": db_name,
                        "collection": coll_name,
                        "doc_id": doc_id,
                    },
                    **{str(k): v for k, v in doc.items()},
                }
                self._cache[doc_uri] = FetchResult(
                    data=_to_json(payload),
                    detected_mime="application/json",
                    source_path=f"{db_name}/{coll_name}/{doc_id}",
                )
                total += 1

        # ── Schema chunks ─────────────────────────────────────────────────────
        for coll_name in collection_names:
            schema_text = self._schema_from_validator(db, coll_name, db_name)
            schema_uri = f"mongodb://{db_name}/{coll_name}/_schema"
            self._cache[schema_uri] = FetchResult(
                data=schema_text.encode("utf-8"),
                detected_mime="text/plain",
                source_path=f"{db_name}/{coll_name}/_schema",
            )

        client.close()
        _log.info(
            "MongoCrawler: indexed %d document(s) across %d collection(s)",
            total, len(collection_names),
        )
        return list(self._cache.keys())

    # ── NER vocabulary ────────────────────────────────────────────────────────

    def get_field_names(self) -> list[str]:
        """Return normalized field names, collection names, and database name.

        Suitable for seeding a ``VocabularyMatcher`` for NER.  Call after
        ``crawl()``.  Aliases defined in ``field_aliases`` are applied to
        field names.
        """
        normalized = {self._field_aliases.get(f, f) for f in self._known_fields}
        result = normalized | set(self._known_collections)
        if self._known_database:
            result.add(self._known_database)
        return sorted(result)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _schema_from_validator(self, db: Any, coll_name: str, db_name: str) -> str:
        """Return schema text: $jsonSchema validator if available, else sample inference."""
        try:
            for info in db.list_collections(filter={"name": coll_name}):
                json_schema = info.get("options", {}).get("validator", {}).get("$jsonSchema")
                if json_schema:
                    self._known_fields.update(self._fields_from_json_schema(json_schema))
                    return (
                        f"Source: {db_name}/{coll_name}\n"
                        f"Schema (validator $jsonSchema):\n"
                        f"{json.dumps(json_schema, indent=2)}\n"
                    )
        except Exception:
            pass

        # Fallback: $sample aggregation
        try:
            sample = list(db[coll_name].aggregate([{"$sample": {"size": _SCHEMA_SAMPLE_SIZE}}]))
        except Exception:
            sample = []
        self._known_fields.update(collect_field_paths(sample))
        total = db[coll_name].estimated_document_count()
        return infer_schema_text(
            sample, f"{db_name}/{coll_name}",
            total_docs=total, sample_size=len(sample),
        )

    @staticmethod
    def _fields_from_json_schema(schema: dict, prefix: str = "") -> set[str]:
        result: set[str] = set()
        for name, sub in schema.get("properties", {}).items():
            path = f"{prefix}.{name}" if prefix else name
            result.add(path)
            if isinstance(sub, dict):
                result |= MongoCrawler._fields_from_json_schema(sub, path)
        return result

    @staticmethod
    def _db_from_uri(uri: str) -> str | None:
        try:
            from urllib.parse import urlparse
            path = urlparse(uri).path.lstrip("/").split("?")[0]
            return path or None
        except Exception:
            return None
