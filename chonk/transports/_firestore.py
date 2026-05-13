# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: e5b7c0d4-6f9a-4e1b-d8f5-4a3c9b2e7f0d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""FirestoreCrawler — index documents from Google Cloud Firestore.

Implements both the ``Crawler`` and ``Transport`` protocols. Streams all
documents from one or more Firestore collections (or collection groups) and
returns each as a JSON ``FetchResult``.

Schema is inferred by sampling up to ``schema_sample_size`` documents per
collection.  One schema chunk is emitted per collection.

Usage::

    from chonk.transports import FirestoreCrawler
    from chonk.loader import DocumentLoader

    # Credentials from GOOGLE_APPLICATION_CREDENTIALS env var or ADC
    crawler = FirestoreCrawler(project="my-gcp-project", collections=["articles"])
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl("firestore://my-gcp-project/articles", crawler=crawler)

    # Multiple collections + explicit credentials
    crawler = FirestoreCrawler(
        project="my-gcp-project",
        collections=["articles", "reports", "kb"],
        credentials_path="/path/to/service-account.json",
        field_aliases={"createdAt": "created_at", "userId": "user_id"},
    )

    # NER vocabulary
    vocab = crawler.get_field_names()  # after crawl()

Requires: google-cloud-firestore>=2.11  (``pip install google-cloud-firestore``)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ._protocol import FetchResult
from ._schema_infer import collect_field_paths, infer_schema_text

_log = logging.getLogger(__name__)

_SCHEMA_SAMPLE_SIZE = 500


def _default(obj: Any) -> Any:
    from datetime import date, datetime
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    try:
        return str(obj)
    except Exception:
        return repr(obj)


class FirestoreCrawler:
    """Crawl Google Cloud Firestore collections and index each document as JSON.

    Args:
        project:            GCP project ID.
        collections:        List of top-level collection names to crawl.
        credentials_path:   Path to a service-account JSON key file.
                            If ``None``, uses Application Default Credentials.
        database:           Firestore database ID (default ``(default)``).
        schema_sample_size: Max docs sampled per collection for schema inference (default 500).
        field_aliases:      Map raw field names to normalized names for NER vocabulary.
    """

    SCHEME = "firestore"

    def __init__(
        self,
        project: str,
        collections: list[str],
        credentials_path: str | None = None,
        database: str = "(default)",
        schema_sample_size: int = _SCHEMA_SAMPLE_SIZE,
        field_aliases: dict[str, str] | None = None,
    ):
        self._project = project
        self._collections = collections
        self._credentials_path = credentials_path
        self._database = database
        self._schema_sample_size = schema_sample_size
        self._field_aliases: dict[str, str] = field_aliases or {}
        self._cache: dict[str, FetchResult] = {}
        self._known_fields: set[str] = set()

    # ── Transport protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(f"{self.SCHEME}://")

    def fetch(self, uri: str, **__) -> FetchResult:
        if uri not in self._cache:
            raise KeyError(f"FirestoreCrawler: unknown URI {uri!r} — call crawl() first")
        return self._cache[uri]

    # ── Crawler protocol ──────────────────────────────────────────────────────

    def crawl(self, uri: str = "", **__) -> list[str]:
        """Stream all documents from the configured collections and emit schema chunks.

        Returns:
            List of ``firestore://{project}/{collection}/{doc_id}`` URIs followed
            by one schema URI per collection.
        """
        try:
            from google.cloud import firestore as _fs
        except ImportError:
            raise ImportError(
                "google-cloud-firestore is required for FirestoreCrawler. "
                "Install with: pip install google-cloud-firestore"
            )

        if self._credentials_path:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                self._credentials_path
            )
            client = _fs.Client(project=self._project, credentials=creds, database=self._database)
        else:
            client = _fs.Client(project=self._project, database=self._database)

        self._cache.clear()
        self._known_fields.clear()
        total = 0

        for coll_name in self._collections:
            coll_ref = client.collection(coll_name)
            sample: list[dict] = []

            for doc_snapshot in coll_ref.stream():
                doc_id = doc_snapshot.id
                data = doc_snapshot.to_dict() or {}
                if len(sample) < self._schema_sample_size:
                    # Serialize then re-parse to normalize types for schema inference
                    sample.append(json.loads(json.dumps(data, default=_default)))
                doc_uri = f"{self.SCHEME}://{self._project}/{coll_name}/{doc_id}"
                payload = {
                    "_source_meta": {
                        "type": "firestore",
                        "project": self._project,
                        "database": self._database,
                        "collection": coll_name,
                        "doc_id": doc_id,
                    },
                    **data,
                }
                self._cache[doc_uri] = FetchResult(
                    data=json.dumps(payload, indent=2, default=_default).encode("utf-8"),
                    detected_mime="application/json",
                    source_path=f"{coll_name}/{doc_id}",
                )
                total += 1

            # ── Schema chunk per collection ───────────────────────────────────
            self._known_fields.update(collect_field_paths(sample))
            schema_text = infer_schema_text(
                sample, f"{self._project}/{coll_name}",
                total_docs=total, sample_size=len(sample),
            )
            schema_uri = f"{self.SCHEME}://{self._project}/{coll_name}/_schema"
            self._cache[schema_uri] = FetchResult(
                data=schema_text.encode("utf-8"),
                detected_mime="text/plain",
                source_path=f"{coll_name}/_schema",
            )

        _log.info("FirestoreCrawler: indexed %d document(s)", total)
        return list(self._cache.keys())

    # ── NER vocabulary ────────────────────────────────────────────────────────

    def get_field_names(self) -> list[str]:
        """Return normalized field names, collection names, and project name.

        Call after ``crawl()``.  Aliases defined in ``field_aliases`` are applied.
        """
        normalized = {self._field_aliases.get(f, f) for f in self._known_fields}
        normalized.update(self._collections)
        normalized.add(self._project)
        return sorted(normalized)
