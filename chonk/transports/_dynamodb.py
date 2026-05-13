# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d4a6b9c3-5e8f-4d0a-c7e4-3f2b8a1d6e9c
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DynamoDBCrawler — index items from AWS DynamoDB tables via boto3.

Implements both the ``Crawler`` and ``Transport`` protocols. Uses a full table
scan with pagination (``ExclusiveStartKey``).  For large tables, prefer
``filter_expression`` to reduce scan cost.

Schema is inferred by sampling up to ``schema_sample_size`` items and running
union-schema inference.  One schema chunk is emitted per table.

Usage::

    from chonk.transports import DynamoDBCrawler
    from chonk.loader import DocumentLoader

    # Credentials from environment / IAM role
    crawler = DynamoDBCrawler(table="articles", region="us-east-1")
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl("dynamodb://articles", crawler=crawler)

    # Explicit credentials + projection
    crawler = DynamoDBCrawler(
        table="kb-docs",
        region="us-west-2",
        aws_access_key_id="AKIA...",
        aws_secret_access_key="secret",
        projection_expression="docId, title, body, #ts",
        expression_attribute_names={"#ts": "timestamp"},
        field_aliases={"pk": "id", "sk": "sort_key"},
    )

    # NER vocabulary
    vocab = crawler.get_field_names()  # after crawl()

Requires: boto3>=1.26  (``pip install boto3``)
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

from ._protocol import FetchResult
from ._schema_infer import collect_field_paths, infer_schema_text

_log = logging.getLogger(__name__)

_SCHEMA_SAMPLE_SIZE = 500


def _default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)


class DynamoDBCrawler:
    """Crawl an AWS DynamoDB table and index each item as a JSON FetchResult.

    Args:
        table:                      DynamoDB table name.
        region:                     AWS region (e.g. ``us-east-1``).
        aws_access_key_id:          Optional explicit AWS credentials.
        aws_secret_access_key:      Optional explicit AWS credentials.
        aws_session_token:          Optional STS session token.
        endpoint_url:               Override endpoint (useful for DynamoDB Local).
        filter_expression:          ``boto3`` ``Attr`` filter expression.
        projection_expression:      Comma-separated attribute names to return.
        expression_attribute_names: Substitution map for reserved words.
        page_size:                  Items per scan page (default 100).
        schema_sample_size:         Max items sampled for schema inference (default 500).
        field_aliases:              Map raw attribute names to normalized names for NER.
    """

    SCHEME = "dynamodb"

    def __init__(
        self,
        table: str,
        region: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpoint_url: str | None = None,
        filter_expression: Any = None,
        projection_expression: str | None = None,
        expression_attribute_names: dict | None = None,
        page_size: int = 100,
        schema_sample_size: int = _SCHEMA_SAMPLE_SIZE,
        field_aliases: dict[str, str] | None = None,
    ):
        self._table = table
        self._region = region
        self._ak = aws_access_key_id
        self._sk = aws_secret_access_key
        self._token = aws_session_token
        self._endpoint = endpoint_url
        self._filter_expr = filter_expression
        self._projection = projection_expression
        self._attr_names = expression_attribute_names
        self._page_size = page_size
        self._schema_sample_size = schema_sample_size
        self._field_aliases: dict[str, str] = field_aliases or {}
        self._cache: dict[str, FetchResult] = {}
        self._known_fields: set[str] = set()

    # ── Transport protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith(f"{self.SCHEME}://")

    def fetch(self, uri: str, **__) -> FetchResult:
        if uri not in self._cache:
            raise KeyError(f"DynamoDBCrawler: unknown URI {uri!r} — call crawl() first")
        return self._cache[uri]

    # ── Crawler protocol ──────────────────────────────────────────────────────

    def crawl(self, uri: str = "", **__) -> list[str]:
        """Scan the DynamoDB table, cache all items, and emit a schema chunk.

        Returns:
            List of ``dynamodb://{table}/{item_key}`` URIs followed by one schema URI.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for DynamoDBCrawler. "
                "Install with: pip install boto3"
            )

        kwargs: dict = dict(region_name=self._region)
        if self._ak:
            kwargs["aws_access_key_id"] = self._ak
            kwargs["aws_secret_access_key"] = self._sk
        if self._token:
            kwargs["aws_session_token"] = self._token
        if self._endpoint:
            kwargs["endpoint_url"] = self._endpoint

        ddb = boto3.resource("dynamodb", **kwargs)
        table = ddb.Table(self._table)

        scan_kwargs: dict = {"Limit": self._page_size}
        if self._filter_expr is not None:
            scan_kwargs["FilterExpression"] = self._filter_expr
        if self._projection:
            scan_kwargs["ProjectionExpression"] = self._projection
        if self._attr_names:
            scan_kwargs["ExpressionAttributeNames"] = self._attr_names

        self._cache.clear()
        self._known_fields.clear()
        total = 0
        last_key = None
        sample: list[dict] = []

        while True:
            if last_key:
                scan_kwargs["ExclusiveStartKey"] = last_key
            response = table.scan(**scan_kwargs)
            for item in response.get("Items", []):
                # Coerce Decimals for JSON serialization
                item_serializable = json.loads(json.dumps(item, default=_default))
                if len(sample) < self._schema_sample_size:
                    sample.append(item_serializable)
                item_key = str(next(iter(item.values()), total))
                doc_uri = f"{self.SCHEME}://{self._table}/{item_key}"
                payload = {
                    "_source_meta": {
                        "type": "dynamodb",
                        "table": self._table,
                        "region": self._region,
                        "endpoint": self._endpoint,
                    },
                    **item_serializable,
                }
                self._cache[doc_uri] = FetchResult(
                    data=json.dumps(payload, indent=2).encode("utf-8"),
                    detected_mime="application/json",
                    source_path=f"{self._table}/{item_key}",
                )
                total += 1

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break

        # ── Schema chunk ──────────────────────────────────────────────────────
        self._known_fields.update(collect_field_paths(sample))
        schema_text = infer_schema_text(
            sample, self._table,
            total_docs=total, sample_size=len(sample),
        )
        schema_uri = f"{self.SCHEME}://{self._table}/_schema"
        self._cache[schema_uri] = FetchResult(
            data=schema_text.encode("utf-8"),
            detected_mime="text/plain",
            source_path=f"{self._table}/_schema",
        )

        _log.info("DynamoDBCrawler: indexed %d item(s) from %r", total, self._table)
        return list(self._cache.keys())

    # ── NER vocabulary ────────────────────────────────────────────────────────

    def get_field_names(self) -> list[str]:
        """Return normalized attribute names and table name.

        Call after ``crawl()``.  Aliases defined in ``field_aliases`` are applied.
        """
        normalized = {self._field_aliases.get(f, f) for f in self._known_fields}
        normalized.add(self._table)
        return sorted(normalized)
