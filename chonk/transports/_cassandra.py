# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d0a45f51-d332-4fd9-8054-e11240a89488
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CassandraCrawler — index Cassandra keyspace schemas and query results.

Implements both the ``Crawler`` and ``Transport`` protocols. Pass the same
instance as both ``crawler=`` and in ``extra_transports=``.

Four capabilities (mirroring DatabaseSchemaCrawler + SqlQueryTransport):

1. **Schema chunks** — keyspace/table/column metadata indexed as text chunks,
   seeding NER vocabulary with table and column names.
2. **NER vocabulary** — ``get_table_meta()`` returns ``list[TableMeta]`` ready
   for ``SchemaVocabBuilder.add_tables()``. ``get_field_names()`` returns a
   flat list of column, table, and keyspace names.
3. **Entity queries** — CQL SELECTs whose results seed NER vocabulary via
   ``get_entity_vocab()`` (same role as ``NerPipeline.add_from_db()``).
4. **Dataset queries** — CQL SELECTs whose results are chunked as documents
   (same role as ``SqlQueryTransport`` / ``loader.load_from_db()``).

All chunks carry ``source_detail`` annotations: host, port, keyspace, table.

Usage::

    from chonk.transports import CassandraCrawler
    from chonk.loader import DocumentLoader
    from chonk.ner import NerPipeline

    crawler = CassandraCrawler(
        contact_points=["10.0.0.1"],
        keyspace="my_keyspace",
        dataset_queries={
            "patient_notes": "SELECT patient_id, note_text FROM clinical_notes",
        },
        entity_queries={
            "physician": "SELECT full_name FROM physicians",
            "drug":      "SELECT drug_name FROM formulary",
        },
    )
    loader = DocumentLoader(extra_transports=[crawler])
    chunks = loader.load_crawl("cassandra://10.0.0.1/my_keyspace", crawler=crawler)

    # NER with Cassandra table vocab + entity vocab
    pipeline = NerPipeline(db_enrich=True, spacy_entities=True)
    pipeline.add_tables(crawler.get_table_meta())
    for entity_type, names in crawler.get_entity_vocab().items():
        pipeline.add_entities(names, entity_type=entity_type)

Requires: cassandra-driver>=3.25  (``pip install cassandra-driver``)
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

from ._protocol import FetchResult

_log = logging.getLogger(__name__)

_SYSTEM_KEYSPACES = frozenset(
    {"system", "system_auth", "system_distributed", "system_schema", "system_traces"}
)


def _rows_to_csv(column_names: list[str], rows: list[Any]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(column_names)
    for row in rows:
        writer.writerow([str(v) if v is not None else "" for v in row])
    return buf.getvalue().encode("utf-8")


class CassandraCrawler:
    """Crawl Cassandra keyspace schemas and optional query datasets.

    Implements both ``Crawler`` and ``Transport`` — pass the same instance to both::

        crawler = CassandraCrawler(contact_points=["localhost"], keyspace="myks")
        loader = DocumentLoader(extra_transports=[crawler])
        chunks = loader.load_crawl("cassandra://localhost/myks", crawler=crawler)

    Args:
        contact_points:   One or more Cassandra hosts.
        port:             CQL native transport port (default 9042).
        keyspace:         Index this keyspace only.  ``None`` = all non-system keyspaces.
        keyspaces:        Whitelist of keyspace names (overrides ``keyspace``).
        username:         Plain-text authentication username.
        password:         Plain-text authentication password.
        include_views:    Include materialized views in schema chunks (default True).
        dataset_queries:  ``{document_name: CQL}`` — query results become chunked
                          documents (equivalent to ``loader.load_from_db()`` queries).
        entity_queries:   ``{entity_type: CQL}`` — first column of each result row
                          seeds NER vocabulary (equivalent to
                          ``NerPipeline.add_from_db()`` queries).
        local_dc:         Datacenter name for ``DCAwareRoundRobinPolicy``; improves
                          latency in multi-DC clusters.  ``None`` = default policy.
    """

    SCHEME = "cassandra"

    def __init__(
        self,
        contact_points: list[str],
        port: int = 9042,
        keyspace: str | None = None,
        keyspaces: list[str] | None = None,
        username: str | None = None,
        password: str | None = None,
        include_views: bool = True,
        dataset_queries: dict[str, str] | None = None,
        entity_queries: dict[str, str] | None = None,
        local_dc: str | None = None,
    ) -> None:
        self._contact_points = contact_points
        self._port = port
        self._keyspace_filter: set[str] | None = (
            set(keyspaces) if keyspaces else ({keyspace} if keyspace else None)
        )
        self._username = username
        self._password = password
        self._include_views = include_views
        self._dataset_queries: dict[str, str] = dataset_queries or {}
        self._entity_queries: dict[str, str] = entity_queries or {}
        self._local_dc = local_dc

        self._cache: dict[str, FetchResult] = {}
        self._table_meta: list[Any] = []  # list[TableMeta] — populated after crawl
        self._entity_vocab: dict[str, list[str]] = {}  # entity_type → names
        self._known_keyspaces: list[str] = []

    # ── Transport protocol ────────────────────────────────────────────────────

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("cassandra://")

    def fetch(self, uri: str, **__: object) -> FetchResult:
        if uri not in self._cache:
            raise KeyError(f"CassandraCrawler: unknown URI {uri!r} — call crawl() first")
        return self._cache[uri]

    # ── Crawler protocol ──────────────────────────────────────────────────────

    def crawl(self, uri: str = "", **__: object) -> list[str]:
        """Connect to Cassandra, index schema and dataset query results.

        Args:
            uri:  ``cassandra://<host>/<keyspace>`` — contact point / keyspace
                  may override constructor values if both are present.

        Returns:
            List of ``cassandra://`` URIs — schema URIs + dataset chunk URIs.
        """
        try:
            from cassandra.cluster import Cluster
            from cassandra.policies import DCAwareRoundRobinPolicy
        except ImportError as exc:
            raise ImportError(
                "cassandra-driver is required for CassandraCrawler. "
                "Install with: pip install cassandra-driver"
            ) from exc

        contact_points = self._contact_points
        if uri.startswith("cassandra://"):
            from urllib.parse import urlparse

            parsed = urlparse(uri)
            if parsed.hostname:
                contact_points = [parsed.hostname]
            if parsed.path.lstrip("/") and self._keyspace_filter is None:
                self._keyspace_filter = {parsed.path.lstrip("/")}

        kwargs: dict[str, object] = {"contact_points": contact_points, "port": self._port}
        if self._local_dc:
            kwargs["load_balancing_policy"] = DCAwareRoundRobinPolicy(local_dc=self._local_dc)

        auth_provider = None
        if self._username:
            from cassandra.auth import PlainTextAuthProvider

            auth_provider = PlainTextAuthProvider(
                username=self._username, password=self._password or ""
            )
            kwargs["auth_provider"] = auth_provider

        cluster = Cluster(**kwargs)
        session = cluster.connect()

        self._cache.clear()
        self._table_meta.clear()
        self._entity_vocab.clear()
        self._known_keyspaces.clear()

        try:
            self._crawl_schema(session, contact_points[0])
            self._crawl_datasets(session, contact_points[0])
            self._collect_entity_vocab(session)
        finally:
            cluster.shutdown()

        _log.info(
            "CassandraCrawler: indexed %d URI(s) across %d keyspace(s)",
            len(self._cache),
            len(self._known_keyspaces),
        )
        return list(self._cache.keys())

    # ── NER vocabulary ────────────────────────────────────────────────────────

    def get_table_meta(self) -> list[Any]:
        """Return ``list[TableMeta]`` for all crawled tables.

        Pass to ``SchemaVocabBuilder.add_tables()`` or
        ``NerPipeline.add_tables()`` to seed NER vocabulary with table and
        column names.  Call after ``crawl()``.
        """
        return list(self._table_meta)

    def get_field_names(self) -> list[str]:
        """Return flat list of column names, table names, and keyspace names.

        Suitable for direct use with ``NerPipeline.add_entities()``.
        Call after ``crawl()``.
        """
        names: set[str] = set(self._known_keyspaces)
        for tm in self._table_meta:
            names.add(tm.name)
            for col in tm.columns:
                names.add(col.name)
        return sorted(names)

    def get_entity_vocab(self) -> dict[str, list[str]]:
        """Return ``{entity_type: [name, ...]}`` from ``entity_queries`` results.

        Each query's first column becomes NER entity names of the given type.
        Pass to ``NerPipeline.add_entities(names, entity_type=...)``.
        Call after ``crawl()``.
        """
        return dict(self._entity_vocab)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _connect_session(self, session: Any, keyspace: str) -> Any:  # noqa: ANN401
        """Return session connected to *keyspace* (no-op if already connected)."""
        session.set_keyspace(keyspace)
        return session

    def _keyspaces_to_crawl(self, session: Any) -> list[str]:  # noqa: ANN401
        rows = session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        all_ks = [r.keyspace_name for r in rows if r.keyspace_name not in _SYSTEM_KEYSPACES]
        if self._keyspace_filter:
            return [k for k in all_ks if k in self._keyspace_filter]
        return all_ks

    def _crawl_schema(self, session: Any, host: str) -> None:  # noqa: ANN401
        from chonk.schema import ColumnMeta, TableMeta

        keyspaces = self._keyspaces_to_crawl(session)
        self._known_keyspaces = list(keyspaces)

        for ks in keyspaces:
            tables = list(
                session.execute(
                    "SELECT table_name, comment FROM system_schema.tables WHERE keyspace_name = %s",
                    (ks,),
                )
            )
            if self._include_views:
                views = list(
                    session.execute(
                        "SELECT view_name, where_clause FROM system_schema.views "
                        "WHERE keyspace_name = %s",
                        (ks,),
                    )
                )
            else:
                views = []

            all_tables = [(r.table_name, getattr(r, "comment", None), False) for r in tables]
            all_tables += [(r.view_name, getattr(r, "where_clause", None), True) for r in views]

            for table_name, comment, is_view in all_tables:
                col_rows = list(
                    session.execute(
                        "SELECT column_name, type, kind FROM system_schema.columns "
                        "WHERE keyspace_name = %s AND table_name = %s",
                        (ks, table_name),
                    )
                )
                columns = []
                col_lines = []
                for col in col_rows:
                    kind = getattr(col, "kind", "")
                    is_pk = kind in ("partition_key", "clustering")
                    cm = ColumnMeta(
                        name=col.column_name,
                        data_type=str(col.type),
                        is_primary_key=is_pk,
                    )
                    columns.append(cm)
                    pk_marker = (
                        " [partition key]"
                        if kind == "partition_key"
                        else (" [clustering]" if kind == "clustering" else "")
                    )
                    col_lines.append(f"  {col.column_name} {col.type}{pk_marker}")

                tm = TableMeta(
                    name=table_name,
                    schema_name=ks,
                    description=comment or None,
                    columns=columns,
                    source_db=host,
                )
                self._table_meta.append(tm)

                obj_type = "MATERIALIZED VIEW" if is_view else "TABLE"
                schema_text = (
                    f"Source: cassandra://{host}/{ks}/{table_name}\n"
                    f"Type: {obj_type}\n"
                    f"Keyspace: {ks}\n"
                    f"Table: {table_name}\n"
                    + (f"Comment: {comment}\n" if comment else "")
                    + "Columns:\n"
                    + "\n".join(col_lines)
                    + "\n"
                )

                schema_uri = f"cassandra://{host}/{ks}/{table_name}/_schema"
                self._cache[schema_uri] = FetchResult(
                    data=schema_text.encode("utf-8"),
                    detected_mime="text/plain",
                    source_path=f"{ks}/{table_name}/_schema",
                )

    def _crawl_datasets(self, session: Any, host: str) -> None:  # noqa: ANN401
        for doc_name, cql in self._dataset_queries.items():
            try:
                rows = list(session.execute(cql))
                if not rows:
                    continue
                col_names = list(rows[0]._fields)
                csv_bytes = _rows_to_csv(col_names, rows)
                uri = f"cassandra://{host}/query/{doc_name}"
                self._cache[uri] = FetchResult(
                    data=csv_bytes,
                    detected_mime="text/csv",
                    source_path=doc_name,
                )
                _log.debug("CassandraCrawler: dataset %r → %d rows", doc_name, len(rows))
            except Exception as exc:
                _log.warning("CassandraCrawler: dataset query %r failed: %s", doc_name, exc)

    def _collect_entity_vocab(self, session: Any) -> None:  # noqa: ANN401
        for entity_type, cql in self._entity_queries.items():
            try:
                rows = list(session.execute(cql))
                if not rows:
                    continue
                # First column of each row becomes an entity name
                names = [str(row[0]) for row in rows if row[0] is not None]
                if names:
                    self._entity_vocab[entity_type] = names
                    _log.debug(
                        "CassandraCrawler: entity_query %r → %d names",
                        entity_type,
                        len(names),
                    )
            except Exception as exc:
                _log.warning("CassandraCrawler: entity_query %r failed: %s", entity_type, exc)

    @staticmethod
    def cassandra_provenance(
        contact_points: list[str], port: int, keyspace: str | None
    ) -> dict[str, object]:
        """Return source annotation dict for ``chunk.source_detail``."""
        info: dict[str, object] = {
            "db_dialect": "cassandra",
            "db_host": contact_points[0] if contact_points else "",
            "db_port": port,
        }
        if keyspace:
            info["db_name"] = keyspace
        return info
