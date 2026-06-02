# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b2291ee0-eb77-4b88-957b-d313933ce89d
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""RelationshipIndex — in-memory directed graph of SVOTriples."""

from __future__ import annotations

from collections import defaultdict

from ._svo import SVOTriple


class RelationshipIndex:
    """Bidirectional index over SVOTriples.

    Supports lookup by subject (forward traversal) and by object (reverse
    traversal), with optional verb filtering on each.
    """

    def __init__(self) -> None:
        self._by_subject: dict[str, list[SVOTriple]] = defaultdict(list)
        self._by_object: dict[str, list[SVOTriple]] = defaultdict(list)

    def add(self, triple: SVOTriple) -> None:
        self._by_subject[triple.subject_id].append(triple)
        self._by_object[triple.object_id].append(triple)

    def get_objects(self, subject_id: str, verb: str | None = None) -> list[SVOTriple]:
        """Return triples where subject_id is the subject.

        If verb is given, restrict to that verb.
        """
        triples = self._by_subject.get(subject_id, [])
        if verb is not None:
            triples = [t for t in triples if t.verb == verb]
        return triples

    def get_subjects(self, object_id: str, verb: str | None = None) -> list[SVOTriple]:
        """Return triples where object_id is the object.

        If verb is given, restrict to that verb.
        """
        triples = self._by_object.get(object_id, [])
        if verb is not None:
            triples = [t for t in triples if t.verb == verb]
        return triples

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_subject.values())

    def save_to_db(self, con, incremental: bool = False) -> int:
        """Upsert all triples into svo_triples table. Returns count written.

        Args:
            con: Open DuckDB connection.
            incremental: If True, append rows without clearing the table first.
                Used for periodic checkpointing during extraction.
        """
        con.execute("""
            CREATE TABLE IF NOT EXISTS svo_triples (
                chunk_id    VARCHAR,
                subject_id  VARCHAR NOT NULL,
                verb        VARCHAR NOT NULL,
                object_id   VARCHAR NOT NULL,
                confidence  FLOAT   NOT NULL DEFAULT 1.0,
                namespace   VARCHAR,
                description TEXT    NOT NULL DEFAULT ''
            )
        """)
        con.execute("ALTER TABLE svo_triples ADD COLUMN IF NOT EXISTS namespace VARCHAR")
        con.execute("ALTER TABLE svo_triples ADD COLUMN IF NOT EXISTS description TEXT")
        if not incremental:
            con.execute("DELETE FROM svo_triples")
        rows = []
        for triples in self._by_subject.values():
            for t in triples:
                ns_row = con.execute(
                    "SELECT namespace FROM embeddings WHERE chunk_id = ?", [t.source_chunk_id]
                ).fetchone()
                namespace = ns_row[0] if ns_row else None
                rows.append(
                    (
                        t.source_chunk_id,
                        t.subject_id,
                        t.verb,
                        t.object_id,
                        t.confidence,
                        namespace,
                        t.description,
                    )
                )
        if rows:
            con.executemany("INSERT INTO svo_triples VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    @classmethod
    def load_from_db(cls, con, namespaces: list[str] | None = None) -> RelationshipIndex:
        """Load RelationshipIndex from svo_triples table. Returns empty index if table absent."""
        idx = cls()
        try:
            _view_exists = (
                con.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = 'all_svo_triples'"
                ).fetchone()[0]
                > 0
            )
            table = "all_svo_triples" if _view_exists else "svo_triples"
            if namespaces is not None:
                placeholders = ", ".join(["?" for _ in namespaces])
                rows = con.execute(
                    f"SELECT chunk_id, subject_id, verb, object_id, confidence, "
                    f"COALESCE(description, '') FROM {table} WHERE namespace IN ({placeholders})",
                    namespaces,
                ).fetchall()
            else:
                rows = con.execute(
                    f"SELECT chunk_id, subject_id, verb, object_id, confidence, "
                    f"COALESCE(description, '') FROM {table}"
                ).fetchall()
        except Exception:
            return idx
        for chunk_id, subject_id, verb, object_id, confidence, description in rows:
            idx.add(
                SVOTriple(
                    subject_id=subject_id,
                    verb=verb,
                    object_id=object_id,
                    confidence=confidence,
                    source_chunk_id=chunk_id,
                    description=description,
                )
            )
        return idx
