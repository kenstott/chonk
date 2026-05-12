# Copyright (c) 2025 Kenneth Stott. MIT License.
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

    def save_to_db(self, con) -> int:
        """Upsert all triples into svo_triples table. Returns count written."""
        con.execute("""
            CREATE TABLE IF NOT EXISTS svo_triples (
                chunk_id    VARCHAR,
                subject_id  VARCHAR NOT NULL,
                verb        VARCHAR NOT NULL,
                object_id   VARCHAR NOT NULL,
                confidence  FLOAT   NOT NULL DEFAULT 1.0,
                namespace   VARCHAR
            )
        """)
        con.execute("ALTER TABLE svo_triples ADD COLUMN IF NOT EXISTS namespace VARCHAR")
        con.execute("DELETE FROM svo_triples")
        rows = []
        for triples in self._by_subject.values():
            for t in triples:
                ns_row = con.execute(
                    "SELECT namespace FROM embeddings WHERE chunk_id = ?", [t.source_chunk_id]
                ).fetchone()
                namespace = ns_row[0] if ns_row else None
                rows.append((t.source_chunk_id, t.subject_id, t.verb, t.object_id, t.confidence, namespace))
        if rows:
            con.executemany("INSERT INTO svo_triples VALUES (?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    @classmethod
    def load_from_db(cls, con, namespaces: list[str] | None = None) -> RelationshipIndex:
        """Load RelationshipIndex from svo_triples table. Returns empty index if table absent."""
        idx = cls()
        try:
            if namespaces is not None:
                placeholders = ", ".join(["?" for _ in namespaces])
                rows = con.execute(
                    f"SELECT chunk_id, subject_id, verb, object_id, confidence FROM svo_triples WHERE namespace IN ({placeholders})",
                    namespaces,
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT chunk_id, subject_id, verb, object_id, confidence FROM svo_triples"
                ).fetchall()
        except Exception:
            return idx
        for chunk_id, subject_id, verb, object_id, confidence in rows:
            idx.add(
                SVOTriple(
                    subject_id=subject_id,
                    verb=verb,
                    object_id=object_id,
                    confidence=confidence,
                    source_chunk_id=chunk_id,
                )
            )
        return idx
