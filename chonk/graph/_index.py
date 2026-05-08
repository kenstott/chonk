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
