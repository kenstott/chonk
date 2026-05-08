# Copyright (c) 2025 Kenneth Stott. MIT License.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""RelationshipIndexBuilder — background hot-swap builder for RelationshipIndex."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from .._versioning import VersionedRef
from ._index import RelationshipIndex
from ._svo import SVOTriple

logger = logging.getLogger(__name__)


class RelationshipIndexBuilder:
    """Builds a RelationshipIndex in a background thread; atomically hot-swaps when done.

    Uses :class:`VersionedRef` so consumers can detect stale reads by version number.

    Typical use::

        extractor = SVOExtractor(llm=my_llm)
        builder = RelationshipIndexBuilder()

        # Build from (chunk_id, text) pairs — runs extraction + indexing in background:
        builder.build(extractor, chunks)

        # Or supply pre-extracted triples directly:
        builder.build_from_triples(triples)

        # Serve current index while rebuild runs:
        rel_index = builder.ref.current

        # Block until done:
        builder.wait()
        print(builder.ref.version)
    """

    def __init__(self, initial: RelationshipIndex | None = None) -> None:
        self.ref: VersionedRef[RelationshipIndex] = VersionedRef(initial)
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Convenience passthrough
    # ------------------------------------------------------------------

    @property
    def index(self) -> RelationshipIndex | None:
        """Current index. Shorthand for ``self.ref.current``."""
        return self.ref.current

    @property
    def is_building(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Build from extractor
    # ------------------------------------------------------------------

    def build(
        self,
        extractor,
        chunks: list[tuple[str, str]],
        *,
        on_complete: Callable[[RelationshipIndex], None] | None = None,
    ) -> None:
        """Extract SVO triples from chunks and build a new RelationshipIndex in background.

        Args:
            extractor: An :class:`SVOExtractor` instance.
            chunks: List of ``(chunk_id, chunk_text)`` pairs to extract from.
            on_complete: Optional callback fired with the new index (on build thread).
        """
        def _run() -> None:
            try:
                idx = RelationshipIndex()
                for chunk_id, text in chunks:
                    for triple in extractor.extract(chunk_id, text):
                        idx.add(triple)
                new_version = self.ref.update(idx)
                logger.debug("RelationshipIndex version %d ready (%d triples)",
                             new_version, len(idx))
                if on_complete is not None:
                    on_complete(idx)
            except Exception:
                logger.exception("RelationshipIndexBuilder background build failed")

        t = threading.Thread(target=_run, daemon=True, name="relationship-index-builder")
        self._thread = t
        t.start()

    # ------------------------------------------------------------------
    # Build from pre-extracted triples
    # ------------------------------------------------------------------

    def build_from_triples(
        self,
        triples: list[SVOTriple],
        *,
        on_complete: Callable[[RelationshipIndex], None] | None = None,
    ) -> None:
        """Build a new RelationshipIndex from pre-extracted triples in background.

        Useful when triples have already been extracted and persisted, and you
        only need to reconstruct the in-memory index (e.g., on startup).

        Args:
            triples: Pre-extracted :class:`SVOTriple` objects.
            on_complete: Optional callback fired with the new index (on build thread).
        """
        def _run() -> None:
            try:
                idx = RelationshipIndex()
                for triple in triples:
                    idx.add(triple)
                new_version = self.ref.update(idx)
                logger.debug("RelationshipIndex version %d ready (%d triples from pre-extracted)",
                             new_version, len(idx))
                if on_complete is not None:
                    on_complete(idx)
            except Exception:
                logger.exception("RelationshipIndexBuilder (from_triples) background build failed")

        t = threading.Thread(target=_run, daemon=True, name="relationship-index-builder")
        self._thread = t
        t.start()

    # ------------------------------------------------------------------
    # Wait
    # ------------------------------------------------------------------

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the running build finishes (or timeout elapses).

        Returns:
            True if build is done (or never started), False if timed out.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True
