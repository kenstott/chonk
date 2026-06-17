# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 371c62a4-e9a3-4290-904f-25093f471a7f
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CommunityIndexBuilder — background hot-swap builder for CommunityIndex."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .._versioning import VersionedRef
from ._index import CommunityIndex

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CommunityIndexBuilder:
    """Builds CommunityIndex in a background thread; atomically hot-swaps when done.

    Uses :class:`VersionedRef` for versioned access — consumers can detect when
    the index has been refreshed by comparing :attr:`ref.version`.

    Typical use::

        builder = CommunityIndexBuilder()
        builder.build(chunk_ids, vecs, n_levels=3)

        # Serve current index while next version builds:
        idx = builder.ref.current   # None until first build completes

        # Corpus updated — kick off a rebuild:
        builder.build(new_chunk_ids, new_vecs, n_levels=3)

        # Optionally block until done:
        builder.wait()
        print(builder.ref.version)   # incremented
    """

    def __init__(self, initial: CommunityIndex | None = None) -> None:
        self.ref: VersionedRef[CommunityIndex] = VersionedRef(initial)
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Convenience passthrough
    # ------------------------------------------------------------------

    @property
    def index(self) -> CommunityIndex | None:
        """Current index. Shorthand for ``self.ref.current``."""
        return self.ref.current

    @property
    def is_building(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        chunk_ids: list[str],
        content_vecs: np.ndarray,
        *,
        on_complete: Callable[[CommunityIndex], None] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Start a background build. Hot-swaps the ref when complete.

        Args:
            chunk_ids: Chunk IDs to cluster.
            content_vecs: (n, dim) float32 embeddings.
            on_complete: Optional callback fired with the new index (on build thread).
            **kwargs: Forwarded verbatim to :meth:`CommunityIndex.build`.
        """

        def _run() -> None:
            try:
                idx = CommunityIndex.build(chunk_ids, content_vecs, **kwargs)
                new_version = self.ref.update(idx)
                logger.debug(
                    "CommunityIndex version %d ready (%d levels, %d chunks)",
                    new_version,
                    idx.level_count(),
                    idx.chunk_count(),
                )
                if on_complete is not None:
                    on_complete(idx)
            except Exception:
                logger.exception("CommunityIndexBuilder background build failed")

        t = threading.Thread(target=_run, daemon=True, name="community-index-builder")
        self._thread = t
        t.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the running build finishes (or timeout elapses).

        Returns:
            True if build is done (or never started), False if timed out.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True
