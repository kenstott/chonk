# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: ce55af9b-d962-4314-9956-f83c1bb12fbd
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""VersionedRef — thread-safe versioned reference with stage/promote semantics.

Supports two update patterns:

  1. Immediate swap::

        ref.update(new_value)          # atomically replaces current, version++

  2. Stage → promote (background build)::

        ref.stage(new_value)           # set pending; current unchanged
        # ... wait for validation, or just promote when ready ...
        ref.promote()                  # atomically activates pending, version++
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class VersionedRef(Generic[T]):
    """Thread-safe versioned holder for any object.

    The *version* counter increments on every :meth:`update` or :meth:`promote`
    call, providing a cheap way to detect staleness without comparing values.

    Args:
        initial: Optional starting value. Counts as version 0 if supplied,
                 version -1 (unset) if None.
    """

    def __init__(self, initial: T | None = None) -> None:
        self._current: T | None = initial
        self._pending: T | None = None
        self._version: int = 0 if initial is not None else -1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def current(self) -> T | None:
        """Active value. Thread-safe read."""
        with self._lock:
            return self._current

    @property
    def version(self) -> int:
        """Monotonically increasing version counter. -1 means never set."""
        with self._lock:
            return self._version

    @property
    def has_pending(self) -> bool:
        """True if a staged value is waiting to be promoted."""
        with self._lock:
            return self._pending is not None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update(self, value: T) -> int:
        """Atomically replace current value. Returns new version number."""
        with self._lock:
            self._current = value
            self._pending = None
            self._version += 1
            return self._version

    def stage(self, value: T) -> None:
        """Stage a pending value without activating it.

        The current value is unchanged until :meth:`promote` is called.
        Calling :meth:`stage` again overwrites any previously staged value.
        """
        with self._lock:
            self._pending = value

    def promote(self) -> int:
        """Activate the staged value as current. Returns new version number.

        Raises:
            RuntimeError: If no value has been staged.
        """
        with self._lock:
            if self._pending is None:
                raise RuntimeError("promote() called with nothing staged")
            self._current = self._pending
            self._pending = None
            self._version += 1
            return self._version

    def discard_staged(self) -> None:
        """Discard staged value without promoting."""
        with self._lock:
            self._pending = None
