# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4b1e7d6a-2c9f-4a3e-8d5b-1f0c7e2a9b64
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Shared structural types used across the public API.

`EmbedModel` documents the embedding contract as either a model name (str) or a
SentenceTransformer-compatible object exposing `encode(...)`. Replaces the
information-erasing `str | Any` annotation so the Go migration has an explicit
`Embedder` interface to port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """A SentenceTransformer-compatible embedding model."""

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = ...,
        show_progress_bar: bool = ...,
        batch_size: int = ...,
    ) -> np.ndarray: ...


EmbedModel = str | Embedder
