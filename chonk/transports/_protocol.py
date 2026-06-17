# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 9ccf08ef-ae1c-4864-a753-a99ab0300447
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Transport protocol and FetchResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class FetchOptions:
    """Options forwarded to Transport.fetch()."""

    sql: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    profile: str | None = None
    region: str | None = None
    endpoint_url: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    key_path: str | None = None


@dataclass
class FetchResult:
    """Result of fetching a document via any transport."""

    data: bytes
    detected_mime: str | None = None
    source_path: str | None = None


@runtime_checkable
class Transport(Protocol):
    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult: ...
    def can_handle(self, uri: str) -> bool: ...
