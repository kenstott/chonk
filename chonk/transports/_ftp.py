# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: ba16be9a-aa2f-4b4a-a326-39ba4260edab
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""FTP transport — stdlib only."""

from __future__ import annotations

from ftplib import FTP  # nosec B402 — FTP transport intentionally implements FTP protocol
from io import BytesIO
from typing import cast
from urllib.parse import urlparse

from ._protocol import FetchOptions, FetchResult


class FtpTransport:
    """Fetch documents via FTP."""

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("ftp://")

    def fetch(self, uri: str, options: FetchOptions | None = None) -> FetchResult:
        parsed = urlparse(uri)
        host = parsed.hostname
        port = cast("int", (options.port if options else None) or parsed.port or 21)
        remote_path = parsed.path
        opt_user = options.username if options else None
        username = cast("str", opt_user or parsed.username or "anonymous")
        password = cast("str", (options.password if options else None) or parsed.password or "")

        if host is None:
            raise ValueError(f"FtpTransport: could not parse host from URI: {uri!r}")
        ftp = FTP()  # nosec B321
        ftp.connect(host, port)
        ftp.login(username, password)

        buf = BytesIO()
        ftp.retrbinary(f"RETR {remote_path}", buf.write)
        ftp.quit()

        return FetchResult(
            data=buf.getvalue(),
            detected_mime=None,
            source_path=uri,
        )
