# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 8c3e1f7a-2b4d-4f9e-a0c5-6d8b1e3f5a9c
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Live integration tests for ImapTransport against FastMail.

IMAP host is derived from SMTP_HOST by replacing the 'smtp.' prefix with
'imap.' — FastMail uses smtp.fastmail.com / imap.fastmail.com.

Skipped automatically when SMTP_USER is not in the environment.

    pytest tests/integration/test_imap.py -v -s
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import pytest

_ENV_FILE = Path(__file__).parents[2] / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE)
    except ImportError:
        pass

_CREDS = pytest.mark.skipif(
    not os.environ.get("SMTP_USER"),
    reason="SMTP_USER not set — skipping live IMAP tests",
)


def _imap_host() -> str:
    """Derive IMAP host from SMTP_HOST (smtp.X → imap.X)."""
    smtp_host = os.environ.get("SMTP_HOST", "")
    if smtp_host.startswith("smtp."):
        return "imap." + smtp_host[5:]
    return smtp_host


def _imap_uri(mailbox: str = "INBOX", search: str | None = None, limit: int | None = None) -> str:
    user = quote(os.environ["SMTP_USER"], safe="")
    password = quote(os.environ["SMTP_PASSWORD"], safe="")
    host = _imap_host()
    uri = f"imaps://{user}:{password}@{host}:993/{mailbox}"
    params = []
    if search:
        params.append(f"search={quote(search)}")
    if limit:
        params.append(f"limit={limit}")
    if params:
        uri += "?" + "&".join(params)
    return uri


@_CREDS
class TestImapTransportLive:
    def test_can_handle(self):
        from chonk.transports import ImapTransport

        t = ImapTransport()
        assert t.can_handle("imap://user:pass@host/INBOX")
        assert t.can_handle("imaps://user:pass@host:993/INBOX")
        assert not t.can_handle("https://example.com")
        assert not t.can_handle("s3://bucket/key")

    def test_imap_host_derived_correctly(self):
        assert _imap_host() == "imap.fastmail.com"

    def test_fetch_messages_returns_results(self):
        from chonk.transports import ImapTransport

        t = ImapTransport()
        results = list(t.fetch_messages(_imap_uri(limit=5), limit=5))
        assert len(results) > 0, "expected at least one message in INBOX"
        for r in results:
            assert r.data, "message data must not be empty"
            assert r.detected_mime == "message/rfc822"
            assert r.source_path
        print(f"\n  fetched {len(results)} message(s) from INBOX")
        print(f"  first source_path: {results[0].source_path}")
        print(f"  first message size: {len(results[0].data)} bytes")

    def test_fetch_single_message(self):
        from chonk.transports import ImapTransport

        t = ImapTransport()
        result = t.fetch(_imap_uri())
        assert result.data
        assert result.detected_mime == "message/rfc822"
        print(f"\n  fetch() → {len(result.data)} bytes, source: {result.source_path}")

    def test_fetch_with_search_unseen(self):
        from chonk.transports import ImapTransport

        t = ImapTransport()
        # UNSEEN may be empty — just confirm it doesn't crash
        results = list(t.fetch_messages(_imap_uri(search="UNSEEN", limit=10), limit=10))
        assert isinstance(results, list)
        print(f"\n  UNSEEN messages: {len(results)}")

    def test_fetch_with_limit(self):
        from chonk.transports import ImapTransport

        t = ImapTransport()
        results = list(t.fetch_messages(_imap_uri(limit=3), limit=3))
        assert len(results) <= 3

    def test_message_is_valid_rfc822(self):
        import email

        from chonk.transports import ImapTransport

        t = ImapTransport()
        result = t.fetch(_imap_uri())
        msg = email.message_from_bytes(result.data)
        assert msg["Subject"] is not None or msg["From"] is not None, (
            "expected at least Subject or From header"
        )
        print(f"\n  From: {msg.get('From', '(none)')}")
        print(f"  Subject: {msg.get('Subject', '(none)')}")
        print(f"  Date: {msg.get('Date', '(none)')}")

    def test_load_imap_produces_chunks(self):
        from chonk import DocumentLoader

        loader = DocumentLoader()
        chunks = loader.load_imap(_imap_uri(), limit=3)
        assert isinstance(chunks, list)
        assert len(chunks) > 0, "expected chunks from INBOX messages"
        print(f"\n  load_imap(limit=3) → {len(chunks)} chunk(s)")
        if chunks:
            print(f"  first chunk document_name: {chunks[0].document_name}")
            print(f"  first chunk preview: {chunks[0].content[:120]}")

    def test_load_imap_with_attachments(self):
        from chonk import DocumentLoader

        loader = DocumentLoader()
        # include_attachments=True — confirm it doesn't crash even if no attachments
        chunks = loader.load_imap(_imap_uri(), limit=5, include_attachments=True)
        assert isinstance(chunks, list)
        print(f"\n  load_imap(include_attachments=True, limit=5) → {len(chunks)} chunk(s)")

    def test_wrong_password_raises(self):
        from chonk.transports import ImapTransport

        t = ImapTransport()
        bad_uri = _imap_uri().replace(
            quote(os.environ["SMTP_PASSWORD"], safe=""), "wrongpassword"
        )
        with pytest.raises(Exception):
            list(t.fetch_messages(bad_uri, limit=1))
