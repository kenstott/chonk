# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 7d2f4a1e-5c8b-4e3f-b9a0-1c6d2e8f3b7a
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Live integration tests for GmailCrawler against kennethstott@gmail.com.

Requires OAuth2 credentials in the environment (GOOGLE_EMAIL_CLIENT_ID,
GOOGLE_EMAIL_CLIENT_SECRET) and a valid token at ~/.chonk/gmail_token.json.

On the first run a browser window opens for the OAuth2 consent flow.

    pytest tests/integration/test_gmail.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ENV_FILE = Path(__file__).parents[2] / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE)
    except ImportError:
        pass

_CREDS = pytest.mark.skipif(
    not os.environ.get("GOOGLE_EMAIL_CLIENT_ID"),
    reason="GOOGLE_EMAIL_CLIENT_ID not set — skipping live Gmail tests",
)


@_CREDS
class TestGmailCrawlerLive:
    def test_can_handle(self):
        from chonk.transports import GmailCrawler

        c = GmailCrawler()
        assert c.can_handle("gmail://me/INBOX")
        assert not c.can_handle("https://example.com")
        assert not c.can_handle("imap://user:pass@host/INBOX")

    def test_crawl_inbox_returns_uris(self):
        from chonk.transports import GmailCrawler

        c = GmailCrawler()
        uris = c.crawl("gmail://me/INBOX", limit=5)
        assert len(uris) > 0, "expected at least one message in INBOX"
        assert all(u.startswith("gmsg://") for u in uris)
        assert c.can_handle(uris[0])
        print(f"\n  crawl(INBOX, limit=5) → {len(uris)} URI(s)")
        print(f"  first URI: {uris[0]}")

    def test_crawl_with_query(self):
        from chonk.transports import GmailCrawler

        c = GmailCrawler()
        uris = c.crawl("gmail://me/INBOX", query="is:unread", limit=10)
        assert isinstance(uris, list)
        print(f"\n  crawl(query='is:unread', limit=10) → {len(uris)} URI(s)")

    def test_crawl_sent(self):
        from chonk.transports import GmailCrawler

        c = GmailCrawler()
        uris = c.crawl("gmail://me/SENT", limit=5)
        assert isinstance(uris, list)
        print(f"\n  crawl(SENT, limit=5) → {len(uris)} URI(s)")

    def test_fetch_message_returns_result(self):
        from chonk.transports import GmailCrawler

        c = GmailCrawler()
        uris = c.crawl("gmail://me/INBOX", limit=1)
        assert uris, "need at least one message to fetch"
        result = c.fetch(uris[0])
        assert result.data, "message data must not be empty"
        assert result.detected_mime == "text/plain"
        assert result.source_path
        print(f"\n  fetch({uris[0]}) → {len(result.data)} bytes")
        print(f"  source_path: {result.source_path}")
        print(f"  preview: {result.data.decode('utf-8', errors='replace')[:200]}")

    def test_fetch_contains_headers(self):
        from chonk.transports import GmailCrawler

        c = GmailCrawler()
        uris = c.crawl("gmail://me/INBOX", limit=1)
        assert uris
        result = c.fetch(uris[0])
        text = result.data.decode("utf-8", errors="replace")
        assert any(h in text for h in ("From:", "Subject:", "Date:")), (
            "expected at least one email header in fetched text"
        )

    def test_fetch_cached(self):
        from chonk.transports import GmailCrawler

        c = GmailCrawler()
        uris = c.crawl("gmail://me/INBOX", limit=1)
        assert uris
        r1 = c.fetch(uris[0])
        r2 = c.fetch(uris[0])
        assert r1 is r2, "second fetch should return cached result"

    def test_crawl_limit_respected(self):
        from chonk.transports import GmailCrawler

        c = GmailCrawler()
        uris = c.crawl("gmail://me/INBOX", limit=3)
        assert len(uris) <= 3

    def test_load_crawl_produces_chunks(self):
        from chonk import DocumentLoader
        from chonk.transports import GmailCrawler

        crawler = GmailCrawler()
        loader = DocumentLoader(extra_transports=[crawler])
        chunks = loader.load_crawl("gmail://me/INBOX", crawler=crawler, limit=3)
        assert isinstance(chunks, list)
        assert len(chunks) > 0, "expected chunks from Gmail messages"
        print(f"\n  load_crawl(INBOX, limit=3) → {len(chunks)} chunk(s)")
        if chunks:
            print(f"  first chunk document_name: {chunks[0].document_name}")
            print(f"  first chunk preview: {chunks[0].content[:120]}")
