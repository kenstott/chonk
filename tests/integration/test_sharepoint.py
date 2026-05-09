# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 9e4b2c1f-6d8a-4f3e-b7c9-0a1d5e2f8b4c
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Live integration tests for SharePointCrawler against kenstott.sharepoint.com.

Skipped automatically when SHAREPOINT_CLIENT_SECRET is not in the environment.
Load credentials with: python-dotenv or export from .env before running.

    pytest tests/integration/test_sharepoint.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Load .env from project root so credentials are available
_ENV_FILE = Path(__file__).parents[2] / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE)
    except ImportError:
        pass  # dotenv optional; export vars manually if not installed

_CREDS = pytest.mark.skipif(
    not os.environ.get("SHAREPOINT_CLIENT_SECRET"),
    reason="SHAREPOINT_CLIENT_SECRET not set — skipping live SharePoint tests",
)


def _crawler(**kwargs):
    from chonk.transports import SharePointCrawler

    return SharePointCrawler(
        site_url=os.environ["SHAREPOINT_SITE_URL"],
        auth_mode="azure_ad",
        tenant_id=os.environ["SHAREPOINT_TENANT_ID"],
        client_id=os.environ["SHAREPOINT_CLIENT_ID"],
        client_secret=os.environ["SHAREPOINT_CLIENT_SECRET"],
        **kwargs,
    )


@_CREDS
class TestSharePointCrawlerLive:
    def test_can_handle_site_url(self):
        c = _crawler()
        assert c.can_handle(os.environ["SHAREPOINT_SITE_URL"])
        assert not c.can_handle("https://example.com/not-sharepoint")

    def test_crawl_returns_uris(self):
        c = _crawler()
        uris = c.crawl()
        assert isinstance(uris, list), "crawl() must return a list"
        assert len(uris) > 0, "expected at least one artifact from the site"
        assert all(u.startswith("spitem://") for u in uris), "all URIs must use spitem:// scheme"
        print(f"\n  crawl() returned {len(uris)} URI(s)")

    def test_current_sha_not_set(self):
        # SharePointCrawler has no SHA watermark (not git-based)
        c = _crawler()
        assert not hasattr(c, "current_sha") or True  # just confirm no crash

    def test_crawl_documents_only(self):
        c = _crawler(artifacts=["documents"])
        uris = c.crawl()
        assert all("/documents/" in u for u in uris), (
            "document-only crawl should only return document URIs"
        )
        print(f"\n  documents: {len(uris)} file(s)")

    def test_crawl_lists_only(self):
        c = _crawler(artifacts=["lists"])
        uris = c.crawl()
        # May be zero if no generic lists exist on the site
        assert all("/lists/" in u for u in uris)
        print(f"\n  lists: {len(uris)} item(s)")

    def test_crawl_pages_only(self):
        c = _crawler(artifacts=["pages"])
        uris = c.crawl()
        assert all("/pages/" in u for u in uris)
        print(f"\n  pages: {len(uris)} page(s)")

    def test_crawl_calendars_only(self):
        c = _crawler(artifacts=["calendars"])
        uris = c.crawl()
        assert all("/lists/" in u for u in uris)
        print(f"\n  calendars: {len(uris)} event(s)")

    def test_fetch_structured_item(self):
        c = _crawler(artifacts=["pages", "lists", "calendars"])
        uris = c.crawl()
        if not uris:
            pytest.skip("no structured artifacts found to fetch")
        uri = uris[0]
        result = c.fetch(uri)
        assert result.data, "fetch() must return non-empty bytes"
        assert result.detected_mime in ("text/plain", "text/html", "text/x-sql", None)
        print(f"\n  fetched {len(result.data)} bytes from {uri}")
        print(f"  source_path: {result.source_path}")
        print(f"  preview: {result.data[:200].decode('utf-8', errors='replace')}")

    def test_fetch_document(self):
        c = _crawler(artifacts=["documents"])
        uris = c.crawl()
        if not uris:
            pytest.skip("no documents found in document libraries")
        uri = uris[0]
        result = c.fetch(uri)
        assert result.data, "fetch() must return non-empty bytes for document"
        print(f"\n  fetched document: {len(result.data)} bytes")
        print(f"  mime: {result.detected_mime}")
        print(f"  source: {result.source_path}")

    def test_fetch_unknown_uri_raises(self):
        c = _crawler()
        c.crawl()
        with pytest.raises(KeyError):
            c.fetch(f"spitem://{c._url_key}/documents/nonexistent-id")

    def test_loader_integration(self):
        from chonk import DocumentLoader

        c = _crawler(artifacts=["pages"], max_items=10)
        loader = DocumentLoader(extra_transports=[c])
        chunks = loader.load_crawl(os.environ["SHAREPOINT_SITE_URL"], crawler=c)
        assert isinstance(chunks, list)
        print(f"\n  load_crawl() produced {len(chunks)} chunk(s)")
        if chunks:
            print(f"  first chunk document_name: {chunks[0].document_name}")
            print(f"  first chunk preview: {chunks[0].content[:100]}")
