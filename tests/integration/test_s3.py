# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4f7a2e9b-1c3d-4e8f-a6b0-5d2c9f1e7a3b
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Live integration tests for S3Transport and DirectoryCrawler (S3) against
chinook-athena-us-west-1.

Skipped automatically when AWS_ACCESS_KEY_ID_TEST is not in the environment.

    pytest tests/integration/test_s3.py -v -s
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

# Inject test credentials into boto3-visible env vars for the duration of the test session
if os.environ.get("AWS_ACCESS_KEY_ID_TEST"):
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["AWS_ACCESS_KEY_ID_TEST"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_SECRET_ACCESS_KEY_TEST"]
    os.environ["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION_TEST", "us-west-1")

_CREDS = pytest.mark.skipif(
    not os.environ.get("AWS_ACCESS_KEY_ID_TEST"),
    reason="AWS_ACCESS_KEY_ID_TEST not set — skipping live S3 tests",
)

_BUCKET = "chinook-athena-us-west-1"
_PREFIX = "data/"
_KNOWN_KEY = "data/artist/artist.csv"
_KNOWN_URI = f"s3://{_BUCKET}/{_KNOWN_KEY}"


@_CREDS
class TestS3TransportLive:
    def test_can_handle(self):
        from chonk.transports import S3Transport

        t = S3Transport()
        assert t.can_handle("s3://bucket/key")
        assert t.can_handle("s3a://bucket/key")
        assert not t.can_handle("https://example.com")
        assert not t.can_handle("/local/path")

    def test_fetch_csv(self):
        from chonk.transports import S3Transport

        t = S3Transport()
        result = t.fetch(_KNOWN_URI)
        assert result.data, "expected non-empty bytes"
        assert result.source_path == _KNOWN_URI
        text = result.data.decode("utf-8")
        assert "ArtistId" in text or "Artist" in text
        print(f"\n  fetched {len(result.data)} bytes from {_KNOWN_URI}")
        print(f"  mime: {result.detected_mime}")
        print(f"  preview: {text[:120]}")

    def test_fetch_produces_chunks(self):
        from chonk import DocumentLoader
        from chonk.transports import S3Transport

        loader = DocumentLoader(extra_transports=[S3Transport()])
        chunks = loader.load(_KNOWN_URI)
        assert len(chunks) > 0
        print(f"\n  {_KNOWN_URI} → {len(chunks)} chunk(s)")
        print(f"  document_name: {chunks[0].document_name}")

    def test_fetch_nonexistent_raises(self):
        from chonk.transports import S3Transport

        t = S3Transport()
        with pytest.raises(Exception):
            t.fetch(f"s3://{_BUCKET}/nonexistent/path/that/does/not/exist.csv")


@_CREDS
class TestS3DirectoryCrawlerLive:
    def test_crawl_prefix_returns_uris(self):
        from chonk.transports import DirectoryCrawler

        c = DirectoryCrawler(extensions=[".csv"])
        uris = c.crawl(f"s3://{_BUCKET}/{_PREFIX}")
        assert len(uris) > 0
        assert all(u.startswith("s3://") for u in uris)
        assert all(u.endswith(".csv") for u in uris)
        print(f"\n  crawl(s3://{_BUCKET}/{_PREFIX}) → {len(uris)} CSV(s)")
        for u in uris:
            print(f"    {u}")

    def test_crawl_extension_filter(self):
        from chonk.transports import DirectoryCrawler

        csv_crawler = DirectoryCrawler(extensions=[".csv"])
        all_crawler = DirectoryCrawler()
        csv_uris = csv_crawler.crawl(f"s3://{_BUCKET}/queries/")
        all_uris = all_crawler.crawl(f"s3://{_BUCKET}/queries/")
        # queries/ has .csv and .csv.metadata files; .metadata not in default extensions
        assert len(csv_uris) <= len(all_uris)
        print(f"\n  queries/: {len(csv_uris)} CSV(s), {len(all_uris)} total")

    def test_crawl_max_files(self):
        from chonk.transports import DirectoryCrawler

        c = DirectoryCrawler(max_files=3)
        uris = c.crawl(f"s3://{_BUCKET}/")
        assert len(uris) <= 3

    def test_crawl_nonrecursive(self):
        from chonk.transports import DirectoryCrawler

        c = DirectoryCrawler(recursive=False)
        # data/ has only subdirectories at the top level — no direct files
        uris = c.crawl(f"s3://{_BUCKET}/{_PREFIX}")
        assert all(_PREFIX in u for u in uris)
        print(f"\n  non-recursive data/: {len(uris)} file(s)")

    def test_load_directory_s3(self):
        from chonk import DocumentLoader

        loader = DocumentLoader()
        chunks = loader.load_directory(
            f"s3://{_BUCKET}/{_PREFIX}",
            extensions=[".csv"],
        )
        assert len(chunks) > 0
        doc_names = {c.document_name for c in chunks}
        assert any("artist" in n for n in doc_names)
        assert any("album" in n for n in doc_names)
        print(f"\n  load_directory(s3://{_BUCKET}/{_PREFIX}) → {len(chunks)} chunk(s)")
        print(f"  documents: {sorted(doc_names)}")
