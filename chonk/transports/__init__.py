# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 4b2ec232-173b-4c2f-bf13-5bed0a1172d0
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Document transport backends."""

from ._cassandra import CassandraCrawler
from ._cosmos import CosmosCrawler
from ._crawler_protocol import Crawler
from ._db_schema import DatabaseSchemaCrawler
from ._directory_crawler import DirectoryCrawler
from ._dynamodb import DynamoDBCrawler
from ._elasticsearch import ElasticsearchCrawler
from ._firestore import FirestoreCrawler
from ._ftp import FtpTransport
from ._github import GitHubCrawler
from ._gmail import GmailCrawler
from ._http import HttpTransport
from ._imap import ImapTransport
from ._import_crawler import ImportCrawler
from ._local import LocalTransport
from ._mongodb import MongoCrawler
from ._protocol import FetchResult, Transport
from ._s3 import S3Transport
from ._sftp import SftpTransport
from ._sharepoint import SharePointCrawler
from ._solr import SolrCrawler
from ._sql_query import SqlQueryTransport
from ._sqlalchemy import SqlAlchemyTransport
from ._web_crawler import WebCrawler

_DEFAULT_REGISTRY = [
    LocalTransport(),
    HttpTransport(),
    S3Transport(),
    FtpTransport(),
    SftpTransport(),
]


def detect_transport(uri: str) -> Transport:
    """Return the first registered transport that can handle the given URI."""
    for t in _DEFAULT_REGISTRY:
        if t.can_handle(uri):
            return t
    raise ValueError(f"No transport found for URI: {uri!r}")


__all__ = [
    "Transport",
    "FetchResult",
    "Crawler",
    "CassandraCrawler",
    "LocalTransport",
    "HttpTransport",
    "S3Transport",
    "FtpTransport",
    "SftpTransport",
    "WebCrawler",
    "DirectoryCrawler",
    "GitHubCrawler",
    "DatabaseSchemaCrawler",
    "SharePointCrawler",
    "GmailCrawler",
    "ImportCrawler",
    "SqlAlchemyTransport",
    "SqlQueryTransport",
    "ImapTransport",
    "MongoCrawler",
    "ElasticsearchCrawler",
    "SolrCrawler",
    "DynamoDBCrawler",
    "FirestoreCrawler",
    "CosmosCrawler",
    "detect_transport",
]
