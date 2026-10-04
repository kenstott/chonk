# Copyright (c) 2025 Kenneth Stott. MIT License.

"""singularize() uses a dictionary (WordNet), not suffix rules.

Stripping a trailing "s" turns ``status`` into ``statu``; ``inflect`` assumes its
input is plural and turns ``address`` into ``addres``. A singular word must come
back unchanged, and an irregular plural must come back as its real singular.
"""

from __future__ import annotations

import pytest

from chonk.ner._schema import normalize_schema_term
from chonk.ner._singular import singularize


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # regular plurals
        ("items", "item"),
        ("categories", "category"),
        ("companies", "company"),
        ("addresses", "address"),
        ("statuses", "status"),
        ("aliases", "alias"),
        ("filings", "filing"),
        ("patents", "patent"),
        # already singular, ending in s
        ("status", "status"),
        ("address", "address"),
        ("class", "class"),
        ("process", "process"),
        ("bus", "bus"),
        ("canvas", "canvas"),
        ("series", "series"),
        ("news", "news"),
        # irregular plurals
        ("indices", "index"),
        ("vertices", "vertex"),
        ("matrices", "matrix"),
        ("analyses", "analysis"),
        ("children", "child"),
        # words used as singular in data work
        ("data", "data"),
        ("metadata", "metadata"),
        ("criteria", "criteria"),
        ("media", "media"),
        # unknown to the dictionary: a regular plural only if it pluralises back
        ("cves", "cve"),
        ("lei", "lei"),
        ("cik", "cik"),
    ],
)
def test_singularize(word: str, expected: str) -> None:
    assert singularize(word) == expected


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("order_statuses", "order status"),
        ("order_status", "order status"),
        ("mailing_address", "mailing address"),
        ("patent_assignees", "patent assignee"),
        ("vulnerability_indices", "vulnerability index"),
        ("orderItems", "order item"),
    ],
)
def test_schema_terms_singularise_the_head_noun(term: str, expected: str) -> None:
    assert normalize_schema_term(term, to_singular=True) == expected
