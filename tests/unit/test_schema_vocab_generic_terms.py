# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Generic table and column names are not made into entities.

A column called ``id``, ``name``, or ``created_at`` appears in nearly every table
and in most text. As an entity it would link almost every chunk to every other,
so it carries no signal and drowns the names that do.
"""

from __future__ import annotations

from chonk.ner import SchemaVocabBuilder
from chonk.ner._schema import normalize_schema_term
from chonk.ner._schema_vocab import DEFAULT_GENERIC_SCHEMA_TERMS
from chonk.schema import ColumnMeta, TableMeta


def _table(name: str, *columns: str) -> TableMeta:
    return TableMeta(name=name, columns=[ColumnMeta(name=c, data_type="VARCHAR") for c in columns])


def _entity_names(builder: SchemaVocabBuilder) -> set[str]:
    return {alias for _eid, alias, _ns in builder.namespaced_entities()}


def _canon(*terms: str) -> set[str]:
    """Entity names as namespaced_entities() spells them (normalised, singular)."""
    return {normalize_schema_term(t, to_singular=True) for t in terms}


def test_generic_column_names_are_not_entities():
    builder = SchemaVocabBuilder().add_tables(
        [_table("patent_assignees", "id", "name", "created_at", "Status", "assignee_name")]
    )

    names = _entity_names(builder)

    assert _canon("assignee_name", "patent_assignees") <= names
    assert not names & _canon("id", "name", "created_at", "Status")


def test_generic_names_are_matched_in_any_spelling():
    # camelCase, SCREAMING_SNAKE, and plural spellings normalise to a generic term
    builder = SchemaVocabBuilder().add_tables(
        [_table("filings", "createdAt", "UPDATED_AT", "descriptions", "filing_date")]
    )

    names = _entity_names(builder)

    assert _canon("filing_date") <= names
    assert not names & _canon("createdAt", "UPDATED_AT", "descriptions")


def test_generic_names_are_filtered_from_ddl_and_matcher():
    builder = SchemaVocabBuilder().add_sql(
        "CREATE TABLE cves (\n  id INTEGER,\n  vendor_name VARCHAR(80),\n  description TEXT\n);"
    )

    matches = builder.build().match("The description lists the vendor name and the id.")
    matched = {normalize_schema_term(m.display_name) for m in matches}

    assert "vendor name" in matched
    assert not matched & {"description", "id"}


def test_a_generic_word_inside_a_specific_name_is_kept():
    builder = SchemaVocabBuilder().add_tables([_table("orders", "customer_id", "order_status")])

    names = _entity_names(builder)

    assert _canon("customer_id", "order_status") <= names


def test_generic_filter_can_be_replaced_or_disabled():
    tables = [_table("orders", "id", "region")]

    assert _canon("id") <= _entity_names(
        SchemaVocabBuilder(generic_terms=frozenset()).add_tables(tables)
    )
    assert not _canon("region") & _entity_names(
        SchemaVocabBuilder(generic_terms=frozenset({"region"})).add_tables(tables)
    )


def test_glossary_terms_are_not_filtered():
    # a glossary entry is a deliberate choice, even when it is a common word
    builder = SchemaVocabBuilder().add_business_terms(["Status"])

    assert _canon("Status") <= _entity_names(builder)


def test_default_list_is_lowercase_normalised_terms():
    assert all(t == t.strip().lower() and "_" not in t for t in DEFAULT_GENERIC_SCHEMA_TERMS)
