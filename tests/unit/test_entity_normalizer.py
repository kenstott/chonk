# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 1b4a75e7-8644-479d-b039-4cb3106c19d4
"""Unit tests for EntityNormalizer, normalize_entity, canonical_key."""

from __future__ import annotations

from chonk.ner import EntityNormalizer, canonical_key, normalize_entity

# ---------------------------------------------------------------------------
# normalize_entity — module-level function
# ---------------------------------------------------------------------------


class TestNormalizeEntity:
    def test_strips_leading_trailing_symbols(self):
        assert normalize_entity("(CustomerRisk)") == "CustomerRisk"
        assert normalize_entity("'entity'") == "entity"
        assert normalize_entity("[IBM]") == "IBM"

    def test_strips_whitespace(self):
        assert normalize_entity("  customer  ") == "customer"

    def test_singularizes_plural_noun(self):
        assert normalize_entity("policies") == "policy"
        assert normalize_entity("employees") == "employee"

    def test_singularizes_head_noun_in_phrase(self):
        assert normalize_entity("compliance policies") == "compliance policy"

    def test_preserves_acronym_in_phrase(self):
        result = normalize_entity("IBM employees")
        assert result == "IBM employee"

    def test_acronym_preserved(self):
        assert normalize_entity("IBM") == "IBM"
        assert normalize_entity("FASB") == "FASB"
        assert normalize_entity("API") == "API"

    def test_already_singular_unchanged(self):
        assert normalize_entity("policy") == "policy"
        assert normalize_entity("employee") == "employee"

    def test_exception_data_not_singularized(self):
        # "data" → "datum" would be wrong in data-domain context
        assert normalize_entity("data") == "data"
        assert normalize_entity("metadata") == "metadata"

    def test_snake_case_singularizes_last_token(self):
        assert normalize_entity("customer_risk_scores") == "customer_risk_score"

    def test_camel_case_singularizes_last_token(self):
        assert normalize_entity("CustomerRiskScores") == "CustomerRiskScore"

    def test_empty_string_returns_empty(self):
        assert normalize_entity("") == ""

    def test_only_symbols_returns_empty(self):
        assert normalize_entity("()[]{}") == ""

    def test_collapses_internal_whitespace(self):
        result = normalize_entity("customer   risk   score")
        assert "  " not in result

    def test_dots_in_acronym_stripped(self):
        # "U.S.A." → strip dots → "USA" → acronym preserved
        result = normalize_entity("U.S.A.")
        assert "." not in result


# ---------------------------------------------------------------------------
# canonical_key — lowercase dedup key
# ---------------------------------------------------------------------------


class TestCanonicalKey:
    def test_lowercases_result(self):
        assert canonical_key("IBM") == "ibm"
        assert canonical_key("CustomerRiskScore") == "customerriskscore"

    def test_deduplicates_plural_and_singular(self):
        assert canonical_key("policies") == canonical_key("policy")
        assert canonical_key("employees") == canonical_key("employee")

    def test_deduplicates_with_stripped_symbols(self):
        assert canonical_key("(CustomerRisk)") == canonical_key("CustomerRisk")


# ---------------------------------------------------------------------------
# EntityNormalizer class
# ---------------------------------------------------------------------------


class TestEntityNormalizer:
    def test_normalize_delegates_to_functions(self):
        n = EntityNormalizer()
        assert n.normalize("policies") == "policy"
        assert n.canonical_key("IBM") == "ibm"

    def test_extra_exceptions_respected(self):
        n = EntityNormalizer(extra_exceptions=frozenset({"criteria"}))
        # "criteria" already in SINGULAR_EXCEPTIONS but also works via extra
        assert n.normalize("criteria") == "criteria"

    def test_custom_exception_not_singularized(self):
        n = EntityNormalizer(extra_exceptions=frozenset({"stamina"}))
        assert n.normalize("stamina") == "stamina"


# ---------------------------------------------------------------------------
# set_entity_description — description stored on entities table
# ---------------------------------------------------------------------------


class TestSetEntityDescription:
    def _seed(self, store, eid: str) -> None:
        store.vector._conn.execute(
            "INSERT OR IGNORE INTO entities(id, name, display_name) VALUES (?, ?, ?)",
            [eid, eid, eid],
        )

    def test_set_and_get(self, tmp_path):
        from chonk.storage._store import Store

        with Store(tmp_path / "t.duckdb") as store:
            self._seed(store, "ent")
            store.set_entity_description("ent", "A description")
            result = store.get_entity_descriptions(["ent"])
        assert result["ent"] == "A description"

    def test_overwrites_previous(self, tmp_path):
        from chonk.storage._store import Store

        with Store(tmp_path / "t.duckdb") as store:
            self._seed(store, "ent")
            store.set_entity_description("ent", "First")
            store.set_entity_description("ent", "Second")
            result = store.get_entity_descriptions(["ent"])
        assert result["ent"] == "Second"
