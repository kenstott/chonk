# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Unit tests for LLMClient protocol and SVOExtractor."""

import json

import pytest

from chonk.graph import VERB_SET, LLMClient, SVOExtractor

# ---------------------------------------------------------------------------
# Stub LLM clients
# ---------------------------------------------------------------------------

class StubLLM:
    """Returns a fixed JSON payload."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, prompt: str) -> str:
        return self._payload


class EchoLLM:
    """Captures prompts for inspection."""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._payload


def _good_json(triples: list[dict]) -> str:
    return json.dumps(triples)


# ---------------------------------------------------------------------------
# LLMClient protocol
# ---------------------------------------------------------------------------

class TestLLMClientProtocol:
    def test_stub_satisfies_protocol(self):
        assert isinstance(StubLLM("[]"), LLMClient)

    def test_object_without_complete_fails(self):
        class Bad:
            pass
        assert not isinstance(Bad(), LLMClient)

    def test_extractor_rejects_non_client(self):
        with pytest.raises(TypeError):
            SVOExtractor("not_a_client")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SVOExtractor.extract
# ---------------------------------------------------------------------------

class TestSVOExtractorExtract:
    def _extractor(self, payload: str) -> SVOExtractor:
        return SVOExtractor(StubLLM(payload))

    def test_returns_valid_triple(self):
        payload = _good_json([
            {"subject_id": "orders", "verb": "references", "object_id": "customers", "confidence": 0.95}
        ])
        results = self._extractor(payload).extract("orders.customer_id FK customers.id")
        assert len(results) == 1
        t = results[0]
        assert t.subject_id == "orders"
        assert t.verb == "references"
        assert t.object_id == "customers"
        assert t.confidence == 0.95

    def test_chunk_id_stored_on_triple(self):
        payload = _good_json([
            {"subject_id": "a", "verb": "type_of", "object_id": "b", "confidence": 0.8}
        ])
        results = self._extractor(payload).extract("text", chunk_id="chunk_42")
        assert results[0].source_chunk_id == "chunk_42"

    def test_no_chunk_id_is_none(self):
        payload = _good_json([
            {"subject_id": "a", "verb": "type_of", "object_id": "b", "confidence": 0.8}
        ])
        results = self._extractor(payload).extract("text")
        assert results[0].source_chunk_id is None

    def test_empty_array_returns_empty(self):
        results = self._extractor("[]").extract("text")
        assert results == []

    def test_invalid_json_returns_empty(self):
        results = self._extractor("not json at all").extract("text")
        assert results == []

    def test_markdown_fences_stripped(self):
        payload = "```json\n" + _good_json([
            {"subject_id": "x", "verb": "governs", "object_id": "y", "confidence": 0.7}
        ]) + "\n```"
        results = self._extractor(payload).extract("text")
        assert len(results) == 1

    def test_invalid_verb_row_dropped(self):
        payload = _good_json([
            {"subject_id": "a", "verb": "invented_verb", "object_id": "b", "confidence": 0.9},
            {"subject_id": "c", "verb": "type_of", "object_id": "d", "confidence": 0.6},
        ])
        results = self._extractor(payload).extract("text")
        assert len(results) == 1
        assert results[0].subject_id == "c"

    def test_missing_field_row_dropped(self):
        payload = _good_json([
            {"subject_id": "a", "verb": "type_of", "confidence": 0.9},  # no object_id
            {"subject_id": "c", "verb": "contains", "object_id": "d", "confidence": 0.5},
        ])
        results = self._extractor(payload).extract("text")
        assert len(results) == 1

    def test_confidence_out_of_range_row_dropped(self):
        payload = _good_json([
            {"subject_id": "a", "verb": "type_of", "object_id": "b", "confidence": 1.5},
        ])
        results = self._extractor(payload).extract("text")
        assert results == []

    def test_non_list_response_returns_empty(self):
        results = self._extractor('{"subject_id": "a"}').extract("text")
        assert results == []

    def test_multiple_triples_all_returned(self):
        payload = _good_json([
            {"subject_id": "invoice", "verb": "references", "object_id": "customer", "confidence": 1.0},
            {"subject_id": "invoice_line", "verb": "part_of", "object_id": "invoice", "confidence": 1.0},
            {"subject_id": "invoice", "verb": "governed_by", "object_id": "GAAP", "confidence": 0.6},
        ])
        results = self._extractor(payload).extract("text")
        # "governed_by" is not in VERB_SET — dropped
        assert len(results) == 2

    def test_all_valid_verbs_accepted(self):
        rows = [
            {"subject_id": "a", "verb": v, "object_id": "b", "confidence": 0.5}
            for v in VERB_SET
        ]
        results = self._extractor(_good_json(rows)).extract("text")
        assert len(results) == len(VERB_SET)


# ---------------------------------------------------------------------------
# SVOExtractor.extract_batch
# ---------------------------------------------------------------------------

class TestSVOExtractorBatch:
    def test_batch_aggregates_results(self):
        call_count = 0

        class CountingLLM:
            def complete(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                return _good_json([
                    {"subject_id": f"s{call_count}", "verb": "type_of",
                     "object_id": "entity", "confidence": 0.8}
                ])

        extractor = SVOExtractor(CountingLLM())
        results = extractor.extract_batch([
            ("text one", "c1"),
            ("text two", "c2"),
            ("text three", "c3"),
        ])
        assert len(results) == 3
        assert call_count == 3

    def test_batch_preserves_chunk_ids(self):
        extractor = SVOExtractor(StubLLM(_good_json([
            {"subject_id": "x", "verb": "depends_on", "object_id": "y", "confidence": 0.9}
        ])))
        results = extractor.extract_batch([("text", "my_chunk")])
        assert results[0].source_chunk_id == "my_chunk"

    def test_empty_batch_returns_empty(self):
        extractor = SVOExtractor(StubLLM("[]"))
        assert extractor.extract_batch([]) == []


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------

class TestPromptContent:
    def test_prompt_contains_all_verbs(self):
        llm = EchoLLM("[]")
        extractor = SVOExtractor(llm)
        extractor.extract("some text")
        assert len(llm.prompts) == 1
        prompt = llm.prompts[0]
        for verb in VERB_SET:
            assert verb in prompt, f"verb {verb!r} missing from prompt"

    def test_prompt_contains_chunk_id(self):
        llm = EchoLLM("[]")
        SVOExtractor(llm).extract("text", chunk_id="sentinel_id_xyz")
        assert "sentinel_id_xyz" in llm.prompts[0]

    def test_prompt_contains_text(self):
        llm = EchoLLM("[]")
        SVOExtractor(llm).extract("unique_passage_content_abc123")
        assert "unique_passage_content_abc123" in llm.prompts[0]


# ---------------------------------------------------------------------------
# Override / customisation
# ---------------------------------------------------------------------------

class TestSVOExtractorOverrides:
    def test_custom_verb_set_accepted(self):
        custom_verbs = frozenset({"custom_rel", "another_rel"})
        payload = _good_json([
            {"subject_id": "a", "verb": "custom_rel", "object_id": "b", "confidence": 0.9}
        ])
        extractor = SVOExtractor(StubLLM(payload), verb_set=custom_verbs)
        results = extractor.extract("text")
        assert len(results) == 1
        assert results[0].verb == "custom_rel"

    def test_custom_verb_set_rejects_default_verbs(self):
        custom_verbs = frozenset({"custom_rel"})
        payload = _good_json([
            {"subject_id": "a", "verb": "type_of", "object_id": "b", "confidence": 0.9}
        ])
        extractor = SVOExtractor(StubLLM(payload), verb_set=custom_verbs)
        assert extractor.extract("text") == []

    def test_custom_verb_set_appears_in_prompt(self):
        custom_verbs = frozenset({"benchmark_verb_xyz"})
        llm = EchoLLM("[]")
        SVOExtractor(llm, verb_set=custom_verbs).extract("text")
        assert "benchmark_verb_xyz" in llm.prompts[0]
        assert "type_of" not in llm.prompts[0]

    def test_custom_system_prompt_used(self):
        custom_tmpl = "Custom instructions. Verbs: {verbs}\n"
        llm = EchoLLM("[]")
        SVOExtractor(llm, system_prompt_template=custom_tmpl).extract("text")
        assert llm.prompts[0].startswith("Custom instructions.")

    def test_custom_user_template_used(self):
        custom_user = "ID={chunk_id} TEXT={text} -> triples"
        llm = EchoLLM("[]")
        SVOExtractor(llm, user_template=custom_user).extract("my_text", chunk_id="c99")
        assert "ID=c99" in llm.prompts[0]
        assert "TEXT=my_text" in llm.prompts[0]

    def test_default_verb_set_unchanged_without_override(self):
        extractor = SVOExtractor(StubLLM("[]"))
        assert extractor._verb_set is VERB_SET


# ---------------------------------------------------------------------------
# Entity-anchored extraction
# ---------------------------------------------------------------------------

def _ea_payload(triples=None, descriptions=None, aliases=None):
    return json.dumps({
        "triples": triples or [],
        "descriptions": descriptions or {},
        "aliases": aliases or {},
    })


class TestExtractEntityAnchored:
    def _entities(self):
        return [
            {"id": "CustomerRiskScore", "type": "db_column", "description": ""},
            {"id": "FactTable",         "type": "db_table",  "description": "Central fact table"},
            {"id": "CompliancePolicy",  "type": "concept",   "description": ""},
        ]

    def test_returns_triples_and_descriptions(self):
        payload = _ea_payload(
            triples=[{"subject_id": "CustomerRiskScore", "verb": "part_of",
                      "object_id": "FactTable", "confidence": 0.9}],
            descriptions={"CustomerRiskScore": "Risk score per customer", "CompliancePolicy": "Regulatory rules"},
        )
        triples, descs, _ = SVOExtractor(StubLLM(payload)).extract_entity_anchored(
            "some text", "c1", self._entities()
        )
        assert len(triples) == 1
        assert triples[0].subject_id == "CustomerRiskScore"
        assert triples[0].verb == "part_of"
        assert triples[0].object_id == "FactTable"
        assert descs["CustomerRiskScore"] == "Risk score per customer"
        # FactTable already had a description — not in new_descriptions
        assert "FactTable" not in descs

    def test_filters_triples_with_unknown_subject(self):
        payload = _ea_payload(
            triples=[{"subject_id": "UnknownEntity", "verb": "part_of",
                      "object_id": "FactTable", "confidence": 0.8}]
        )
        triples, _, _2 = SVOExtractor(StubLLM(payload)).extract_entity_anchored(
            "text", "c1", self._entities()
        )
        assert triples == []

    def test_filters_triples_with_invalid_verb(self):
        payload = _ea_payload(
            triples=[{"subject_id": "CustomerRiskScore", "verb": "invented_verb",
                      "object_id": "FactTable", "confidence": 0.9}]
        )
        triples, _, _2 = SVOExtractor(StubLLM(payload)).extract_entity_anchored(
            "text", "c1", self._entities()
        )
        assert triples == []

    def test_fewer_than_two_entities_returns_empty(self):
        triples, descs, aliases = SVOExtractor(StubLLM(_ea_payload())).extract_entity_anchored(
            "text", "c1", [{"id": "OnlyOne", "type": "concept", "description": ""}]
        )
        assert triples == []
        assert descs == {}
        assert aliases == {}

    def test_malformed_json_returns_empty(self):
        triples, descs, aliases = SVOExtractor(StubLLM("not json")).extract_entity_anchored(
            "text", "c1", self._entities()
        )
        assert triples == []
        assert descs == {}
        assert aliases == {}

    def test_entity_with_description_marked_in_prompt(self):
        captured = []
        class CaptureLLM:
            def complete(self, prompt):
                captured.append(prompt)
                return _ea_payload()
        SVOExtractor(CaptureLLM()).extract_entity_anchored("text", "c1", self._entities())
        assert "[✓] FactTable" in captured[0]
        assert "[ ] CustomerRiskScore" in captured[0]

    def test_source_chunk_id_set_on_triples(self):
        payload = _ea_payload(
            triples=[{"subject_id": "CustomerRiskScore", "verb": "part_of",
                      "object_id": "FactTable", "confidence": 0.7}]
        )
        triples, _, _2 = SVOExtractor(StubLLM(payload)).extract_entity_anchored(
            "text", "chunk-42", self._entities()
        )
        assert triples[0].source_chunk_id == "chunk-42"


# ---------------------------------------------------------------------------
# entity_descriptions Store methods
# ---------------------------------------------------------------------------

class TestEntityDescriptionsStore:
    def test_upsert_and_get(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.upsert_entity_description("ent_a", "A description", source="llm")
            result = store.get_entity_descriptions(["ent_a"])
        assert result["ent_a"] == "A description"

    def test_user_not_overwritten_by_llm(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.upsert_entity_description("e", "User desc", source="user")
            store.upsert_entity_description("e", "LLM desc",  source="llm")
            result = store.get_entity_descriptions(["e"])
        assert result["e"] == "User desc"

    def test_schema_not_overwritten_by_llm(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.upsert_entity_description("e", "Schema desc", source="schema")
            store.upsert_entity_description("e", "LLM desc",    source="llm")
            result = store.get_entity_descriptions(["e"])
        assert result["e"] == "Schema desc"

    def test_llm_overwrites_nothing_higher_priority(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.upsert_entity_description("e", "First LLM", source="llm")
            store.upsert_entity_description("e", "Second LLM", source="llm")
            result = store.get_entity_descriptions(["e"])
        # second llm call same priority — does NOT overwrite (>=)
        assert result["e"] == "First LLM"

    def test_user_overwrites_schema(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.upsert_entity_description("e", "Schema desc", source="schema")
            store.upsert_entity_description("e", "User override", source="user")
            result = store.get_entity_descriptions(["e"])
        assert result["e"] == "User override"

    def test_batch_upsert(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            n = store.upsert_entity_descriptions_batch(
                {"a": "desc a", "b": "desc b", "c": "desc c"}, source="llm"
            )
        assert n == 3

    def test_get_missing_entity_returns_empty(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            result = store.get_entity_descriptions(["nonexistent"])
        assert result == {}


# ---------------------------------------------------------------------------
# Entity aliases — extractor parses aliases from LLM response
# ---------------------------------------------------------------------------

class TestExtractEntityAnchoredAliases:
    def _entities(self):
        return [
            {"id": "CustomerRiskScore", "type": "db_column", "description": ""},
            {"id": "FactTable",         "type": "db_table",  "description": "Central fact table"},
        ]

    def test_aliases_parsed_from_response(self):
        payload = _ea_payload(
            aliases={"CustomerRiskScore": ["CRS", "Risk Score"], "FactTable": ["FT"]},
        )
        _, _, aliases = SVOExtractor(StubLLM(payload)).extract_entity_anchored(
            "text", "c1", self._entities()
        )
        assert aliases["CustomerRiskScore"] == ["CRS", "Risk Score"]
        assert aliases["FactTable"] == ["FT"]

    def test_unknown_entity_aliases_dropped(self):
        payload = _ea_payload(
            aliases={"UnknownEntity": ["UE"]},
        )
        _, _, aliases = SVOExtractor(StubLLM(payload)).extract_entity_anchored(
            "text", "c1", self._entities()
        )
        assert "UnknownEntity" not in aliases

    def test_empty_alias_list_dropped(self):
        payload = _ea_payload(
            aliases={"CustomerRiskScore": []},
        )
        _, _, aliases = SVOExtractor(StubLLM(payload)).extract_entity_anchored(
            "text", "c1", self._entities()
        )
        assert "CustomerRiskScore" not in aliases

    def test_no_aliases_key_returns_empty_dict(self):
        raw = json.dumps({"triples": [], "descriptions": {}})
        _, _, aliases = SVOExtractor(StubLLM(raw)).extract_entity_anchored(
            "text", "c1", self._entities()
        )
        assert aliases == {}


# ---------------------------------------------------------------------------
# Entity aliases — Store methods
# ---------------------------------------------------------------------------

class TestEntityAliasesStore:
    def test_add_and_get_alias(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.add_entity_alias("CRS", "CustomerRiskScore")
            result = store.get_entity_aliases("CustomerRiskScore")
        assert "CRS" in result

    def test_resolve_alias(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.add_entity_alias("CRS", "CustomerRiskScore")
            result = store.resolve_entity_alias("CRS")
        assert result == "CustomerRiskScore"

    def test_resolve_missing_alias_returns_none(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            result = store.resolve_entity_alias("nonexistent")
        assert result is None

    def test_llm_first_registration_wins(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.add_entity_alias("CRS", "CustomerRiskScore", source="llm")
            store.add_entity_alias("CRS", "OtherEntity", source="llm")
            result = store.resolve_entity_alias("CRS")
        assert result == "CustomerRiskScore"

    def test_user_source_overwrites_llm(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            store.add_entity_alias("CRS", "CustomerRiskScore", source="llm")
            store.add_entity_alias("CRS", "CorrectEntity", source="user")
            result = store.resolve_entity_alias("CRS")
        assert result == "CorrectEntity"

    def test_batch_aliases(self, tmp_path):
        from chonk.storage._store import Store
        with Store(tmp_path / "t.duckdb") as store:
            n = store.add_entity_aliases_batch(
                {"alias_a": "EntityA", "alias_b": "EntityB"}, source="llm"
            )
        assert n == 2


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

class TestTopLevelImport:
    def test_top_level_import(self):
        import chonk
        assert chonk.LLMClient is LLMClient
        assert chonk.SVOExtractor is SVOExtractor
