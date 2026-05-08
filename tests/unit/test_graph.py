# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Unit tests for SVOTriple and RelationshipIndex (Phase 4.1)."""

import json
import pytest

from chonk.graph import SVOTriple, VERB_SET, RelationshipIndex, SVOExtractor


# ---------------------------------------------------------------------------
# SVOTriple
# ---------------------------------------------------------------------------

class TestSVOTriple:
    def test_valid_construction(self):
        t = SVOTriple("card_number", "type_of", "personal_data", 0.95)
        assert t.subject_id == "card_number"
        assert t.verb == "type_of"
        assert t.object_id == "personal_data"
        assert t.confidence == 0.95
        assert t.source_chunk_id is None

    def test_source_chunk_id_stored(self):
        t = SVOTriple("orders", "references", "customers", 1.0, source_chunk_id="chunk_001")
        assert t.source_chunk_id == "chunk_001"

    def test_invalid_verb_accepted_by_dataclass(self):
        # SVOTriple is a pure storage primitive; verb enforcement is SVOExtractor's job
        t = SVOTriple("a", "relates_to", "b", 0.9)
        assert t.verb == "relates_to"

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            SVOTriple("a", "type_of", "b", -0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            SVOTriple("a", "type_of", "b", 1.001)

    def test_confidence_boundary_values(self):
        SVOTriple("a", "governs", "b", 0.0)
        SVOTriple("a", "governs", "b", 1.0)

    def test_all_verbs_accepted(self):
        for verb in VERB_SET:
            t = SVOTriple("s", verb, "o", 0.5)
            assert t.verb == verb

    def test_verb_set_contents(self):
        # spot-check one from each category
        assert "type_of" in VERB_SET           # taxonomy
        assert "contains" in VERB_SET          # structure
        assert "references" in VERB_SET        # lineage
        assert "governs" in VERB_SET           # governance
        assert "manages" in VERB_SET           # ownership
        assert "equivalent_to" in VERB_SET     # equivalence
        assert "depends_on" in VERB_SET        # causation
        assert "produces" in VERB_SET          # data flow
        assert "located_in" in VERB_SET
        assert "used_for" in VERB_SET
        assert "member_of" in VERB_SET
        assert "inverse_of" in VERB_SET
        assert len(VERB_SET) == 48

    def test_removed_verbs_dropped_by_extractor(self):
        # Removed verbs are rejected by SVOExtractor (not the dataclass)
        removed = ("categorized_under", "inherits_from", "regulated_by",
                   "subject_to", "synonym_of", "replaces", "computed_from",
                   "authored_by", "joins_to")

        class _Stub:
            def __init__(self, verb): self._verb = verb
            def complete(self, p): return json.dumps([
                {"subject_id": "a", "verb": self._verb, "object_id": "b", "confidence": 0.9}
            ])

        for verb in removed:
            extractor = SVOExtractor(_Stub(verb))
            assert extractor.extract("text") == [], f"extractor should drop removed verb {verb!r}"


# ---------------------------------------------------------------------------
# RelationshipIndex
# ---------------------------------------------------------------------------

class TestRelationshipIndex:
    def _make_triple(self, subj, verb, obj, conf=0.9, chunk=None):
        return SVOTriple(subj, verb, obj, conf, source_chunk_id=chunk)

    def test_empty_index(self):
        idx = RelationshipIndex()
        assert len(idx) == 0
        assert idx.get_objects("x") == []
        assert idx.get_subjects("x") == []

    def test_add_and_get_objects(self):
        idx = RelationshipIndex()
        t = self._make_triple("card_number", "type_of", "personal_data")
        idx.add(t)
        result = idx.get_objects("card_number")
        assert result == [t]

    def test_add_and_get_subjects(self):
        idx = RelationshipIndex()
        t = self._make_triple("card_number", "type_of", "personal_data")
        idx.add(t)
        result = idx.get_subjects("personal_data")
        assert result == [t]

    def test_get_objects_unknown_subject_returns_empty(self):
        idx = RelationshipIndex()
        idx.add(self._make_triple("a", "type_of", "b"))
        assert idx.get_objects("unknown") == []

    def test_get_subjects_unknown_object_returns_empty(self):
        idx = RelationshipIndex()
        idx.add(self._make_triple("a", "type_of", "b"))
        assert idx.get_subjects("unknown") == []

    def test_get_objects_verb_filter(self):
        idx = RelationshipIndex()
        t1 = self._make_triple("orders", "references", "customers")
        t2 = self._make_triple("orders", "part_of", "invoices")
        idx.add(t1)
        idx.add(t2)
        assert idx.get_objects("orders", verb="references") == [t1]
        assert idx.get_objects("orders", verb="part_of") == [t2]
        assert set(idx.get_objects("orders")) == {t1, t2}

    def test_get_subjects_verb_filter(self):
        idx = RelationshipIndex()
        t1 = self._make_triple("PCI_DSS", "governs", "card_number")
        t2 = self._make_triple("GDPR", "governs", "card_number")
        idx.add(t1)
        idx.add(t2)
        subjects = idx.get_subjects("card_number", verb="governs")
        assert set(subjects) == {t1, t2}

    def test_get_objects_verb_filter_no_match(self):
        idx = RelationshipIndex()
        idx.add(self._make_triple("a", "type_of", "b"))
        assert idx.get_objects("a", verb="governs") == []

    def test_len_counts_all_triples(self):
        idx = RelationshipIndex()
        idx.add(self._make_triple("a", "type_of", "b"))
        idx.add(self._make_triple("a", "governs", "c"))
        idx.add(self._make_triple("d", "references", "e"))
        assert len(idx) == 3

    def test_multiple_objects_same_subject(self):
        idx = RelationshipIndex()
        t1 = self._make_triple("policy", "governs", "field_a")
        t2 = self._make_triple("policy", "governs", "field_b")
        idx.add(t1)
        idx.add(t2)
        assert set(idx.get_objects("policy", verb="governs")) == {t1, t2}

    def test_deterministic_fk_verbs(self):
        """references and part_of verbs are accepted (FK-derived, no LLM needed)."""
        idx = RelationshipIndex()
        idx.add(SVOTriple("invoice_lines", "part_of", "invoices", 1.0))
        idx.add(SVOTriple("invoices", "references", "customers", 1.0))
        assert len(idx) == 2

    def test_top_level_import(self):
        import chonk
        assert chonk.SVOTriple is SVOTriple
        assert chonk.RelationshipIndex is RelationshipIndex
        assert chonk.VERB_SET is VERB_SET
