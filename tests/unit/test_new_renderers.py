# Copyright (c) 2025 Kenneth Stott. MIT License.
"""Tests for CweRenderer, NistRenderer, ClinicalTrialRenderer, FdaLabelRenderer, FhirRenderer."""

from __future__ import annotations

from chonk.extractors._clinical_trial import ClinicalTrialRenderer, _iter_studies
from chonk.extractors._cwe import CweRenderer, _iter_weaknesses_from_dict
from chonk.extractors._fda_label import FdaLabelRenderer, _iter_labels
from chonk.extractors._fhir import FhirRenderer, _iter_resources
from chonk.extractors._nist import NistRenderer, _iter_controls
from chonk.extractors._renderer import Renderer
from chonk.models import DocumentChunk

# ---------------------------------------------------------------------------
# CWE fixtures
# ---------------------------------------------------------------------------

_CWE_BUNDLE = {
    "_tag": "Weakness_Catalog",
    "Weaknesses": {
        "Weakness": [
            {
                "ID": "119",
                "Name": "Improper Restriction of Operations within the Bounds of a Memory Buffer",
                "Description": {"_text": "The product performs operations on a memory buffer."},
                "Applicable_Platforms": {
                    "Language": [
                        {"Name": "C"},
                        {"Name": "C++"},
                    ]
                },
                "Common_Consequences": {
                    "Consequence": [
                        {"Scope": {"_text": "Integrity"}},
                        {"Scope": {"_text": "Availability"}},
                    ]
                },
                "Related_Weaknesses": {
                    "Related_Weakness": [
                        {"CWE_ID": "787", "Nature": "ChildOf"},
                    ]
                },
            },
            {
                "ID": "787",
                "Name": "Out-of-bounds Write",
                "Description": {"_text": "The product writes data past the end of a buffer."},
            },
        ]
    },
}


# ---------------------------------------------------------------------------
# NIST fixtures
# ---------------------------------------------------------------------------

_NIST_BUNDLE = {
    "catalog": {
        "groups": [
            {
                "id": "ac",
                "title": "Access Control",
                "controls": [
                    {
                        "id": "ac-1",
                        "title": "Policy and Procedures",
                        "parts": [
                            {
                                "name": "statement",
                                "prose": "The organization develops and implements access control policies.",
                                "parts": [],
                            }
                        ],
                        "controls": [
                            {
                                "id": "ac-1.1",
                                "title": "Policy Review",
                                "parts": [{"name": "statement", "prose": "Review annually."}],
                            }
                        ],
                    }
                ],
            }
        ]
    }
}

# ---------------------------------------------------------------------------
# ClinicalTrial fixtures
# ---------------------------------------------------------------------------

_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT04280705",
            "briefTitle": "A Study of Remdesivir in COVID-19",
        },
        "statusModule": {"overallStatus": "COMPLETED"},
        "descriptionModule": {
            "briefSummary": "This study evaluates the safety and efficacy of remdesivir."
        },
        "conditionsModule": {"conditions": ["COVID-19"]},
        "designModule": {"phases": ["PHASE3"]},
        "armsInterventionsModule": {
            "interventions": [{"type": "DRUG", "name": "Remdesivir"}]
        },
        "eligibilityModule": {
            "eligibilityCriteria": "Inclusion Criteria: Adults >= 18 years.",
            "sex": "ALL",
            "minimumAge": "18 Years",
        },
    }
}

_CT_BUNDLE = {"studies": [_STUDY]}

# ---------------------------------------------------------------------------
# FDA fixtures
# ---------------------------------------------------------------------------

_FDA_LABEL = {
    "openfda": {
        "brand_name": ["Tylenol"],
        "generic_name": ["Acetaminophen"],
        "manufacturer_name": ["McNeil Consumer Healthcare"],
        "route": ["ORAL"],
        "product_type": ["HUMAN OTC DRUG"],
        "application_number": ["NDA019837"],
    },
    "indications_and_usage": ["For temporary relief of minor aches and pains."],
    "dosage_and_administration": ["Adults and children 12 years and over: take 2 tablets every 4 to 6 hours."],
    "warnings": ["Liver warning: This product contains acetaminophen."],
}

_FDA_BUNDLE = {"results": [_FDA_LABEL]}

# ---------------------------------------------------------------------------
# FHIR fixtures
# ---------------------------------------------------------------------------

_FHIR_PATIENT = {
    "resourceType": "Patient",
    "id": "patient-1",
    "name": [{"given": ["John"], "family": "Doe"}],
    "gender": "male",
    "birthDate": "1980-01-15",
}

_FHIR_OBS = {
    "resourceType": "Observation",
    "id": "obs-1",
    "status": "final",
    "code": {"coding": [{"display": "Blood pressure"}], "text": "Blood pressure"},
    "subject": {"reference": "Patient/patient-1", "display": "John Doe"},
    "effectiveDateTime": "2024-03-15T10:00:00Z",
    "valueQuantity": {"value": 120, "unit": "mmHg"},
}

_FHIR_BUNDLE = {
    "resourceType": "Bundle",
    "type": "searchset",
    "entry": [
        {"resource": _FHIR_PATIENT},
        {"resource": _FHIR_OBS},
    ],
}


# ===========================================================================
# CweRenderer
# ===========================================================================


class TestCweRenderer:
    def setup_method(self):
        self.r = CweRenderer()

    def test_can_render_weakness_catalog(self):
        assert self.r.can_render(None, _CWE_BUNDLE)

    def test_cannot_render_plain_dict(self):
        assert not self.r.can_render(None, {"foo": "bar"})

    def test_cannot_render_attack_bundle(self):
        assert not self.r.can_render(None, {"_tag": "other", "Weaknesses": {}})

    def test_render_contains_cwe_id(self):
        md = self.r.render(_CWE_BUNDLE)
        assert "CWE-119" in md

    def test_render_h1_format(self):
        md = self.r.render(_CWE_BUNDLE)
        assert "# CWE-119" in md

    def test_render_description_present(self):
        md = self.r.render(_CWE_BUNDLE)
        assert "memory buffer" in md

    def test_render_platforms_present(self):
        md = self.r.render(_CWE_BUNDLE)
        assert "C++" in md

    def test_render_consequences_present(self):
        md = self.r.render(_CWE_BUNDLE)
        assert "Integrity" in md

    def test_render_related_present(self):
        md = self.r.render(_CWE_BUNDLE)
        assert "CWE-787" in md

    def test_annotate_stamps_source_detail(self):
        chunk = DocumentChunk(
            document_name="cwe", content="# CWE-119 Buffer Overflow\n\nDetails here.", chunk_index=0
        )
        self.r.annotate([chunk], _CWE_BUNDLE)
        assert chunk.source_detail is not None
        assert chunk.source_detail["cwe_id"] == "CWE-119"
        assert "C" in chunk.source_detail["platforms"]

    def test_annotate_sets_rendered_source(self):
        chunk = DocumentChunk(
            document_name="cwe", content="# CWE-119 Memory buffer.", chunk_index=0
        )
        self.r.annotate([chunk], _CWE_BUNDLE)
        assert chunk.rendered_source is not None
        assert "CWE-119" in chunk.rendered_source

    def test_annotate_section_fallback(self):
        chunk = DocumentChunk(
            document_name="cwe",
            content="memory operations",
            section=["CWE-119 Buffer", "Description"],
            chunk_index=1,
        )
        self.r.annotate([chunk], _CWE_BUNDLE)
        assert chunk.source_detail is not None
        assert chunk.source_detail["cwe_id"] == "CWE-119"

    def test_annotate_no_match_unchanged(self):
        chunk = DocumentChunk(document_name="x", content="nothing", chunk_index=0)
        self.r.annotate([chunk], _CWE_BUNDLE)
        assert chunk.source_detail is None

    def test_satisfies_renderer_protocol(self):
        assert isinstance(self.r, Renderer)

    def test_iter_weaknesses(self):
        ws = _iter_weaknesses_from_dict(_CWE_BUNDLE)
        assert len(ws) == 2
        ids = {w["ID"] for w in ws}
        assert "119" in ids
        assert "787" in ids


# ===========================================================================
# NistRenderer
# ===========================================================================


class TestNistRenderer:
    def setup_method(self):
        self.r = NistRenderer()

    def test_can_render_oscal(self):
        assert self.r.can_render(None, _NIST_BUNDLE)

    def test_cannot_render_plain_dict(self):
        assert not self.r.can_render(None, {"name": "Alice"})

    def test_render_contains_control_id(self):
        md = self.r.render(_NIST_BUNDLE)
        assert "AC-1" in md

    def test_render_h1_format(self):
        md = self.r.render(_NIST_BUNDLE)
        assert "# AC-1 Policy and Procedures" in md

    def test_render_statement_prose(self):
        md = self.r.render(_NIST_BUNDLE)
        assert "access control policies" in md

    def test_render_sub_control(self):
        md = self.r.render(_NIST_BUNDLE)
        assert "AC-1.1" in md

    def test_annotate_stamps_source_detail(self):
        chunk = DocumentChunk(
            document_name="nist", content="# AC-1 Policy and Procedures\n\nPolicies.", chunk_index=0
        )
        self.r.annotate([chunk], _NIST_BUNDLE)
        assert chunk.source_detail is not None
        assert chunk.source_detail["control_id"] == "AC-1"
        assert chunk.source_detail["group"] == "Access Control"

    def test_annotate_sets_rendered_source(self):
        chunk = DocumentChunk(
            document_name="nist", content="AC-1 access control", chunk_index=0
        )
        self.r.annotate([chunk], _NIST_BUNDLE)
        assert chunk.rendered_source is not None
        assert "AC-1" in chunk.rendered_source

    def test_annotate_no_match_unchanged(self):
        chunk = DocumentChunk(document_name="x", content="nothing", chunk_index=0)
        self.r.annotate([chunk], _NIST_BUNDLE)
        assert chunk.source_detail is None

    def test_satisfies_renderer_protocol(self):
        assert isinstance(self.r, Renderer)

    def test_iter_controls_flattens(self):
        controls = _iter_controls(_NIST_BUNDLE)
        ids = {c["id"] for c in controls}
        assert "ac-1" in ids
        assert "ac-1.1" in ids


# ===========================================================================
# ClinicalTrialRenderer
# ===========================================================================


class TestClinicalTrialRenderer:
    def setup_method(self):
        self.r = ClinicalTrialRenderer()

    def test_can_render_bundle(self):
        assert self.r.can_render(None, _CT_BUNDLE)

    def test_can_render_single_study(self):
        assert self.r.can_render(None, _STUDY)

    def test_cannot_render_plain_dict(self):
        assert not self.r.can_render(None, {"name": "Alice"})

    def test_render_contains_nct_id(self):
        md = self.r.render(_CT_BUNDLE)
        assert "NCT04280705" in md

    def test_render_h1_format(self):
        md = self.r.render(_CT_BUNDLE)
        assert "# NCT04280705" in md

    def test_render_status_present(self):
        md = self.r.render(_CT_BUNDLE)
        assert "COMPLETED" in md

    def test_render_summary_present(self):
        md = self.r.render(_CT_BUNDLE)
        assert "remdesivir" in md.lower()

    def test_render_conditions_present(self):
        md = self.r.render(_CT_BUNDLE)
        assert "COVID-19" in md

    def test_render_intervention_present(self):
        md = self.r.render(_CT_BUNDLE)
        assert "Remdesivir" in md

    def test_render_eligibility_present(self):
        md = self.r.render(_CT_BUNDLE)
        assert "Inclusion Criteria" in md

    def test_annotate_stamps_source_detail(self):
        chunk = DocumentChunk(
            document_name="ct", content="# NCT04280705 Remdesivir Study", chunk_index=0
        )
        self.r.annotate([chunk], _CT_BUNDLE)
        assert chunk.source_detail is not None
        assert chunk.source_detail["nct_id"] == "NCT04280705"
        assert chunk.source_detail["status"] == "COMPLETED"
        assert "COVID-19" in chunk.source_detail["conditions"]

    def test_annotate_sets_rendered_source(self):
        chunk = DocumentChunk(
            document_name="ct", content="NCT04280705 safety", chunk_index=0
        )
        self.r.annotate([chunk], _CT_BUNDLE)
        assert chunk.rendered_source is not None
        assert "NCT04280705" in chunk.rendered_source

    def test_annotate_no_match_unchanged(self):
        chunk = DocumentChunk(document_name="x", content="nothing", chunk_index=0)
        self.r.annotate([chunk], _CT_BUNDLE)
        assert chunk.source_detail is None

    def test_satisfies_renderer_protocol(self):
        assert isinstance(self.r, Renderer)

    def test_iter_studies(self):
        studies = _iter_studies(_CT_BUNDLE)
        assert len(studies) == 1

    def test_iter_studies_single(self):
        studies = _iter_studies(_STUDY)
        assert len(studies) == 1


# ===========================================================================
# FdaLabelRenderer
# ===========================================================================


class TestFdaLabelRenderer:
    def setup_method(self):
        self.r = FdaLabelRenderer()

    def test_can_render_bundle(self):
        assert self.r.can_render(None, _FDA_BUNDLE)

    def test_can_render_single(self):
        assert self.r.can_render(None, _FDA_LABEL)

    def test_cannot_render_plain_dict(self):
        assert not self.r.can_render(None, {"name": "Alice"})

    def test_render_brand_name(self):
        md = self.r.render(_FDA_BUNDLE)
        assert "Tylenol" in md

    def test_render_h1_is_brand(self):
        md = self.r.render(_FDA_BUNDLE)
        assert "# Tylenol" in md

    def test_render_indications_present(self):
        md = self.r.render(_FDA_BUNDLE)
        assert "aches and pains" in md

    def test_render_warnings_present(self):
        md = self.r.render(_FDA_BUNDLE)
        assert "acetaminophen" in md.lower()

    def test_render_manufacturer_present(self):
        md = self.r.render(_FDA_BUNDLE)
        assert "McNeil" in md

    def test_annotate_by_application_number(self):
        chunk = DocumentChunk(
            document_name="fda", content="Application NDA019837 Tylenol.", chunk_index=0
        )
        self.r.annotate([chunk], _FDA_BUNDLE)
        assert chunk.source_detail is not None
        assert chunk.source_detail["brand_name"] == "Tylenol"

    def test_annotate_sets_rendered_source(self):
        chunk = DocumentChunk(
            document_name="fda", content="NDA019837 pain relief", chunk_index=0
        )
        self.r.annotate([chunk], _FDA_BUNDLE)
        assert chunk.rendered_source is not None
        assert "Tylenol" in chunk.rendered_source

    def test_annotate_no_match_unchanged(self):
        chunk = DocumentChunk(document_name="x", content="nothing here", chunk_index=0)
        self.r.annotate([chunk], _FDA_BUNDLE)
        assert chunk.source_detail is None

    def test_satisfies_renderer_protocol(self):
        assert isinstance(self.r, Renderer)

    def test_iter_labels(self):
        labels = _iter_labels(_FDA_BUNDLE)
        assert len(labels) == 1

    def test_iter_labels_single(self):
        labels = _iter_labels(_FDA_LABEL)
        assert len(labels) == 1


# ===========================================================================
# FhirRenderer
# ===========================================================================


class TestFhirRenderer:
    def setup_method(self):
        self.r = FhirRenderer()

    def test_can_render_bundle(self):
        assert self.r.can_render(None, _FHIR_BUNDLE)

    def test_can_render_single_resource(self):
        assert self.r.can_render(None, _FHIR_PATIENT)

    def test_cannot_render_plain_dict(self):
        assert not self.r.can_render(None, {"name": "Alice"})

    def test_cannot_render_empty_bundle(self):
        assert not self.r.can_render(None, {"resourceType": "Bundle", "entry": []})

    def test_render_patient_h1(self):
        md = self.r.render(_FHIR_BUNDLE)
        assert "# Patient/patient-1" in md

    def test_render_patient_name(self):
        md = self.r.render(_FHIR_BUNDLE)
        assert "John Doe" in md

    def test_render_observation_h1(self):
        md = self.r.render(_FHIR_BUNDLE)
        assert "# Observation/obs-1" in md

    def test_render_observation_code(self):
        md = self.r.render(_FHIR_BUNDLE)
        assert "Blood pressure" in md

    def test_render_observation_value(self):
        md = self.r.render(_FHIR_BUNDLE)
        assert "120" in md

    def test_annotate_stamps_patient(self):
        chunk = DocumentChunk(
            document_name="fhir",
            content="# Patient/patient-1 John Doe\n\nGender: male",
            chunk_index=0,
        )
        self.r.annotate([chunk], _FHIR_BUNDLE)
        assert chunk.source_detail is not None
        assert chunk.source_detail["resource_type"] == "Patient"
        assert chunk.source_detail["resource_id"] == "patient-1"

    def test_annotate_stamps_observation(self):
        chunk = DocumentChunk(
            document_name="fhir",
            content="# Observation/obs-1 Blood pressure",
            chunk_index=1,
        )
        self.r.annotate([chunk], _FHIR_BUNDLE)
        assert chunk.source_detail is not None
        assert chunk.source_detail["resource_type"] == "Observation"

    def test_annotate_sets_rendered_source(self):
        chunk = DocumentChunk(
            document_name="fhir",
            content="Observation/obs-1 measurement",
            chunk_index=0,
        )
        self.r.annotate([chunk], _FHIR_BUNDLE)
        assert chunk.rendered_source is not None
        assert "Blood pressure" in chunk.rendered_source

    def test_annotate_no_match_unchanged(self):
        chunk = DocumentChunk(document_name="x", content="nothing", chunk_index=0)
        self.r.annotate([chunk], _FHIR_BUNDLE)
        assert chunk.source_detail is None

    def test_satisfies_renderer_protocol(self):
        assert isinstance(self.r, Renderer)

    def test_iter_resources(self):
        resources = _iter_resources(_FHIR_BUNDLE)
        assert len(resources) == 2
        types = {r["resourceType"] for r in resources}
        assert "Patient" in types
        assert "Observation" in types
