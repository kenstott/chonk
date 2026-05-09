# Copyright (c) 2025 Kenneth Stott. MIT License.
"""Tests for AttackRenderer — MITRE ATT&CK STIX bundle rendering."""

from __future__ import annotations

import json

from chonk.extractors._attack import AttackRenderer, _ext_id, _iter_techniques
from chonk.extractors._renderer import Renderer
from chonk.models import DocumentChunk

# ---------------------------------------------------------------------------
# Minimal STIX bundle fixture
# ---------------------------------------------------------------------------

_TECH_ID = "attack-pattern--aaaaaaaa-0000-0000-0000-000000000001"
_MIT_ID = "course-of-action--bbbbbbbb-0000-0000-0000-000000000001"
_PARENT_ID = "attack-pattern--cccccccc-0000-0000-0000-000000000001"
_REL_MIT_ID = "relationship--11111111-0000-0000-0000-000000000001"
_REL_SUB_ID = "relationship--22222222-0000-0000-0000-000000000002"

_TECHNIQUE = {
    "type": "attack-pattern",
    "id": _TECH_ID,
    "name": "Extra Window Memory Injection",
    "description": "Adversaries inject code via EWM to evade defenses.",
    "x_mitre_platforms": ["Windows"],
    "x_mitre_is_subtechnique": True,
    "x_mitre_deprecated": False,
    "revoked": False,
    "kill_chain_phases": [
        {"phase_name": "privilege-escalation", "kill_chain_name": "mitre-attack"},
        {"phase_name": "defense-evasion", "kill_chain_name": "mitre-attack"},
    ],
    "external_references": [
        {
            "source_name": "mitre-attack",
            "external_id": "T1055.011",
            "url": "https://attack.mitre.org/techniques/T1055/011",
        }
    ],
}

_PARENT_TECH = {
    "type": "attack-pattern",
    "id": _PARENT_ID,
    "name": "Process Injection",
    "description": "Parent technique.",
    "x_mitre_platforms": ["Windows", "Linux", "macOS"],
    "x_mitre_is_subtechnique": False,
    "x_mitre_deprecated": False,
    "revoked": False,
    "kill_chain_phases": [
        {"phase_name": "privilege-escalation", "kill_chain_name": "mitre-attack"},
    ],
    "external_references": [
        {
            "source_name": "mitre-attack",
            "external_id": "T1055",
            "url": "https://attack.mitre.org/techniques/T1055",
        }
    ],
}

_MITIGATION = {
    "type": "course-of-action",
    "id": _MIT_ID,
    "name": "Behavior Prevention on Endpoint",
    "description": "Use endpoint tools to prevent suspicious patterns. Additional context here.",
    "x_mitre_deprecated": False,
    "revoked": False,
    "external_references": [],
}

_REL_MITIGATES = {
    "type": "relationship",
    "id": _REL_MIT_ID,
    "relationship_type": "mitigates",
    "source_ref": _MIT_ID,
    "target_ref": _TECH_ID,
}

_REL_SUBTECHNIQUE = {
    "type": "relationship",
    "id": _REL_SUB_ID,
    "relationship_type": "subtechnique-of",
    "source_ref": _TECH_ID,
    "target_ref": _PARENT_ID,
}

_BUNDLE = {
    "type": "bundle",
    "spec_version": "2.1",
    "objects": [
        _TECHNIQUE,
        _PARENT_TECH,
        _MITIGATION,
        _REL_MITIGATES,
        _REL_SUBTECHNIQUE,
    ],
}


def _bytes(obj: object) -> bytes:
    return json.dumps(obj).encode()


# ---------------------------------------------------------------------------
# _ext_id
# ---------------------------------------------------------------------------


class TestExtId:
    def test_returns_attack_id(self):
        assert _ext_id(_TECHNIQUE) == "T1055.011"

    def test_returns_none_when_absent(self):
        assert _ext_id({"external_references": []}) is None


# ---------------------------------------------------------------------------
# _iter_techniques
# ---------------------------------------------------------------------------


class TestIterTechniques:
    def test_returns_live_techniques(self):
        techs, _ = _iter_techniques(_BUNDLE)
        ids = {_ext_id(t) for t in techs}
        assert "T1055.011" in ids
        assert "T1055" in ids

    def test_excludes_revoked(self):
        bundle = {
            "objects": [{**_TECHNIQUE, "revoked": True}, _PARENT_TECH]
        }
        techs, _ = _iter_techniques(bundle)
        ids = {_ext_id(t) for t in techs}
        assert "T1055.011" not in ids

    def test_excludes_deprecated(self):
        bundle = {
            "objects": [{**_TECHNIQUE, "x_mitre_deprecated": True}, _PARENT_TECH]
        }
        techs, _ = _iter_techniques(bundle)
        ids = {_ext_id(t) for t in techs}
        assert "T1055.011" not in ids

    def test_index_builds_parent_map(self):
        _, index = _iter_techniques(_BUNDLE)
        assert index.parent[_TECH_ID] == _PARENT_ID

    def test_index_builds_mitigation_map(self):
        _, index = _iter_techniques(_BUNDLE)
        assert _TECH_ID in index.tech_mitigations
        assert index.tech_mitigations[_TECH_ID][0]["name"] == "Behavior Prevention on Endpoint"


# ---------------------------------------------------------------------------
# AttackRenderer
# ---------------------------------------------------------------------------


class TestAttackRenderer:
    def setup_method(self):
        self.r = AttackRenderer()

    def test_can_render_attack_bundle(self):
        assert self.r.can_render(None, _BUNDLE)

    def test_cannot_render_cve_json(self):
        cve = {"vulnerabilities": [{"cve": {"id": "CVE-2024-1"}}]}
        assert not self.r.can_render(None, cve)

    def test_cannot_render_plain_dict(self):
        assert not self.r.can_render(None, {"name": "Alice"})

    def test_cannot_render_empty_bundle(self):
        assert not self.r.can_render(None, {"objects": []})

    def test_render_contains_attack_id(self):
        md = self.r.render(_BUNDLE)
        assert "T1055.011" in md

    def test_render_h1_format(self):
        md = self.r.render(_BUNDLE)
        assert "# T1055.011 Extra Window Memory Injection" in md

    def test_render_tactics_present(self):
        md = self.r.render(_BUNDLE)
        assert "privilege-escalation" in md

    def test_render_platforms_present(self):
        md = self.r.render(_BUNDLE)
        assert "Windows" in md

    def test_render_parent_id_present(self):
        md = self.r.render(_BUNDLE)
        assert "T1055" in md

    def test_render_mitigation_present(self):
        md = self.r.render(_BUNDLE)
        assert "Behavior Prevention on Endpoint" in md

    def test_render_url_present(self):
        md = self.r.render(_BUNDLE)
        assert "attack.mitre.org" in md

    def test_render_description_present(self):
        md = self.r.render(_BUNDLE)
        assert "inject code via EWM" in md

    def test_annotate_stamps_source_detail(self):
        from chonk import DocumentLoader

        loader = DocumentLoader(enrich_context=False)
        chunks = loader.load_bytes(_bytes(_BUNDLE), name="attack", doc_type="json")
        chunks = self.r.annotate(chunks, _BUNDLE)
        annotated = [c for c in chunks if c.source_detail and c.source_detail.get("attack_id") == "T1055.011"]
        assert annotated, "no chunk annotated with T1055.011"
        d = annotated[0].source_detail
        assert d["name"] == "Extra Window Memory Injection"
        assert "privilege-escalation" in d["tactics"]
        assert d["is_subtechnique"] is True
        assert d["parent_id"] == "T1055"

    def test_annotate_uses_section_fallback(self):
        chunk = DocumentChunk(
            document_name="attack",
            content="## Description\n\nSome injected code text.",
            section=["T1055.011 Extra Window Memory Injection", "Description"],
            chunk_index=1,
        )
        self.r.annotate([chunk], _BUNDLE)
        assert chunk.source_detail is not None
        assert chunk.source_detail["attack_id"] == "T1055.011"

    def test_annotate_no_match_leaves_unchanged(self):
        chunk = DocumentChunk(
            document_name="x", content="nothing here", chunk_index=0
        )
        self.r.annotate([chunk], _BUNDLE)
        assert chunk.source_detail is None

    def test_satisfies_renderer_protocol(self):
        assert isinstance(self.r, Renderer)


# ---------------------------------------------------------------------------
# JsonExtractor dispatch
# ---------------------------------------------------------------------------


class TestJsonExtractorDispatch:
    def test_attack_bundle_dispatched_to_renderer(self):
        from chonk.extractors._json import JsonExtractor

        ext = JsonExtractor(renderers=[AttackRenderer()])
        md = ext.extract(_bytes(_BUNDLE))
        assert "# T1055" in md
        assert "inject code" in md

    def test_non_attack_json_falls_back_to_walk(self):
        from chonk.extractors._json import JsonExtractor

        ext = JsonExtractor(renderers=[AttackRenderer()])
        md = ext.extract(_bytes({"foo": "bar"}))
        assert "bar" in md
        assert "# T1055" not in md
