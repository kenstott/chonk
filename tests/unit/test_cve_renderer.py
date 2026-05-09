# Copyright (c) 2025 Kenneth Stott. MIT License.
"""Tests for CveRenderer and the Renderer plug-in pattern on JsonExtractor."""

from __future__ import annotations

import json

import pytest

from chonk.extractors._cve import CveRenderer, _iter_cves, _render_one
from chonk.extractors._json import JsonExtractor
from chonk.extractors._renderer import Renderer
from chonk.models import DocumentChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ONE_CVE = {
    "id": "CVE-2024-99999",
    "published": "2024-03-15T00:00:00.000",
    "lastModified": "2024-04-01T00:00:00.000",
    "vulnStatus": "Analyzed",
    "descriptions": [
        {"lang": "en", "value": "A critical buffer overflow in Acme widget."},
        {"lang": "es", "value": "Desbordamiento de buffer en Acme widget."},
    ],
    "metrics": {
        "cvssMetricV31": [
            {
                "cvssData": {"baseScore": 9.8, "baseSeverity": "CRITICAL"},
                "exploitabilityScore": 3.9,
            }
        ]
    },
    "weaknesses": [
        {"description": [{"lang": "en", "value": "CWE-119"}]}
    ],
    "configurations": [
        {
            "nodes": [
                {
                    "operator": "OR",
                    "negate": False,
                    "cpeMatch": [
                        {
                            "vulnerable": True,
                            "criteria": "cpe:2.3:a:acme:widget:*:*:*:*:*:*:*:*",
                            "versionEndExcluding": "3.2.1",
                        }
                    ],
                }
            ]
        }
    ],
    "references": [
        {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-99999"},
        {"url": "https://acme.example.com/advisory/2024-001"},
    ],
}

_NVD_RESPONSE = {"vulnerabilities": [{"cve": _ONE_CVE}]}


def _bytes(obj: object) -> bytes:
    return json.dumps(obj).encode()


# ---------------------------------------------------------------------------
# _iter_cves
# ---------------------------------------------------------------------------


class TestIterCves:
    def test_nvd_response_shape(self):
        cves = _iter_cves(_NVD_RESPONSE)
        assert len(cves) == 1
        assert cves[0]["id"] == "CVE-2024-99999"

    def test_unwrapped_single_record(self):
        cves = _iter_cves(_ONE_CVE)
        assert len(cves) == 1

    def test_wrapped_single(self):
        cves = _iter_cves({"cve": _ONE_CVE})
        assert len(cves) == 1

    def test_plain_list(self):
        cves = _iter_cves([{"cve": _ONE_CVE}, {"cve": {**_ONE_CVE, "id": "CVE-2024-00002"}}])
        assert len(cves) == 2

    def test_unrecognised_returns_empty(self):
        assert _iter_cves({"foo": "bar"}) == []

    def test_empty_vulnerabilities(self):
        assert _iter_cves({"vulnerabilities": []}) == []


# ---------------------------------------------------------------------------
# _render_one
# ---------------------------------------------------------------------------


class TestRenderOne:
    def test_h1_is_cve_id(self):
        md = _render_one(_ONE_CVE)
        assert md.startswith("# CVE-2024-99999")

    def test_cvss_score_present(self):
        md = _render_one(_ONE_CVE)
        assert "9.8" in md
        assert "CRITICAL" in md

    def test_english_description_only(self):
        md = _render_one(_ONE_CVE)
        assert "buffer overflow in Acme widget" in md
        assert "Desbordamiento" not in md

    def test_cwe_present(self):
        md = _render_one(_ONE_CVE)
        assert "CWE-119" in md

    def test_affected_version_range(self):
        md = _render_one(_ONE_CVE)
        assert "< 3.2.1" in md

    def test_references_present(self):
        md = _render_one(_ONE_CVE)
        assert "nvd.nist.gov" in md

    def test_no_cvss_graceful(self):
        cve = {**_ONE_CVE, "metrics": {}}
        md = _render_one(cve)
        assert "# CVE-2024-99999" in md
        assert "CVSS" not in md

    def test_non_vulnerable_cpe_excluded(self):
        cve = {
            **_ONE_CVE,
            "configurations": [
                {
                    "nodes": [
                        {
                            "operator": "OR",
                            "negate": False,
                            "cpeMatch": [
                                {
                                    "vulnerable": False,
                                    "criteria": "cpe:2.3:a:acme:other:1.0:*:*:*:*:*:*:*",
                                }
                            ],
                        }
                    ]
                }
            ],
        }
        md = _render_one(cve)
        assert "## Affected" not in md


# ---------------------------------------------------------------------------
# CveRenderer
# ---------------------------------------------------------------------------


class TestCveRenderer:
    def setup_method(self):
        self.r = CveRenderer()

    def test_can_render_nvd_response(self):
        assert self.r.can_render(None, _NVD_RESPONSE)

    def test_can_render_single_record(self):
        assert self.r.can_render(None, _ONE_CVE)

    def test_cannot_render_unrelated_json(self):
        assert not self.r.can_render(None, {"name": "Alice", "age": 30})

    def test_render_multiple_cves_present(self):
        two = {"vulnerabilities": [{"cve": _ONE_CVE}, {"cve": {**_ONE_CVE, "id": "CVE-2024-00002"}}]}
        md = self.r.render(two)
        assert "CVE-2024-99999" in md
        assert "CVE-2024-00002" in md

    def test_annotate_stamps_source_detail(self):
        from chonk import DocumentLoader

        loader = DocumentLoader(enrich_context=False)
        chunks = loader.load_bytes(_bytes(_NVD_RESPONSE), name="CVE-2024-99999", doc_type="json")
        chunks = self.r.annotate(chunks, _bytes(_NVD_RESPONSE))
        annotated = [c for c in chunks if c.source_detail]
        assert annotated, "no chunks received source_detail"
        detail = annotated[0].source_detail
        assert detail["cve_id"] == "CVE-2024-99999"
        assert detail["cvss_score"] == pytest.approx(9.8)
        assert detail["severity"] == "CRITICAL"

    def test_annotate_no_match_leaves_chunks_unchanged(self):
        chunk = DocumentChunk(document_name="x", content="no cve here", chunk_index=0)
        result = self.r.annotate([chunk], _NVD_RESPONSE)
        assert result[0].source_detail is None


# ---------------------------------------------------------------------------
# Renderer protocol
# ---------------------------------------------------------------------------


class TestRendererProtocol:
    def test_cve_renderer_satisfies_protocol(self):
        assert isinstance(CveRenderer(), Renderer)


# ---------------------------------------------------------------------------
# JsonExtractor with CveRenderer
# ---------------------------------------------------------------------------


class TestJsonExtractorWithCveRenderer:
    def setup_method(self):
        self.extractor = JsonExtractor(renderers=[CveRenderer()])

    def test_cve_json_renders_markdown(self):
        text = self.extractor.extract(_bytes(_NVD_RESPONSE))
        assert "# CVE-2024-99999" in text
        assert "buffer overflow" in text

    def test_plain_json_falls_back_to_walk(self):
        plain = {"name": "Alice", "age": 30}
        text = self.extractor.extract(_bytes(plain))
        assert "Alice" in text
        assert "30" in text
        assert "#" in text  # key-path headings

    def test_annotate_delegates_to_renderer(self):
        from chonk import DocumentLoader

        loader = DocumentLoader(enrich_context=False)
        chunks = loader.load_bytes(_bytes(_NVD_RESPONSE), name="feed", doc_type="json")
        chunks = self.extractor.annotate(chunks, _bytes(_NVD_RESPONSE))
        annotated = [c for c in chunks if c.source_detail]
        assert annotated
        assert annotated[0].source_detail["cve_id"] == "CVE-2024-99999"

    def test_plain_json_annotate_noop(self):
        plain = {"name": "Alice"}
        chunk = DocumentChunk(document_name="x", content="Alice", chunk_index=0)
        result = self.extractor.annotate([chunk], _bytes(plain))
        assert result[0].source_detail is None
