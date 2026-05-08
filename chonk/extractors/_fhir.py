# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: e5f6a7b8-c9d0-1234-ef01-234567890123
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""FhirRenderer — renders FHIR R4 Bundle JSON into per-resource markdown."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentChunk

_FHIR_ID_RE = re.compile(r"\b([A-Za-z][a-zA-Z]+)/([A-Za-z0-9\-\.]+)\b")

_RENDERABLE_TYPES = frozenset(
    {
        "AllergyIntolerance",
        "Condition",
        "DiagnosticReport",
        "Encounter",
        "ImagingStudy",
        "Immunization",
        "MedicationRequest",
        "MedicationStatement",
        "Observation",
        "Patient",
        "Procedure",
        "ServiceRequest",
    }
)


def _iter_resources(obj: object) -> list[dict]:
    if isinstance(obj, dict):
        rtype = obj.get("resourceType", "")
        if rtype == "Bundle":
            return [
                e["resource"]
                for e in obj.get("entry", [])
                if isinstance(e, dict) and isinstance(e.get("resource"), dict)
            ]
        if rtype in _RENDERABLE_TYPES:
            return [obj]
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict) and r.get("resourceType")]
    return []


def _coding_display(coding_list: list[dict]) -> str:
    for c in coding_list:
        d = c.get("display") or c.get("code")
        if d:
            return d
    return ""


def _code_text(code_obj: dict | None) -> str:
    if not code_obj:
        return ""
    return code_obj.get("text") or _coding_display(code_obj.get("coding", []))


def _reference_display(ref: dict | None) -> str:
    if not ref:
        return ""
    return ref.get("display") or ref.get("reference") or ""


def _render_patient(res: dict) -> str:
    rid = res.get("id", "UNKNOWN")
    names = res.get("name", [])
    name_str = ""
    if names:
        n = names[0]
        given = " ".join(n.get("given", []))
        family = n.get("family", "")
        name_str = f"{given} {family}".strip()

    gender = res.get("gender", "")
    birth_date = res.get("birthDate", "")
    identifiers: list[str] = []
    for ident in res.get("identifier", []):
        sys = ident.get("system", "")
        val = ident.get("value", "")
        if val:
            identifiers.append(f"{val} ({sys})" if sys else val)

    lines = [f"# Patient/{rid} {name_str}", ""]
    if gender:
        lines.append(f"**Gender:** {gender}")
    if birth_date:
        lines.append(f"**DOB:** {birth_date}")
    if identifiers:
        lines.append(f"**Identifiers:** {', '.join(identifiers[:3])}")
    return "\n".join(lines)


def _render_generic(res: dict) -> str:
    rtype = res.get("resourceType", "Resource")
    rid = res.get("id", "UNKNOWN")

    code = _code_text(res.get("code") or res.get("medicationCodeableConcept"))
    subject = _reference_display(res.get("subject") or res.get("patient"))

    status = res.get("status") or res.get("clinicalStatus", {})
    if isinstance(status, dict):
        status = _code_text(status)
    status = str(status) if status else ""

    date = (
        res.get("recordedDate")
        or res.get("effectiveDateTime")
        or res.get("occurrenceDateTime")
        or res.get("authoredOn")
        or res.get("date")
        or res.get("issued")
        or res.get("onsetDateTime")
        or ""
    )

    category_list = res.get("category", [])
    category = ""
    if category_list:
        c = category_list[0] if isinstance(category_list, list) else category_list
        category = _code_text(c) if isinstance(c, dict) else str(c)

    title = code or rid
    lines: list[str] = [f"# {rtype}/{rid} {title}", ""]

    meta: list[str] = []
    if status:
        meta.append(f"**Status:** {status}")
    if subject:
        meta.append(f"**Subject:** {subject}")
    if date:
        meta.append(f"**Date:** {date[:10]}")
    if category:
        meta.append(f"**Category:** {category}")
    lines.extend(meta)

    # Observations: value
    if rtype == "Observation":
        val = res.get("valueQuantity")
        if val:
            v = val.get("value")
            unit = val.get("unit") or val.get("code", "")
            if v is not None:
                lines += ["", f"**Value:** {v} {unit}".strip()]
        val_str = res.get("valueString")
        if val_str:
            lines += ["", f"**Value:** {val_str}"]
        # Components (vital panel, etc.)
        components = res.get("component", [])
        if components:
            lines += ["", "## Components", ""]
            for comp in components[:20]:
                comp_code = _code_text(comp.get("code"))
                comp_val = comp.get("valueQuantity", {})
                comp_v = comp_val.get("value")
                comp_u = comp_val.get("unit") or comp_val.get("code", "")
                if comp_code and comp_v is not None:
                    lines.append(f"- {comp_code}: {comp_v} {comp_u}".strip())

    # Diagnostic report: result references
    if rtype == "DiagnosticReport":
        results = res.get("result", [])
        if results:
            lines += ["", "## Results", ""]
            for r in results[:20]:
                lines.append(f"- {_reference_display(r)}")

    # Note / text
    note_list = res.get("note", [])
    notes = [n.get("text", "") for n in note_list if n.get("text")]
    if notes:
        lines += ["", "## Notes", ""] + [n[:500] for n in notes[:3]]

    return "\n".join(lines)


def _render_one(res: dict) -> str:
    rtype = res.get("resourceType", "")
    if rtype == "Patient":
        return _render_patient(res)
    return _render_generic(res)


class FhirRenderer:
    """Renderer for FHIR R4 Bundle JSON.

    Detects ``{"resourceType": "Bundle", "entry": [...]}`` or a single FHIR
    resource.  Renders each resource as an H1-headed markdown section.

    ``source_detail`` per chunk::

        {
            "resource_type": "Observation",
            "resource_id":   "obs-123",
            "code":          "Blood pressure",
            "subject":       "Patient/patient-1",
        }
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:  # noqa: ARG002
        if not isinstance(obj, dict):
            return False
        rtype = obj.get("resourceType", "")
        if rtype == "Bundle":
            entries = obj.get("entry", [])
            return isinstance(entries, list) and len(entries) > 0
        return rtype in _RENDERABLE_TYPES

    def render(self, obj: object) -> str:
        resources = _iter_resources(obj)
        return "\n\n".join(_render_one(r) for r in resources)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        meta: dict[str, dict] = {}
        rendered: dict[str, str] = {}
        for res in _iter_resources(obj):
            rtype = res.get("resourceType", "")
            rid = res.get("id", "")
            if not rid:
                continue
            key = f"{rtype}/{rid}"
            code = _code_text(res.get("code") or res.get("medicationCodeableConcept"))
            subject = _reference_display(res.get("subject") or res.get("patient"))
            meta[key] = {
                "resource_type": rtype,
                "resource_id": rid,
                "code": code,
                "subject": subject,
            }
            rendered[key] = _render_one(res)

        for chunk in chunks:
            matched_key = None
            m = _FHIR_ID_RE.search(chunk.content)
            if m:
                candidate = m.group(0)
                if candidate in meta:
                    matched_key = candidate
            if not matched_key:
                for part in chunk.section or []:
                    sm = _FHIR_ID_RE.match(part.strip())
                    if sm:
                        candidate = sm.group(0)
                        if candidate in meta:
                            matched_key = candidate
                            break
            if matched_key:
                chunk.source_detail = meta[matched_key]
                chunk.rendered_source = rendered[matched_key]

        return chunks
