# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d4e5f6a7-b8c9-0123-def0-123456789012
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""FdaLabelRenderer — renders openFDA drug label JSON into per-label markdown."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import DocumentChunk

_APPL_RE = re.compile(r"\bNDA\d{6}\b|\bBLA\d{6}\b|\bANDA\d{6}\b", re.IGNORECASE)

# Ordered sections to render — (json_key, display_heading)
_LABEL_SECTIONS: list[tuple[str, str]] = [
    ("indications_and_usage", "Indications and Usage"),
    ("dosage_and_administration", "Dosage and Administration"),
    ("dosage_forms_and_strengths", "Dosage Forms and Strengths"),
    ("contraindications", "Contraindications"),
    ("warnings_and_cautions", "Warnings and Precautions"),
    ("warnings", "Warnings"),
    ("adverse_reactions", "Adverse Reactions"),
    ("drug_interactions", "Drug Interactions"),
    ("use_in_specific_populations", "Use in Specific Populations"),
    ("description", "Description"),
    ("clinical_pharmacology", "Clinical Pharmacology"),
    ("mechanism_of_action", "Mechanism of Action"),
    ("clinical_studies", "Clinical Studies"),
    ("how_supplied", "How Supplied"),
    ("storage_and_handling", "Storage and Handling"),
    ("patient_counseling_information", "Patient Counseling Information"),
]


def _join_field(val: object) -> str:
    """openFDA fields are lists of strings; join and strip."""
    if isinstance(val, list):
        return " ".join(str(v) for v in val).strip()
    return str(val).strip() if val else ""


def _iter_labels(obj: object) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        if "results" in obj:
            return [r for r in obj["results"] if isinstance(r, dict)]
        if "openfda" in obj or "indications_and_usage" in obj:
            return [obj]
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    return []


def _label_id(label: dict[str, Any]) -> str:
    openfda = label.get("openfda", {})
    appl = openfda.get("application_number", [])
    if appl:
        return appl[0] if isinstance(appl, list) else str(appl)
    # Fallback to set_id or id
    return label.get("set_id") or label.get("id") or "UNKNOWN"


def _render_one(label: dict[str, Any]) -> str:
    openfda = label.get("openfda", {})

    brand_names = openfda.get("brand_name", [])
    brand = brand_names[0] if brand_names else ""
    generic_names = openfda.get("generic_name", [])
    generic = generic_names[0] if generic_names else ""
    appl_id = _label_id(label)
    name = brand or generic or appl_id
    manufacturer = (openfda.get("manufacturer_name") or [""])[0]
    route = ", ".join(openfda.get("route", []))
    product_type = (openfda.get("product_type") or [""])[0]

    lines: list[str] = [f"# {name}", ""]

    meta: list[str] = []
    if appl_id and appl_id != "UNKNOWN":
        meta.append(f"**Application:** {appl_id}")
    if generic and generic != brand:
        meta.append(f"**Generic:** {generic}")
    if manufacturer:
        meta.append(f"**Manufacturer:** {manufacturer}")
    if route:
        meta.append(f"**Route:** {route}")
    if product_type:
        meta.append(f"**Type:** {product_type}")
    lines.extend(meta)

    for key, heading in _LABEL_SECTIONS:
        text = _join_field(label.get(key))
        if text:
            lines += ["", f"## {heading}", "", text[:3000]]

    return "\n".join(lines)


class FdaLabelRenderer:
    """Renderer for openFDA drug label JSON.

    Detects ``{"results": [...]}`` from the openFDA /drug/label API or a
    single label record with ``openfda`` or ``indications_and_usage`` keys.
    Renders each label as an H1-headed markdown section with clinical sections
    in a consistent order.

    ``source_detail`` per chunk::

        {
            "application_id": "NDA019837",
            "brand_name":     "Tylenol",
            "generic_name":   "Acetaminophen",
            "manufacturer":   "...",
        }
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:
        labels = _iter_labels(obj)
        if not labels:
            return False
        for label in labels[:3]:
            if "openfda" in label or "indications_and_usage" in label:
                return True
        return False

    def render(self, obj: object) -> str:
        labels = _iter_labels(obj)
        return "\n\n".join(_render_one(lb) for lb in labels)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        meta: dict[str, dict[str, Any]] = {}
        rendered: dict[str, str] = {}
        for label in _iter_labels(obj):
            appl_id = _label_id(label)
            openfda = label.get("openfda", {})
            brand_names = openfda.get("brand_name", [])
            generic_names = openfda.get("generic_name", [])
            brand = brand_names[0] if brand_names else ""
            generic = generic_names[0] if generic_names else ""
            manufacturer = (openfda.get("manufacturer_name") or [""])[0]
            detail = {
                "application_id": appl_id,
                "brand_name": brand,
                "generic_name": generic,
                "manufacturer": manufacturer,
            }
            name_key = (brand or generic or appl_id).upper()
            meta[name_key] = {k: v for k, v in detail.items() if v}
            rendered[name_key] = _render_one(label)
            if appl_id and appl_id != "UNKNOWN":
                meta[appl_id.upper()] = meta[name_key]
                rendered[appl_id.upper()] = rendered[name_key]

        for chunk in chunks:
            matched_key = None
            m = _APPL_RE.search(chunk.content)
            if m:
                matched_key = m.group(0).upper()
            if not matched_key:
                for key in meta:
                    if key in chunk.content.upper():
                        matched_key = key
                        break
            if not matched_key:
                for part in chunk.section or []:
                    part_upper = part.strip().upper()
                    for key in meta:
                        if part_upper.startswith(key):
                            matched_key = key
                            break
                    if matched_key:
                        break
            if matched_key and matched_key in meta:
                chunk.source_detail = meta[matched_key]
                chunk.rendered_source = rendered[matched_key]

        return chunks
