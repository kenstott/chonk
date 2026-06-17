# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: c3d4e5f6-a7b8-9012-cdef-012345678901
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ClinicalTrialRenderer — renders ClinicalTrials.gov API v2 study records."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import DocumentChunk

_NCT_ID_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


def _iter_studies(obj: object) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        if "studies" in obj:
            return [s for s in obj["studies"] if isinstance(s, dict)]
        # Single study
        if "protocolSection" in obj:
            return [obj]
    if isinstance(obj, list):
        return [s for s in obj if isinstance(s, dict) and "protocolSection" in s]
    return []


def _render_one(study: dict[str, Any]) -> str:
    ps = study.get("protocolSection", {})

    ident = ps.get("identificationModule", {})
    nct_id = ident.get("nctId", "UNKNOWN")
    title = ident.get("briefTitle", ident.get("officialTitle", ""))

    status_mod = ps.get("statusModule", {})
    status = status_mod.get("overallStatus", "")
    phase_list = ps.get("designModule", {}).get("phases", [])
    phases = ", ".join(phase_list) if phase_list else ""

    desc_mod = ps.get("descriptionModule", {})
    brief = (desc_mod.get("briefSummary") or "").strip()
    detailed = (desc_mod.get("detailedDescription") or "").strip()

    conditions = ps.get("conditionsModule", {}).get("conditions", [])

    arms_mod = ps.get("armsInterventionsModule", {})
    interventions: list[str] = []
    for iv in arms_mod.get("interventions", []):
        itype = iv.get("type", "")
        iname = iv.get("name", "")
        if iname:
            interventions.append(f"{itype}: {iname}" if itype else iname)

    elig_mod = ps.get("eligibilityModule", {})
    eligibility = (elig_mod.get("eligibilityCriteria") or "").strip()
    min_age = elig_mod.get("minimumAge", "")
    max_age = elig_mod.get("maximumAge", "")
    sex = elig_mod.get("sex", "")

    lines: list[str] = [f"# {nct_id} {title}", ""]

    meta: list[str] = []
    if status:
        meta.append(f"**Status:** {status}")
    if phases:
        meta.append(f"**Phase:** {phases}")
    if conditions:
        meta.append(f"**Conditions:** {', '.join(conditions[:5])}")
    lines.extend(meta)

    if brief:
        lines += ["", "## Summary", "", brief]

    if detailed:
        lines += ["", "## Detailed Description", "", detailed]

    if interventions:
        lines += ["", "## Interventions", ""]
        lines.extend(f"- {iv}" for iv in interventions[:10])

    if eligibility:
        age_parts = [p for p in [min_age, max_age] if p]
        age_str = " to ".join(age_parts) if age_parts else ""
        elig_header = f"## Eligibility{' (' + sex + ', ' + age_str + ')' if sex or age_str else ''}"
        lines += ["", elig_header, "", eligibility[:2000]]

    return "\n".join(lines)


class ClinicalTrialRenderer:
    """Renderer for ClinicalTrials.gov API v2 study JSON.

    Detects ``{"studies": [...]}`` or single study with ``protocolSection``.
    Renders each study as an H1-headed markdown section with status, phases,
    conditions, summary, interventions, and eligibility criteria.

    ``source_detail`` per chunk::

        {
            "nct_id":     "NCT04280705",
            "title":      "...",
            "status":     "COMPLETED",
            "phases":     ["PHASE3"],
            "conditions": ["COVID-19"],
        }
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:
        studies = _iter_studies(obj)
        if not studies:
            return False
        # Verify at least one has a protocolSection with identificationModule
        for s in studies[:3]:
            ps = s.get("protocolSection", {})
            if ps.get("identificationModule", {}).get("nctId"):
                return True
        return False

    def render(self, obj: object) -> str:
        studies = _iter_studies(obj)
        return "\n\n".join(_render_one(s) for s in studies)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        meta: dict[str, dict[str, Any]] = {}
        rendered: dict[str, str] = {}
        for study in _iter_studies(obj):
            ps = study.get("protocolSection", {})
            ident = ps.get("identificationModule", {})
            nct_id = ident.get("nctId")
            if not nct_id:
                continue
            phase_list = ps.get("designModule", {}).get("phases", [])
            conditions = ps.get("conditionsModule", {}).get("conditions", [])
            meta[nct_id.upper()] = {
                "nct_id": nct_id,
                "title": ident.get("briefTitle", ""),
                "status": ps.get("statusModule", {}).get("overallStatus", ""),
                "phases": phase_list,
                "conditions": conditions,
            }
            rendered[nct_id.upper()] = _render_one(study)

        for chunk in chunks:
            nct_id = None
            m = _NCT_ID_RE.search(chunk.content)
            if m:
                nct_id = m.group(0).upper()
            if not nct_id:
                for part in chunk.section or []:
                    sm = _NCT_ID_RE.match(part.strip())
                    if sm:
                        nct_id = sm.group(0).upper()
                        break
            if nct_id and nct_id in meta:
                chunk.source_detail = meta[nct_id]
                chunk.rendered_source = rendered[nct_id]

        return chunks
