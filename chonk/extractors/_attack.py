# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: d4e5f6a7-b8c9-0123-4567-89abcdef0123
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""AttackRenderer — renders MITRE ATT&CK STIX bundles into per-technique markdown."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentChunk

_ATTACK_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def _ext_id(obj: dict) -> str | None:
    """Return the ATT&CK technique ID (e.g. T1055.011) from external_references."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _ext_url(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("url")
    return None


def _is_live(obj: dict) -> bool:
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


class _Index:
    """Pre-built lookup tables over a STIX bundle."""

    def __init__(self, objects: list[dict]) -> None:
        self.by_id: dict[str, dict] = {o["id"]: o for o in objects if "id" in o}

        # course-of-action id → mitigation dict
        self.mitigations: dict[str, dict] = {
            o["id"]: o for o in objects if o.get("type") == "course-of-action" and _is_live(o)
        }

        # attack-pattern id → list[mitigation dict]
        self.tech_mitigations: dict[str, list[dict]] = {}
        # attack-pattern id → parent attack-pattern id (for sub-techniques)
        self.parent: dict[str, str] = {}

        for rel in objects:
            if rel.get("type") != "relationship":
                continue
            rtype = rel.get("relationship_type", "")
            src, tgt = rel.get("source_ref", ""), rel.get("target_ref", "")

            if rtype == "mitigates" and src in self.mitigations:
                self.tech_mitigations.setdefault(tgt, []).append(self.mitigations[src])
            elif rtype == "subtechnique-of":
                self.parent[src] = tgt


def _iter_techniques(obj: object) -> tuple[list[dict], _Index]:
    """Return (technique_list, index) from a STIX bundle dict."""
    if not isinstance(obj, dict):
        return [], _Index([])
    objects: list[dict] = obj.get("objects", [])
    techniques = [o for o in objects if o.get("type") == "attack-pattern" and _is_live(o)]
    return techniques, _Index(objects)


def _render_one(tech: dict, index: _Index) -> str:
    attack_id = _ext_id(tech) or tech.get("id", "UNKNOWN")
    name = tech.get("name", "")
    description = (tech.get("description") or "").strip()
    platforms = tech.get("x_mitre_platforms", [])
    tactics = [p["phase_name"] for p in tech.get("kill_chain_phases", [])]
    is_sub = tech.get("x_mitre_is_subtechnique", False)
    url = _ext_url(tech)

    parent_id: str | None = None
    if is_sub:
        parent_stix = index.parent.get(tech["id"])
        if parent_stix and parent_stix in index.by_id:
            parent_id = _ext_id(index.by_id[parent_stix])

    mitigations = index.tech_mitigations.get(tech["id"], [])

    lines: list[str] = [f"# {attack_id} {name}", ""]

    meta: list[str] = []
    if tactics:
        meta.append(f"**Tactics:** {', '.join(tactics)}")
    if platforms:
        meta.append(f"**Platforms:** {', '.join(platforms)}")
    if parent_id:
        meta.append(f"**Sub-technique of:** {parent_id}")
    lines.extend(meta)

    if description:
        lines += ["", "## Description", "", description]

    if mitigations:
        lines += ["", "## Mitigations", ""]
        for m in mitigations:
            mname = m.get("name", "")
            mdesc = (m.get("description") or "").strip()
            # first sentence only to keep chunks tight
            first_sentence = mdesc.split(". ")[0].rstrip(".") + "." if mdesc else ""
            if first_sentence:
                lines.append(f"- **{mname}** — {first_sentence}")
            else:
                lines.append(f"- {mname}")

    if url:
        lines += ["", "## References", "", f"- {url}"]

    return "\n".join(lines)


class AttackRenderer:
    """Renderer for MITRE ATT&CK STIX 2.x bundles (enterprise, mobile, ics).

    Detects any STIX bundle containing ``attack-pattern`` objects with
    MITRE ATT&CK external references.  Renders each non-deprecated technique
    as an H1-headed markdown section with tactics, platforms, parent
    sub-technique link, mitigations (first sentence), and ATT&CK URL.

    ``source_detail`` per chunk::

        {
            "attack_id":      "T1055.011",
            "name":           "Extra Window Memory Injection",
            "tactics":        ["privilege-escalation", "defense-evasion"],
            "platforms":      ["Windows"],
            "is_subtechnique": True,
            "parent_id":      "T1055",   # omitted if not a sub-technique
            "stix_id":        "attack-pattern--...",
        }
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:  # noqa: ARG002
        if not isinstance(obj, dict):
            return False
        objects = obj.get("objects", [])
        if not isinstance(objects, list) or not objects:
            return False
        # Require at least one live attack-pattern with a mitre-attack external ref
        for o in objects:
            if o.get("type") == "attack-pattern" and _is_live(o) and _ext_id(o):
                return True
        return False

    def render(self, obj: object) -> str:
        techniques, index = _iter_techniques(obj)
        return "\n\n".join(_render_one(t, index) for t in techniques)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        techniques, index = _iter_techniques(obj)

        meta: dict[str, dict] = {}
        rendered_map: dict[str, str] = {}
        for tech in techniques:
            attack_id = _ext_id(tech)
            if not attack_id:
                continue
            tactics = [p["phase_name"] for p in tech.get("kill_chain_phases", [])]
            platforms = tech.get("x_mitre_platforms", [])
            is_sub = tech.get("x_mitre_is_subtechnique", False)
            parent_id: str | None = None
            if is_sub:
                parent_stix = index.parent.get(tech["id"])
                if parent_stix and parent_stix in index.by_id:
                    parent_id = _ext_id(index.by_id[parent_stix])
            detail: dict = {
                "attack_id": attack_id,
                "name": tech.get("name", ""),
                "tactics": tactics,
                "platforms": platforms,
                "is_subtechnique": is_sub,
                "stix_id": tech["id"],
            }
            if parent_id:
                detail["parent_id"] = parent_id
            meta[attack_id] = detail
            rendered_map[attack_id] = _render_one(tech, index)

        for chunk in chunks:
            attack_id = None
            m = _ATTACK_ID_RE.search(chunk.content)
            if m:
                attack_id = m.group(0)
            if not attack_id:
                for part in chunk.section or []:
                    sm = _ATTACK_ID_RE.match(part.strip())
                    if sm:
                        attack_id = sm.group(0)
                        break
            if attack_id and attack_id in meta:
                chunk.source_detail = meta[attack_id]
                chunk.rendered_source = rendered_map[attack_id]

        return chunks
