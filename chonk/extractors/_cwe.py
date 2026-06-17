# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: a1b2c3d4-e5f6-7890-abcd-ef0123456789
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CweRenderer — renders MITRE CWE XML catalog into per-weakness markdown."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import DocumentChunk

_CWE_ID_RE = re.compile(r"\bCWE-(\d+)\b", re.IGNORECASE)


def _text(el: object) -> str:
    """Return stripped text content from an ElementTree element or dict."""
    if el is None:
        return ""
    if isinstance(el, str):
        return el.strip()
    if isinstance(el, dict):
        return (el.get("_text") or "").strip()
    # ElementTree element
    return (getattr(el, "text", None) or "").strip()


def _iter_weaknesses_from_dict(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract weakness dicts from a _to_dict() result of CWE XML root."""
    # Root is Weakness_Catalog, child Weaknesses contains Weakness elements
    weaknesses_container = obj.get("Weaknesses")
    if not weaknesses_container:
        return []
    if isinstance(weaknesses_container, list):
        weaknesses_container = weaknesses_container[0]
    raw = weaknesses_container.get("Weakness", [])
    if isinstance(raw, dict):
        raw = [raw]
    return raw


def _get_child_text(weakness: dict[str, Any], child_key: str) -> str:
    child = weakness.get(child_key)
    if child is None:
        return ""
    if isinstance(child, list):
        child = child[0]
    return _text(child)


def _render_one_dict(weakness: dict[str, Any]) -> str:
    cwe_id = weakness.get("ID", "")
    name = weakness.get("Name", "")
    description = _get_child_text(weakness, "Description")
    extended = _get_child_text(weakness, "Extended_Description")

    # Applicable_Platforms > Language
    platforms: list[str] = []
    ap = weakness.get("Applicable_Platforms")
    if ap:
        if isinstance(ap, list):
            ap = ap[0]
        for lang in ap.get("Language", []) if isinstance(ap, dict) else []:
            if isinstance(lang, dict):
                n = lang.get("Name") or lang.get("Class")
                if n:
                    platforms.append(n)

    # Related_Weaknesses > Related_Weakness
    related: list[str] = []
    rw = weakness.get("Related_Weaknesses")
    if rw:
        if isinstance(rw, list):
            rw = rw[0]
        for r in rw.get("Related_Weakness", []) if isinstance(rw, dict) else []:
            if isinstance(r, dict):
                rid = r.get("CWE_ID")
                nature = r.get("Nature", "")
                if rid:
                    related.append(f"CWE-{rid} ({nature})" if nature else f"CWE-{rid}")

    # Common_Consequences > Consequence
    consequences: list[str] = []
    cc = weakness.get("Common_Consequences")
    if cc:
        if isinstance(cc, list):
            cc = cc[0]
        for c in cc.get("Consequence", []) if isinstance(cc, dict) else []:
            if isinstance(c, dict):
                scope = c.get("Scope")
                if scope:
                    if isinstance(scope, list):
                        scope = scope[0]
                    s = _text(scope)
                    if s and s not in consequences:
                        consequences.append(s)

    lines: list[str] = [f"# CWE-{cwe_id} {name}", ""]

    if platforms:
        lines.append(f"**Platforms:** {', '.join(platforms)}")
    if related:
        lines.append(f"**Related:** {', '.join(related[:5])}")

    if description:
        lines += ["", "## Description", "", description]
    if extended:
        lines += ["", extended]

    if consequences:
        lines += ["", "## Consequences", ""]
        lines.extend(f"- {c}" for c in consequences)

    return "\n".join(lines)


class CweRenderer:
    """Renderer for MITRE CWE XML catalog (CWE List XML format).

    Detects the ``Weakness_Catalog`` root element.  Renders each non-deprecated
    weakness as an H1-headed markdown section with description, platforms,
    related weaknesses, and common consequences.

    ``source_detail`` per chunk::

        {
            "cwe_id":    "CWE-119",
            "name":      "Improper Restriction of Operations...",
            "platforms": ["C", "C++"],
        }
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:  # noqa: ARG002
        if not isinstance(obj, dict):
            return False
        tag = obj.get("_tag", "")
        return tag == "Weakness_Catalog" and "Weaknesses" in obj

    def render(self, obj: object) -> str:
        if not isinstance(obj, dict):
            return ""
        weaknesses = _iter_weaknesses_from_dict(obj)
        return "\n\n".join(_render_one_dict(w) for w in weaknesses)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        if not isinstance(obj, dict):
            return chunks
        weaknesses = _iter_weaknesses_from_dict(obj)

        meta: dict[str, dict[str, Any]] = {}
        rendered: dict[str, str] = {}
        for w in weaknesses:
            cwe_id = w.get("ID", "")
            if not cwe_id:
                continue
            ap = w.get("Applicable_Platforms")
            platforms: list[str] = []
            if ap:
                if isinstance(ap, list):
                    ap = ap[0]
                for lang in ap.get("Language", []) if isinstance(ap, dict) else []:
                    if isinstance(lang, dict):
                        n = lang.get("Name") or lang.get("Class")
                        if n:
                            platforms.append(n)
            meta[cwe_id] = {
                "cwe_id": f"CWE-{cwe_id}",
                "name": w.get("Name", ""),
                "platforms": platforms,
            }
            rendered[cwe_id] = _render_one_dict(w)

        for chunk in chunks:
            cwe_id = None
            m = _CWE_ID_RE.search(chunk.content)
            if m:
                cwe_id = m.group(1)
            if not cwe_id:
                for part in chunk.section or []:
                    sm = _CWE_ID_RE.match(part.strip())
                    if sm:
                        cwe_id = sm.group(1)
                        break
            if cwe_id and cwe_id in meta:
                chunk.source_detail = meta[cwe_id]
                chunk.rendered_source = rendered[cwe_id]

        return chunks
