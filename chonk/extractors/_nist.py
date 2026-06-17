# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: b2c3d4e5-f6a7-8901-bcde-f01234567890
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""NistRenderer — renders NIST OSCAL SP 800-53 catalog JSON into per-control markdown."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import DocumentChunk

# e.g. "ac-1", "ac-1.a", "si-3.2"
_CTRL_ID_RE = re.compile(r"\b([a-z]{2}-\d+(?:\.\d+)?)\b", re.IGNORECASE)


def _iter_controls(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten OSCAL catalog groups[] → controls[] (with nested controls)."""
    catalog = obj.get("catalog", obj)
    controls: list[dict[str, Any]] = []
    for group in catalog.get("groups", []):
        _collect_controls(group, controls)
    return controls


def _collect_controls(node: dict[str, Any], out: list[dict[str, Any]]) -> None:
    for ctrl in node.get("controls", []):
        out.append(ctrl)
        _collect_controls(ctrl, out)


def _parts_text(parts: list[dict[str, Any]], depth: int = 0) -> list[str]:
    """Recursively extract prose from OSCAL part trees."""
    lines: list[str] = []
    for part in parts:
        prose = (part.get("prose") or "").strip()
        sub = part.get("parts", [])
        if prose:
            prefix = "  " * depth
            lines.append(f"{prefix}{prose}")
        if sub:
            lines.extend(_parts_text(sub, depth + 1))
    return lines


def _render_one(ctrl: dict[str, Any]) -> str:
    ctrl_id = ctrl.get("id", "UNKNOWN").upper()
    title = ctrl.get("title", "")
    parts = ctrl.get("parts", [])

    lines: list[str] = [f"# {ctrl_id} {title}", ""]

    prose_lines = _parts_text(parts)
    if prose_lines:
        lines += ["## Statement", ""] + prose_lines

    # Sub-controls as ## sections
    for sub in ctrl.get("controls", []):
        sub_id = sub.get("id", "").upper()
        sub_title = sub.get("title", "")
        lines += ["", f"## {sub_id} {sub_title}"]
        sub_prose = _parts_text(sub.get("parts", []))
        if sub_prose:
            lines += [""] + sub_prose

    return "\n".join(lines)


class NistRenderer:
    """Renderer for NIST OSCAL SP 800-53 catalog JSON.

    Detects any JSON with a ``catalog.groups`` structure containing OSCAL
    controls.  Renders each control as an H1 section with statement prose
    and any nested sub-controls.

    ``source_detail`` per chunk::

        {
            "control_id": "AC-1",
            "title":      "Policy and Procedures",
            "group":      "Access Control",
        }
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:  # noqa: ARG002
        if not isinstance(obj, dict):
            return False
        catalog = obj.get("catalog", obj)
        if not isinstance(catalog, dict):
            return False
        groups = catalog.get("groups", [])
        if not isinstance(groups, list) or not groups:
            return False
        # Must have at least one control with an OSCAL-style id
        for group in groups[:3]:
            for ctrl in group.get("controls", [])[:3]:
                cid = ctrl.get("id", "")
                if re.match(r"^[a-z]{2}-\d+", cid):
                    return True
        return False

    def render(self, obj: object) -> str:
        if not isinstance(obj, dict):
            return ""
        controls = _iter_controls(obj)
        return "\n\n".join(_render_one(c) for c in controls)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        if not isinstance(obj, dict):
            return chunks

        # Build group title map: control_id → group title
        group_map: dict[str, str] = {}
        catalog = obj.get("catalog", obj)
        for group in catalog.get("groups", []):
            gtitle = group.get("title", "")
            for ctrl in group.get("controls", []):
                cid = (ctrl.get("id") or "").upper()
                if cid:
                    group_map[cid] = gtitle

        meta: dict[str, dict[str, Any]] = {}
        rendered: dict[str, str] = {}
        for ctrl in _iter_controls(obj):
            cid = (ctrl.get("id") or "").upper()
            if not cid:
                continue
            meta[cid] = {
                "control_id": cid,
                "title": ctrl.get("title", ""),
                "group": group_map.get(cid, ""),
            }
            rendered[cid] = _render_one(ctrl)

        for chunk in chunks:
            ctrl_id = None
            m = _CTRL_ID_RE.search(chunk.content)
            if m:
                ctrl_id = m.group(1).upper()
            if not ctrl_id:
                for part in chunk.section or []:
                    sm = _CTRL_ID_RE.match(part.strip())
                    if sm:
                        ctrl_id = sm.group(1).upper()
                        break
            if ctrl_id and ctrl_id in meta:
                chunk.source_detail = meta[ctrl_id]
                chunk.rendered_source = rendered[ctrl_id]

        return chunks
