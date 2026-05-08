# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: f1e2d3c4-b5a6-7890-abcd-ef1234567890
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CveRenderer — renders NVD API v2 CVE records into per-record markdown."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentChunk

_CVE_ID_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)

# NVD CVSS metric keys in descending preference (highest version first)
_CVSS_KEYS = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")


def _en(items: list[dict]) -> str:
    for item in items:
        if item.get("lang") == "en":
            return item.get("value", "").strip()
    return items[0].get("value", "").strip() if items else ""


def _best_cvss(metrics: dict) -> tuple[float | None, str | None]:
    for key in _CVSS_KEYS:
        for entry in metrics.get(key, []):
            data = entry.get("cvssData", {})
            score = data.get("baseScore")
            severity = (
                data.get("baseSeverity") or entry.get("baseSeverity") or entry.get("severity")
            )
            if score is not None:
                return float(score), (severity or "").upper() or None
    return None, None


def _cpe_readable(criteria: str, match: dict) -> str:
    parts = criteria.split(":")
    base = ""
    if len(parts) >= 6:
        vendor = parts[3].replace("_", " ")
        product = parts[4].replace("_", " ")
        base = f"{vendor}/{product}"
    else:
        base = criteria

    ranges = []
    if match.get("versionStartIncluding"):
        ranges.append(f">= {match['versionStartIncluding']}")
    if match.get("versionStartExcluding"):
        ranges.append(f"> {match['versionStartExcluding']}")
    if match.get("versionEndIncluding"):
        ranges.append(f"<= {match['versionEndIncluding']}")
    if match.get("versionEndExcluding"):
        ranges.append(f"< {match['versionEndExcluding']}")

    if ranges:
        return f"{base} ({', '.join(ranges)})"
    version = parts[5] if len(parts) >= 6 and parts[5] != "*" else ""
    return f"{base} {version}".strip()


def _iter_cves(obj: object) -> list[dict]:
    if isinstance(obj, dict):
        if "vulnerabilities" in obj:
            return [v["cve"] for v in obj["vulnerabilities"] if "cve" in v]
        if str(obj.get("id", "")).upper().startswith("CVE-"):
            return [obj]
        if "cve" in obj:
            return [obj["cve"]]
    if isinstance(obj, list):
        out: list[dict] = []
        for item in obj:
            out.extend(_iter_cves(item))
        return out
    return []


def _render_one(cve: dict) -> str:
    cve_id = cve.get("id", "UNKNOWN")
    published = (cve.get("published") or "")[:10]
    modified = (cve.get("lastModified") or "")[:10]
    status = cve.get("vulnStatus", "")

    description = _en(cve.get("descriptions", []))
    score, severity = _best_cvss(cve.get("metrics", {}))

    weaknesses: list[str] = []
    for w in cve.get("weaknesses", []):
        val = _en(w.get("description", []))
        if val and val not in weaknesses:
            weaknesses.append(val)

    affected: list[str] = []
    for cfg in cve.get("configurations", []):
        for node in cfg.get("nodes", []):
            for m in node.get("cpeMatch", []):
                if not m.get("vulnerable"):
                    continue
                readable = _cpe_readable(m.get("criteria", ""), m)
                if readable not in affected:
                    affected.append(readable)

    references = [r["url"] for r in cve.get("references", []) if r.get("url")]

    lines: list[str] = [f"# {cve_id}", ""]

    meta: list[str] = []
    if status:
        meta.append(f"**Status:** {status}")
    if published:
        meta.append(f"**Published:** {published}")
    if modified:
        meta.append(f"**Last Modified:** {modified}")
    if score is not None:
        sev = f" ({severity})" if severity else ""
        meta.append(f"**CVSS Score:** {score}{sev}")
    if weaknesses:
        meta.append(f"**CWE:** {', '.join(weaknesses)}")
    lines.extend(meta)

    if description:
        lines += ["", "## Description", "", description]

    if affected:
        lines += ["", "## Affected", ""]
        lines.extend(f"- {a}" for a in affected)

    if references:
        lines += ["", "## References", ""]
        lines.extend(f"- {u}" for u in references[:10])

    return "\n".join(lines)


class CveRenderer:
    """Renderer for NVD CVE JSON (API v2).

    Detects any JSON object containing a ``vulnerabilities`` array or a
    top-level ``id`` that looks like a CVE ID.  Renders each record as an
    H1-headed markdown section so ``chunk_document`` splits naturally at
    Description / Affected / References.
    """

    def can_render(self, source_path: str | None, obj: object) -> bool:
        if not isinstance(obj, (dict, list)):
            return False
        cves = _iter_cves(obj)
        return bool(cves)

    def render(self, obj: object) -> str:
        cves = _iter_cves(obj)
        # H1 headings reset section context; no separator needed between records
        return "\n\n".join(_render_one(c) for c in cves)

    def annotate(
        self,
        chunks: list[DocumentChunk],
        obj: object,
    ) -> list[DocumentChunk]:
        meta: dict[str, dict] = {}
        for cve in _iter_cves(obj):
            cve_id = cve.get("id")
            if not cve_id:
                continue
            score, severity = _best_cvss(cve.get("metrics", {}))
            detail = {
                "cve_id": cve_id,
                "published": (cve.get("published") or "")[:10] or None,
                "cvss_score": score,
                "severity": severity,
                "vuln_status": cve.get("vulnStatus") or None,
            }
            meta[cve_id.upper()] = {k: v for k, v in detail.items() if v is not None}

        for chunk in chunks:
            # Try content first, then section path (sub-sections lack the H1 text)
            cve_id = None
            m = _CVE_ID_RE.search(chunk.content)
            if m:
                cve_id = m.group(0).upper()
            if not cve_id:
                for part in chunk.section or []:
                    sm = _CVE_ID_RE.match(part.strip())
                    if sm:
                        cve_id = sm.group(0).upper()
                        break
            if cve_id and cve_id in meta:
                chunk.source_detail = meta[cve_id]

        return chunks
