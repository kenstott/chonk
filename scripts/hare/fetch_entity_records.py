# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Build the HARE-Bench entity-resolution records (the "gleif" corpus source).

Source data:
  - GLEIF golden copy (goldencopy.gleif.org, CC0): the Level 1 LEI records file
    (lei2) and the Level 2 relationship records file (rr).
  - SEC EDGAR submissions (data.sec.gov) for the five parents (CIKs from the
    corpus's 10-K manifest) and for any entity whose GLEIF registration authority
    is SEC EDGAR (RA000665), which makes its registration number a CIK.

Selection: the five FANG parent companies plus every entity GLEIF reports as
consolidated under them (ACTIVE IS_DIRECTLY_CONSOLIDATED_BY edges, followed
transitively, and ACTIVE IS_ULTIMATELY_CONSOLIDATED_BY edges).

Output (in --out-dir): one Markdown document per entity, the selected lei2 and rr
rows as CSV, the raw SEC responses, and provenance.json recording the golden-copy
files and their SHA-256 digests.

This script regenerates the source data; it is not part of reproducing the
benchmark. The golden copy is republished three times a day and GLEIF keeps no
public archive, so the benchmark uses the frozen output published with the
corpus (scripts/hare/artifacts.py download).

SEC requires a User-Agent that identifies the requester with a contact email:
https://www.sec.gov/os/accessing-edgar-data

Usage:
    python scripts/hare/fetch_entity_records.py --sec-user-agent "Name email@example.com"
    python scripts/hare/fetch_entity_records.py --sec-user-agent "..." \\
        --lei2-zip path/to/lei2.csv.zip --rr-zip path/to/rr.csv.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import time
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "work" / "corpus" / "gleif"
GOLDEN_COPY = "https://goldencopy.gleif.org/api/v2/golden-copies/publishes/{kind}/latest.csv"
SEC_API = "https://data.sec.gov/submissions"
GLEIF_USER_AGENT = "chonk-benchmark https://github.com/kenstott/chonk"
SEC_RA = "RA000665"  # GLEIF registration-authority code for SEC EDGAR

SEED_LEIS = {
    "HWUPKR0MPOU8FGXBT394": "Apple Inc.",
    "5493006MHB84DD0ZWV18": "Alphabet Inc.",
    "BQ4BKCS1HXDV9HN80Z93": "Meta Platforms, Inc.",
    "ZXTILKJKG63JELOEG630": "Amazon.com, Inc.",
    "549300Y7VHGU0I7CE873": "Netflix, Inc.",
}
# The seeds are registered with state registries, not SEC EDGAR, so GLEIF carries
# no CIK for them. These are the CIKs of the 10-K filers in the corpus
# (work/corpus/10k/manifest.json).
SEED_CIKS = {
    "HWUPKR0MPOU8FGXBT394": 320193,
    "5493006MHB84DD0ZWV18": 1652044,
    "BQ4BKCS1HXDV9HN80Z93": 1326801,
    "ZXTILKJKG63JELOEG630": 1018724,
    "549300Y7VHGU0I7CE873": 1065280,
}
DIRECT = "IS_DIRECTLY_CONSOLIDATED_BY"
ULTIMATE = "IS_ULTIMATELY_CONSOLIDATED_BY"

Row = dict[str, str]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(kind: str, dest_dir: Path) -> tuple[Path, str]:
    """Download the latest golden-copy CSV zip; return (path, resolved URL)."""
    req = urllib.request.Request(
        GOLDEN_COPY.format(kind=kind), headers={"User-Agent": GLEIF_USER_AGENT}
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        url = resp.geturl()
        path = dest_dir / url.rsplit("/", 1)[-1]
        with path.open("wb") as out:
            shutil.copyfileobj(resp, out)
    return path, url


def _rows(zip_path: Path) -> Iterator[Row]:
    with zipfile.ZipFile(zip_path) as z:
        (name,) = z.namelist()
        with z.open(name) as fh:
            yield from csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))


def select_family(rr_rows: list[Row]) -> set[str]:
    """Seed LEIs plus every entity consolidated under them."""
    children: dict[str, set[str]] = defaultdict(set)
    for r in rr_rows:
        children[r["Relationship.EndNode.NodeID"]].add(r["Relationship.StartNode.NodeID"])
    family = set(SEED_LEIS)
    stack = list(SEED_LEIS)
    while stack:
        for child in children[stack.pop()]:
            if child not in family:
                family.add(child)
                stack.append(child)
    return family


def _address(row: Row, prefix: str) -> str:
    keys = [
        "FirstAddressLine",
        "AdditionalAddressLine.1",
        "AdditionalAddressLine.2",
        "AdditionalAddressLine.3",
        "City",
        "Region",
        "PostalCode",
        "Country",
    ]
    return ", ".join(row[f"{prefix}.{k}"] for k in keys if row[f"{prefix}.{k}"])


def _other_names(row: Row) -> list[str]:
    names = []
    for i in range(1, 6):
        n = row[f"Entity.OtherEntityNames.OtherEntityName.{i}"]
        if n:
            kind = row[f"Entity.OtherEntityNames.OtherEntityName.{i}.type"]
            names.append(f"{n} ({kind.replace('_', ' ').lower()})")
    return names


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]


def _sec_lines(sec: dict[str, Any]) -> list[str]:
    tickers = [f"{t} ({x})" for t, x in zip(sec["tickers"], sec["exchanges"], strict=True)]
    former = [f"{f['name']} (until {f['to'][:10]})" for f in sec["formerNames"]]
    return [
        f"**SEC name:** {sec['name']}",
        f"**CIK:** {int(sec['cik']):010d}",
        f"**Tickers:** {', '.join(tickers) if tickers else 'none'}",
        f"**Former names:** {'; '.join(former) if former else 'none'}",
        f"**State of incorporation:** {sec['stateOfIncorporation'] or 'not reported'}",
        f"**SIC:** {sec['sic']} {sec['sicDescription']}",
    ]


def render(
    row: Row,
    names: dict[str, str],
    parents: dict[str, str],
    ultimate: dict[str, str],
    children: dict[str, list[str]],
    sec: dict[str, Any] | None,
    publish: str,
) -> str:
    lei = row["LEI"]
    name = row["Entity.LegalName"]
    other = _other_names(row)
    ra = row["Entity.RegistrationAuthority.RegistrationAuthorityID"]
    ra_id = row["Entity.RegistrationAuthority.RegistrationAuthorityEntityID"]

    def who(parent: str | None) -> str:
        return f"{names[parent]} (LEI {parent})" if parent else "none reported"

    subs = sorted(names[c] for c in children.get(lei, []))
    lines = [
        f"# {name} — Legal Entity Record",
        "",
        "## GLEIF LEI record",
        f"**LEI:** {lei}",
        f"**Legal name:** {name}",
        f"**Other names:** {'; '.join(other) if other else 'none'}",
        f"**Legal jurisdiction:** {row['Entity.LegalJurisdiction']}",
        f"**Legal form code:** {row['Entity.LegalForm.EntityLegalFormCode']}",
        f"**Entity category:** {row['Entity.EntityCategory'] or 'not reported'}",
        f"**Entity status:** {row['Entity.EntityStatus']}",
        f"**Registration status:** {row['Registration.RegistrationStatus']}",
        f"**Registration authority:** {ra} (entity ID {ra_id or 'not reported'})",
        f"**Legal address:** {_address(row, 'Entity.LegalAddress')}",
        f"**Headquarters address:** {_address(row, 'Entity.HeadquartersAddress')}",
        f"**Direct parent:** {who(parents.get(lei))}",
        f"**Ultimate parent:** {who(ultimate.get(lei))}",
        f"**Direct subsidiaries:** {'; '.join(subs) if subs else 'none reported'}",
        "",
        "## SEC EDGAR record",
    ]
    if sec is None:
        lines.append("No SEC EDGAR record.")
    else:
        lines += _sec_lines(sec)
    lines += ["", f"*Sources: GLEIF golden copy {publish} and SEC EDGAR.*", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HARE-Bench entity-resolution records.")
    parser.add_argument(
        "--sec-user-agent",
        required=True,
        help='SEC-compliant User-Agent, e.g. "Name email@example.com"',
    )
    parser.add_argument("--lei2-zip", type=Path, help="local lei2 golden-copy CSV zip")
    parser.add_argument("--rr-zip", type=Path, help="local rr golden-copy CSV zip")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "work" / "gleif_golden_copy",
        help="where downloaded golden-copy zips are kept",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if (args.lei2_zip is None) != (args.rr_zip is None):
        parser.error("--lei2-zip and --rr-zip must be given together")

    if args.lei2_zip is not None:
        lei2_zip, rr_zip = args.lei2_zip, args.rr_zip
        lei2_url = rr_url = "local file"
    else:
        print("Downloading GLEIF golden copy (lei2 ~480 MB, rr ~25 MB)...")
        lei2_zip, lei2_url = _download("lei2", args.cache_dir)
        rr_zip, rr_url = _download("rr", args.cache_dir)
    publish = lei2_zip.name[:13]  # YYYYMMDD-HHMM prefix of the golden-copy file name

    rr_rows = [
        r
        for r in _rows(rr_zip)
        if r["Relationship.RelationshipStatus"] == "ACTIVE"
        and r["Relationship.RelationshipType"] in (DIRECT, ULTIMATE)
    ]
    family = select_family(rr_rows)
    fam_rr = [
        r
        for r in rr_rows
        if r["Relationship.StartNode.NodeID"] in family
        or r["Relationship.EndNode.NodeID"] in family
    ]
    parents = {
        r["Relationship.StartNode.NodeID"]: r["Relationship.EndNode.NodeID"]
        for r in fam_rr
        if r["Relationship.RelationshipType"] == DIRECT
    }
    ultimate = {
        r["Relationship.StartNode.NodeID"]: r["Relationship.EndNode.NodeID"]
        for r in fam_rr
        if r["Relationship.RelationshipType"] == ULTIMATE
    }
    children: dict[str, list[str]] = defaultdict(list)
    for child, parent in parents.items():
        children[parent].append(child)

    fam_lei2 = [r for r in _rows(lei2_zip) if r["LEI"] in family]
    missing = family - {r["LEI"] for r in fam_lei2}
    if missing:
        raise RuntimeError(f"LEIs in rr but not in lei2: {sorted(missing)}")
    names = {r["LEI"]: r["Entity.LegalName"] for r in fam_lei2}

    out: Path = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "raw").mkdir(parents=True)
    for rows, fname in ((fam_lei2, "family_lei2.csv"), (fam_rr, "family_rr.csv")):
        with (out / "raw" / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: tuple(r.values())))

    n_sec = 0
    for row in sorted(fam_lei2, key=lambda r: r["LEI"]):
        sec = None
        cik: int | None = SEED_CIKS.get(row["LEI"])
        if row["Entity.RegistrationAuthority.RegistrationAuthorityID"] == SEC_RA:
            cik = int(row["Entity.RegistrationAuthority.RegistrationAuthorityEntityID"])
        if cik is not None:
            req = urllib.request.Request(
                f"{SEC_API}/CIK{cik:010d}.json", headers={"User-Agent": args.sec_user_agent}
            )
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                body = resp.read()
            sec = json.loads(body)
            (out / "raw" / f"sec_CIK{cik:010d}.json").write_bytes(body)
            n_sec += 1
            time.sleep(0.2)  # SEC fair-access limit is 10 requests/second
        doc = render(row, names, parents, ultimate, children, sec, publish)
        (out / f"{_slug(row['Entity.LegalName'])}_{row['LEI']}.md").write_text(doc)

    provenance = {
        "generated": date.today().isoformat(),
        "selection": "seed LEIs plus entities consolidated under them (ACTIVE "
        f"{DIRECT}, transitive; ACTIVE {ULTIMATE})",
        "seed_leis": SEED_LEIS,
        "entities": len(fam_lei2),
        "sec_records": n_sec,
        "golden_copy": {
            "lei2": {"file": lei2_zip.name, "url": lei2_url, "sha256": _sha256(lei2_zip)},
            "rr": {"file": rr_zip.name, "url": rr_url, "sha256": _sha256(rr_zip)},
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(fam_lei2)} entity records ({n_sec} with SEC data) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
