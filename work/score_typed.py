#!/usr/bin/env python3
"""
Type-aware scorer for FANG gold answers.

Loads gold schemas (from annotate_gold_schemas.py) and a results JSONL,
scores each generated answer using the appropriate check type, and outputs
per-question scores + aggregate domain breakdown.

Check types:
  boolean   → extracts yes/no from generated answer; exact match with gold bool
  number    → extracts first float; score 1.0 if within tolerance, 0.0 if wrong,
               0.0 if answer is an abstention ("not in context" / "cannot determine")
  entity    → normalizes entity strings; score = F1 over gold set
  text      → cosine similarity between sentence embeddings (falls back to ROUGE-L)

Usage:
    python work/score_typed.py \
        --schemas work/fang2026/data/fang2026_gold_schemas.jsonl \
        --results work/fang2026/results/<run>_rp.jsonl \
        --out work/fang2026/results/<run>_typed_scores.json \
        [--local-embed-model BAAI/bge-large-en-v1.5]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Abstention / hallucination detection ────────────────────────────────────

_ABSTENTION_PHRASES = [
    "not in context", "not in the context", "not mentioned", "not appear",
    "no information", "cannot determine", "cannot be answered", "unable to",
    "does not contain", "not provided", "not available", "context does not",
    "context provided does not", "not found in", "no evidence",
    "not present in", "does not exist in", "not exist in", "no mention of",
    "not included in", "absent from the", "not referenced", "not listed in",
]

# Model hedges its source while still asserting an answer — signals confabulation.
_HALLUCINATION_PHRASES = [
    "based on the context", "based on the information provided",
    "based on the provided context", "based on the given context",
    "according to the context", "the context suggests", "the context indicates",
    "from the context", "in the context provided", "as per the context",
    "the provided context", "context provided", "and based on the context",
]


def _is_abstention(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _ABSTENTION_PHRASES)


def _is_hallucination_hedge(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _HALLUCINATION_PHRASES)


# ── Boolean scorer ──────────────────────────────────────────────────────────

_YES_RE = re.compile(r'\b(yes|true|correct|affirmative|indeed|both|same)\b', re.I)
_NO_RE  = re.compile(r'\b(no\b|false|incorrect|not the same|different|neither|absent|none)\b', re.I)


def _extract_bool(text: str) -> bool | None:
    t = text.strip()
    yes = bool(_YES_RE.search(t))
    no  = bool(_NO_RE.search(t))
    if yes and not no:
        return True
    if no and not yes:
        return False
    # first word heuristic
    first = t.split()[0].lower().rstrip(".,!") if t else ""
    if first in ("yes", "true"):
        return True
    if first in ("no", "false"):
        return False
    # last-sentence heuristic: hedged answers often conclude with the real verdict
    sentences = re.split(r'(?<=[.!?])\s+', t)
    for sent in reversed(sentences):
        s = sent.strip()
        if not s:
            continue
        s_yes = bool(_YES_RE.search(s))
        s_no  = bool(_NO_RE.search(s))
        if s_yes and not s_no:
            return True
        if s_no and not s_yes:
            return False
    return None


def score_boolean(generated: str, gold_value: bool) -> float:
    # Epistemic Fidelity: abstention earns partial credit; hallucination earns none.
    if _is_abstention(generated):
        return 0.3
    pred = _extract_bool(generated)
    if pred is not None:
        # Hallucination hedge with assertable answer → 0.0 regardless of match.
        if _is_hallucination_hedge(generated):
            return 0.0
        return 1.0 if pred == gold_value else 0.0
    # Unparseable ("maybe") — including hedge-without-answer — earns 0.3.
    return 0.3


# ── Date scorer ─────────────────────────────────────────────────────────────

import re as _re

_DATE_RE = _re.compile(
    r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})'          # YYYY-MM-DD / YYYY/M/D
    r'|(\d{1,2})[-/](\d{1,2})[-/](\d{4})'            # MM-DD-YYYY / M/D/YYYY
    r'|(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4})'  # Month D, YYYY
    r'|(\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{4})',   # D Month YYYY
    _re.IGNORECASE,
)

def _parse_date(text: str):
    m = _DATE_RE.search(text)
    if not m:
        return None
    s = m.group(0).strip().rstrip(",")
    # Try ISO-ish first
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%m/%d/%Y",
                "%B %d %Y", "%B %d, %Y", "%b %d %Y", "%b %d, %Y",
                "%d %B %Y", "%d %b %Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def score_date(generated: str, gold_value: str, tolerance_days: int = 0) -> float:
    if _is_abstention(generated):
        return 0.3
    gold_date = _parse_date(gold_value)
    if gold_date is None:
        return float("nan")
    pred_date = _parse_date(generated)
    if pred_date is None:
        return 0.0
    diff = abs((pred_date - gold_date).days)
    return 1.0 if diff <= tolerance_days else 0.0


# ── Number scorer ───────────────────────────────────────────────────────────

_FLOAT_RE = re.compile(r'(?<![A-Za-z])-?\d+(?:\.\d+)?')
_CVE_RE  = re.compile(r'CVE-\d{4}-\d+', re.IGNORECASE)
_PATENT_ID_RE = re.compile(r'\b(US)?\d{7,8}\b')
_DATE_STR_RE = re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b')
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')

# Expand bare financial abbreviations to full words before quantulum3 parsing.
# MM and M are both treated as million; B as billion; K as thousand; T as trillion.
# Only matches when immediately after a digit (no $ prefix — those work natively).
_FIN_ABBREV_RE = re.compile(r'(?<=\d)\s*(MM|B|M|K|T)\b', re.IGNORECASE)
_FIN_ABBREV_MAP = {'mm': 'million', 'b': 'billion', 'm': 'million', 'k': 'thousand', 't': 'trillion'}

# Scale words → multiplier — used as fallback when quantulum3 is unavailable
_SCALE_MAP = {
    'trillion': 1e12, 'trillions': 1e12,
    'billion':  1e9,  'billions':  1e9,
    'million':  1e6,  'millions':  1e6,
    'thousand': 1e3,  'thousands': 1e3,
    'mm': 1e6, 'b': 1e9, 'm': 1e6, 'k': 1024, 't': 1e12,
}
_SCALE_WORD_RE = re.compile(
    r'\b(trillion|billion|million|thousand)s?\b'
    r'|(?<=\d)(MM|B|M|K|T)\b',
    re.IGNORECASE,
)

_QPARSER = None


def _get_qparser():
    global _QPARSER
    if _QPARSER is None:
        try:
            from quantulum3 import parser as qp
            _QPARSER = qp
        except ImportError:
            _QPARSER = False
    return _QPARSER if _QPARSER is not False else None


_Q_GOOD_UNITS = {
    'dollar', 'united states dollar', 'euro', 'pound sterling',
    'canadian dollar', 'australian dollar', 'dimensionless',
}


def _parse_financial(text: str) -> float | None:
    """Extract a financial quantity from text, resolving scale and denomination.

    Tries quantulum3 first (handles '$X billion', 'X million USD', scientific
    notation, etc.), pre-expanding bare abbreviations (B/M/MM/K/T) to full
    words so quantulum3 doesn't misread them as physical units.
    Falls back to regex scale-word matching when quantulum3 is unavailable or
    returns no usable quantity.
    """
    # Strip noise patterns before any numeric extraction
    clean = _CVE_RE.sub('', text)
    clean = _PATENT_ID_RE.sub('', clean)
    clean = _DATE_STR_RE.sub('', clean)
    clean = _YEAR_RE.sub('', clean)

    # Expand bare abbreviations: "10.9B" → "10.9 billion", "716.9MM" → "716.9 million"
    expanded = _FIN_ABBREV_RE.sub(
        lambda m: ' ' + _FIN_ABBREV_MAP[m.group(1).lower()], clean
    )

    qp = _get_qparser()
    if qp is not None:
        quants = qp.parse(expanded)
        for q in quants:
            if q.unit.name in _Q_GOOD_UNITS and q.value != 0:
                return float(q.value)

    # Fallback: regex scale-word matching
    num_clean = expanded.replace(',', '').replace('$', '').replace('€', '').replace('£', '')
    m = _FLOAT_RE.search(num_clean)
    if not m:
        return None
    val = float(m.group())
    sm = _SCALE_WORD_RE.search(
        expanded[max(0, expanded.find(m.group()) - 5):
                 expanded.find(m.group()) + len(m.group()) + 30]
    )
    if sm:
        scale = _SCALE_MAP.get(sm.group(0).lower())
        if scale:
            val *= scale
    return val


def _extract_number(text: str) -> float | None:
    # Strip patterns that embed misleading numbers before extracting
    cleaned = _CVE_RE.sub('', text)
    cleaned = _PATENT_ID_RE.sub('', cleaned)
    cleaned = _DATE_STR_RE.sub('', cleaned)
    cleaned = _YEAR_RE.sub('', cleaned)
    cleaned = cleaned.replace(',', '')
    m = _FLOAT_RE.search(cleaned)
    return float(m.group()) if m else None


def _number_matches(val: float, gold: float, tolerance: float) -> bool:
    if tolerance == 0:
        return val == gold
    return abs(val - gold) <= tolerance


def score_number(generated: str, gold_value: float, tolerance: float = 0.0,
                 unit: str | None = None) -> float:
    if _is_abstention(generated):
        return 0.3

    # For financial (billion USD) values: parse with full denomination resolution,
    # then normalise to billions for comparison.
    if unit == 'billion USD':
        raw = _parse_financial(generated)
        if raw is None:
            return 0.0
        pred_billions = raw / 1e9
        if _number_matches(pred_billions, gold_value, tolerance):
            return 1.0
        # Fallback: raw number might be in millions or thousands (no unit word in text).
        for divisor in (1e6, 1e3):
            if _number_matches(raw / divisor, gold_value, tolerance):
                return 1.0
        return 0.0

    # Primary extraction: first number in text.
    pred = _extract_number(generated)
    if pred is not None and _number_matches(pred, gold_value, tolerance):
        return 1.0

    # Partial credit: gold value appears somewhere in the text but wasn't first.
    # Useful for agentic planners — the correct fact is present but buried.
    cleaned = _CVE_RE.sub('', generated)
    cleaned = _PATENT_ID_RE.sub('', cleaned)
    cleaned = _DATE_STR_RE.sub('', cleaned)
    cleaned = _YEAR_RE.sub('', cleaned)
    cleaned = cleaned.replace(',', '')
    all_nums = [float(m) for m in _FLOAT_RE.findall(cleaned)]
    if any(_number_matches(n, gold_value, max(tolerance, abs(gold_value) * 0.01)) for n in all_nums):
        return 0.5

    return 0.0


# ── Entity scorer ───────────────────────────────────────────────────────────

_FILLER_WORDS_RE = re.compile(
    r'\b(version|ver|v|release|update|patch|build|edition|rev|revision)\b\.?',
    re.I
)


def _normalize_entity(s: str) -> str:
    s = _FILLER_WORDS_RE.sub('', s)
    return re.sub(r'[^a-z0-9]', '', s.lower())


def score_entity(generated: str, gold_values: list[str], match_mode: str = "exact") -> float:
    if _is_abstention(generated[:200]):
        return 0.3
    norm_gold = {_normalize_entity(g) for g in gold_values}
    # extract candidate entities: capitalised tokens, CVE-like, patent-like
    tokens = re.findall(r'CVE-[\d-]+|patent[_\s]?\w+|\b[A-Z][A-Za-z0-9\s,\.]+', generated)
    tokens += re.findall(r'\b\d+\b', generated)  # bare numbers (counts, IDs)
    # Version strings (e.g. "26.4", "147.0.7727.55") — only when gold contains versions,
    # to avoid diluting F1 precision on non-version entity questions.
    if any(re.search(r'\d+\.\d+', g) for g in gold_values):
        tokens += re.findall(r'\d+(?:\.\d+)+', generated)
    # Also add individual words so "Google Chrome product." → {'google', 'chrome', 'product'}
    tokens += re.findall(r'\b[a-zA-Z][a-zA-Z0-9]+\b', generated)
    # N-gram sliding window to catch multi-word entities like "macOS Tahoe 26.4"
    words = re.findall(r'\S+', generated)
    max_ng = max((len(g.split()) for g in gold_values), default=1)
    for n in range(2, min(max_ng + 1, 8)):
        for i in range(len(words) - n + 1):
            tokens.append(' '.join(words[i:i + n]))
    norm_pred = {_normalize_entity(t) for t in tokens if t.strip()}

    if match_mode == "any":
        return 1.0 if norm_gold & norm_pred else 0.0

    if match_mode in ("all", "exact"):
        # Per-value presence: each gold value scored independently, return mean.
        # Avoids F1 precision dilution from long generated answers.
        norm_gen = _normalize_entity(generated)
        hits = [
            1.0 if (_normalize_entity(g) in norm_pred or _normalize_entity(g) in norm_gen)
            else 0.0
            for g in gold_values
        ]
        return sum(hits) / len(hits) if hits else 0.0

    # F1 over token sets (legacy path — not reached by any, all, or exact)
    tp = len(norm_gold & norm_pred)
    if tp == 0:
        # Partial credit: any gold entity appears as a substring in generated text.
        # The correct fact is present even if surrounding context is wrong — useful
        # signal for an agentic planner to refine on.
        if any(_normalize_entity(g) in _normalize_entity(generated) for g in gold_values):
            return 0.3
        return 0.0
    precision = tp / len(norm_pred) if norm_pred else 0.0
    recall    = tp / len(norm_gold) if norm_gold else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# ── Text scorer ─────────────────────────────────────────────────────────────

def score_text(generated: str, gold_value: str, threshold: float = 0.75,
               embedder=None) -> float:  # threshold reserved for future min-score cutoff
    if _is_abstention(generated[:200]):
        return 0.3
    if not generated.strip() or not gold_value.strip():
        return 0.0
    if embedder is not None:
        import numpy as np
        vg = embedder(gold_value)
        vp = embedder(generated)
        cos = float(np.dot(vg, vp) / (np.linalg.norm(vg) * np.linalg.norm(vp) + 1e-9))
        return max(0.0, cos)

    # ROUGE-L fallback
    def _lcs(a, b):
        a, b = a.lower().split(), b.lower().split()
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    ref_len = len(gold_value.split())
    hyp_len = len(generated.split())
    lcs = _lcs(gold_value, generated)
    if ref_len == 0 or hyp_len == 0:
        return 0.0
    p = lcs / hyp_len
    r = lcs / ref_len
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Evidence grounding check ─────────────────────────────────────────────────

# Corpus-specific identifiers and field markers that indicate retrieved evidence.
# SRR evidence_used entries contain passage text rather than document IDs, so we
# match corpus-characteristic strings across CVE, patent, SEC, and CPE corpora.
_EVIDENCE_ID_RE = re.compile(
    r'CVE-\d{4}-\d+'                                               # full CVE ID
    r'|\bCVE[s]?\b'                                                # bare CVE mention
    r'|\b(?:NVD|CPE)\b'                                           # NVD/CPE databases
    r'|US\d{6,8}'                                                  # US patent number
    r'|\b\w{2,8}_10[kK]'                                          # ticker_10k filing
    r'|Form\s+10-[KQ]'                                            # SEC form reference
    r'|\bassignee\b'                                               # patent assignee field
    r'|\b(?:apple|google|alphabet|meta|amazon|netflix|microsoft)/\w+'  # CPE vendor/product
    r'|This issue (?:is fixed|affects)'                            # CVE advisory text
    r'|\bvulnerability\b'                                          # CVE description keyword
    r'|votes?\s+per\s+share'                                      # SEC stock structure
    r'|\bticker\s+[A-Z]{1,6}\b'                                  # SEC ticker reference
    r'|\$\s*\d[\d,.]*\s*(?:million|billion)'                      # financial figures
    r'|\bLEI[:\s]'                                                # GLEIF LEI reference
    r'|\bpat(?:ent)?[_\s]\d+',                                   # patent reference
    re.IGNORECASE,
)

# Fallback: combined evidence text > 200 chars signals substantive retrieval even
# when no specific corpus marker matches (e.g. narrative SEC passages, CMO text).
_EVIDENCE_BULK_CHARS = 150


def _evidence_is_grounded(srr_data: dict) -> bool:
    """Return True if evidence_used contains a corpus marker or substantive bulk text."""
    entries = srr_data.get("evidence_used") or []
    if not entries:
        return False
    if any(_EVIDENCE_ID_RE.search(str(e)) for e in entries):
        return True
    return sum(len(str(e)) for e in entries) >= _EVIDENCE_BULK_CHARS


# ── Dispatcher ──────────────────────────────────────────────────────────────

def score_one(generated: str, schema: dict, embedder=None,
              srr_data: dict | None = None) -> float:
    ct = schema.get("check_type", "text")
    val = schema.get("value")
    if ct == "boolean":
        base = score_boolean(generated, bool(val))
    elif ct == "number":
        try:
            gold_num = float(val) if val is not None else _extract_number(str(schema.get("gold_answer", "")))
        except (TypeError, ValueError):
            gold_num = None
        if gold_num is None:
            return float("nan")
        base = score_number(generated, gold_num, schema.get("tolerance", 0.0),
                            unit=schema.get("unit"))
    elif ct == "date":
        base = score_date(generated, str(val) if val else "", schema.get("tolerance_days", 0))
    elif ct == "entity":
        values = val if isinstance(val, list) else [str(val)]
        base = score_entity(generated, values, schema.get("match_mode", "exact"))
    else:
        # text (default)
        base = score_text(generated, str(val) if val else "", schema.get("similarity_threshold", 0.75),
                          embedder=embedder)

    # require_evidence: cap ungrounded answers at 0.3 (abstention equivalent).
    # Only applied when srr_data is present (SRR run); non-SRR runs are not penalised.
    if schema.get("require_evidence") and srr_data is not None and base >= 1.0:
        if not _evidence_is_grounded(srr_data):
            return 0.3

    return base


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-embed-model", default=None,
                    help="Local sentence-transformer model for text check_type "
                         "(e.g. BAAI/bge-large-en-v1.5). Falls back to ROUGE-L if omitted.")
    args = ap.parse_args()

    embedder = None
    if args.local_embed_model:
        from sentence_transformers import SentenceTransformer as _ST
        _model = _ST(args.local_embed_model)
        _embed_cache: dict[str, list[float]] = {}

        def _embed(text: str) -> list[float]:
            if text not in _embed_cache:
                _embed_cache[text] = _model.encode(text, normalize_embeddings=True).tolist()
            return _embed_cache[text]

        embedder = _embed

    schemas = {}
    with open(args.schemas) as f:
        for line in f:
            r = json.loads(line)
            if "answer_schema" in r:
                schemas[r["id"]] = r["answer_schema"]

    results = {}
    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            results[r["id"]] = r

    scores_by_type = defaultdict(list)
    per_question = []

    for qid, schema in schemas.items():
        if qid not in results:
            continue
        r = results[qid]
        generated = r.get("generated_answer", "")
        s = score_one(generated, schema, embedder=embedder, srr_data=r.get("srr"))
        qt = r.get("question_type", "?")
        scores_by_type[qt].append(s)
        per_question.append({
            "id": qid,
            "question_type": qt,
            "check_type": schema.get("check_type"),
            "score": s,
            "generated": generated[:120],
            "gold": r.get("gold_answer", "")[:80],
        })

    overall_scores = [s for ss in scores_by_type.values() for s in ss
                      if s == s]  # exclude NaN
    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    output = {
        "overall": overall,
        "by_type": {qt: sum(ss)/len(ss) for qt, ss in scores_by_type.items() if ss},
        "per_question": per_question,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n── Typed scores: {Path(args.results).name} ──")
    print(f"  Overall: {overall:.3f}")
    for qt, ss in sorted(scores_by_type.items()):
        avg = sum(ss) / len(ss) if ss else 0
        print(f"  {qt:<40} {avg:.3f}  (n={len(ss)})")
    print(f"\nWritten → {args.out}")


if __name__ == "__main__":
    main()
