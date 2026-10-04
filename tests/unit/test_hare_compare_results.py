# Copyright (c) 2025 Kenneth Stott. MIT License.

"""scripts/hare/compare_results.py reads both score file layouts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "hare" / "compare_results.py"
_spec = importlib.util.spec_from_file_location("hare_compare_results", _SCRIPT)
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)

_BY_TYPE = {
    "Cross-Domain Entity Resolution": 0.8,
    "Multi-Document Join": 0.9,
    "Temporal Versioning": 0.7,
    "Targeted Attribute Lookup": 0.6,
    "Descriptive Attribute Lookup": 0.5,
}
_EXPECTED = {"overall": 0.7, "CE": 0.8, "MDJ": 0.9, "TVR": 0.7, "TAL": 0.6, "DAL": 0.5}


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "bench_eval_r_rp.json"
    path.write_text(json.dumps(payload))
    return path


def test_reads_typed_scorer_layout(tmp_path):
    path = _write(tmp_path, {"overall": 0.7, "by_type": _BY_TYPE, "per_question": []})

    assert _compare._scores(path) == pytest.approx(_EXPECTED)


def test_reads_evaluator_layout(tmp_path):
    # what `graphrag_bench.py run-all` writes: one block per question type
    payload = {"_params": {"n_evaluated": 500}}
    payload.update({name: {"typed_score": score} for name, score in _BY_TYPE.items()})

    assert _compare._scores(_write(tmp_path, payload)) == pytest.approx(_EXPECTED)


def test_unknown_layout_raises(tmp_path):
    with pytest.raises(KeyError):
        _compare._scores(_write(tmp_path, {"_params": {}}))


def test_type_without_typed_score_is_nan_and_left_out_of_overall(tmp_path):
    # runs without SRR cannot be scored on cross-domain questions, which require
    # cited evidence; the typed scorer leaves that type out of the overall score
    payload = {"_params": {"n_evaluated": 500}}
    payload.update({name: {"typed_score": score} for name, score in _BY_TYPE.items()})
    payload["Cross-Domain Entity Resolution"] = {}

    scores = _compare._scores(_write(tmp_path, payload))

    assert scores["CE"] != scores["CE"]  # NaN
    assert scores["overall"] == pytest.approx((0.9 + 0.7 + 0.6 + 0.5) / 4)
