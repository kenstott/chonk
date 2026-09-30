# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Tests for the `chonk build` command-line entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from chonk import _cli


class _FakeIndex:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_build_passes_config_path_and_force(monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("store: {path: x.duckdb}\n")
    calls = []
    idx = _FakeIndex()

    def fake_build(config, *, force):
        calls.append((config, force))
        return idx

    monkeypatch.setattr(_cli, "build", fake_build)

    assert _cli.main(["build", str(cfg), "--force"]) == 0
    assert calls == [(cfg, True)]
    assert idx.closed


def test_build_force_defaults_to_false(monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("store: {path: x.duckdb}\n")
    calls = []

    def fake_build(config, *, force):
        calls.append(force)
        return _FakeIndex()

    monkeypatch.setattr(_cli, "build", fake_build)

    assert _cli.main(["build", str(cfg)]) == 0
    assert calls == [False]


def test_build_missing_config_is_an_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _cli.main(["build", str(tmp_path / "nope.yaml")])
    assert exc.value.code == 2


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exc:
        _cli.main([])
    assert exc.value.code == 2


def test_python_dash_m_chonk_help():
    out = subprocess.run(
        [sys.executable, "-m", "chonk", "build", "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert "--force" in out.stdout


def test_build_store_overrides_config_store_path(monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("store:\n  path: original.duckdb\n  embedding_dim: 8\nsources: []\n")
    calls = []

    def fake_build(config, *, force):
        calls.append(config)
        return _FakeIndex()

    monkeypatch.setattr(_cli, "build", fake_build)

    assert _cli.main(["build", str(cfg), "--store", "elsewhere/x.duckdb"]) == 0
    (config,) = calls
    assert config["store"] == {"path": "elsewhere/x.duckdb", "embedding_dim": 8}
    assert config["sources"] == []
