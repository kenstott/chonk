# Copyright (c) 2025 Kenneth Stott. MIT License.

"""Command-line entry point: ``chonk build <config.yaml> [--force]``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .ingest import build


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"config file not found: {value}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chonk", description="Chonk command-line tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "build",
        help="Build an index from a YAML config: ingest, embed, FTS, NER, community.",
    )
    p.add_argument("config", type=_existing_file, help="path to the YAML config")
    p.add_argument(
        "--store",
        help="write the store here instead of the config's store.path",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="delete the existing store and rebuild every phase",
    )

    args = parser.parse_args(argv)
    config: Path | dict[str, Any] = args.config
    if args.store:
        import yaml

        raw: dict[str, Any] = yaml.safe_load(args.config.read_text())
        raw["store"] = {**raw["store"], "path": args.store}
        config = raw
    index = build(config, force=args.force)
    index.close()
    return 0
