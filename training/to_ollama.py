#!/usr/bin/env python3
"""Convert a Together.ai merged checkpoint to an Ollama model.

Usage:
    python3 to_ollama.py \\
        --checkpoint path/to/merged_checkpoint.tar.zst \\
        --base-model Qwen/Qwen3-14B

    python3 to_ollama.py \\
        --checkpoint path/to/extracted_dir/ \\
        --base-model Qwen/Qwen3-14B \\
        --name py2go \\
        --quant Q4_K_M \\
        --system-prompt "You convert Python to idiomatic, compilable Go."
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "You convert Python to idiomatic, compilable Go."
DEFAULT_QUANT = "Q4_K_M"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)  # type: ignore[call-overload]


def find_llama_cpp() -> tuple[Path, Path]:
    """Return (convert_script, quantize_binary). Exits with instructions if missing."""
    # brew install llama.cpp puts binaries in $(brew --prefix)/bin
    brew = shutil.which("brew")
    if brew:
        prefix = subprocess.run(
            [brew, "--prefix", "llama.cpp"], capture_output=True, text=True
        ).stdout.strip()
        if prefix:
            prefix_path = Path(prefix)
            convert = prefix_path / "bin" / "convert_hf_to_gguf.py"
            quantize = prefix_path / "bin" / "llama-quantize"
            if not convert.exists():
                # older brew layout
                convert = prefix_path / "convert_hf_to_gguf.py"
            if convert.exists() and quantize.exists():
                return convert, quantize

    # fallback: local llama.cpp clone
    for candidate in [Path("llama.cpp"), Path("../llama.cpp"), Path.home() / "llama.cpp"]:
        convert = candidate / "convert_hf_to_gguf.py"
        quantize = candidate / "llama-quantize"
        if convert.exists() and quantize.exists():
            return convert, quantize

    die(
        "llama.cpp not found.\n"
        "Install with:  brew install llama.cpp\n"
        "Or clone:      git clone https://github.com/ggerganov/llama.cpp && "
        "cd llama.cpp && make -j"
    )
    raise SystemExit(1)  # unreachable


def extract_checkpoint(checkpoint: Path, build_dir: Path) -> Path:
    """Extract archive to build_dir/model_src/. Returns the extracted directory."""
    src_dir = build_dir / "model_src"
    if src_dir.exists():
        print(f"  using existing extracted dir: {src_dir}")
        return src_dir

    src_dir.mkdir(parents=True)

    suffix = "".join(checkpoint.suffixes)
    if ".tar" in suffix:
        mode = "r:*"
        print(f"  extracting {checkpoint.name} …")
        with tarfile.open(checkpoint, mode) as tf:
            tf.extractall(src_dir)
        # If the archive has a single top-level dir, descend into it
        children = list(src_dir.iterdir())
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return src_dir
    die(f"Unrecognised archive format: {suffix}. Pass an extracted directory instead.")
    raise SystemExit(1)


def slug(base_model: str) -> str:
    """Qwen/Qwen3-14B  →  qwen3-14b"""
    name = base_model.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name).lower()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Together.ai merged checkpoint → Ollama model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Path to merged checkpoint (.tar.zst / .tar.gz / extracted directory)",
    )
    p.add_argument(
        "--base-model",
        "-m",
        required=True,
        help="Base model name used during fine-tuning (e.g. Qwen/Qwen3-14B)",
    )
    p.add_argument(
        "--name",
        "-n",
        default=None,
        help="Ollama model name (default: derived from base model)",
    )
    p.add_argument(
        "--quant",
        "-q",
        default=DEFAULT_QUANT,
        help=f"GGUF quantization type (default: {DEFAULT_QUANT})",
    )
    p.add_argument(
        "--system-prompt",
        "-s",
        default=DEFAULT_SYSTEM_PROMPT,
        help=f'System prompt written into the Modelfile (default: "{DEFAULT_SYSTEM_PROMPT}")',
    )
    p.add_argument(
        "--output-dir",
        "-o",
        default="ollama_build",
        help="Directory for intermediate files (default: ollama_build/)",
    )
    p.add_argument(
        "--skip-create",
        action="store_true",
        help="Build GGUF but skip `ollama create` (useful if Ollama is not installed)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        die(f"checkpoint not found: {checkpoint}")

    build_dir = Path(args.output_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    model_slug = slug(args.base_model)
    ollama_name = args.name or model_slug
    quant_lower = args.quant.lower().replace("_", "-")

    f16_gguf = build_dir / f"{model_slug}-f16.gguf"
    quant_gguf = build_dir / f"{model_slug}-{quant_lower}.gguf"
    modelfile = build_dir / "Modelfile"

    convert_script, quantize_bin = find_llama_cpp()

    # -------------------------------------------------------------------------
    # 1. Extract if needed
    # -------------------------------------------------------------------------
    if checkpoint.is_dir():
        model_src = checkpoint
    else:
        print("\n[1/4] Extracting checkpoint …")
        model_src = extract_checkpoint(checkpoint, build_dir)

    # -------------------------------------------------------------------------
    # 2. Convert HF → GGUF (F16)
    # -------------------------------------------------------------------------
    if f16_gguf.exists():
        print(f"\n[2/4] Skipping conversion — {f16_gguf.name} already exists")
    else:
        print("\n[2/4] Converting to GGUF (F16) …")
        run([sys.executable, str(convert_script), str(model_src), "--outfile", str(f16_gguf)])

    # -------------------------------------------------------------------------
    # 3. Quantize
    # -------------------------------------------------------------------------
    if quant_gguf.exists():
        print(f"\n[3/4] Skipping quantization — {quant_gguf.name} already exists")
    else:
        print(f"\n[3/4] Quantizing to {args.quant} …")
        run([str(quantize_bin), str(f16_gguf), str(quant_gguf), args.quant])

    # -------------------------------------------------------------------------
    # 4. Write Modelfile + ollama create
    # -------------------------------------------------------------------------
    print("\n[4/4] Writing Modelfile …")
    modelfile.write_text(f'FROM {quant_gguf.resolve()}\nSYSTEM "{args.system_prompt}"\n')
    print(f"  {modelfile}")

    if args.skip_create:
        print("\nDone. To create the Ollama model manually:")
        print(f"  ollama create {ollama_name} -f {modelfile}")
        return

    ollama = shutil.which("ollama")
    if not ollama:
        print("\nOllama not found in PATH. Install from https://ollama.com, then run:")
        print(f"  ollama create {ollama_name} -f {modelfile}")
        return

    print(f"\n  Creating Ollama model '{ollama_name}' …")
    run([ollama, "create", ollama_name, "-f", str(modelfile)])

    print(f"\nDone. Run with:  ollama run {ollama_name}")


if __name__ == "__main__":
    main()
