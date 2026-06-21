"""
Local fine-tuning on Apple Silicon via mlx-lm.

Modes:
  --mode lora   (default) — train a LoRA adapter, saved to <output-dir>/adapters/
  --mode full             — full fine-tune, merged model saved to <output-dir>/model/

Pipeline:
  1. Convert HF model to MLX format (cached after first run)
  2. Prepare train/val JSONL in mlx-lm chat format
  3. Run mlx_lm.lora (LoRA) or mlx_lm.lora --train (full)
  4. Fuse adapter into base model (LoRA only)

Requirements:
  pip install mlx-lm huggingface_hub

Usage:
  python local_fine_tuning.py
  python local_fine_tuning.py --base-model Qwen/Qwen3-4B-Instruct-2507 --mode full
  python local_fine_tuning.py --iters 1000 --learning-rate 2e-5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    with _ENV_FILE.open() as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = "You convert Python to idiomatic, compilable Go."

DEFAULT_INPUT = str(Path(__file__).parent / "training_data_sft_A_python_go_20260611_final.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).parent / "mlx_output")
DEFAULT_ITERS = 600
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUMULATION = 4  # effective batch = 4 despite batch_size=1
DEFAULT_LORA_RANK = 8
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_VALIDATION_SPLIT = 0.05


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def check_mlx() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        die("mlx-lm not installed. Run: pip install mlx-lm")


def format_example(row: dict[str, Any]) -> dict[str, Any]:
    instruction = row.get("instruction", "").strip()
    python_src = row.get("input", "").strip()
    go_src = row.get("output", "").strip()
    user_content = f"{instruction}\n\n{python_src}" if instruction else python_src
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": go_src},
        ]
    }


def prepare_data(
    input_path: Path,
    output_dir: Path,
    validation_split: float,
) -> tuple[Path, Path]:
    import csv

    raw: list[dict[str, Any]] = []
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        with input_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    elif suffix == ".csv":
        with input_path.open(newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
    else:
        die(f"Unsupported format: {suffix}. Expected .jsonl or .csv.")

    if not raw:
        die("Dataset is empty.")

    required = {"instruction", "input", "output"}
    missing = required - set(raw[0].keys())
    if missing:
        die(f"Missing required fields: {missing}")

    examples = [format_example(r) for r in raw]

    # Filter sequences that would OOM at DEFAULT_MAX_SEQ_LENGTH
    # Use chars/3.5 approximation with 10% safety margin
    token_limit = int(DEFAULT_MAX_SEQ_LENGTH * 0.9)
    before = len(examples)
    examples = [e for e in examples if len(json.dumps(e["messages"])) / 3.5 <= token_limit]
    dropped = before - len(examples)
    if dropped:
        print(f"Dropped {dropped} examples exceeding ~{token_limit} tokens (would OOM).")

    n_val = max(1, int(len(examples) * validation_split))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    val_path = output_dir / "valid.jsonl"  # mlx-lm expects valid.jsonl

    with train_path.open("w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with val_path.open("w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Train: {len(train_examples)} examples → {train_path}")
    print(f"Val:   {len(val_examples)} examples → {val_path}")
    return train_path, val_path


def convert_model(base_model: str, mlx_model_dir: Path) -> Path:
    """Download + convert HF model to MLX format. Cached after first run."""
    # Only skip if config.json exists (indicates a complete conversion)
    if (mlx_model_dir / "config.json").exists():
        print(f"MLX model already cached at {mlx_model_dir}")
        return mlx_model_dir

    # Remove incomplete directory so mlx-lm doesn't refuse to write
    if mlx_model_dir.exists():
        import shutil

        shutil.rmtree(mlx_model_dir)

    print(f"Converting {base_model} to MLX format → {mlx_model_dir} ...")
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        base_model,
        "--mlx-path",
        str(mlx_model_dir),
        "--quantize",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        cmd_nq = [
            sys.executable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            base_model,
            "--mlx-path",
            str(mlx_model_dir),
        ]
        subprocess.run(cmd_nq, check=True)
    return mlx_model_dir


def run_lora(
    mlx_model_dir: Path,
    data_dir: Path,
    adapter_dir: Path,
    iters: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int,
    resume_adapter: str | None = None,
) -> None:
    import yaml

    adapter_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "model": str(mlx_model_dir),
        "train": True,
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),
        "iters": iters,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "grad_accumulation_steps": DEFAULT_GRAD_ACCUMULATION,
        "val_batches": 5,
        "fine_tune_type": "lora",
        "max_seq_length": DEFAULT_MAX_SEQ_LENGTH,
        "grad_checkpoint": True,
        "lora_parameters": {"rank": lora_rank, "scale": float(lora_rank), "dropout": 0.0},
    }
    if resume_adapter:
        config["resume_adapter_file"] = resume_adapter
    config_path = adapter_dir / "train_config.yaml"
    config_path.write_text(yaml.dump(config))
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "-c",
        str(config_path),
    ]
    print("\nRunning LoRA fine-tuning...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_full_finetune(
    mlx_model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    iters: int,
    learning_rate: float,
    batch_size: int,
) -> None:
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": str(mlx_model_dir),
        "train": True,
        "data": str(data_dir),
        "adapter_path": str(output_dir),
        "iters": iters,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "val_batches": 5,
        "fine_tune_type": "full",
    }
    config_path = output_dir / "train_config.yaml"
    config_path.write_text(yaml.dump(config))
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "-c",
        str(config_path),
    ]
    print("\nRunning full fine-tune...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def fuse_adapter(mlx_model_dir: Path, adapter_dir: Path, fused_dir: Path) -> None:
    fused_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        str(mlx_model_dir),
        "--adapter-path",
        str(adapter_dir),
        "--save-path",
        str(fused_dir),
        "--dequantize",  # convert back to bf16 for vLLM compat
    ]
    print(f"\nFusing adapter → {fused_dir} ...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        # Try without --de-quantize
        cmd_nq = [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(mlx_model_dir),
            "--adapter-path",
            str(adapter_dir),
            "--save-path",
            str(fused_dir),
        ]
        subprocess.run(cmd_nq, check=True)
    print(f"Fused model saved to {fused_dir}")
    print("Load in vLLM with: vllm serve " + str(fused_dir))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local fine-tuning on Apple Silicon via mlx-lm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT, metavar="PATH")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument(
        "--mode",
        choices=["lora", "full"],
        default="lora",
        help="lora: train adapter only; full: full fine-tune (slower, more memory)",
    )
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    p.add_argument("--validation-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    p.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip model conversion if MLX model already exists.",
    )
    p.add_argument(
        "--fuse",
        action="store_true",
        default=True,
        help="After LoRA training, fuse adapter into base model for vLLM deployment.",
    )
    p.add_argument(
        "--no-fuse",
        action="store_false",
        dest="fuse",
        help="Keep adapter separate (do not fuse).",
    )
    p.add_argument(
        "--resume-adapter",
        default=None,
        metavar="PATH",
        help="Resume from a saved adapter checkpoint (e.g. adapters/0000300_adapters.safetensors).",
    )
    return p


def main() -> None:
    check_mlx()
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    input_path = Path(args.input)
    mlx_model_dir = output_dir / "base_mlx"
    adapter_dir = output_dir / "adapters"
    fused_dir = output_dir / "fused_model"

    train_path, _val_path = prepare_data(input_path, output_dir, args.validation_split)

    if not args.skip_convert:
        convert_model(args.base_model, mlx_model_dir)
    else:
        if not mlx_model_dir.exists():
            die(f"--skip-convert set but {mlx_model_dir} does not exist.")

    data_dir = train_path.parent

    if args.mode == "lora":
        run_lora(
            mlx_model_dir,
            data_dir,
            adapter_dir,
            args.iters,
            args.learning_rate,
            args.batch_size,
            args.lora_rank,
            resume_adapter=args.resume_adapter,
        )
        if args.fuse:
            fuse_adapter(mlx_model_dir, adapter_dir, fused_dir)
        else:
            print(f"\nAdapter saved to {adapter_dir}")
            print(
                "Load adapter with mlx-lm: mlx_lm.generate --model <base> --adapter-path "
                + str(adapter_dir)
            )
    else:
        run_full_finetune(
            mlx_model_dir,
            data_dir,
            output_dir / "full_model",
            args.iters,
            args.learning_rate,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
