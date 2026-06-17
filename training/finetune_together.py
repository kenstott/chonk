"""
Together.ai LoRA fine-tuning script for Qwen3-30B-A3B-Instruct-2507.

Pipeline: validate → prepare → upload → launch → monitor → report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Load ../.env (project root) before anything reads os.environ
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    with _ENV_FILE.open() as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_SEQ_LEN = 8192
BATCH_SIZE_MIN = 8
BATCH_SIZE_MAX = 16
GRADIENT_ACCUMULATION_STEPS = 1
SYSTEM_PROMPT = "You convert Python to idiomatic, compilable Go."

DEFAULT_BATCH_SIZE = 16
DEFAULT_N_EPOCHS = 3
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_N_CHECKPOINTS = 1
DEFAULT_SUFFIX = "py2go"
DEFAULT_VALIDATION_SPLIT = 0.05

METADATA_FILE = "run_metadata.json"
POLL_INTERVAL_SECONDS = 30
TERMINAL_STATES = {"completed", "failed", "cancelled", "error"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune a LoRA adapter on Together.ai (Qwen3-30B-A3B-Instruct-2507).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default=str(Path(__file__).parent / "training_data_sft_A_python_go_20260611_final.jsonl"),  # noqa: E501
        metavar="PATH",
        help="Path to raw dataset (JSONL with instruction/input/output fields).",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory for train.jsonl, val.jsonl, run_metadata.json.",
    )
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--n-checkpoints", type=int, default=DEFAULT_N_CHECKPOINTS)
    p.add_argument("--suffix", default=DEFAULT_SUFFIX)
    p.add_argument(
        "--full-finetune",
        action="store_true",
        help="Full fine-tune instead of LoRA. Default is LoRA.",
    )
    p.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        metavar="FRAC",
        help="Fraction of data for validation (0 = no val split).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare + validate data, print planned config, then exit.",
    )
    p.add_argument(
        "--check-hardware",
        action="store_true",
        help="Query Together endpoint hardware availability and exit.",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def get_api_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        die("TOGETHER_API_KEY is not set. Export it before running.")
    return key


def load_metadata(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _json_default(obj: Any) -> Any:  # noqa: ANN401
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_metadata(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def count_tokens_approx(text: str) -> int:
    """BPE approximation: ~4 chars per token. Conservative for code."""
    return max(1, len(text) // 3)  # tighter bound for code-heavy text


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


# ---------------------------------------------------------------------------
# Stage 1: prepare_data
# ---------------------------------------------------------------------------
def prepare_data(
    input_path: Path,
    output_dir: Path,
    max_seq_len: int,
    validation_split: float,
) -> tuple[Path, Path | None]:
    raw: list[dict[str, Any]] = []

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        with input_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    elif suffix == ".csv":
        import csv

        with input_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw = list(reader)
    else:
        die(f"Unsupported format: {suffix}. Expected .jsonl or .csv.")

    if not raw:
        die("Dataset is empty.")

    sample = raw[0]
    print("\n--- Detected schema (first row keys) ---")
    for k, v in sample.items():
        preview = str(v)[:80].replace("\n", "\\n")
        print(f"  {k!r}: {preview!r}")

    required = {"instruction", "input", "output"}
    missing = required - set(sample.keys())
    if missing:
        die(
            f"Dataset is missing required fields: {missing}. "
            "Expected keys: instruction, input, output. "
            "Inspect your dataset and re-run."
        )
    print(f"\nSchema OK. {len(raw)} raw examples.\n")

    formatted: list[dict[str, Any]] = []
    dropped = 0
    for row in raw:
        ex = format_example(row)
        full_text = " ".join(m["content"] for m in ex["messages"])
        n_tokens = count_tokens_approx(full_text)
        if n_tokens > max_seq_len:
            dropped += 1
            continue
        formatted.append(ex)

    if dropped:
        print(f"Dropped {dropped} examples exceeding {max_seq_len} tokens.")
    print(f"Retained {len(formatted)} examples.")

    if not formatted:
        die("No examples remain after token filtering.")

    val_path: Path | None = None
    if validation_split and validation_split > 0:
        n_val = max(1, int(len(formatted) * validation_split))
        n_train = len(formatted) - n_val
        train_rows = formatted[:n_train]
        val_rows = formatted[n_train:]
        val_path = output_dir / "val.jsonl"
        val_path.write_text("\n".join(json.dumps(r) for r in val_rows) + "\n")
        print(f"Validation split: {len(val_rows)} → {val_path}")
    else:
        train_rows = formatted

    train_path = output_dir / "train.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    print(f"Train split: {len(train_rows)} → {train_path}")

    return train_path, val_path


# ---------------------------------------------------------------------------
# Stage 2: validate format via Together CLI (optional, non-fatal if missing)
# ---------------------------------------------------------------------------
def validate_format(train_path: Path) -> None:
    import subprocess

    print("\nRunning Together format check...")
    result = subprocess.run(
        ["together", "files", "check", str(train_path)],
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output)
    if result.returncode != 0:
        die(
            f"Together format check failed (exit {result.returncode}). Fix the dataset before uploading."  # noqa: E501
        )
    print("Format check passed.")


# ---------------------------------------------------------------------------
# Stage 3: upload
# ---------------------------------------------------------------------------
def upload(
    client: Any,  # noqa: ANN401
    train_path: Path,
    val_path: Path | None,
    metadata: dict[str, Any],
) -> tuple[str, str | None]:
    train_id: str | None = metadata.get("train_file_id")
    val_id: str | None = metadata.get("val_file_id")

    if train_id:
        print(f"Reusing existing train file: {train_id}")
    else:
        print(f"Uploading {train_path} ...")
        try:
            resp = client.files.upload(file=train_path, check=False)
            train_id = resp.id
            print(f"Train file ID: {train_id}")
        except Exception as e:
            die(f"Upload failed for train file: {e}")

    if val_path:
        if val_id:
            print(f"Reusing existing val file: {val_id}")
        else:
            print(f"Uploading {val_path} ...")
            try:
                resp = client.files.upload(file=val_path, check=False)
                val_id = resp.id
                print(f"Val file ID: {val_id}")
            except Exception as e:
                die(f"Upload failed for val file: {e}")

    return train_id, val_id  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stage 4: launch
# ---------------------------------------------------------------------------
def launch(
    client: Any,  # noqa: ANN401
    train_file_id: str,
    val_file_id: str | None,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> str:
    existing_job_id: str | None = metadata.get("job_id")
    if existing_job_id:
        print(f"\nExisting job found in metadata: {existing_job_id}")
        answer = input("Launch a new job anyway? [y/N] ").strip().lower()
        if answer != "y":
            print("Reusing existing job. Proceeding to monitor.")
            return existing_job_id

    kwargs: dict[str, Any] = dict(
        training_file=train_file_id,
        model=args.base_model,
        lora=not args.full_finetune,
        n_epochs=args.n_epochs,
        n_checkpoints=args.n_checkpoints,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_on_inputs="auto",
        suffix=args.suffix,
    )
    if val_file_id:
        kwargs["validation_file"] = val_file_id

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        kwargs["wandb_api_key"] = wandb_key
        print("W&B logging enabled.")

    print("\nLaunching fine-tuning job with config:")
    for k, v in kwargs.items():
        if k == "wandb_api_key":
            print(f"  {k}: ***")
        else:
            print(f"  {k}: {v}")

    try:
        job = client.fine_tuning.create(**kwargs)
        job_id: str = job.id
    except Exception as e:
        die(f"Failed to launch fine-tuning job: {e}")
        raise  # unreachable; satisfies type checker

    print(f"\nJob launched: {job_id}")
    return job_id


# ---------------------------------------------------------------------------
# Stage 5: monitor
# ---------------------------------------------------------------------------
def monitor(client: Any, job_id: str) -> dict[str, Any]:  # noqa: ANN401
    print(f"\nMonitoring job {job_id} (Ctrl-C to stop watching — job continues remotely)")
    seen_events: set[str] = set()

    try:
        while True:
            try:
                job = client.fine_tuning.retrieve(job_id)
            except Exception as e:
                print(f"  [warn] retrieve failed: {e}. Retrying in {POLL_INTERVAL_SECONDS}s...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            status = getattr(job, "status", "unknown")
            print(f"  status: {status}")

            try:
                events = client.fine_tuning.list_events(job_id)
                event_list = getattr(events, "data", events) or []
                for ev in event_list:
                    ev_id = getattr(ev, "id", None) or getattr(ev, "created_at", id(ev))
                    ev_key = str(ev_id)
                    if ev_key not in seen_events:
                        seen_events.add(ev_key)
                        msg = getattr(ev, "message", "") or getattr(ev, "type", "")
                        ts = getattr(ev, "created_at", "")
                        print(f"  [{ts}] {msg}")
            except Exception as e:
                print(f"  [warn] list_events failed: {e}")

            if status in TERMINAL_STATES:
                return job.__dict__ if hasattr(job, "__dict__") else {}

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Job {job_id} is still running remotely.")
        print("Resume monitoring with: python finetune_together.py --job-id <id>")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Hardware check helper
# ---------------------------------------------------------------------------
def check_hardware(client: Any, base_model: str) -> None:  # noqa: ANN401
    print(f"Checking endpoint hardware for {base_model} ...")
    try:
        hw = client.endpoints.list_hardware(model=base_model)
        print(json.dumps(hw if isinstance(hw, (list, dict)) else hw.__dict__, indent=2))
    except Exception as e:
        print(f"Hardware check failed: {e}")
        print(
            "This may mean the model is not available on Dedicated Endpoints. "
            "Confirm at https://docs.together.ai/docs/dedicated-endpoints before training."
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    # Allow --job-id for resume-monitor mode
    parser.add_argument(
        "--job-id",
        metavar="ID",
        help="Existing job ID to resume monitoring (skips prepare/upload/launch).",
    )
    args = parser.parse_args()

    # Validate config
    if args.batch_size <= 0:
        die(f"--batch-size must be > 0, got {args.batch_size}.")
    if args.n_epochs <= 0:
        die("--n-epochs must be > 0.")

    api_key = get_api_key()

    try:
        from together import Together  # type: ignore[import-untyped]

        _Together = Together
    except ImportError:
        die("together package not installed. Run: pip install -U together")
        raise  # unreachable

    client = _Together(api_key=api_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILE
    metadata = load_metadata(metadata_path)

    # --- hardware check only ---
    if args.check_hardware:
        check_hardware(client, args.base_model)
        return

    # --- resume-monitor only ---
    if args.job_id:
        summary = monitor(client, args.job_id)
        output_model = summary.get("output_name") or summary.get("fine_tuned_model") or "unknown"
        print(f"\nOutput model: {output_model}")
        return

    # --- full pipeline ---
    input_path = Path(args.input)
    if not input_path.exists():
        die(f"Input file not found: {input_path}")

    train_path, val_path = prepare_data(
        input_path, output_dir, args.max_seq_len, args.validation_split
    )

    # Try format check; non-fatal if together CLI absent
    try:
        validate_format(train_path)
    except FileNotFoundError:
        print("[warn] 'together' CLI not found; skipping shell-based format check.")

    if args.dry_run:
        print("\n--- DRY RUN — planned job config ---")
        config = dict(
            base_model=args.base_model,
            lora=not args.full_finetune,
            n_epochs=args.n_epochs,
            n_checkpoints=args.n_checkpoints,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            train_on_inputs="auto",
            suffix=args.suffix,
            validation_split=args.validation_split,
            train_file=str(train_path),
            val_file=str(val_path) if val_path else None,
            wandb_enabled=bool(os.environ.get("WANDB_API_KEY")),
        )
        print(json.dumps(config, indent=2))
        print("\nDry run complete. No files uploaded, no job launched.")
        return

    train_file_id, val_file_id = upload(client, train_path, val_path, metadata)
    metadata["train_file_id"] = train_file_id
    if val_file_id:
        metadata["val_file_id"] = val_file_id
    save_metadata(metadata_path, metadata)

    job_id = launch(client, train_file_id, val_file_id, args, metadata)
    metadata["job_id"] = job_id
    save_metadata(metadata_path, metadata)

    summary = monitor(client, job_id)

    output_model = summary.get("output_name") or summary.get("fine_tuned_model") or "unknown"
    metadata["output_model"] = output_model
    metadata["job_summary"] = summary
    save_metadata(metadata_path, metadata)

    print(f"\nOutput model: {output_model}")
    print(
        "\nNext steps: provision a Dedicated Endpoint for the base model "
        f"({args.base_model}), then load the LoRA adapter.\n"
        "Run `python finetune_together.py --check-hardware` to query available hardware."
    )


if __name__ == "__main__":
    main()
