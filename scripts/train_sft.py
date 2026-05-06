#!/usr/bin/env python3
"""
SFT Training Script using MLX LoRA fine-tuning.

Uses mlx_lm.lora (built-in MLX fine-tuning) to train on tax law SFT data.
Supports both MLX-converted model and HuggingFace format (auto-converts).

Usage:
    python scripts/train_sft.py [--iters 1000] [--batch-size 4] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_MLX = PROJECT_ROOT / "models" / "qwen25-3b-mlx"
MODEL_HF = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct"
DATA_DIR = PROJECT_ROOT / "data" / "v5"
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "sft" / "adapters"
LOG_FILE = PROJECT_ROOT / "outputs" / "sft" / "train.log"

# CLI override (set in main())
_CLI_DATA_DIR: Path | None = None

# ---------------------------------------------------------------------------
# Default hyperparameters (tuned for M4 Max 128 GB, bf16, rank-32 LoRA)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "iters": 1000,
    "batch_size": 4,
    # Fix 2-A: raised from 16 → 24 (all layers for Qwen-3B) so LoRA covers the
    # full network. With only 16/24 layers frozen, the model lacks capacity to
    # memorise precise IRC numeric tables. Use 16 if memory-constrained.
    "lora_layers": 24,         # number of transformer layers to apply LoRA
    "lora_rank": 32,
    "learning_rate": 1e-5,
    "val_batches": 25,
    "steps_per_eval": 100,
    "save_every": 200,
    "max_seq_length": 2048,
    "grad_checkpoint": True,   # enable gradient checkpointing to save memory
}


def check_dependencies() -> None:
    """Verify mlx_lm is importable."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("ERROR: mlx_lm not found. Install with: pip install mlx-lm")
        sys.exit(1)


def resolve_model_path() -> Path:
    """Return the model path to use, preferring the MLX-converted version."""
    if MODEL_MLX.exists() and (MODEL_MLX / "config.json").exists():
        print(f"Using MLX model: {MODEL_MLX}")
        return MODEL_MLX
    if MODEL_HF.exists() and (MODEL_HF / "config.json").exists():
        print(f"MLX model not found. Using HF model (mlx_lm will convert): {MODEL_HF}")
        return MODEL_HF
    print("ERROR: No model found. Expected one of:")
    print(f"  {MODEL_MLX}")
    print(f"  {MODEL_HF}")
    sys.exit(1)


def get_data_dir() -> Path:
    """Return the data directory to use (CLI override or default)."""
    if _CLI_DATA_DIR is not None:
        return _CLI_DATA_DIR
    return DATA_DIR


def check_data() -> None:
    """Verify SFT training data exists."""
    data_dir = get_data_dir()
    required = [
        data_dir / "train.jsonl",
        data_dir / "valid.jsonl",
    ]
    # mlx_lm.lora expects train.jsonl and valid.jsonl in the data directory.
    # If the user created sft.jsonl, we check for that too and give a hint.
    missing = [p for p in required if not p.exists()]
    if missing:
        # Check for the raw sft.jsonl as fallback
        sft_file = data_dir / "sft.jsonl"
        if sft_file.exists():
            print(
                f"WARNING: mlx_lm.lora expects {data_dir}/train.jsonl and "
                f"{data_dir}/valid.jsonl.\n"
                f"Found {sft_file} — run scripts/prepare_mlx_data.py to split it."
            )
        else:
            print(f"ERROR: Missing required data files: {missing}")
            print(
                "Run the data pipeline first: python scripts/parse_irc.py && "
                "python scripts/generate_sft.py"
            )
        sys.exit(1)
    # Quick sanity-check the first record
    with open(data_dir / "train.jsonl") as f:
        first = json.loads(f.readline())
    if "text" not in first and "messages" not in first:
        print(
            "WARNING: train.jsonl records should have a 'text' or 'messages' key. "
            f"Got keys: {list(first.keys())}"
        )
    print(f"Data OK — {data_dir}")


def build_lora_config(args: argparse.Namespace) -> Path:
    """
    Write a temporary YAML config for mlx_lm.lora with the requested LoRA rank.

    The rank cannot be passed as a CLI flag to mlx_lm.lora directly; it must
    be specified in the YAML config file.  This function generates a config
    from the CLI --lora-rank argument so the flag is actually honoured.

    Fix LOW: previously --lora-rank was accepted by the SFT parser but silently
    ignored — the static configs/mlx_lora_rank32.yaml was always used regardless
    of the user's choice.
    """
    config_data = {
        "lora_parameters": {
            "rank": args.lora_rank,
            "dropout": 0.05,
            "scale": 20.0,
        }
    }

    # Write to a temp file so we don't clobber the static configs/ file.
    configs_dir = PROJECT_ROOT / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"mlx_lora_rank{args.lora_rank}.yaml"

    try:
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
    except ImportError:
        # PyYAML not available — write manually (safe for simple flat dict)
        with open(config_path, "w") as f:
            f.write("lora_parameters:\n")
            f.write(f"  rank: {args.lora_rank}\n")
            f.write("  dropout: 0.05\n")
            f.write("  scale: 20.0\n")

    return config_path


def build_command(args: argparse.Namespace, model_path: Path) -> list[str]:
    """Construct the mlx_lm.lora command."""
    # Note: --lora-layers was renamed to --num-layers in mlx_lm >= 0.19
    # LoRA rank is wired via a generated YAML config file (-c flag).
    lora_config = build_lora_config(args)
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", str(model_path),
        "--data", str(get_data_dir()),
        "--train",
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.lora_layers),
        "--iters", str(args.iters),
        "--val-batches", str(args.val_batches),
        "--learning-rate", str(args.learning_rate),
        "--steps-per-eval", str(args.steps_per_eval),
        "--adapter-path", str(ADAPTER_PATH),
        "--save-every", str(args.save_every),
        "--max-seq-length", str(args.max_seq_length),
    ]
    if lora_config.exists():
        cmd += ["-c", str(lora_config)]
    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    return cmd


def run_training(cmd: list[str], dry_run: bool = False) -> None:
    """Execute the training command, streaming output to stdout and log file."""
    print("\n" + "=" * 70)
    print("SFT TRAINING COMMAND:")
    print(" ".join(cmd))
    print("=" * 70 + "\n")

    if dry_run:
        print("DRY RUN — command not executed.")
        return

    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with open(LOG_FILE, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()

    elapsed = time.time() - start
    if proc.returncode != 0:
        print(f"\nERROR: Training failed (exit code {proc.returncode}). "
              f"See {LOG_FILE} for details.")
        sys.exit(proc.returncode)

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Adapters saved to: {ADAPTER_PATH}")
    print(f"Log saved to: {LOG_FILE}")


def test_generation(model_path: Path) -> None:
    """Run a quick generation test with the trained adapter."""
    print("\n" + "=" * 70)
    print("POST-TRAINING GENERATION TEST")
    print("=" * 70)

    test_prompt = (
        "What is the standard deduction for a single filer under IRC Section 63?"
    )

    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", str(model_path),
        "--adapter-path", str(ADAPTER_PATH),
        "--max-tokens", "256",
        "--prompt", test_prompt,
    ]
    print(f"Prompt: {test_prompt}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print("WARNING: Generation test failed — adapter may still be usable.")


def main() -> None:
    global _CLI_DATA_DIR
    parser = argparse.ArgumentParser(description="SFT training via mlx_lm.lora")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data directory containing train.jsonl and valid.jsonl "
                             "(default: data/processed/train)")
    parser.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lora-layers", type=int, default=DEFAULTS["lora_layers"])
    parser.add_argument("--lora-rank", type=int, default=DEFAULTS["lora_rank"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--val-batches", type=int, default=DEFAULTS["val_batches"])
    parser.add_argument("--steps-per-eval", type=int, default=DEFAULTS["steps_per_eval"])
    parser.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["grad_checkpoint"],
        help="Enable gradient checkpointing (saves memory at slight speed cost)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip post-training generation test",
    )
    args = parser.parse_args()

    if args.data_dir:
        _CLI_DATA_DIR = Path(args.data_dir)

    check_dependencies()
    model_path = resolve_model_path()
    check_data()

    cmd = build_command(args, model_path)
    run_training(cmd, dry_run=args.dry_run)

    if not args.dry_run and not args.skip_test:
        test_generation(model_path)


if __name__ == "__main__":
    main()
