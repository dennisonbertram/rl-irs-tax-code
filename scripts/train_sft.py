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
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "train"
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "sft" / "adapters"
LOG_FILE = PROJECT_ROOT / "outputs" / "sft" / "train.log"

# ---------------------------------------------------------------------------
# Default hyperparameters (tuned for M4 Max 128 GB, bf16, rank-32 LoRA)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "iters": 1000,
    "batch_size": 4,
    "lora_layers": 16,         # number of transformer layers to apply LoRA
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


def check_data() -> None:
    """Verify SFT training data exists."""
    required = [
        DATA_DIR / "train.jsonl",
        DATA_DIR / "valid.jsonl",
    ]
    # mlx_lm.lora expects train.jsonl and valid.jsonl in the data directory.
    # If the user created sft.jsonl, we check for that too and give a hint.
    missing = [p for p in required if not p.exists()]
    if missing:
        # Check for the raw sft.jsonl as fallback
        sft_file = DATA_DIR / "sft.jsonl"
        if sft_file.exists():
            print(
                f"WARNING: mlx_lm.lora expects {DATA_DIR}/train.jsonl and "
                f"{DATA_DIR}/valid.jsonl.\n"
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
    with open(DATA_DIR / "train.jsonl") as f:
        first = json.loads(f.readline())
    if "text" not in first and "messages" not in first:
        print(
            "WARNING: train.jsonl records should have a 'text' or 'messages' key. "
            f"Got keys: {list(first.keys())}"
        )
    print(f"Data OK — {DATA_DIR}")


def build_command(args: argparse.Namespace, model_path: Path) -> list[str]:
    """Construct the mlx_lm.lora command."""
    # Note: --lora-layers was renamed to --num-layers in mlx_lm >= 0.19
    # LoRA rank is set via a YAML config file (-c flag)
    lora_config = PROJECT_ROOT / "configs" / "mlx_lora_rank32.yaml"
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", str(model_path),
        "--data", str(DATA_DIR),
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
    parser = argparse.ArgumentParser(description="SFT training via mlx_lm.lora")
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

    check_dependencies()
    model_path = resolve_model_path()
    check_data()

    cmd = build_command(args, model_path)
    run_training(cmd, dry_run=args.dry_run)

    if not args.dry_run and not args.skip_test:
        test_generation(model_path)


if __name__ == "__main__":
    main()
