# Training Pipeline Review Package v4

## Context
Tax-law LLM fine-tuning pipeline using MLX on Apple Silicon.
Model: Qwen2.5-3B-Instruct with LoRA. Pipeline: SFT -> DPO -> GRPO RL
Python 3.14, MLX 0.31.1

## Round 3 -> Round 4 Fixes

### HIGH fix: save_lora_weights robustness
Replaced try/except import pattern with getattr-based discovery of tree_flatten_items
or tree_flatten. Added explicit type-checking on the result to catch any future API
changes. Raises RuntimeError if no LoRA parameters found (prevents silent 6GB saves).

### HIGH fix: GRPO and DPO adapter_config.json on-disk files
Updated both on-disk files to use the correct minimal format:
  {num_layers, lora_parameters:{rank, scale:20.0, dropout:0.05}, training, ...}
These are the values that train_grpo.py and train_dpo.py will write on next training run.

### MEDIUM fix: evaluation _base_section() uses re.search not re.match
Now handles '§ 179' and other non-digit-prefixed strings correctly.

### LOW fix: sequence_log_prob zero-length edge case in DPO
Returns zero tensor if shifted sequence has length 0 (single-token inputs).

## Source Files

### scripts/train_sft.py

```
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
    import tempfile, yaml as _yaml  # yaml from PyYAML (bundled with mlx-lm)

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

```

### scripts/train_dpo.py

```
#!/usr/bin/env python3
"""
DPO Training Script using MLX.

Loads the SFT adapter as a starting point, then runs Direct Preference
Optimization (DPO) on preference pairs from data/processed/train/dpo.jsonl.

DPO loss (Rafailov et al., 2023):
    L_DPO = -E[log σ(β · (log π(y_w|x) - log π_ref(y_w|x)
                          - log π(y_l|x) + log π_ref(y_l|x)))]

where:
    π       = current (trainable) policy
    π_ref   = frozen reference policy (SFT checkpoint)
    y_w     = chosen (preferred) response
    y_l     = rejected response
    β       = KL penalty coefficient

Input data format (JSONL, one record per line):
    {
        "prompt":   "...",
        "chosen":   "...",
        "rejected": "..."
    }

Usage:
    python scripts/train_dpo.py [--iters 500] [--beta 0.1] [--dry-run]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_MLX = PROJECT_ROOT / "models" / "qwen25-3b-mlx"
MODEL_HF = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct"
SFT_ADAPTER = PROJECT_ROOT / "outputs" / "sft" / "adapters"
DPO_ADAPTER = PROJECT_ROOT / "outputs" / "dpo" / "adapters"
DPO_DATA = PROJECT_ROOT / "data" / "processed" / "train" / "dpo.jsonl"
LOG_FILE = PROJECT_ROOT / "outputs" / "dpo" / "train.log"

# These can be overridden by CLI args
_CLI_MODEL_PATH = None
_CLI_SFT_ADAPTER = None
_CLI_OUTPUT_DIR = None
_CLI_LOG_FILE = None

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = {
    "iters": 500,
    "batch_size": 2,           # DPO is memory-heavy (2x forward passes per step)
    "lora_layers": 16,
    "learning_rate": 5e-6,
    "beta": 0.5,               # KL penalty coefficient (increased from 0.1 — was causing clip saturation)
    "max_seq_length": 1024,
    "save_every": 100,
    "log_every": 10,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_model_items(model) -> list:
    """
    Return a flat list of (key, value) pairs from a model's parameters.

    Handles both modern MLX (tree_flatten returns [(k,v)...]) and future
    versions that may rename the function.  Always returns (key, value) pairs.
    """
    import mlx.utils as mu
    fn = getattr(mu, "tree_flatten_items", None) or getattr(mu, "tree_flatten")
    result = fn(model.parameters())
    if not isinstance(result, (list, tuple)):
        raise RuntimeError(f"Unexpected tree_flatten return type: {type(result)}")
    if result and not isinstance(result[0], (list, tuple)):
        raise RuntimeError(
            f"tree_flatten returned non-pair items (got {type(result[0]).__name__}). "
            "Cannot extract LoRA weights. Check MLX version compatibility."
        )
    return result


def save_lora_weights(model, path: str) -> None:
    """Save only LoRA adapter weights, not the full model.

    Raises RuntimeError if no LoRA parameters are found, to avoid silently
    saving the full model (6+ GB) when checkpointing was intended to save
    a small adapter (< 50 MB).
    """
    import mlx.core as mx
    all_params = _get_model_items(model)
    lora_params = {k: v for k, v in all_params if "lora" in k.lower()}
    if not lora_params:
        raise RuntimeError(
            "save_lora_weights: no LoRA parameters found in model. "
            "Ensure linear_to_lora_layers() has been called before saving. "
            f"Available parameter keys: {[k for k, _ in all_params[:10]]!r}"
        )
    mx.save_safetensors(path, lora_params)


def check_dependencies() -> None:
    try:
        import mlx.core  # noqa: F401
        import mlx.nn    # noqa: F401
        import mlx_lm    # noqa: F401
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}")
        print("Install with: pip install mlx mlx-lm")
        sys.exit(1)


def resolve_model_path() -> Path:
    if _CLI_MODEL_PATH is not None:
        p = Path(_CLI_MODEL_PATH)
        if p.exists() and (p / "config.json").exists():
            return p
        print(f"ERROR: No model found at {p}")
        sys.exit(1)
    if MODEL_MLX.exists() and (MODEL_MLX / "config.json").exists():
        return MODEL_MLX
    if MODEL_HF.exists() and (MODEL_HF / "config.json").exists():
        print(f"Using HF model (mlx_lm will convert): {MODEL_HF}")
        return MODEL_HF
    print(f"ERROR: No model found at {MODEL_MLX} or {MODEL_HF}")
    sys.exit(1)


def check_data() -> None:
    if not DPO_DATA.exists():
        print(f"ERROR: DPO data not found at {DPO_DATA}")
        print("Run the data pipeline to generate DPO preference pairs.")
        sys.exit(1)
    with open(DPO_DATA) as f:
        first = json.loads(f.readline())
    required_keys = {"prompt", "chosen", "rejected"}
    missing = required_keys - set(first.keys())
    if missing:
        print(f"ERROR: dpo.jsonl records must have keys {required_keys}. Missing: {missing}")
        sys.exit(1)
    print(f"DPO data OK: {DPO_DATA}")


def get_sft_adapter_path() -> Path:
    if _CLI_SFT_ADAPTER is not None:
        return Path(_CLI_SFT_ADAPTER)
    return SFT_ADAPTER


def get_dpo_adapter_path() -> Path:
    if _CLI_OUTPUT_DIR is not None:
        return Path(_CLI_OUTPUT_DIR)
    return DPO_ADAPTER


def get_log_file() -> Path:
    if _CLI_LOG_FILE is not None:
        return Path(_CLI_LOG_FILE)
    return LOG_FILE


def check_sft_adapter() -> None:
    sft = get_sft_adapter_path()
    adapter_config = sft / "adapter_config.json"
    if not adapter_config.exists():
        print(
            f"WARNING: SFT adapter not found at {sft}.\n"
            "DPO will train from the base model without SFT initialization.\n"
            "Run train_sft.py first for best results."
        )
    else:
        print(f"SFT adapter found: {sft}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dpo_data(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def batch_iterator(
    records: list[dict],
    batch_size: int,
    tokenizer,
    max_seq_length: int,
    epoch: int = 0,
) -> Iterator[dict]:
    """
    Yield batches of tokenized (prompt+chosen, prompt+rejected) pairs.

    Each batch is a dict with keys:
        chosen_ids:   (B, T_c) int32
        rejected_ids: (B, T_r) int32
        chosen_mask:  (B, T_c) float32
        rejected_mask:(B, T_r) float32

    The `epoch` parameter varies the shuffle seed per epoch so that repeated
    passes through the dataset produce different orderings (fix for
    deterministic-but-identical epoch shuffles).
    """
    import mlx.core as mx
    import numpy as np

    # Fix LOW: vary seed by epoch so repeated passes produce different orderings.
    # Previously recreated RNG with the same DEFAULTS["seed"] every StopIteration,
    # meaning all epochs had identical batch order — hurts generalisation.
    epoch_seed = DEFAULTS["seed"] + epoch
    rng = np.random.default_rng(epoch_seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    batch = []
    for idx in indices:
        batch.append(records[idx])
        if len(batch) == batch_size:
            yield _collate_batch(batch, tokenizer, max_seq_length, mx)
            batch = []
    if batch:
        yield _collate_batch(batch, tokenizer, max_seq_length, mx)


def _extract_dpo_text(record: dict, key: str) -> str:
    """
    Extract a text field from a DPO record.
    Supports both plain strings and messages-list format.
    For 'prompt' as messages list, applies the chat template to get the prompt string.
    For 'chosen'/'rejected' as lists, concatenates assistant content.
    Falls back to str() conversion if unknown format.
    """
    val = record.get(key, "")
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        # messages format: extract content based on role
        if key == "prompt":
            # Extract user (and system) messages as the prompt text
            parts = []
            for msg in val:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("system", "user"):
                    parts.append(content)
            return " ".join(parts)
        else:
            # chosen/rejected: extract assistant message content
            for msg in val:
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            # fallback: last message content
            return val[-1].get("content", "") if val else ""
    return str(val)


def _collate_batch(
    records: list[dict],
    tokenizer,
    max_seq_length: int,
    mx,
) -> dict:
    import numpy as np

    def encode(text: str) -> list[int]:
        ids = tokenizer.encode(text)
        return ids[:max_seq_length]

    def pad_sequences(seqs: list[list[int]]) -> tuple:
        max_len = max(len(s) for s in seqs)
        padded = []
        masks = []
        for s in seqs:
            pad_len = max_len - len(s)
            masks.append([1.0] * len(s) + [0.0] * pad_len)
            padded.append(s + [tokenizer.pad_token_id or 0] * pad_len)
        return (
            mx.array(np.array(padded, dtype=np.int32)),
            mx.array(np.array(masks, dtype=np.float32)),
        )

    # Fix 1-D: insert a separator between prompt and response so the model
    # can distinguish where the prompt ends and the assistant turn begins.
    # Without this, gradients leak across role boundaries during DPO training.
    separator = getattr(tokenizer, "eos_token", None) or "\n\n"

    chosen_seqs = [
        encode(_extract_dpo_text(r, "prompt") + separator + _extract_dpo_text(r, "chosen")) for r in records
    ]
    rejected_seqs = [
        encode(_extract_dpo_text(r, "prompt") + separator + _extract_dpo_text(r, "rejected")) for r in records
    ]
    chosen_ids, chosen_mask = pad_sequences(chosen_seqs)
    rejected_ids, rejected_mask = pad_sequences(rejected_seqs)
    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
    }


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def sequence_log_prob(
    model,
    input_ids,    # (B, T)
    mask,         # (B, T)  1=real token, 0=pad
) -> "mx.array":
    """
    Compute mean per-token log-probability for each sequence in the batch.
    Returns shape (B,).

    NOTE: We use MEAN (not sum) over tokens. Summing over tokens causes log-ratios
    to scale with sequence length — long sequences produce huge log-ratio magnitudes
    that saturate the sigmoid regardless of beta, causing all losses to collapse to
    {0.3133, 0.8133, 1.3133} (the sum-clipping artifact seen with the old code).
    Length-normalised (average) log-probs keep the log-ratio in a bounded range.
    """
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(input_ids).astype(mx.float32)  # (B, T, V) - ensure float32
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_labels = input_ids[:, 1:]    # (B, T-1)
    shift_mask = mask[:, 1:]           # (B, T-1)

    # Edge case: sequence length of 1 produces empty (B, 0, V) shifted tensors.
    # Return zero log-prob for each sequence in the batch to avoid empty-index error.
    B = input_ids.shape[0]
    T = shift_labels.shape[1]
    if T == 0:
        return mx.zeros((B,), dtype=mx.float32)

    log_probs = nn.log_softmax(shift_logits, axis=-1)  # (B, T-1, V)

    # Gather log probs of actual tokens
    token_log_probs = log_probs[mx.arange(B)[:, None], mx.arange(T)[None, :], shift_labels]
    # (B, T-1)

    # Mask padding and compute mean over real tokens (length normalisation)
    token_sum = (token_log_probs * shift_mask).sum(axis=-1)   # (B,)
    token_count = mx.clip(shift_mask.sum(axis=-1), 1.0, None)  # (B,) avoid div-by-zero
    return token_sum / token_count                             # (B,)


def dpo_loss(
    policy_model,
    ref_model,
    batch: dict,
    beta: float,
) -> "mx.array":
    """
    Compute DPO loss for a batch.

    L = -mean[ log σ(β * (log π(y_w|x) - log π_ref(y_w|x)
                          - log π(y_l|x) + log π_ref(y_l|x))) ]
    """
    import mlx.core as mx
    import mlx.nn as nn

    log_pi_chosen = sequence_log_prob(
        policy_model, batch["chosen_ids"], batch["chosen_mask"]
    )
    log_pi_rejected = sequence_log_prob(
        policy_model, batch["rejected_ids"], batch["rejected_mask"]
    )
    log_ref_chosen = mx.stop_gradient(sequence_log_prob(
        ref_model, batch["chosen_ids"], batch["chosen_mask"]
    ))
    log_ref_rejected = mx.stop_gradient(sequence_log_prob(
        ref_model, batch["rejected_ids"], batch["rejected_mask"]
    ))

    log_ratio = (log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected)
    # Scale by beta. With length-normalised log-probs the ratio is typically small
    # (|log_ratio| << 5), so clipping the reward at ±10 is a safety net only.
    rewards = beta * log_ratio
    rewards = mx.clip(rewards, -10.0, 10.0)
    loss = -nn.log_sigmoid(rewards).mean()
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, model_path: Path) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.utils import linear_to_lora_layers

    dpo_adapter = get_dpo_adapter_path()
    log_file = get_log_file()
    sft_adapter = get_sft_adapter_path()

    dpo_adapter.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model from {model_path} ...")
    policy_model, tokenizer = load(str(model_path))

    # Read SFT adapter config to match LoRA settings
    sft_adapter_config_file = sft_adapter / "adapter_config.json"
    sft_has_adapter = sft_adapter_config_file.exists() or (sft_adapter / "adapters.safetensors").exists()

    # Determine LoRA config from SFT adapter if available
    sft_num_layers = args.lora_layers
    sft_lora_rank = args.lora_rank
    sft_lora_scale = 20.0  # safe default matching mlx_lm SFT rank-32 default
    if sft_adapter_config_file.exists():
        with open(sft_adapter_config_file) as f:
            sft_cfg = json.load(f)
        # mlx_lm style config
        if "lora_parameters" in sft_cfg:
            sft_lora_rank = sft_cfg["lora_parameters"].get("rank", args.lora_rank)
            sft_lora_scale = sft_cfg["lora_parameters"].get("scale", sft_lora_scale)
        if "num_layers" in sft_cfg:
            sft_num_layers = sft_cfg["num_layers"]
        print(f"SFT adapter config: rank={sft_lora_rank}, layers={sft_num_layers}, scale={sft_lora_scale}")

    # Write adapter config AFTER reading SFT config so scale/rank are correct.
    # Fix 2-B / 5-B: record actual dropout and use lora_parameters sub-key
    # so GRPO can read scale/rank/num_layers correctly (fix 1-B in train_grpo.py).
    adapter_config = {
        "num_layers": sft_num_layers,
        "lora_parameters": {
            "rank": sft_lora_rank,
            "scale": sft_lora_scale,
            "dropout": 0.05,
        },
        "training": "dpo",
        "beta": args.beta,
    }
    with open(dpo_adapter / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Fix 2-B: use dropout=0.05 to match SFT stage; changing dropout between
    # stages alters the effective scale of new delta updates even with same weights.
    dpo_lora_config = {"rank": sft_lora_rank, "scale": sft_lora_scale, "dropout": 0.05}

    # Apply LoRA FIRST, then load SFT weights (LoRA keys must exist before loading).
    # Fix 5-B: guard against re-initialising LoRA layers that already exist
    # (e.g. when resuming training from a checkpoint with fused weights).
    def _apply_lora_if_needed(mdl, num_layers, config):
        """Apply LoRA only if layers don't already have lora_A (fix 5-B)."""
        first_layer = next(iter(mdl.model.layers), None) if hasattr(mdl, "model") else None
        if first_layer is not None and hasattr(first_layer, "lora_A"):
            print("LoRA layers already present — skipping re-initialisation (fix 5-B).")
            return
        linear_to_lora_layers(mdl, num_layers, config)

    _apply_lora_if_needed(policy_model, sft_num_layers, dpo_lora_config)

    if sft_has_adapter:
        print(f"Loading SFT adapter from {sft_adapter} ...")
        policy_model.load_weights(str(sft_adapter / "adapters.safetensors"), strict=False)

    policy_model.train()

    # Reference model is a frozen copy (base + SFT adapter)
    print("Loading reference model ...")
    ref_model, _ = load(str(model_path))
    _apply_lora_if_needed(ref_model, sft_num_layers, dpo_lora_config)
    if sft_has_adapter:
        ref_model.load_weights(str(sft_adapter / "adapters.safetensors"), strict=False)
    ref_model.eval()
    # Freeze all ref model params
    ref_model.freeze()

    records = load_dpo_data(DPO_DATA)
    print(f"Loaded {len(records)} DPO preference pairs.")

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    def loss_fn(policy_model, batch):
        return dpo_loss(policy_model, ref_model, batch, args.beta)

    loss_and_grad = nn.value_and_grad(policy_model, loss_fn)

    best_loss = float("inf")
    start = time.time()

    with open(log_file, "w") as log:
        log.write(f"DPO Training — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"beta={args.beta}, lr={args.learning_rate}, iters={args.iters}\n\n")

        step = 0
        epoch = 0
        data_iter = batch_iterator(records, args.batch_size, tokenizer, args.max_seq_length, epoch=epoch)

        while step < args.iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reshuffle with a new epoch seed to produce a different ordering
                epoch += 1
                data_iter = batch_iterator(
                    records, args.batch_size, tokenizer, args.max_seq_length, epoch=epoch
                )
                batch = next(data_iter)

            loss, grads = loss_and_grad(policy_model, batch)
            # Gradient clipping to prevent NaN with large models
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(policy_model, grads)
            mx.eval(policy_model.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            step += 1

            # NaN detection — abort early instead of wasting compute
            import math
            if math.isnan(loss_val) or math.isinf(loss_val):
                msg = f"ERROR: NaN/Inf loss detected at step {step}. Aborting."
                print(msg)
                log.write(msg + "\n")
                sys.exit(1)

            if step % args.log_every == 0:
                elapsed = time.time() - start
                msg = f"step {step:5d}/{args.iters} | loss={loss_val:.4f} | {elapsed:.0f}s"
                print(msg)
                log.write(msg + "\n")
                log.flush()

            if step % args.save_every == 0 or step == args.iters:
                save_lora_weights(policy_model, str(dpo_adapter / "adapters.safetensors"))
                print(f"  Saved adapter checkpoint at step {step}")
                if loss_val < best_loss:
                    best_loss = loss_val
                    save_lora_weights(policy_model, str(dpo_adapter / "adapters_best.safetensors"))

    total = time.time() - start
    print(f"\nDPO training complete in {total:.1f}s")
    print(f"Final adapter saved to: {dpo_adapter}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _CLI_MODEL_PATH, _CLI_SFT_ADAPTER, _CLI_OUTPUT_DIR, _CLI_LOG_FILE

    parser = argparse.ArgumentParser(description="DPO training via MLX")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to base model directory (default: models/qwen25-3b-mlx)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to SFT adapter to start from (default: outputs/sft/adapters)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to save DPO adapters (default: outputs/dpo/adapters)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to DPO data JSONL (default: data/processed/train/dpo.jsonl)")
    parser.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lora-layers", type=int, default=DEFAULTS["lora_layers"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--beta", type=float, default=DEFAULTS["beta"],
                        help="KL penalty coefficient for DPO")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    parser.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without running training")
    args = parser.parse_args()

    # Set CLI overrides
    if args.model:
        _CLI_MODEL_PATH = args.model
    if args.adapter_path:
        _CLI_SFT_ADAPTER = args.adapter_path
    if args.output_dir:
        _CLI_OUTPUT_DIR = args.output_dir
        _CLI_LOG_FILE = str(Path(args.output_dir).parent / "train.log")
    if args.data:
        global DPO_DATA
        DPO_DATA = Path(args.data)

    check_dependencies()
    model_path = resolve_model_path()
    check_data()
    check_sft_adapter()

    if args.dry_run:
        print("\nDRY RUN — all checks passed. Training not started.")
        return

    train(args, model_path)


if __name__ == "__main__":
    main()

```

### scripts/train_grpo.py

```
#!/usr/bin/env python3
"""
GRPO Training Script using MLX.

Group Relative Policy Optimization (Shao et al., 2024) for tax-law RL.

Algorithm per step:
    1. Sample a prompt from the dataset.
    2. Generate K completions using the current policy (temperature sampling).
    3. Score each completion with the reward function from grpo_reward.py.
    4. Normalise rewards within the group: r̂_i = (r_i - mean(r)) / (std(r) + ε)
    5. Compute policy-gradient loss with KL-from-reference clipping (PPO-style):
         L = -mean_i[ min(ρ_i · r̂_i, clip(ρ_i, 1-ε_clip, 1+ε_clip) · r̂_i) ]
       where ρ_i = π(y_i|x) / π_ref(y_i|x)
    6. Update policy; repeat.

Input data format (JSONL):
    { "prompt": "..." }

Usage:
    python scripts/train_grpo.py [--iters 300] [--group-size 4] [--dry-run]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_MLX = PROJECT_ROOT / "models" / "qwen25-3b-mlx"
MODEL_HF = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct"
SFT_ADAPTER = PROJECT_ROOT / "outputs" / "sft" / "adapters"
DPO_ADAPTER = PROJECT_ROOT / "outputs" / "dpo" / "adapters"
GRPO_ADAPTER = PROJECT_ROOT / "outputs" / "grpo" / "adapters"
GRPO_DATA = PROJECT_ROOT / "data" / "processed" / "train" / "grpo.jsonl"
SFT_DATA = PROJECT_ROOT / "data" / "train_v2" / "sft.jsonl"
LOG_FILE = PROJECT_ROOT / "outputs" / "grpo" / "train.log"

# CLI overrides
_CLI_MODEL_PATH = None
_CLI_START_ADAPTER = None
_CLI_OUTPUT_DIR = None
_CLI_LOG_FILE = None
_CLI_SFT_DATA = None

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = {
    "iters": 300,
    "group_size": 8,           # K completions per prompt; use 4 if OOM (fix 2-D: 8 gives better sample efficiency on 3B vs 4)
    "batch_size": 1,           # prompts per gradient step
    "learning_rate": 1e-6,
    "lora_layers": 16,
    "lora_rank": 32,
    "max_new_tokens": 512,
    "temperature": 0.8,
    "epsilon_clip": 0.2,       # PPO clip epsilon
    "kl_coeff": 0.01,          # additional KL penalty weight
    "save_every": 50,
    "log_every": 5,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Dependency and path checks
# ---------------------------------------------------------------------------

def _get_model_items(model) -> list:
    """
    Return a flat list of (key, value) pairs from a model's parameters.

    Handles both modern MLX (tree_flatten returns [(k,v)...]) and future
    versions that may rename the function.  Always returns (key, value) pairs.
    """
    import mlx.utils as mu
    # Modern MLX (>=0.17): tree_flatten returns list[(str, array)]
    # Older MLX: same API — tree_flatten has always returned (k,v) pairs
    # tree_flatten_items is an alias added in some intermediate version.
    # We prefer tree_flatten_items if available, otherwise tree_flatten.
    fn = getattr(mu, "tree_flatten_items", None) or getattr(mu, "tree_flatten")
    result = fn(model.parameters())
    # Sanity-check: result must be a list/sequence of 2-tuples
    if not isinstance(result, (list, tuple)):
        raise RuntimeError(f"Unexpected tree_flatten return type: {type(result)}")
    if result and not isinstance(result[0], (list, tuple)):
        raise RuntimeError(
            f"tree_flatten returned non-pair items (got {type(result[0]).__name__}). "
            "Cannot extract LoRA weights. Check MLX version compatibility."
        )
    return result


def save_lora_weights(model, path: str) -> None:
    """Save only LoRA adapter weights, not the full model.

    Raises RuntimeError if no LoRA parameters are found, to avoid silently
    saving the full model (6+ GB) when checkpointing was intended to save
    a small adapter (< 50 MB).
    """
    import mlx.core as mx
    all_params = _get_model_items(model)
    lora_params = {k: v for k, v in all_params if "lora" in k.lower()}
    if not lora_params:
        # Fail hard rather than silently saving 6 GB of full model weights.
        raise RuntimeError(
            "save_lora_weights: no LoRA parameters found in model. "
            "Ensure linear_to_lora_layers() has been called before saving. "
            f"Available parameter keys: {[k for k, _ in all_params[:10]]!r}"
        )
    mx.save_safetensors(path, lora_params)


def check_dependencies() -> None:
    missing = []
    for pkg in ["mlx", "mlx.core", "mlx.nn", "mlx_lm"]:
        try:
            __import__(pkg.replace(".", "_") if "." not in pkg else pkg.split(".")[0])
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {missing}")
        print("Install with: pip install mlx mlx-lm")
        sys.exit(1)
    # Check reward module
    reward_path = PROJECT_ROOT / "scripts" / "grpo_reward.py"
    if not reward_path.exists():
        print(f"ERROR: Reward function not found at {reward_path}")
        print("Create scripts/grpo_reward.py before running GRPO training.")
        sys.exit(1)


def resolve_model_path() -> Path:
    if _CLI_MODEL_PATH is not None:
        p = Path(_CLI_MODEL_PATH)
        if p.exists() and (p / "config.json").exists():
            return p
        print(f"ERROR: No model found at {p}")
        sys.exit(1)
    if MODEL_MLX.exists() and (MODEL_MLX / "config.json").exists():
        return MODEL_MLX
    if MODEL_HF.exists() and (MODEL_HF / "config.json").exists():
        print(f"Using HF model (mlx_lm will convert): {MODEL_HF}")
        return MODEL_HF
    print(f"ERROR: No model found at {MODEL_MLX} or {MODEL_HF}")
    sys.exit(1)


def get_grpo_adapter_path() -> Path:
    if _CLI_OUTPUT_DIR is not None:
        return Path(_CLI_OUTPUT_DIR)
    return GRPO_ADAPTER


def get_log_file() -> Path:
    if _CLI_LOG_FILE is not None:
        return Path(_CLI_LOG_FILE)
    return LOG_FILE


def resolve_start_adapter() -> Path | None:
    """Return the best available starting adapter (DPO > SFT > None)."""
    if _CLI_START_ADAPTER is not None:
        p = Path(_CLI_START_ADAPTER)
        if (p / "adapter_config.json").exists():
            print(f"Starting from adapter: {p}")
            return p
        # Also check if the safetensors file exists (even without adapter_config.json)
        if (p / "adapters.safetensors").exists():
            print(f"Starting from adapter: {p} (no adapter_config.json found, using safetensors)")
            return p
        print(f"WARNING: Specified adapter path {p} has no adapter_config.json or safetensors")
        return None
    for adapter_dir in [DPO_ADAPTER, SFT_ADAPTER]:
        if (adapter_dir / "adapter_config.json").exists():
            print(f"Starting from adapter: {adapter_dir}")
            return adapter_dir
    print("WARNING: No prior adapter found. Training from base model.")
    return None


def extract_prompt(rec: dict) -> str | None:
    """Extract prompt string from a record (supports 'prompt' key or 'messages' format)."""
    if "prompt" in rec:
        return rec["prompt"]
    if "messages" in rec:
        # Extract user message content as prompt
        msgs = rec["messages"]
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
        return user_msg
    return None


def check_data() -> None:
    if not GRPO_DATA.exists():
        print(f"ERROR: GRPO data not found at {GRPO_DATA}")
        print("Run the data pipeline to generate GRPO prompts.")
        sys.exit(1)
    with open(GRPO_DATA) as f:
        first = json.loads(f.readline())
    prompt = extract_prompt(first)
    if prompt is None:
        print(f"ERROR: grpo.jsonl records must have a 'prompt' key or 'messages' list. Got: {list(first.keys())}")
        sys.exit(1)
    print(f"GRPO data OK: {GRPO_DATA}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_sft_data_path() -> Path | None:
    """Return the SFT data path (CLI override or default). Returns None if disabled."""
    if _CLI_SFT_DATA is not None:
        return None if _CLI_SFT_DATA == "" else Path(_CLI_SFT_DATA)
    return SFT_DATA


def build_reference_lookup(sft_path: Path | None) -> dict[str, str]:
    """
    Build a mapping of {user_prompt -> reference_answer} from an SFT JSONL file.

    The SFT format is:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    Returns an empty dict if sft_path is None or the file does not exist.
    """
    if sft_path is None:
        print("SFT reference file disabled. Factual accuracy scoring will use neutral 0.5.")
        return {}
    if not sft_path.exists():
        print(f"WARNING: SFT reference file not found at {sft_path}. "
              "Factual accuracy scoring will use neutral 0.5 for all examples.")
        return {}

    lookup: dict[str, str] = {}
    with open(sft_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if user_msg and asst_msg:
                lookup[user_msg] = asst_msg

    print(f"Loaded {len(lookup)} reference answers from {sft_path}")
    return lookup


def load_prompts(path: Path, reference_lookup: dict[str, str] | None = None) -> list[dict]:
    """
    Load GRPO data as list of dicts with 'prompt', optional 'expected_section',
    and optional 'reference' answer (populated from reference_lookup if provided).
    Supports both 'prompt' key and 'messages' format.
    """
    if reference_lookup is None:
        reference_lookup = {}
    records = []
    matched = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                prompt = extract_prompt(rec)
                if prompt is None:
                    continue
                reference = reference_lookup.get(prompt, None)
                if reference:
                    matched += 1
                records.append({
                    "prompt": prompt,
                    "expected_section": rec.get("expected_section", None),
                    "reference": reference,
                })
    if reference_lookup:
        print(f"Reference answers matched: {matched}/{len(records)} GRPO prompts "
              f"({matched / max(len(records), 1):.1%})")
    return records


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_completions(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """
    Generate `group_size` completions for a single prompt using temperature sampling.
    Returns list of completion strings (excluding the prompt).
    """
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)

    completions = []
    for _ in range(group_size):
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        # Strip the prompt prefix if the model echoes it
        if output.startswith(prompt):
            output = output[len(prompt):]
        completions.append(output.strip())
    return completions


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------

def sequence_log_prob(model, tokenizer, text: str, max_seq_length: int):
    """
    Compute the mean per-token log-prob for a sequence.
    Returns a scalar mlx array.

    Fix 1-A: Use MEAN (not sum) over tokens. Summing causes importance ratio
    rho = exp(log_pi - log_ref) to scale with sequence length, making PPO clip
    fire constantly on long completions and destroying the learning signal.
    Length-normalised (average) log-probs keep rho in a bounded range.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    ids = tokenizer.encode(text)[:max_seq_length]
    ids_arr = mx.array(np.array([ids], dtype=np.int32))  # (1, T)

    logits = model(ids_arr)             # (1, T, V)
    shift_logits = logits[:, :-1, :]    # (1, T-1, V)
    shift_labels = ids_arr[:, 1:]       # (1, T-1)

    # Build a real-token mask (all 1s — no padding in a single sequence)
    mask = mx.ones_like(shift_labels, dtype=mx.float32)  # (1, T-1)

    log_probs = nn.log_softmax(shift_logits, axis=-1)
    T = shift_labels.shape[1]
    token_log_probs = log_probs[0, mx.arange(T), shift_labels[0]]  # (T-1,)

    # Fix 1-A: length-normalise so that rho does not explode with long completions
    token_sum = (token_log_probs * mask[0]).sum(axis=-1)
    token_cnt = mx.clip(mask[0].sum(axis=-1), 1.0, None)
    return token_sum / token_cnt


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss_for_prompt(
    policy_model,
    ref_model,
    tokenizer,
    prompt: str,
    completions: list[str],
    rewards: list[float],
    args: argparse.Namespace,
):
    """
    Compute GRPO / PPO-style policy gradient loss for one prompt group.

    Uses importance-weighted advantage with PPO clipping.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    r = np.array(rewards, dtype=np.float32)
    r_std = r.std() + 1e-8
    advantages = (r - r.mean()) / r_std

    total_loss = mx.array(0.0)

    for completion, adv in zip(completions, advantages):
        # Fix 1-E: insert separator between prompt and completion so the model
        # can distinguish where the prompt ends and the assistant turn begins.
        # Using eos_token as delimiter; falls back to "\n\n" if not available.
        separator = getattr(tokenizer, "eos_token", None) or "\n\n"
        full_text = prompt + separator + completion
        max_len = args.max_new_tokens + 128  # rough bound

        log_pi = sequence_log_prob(policy_model, tokenizer, full_text, max_len)
        log_ref = mx.stop_gradient(sequence_log_prob(ref_model, tokenizer, full_text, max_len))

        # Importance ratio (scalar)
        rho = mx.exp(log_pi - log_ref)

        adv_tensor = mx.array(float(adv))

        # PPO clipped objective
        unclipped = rho * adv_tensor
        clipped = mx.clip(rho, 1 - args.epsilon_clip, 1 + args.epsilon_clip) * adv_tensor
        pg_loss = -mx.minimum(unclipped, clipped)

        # KL penalty: KL(π || π_ref) ≈ log(ρ)
        kl_penalty = args.kl_coeff * (log_pi - log_ref)

        total_loss = total_loss + pg_loss + kl_penalty

    return total_loss / len(completions)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, model_path: Path) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    # Import reward function
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from grpo_reward import compute_reward  # noqa: E402

    grpo_adapter = get_grpo_adapter_path()
    log_file = get_log_file()

    grpo_adapter.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading policy model from {model_path} ...")
    policy_model, tokenizer = load(str(model_path))

    start_adapter = resolve_start_adapter()

    # Fix 1-B: read LoRA scale/rank/num_layers from the prior adapter's config
    # so we don't silently shrink SFT/DPO deltas by hardcoding scale=1.0.
    # The SFT adapter is trained with scale=20.0; using 1.0 would wipe out all
    # inherited knowledge before the first GRPO update.
    lora_scale = 20.0   # safe default matching mlx_lm rank-32 SFT default
    lora_rank = args.lora_rank
    lora_num_layers = args.lora_layers
    if start_adapter is not None:
        adapter_config_path = start_adapter / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            # mlx_lm stores these under "lora_parameters" sub-key
            lora_params_cfg = adapter_cfg.get("lora_parameters", {})
            lora_scale = lora_params_cfg.get("scale", adapter_cfg.get("scale", lora_scale))
            lora_rank = lora_params_cfg.get("rank", adapter_cfg.get("lora_rank", lora_rank))
            lora_num_layers = adapter_cfg.get("num_layers", adapter_cfg.get("lora_layers", lora_num_layers))
            print(f"Adapter config: scale={lora_scale}, rank={lora_rank}, num_layers={lora_num_layers}")

    # Fix 2-B: use dropout=0.05 to match SFT stage; changing dropout between
    # stages alters the effective scale of new delta updates.
    lora_config = {"rank": lora_rank, "scale": lora_scale, "dropout": 0.05}

    # Apply LoRA to policy BEFORE loading adapter weights.
    # linear_to_lora_layers must run first so that LoRA parameter keys
    # (lora_A, lora_B, etc.) exist in the model before load_weights tries
    # to populate them. Loading weights before this call silently discards
    # all LoRA keys because the layers don't exist yet (strict=False).
    linear_to_lora_layers(policy_model, lora_num_layers, lora_config)

    if start_adapter is not None:
        print(f"Initializing policy LoRA from adapter: {start_adapter}")
        policy_model.load_weights(str(start_adapter / "adapters.safetensors"), strict=False)

    policy_model.train()

    # Frozen reference model
    print("Loading reference model ...")
    ref_model, _ = load(str(model_path))
    # Apply LoRA to reference model first, then load adapter weights.
    linear_to_lora_layers(ref_model, lora_num_layers, lora_config)
    if start_adapter is not None:
        ref_model.load_weights(str(start_adapter / "adapters.safetensors"), strict=False)
    ref_model.eval()
    ref_model.freeze()

    # Build reference lookup from SFT data for factual accuracy scoring.
    # Falls back gracefully: if the file is absent, all references will be None
    # and factual_accuracy_score() returns the neutral 0.5 score.
    sft_path = get_sft_data_path()
    reference_lookup = build_reference_lookup(sft_path)

    prompt_records = load_prompts(GRPO_DATA, reference_lookup)
    print(f"Loaded {len(prompt_records)} GRPO prompts.")

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    import numpy as np
    rng = np.random.default_rng(args.seed)

    start_time = time.time()
    best_avg_reward = -float("inf")

    with open(log_file, "w") as log:
        log.write(f"GRPO Training — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(
            f"group_size={args.group_size}, lr={args.learning_rate}, "
            f"iters={args.iters}, eps_clip={args.epsilon_clip}\n\n"
        )

        for step in range(1, args.iters + 1):
            # Sample a prompt record
            rec = prompt_records[rng.integers(len(prompt_records))]
            prompt = rec["prompt"]
            expected_section = rec.get("expected_section", None)
            reference = rec.get("reference", None)

            # Generate K completions (no grad)
            policy_model.eval()
            completions = generate_completions(
                policy_model, tokenizer, prompt,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            policy_model.train()

            # Score completions with citation and factual accuracy
            rewards = [
                compute_reward(prompt, c, reference=reference, expected_section=expected_section)
                for c in completions
            ]

            # Compute loss
            # Fix CRITICAL: loss_fn must NOT take model as an argument when
            # using nn.value_and_grad(model, fn). The returned callable from
            # value_and_grad takes the same args as fn; passing policy_model
            # twice (as bind arg AND as call arg) is at minimum redundant and
            # can raise TypeError on some MLX versions. Use a closure instead
            # so loss_fn() takes zero arguments, matching the zero-arg call.
            def loss_fn():
                return grpo_loss_for_prompt(
                    policy_model, ref_model, tokenizer,
                    prompt, completions, rewards, args,
                )

            loss, grads = nn.value_and_grad(policy_model, loss_fn)()
            optimizer.update(policy_model, grads)
            mx.eval(policy_model.parameters(), optimizer.state, loss)

            avg_reward = float(np.mean(rewards))

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                msg = (
                    f"step {step:4d}/{args.iters} | "
                    f"loss={loss.item():.4f} | "
                    f"avg_reward={avg_reward:.3f} | "
                    f"max_reward={max(rewards):.3f} | "
                    f"{elapsed:.0f}s"
                )
                print(msg)
                log.write(msg + "\n")
                log.flush()

            if step % args.save_every == 0 or step == args.iters:
                save_lora_weights(policy_model, str(grpo_adapter / "adapters.safetensors"))
                # Fix HIGH: write adapter_config.json at every checkpoint so
                # downstream scripts (export_to_ollama, evaluate, future RL
                # stages) read the correct scale/rank/dropout rather than a
                # stale file from a previous mlx_lm.lora run.
                grpo_adapter_config = {
                    "num_layers": lora_num_layers,
                    "lora_parameters": {
                        "rank": lora_rank,
                        "scale": lora_scale,
                        "dropout": 0.05,
                    },
                    "training": "grpo",
                    "group_size": args.group_size,
                    "eps_clip": args.epsilon_clip,
                    "step": step,
                }
                with open(grpo_adapter / "adapter_config.json", "w") as _f:
                    json.dump(grpo_adapter_config, _f, indent=2)
                print(f"  Saved adapter checkpoint at step {step}")
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_lora_weights(policy_model, str(grpo_adapter / "adapters_best.safetensors"))

    total = time.time() - start_time
    print(f"\nGRPO training complete in {total:.1f}s")
    print(f"Best average reward: {best_avg_reward:.3f}")
    print(f"Final adapter saved to: {grpo_adapter}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _CLI_MODEL_PATH, _CLI_START_ADAPTER, _CLI_OUTPUT_DIR, _CLI_LOG_FILE, _CLI_SFT_DATA

    parser = argparse.ArgumentParser(description="GRPO training via MLX")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to base model directory (default: models/qwen25-3b-mlx)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to starting adapter (DPO or SFT) to load weights from")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to save GRPO adapters (default: outputs/grpo/adapters)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to GRPO data JSONL (default: data/processed/train/grpo.jsonl)")
    parser.add_argument("--sft-data", type=str, default=None,
                        help="Path to SFT JSONL with reference answers for factual accuracy scoring "
                             "(default: data/train_v2/sft.jsonl). Pass 'none' to disable.")
    parser.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    parser.add_argument("--group-size", type=int, default=DEFAULTS["group_size"],
                        help="Number of completions to generate per prompt (K)")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--lora-layers", type=int, default=DEFAULTS["lora_layers"])
    parser.add_argument("--lora-rank", type=int, default=DEFAULTS["lora_rank"])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--epsilon-clip", type=float, default=DEFAULTS["epsilon_clip"])
    parser.add_argument("--kl-coeff", type=float, default=DEFAULTS["kl_coeff"])
    parser.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    parser.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without running training")
    args = parser.parse_args()

    # Set CLI overrides
    if args.model:
        _CLI_MODEL_PATH = args.model
    if args.adapter_path:
        _CLI_START_ADAPTER = args.adapter_path
    if args.output_dir:
        _CLI_OUTPUT_DIR = args.output_dir
        _CLI_LOG_FILE = str(Path(args.output_dir).parent / "train.log")
    if args.data:
        global GRPO_DATA
        GRPO_DATA = Path(args.data)
    if args.sft_data:
        if args.sft_data.lower() == "none":
            _CLI_SFT_DATA = ""  # Empty string signals: no SFT reference file
        else:
            _CLI_SFT_DATA = args.sft_data

    check_dependencies()
    model_path = resolve_model_path()
    check_data()

    if args.dry_run:
        print("\nDRY RUN — all checks passed. Training not started.")
        return

    train(args, model_path)


if __name__ == "__main__":
    main()

```

### scripts/grpo_reward.py

```
#!/usr/bin/env python3
"""
GRPO reward function for tax law responses.

Rewards higher-quality responses that:
1. Cite specific IRC/CFR sections                (citation_format)
2. Cite the *correct* section for the question  (citation_accuracy)
3. Reproduce key factual numbers from reference  (factual_accuracy)  [NEW v4]
4. Are sufficiently detailed                     (length)
5. Avoid vague non-answers                       (vague_penalty)

Weight breakdown (v4):
    factual_accuracy  = 0.30
    citation_accuracy = 0.25
    citation_format   = 0.20
    length            = 0.15
    vague_penalty     = 0.10  (applied as a deduction)
"""
import math
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Shared utilities (canonical citation regex lives here)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from citation_utils import (  # noqa: E402
    count_citations,
    extract_irc_sections,
    extract_numbers,
    extract_section_number,
    IRC_CITATION_PATTERN,
    CFR_CITATION_PATTERN,
)

# ---------------------------------------------------------------------------
# Vague non-answer phrases
# ---------------------------------------------------------------------------
VAGUE_PHRASES = [
    "consult a tax professional",
    "depends on your circumstances",
    "complex and vary",
    "facts and circumstances",
    "i cannot provide",
    "i am not able to",
    "please seek professional advice",
    "this is not legal advice",
    "i'm not able to give",
    "you should talk to",
]

# Precision legal language that indicates quality
LEGAL_PRECISION_TERMS = [
    "taxable income",
    "gross income",
    "adjusted gross income",
    "deduction",
    "exclusion",
    "credit",
    "basis",
    "recognition",
    "realization",
    "ordinary income",
    "capital gain",
    "tax liability",
    "filing status",
    "taxpayer",
    "fiscal year",
    "taxable year",
    "withholding",
    "estimated tax",
    "penalty",
    "interest",
    "statute of limitations",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_vague_language(response: str) -> bool:
    """Check if response contains vague non-answer language."""
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in VAGUE_PHRASES)


def count_legal_terms(response: str) -> int:
    """Count precision legal terms used."""
    response_lower = response.lower()
    return sum(1 for term in LEGAL_PRECISION_TERMS if term in response_lower)


def extract_cited_sections(response: str) -> list[str]:
    """
    Return a list of base IRC section numbers cited in *response*.

    Uses the canonical IRC_CITATION_PATTERN from citation_utils.
    """
    return list(extract_irc_sections(response))


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def citation_accuracy_score(response: str, expected_section: Optional[str]) -> float:
    """
    Check if the model cites the correct IRC section.

    Returns:
        1.0  — expected section is among cited sections
        0.25 — no expected section provided (uncertain/neutral; cannot verify)
        0.2  — model cited *some* sections but none match the expected one
        0.0  — model cited no sections at all

    Fix 3-B (review item 3-B MEDIUM): default when expected_section is None
    was 0.5 (half credit), which biased the model toward always citing
    something even on unannotated questions.  Changed to 0.25 (uncertain)
    to reduce that bias.
    """
    if not expected_section:
        return 0.25  # No ground truth; uncertain/neutral (fix 3-B)

    expected_num = extract_section_number(expected_section)
    if not expected_num:
        return 0.25  # Cannot parse expected section; uncertain/neutral (fix 3-B)

    cited = extract_cited_sections(response)
    if not cited:
        return 0.0  # No citations whatsoever

    if expected_num in cited:
        return 1.0  # Correct section cited

    return 0.2  # Wrong sections cited


def normalize_number(s: str) -> str:
    """
    Normalise a number string for comparison.

    Fix 1-F (review item 1-F MEDIUM): raw string comparison failed to match
    equivalent values like "$1,160,000" and "1160000".  Stripping "$", ","
    and leading zeros ensures that different textual representations of the
    same value are treated as equal.

    Examples
    --------
    >>> normalize_number("$1,160,000")
    '1160000'
    >>> normalize_number("0050")
    '50'
    >>> normalize_number("0")
    '0'
    """
    return s.replace("$", "").replace(",", "").lstrip("0") or "0"


def factual_accuracy_score(response: str, reference: Optional[str]) -> float:
    """
    Measure how many key numbers from *reference* appear in *response*.

    Key numbers = dollar amounts ($25,000) and percentages (20%).

    Numbers are normalised before comparison so that "$1,160,000" and
    "1160000" are treated as equal (fix 1-F).

    Returns:
        float in [0.0, 1.0] — fraction of reference numbers present in response.
        0.5 if reference is absent or contains no numbers (neutral; cannot verify).
    """
    if not reference or not reference.strip():
        return 0.5  # No reference; neutral

    ref_numbers = extract_numbers(reference)
    if not ref_numbers:
        return 0.5  # Reference has no numbers to verify against; neutral

    resp_numbers = extract_numbers(response)
    if not resp_numbers:
        return 0.0  # Reference has numbers but response has none

    # Normalise both sets before computing overlap (fix 1-F)
    ref_normalized = {normalize_number(n) for n in ref_numbers}
    resp_normalized = {normalize_number(n) for n in resp_numbers}
    matched = ref_normalized.intersection(resp_normalized)
    return len(matched) / len(ref_normalized)


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    prompt: str,
    response: str,
    reference: Optional[str] = None,
    expected_section: Optional[str] = None,
) -> float:
    """
    Compute a scalar reward for a tax law response.

    Returns a float clamped to [0.0, 1.0].

    Weight breakdown (v4):
        factual_accuracy  = 0.30  (key numbers from reference present in response)
        citation_accuracy = 0.25  (correct IRC section cited)
        citation_format   = 0.20  (citations present, up to 4 for full score)
        length            = 0.15  (200–1500 chars ideal)
        vague_penalty     = 0.10  (deducted if vague language detected)

    Args:
        prompt:           The user question (unused in scoring currently but
                          kept for API symmetry).
        response:         The model's answer.
        reference:        Gold-standard reference answer; used for factual
                          accuracy (number overlap).
        expected_section: The IRC section the question is about (e.g. "179").
                          Used for citation accuracy scoring.
    """
    if not response or not response.strip():
        return 0.0

    # 1. Factual accuracy (0.0 – 0.30)
    factual = factual_accuracy_score(response, reference)
    factual_score = factual * 0.30

    # 2. Citation accuracy (0.0 – 0.25)
    accuracy = citation_accuracy_score(response, expected_section)
    citation_accuracy = accuracy * 0.25

    # 3. Citation format (0.0 – 0.20)
    # Fix 3-C (review item 3-C LOW): use diminishing-returns curve instead of
    # linear up to 4 citations.  score = 1 - exp(-n/2) means each additional
    # citation has less marginal value, discouraging padding with irrelevant refs.
    n_citations = count_citations(response)
    citation_format_score = (1.0 - math.exp(-n_citations / 2.0)) * 0.20

    # 4. Length / detail (0.0 – 0.15)
    response_len = len(response)
    if response_len < 50:
        length_score = 0.0
    elif response_len < 200:
        length_score = (response_len - 50) / 150 * 0.10
    elif response_len <= 1500:
        length_score = 0.15
    elif response_len <= 3000:
        length_score = 0.15 - (response_len - 1500) / 1500 * 0.05
    else:
        length_score = 0.10

    # 5. Vague language penalty (-0.10)
    vague_penalty = -0.10 if has_vague_language(response) else 0.0

    # Fix 3-A (review item 3-A HIGH): component weights sum to at most 0.90
    # (0.30+0.25+0.20+0.15) before the vague_penalty adjustment, so the total
    # cannot exceed 1.0 with the current weights.  The clamp below ensures the
    # final score stays in [0.0, 1.0] regardless of future weight changes.
    # Note: citation_format uses a diminishing-returns curve (fix 3-C) that
    # approaches 0.20 asymptotically, so it also cannot push the total above 1.0.
    total = factual_score + citation_accuracy + citation_format_score + length_score + vague_penalty
    return max(0.0, min(1.0, total))


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def batch_reward(
    prompts: list[str],
    responses: list[str],
    references: Optional[list[str]] = None,
    expected_sections: Optional[list[Optional[str]]] = None,
) -> list[float]:
    """
    Compute rewards for a batch of (prompt, response) pairs.

    Args:
        prompts:           List of input prompts.
        responses:         List of model responses.
        references:        Optional list of reference answers (for factual
                           accuracy).  Defaults to all-None.
        expected_sections: Optional list of expected IRC section strings (for
                           citation accuracy).  Defaults to all-None.

    Returns:
        List of float rewards in [0.0, 1.0].

    Note:
        Previously *expected_sections* was missing from this function, so
        citation_accuracy_score always returned the neutral 0.5.  This is
        now fixed.
    """
    n = len(prompts)
    if references is None:
        references = [None] * n
    if expected_sections is None:
        expected_sections = [None] * n

    return [
        compute_reward(p, r, ref, sec)
        for p, r, ref, sec in zip(prompts, responses, references, expected_sections)
    ]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing GRPO reward function (v4)...\n")

    REFERENCE_179 = (
        "Under IRC Section 179, a taxpayer may elect to expense the cost of qualifying "
        "depreciable property placed in service during the tax year. For 2023, the "
        "maximum deduction is $1,160,000, subject to a phase-out when qualifying "
        "property exceeds $2,890,000. The property must be used more than 50% for "
        "business purposes."
    )

    test_cases = [
        {
            "name": "High quality — correct section + all numbers",
            "prompt": "What is the Section 179 expensing limit?",
            "response": (
                "IRC Section 179 allows taxpayers to immediately expense qualifying "
                "depreciable property. The 2023 deduction limit is $1,160,000. "
                "This limit phases out dollar-for-dollar when total property placed "
                "in service exceeds $2,890,000. Property must exceed 50% business use. "
                "See also 26 CFR § 1.179-1 for Treasury Regulation details."
            ),
            "reference": REFERENCE_179,
            "expected_section": "179",
        },
        {
            "name": "Wrong numbers — correct section cited",
            "prompt": "What is the Section 179 expensing limit?",
            "response": (
                "Under IRC Section 179 you can deduct up to $500,000 of equipment "
                "costs, with a phase-out starting at $2,000,000."
            ),
            "reference": REFERENCE_179,
            "expected_section": "179",
        },
        {
            "name": "Vague non-answer",
            "prompt": "What is the Section 179 expensing limit?",
            "response": (
                "This is a complex area of tax law that depends on your circumstances. "
                "You should consult a tax professional for advice specific to your situation."
            ),
            "reference": REFERENCE_179,
            "expected_section": "179",
        },
        {
            "name": "Moderate quality — no citations, some numbers",
            "prompt": "What is the Section 179 expensing limit?",
            "response": (
                "Businesses can expense up to $1,160,000 of qualifying property in the "
                "year it is placed in service. A phase-out applies above $2,890,000."
            ),
            "reference": REFERENCE_179,
            "expected_section": "179",
        },
        {
            "name": "Empty response",
            "prompt": "What is IRC Section 1?",
            "response": "",
            "reference": None,
            "expected_section": "1",
        },
    ]

    for tc in test_cases:
        reward = compute_reward(
            tc["prompt"], tc["response"],
            reference=tc.get("reference"),
            expected_section=tc.get("expected_section"),
        )
        print(f"Test: {tc['name']}")
        print(f"  Citations found:    {count_citations(tc['response'])}")
        print(f"  Cited sections:     {extract_cited_sections(tc['response'])}")
        print(f"  Numbers in resp:    {extract_numbers(tc['response'])}")
        print(f"  Vague:             {has_vague_language(tc['response'])}")
        print(f"  Factual accuracy:  {factual_accuracy_score(tc['response'], tc.get('reference')):.3f}")
        print(f"  Citation accuracy: {citation_accuracy_score(tc['response'], tc.get('expected_section')):.3f}")
        print(f"  Reward:            {reward:.3f}")
        print()

```

### scripts/citation_utils.py

```
#!/usr/bin/env python3
"""
Shared citation utilities for the IRS tax-code RL project.

All scripts that need to detect or extract IRC/CFR section citations
should import from this module rather than defining their own patterns.
This ensures consistent behaviour across grpo_reward.py, evaluate.py,
and generate_onpolicy_dpo.py.
"""
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Canonical citation regex
# ---------------------------------------------------------------------------
# Matches all of the following (case-insensitive):
#   Section 179          →  bare "Section N"
#   §179 / § 179         →  bare section-sign with optional space
#   IRC §179 / IRC 179   →  "IRC" prefix
#   I.R.C. §179          →  dotted abbreviation
#   I.R.C. § 179(d)(1)  →  with subsections
#   26 U.S.C. §179       →  title-26 USC citation
#   Internal Revenue Code Section 179
#   Sec. 179             →  abbreviated "Sec."
#   CFR / Treasury Regs citations are handled by CFR_CITATION_PATTERN below
#
# Capturing group 1: the numeric section (digits + optional trailing letter),
#                    e.g.  "179", "168", "199A", "408A".
# Capturing group 2 (non-captured internally): optional subsection string,
#                    e.g.  "(d)(1)", "(k)", "(t)".
# The public API always returns the base section number only (group 1).

_IRC_PREFIXES = r"""
    (?:
        (?:IRC|I\.R\.C\.)                       # IRC or I.R.C.
        |
        (?:26\s+U\.S\.C\.)                      # 26 U.S.C.
        |
        (?:Internal\s+Revenue\s+Code)           # spelled out
    )
    \s*
    (?:Section|Sec\.?|§)?                       # optional "Section"/"Sec."/"§"
    \s*
"""

_SECTION_KEYWORD = r"""
    (?:Section|Sec\.?)\s+                       # bare "Section N" or "Sec. N"
"""

_SECTION_SIGN = r"""
    (?<!C\.F\.R\.\s)(?<!C\.F\.R\.)             # Fix 1-I (review item 1-I LOW): negative
    (?<!CFR\s)(?<!CFR)                          # lookbehind so that CFR/C.F.R. section
    §\s*                                        # signs are NOT counted as IRC citations
"""

IRC_CITATION_PATTERN = re.compile(
    rf"""
    (?:
        {_IRC_PREFIXES}
        |
        {_SECTION_KEYWORD}
        |
        {_SECTION_SIGN}
    )
    (\d+[A-Za-z]?)                              # base section number (group 1)
    (?:\([^\)]*\))*                             # optional subsection(s) like (d)(1)
    """,
    re.IGNORECASE | re.VERBOSE,
)

CFR_CITATION_PATTERN = re.compile(
    r"(?:26\s*C\.?F\.?R\.?|Treasury\s*Reg(?:ulation)?s?)\s*[§\s]*(\d+[\.\w\-]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_irc_sections(text: str) -> set[str]:
    """
    Return the set of base IRC section numbers cited in *text*.

    Examples
    --------
    >>> extract_irc_sections("Under IRC §179 and Section 168(k)...")
    {'179', '168'}
    >>> extract_irc_sections("See 26 U.S.C. §1031 and I.R.C. Section 408A")
    {'1031', '408A'}
    """
    sections: set[str] = set()
    for m in IRC_CITATION_PATTERN.finditer(text):
        sections.add(m.group(1))
    return sections


def count_citations(text: str) -> int:
    """
    Count total IRC + CFR citations in *text*.

    IRC citations detected via IRC_CITATION_PATTERN; CFR citations via
    CFR_CITATION_PATTERN.  Overlapping matches are not double-counted.
    """
    irc_count = len(IRC_CITATION_PATTERN.findall(text))
    cfr_count = len(CFR_CITATION_PATTERN.findall(text))
    return irc_count + cfr_count


def extract_section_number(section_str: str) -> Optional[str]:
    """
    Extract the base section number from a free-form string.

    Useful for parsing values like "IRC §179" or "Section 408A" that
    are stored in training-data metadata fields.

    Returns the first match or *None* if no number is found.
    """
    m = re.search(r"(\d+[A-Za-z]?)", section_str)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Number / amount extraction (shared between reward and DPO generation)
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> set[str]:
    """
    Extract dollar amounts and percentages from *text*.

    Returns normalised strings, e.g. "$25,000", "20%", "10%".
    """
    amounts: set[str] = set()
    # Dollar amounts: $1,000  $1,000.50  $1000
    amounts.update(re.findall(r'\$[\d,]+(?:\.\d+)?', text))
    # Percentages: 20%  10.5%  59½% is unusual but guard against it
    amounts.update(re.findall(r'\d+(?:\.\d+)?%', text))
    return amounts

```

### scripts/evaluate.py

```
#!/usr/bin/env python3
"""
Evaluation Script — Tax Law LLM.

Runs a suite of 25 tax-law questions, scores responses, and compares the
fine-tuned model against the baseline (no adapter).

Scoring:
    - IRC section citation presence (+0.4)
    - Factual keyword coverage       (+0.4)
    - Response length plausibility   (+0.2)

Results written to: outputs/eval_results.json

Usage:
    python scripts/evaluate.py [--adapter-path outputs/grpo/adapters]
    python scripts/evaluate.py --baseline-only   # evaluate base model only
    python scripts/evaluate.py --max-tokens 512
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# [1-G] Import mlx.core for GPU cache clearing between model loads
import mlx.core as mx

# Shared citation utilities (canonical regex)
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from citation_utils import extract_irc_sections  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_MLX = PROJECT_ROOT / "models" / "qwen25-3b-mlx"
MODEL_HF = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct"
EVAL_RESULTS = PROJECT_ROOT / "outputs" / "eval_results.json"

ADAPTER_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "grpo" / "adapters",
    PROJECT_ROOT / "outputs" / "dpo" / "adapters",
    PROJECT_ROOT / "outputs" / "sft" / "adapters",
]


# ---------------------------------------------------------------------------
# Evaluation questions with expected signals
# ---------------------------------------------------------------------------
# Each entry: (question, [expected_irc_sections], [expected_keywords])
EVAL_QUESTIONS: list[tuple[str, list[str], list[str]]] = [
    # Section 61 — Gross income
    (
        "What is included in gross income under IRC Section 61? Give examples.",
        ["61"],
        ["compensation", "wages", "interest", "dividends", "rents", "royalties",
         "gains", "income from whatever source"],
    ),
    # Section 63 — Standard deduction
    (
        "What is the standard deduction and how does it work under IRC Section 63?",
        ["63"],
        ["standard deduction", "itemized", "filing status", "adjusted gross income"],
    ),
    # Section 1 — Capital gains tax rates
    (
        "Explain the capital gains tax rates under IRC Section 1. "
        "What is the rate for long-term capital gains?",
        ["1", "1(h)"],
        ["long-term", "short-term", "0%", "15%", "20%", "holding period"],
    ),
    # Section 162 — Business deductions
    (
        "What ordinary and necessary business expenses are deductible "
        "under IRC Section 162?",
        ["162"],
        ["ordinary", "necessary", "trade or business", "deductible", "expense"],
    ),
    # Section 170 — Charitable contributions
    (
        "What are the rules for deducting charitable contributions under "
        "IRC Section 170? What is the AGI limitation?",
        ["170"],
        ["charitable", "contribution", "501(c)(3)", "60%", "AGI", "deduction limit"],
    ),
    # Section 280A — Home office
    (
        "When can a taxpayer deduct home office expenses under IRC Section 280A? "
        "What is the exclusive use requirement?",
        ["280A"],
        ["exclusive use", "regular basis", "principal place", "home office",
         "trade or business"],
    ),
    # Section 401 — 401(k) qualified plans
    (
        "What are the contribution limits and basic rules for 401(k) plans "
        "under IRC Section 401?",
        ["401", "401(k)"],
        ["elective deferral", "contribution limit", "employer match",
         "vesting", "qualified plan"],
    ),
    # Section 408 — IRAs
    (
        "What is the difference between a traditional IRA and a Roth IRA "
        "under IRC Sections 408 and 408A?",
        ["408", "408A"],
        ["traditional IRA", "Roth IRA", "deductible", "tax-free", "contribution limit",
         "income limit", "distribution"],
    ),
    # Section 501 — Tax-exempt organizations
    (
        "What types of organizations qualify for tax exemption under "
        "IRC Section 501(c)(3)? What is the inurement prohibition?",
        ["501", "501(c)(3)"],
        ["charitable", "religious", "educational", "scientific", "inurement",
         "private benefit", "public charity"],
    ),
    # Section 1031 — Like-kind exchanges
    (
        "How does a like-kind exchange work under IRC Section 1031? "
        "What property qualifies?",
        ["1031"],
        ["like-kind", "real property", "boot", "exchange", "defer", "gain",
         "qualified intermediary"],
    ),
    # Section 179 — Expensing election
    (
        "What is the Section 179 expensing election and what are its limits?",
        ["179"],
        ["expensing", "first-year", "deduction", "phase-out", "business use",
         "tangible personal property"],
    ),
    # Section 168 — MACRS / bonus depreciation
    (
        "How does bonus depreciation work under IRC Section 168(k)? "
        "What is the phase-down schedule?",
        ["168", "168(k)"],
        ["bonus depreciation", "first year", "MACRS", "placed in service",
         "phase-down", "100%", "80%"],
    ),
    # Section 267 — Related party losses
    (
        "What are the restrictions on deducting losses between related parties "
        "under IRC Section 267?",
        ["267"],
        ["related party", "loss disallowance", "constructive ownership",
         "family member", "controlled", "deferral"],
    ),
    # Section 469 — Passive activity losses
    (
        "What are the passive activity loss rules under IRC Section 469? "
        "What is the $25,000 rental exception?",
        ["469"],
        ["passive activity", "material participation", "rental", "$25,000",
         "active participation", "suspended losses"],
    ),
    # Section 121 — Home sale exclusion
    (
        "How does the home sale exclusion work under IRC Section 121? "
        "What is the dollar limit for a married couple?",
        ["121"],
        ["exclusion", "principal residence", "$250,000", "$500,000", "married",
         "2 out of 5 years", "ownership", "use"],
    ),
    # Section 1014 — Step-up in basis
    (
        "Explain the step-up in basis at death under IRC Section 1014. "
        "How does it affect inherited property?",
        ["1014"],
        ["step-up", "fair market value", "date of death", "inherited",
         "basis", "capital gains"],
    ),
    # Section 2503 — Annual gift exclusion
    (
        "What is the annual gift tax exclusion under IRC Section 2503? "
        "How much can a person give tax-free per year per recipient?",
        ["2503", "2501"],
        ["annual exclusion", "$18,000", "$17,000", "gift tax", "per recipient",
         "present interest"],
    ),
    # Section 6662 — Accuracy penalties
    (
        "What penalties apply for substantial understatements of tax under "
        "IRC Section 6662?",
        ["6662"],
        ["substantial understatement", "20%", "accuracy-related", "negligence",
         "reasonable cause", "substantial authority"],
    ),
    # Section 72 — Annuities / early withdrawal
    (
        "What is the 10% early withdrawal penalty for retirement accounts "
        "under IRC Section 72(t)? What are the exceptions?",
        ["72", "72(t)"],
        ["10%", "early withdrawal", "59½", "exception", "substantially equal",
         "disability", "death"],
    ),
    # Section 199A — QBI deduction
    (
        "How does the qualified business income deduction work under "
        "IRC Section 199A for pass-through entities?",
        ["199A"],
        ["qualified business income", "20%", "pass-through", "W-2 wages",
         "specified service", "SSTB", "threshold"],
    ),
    # Section 163 — Interest deduction
    (
        "What types of interest are deductible under IRC Section 163? "
        "What are the limitations on investment interest and mortgage interest?",
        ["163", "163(h)"],
        ["mortgage interest", "qualified residence", "investment interest",
         "business interest", "limitation", "deductible"],
    ),
    # Section 104 — Damages exclusion
    (
        "Are personal injury lawsuit damages taxable? What does IRC Section 104 say?",
        ["104"],
        ["personal physical injury", "physical sickness", "excludable",
         "damages", "compensatory", "punitive", "taxable"],
    ),
    # Section 2056 — Marital deduction
    (
        "What is the unlimited marital deduction for estate tax purposes "
        "under IRC Section 2056?",
        ["2056", "2001"],
        ["marital deduction", "unlimited", "U.S. citizen", "surviving spouse",
         "estate tax", "QTIP", "qualified terminable interest"],
    ),
    # Section 1221 — Capital asset definition
    (
        "What is a capital asset under IRC Section 1221? "
        "What property is excluded from capital asset treatment?",
        ["1221"],
        ["capital asset", "inventory", "accounts receivable", "depreciable property",
         "real property used in trade", "copyrights", "exclusion"],
    ),
    # Section 83 — Property transferred for services
    (
        "How are restricted stock units (RSUs) and stock options taxed under "
        "IRC Section 83? What is the Section 83(b) election?",
        ["83", "83(b)"],
        ["substantial risk of forfeiture", "83(b) election", "vesting",
         "fair market value", "ordinary income", "RSU", "stock option"],
    ),
]

# Total: 25 questions


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_response(
    response: str,
    expected_sections: list[str],
    expected_keywords: list[str],
) -> dict[str, Any]:
    """
    Return a score dict with component scores and overall [0, 1].

    Components:
        citation_score  (0 or 0.4)   — any expected section cited
        keyword_score   (0–0.4)      — fraction of keywords present
        length_score    (0 or 0.2)   — response is 50–2000 chars
    """
    response_lower = response.lower()

    # Citation score — use canonical IRC citation regex from citation_utils.
    # Fix LOW: expected_sections may include subsection qualifiers like "1(h)"
    # or "168(k)", but extract_irc_sections() returns only the base number
    # (e.g. "1", "168").  Strip the subsection before comparing so that
    # "Section 179(b)(1)" correctly matches expected entry "179".
    # Use re.search (not re.match) so that strings starting with "§" or
    # other non-digit prefixes are still handled correctly.
    cited_in_response = extract_irc_sections(response)
    def _base_section(s: str) -> str:
        """Return the numeric base of a section string, stripping subsections."""
        m = re.search(r"(\d+[A-Za-z]?)", s)
        return m.group(1) if m else s
    cited = any(_base_section(sec) in cited_in_response for sec in expected_sections)
    citation_score = 0.4 if cited else 0.0

    # Keyword coverage
    matched_keywords = [
        kw for kw in expected_keywords
        if kw.lower() in response_lower
    ]
    kw_fraction = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0.0
    keyword_score = round(kw_fraction * 0.4, 4)

    # Length score
    length = len(response.strip())
    length_score = 0.2 if 50 <= length <= 2000 else 0.0

    overall = citation_score + keyword_score + length_score

    return {
        "citation_score": citation_score,
        "keyword_score": keyword_score,
        "length_score": length_score,
        "overall": round(overall, 4),
        "cited_sections": cited,
        "matched_keywords": matched_keywords,
        "response_length": length,
    }


# ---------------------------------------------------------------------------
# Model loading and generation
# ---------------------------------------------------------------------------

def resolve_model_path() -> Path:
    if MODEL_MLX.exists() and (MODEL_MLX / "config.json").exists():
        return MODEL_MLX
    if MODEL_HF.exists() and (MODEL_HF / "config.json").exists():
        return MODEL_HF
    print(f"ERROR: No model found at {MODEL_MLX} or {MODEL_HF}")
    sys.exit(1)


def resolve_adapter(override: str | None) -> Path | None:
    if override:
        p = Path(override)
        if not (p / "adapter_config.json").exists():
            print(f"WARNING: No adapter_config.json at {p}. Evaluating base model.")
            return None
        return p
    for candidate in ADAPTER_CANDIDATES:
        if (candidate / "adapter_config.json").exists():
            return candidate
    return None


def load_model(model_path: Path, adapter_path: Path | None):
    from mlx_lm import load as mlx_load

    if adapter_path:
        model, tokenizer = mlx_load(str(model_path), adapter_path=str(adapter_path))
    else:
        model, tokenizer = mlx_load(str(model_path))
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_tokens: int,
    temperature: float = 0.3,
) -> str:
    from mlx_lm import generate

    # Format as chat using the model's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful tax law assistant. Answer questions about "
                    "US federal tax law accurately, citing relevant IRC sections."
                ),
            },
            {"role": "user", "content": question},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = question
    else:
        prompt = question

    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temperature)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    # Strip echoed prompt if present
    if response.startswith(prompt):
        response = response[len(prompt):]
    return response.strip()


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    label: str,
    max_tokens: int,
    temperature: float = 0.3,  # [7-B] explicit temperature for fair A/B comparison
) -> list[dict]:
    results = []
    total = len(EVAL_QUESTIONS)
    print(f"\nEvaluating: {label} ({total} questions, temp={temperature})")
    print("-" * 60)

    for idx, (question, sections, keywords) in enumerate(EVAL_QUESTIONS, 1):
        print(f"  [{idx:2d}/{total}] {question[:70]}...", end=" ", flush=True)
        t0 = time.time()
        response = generate_answer(model, tokenizer, question, max_tokens, temperature)
        elapsed = time.time() - t0
        score = score_response(response, sections, keywords)
        print(f"score={score['overall']:.2f} ({elapsed:.1f}s)")
        results.append({
            "idx": idx,
            "question": question,
            "expected_sections": sections,
            "response": response,
            "score": score,
            "elapsed_s": round(elapsed, 2),
        })
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def summarise(results: list[dict]) -> dict:
    overall_scores = [r["score"]["overall"] for r in results]
    citation_scores = [r["score"]["citation_score"] for r in results]
    keyword_scores = [r["score"]["keyword_score"] for r in results]
    n = len(results)
    return {
        "n_questions": n,
        "mean_overall": round(sum(overall_scores) / n, 4),
        "mean_citation": round(sum(citation_scores) / n, 4),
        "mean_keyword": round(sum(keyword_scores) / n, 4),
        "pct_cited_section": round(
            sum(1 for s in citation_scores if s > 0) / n * 100, 1
        ),
    }


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{label} Summary:")
    print(f"  Mean overall score:  {summary['mean_overall']:.3f} / 1.000")
    print(f"  Mean citation score: {summary['mean_citation']:.3f} / 0.400")
    print(f"  Mean keyword score:  {summary['mean_keyword']:.3f} / 0.400")
    print(f"  Questions citing IRC section: {summary['pct_cited_section']:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned tax law LLM")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (default: auto-detect best available)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate only the baseline model (no adapter)",
    )
    parser.add_argument(
        "--finetuned-only",
        action="store_true",
        help="Evaluate only the fine-tuned model (skip baseline)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EVAL_RESULTS),
        help=f"Output JSON path (default: {EVAL_RESULTS})",
    )
    args = parser.parse_args()

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("ERROR: mlx_lm not found. Install with: pip install mlx-lm")
        sys.exit(1)

    model_path = resolve_model_path()
    adapter_path = None if args.baseline_only else resolve_adapter(args.adapter_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "meta": {
            "model_path": str(model_path),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "n_questions": len(EVAL_QUESTIONS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # --- Baseline ---
    if not args.finetuned_only:
        print(f"\nLoading baseline model from {model_path} ...")
        baseline_model, baseline_tokenizer = load_model(model_path, adapter_path=None)
        baseline_results = evaluate_model(
            baseline_model, baseline_tokenizer, "Baseline (no adapter)",
            args.max_tokens, args.temperature,  # [7-B] use shared temperature
        )
        baseline_summary = summarise(baseline_results)
        print_summary("Baseline", baseline_summary)
        all_results["baseline"] = {
            "summary": baseline_summary,
            "questions": baseline_results,
        }
        del baseline_model  # free memory before loading fine-tuned
        # [1-G] Clear Metal GPU cache so the evicted model is not keeping VRAM
        mx.metal.clear_cache()

    # --- Fine-tuned ---
    if not args.baseline_only and adapter_path is not None:
        print(f"\nLoading fine-tuned model (adapter: {adapter_path}) ...")
        ft_model, ft_tokenizer = load_model(model_path, adapter_path)
        ft_results = evaluate_model(
            ft_model, ft_tokenizer, f"Fine-tuned ({adapter_path.parent.name})",
            args.max_tokens, args.temperature,  # [7-B] same temperature as baseline
        )
        ft_summary = summarise(ft_results)
        print_summary("Fine-tuned", ft_summary)
        all_results["finetuned"] = {
            "adapter_path": str(adapter_path),
            "summary": ft_summary,
            "questions": ft_results,
        }

        # Delta comparison
        if "baseline" in all_results:
            delta = round(
                ft_summary["mean_overall"] - baseline_summary["mean_overall"], 4
            )
            all_results["delta_overall"] = delta
            print(f"\nImprovement over baseline: {delta:+.4f}")

    elif not args.baseline_only and adapter_path is None:
        print(
            "\nNo adapter found. Run SFT/DPO/GRPO training first to evaluate "
            "the fine-tuned model."
        )

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

```

### scripts/export_to_ollama.py

```
#!/usr/bin/env python3
"""
Export the final trained model to Ollama.

Pipeline:
    1. Fuse LoRA adapters into the base model  (mlx_lm.fuse)
    2. Convert fused model to GGUF             (llama.cpp convert_hf_to_gguf.py)
    3. Quantize to Q4_K_M                     (llama-quantize or llama.cpp)
    4. Write Ollama Modelfile
    5. Import into Ollama                      (ollama create)

Usage:
    python scripts/export_to_ollama.py [--adapter-path outputs/grpo/adapters] [--dry-run]
    python scripts/export_to_ollama.py --skip-gguf   # skip if GGUF already done

Prerequisites:
    pip install mlx-lm
    brew install llama.cpp   (or build from source for convert/quantize)
    ollama must be installed and running
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_MLX = PROJECT_ROOT / "models" / "qwen25-3b-mlx"
MODEL_HF = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct"

# Priority order for adapter selection
ADAPTER_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "grpo" / "adapters",
    PROJECT_ROOT / "outputs" / "dpo" / "adapters",
    PROJECT_ROOT / "outputs" / "sft" / "adapters",
]

FUSED_PATH = PROJECT_ROOT / "outputs" / "final" / "fused"
# [6-A] Two-step quantization: bf16 GGUF first, then q6_k (q8_0 cannot be re-quantized by llama-quantize)
GGUF_PATH = PROJECT_ROOT / "outputs" / "final" / "model-bf16.gguf"
GGUF_Q6_PATH = PROJECT_ROOT / "outputs" / "final" / "model-q6_k.gguf"
GGUF_Q4_PATH = PROJECT_ROOT / "outputs" / "final" / "model-q4_k_m.gguf"  # legacy fallback
MODELFILE_PATH = PROJECT_ROOT / "outputs" / "final" / "Modelfile"
OLLAMA_MODEL_NAME = "qwen25-tax-3b"

SYSTEM_PROMPT = """\
You are a tax law assistant trained on the Internal Revenue Code (Title 26) \
and Treasury Regulations (26 CFR). You answer questions about US federal tax law \
accurately, cite relevant IRC sections, and note important exceptions and limitations. \
You do not provide personalised tax advice; always recommend consulting a qualified \
tax professional for individual situations.\
"""


# ---------------------------------------------------------------------------
# Step 1: Fuse adapters
# ---------------------------------------------------------------------------

def resolve_adapter(override: str | None) -> Path | None:
    if override:
        p = Path(override)
        if not p.exists():
            print(f"ERROR: Specified adapter path does not exist: {p}")
            sys.exit(1)
        return p
    for candidate in ADAPTER_CANDIDATES:
        if (candidate / "adapter_config.json").exists():
            print(f"Using adapter: {candidate}")
            return candidate
    print(
        "WARNING: No trained adapter found. "
        "Export will fuse the base model without any fine-tuning."
    )
    return None


def resolve_base_model(override: str | None = None) -> Path:
    if override:
        p = Path(override)
        if p.exists() and (p / "config.json").exists():
            return p
        print(f"ERROR: Base model not found at {p}")
        sys.exit(1)
    if MODEL_MLX.exists() and (MODEL_MLX / "config.json").exists():
        return MODEL_MLX
    if MODEL_HF.exists() and (MODEL_HF / "config.json").exists():
        return MODEL_HF
    print(f"ERROR: Base model not found at {MODEL_MLX} or {MODEL_HF}")
    sys.exit(1)


def fuse_adapters(model_path: Path, adapter_path: Path | None, dry_run: bool) -> Path:
    """Merge LoRA adapter weights into the base model via mlx_lm.fuse."""
    FUSED_PATH.mkdir(parents=True, exist_ok=True)

    if adapter_path is None:
        # No adapter — just copy/symlink the base model
        print("No adapter to fuse. Copying base model to fused path ...")
        if not dry_run:
            if FUSED_PATH.exists():
                shutil.rmtree(FUSED_PATH)
            shutil.copytree(str(model_path), str(FUSED_PATH))
        return FUSED_PATH

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", str(model_path),
        "--adapter-path", str(adapter_path),
        "--save-path", str(FUSED_PATH),
        "--dequantize",   # export as bf16 (no quantization) for best GGUF quality
    ]
    print("\nStep 1: Fusing LoRA adapters")
    print(" ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            print("ERROR: mlx_lm.fuse failed.")
            sys.exit(1)
        print(f"Fused model saved to: {FUSED_PATH}")
    return FUSED_PATH


# ---------------------------------------------------------------------------
# Step 2: Convert to GGUF
# ---------------------------------------------------------------------------

def find_llama_cpp_convert() -> Path | None:
    """Find llama.cpp's convert_hf_to_gguf.py script."""
    candidates = [
        # Homebrew (bin/ — newer llama.cpp formula installs here directly)
        Path("/opt/homebrew/bin/convert_hf_to_gguf.py"),
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
        # Homebrew (share/)
        Path("/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py"),
        Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
        # Common build locations
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/tmp/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # Try locating via 'llama-quantize' binary path
    quantize = shutil.which("llama-quantize")
    if quantize:
        bin_dir = Path(quantize).parent
        # Check the same bin/ directory first (e.g. /opt/homebrew/bin/)
        for name in ["convert_hf_to_gguf.py", "convert.py"]:
            p = bin_dir / name
            if p.exists():
                return p
        # Then check one level up (legacy build layout)
        parent = bin_dir.parent
        for name in ["convert_hf_to_gguf.py", "convert.py"]:
            p = parent / name
            if p.exists():
                return p

    return None


def convert_to_gguf(fused_path: Path, dry_run: bool) -> Path:
    """Convert the fused HF model to GGUF format."""
    print("\nStep 2: Converting to GGUF")

    converter = find_llama_cpp_convert()
    if converter is None:
        print(
            "WARNING: llama.cpp convert_hf_to_gguf.py not found.\n"
            "Install with: brew install llama.cpp\n"
            "Or build from source: https://github.com/ggerganov/llama.cpp\n"
            "Skipping GGUF conversion — you must convert manually."
        )
        return GGUF_PATH  # placeholder; won't exist

    # If using a local llama.cpp clone (e.g. /tmp/llama.cpp), inject its
    # bundled gguf-py into PYTHONPATH so the convert script finds the right
    # gguf package version (brew's PyPI gguf 0.18.0 may lag behind the binary).
    env = None
    gguf_py = converter.parent / "gguf-py"
    if gguf_py.exists():
        import os
        env = {**os.environ, "PYTHONPATH": str(gguf_py) + ":" + os.environ.get("PYTHONPATH", "")}
        print(f"Using bundled gguf-py from: {gguf_py}")

    # [6-A] Export as bf16 first; q6_k quantization happens in a separate step
    # (q8_0 cannot be re-quantized by llama-quantize in brew llama.cpp)
    cmd = [
        sys.executable, str(converter),
        str(fused_path),
        "--outfile", str(GGUF_PATH),
        "--outtype", "bf16",
    ]
    print(" ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=False, env=env)
        if result.returncode != 0:
            print("ERROR: GGUF conversion failed. Check llama.cpp version compatibility.")
            sys.exit(1)
        print(f"bf16 GGUF saved to: {GGUF_PATH}")
    return GGUF_PATH


def quantize_gguf(gguf_path: Path, dry_run: bool) -> Path:
    """Quantize the bf16 GGUF to q6_k for Ollama serving.

    [6-A] Two-step process: bf16 GGUF → q6_k via llama-quantize.
    q6_k preserves numeric precision (e.g. $14,600 standard deduction) that
    q4_k_m loses, while remaining ~half the size of bf16.
    """
    print("\nStep 3: Quantizing to q6_k")

    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        print(
            "WARNING: llama-quantize not found.\n"
            "Install with: brew install llama.cpp\n"
            "Skipping quantization — Ollama will use the bf16 GGUF (larger file)."
        )
        return gguf_path  # fall back to bf16

    cmd = [quantize_bin, str(gguf_path), str(GGUF_Q6_PATH), "q6_k"]
    print(" ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("ERROR: q6_k quantization failed. Using bf16 GGUF.")
            return gguf_path
        print(f"q6_k GGUF saved to: {GGUF_Q6_PATH}")
    return GGUF_Q6_PATH


# ---------------------------------------------------------------------------
# Step 3: Write Modelfile
# ---------------------------------------------------------------------------

def write_modelfile(gguf_path: Path, dry_run: bool) -> Path:
    """Write an Ollama Modelfile."""
    print("\nStep 4: Writing Modelfile")
    MODELFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # [6-B] Include all Qwen chat template stop sequences to avoid streaming until EOS
    content = f"""\
FROM {gguf_path}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
"""
    print(f"Modelfile path: {MODELFILE_PATH}")
    if not dry_run:
        MODELFILE_PATH.write_text(content)
        print("Modelfile written.")
    else:
        print("--- Modelfile contents (dry run) ---")
        print(content)
        print("---")
    return MODELFILE_PATH


# ---------------------------------------------------------------------------
# Step 4: Import to Ollama
# ---------------------------------------------------------------------------

def check_ollama() -> bool:
    ollama = shutil.which("ollama")
    if ollama is None:
        print("WARNING: ollama not found in PATH. Install from https://ollama.com")
        return False
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: ollama is installed but not responding. Is the server running?")
        print("Start with: ollama serve")
        return False
    return True


def import_to_ollama(modelfile_path: Path, model_name: str, dry_run: bool) -> None:
    """Create the Ollama model from the Modelfile."""
    print("\nStep 5: Importing to Ollama")

    if not check_ollama():
        print(f"Skipping Ollama import. Run manually:\n  ollama create {model_name} -f {modelfile_path}")
        return

    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    print(" ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("ERROR: ollama create failed.")
            sys.exit(1)
        print(f"\nModel '{model_name}' imported to Ollama successfully.")
        print(f"Test with: ollama run {model_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model to Ollama")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to base model directory (default: auto-detect qwen25-3b-mlx or qwen2.5-3b-instruct)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (default: auto-detect best available)",
    )
    parser.add_argument(
        "--model-name",
        "--name",
        type=str,
        default=OLLAMA_MODEL_NAME,
        help=f"Ollama model name (default: {OLLAMA_MODEL_NAME})",
    )
    parser.add_argument(
        "--skip-fuse",
        action="store_true",
        help="Skip fusion step (use existing fused model at outputs/final/fused)",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip GGUF conversion (use existing GGUF at outputs/final/)",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip q6_k quantization (use bf16 GGUF directly)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override base output directory (default: outputs/final/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    # [6-A] GGUF_Q6_PATH added to globals for two-step bf16 -> q6_k pipeline
    global FUSED_PATH, GGUF_PATH, GGUF_Q6_PATH, GGUF_Q4_PATH, MODELFILE_PATH

    model_name = args.model_name

    # Override output paths if specified
    if args.output_dir:
        out = Path(args.output_dir)
        FUSED_PATH = out / "fused"
        GGUF_PATH = out / "model-bf16.gguf"
        GGUF_Q6_PATH = out / "model-q6_k.gguf"
        GGUF_Q4_PATH = out / "model-q4_k_m.gguf"  # legacy fallback
        MODELFILE_PATH = out / "Modelfile"

    base_model = resolve_base_model(args.model)
    adapter_path = resolve_adapter(args.adapter_path)

    print("\n" + "=" * 70)
    print("EXPORT TO OLLAMA")
    print(f"  base model:    {base_model}")
    print(f"  adapter:       {adapter_path or 'none'}")
    print(f"  fused output:  {FUSED_PATH}")
    print(f"  gguf output:   {GGUF_Q6_PATH}")
    print(f"  ollama name:   {model_name}")
    print("=" * 70 + "\n")

    # Step 1
    if args.skip_fuse and FUSED_PATH.exists():
        print("Skipping fusion — using existing fused model.")
        fused = FUSED_PATH
    else:
        fused = fuse_adapters(base_model, adapter_path, args.dry_run)

    # Step 2
    if args.skip_gguf and GGUF_PATH.exists():
        print("Skipping GGUF conversion — using existing GGUF.")
        gguf = GGUF_PATH
    else:
        gguf = convert_to_gguf(fused, args.dry_run)

    # Step 3
    if args.skip_quantize:
        final_gguf = gguf
    else:
        final_gguf = quantize_gguf(gguf, args.dry_run)

    # Step 4
    modelfile = write_modelfile(final_gguf, args.dry_run)

    # Step 5
    import_to_ollama(modelfile, model_name, args.dry_run)

    print("\nExport pipeline complete.")
    if not args.dry_run:
        print(f"Test your model: ollama run {model_name}")


if __name__ == "__main__":
    main()

```

### scripts/assemble_v5_dataset.py

```
#!/usr/bin/env python3
"""
Assemble v5 training dataset from all available SFT and DPO sources.

Applies inflation upsampling, deduplication, train/valid splits,
and outputs GRPO prompts alongside SFT/DPO splits.
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

SFT_SOURCES = [
    {
        "path": "data/processed/grounded_sft_full.jsonl",
        "name": "grounded_sft_full",
        "description": "IRC grounded",
        "expected_count": 16909,
        "is_inflation": False,
    },
    {
        "path": "data/processed/grounded_cfr_sft_deduped.jsonl",
        "name": "grounded_cfr_sft_deduped",
        "description": "CFR grounded",
        "expected_count": 45855,
        "is_inflation": False,
    },
    {
        "path": "data/processed/bulk_sft_full.jsonl",
        "name": "bulk_sft_full",
        "description": "Tavily bulk batch",
        "expected_count": 15398,
        "is_inflation": False,
    },
    {
        "path": "data/processed/tavily_sft_full.jsonl",
        "name": "tavily_sft_full",
        "description": "Tavily original",
        "expected_count": 4077,
        "is_inflation": False,
    },
    {
        "path": "data/processed/inflation_sft_v2.jsonl",
        "name": "inflation_sft_v2",
        "description": "inflation batch",
        "expected_count": 1359,
        "is_inflation": True,
    },
    {
        "path": "data/processed/inflation_adjusted_sft.jsonl",
        "name": "inflation_adjusted_sft",
        "description": "inflation v1",
        "expected_count": 70,
        "is_inflation": True,
    },
]

DPO_SOURCES = [
    {
        "path": "data/processed/grounded_dpo_full.jsonl",
        "name": "grounded_dpo_full",
        "description": "IRC grounded",
        "expected_count": 1719,
        "is_inflation": False,
    },
    {
        "path": "data/processed/bulk_dpo_full.jsonl",
        "name": "bulk_dpo_full",
        "description": "Tavily bulk batch",
        "expected_count": 7610,
        "is_inflation": False,
    },
    {
        "path": "data/processed/inflation_dpo_v2.jsonl",
        "name": "inflation_dpo_v2",
        "description": "inflation batch",
        "expected_count": 810,
        "is_inflation": True,
    },
    {
        "path": "data/processed/inflation_adjusted_dpo.jsonl",
        "name": "inflation_adjusted_dpo",
        "description": "inflation v1",
        "expected_count": 17,
        "is_inflation": True,
    },
    {
        "path": "data/processed/onpolicy_dpo_v2.jsonl",
        "name": "onpolicy_dpo_v2",
        "description": "on-policy",
        "expected_count": 86,
        "is_inflation": False,
    },
    {
        "path": "data/processed/tavily_dpo_full.jsonl",
        "name": "tavily_dpo_full",
        "description": "Tavily original",
        "expected_count": 56,
        "is_inflation": False,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed line {line_no} in {path}: {e}")
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_sft_user_message(record: dict) -> str | None:
    """Extract the first user message text from an SFT record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def validate_sft(record: dict) -> bool:
    """Validate an SFT record has required structure."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = {m.get("role") for m in messages}
    return "user" in roles and "assistant" in roles


def validate_dpo(record: dict) -> bool:
    """Validate a DPO record has required keys."""
    return all(k in record for k in ("prompt", "chosen", "rejected"))


# ---------------------------------------------------------------------------
# Core assembly logic
# ---------------------------------------------------------------------------

def load_and_validate_sft(sources: list[dict], base_dir: str) -> tuple[dict, list[dict]]:
    """
    Load all SFT sources.
    Returns (source_counts dict, list of (record, source_name, is_inflation)).
    """
    source_counts = {}
    all_records = []  # list of (record, source_name, is_inflation)

    for src in sources:
        path = os.path.join(base_dir, src["path"])
        if not os.path.exists(path):
            print(f"  ERROR: missing file {path}")
            source_counts[src["name"]] = {"loaded": 0, "valid": 0, "missing": True}
            continue

        raw = load_jsonl(path)
        valid = [r for r in raw if validate_sft(r)]
        invalid = len(raw) - len(valid)

        print(
            f"  {src['name']}: loaded {len(raw):,}  valid {len(valid):,}"
            + (f"  ({invalid} invalid skipped)" if invalid else "")
        )

        source_counts[src["name"]] = {
            "loaded": len(raw),
            "valid": len(valid),
            "is_inflation": src["is_inflation"],
            "description": src["description"],
        }
        for r in valid:
            all_records.append((r, src["name"], src["is_inflation"]))

    return source_counts, all_records


def load_and_validate_dpo(sources: list[dict], base_dir: str) -> tuple[dict, list[dict]]:
    """Load all DPO sources."""
    source_counts = {}
    all_records = []

    for src in sources:
        path = os.path.join(base_dir, src["path"])
        if not os.path.exists(path):
            print(f"  ERROR: missing file {path}")
            source_counts[src["name"]] = {"loaded": 0, "valid": 0, "missing": True}
            continue

        raw = load_jsonl(path)
        valid = [r for r in raw if validate_dpo(r)]
        invalid = len(raw) - len(valid)

        print(
            f"  {src['name']}: loaded {len(raw):,}  valid {len(valid):,}"
            + (f"  ({invalid} invalid skipped)" if invalid else "")
        )

        source_counts[src["name"]] = {
            "loaded": len(raw),
            "valid": len(valid),
            "is_inflation": src["is_inflation"],
            "description": src["description"],
        }
        for r in valid:
            all_records.append((r, src["name"], src["is_inflation"]))

    return source_counts, all_records


def deduplicate_sft(records: list[tuple]) -> tuple[list[tuple], dict]:
    """Deduplicate SFT records by user message text. Keep first occurrence."""
    seen = set()
    deduped = []
    dup_count = 0

    for record, source, is_inflation in records:
        key = get_sft_user_message(record)
        if key is None:
            continue
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append((record, source, is_inflation))

    stats = {"before": len(records), "after": len(deduped), "removed": dup_count}
    return deduped, stats


def _dpo_prompt_key(prompt) -> str:
    """Convert a DPO prompt (str or list of messages) to a hashable string key."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        # Concatenate all content values from message dicts
        return "\n".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in prompt
        )
    return str(prompt)


def deduplicate_dpo(records: list[tuple]) -> tuple[list[tuple], dict]:
    """Deduplicate DPO records by (prompt, chosen, rejected) tuple.

    Fix 4-B (review item 4-B MEDIUM): Previously deduped by prompt text only,
    which discarded alternative hard negatives sharing the same prompt but with
    different chosen/rejected pairs.  Now we keep all unique (prompt, chosen,
    rejected) triples so contrastive signal is preserved.
    """
    seen = set()
    deduped = []
    dup_count = 0

    for record, source, is_inflation in records:
        prompt_key = _dpo_prompt_key(record.get("prompt", ""))
        chosen_key = record.get("chosen", "")
        rejected_key = record.get("rejected", "")
        # Deduplicate on the full (prompt, chosen, rejected) triple
        key = (prompt_key, chosen_key, rejected_key)
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append((record, source, is_inflation))

    stats = {"before": len(records), "after": len(deduped), "removed": dup_count}
    return deduped, stats


def apply_inflation_upsampling(records: list[tuple], multiplier: int) -> list[tuple]:
    """
    Duplicate all inflation records `multiplier` times.
    The original copies are already present so we add (multiplier - 1) extra copies.
    """
    upsampled = []
    extra_count = 0

    for record, source, is_inflation in records:
        upsampled.append((record, source, is_inflation))
        if is_inflation:
            for _ in range(multiplier - 1):
                upsampled.append((record, source, is_inflation))
                extra_count += 1

    return upsampled, extra_count


def train_valid_split(records: list[tuple], ratio: float, seed: int) -> tuple[list, list]:
    """Shuffle and split records into train/valid."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def make_grpo_record(sft_record: dict) -> dict:
    """Extract just the user message as a GRPO prompt record."""
    for msg in sft_record.get("messages", []):
        if msg.get("role") == "user":
            return {"messages": [{"role": "user", "content": msg["content"]}]}
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble v5 training dataset from all available SFT and DPO sources."
    )
    parser.add_argument(
        "--output-dir",
        default="data/v5/",
        help="Output directory for assembled dataset (default: data/v5/)",
    )
    parser.add_argument(
        "--inflation-multiplier",
        type=int,
        default=20,
        # Fix 4-C (review item 4-C LOW): 20x is aggressive (inflation records will
        # dominate >35% of tokens).  Monitor class imbalance and reduce if the model
        # over-indexes on inflation scenarios.  Default kept at 20 for compatibility.
        help="Inflation upsampling multiplier (default: 20, NOTE: 20x is aggressive — monitor class balance)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
        help="Train/valid split ratio (default: 0.9)",
    )
    args = parser.parse_args()

    # Resolve paths relative to script location's parent (repo root)
    script_dir = Path(__file__).resolve().parent
    base_dir = str(script_dir.parent)
    output_dir = Path(base_dir) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v5 Dataset Assembly")
    print("=" * 70)
    print(f"  Output dir       : {output_dir}")
    print(f"  Inflation mult   : {args.inflation_multiplier}x")
    print(f"  Random seed      : {args.seed}")
    print(f"  Train/valid split: {args.split_ratio:.0%} / {1-args.split_ratio:.0%}")
    print()

    report = {
        "config": {
            "output_dir": str(output_dir),
            "inflation_multiplier": args.inflation_multiplier,
            "seed": args.seed,
            "split_ratio": args.split_ratio,
        },
        "sft": {},
        "dpo": {},
    }

    # ------------------------------------------------------------------
    # SFT pipeline
    # Fix 1-C (review item 1-C HIGH): Perform train/valid split FIRST on
    # the deduplicated base records, THEN upsample inflation records within
    # each partition separately.  Previously upsampling happened before the
    # split, which allowed copies of the same record to appear in both train
    # and valid sets, making validation non-independent (data leakage).
    # ------------------------------------------------------------------
    print("--- SFT Sources ---")
    sft_source_counts, sft_records = load_and_validate_sft(SFT_SOURCES, base_dir)
    print(f"  Total raw SFT records: {len(sft_records):,}")
    print()

    print("--- SFT Deduplication ---")
    sft_deduped, sft_dedup_stats = deduplicate_sft(sft_records)
    print(
        f"  Before: {sft_dedup_stats['before']:,}  "
        f"After: {sft_dedup_stats['after']:,}  "
        f"Removed: {sft_dedup_stats['removed']:,}"
    )
    print()

    # Split on BASE records first (before upsampling) to prevent leakage
    print("--- SFT Train/Valid Split (on base records, before upsampling) ---")
    sft_train_base, sft_valid_base = train_valid_split(
        sft_deduped, args.split_ratio, args.seed
    )
    print(
        f"  Base train: {len(sft_train_base):,}  "
        f"Base valid: {len(sft_valid_base):,}  "
        f"Ratio: {len(sft_train_base)/len(sft_deduped):.3f}"
    )
    print()

    # Now upsample inflation records WITHIN each partition
    print("--- SFT Inflation Upsampling (within each partition) ---")
    inflation_sft_base = sum(1 for _, _, is_inf in sft_deduped if is_inf)
    sft_train_tuples, sft_train_extra = apply_inflation_upsampling(
        sft_train_base, args.inflation_multiplier
    )
    sft_valid_tuples, sft_valid_extra = apply_inflation_upsampling(
        sft_valid_base, args.inflation_multiplier
    )
    sft_extra = sft_train_extra + sft_valid_extra
    sft_total = len(sft_train_tuples) + len(sft_valid_tuples)
    print(
        f"  Inflation records (base): {inflation_sft_base:,}  "
        f"Extra copies added (total): {sft_extra:,}  "
        f"Total SFT after upsampling: {sft_total:,}"
    )
    print(
        f"  Train: {len(sft_train_tuples):,}  "
        f"Valid: {len(sft_valid_tuples):,}"
    )
    print()

    # Extract just records (drop source/is_inflation metadata for output)
    sft_train = [r for r, _, _ in sft_train_tuples]
    sft_valid = [r for r, _, _ in sft_valid_tuples]
    sft_upsampled_total = sft_total  # used in report below

    # ------------------------------------------------------------------
    # DPO pipeline
    # Fix 1-C (review item 1-C HIGH): Same split-before-upsample fix as SFT.
    # ------------------------------------------------------------------
    print("--- DPO Sources ---")
    dpo_source_counts, dpo_records = load_and_validate_dpo(DPO_SOURCES, base_dir)
    print(f"  Total raw DPO records: {len(dpo_records):,}")
    print()

    print("--- DPO Deduplication ---")
    dpo_deduped, dpo_dedup_stats = deduplicate_dpo(dpo_records)
    print(
        f"  Before: {dpo_dedup_stats['before']:,}  "
        f"After: {dpo_dedup_stats['after']:,}  "
        f"Removed: {dpo_dedup_stats['removed']:,}"
    )
    print()

    # Split on BASE records first (before upsampling) to prevent leakage
    print("--- DPO Train/Valid Split (on base records, before upsampling) ---")
    dpo_train_base, dpo_valid_base = train_valid_split(
        dpo_deduped, args.split_ratio, args.seed
    )
    print(
        f"  Base train: {len(dpo_train_base):,}  "
        f"Base valid: {len(dpo_valid_base):,}  "
        f"Ratio: {len(dpo_train_base)/len(dpo_deduped):.3f}"
    )
    print()

    # Upsample inflation records WITHIN each partition
    print("--- DPO Inflation Upsampling (within each partition) ---")
    inflation_dpo_base = sum(1 for _, _, is_inf in dpo_deduped if is_inf)
    dpo_train_tuples, dpo_train_extra = apply_inflation_upsampling(
        dpo_train_base, args.inflation_multiplier
    )
    dpo_valid_tuples, dpo_valid_extra = apply_inflation_upsampling(
        dpo_valid_base, args.inflation_multiplier
    )
    dpo_extra = dpo_train_extra + dpo_valid_extra
    dpo_total = len(dpo_train_tuples) + len(dpo_valid_tuples)
    print(
        f"  Inflation records (base): {inflation_dpo_base:,}  "
        f"Extra copies added (total): {dpo_extra:,}  "
        f"Total DPO after upsampling: {dpo_total:,}"
    )
    print(
        f"  Train: {len(dpo_train_tuples):,}  "
        f"Valid: {len(dpo_valid_tuples):,}"
    )
    print()

    dpo_train = [r for r, _, _ in dpo_train_tuples]
    dpo_valid = [r for r, _, _ in dpo_valid_tuples]
    dpo_upsampled_total = dpo_total  # used in report below

    # ------------------------------------------------------------------
    # GRPO prompts
    # ------------------------------------------------------------------
    print("--- GRPO Prompts ---")
    grpo_train = [g for r in sft_train if (g := make_grpo_record(r)) is not None]
    grpo_valid = [g for r in sft_valid if (g := make_grpo_record(r)) is not None]
    print(f"  GRPO train: {len(grpo_train):,}  GRPO valid: {len(grpo_valid):,}")
    print()

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    print("--- Writing Output Files ---")
    files = {
        "sft_train.jsonl": sft_train,
        "sft_valid.jsonl": sft_valid,
        "dpo_train.jsonl": dpo_train,
        "dpo_valid.jsonl": dpo_valid,
        "grpo_train.jsonl": grpo_train,
        "grpo_valid.jsonl": grpo_valid,
    }

    for filename, records in files.items():
        out_path = output_dir / filename
        write_jsonl(records, str(out_path))
        print(f"  Wrote {len(records):,} records -> {out_path}")

    # Compatibility copies
    shutil.copy(output_dir / "sft_train.jsonl", output_dir / "train.jsonl")
    shutil.copy(output_dir / "sft_valid.jsonl", output_dir / "valid.jsonl")
    print(f"  Copied sft_train.jsonl -> train.jsonl (compatibility)")
    print(f"  Copied sft_valid.jsonl -> valid.jsonl (compatibility)")
    print()

    # ------------------------------------------------------------------
    # Assembly report
    # ------------------------------------------------------------------
    report["sft"] = {
        "sources": sft_source_counts,
        "deduplication": sft_dedup_stats,
        "inflation_base_records": inflation_sft_base,
        "extra_inflation_copies": sft_extra,
        "total_after_upsampling": sft_upsampled_total,
        "train_count": len(sft_train),
        "valid_count": len(sft_valid),
        "actual_split_ratio": len(sft_train) / sft_upsampled_total,
    }
    report["dpo"] = {
        "sources": dpo_source_counts,
        "deduplication": dpo_dedup_stats,
        "inflation_base_records": inflation_dpo_base,
        "extra_inflation_copies": dpo_extra,
        "total_after_upsampling": dpo_upsampled_total,
        "train_count": len(dpo_train),
        "valid_count": len(dpo_valid),
        "actual_split_ratio": len(dpo_train) / dpo_upsampled_total,
    }
    report["grpo"] = {
        "train_count": len(grpo_train),
        "valid_count": len(grpo_valid),
    }
    report["output_files"] = {
        "sft_train.jsonl": len(sft_train),
        "sft_valid.jsonl": len(sft_valid),
        "dpo_train.jsonl": len(dpo_train),
        "dpo_valid.jsonl": len(dpo_valid),
        "grpo_train.jsonl": len(grpo_train),
        "grpo_valid.jsonl": len(grpo_valid),
        "train.jsonl": len(sft_train),
        "valid.jsonl": len(sft_valid),
    }

    report_path = output_dir / "assembly_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Assembly report -> {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Assembly Complete — Summary")
    print("=" * 70)
    print(f"  SFT total (after dedup + upsample) : {sft_upsampled_total:,}")
    print(f"    train  : {len(sft_train):,}")
    print(f"    valid  : {len(sft_valid):,}")
    print(f"  DPO total (after dedup + upsample) : {dpo_upsampled_total:,}")
    print(f"    train  : {len(dpo_train):,}")
    print(f"    valid  : {len(dpo_valid):,}")
    print(f"  GRPO train : {len(grpo_train):,}")
    print(f"  GRPO valid : {len(grpo_valid):,}")
    print()


if __name__ == "__main__":
    main()

```

### outputs/grpo/adapters/adapter_config.json

```
{
  "num_layers": 16,
  "lora_parameters": {
    "rank": 32,
    "scale": 20.0,
    "dropout": 0.05
  },
  "training": "grpo",
  "group_size": 8,
  "eps_clip": 0.2,
  "step": 300
}

```

### outputs/dpo/adapters/adapter_config.json

```
{
  "num_layers": 16,
  "lora_parameters": {
    "rank": 32,
    "scale": 20.0,
    "dropout": 0.05
  },
  "training": "dpo",
  "beta": 0.5
}

```

### outputs/sft/adapters/adapter_config.json

```
{
    "adapter_path": "/Users/dennisonbertram/Develop/rl-irs-tax-code/outputs/sft/adapters",
    "batch_size": 4,
    "config": "/Users/dennisonbertram/Develop/rl-irs-tax-code/configs/mlx_lora_rank32.yaml",
    "data": "data/v5",
    "fine_tune_type": "lora",
    "grad_accumulation_steps": 1,
    "grad_checkpoint": true,
    "iters": 1500,
    "learning_rate": 1e-05,
    "lora_parameters": {
        "rank": 32,
        "dropout": 0.05,
        "scale": 20.0
    },
    "lr_schedule": null,
    "mask_prompt": false,
    "max_seq_length": 2048,
    "model": "/Users/dennisonbertram/Develop/rl-irs-tax-code/models/qwen25-3b-mlx",
    "num_layers": 16,
    "optimizer": "adam",
    "optimizer_config": {
        "adam": {},
        "adamw": {},
        "muon": {},
        "sgd": {},
        "adafactor": {}
    },
    "project_name": null,
    "report_to": null,
    "resume_adapter_file": null,
    "save_every": 200,
    "seed": 0,
    "steps_per_eval": 100,
    "steps_per_report": 10,
    "test": false,
    "test_batches": 500,
    "train": true,
    "val_batches": 25
}
```

### outputs/final/Modelfile

```
FROM /Users/dennisonbertram/Develop/rl-irs-tax-code/outputs/final/model-q8.gguf

SYSTEM """You are a tax law assistant trained on the Internal Revenue Code (Title 26) and Treasury Regulations (26 CFR). You answer questions about US federal tax law accurately, cite relevant IRC sections, and note important exceptions and limitations. You do not provide personalised tax advice; always recommend consulting a qualified tax professional for individual situations."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

```
