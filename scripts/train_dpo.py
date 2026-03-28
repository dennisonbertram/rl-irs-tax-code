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
    "beta": 0.1,               # KL penalty coefficient
    "max_seq_length": 1024,
    "save_every": 100,
    "log_every": 10,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def save_lora_weights(model, path: str) -> None:
    """Save only LoRA adapter weights, not the full model."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    all_params = tree_flatten(model.parameters())
    lora_params = {k: v for k, v in all_params if "lora" in k}
    if not lora_params:
        print("WARNING: No LoRA parameters found, saving all trainable params")
        lora_params = dict(tree_flatten(model.trainable_parameters()))
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
) -> Iterator[dict]:
    """
    Yield batches of tokenized (prompt+chosen, prompt+rejected) pairs.

    Each batch is a dict with keys:
        chosen_ids:   (B, T_c) int32
        rejected_ids: (B, T_r) int32
        chosen_mask:  (B, T_c) float32
        rejected_mask:(B, T_r) float32
    """
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(DEFAULTS["seed"])
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

    chosen_seqs = [
        encode(r["prompt"] + r["chosen"]) for r in records
    ]
    rejected_seqs = [
        encode(r["prompt"] + r["rejected"]) for r in records
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
    Compute sum of per-token log-probabilities for each sequence in the batch.
    Returns shape (B,).
    """
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(input_ids).astype(mx.float32)  # (B, T, V) - ensure float32
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_labels = input_ids[:, 1:]    # (B, T-1)
    shift_mask = mask[:, 1:]           # (B, T-1)

    log_probs = nn.log_softmax(shift_logits, axis=-1)  # (B, T-1, V)

    # Gather log probs of actual tokens
    B, T = shift_labels.shape
    token_log_probs = log_probs[mx.arange(B)[:, None], mx.arange(T)[None, :], shift_labels]
    # (B, T-1)

    # Mask padding and sum
    return (token_log_probs * shift_mask).sum(axis=-1)  # (B,)


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
    # Clamp log-ratio to prevent NaN from overflow at low precision (4-bit models)
    log_ratio = mx.clip(log_ratio, -10.0, 10.0)
    rewards = beta * log_ratio
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

    # Write adapter config so downstream scripts can detect this adapter
    adapter_config = {
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank,
        "scale": 1.0,
        "dropout": 0.0,
        "training": "dpo",
        "beta": args.beta,
    }
    with open(dpo_adapter / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"\nLoading model from {model_path} ...")
    policy_model, tokenizer = load(str(model_path))

    # Read SFT adapter config to match LoRA settings
    sft_adapter_config_file = sft_adapter / "adapter_config.json"
    sft_has_adapter = sft_adapter_config_file.exists() or (sft_adapter / "adapters.safetensors").exists()

    # Determine LoRA config from SFT adapter if available
    sft_num_layers = args.lora_layers
    sft_lora_rank = args.lora_rank
    sft_lora_scale = 1.0
    if sft_adapter_config_file.exists():
        with open(sft_adapter_config_file) as f:
            sft_cfg = json.load(f)
        # mlx_lm style config
        if "lora_parameters" in sft_cfg:
            sft_lora_rank = sft_cfg["lora_parameters"].get("rank", args.lora_rank)
            sft_lora_scale = sft_cfg["lora_parameters"].get("scale", 1.0)
        if "num_layers" in sft_cfg:
            sft_num_layers = sft_cfg["num_layers"]
        print(f"SFT adapter config: rank={sft_lora_rank}, layers={sft_num_layers}, scale={sft_lora_scale}")

    # Apply LoRA FIRST, then load SFT weights (LoRA keys must exist before loading)
    linear_to_lora_layers(policy_model, sft_num_layers, {"rank": sft_lora_rank, "scale": sft_lora_scale, "dropout": 0.0})

    if sft_has_adapter:
        print(f"Loading SFT adapter from {sft_adapter} ...")
        policy_model.load_weights(str(sft_adapter / "adapters.safetensors"), strict=False)

    policy_model.train()

    # Reference model is a frozen copy (base + SFT adapter)
    print("Loading reference model ...")
    ref_model, _ = load(str(model_path))
    linear_to_lora_layers(ref_model, sft_num_layers, {"rank": sft_lora_rank, "scale": sft_lora_scale, "dropout": 0.0})
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
        data_iter = batch_iterator(records, args.batch_size, tokenizer, args.max_seq_length)

        while step < args.iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reshuffle and restart
                data_iter = batch_iterator(
                    records, args.batch_size, tokenizer, args.max_seq_length
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
