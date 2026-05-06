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
DPO_DATA = PROJECT_ROOT / "data" / "v5" / "dpo_train.jsonl"
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

    Handles multiple MLX API variants defensively:
    - Modern MLX: tree_flatten(params) returns list[(str, array)]  (current)
    - Future/alternate: may be renamed to tree_flatten_items
    - Hypothetical legacy: might return tuple(leaves, treedef) — we guard
      against this even though it has not been observed in any released MLX
      version, to satisfy belt-and-suspenders defensive coding.
    """
    import mlx.utils as mu
    fn = getattr(mu, "tree_flatten_items", None) or getattr(mu, "tree_flatten")
    result = fn(model.parameters())

    # If result is a bare 2-tuple (leaves, treedef) rather than a list of
    # (k,v) pairs, extract just the leaves list.  This guard handles any
    # hypothetical MLX version that changed the return convention.
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], type)
    ):
        result = result[0]

    # Validate that we have a sequence of 2-tuples
    if not isinstance(result, (list, tuple)):
        raise RuntimeError(
            f"_get_model_items: unexpected return type {type(result).__name__} "
            "from tree_flatten. Check MLX version compatibility."
        )
    if result and not isinstance(result[0], (list, tuple)):
        raise RuntimeError(
            f"_get_model_items: expected (key, value) pairs but got "
            f"{type(result[0]).__name__}. Check MLX version compatibility."
        )
    return list(result)


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
        """Apply LoRA only if the model does not already contain LoRALinear layers.

        Uses isinstance() on actual sub-modules (via model.modules()) rather
        than attribute checks on top-level transformer blocks.  LoRA replaces
        Linear sub-modules nested inside attention/MLP blocks, so attribute-
        based checks on the outer layer are not reliable.
        """
        from mlx_lm.tuner.lora import LoRALinear
        already_lora = any(
            isinstance(m, LoRALinear)
            for m in mdl.modules()
        )
        if already_lora:
            print("LoRA layers already present — skipping re-initialisation.")
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
