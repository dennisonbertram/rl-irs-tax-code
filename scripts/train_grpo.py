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
GRPO_DATA = PROJECT_ROOT / "data" / "v5" / "grpo_train.jsonl"
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
        and isinstance(result[1], type)  # treedef is a type/class in JAX convention
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
    # Note: mx.ones_like() does not accept dtype kwarg in this MLX version;
    # use mx.ones(shape, dtype=...) instead.
    mask = mx.ones(shift_labels.shape, dtype=mx.float32)  # (1, T-1)

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

    def _apply_lora_if_needed(mdl, num_layers, config):
        """Apply LoRA only if the model does not already contain LoRALinear layers.

        Uses isinstance() on actual sub-modules (via model.modules()) rather
        than attribute checks on top-level transformer blocks.  LoRA replaces
        Linear sub-modules nested inside attention/MLP blocks, so attribute-
        based checks on the outer layer are not reliable.
        """
        from mlx_lm.tuner.lora import LoRALinear
        # Walk all sub-modules; stop as soon as we find one LoRALinear
        already_lora = any(
            isinstance(m, LoRALinear)
            for m in mdl.modules()
        )
        if already_lora:
            print("LoRA layers already present — skipping re-initialisation.")
            return
        linear_to_lora_layers(mdl, num_layers, config)

    # Apply LoRA to policy BEFORE loading adapter weights.
    # linear_to_lora_layers must run first so that LoRA parameter keys
    # (lora_A, lora_B, etc.) exist in the model before load_weights tries
    # to populate them. Loading weights before this call silently discards
    # all LoRA keys because the layers don't exist yet (strict=False).
    _apply_lora_if_needed(policy_model, lora_num_layers, lora_config)

    if start_adapter is not None:
        print(f"Initializing policy LoRA from adapter: {start_adapter}")
        policy_model.load_weights(str(start_adapter / "adapters.safetensors"), strict=False)

    policy_model.train()

    # Frozen reference model
    print("Loading reference model ...")
    ref_model, _ = load(str(model_path))
    # Apply LoRA to reference model first, then load adapter weights.
    _apply_lora_if_needed(ref_model, lora_num_layers, lora_config)
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

    # Define the loss function that takes per-step data as arguments so that
    # nn.value_and_grad can be compiled ONCE outside the loop and reused.
    # Fix MEDIUM: previously loss_fn was a new closure every step, which forced
    # value_and_grad to JIT-recompile each iteration (~0.3-0.5 s overhead).
    # Fix CRITICAL (from v2): loss_fn accepts (prompt, completions, rewards) as
    # explicit args, NOT a model arg — value_and_grad differentiates wrt
    # policy_model's trainable parameters; model is NOT passed as a call arg.
    def loss_fn(prompt_arg, completions_arg, rewards_arg):
        return grpo_loss_for_prompt(
            policy_model, ref_model, tokenizer,
            prompt_arg, completions_arg, rewards_arg, args,
        )

    # Compile value_and_grad once before the loop
    loss_and_grad = nn.value_and_grad(policy_model, loss_fn)

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

            # Compute loss and gradients using the pre-compiled value_and_grad fn
            loss, grads = loss_and_grad(prompt, completions, rewards)
            # Gradient clipping to match DPO and prevent NaN on long completions
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
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
