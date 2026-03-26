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
LOG_FILE = PROJECT_ROOT / "outputs" / "grpo" / "train.log"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = {
    "iters": 300,
    "group_size": 4,           # K completions per prompt
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
    if MODEL_MLX.exists() and (MODEL_MLX / "config.json").exists():
        return MODEL_MLX
    if MODEL_HF.exists() and (MODEL_HF / "config.json").exists():
        print(f"Using HF model (mlx_lm will convert): {MODEL_HF}")
        return MODEL_HF
    print(f"ERROR: No model found at {MODEL_MLX} or {MODEL_HF}")
    sys.exit(1)


def resolve_start_adapter() -> Path | None:
    """Return the best available starting adapter (DPO > SFT > None)."""
    for adapter_dir in [DPO_ADAPTER, SFT_ADAPTER]:
        if (adapter_dir / "adapter_config.json").exists():
            print(f"Starting from adapter: {adapter_dir}")
            return adapter_dir
    print("WARNING: No prior adapter found. Training from base model.")
    return None


def check_data() -> None:
    if not GRPO_DATA.exists():
        print(f"ERROR: GRPO data not found at {GRPO_DATA}")
        print("Run the data pipeline to generate GRPO prompts.")
        sys.exit(1)
    with open(GRPO_DATA) as f:
        first = json.loads(f.readline())
    if "prompt" not in first:
        print(f"ERROR: grpo.jsonl records must have a 'prompt' key. Got: {list(first.keys())}")
        sys.exit(1)
    print(f"GRPO data OK: {GRPO_DATA}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prompts(path: Path) -> list[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                prompts.append(rec["prompt"])
    return prompts


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
    Compute the sum of per-token log-probs for a sequence.
    Returns a scalar mlx array.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    ids = tokenizer.encode(text)[:max_seq_length]
    ids_arr = mx.array(np.array([ids], dtype=np.int32))  # (1, T)

    logits = model(ids_arr)             # (1, T, V)
    shift_logits = logits[:, :-1, :]    # (1, T-1, V)
    shift_labels = ids_arr[:, 1:]       # (1, T-1)

    log_probs = nn.log_softmax(shift_logits, axis=-1)
    T = shift_labels.shape[1]
    token_lp = log_probs[0, mx.arange(T), shift_labels[0]]  # (T-1,)
    return token_lp.sum()


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
        full_text = prompt + completion
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

    GRPO_ADAPTER.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading policy model from {model_path} ...")
    policy_model, tokenizer = load(str(model_path))

    start_adapter = resolve_start_adapter()
    if start_adapter is not None:
        policy_model.load_weights(str(start_adapter / "adapters.safetensors"), strict=False)

    # Apply LoRA to policy
    linear_to_lora_layers(policy_model, args.lora_layers, {"rank": args.lora_rank, "scale": 1.0, "dropout": 0.0})
    policy_model.train()

    # Frozen reference model
    print("Loading reference model ...")
    ref_model, _ = load(str(model_path))
    if start_adapter is not None:
        ref_model.load_weights(str(start_adapter / "adapters.safetensors"), strict=False)
    ref_model.eval()
    ref_model.freeze()

    prompts = load_prompts(GRPO_DATA)
    print(f"Loaded {len(prompts)} GRPO prompts.")

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    import numpy as np
    rng = np.random.default_rng(args.seed)

    start_time = time.time()
    best_avg_reward = -float("inf")

    with open(LOG_FILE, "w") as log:
        log.write(f"GRPO Training — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(
            f"group_size={args.group_size}, lr={args.learning_rate}, "
            f"iters={args.iters}, eps_clip={args.epsilon_clip}\n\n"
        )

        for step in range(1, args.iters + 1):
            # Sample a prompt
            prompt = prompts[rng.integers(len(prompts))]

            # Generate K completions (no grad)
            policy_model.eval()
            completions = generate_completions(
                policy_model, tokenizer, prompt,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            policy_model.train()

            # Score completions
            rewards = [compute_reward(prompt, c) for c in completions]

            # Compute loss
            def loss_fn(model):
                return grpo_loss_for_prompt(
                    model, ref_model, tokenizer,
                    prompt, completions, rewards, args,
                )

            loss, grads = nn.value_and_grad(policy_model, loss_fn)(policy_model)
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
                save_lora_weights(policy_model, str(GRPO_ADAPTER / "adapters.safetensors"))
                print(f"  Saved adapter checkpoint at step {step}")
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_lora_weights(policy_model, str(GRPO_ADAPTER / "adapters_best.safetensors"))

    total = time.time() - start_time
    print(f"\nGRPO training complete in {total:.1f}s")
    print(f"Best average reward: {best_avg_reward:.3f}")
    print(f"Final adapter saved to: {GRPO_ADAPTER}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training via MLX")
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

    check_dependencies()
    model_path = resolve_model_path()
    check_data()

    if args.dry_run:
        print("\nDRY RUN — all checks passed. Training not started.")
        return

    train(args, model_path)


if __name__ == "__main__":
    main()
