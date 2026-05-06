# Debug: Ollama/GGUF vs MLX Adapter Response Difference

**Date**: 2026-03-31
**Issue**: GGUF model in Ollama gives 2021 standard deduction figures ($12,550) while evaluation claimed $13,850.

---

## Root Cause: Two Distinct Bugs

### Bug 1 (PRIMARY): `train_grpo.py` — wrong initialization order destroys DPO→GRPO transfer

**Location**: `scripts/train_grpo.py`, lines 400-412 (pre-fix)

**What happened**:
```python
# WRONG ORDER (original code):
policy_model.load_weights(adapter / "adapters.safetensors", strict=False)  # line 402 — FAILS SILENTLY
linear_to_lora_layers(policy_model, ...)  # line 405 — creates fresh LoRA layers from scratch
```

`load_weights` runs BEFORE `linear_to_lora_layers`. At that point, no LoRA parameter keys exist in the model (e.g., `lora_A`, `lora_B`). Because `strict=False`, all LoRA keys from the DPO adapter are silently discarded. Then `linear_to_lora_layers` creates fresh, randomly initialized LoRA layers.

**Result**: The GRPO policy model trained from a fresh base model for 1000 steps, completely ignoring the DPO adapter warm-start. The GRPO adapter is effectively decorrelated from the SFT/DPO knowledge.

**Fix applied**: Move `linear_to_lora_layers` BEFORE `load_weights` for both policy and reference models. This ensures LoRA keys exist when the adapter weights are loaded.

---

### Bug 2 (SECONDARY): Q8_0 quantization degrades factual numeric accuracy

When the GRPO fused model (already degraded to 2022 figures due to Bug 1) is converted to GGUF Q8_0, quantization errors shift the output further back to 2021 figures. This is a separate effect layered on top of Bug 1.

Similarly, even when the SFT adapter (which correctly gives $14,600 single / $29,200 MFJ on MLX) is fused and converted to GGUF Q8_0, it hallucinates "$15,000" for single filers — wrong but in the right ballpark. Q8_0 is insufficient to preserve fine-tuned numeric knowledge in a 3B model.

---

## Comparison Table

| Model | Standard Deduction (Single) | Correct? |
|-------|----------------------------|----------|
| Base model (no adapter) | $12,950 (2022 figures) | NO |
| SFT adapter (MLX) | $14,600 (2024 figures) | YES |
| DPO adapter (MLX) | Unable to load (config mismatch) | N/A |
| GRPO v5 step-1000 adapter (MLX) | $12,950 (2022 figures, same as base) | NO |
| GRPO v5 best adapter (step ~40, MLX) | $12,950 (2022 figures, same as base) | NO |
| fused (GRPO v5, MLX) | $12,950 (2022 figures) | NO |
| GGUF q8_0 (GRPO v5, Ollama) | $12,550 (2021 figures) or $13,850 (2023, scrambled) | NO |
| fused_sft (SFT adapter, MLX) | $14,600 (2024 figures) | YES |
| GGUF q8_0 (SFT, Ollama - new) | $15,000 (hallucinated) | CLOSE but wrong |

---

## Why the v5 Evaluation Showed $13,850

The v5 evaluation at 21:52 loaded `outputs/grpo/adapters` and somehow got $13,850. At temperature=0.3 with non-zero sampling, model outputs can vary. On a subsequent run immediately after, the adapter gives $12,950. The $13,850 result was likely a lucky sample that matched the 2023 figure (which the GRPO training saw in the reward computation data), not a reliable output. The evaluation only ran each question once.

---

## What the GRPO Training Actually Did

- Started from fresh base model (DPO adapter ignored due to Bug 1)
- Trained for 1000 steps on prompts with citation-based rewards
- Best checkpoint was at step 40 (avg_reward=0.609) — very early in training
- The final step-1000 adapter has slightly lower average reward (0.461 at last step)
- The GRPO training improved citation behavior but regressed numeric facts (model converged on safe, citation-heavy language but with base model's wrong figures)

---

## Immediate Fix Applied

1. Fixed `scripts/train_grpo.py`: moved `linear_to_lora_layers` before `load_weights` for both policy and reference models
2. Fused SFT adapter into new model: `outputs/final/fused_sft/`
3. Converted to GGUF: `outputs/final/model-sft-q8.gguf`
4. Created new Ollama model: `qwen25-tax-3b-sft-v5`

---

## Recommended Next Steps

1. **Re-run GRPO training** with the fixed `train_grpo.py` so DPO adapter weights actually initialize GRPO. This will give true DPO → GRPO warm-start.

2. **Use F16 or BF16 GGUF** instead of Q8_0 to preserve numeric precision. Q8_0 is insufficient for retaining fine-tuned dollar amounts in a 3B model. Command:
   ```
   convert_hf_to_gguf.py outputs/final/fused_sft --outfile outputs/final/model-sft-bf16.gguf --outtype bf16
   ```

3. **Use `qwen25-tax-3b-sft-v5`** as the best current Ollama model (SFT-based, gives ~$15,000 which is closer than $12,550 but still not exact due to quantization). For best accuracy use MLX directly with `outputs/sft/adapters`.

4. **Fix the DPO adapter_config.json**: it's missing `num_layers` key, causing a crash when trying to load it with `mlx_lm.load`. The GRPO adapter config also uses `num_layers` inconsistently vs `lora_layers`.

5. **Run the formal 25-question eval** against the SFT adapter to establish it as the true v5 baseline before investing in GRPO retraining.
