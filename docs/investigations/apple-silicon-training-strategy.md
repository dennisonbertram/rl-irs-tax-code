# Apple Silicon M4 Max 128GB: Fine-Tuning & RL Training Strategy

**Date:** 2025-03-25
**Hardware:** Apple M4 Max, 128GB Unified Memory
**Target Model:** Qwen 2.5 3B

---

## Executive Summary

**Recommended path: Use MLX-native tooling (mlx-lm-lora or mlx-tune) for both SFT and GRPO/DPO training. Skip PyTorch MPS entirely. For a 3B model with 128GB unified memory, full-precision LoRA (no quantization) is the sweet spot -- full fine-tuning is possible but LoRA gives faster iteration with comparable results.**

---

## 1. Unsloth on Apple Silicon M4 Max

### Status: Not natively supported. Use mlx-tune instead.

- **Unsloth core** relies on Triton kernels, which are CUDA-only. It does **not** work with MPS backend.
- There is a community PR ([#1289](https://github.com/unslothai/unsloth/pull/1289)) adding Apple Silicon support, but it is not merged.
- Official docs state Apple Silicon/MLX support is "in the works" and "coming soon" to Unsloth Studio.
- The `unsloth-mlx` PyPI package exists but has been **renamed to `mlx-tune`** -- it is a community project (not official Unsloth) that wraps MLX in an Unsloth-compatible `FastLanguageModel` API.
- **Bottom line:** Do not try to use Unsloth on M4 Max. The CUDA kernel dependency is fundamental.

### Sources
- [Unsloth Requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)
- [Unsloth Apple Silicon Issue #685](https://github.com/unslothai/unsloth/issues/685)
- [unsloth-mlx on PyPI](https://pypi.org/project/unsloth-mlx/0.3.5/)

---

## 2. MLX-lm for SFT -- and DPO/GRPO Support

### Status: MLX ecosystem now supports SFT, DPO, GRPO, and more.

There are **three MLX-native training frameworks** worth considering:

### A. `mlx-lm` (Apple Official)
- Apple's official library for inference and LoRA/QLoRA fine-tuning
- Supports **SFT only** (no DPO/GRPO)
- Well-maintained, stable, good Qwen support
- Best for: Quick SFT experiments
- Repo: [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)

### B. `mlx-lm-lora` (Community - Goekdeniz Guelmez)
- **12 training algorithms**: SFT, DPO, CPO, ORPO, GRPO, GSPO, Dr. GRPO, DAPO, Online DPO, XPO, RLHF, PPO
- Built-in synthetic dataset generation and custom judge training
- CLI-first with YAML config support
- Unified memory access to full system RAM
- **This is the most comprehensive option for RL training on Apple Silicon**
- Repo: [Goekdeniz-Guelmez/mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)

### C. `mlx-tune` (Community - ARahim3, formerly unsloth-mlx)
- Supports SFT, DPO, GRPO, and Vision fine-tuning
- **Unsloth-compatible API** -- write code once, run on Mac or push to CUDA cluster
- Good if you want code portability between Mac dev and cloud training
- Repo: [ARahim3/mlx-tune](https://github.com/ARahim3/mlx-tune)

### D. `MLX-GRPO` (Doriandarko)
- Pure MLX GRPO implementation, purpose-built for Apple Silicon
- Implements multiple reward functions (correctness, format-check, XML count)
- Uses GSM8K dataset for multi-step reasoning training
- Good reference implementation but less full-featured
- Repo: [Doriandarko/MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO)

### Recommendation

**Use `mlx-lm-lora` for the broadest algorithm support (especially GRPO/DPO).** If you need Unsloth API compatibility for later migration to CUDA, use `mlx-tune`.

---

## 3. Fastest Training Path for Qwen 2.5 3B on M4 Max

### Recommended Pipeline

```
Step 1: SFT with mlx-lm (or mlx-lm-lora)
  - Use full-precision LoRA (bf16, no quantization -- you have the memory)
  - LoRA rank 16-64 depending on task complexity
  - Expected: ~20-30 min for a typical dataset on M4 Max

Step 2: GRPO/DPO with mlx-lm-lora
  - Use the SFT checkpoint as base
  - GRPO for reasoning tasks (generates multiple responses, compares within group)
  - DPO for preference alignment (requires chosen/rejected pairs)

Step 3: Export & Merge
  - Merge LoRA adapters into base model
  - Convert to GGUF if needed for deployment
```

### Why This Path is Fastest

1. **MLX is native** -- no MPS translation layer, no CUDA emulation
2. **Unified memory** -- zero-copy between CPU and GPU, no transfer overhead
3. **Metal compute shaders** tuned for Apple GPU architecture
4. **MLX is 20-30% faster than llama.cpp** on Apple Silicon (and the gap widens on larger models)

### Alternative: PyTorch + TRL on MPS
- Possible but **not recommended** (see Section 5 for gotchas)
- Significantly slower than MLX for the same operations
- Many TRL features silently fail or produce wrong results on MPS

---

## 4. Memory Strategy: 128GB Unified Memory

### Can We Use the Full 128GB?

**Yes, but with caveats:**
- MLX accesses the full unified memory pool directly
- However, macOS reserves ~8-12GB for system operations
- Practical available: ~116-120GB

### Should We Skip Quantization?

**For a 3B model: Yes, absolutely skip quantization.**

Memory budget for Qwen 2.5 3B:

| Method | Model Memory | Training Overhead | Total Estimate |
|--------|-------------|-------------------|----------------|
| Full fine-tune (bf16) | ~6 GB | ~30-48 GB (optimizer + gradients) | ~36-54 GB |
| LoRA (bf16, no quant) | ~6 GB | ~2-4 GB (LoRA params + gradients) | ~8-10 GB |
| QLoRA (4-bit) | ~1.5 GB | ~1-2 GB | ~2.5-3.5 GB |

**Recommendation:** Use **bf16 LoRA without quantization**.

- You have 128GB -- QLoRA's memory savings are unnecessary and quantization degrades training signal quality
- Full fine-tuning is possible (~48GB) but LoRA gives comparable results with much faster iteration
- With bf16 LoRA, you'll use ~10GB, leaving massive headroom for large batch sizes and long sequences
- Increase batch size to 8-16 and sequence length to 4096-8192 to better utilize the memory

### For GRPO Specifically

GRPO generates multiple completions per prompt (typically 4-16 per group). This multiplies memory usage. With 128GB:
- You can run group_size=16 comfortably at 3B scale
- This is a significant advantage over CUDA GPUs with 24-48GB VRAM

---

## 5. PyTorch MPS Backend Gotchas (Why to Avoid for Training)

### Known Issues

1. **Missing Operations**: Many PyTorch ops are not implemented for MPS. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` silently falls back to CPU, causing massive slowdowns.

2. **No float64 Support**: MPS lacks double precision entirely. Some loss functions and gradient computations silently lose precision.

3. **Kernel Bugs**: `addcmul_` and `addcdiv_` operations silently fail on non-contiguous tensors (fixed in PyTorch 2.4+, but trust is low).

4. **Silent Training Failures**: Model encoder weights can freeze during training due to the above bugs -- training appears to proceed normally but produces garbage.

5. **No Distributed Training**: MPS backend does not support any form of distributed training.

6. **Autograd Debugging Unusable**: `torch.autograd.detect_anomaly` slows MPS execution by ~100x.

7. **Numerical Divergence**: MPS does not match CUDA's stochastic rounding, making cross-platform debugging impossible.

8. **TRL-Specific Issues**: Many TRL (Transformer Reinforcement Learning) library features assume CUDA. Flash attention, gradient checkpointing, and mixed precision training all have MPS compatibility issues.

### Expert Opinion

Sebastian Raschka (AI researcher) has stated: "I would not fine-tune even small LLMs on [Apple Silicon] -- it gets very hot, and MPS on macOS is still unstable, with fine-tuning often failing to converge."

**This is why MLX is the answer** -- it bypasses MPS entirely and uses Metal compute shaders directly.

### Sources
- [Accelerated PyTorch on Mac - Apple Developer](https://developer.apple.com/metal/pytorch/)
- [HuggingFace Apple Silicon Training Guide](https://huggingface.co/docs/transformers/en/perf_train_special)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)

---

## 6. Other Frameworks Worth Knowing About

| Framework | RL Support | Apple Silicon | Notes |
|-----------|-----------|---------------|-------|
| **mlx-lm-lora** | Full (12 algos) | Native MLX | **Best overall for RL on Apple Silicon** |
| **mlx-tune** | SFT/DPO/GRPO | Native MLX | Unsloth API compat, good for portability |
| **MLX-GRPO** | GRPO only | Native MLX | Clean reference implementation |
| **mlx-lm** | SFT only | Native MLX | Apple official, very stable |
| **PyTorch + TRL** | Full | MPS (buggy) | Not recommended on Apple Silicon |
| **Unsloth** | SFT/DPO/GRPO | No (CUDA only) | Cannot run on Apple Silicon |
| **Axolotl** | SFT/DPO | MPS (partial) | Limited Apple Silicon support |
| **LLaMA-Factory** | SFT/DPO/PPO | MPS (partial) | Better on CUDA |

### Framework Not to Miss: mlx-lm-lora

This is the standout option. 12 training algorithms including the latest RL methods (GRPO, DAPO, Dr. GRPO), all running natively on MLX with full unified memory access. This did not exist a year ago and represents a major leap for Apple Silicon training capability.

---

## 7. Recommended Action Plan

```
1. Install mlx-lm-lora:
   pip install mlx-lm-lora

2. Download Qwen 2.5 3B (or convert with mlx_lm.convert):
   mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct --mlx-path ./qwen25-3b-mlx

3. Prepare training data:
   - SFT: JSONL with {"messages": [...]} format
   - DPO: JSONL with {"prompt": ..., "chosen": ..., "rejected": ...}
   - GRPO: Prompts + reward functions (no pre-labeled data needed)

4. Run SFT first:
   mlx-lm-lora train --model ./qwen25-3b-mlx \
     --data ./sft_data.jsonl \
     --method sft \
     --lora-rank 32 \
     --batch-size 8 \
     --num-epochs 3

5. Run GRPO on the SFT checkpoint:
   mlx-lm-lora train --model ./qwen25-3b-mlx \
     --adapter ./sft-adapter \
     --method grpo \
     --group-size 8 \
     --batch-size 4

6. Evaluate and iterate
```

---

## 8. Key Takeaways

1. **MLX, not MPS** -- Use MLX-native frameworks. PyTorch MPS is too buggy for training.
2. **mlx-lm-lora is the best all-in-one** -- 12 algorithms including GRPO, DPO, PPO, all native MLX.
3. **Skip quantization** -- With 128GB, use bf16 LoRA for maximum training quality.
4. **LoRA over full fine-tune** -- Even with enough memory for full FT, LoRA is faster to iterate and nearly as effective.
5. **GRPO is your friend for reasoning** -- It doesn't need labeled preference data, generates its own comparisons, and 128GB lets you run large group sizes.
6. **128GB is a superpower for GRPO** -- Large group sizes (8-16) that would OOM on a 24GB GPU run comfortably here.
7. **Unsloth is a dead end on Apple Silicon** -- Don't waste time trying to make it work.

---

## Sources

- [mlx-lm-lora GitHub](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
- [mlx-tune GitHub](https://github.com/ARahim3/mlx-tune)
- [MLX-GRPO GitHub](https://github.com/Doriandarko/MLX-GRPO)
- [Unsloth Apple Silicon PR #1289](https://github.com/unslothai/unsloth/pull/1289)
- [Unsloth Requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)
- [MLX Framework](https://mlx-framework.org/)
- [WWDC25: Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [HuggingFace RL Post-Training Guide](https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms)
- [Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/)
- [HuggingFace Apple Silicon Training](https://huggingface.co/docs/transformers/en/perf_train_special)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)
- [Modal: VRAM for Fine-Tuning](https://modal.com/blog/how-much-vram-need-fine-tuning)
- [Run and Fine-Tune LLMs on Mac with MLX-LM 2026](https://markaicode.com/run-fine-tune-llms-mac-mlx-lm/)
- [Implementing GRPO in Apple MLX](https://dev.to/lewis_won/implementing-deekseek-r1-grpo-in-apple-mlx-framework-3n97)
- [Sebastian Raschka: State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
