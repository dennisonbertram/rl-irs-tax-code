# Qwen3.5-27B Model Evaluation

**Date**: 2026-03-27
**Source**: https://huggingface.co/Qwen/Qwen3.5-27B
**License**: Apache 2.0
**Released**: 2026-02-24
**Downloads**: ~2.6M (as of evaluation date)

---

## 1. Architecture

Qwen3.5-27B is a **multimodal (vision-language) model** -- NOT a text-only model like Qwen2.5-32B.

| Property | Value |
|---|---|
| **Architecture** | `Qwen3_5ForConditionalGeneration` (Causal LM + Vision Encoder) |
| **Pipeline tag** | `image-text-to-text` |
| **Model type** | Hybrid attention: Gated DeltaNet + Full Attention |
| **Parameters** | 27.78B (27,781,427,952 total) |
| **Hidden size** | 5120 |
| **Layers** | 64 |
| **Layer layout** | 16 blocks of: 3x (Gated DeltaNet -> FFN) + 1x (Full Attention -> FFN) |
| **Attention (full)** | 24 Q heads, 4 KV heads, head dim 256 |
| **DeltaNet (linear)** | 48 V heads, 16 QK heads, head dim 128 |
| **FFN intermediate** | 17,408 |
| **Vocab size** | 248,320 |
| **Context length** | **262,144 tokens** natively, extensible to ~1M tokens |
| **RoPE** | M-RoPE (multimodal rotary), theta=10M, partial rotary factor 0.25 |
| **Activation** | SiLU |
| **Vision encoder** | 27-layer ViT, hidden=1152, patch_size=16, 16 heads |
| **Training** | Pre-training + Post-training (RL-based) |
| **MTP** | Multi-token prediction trained |
| **Dtype** | BF16 |

### Key Architectural Innovation
The hybrid architecture alternates between **Gated DeltaNet** (linear attention, 75% of layers) and **full attention** (25% of layers). This gives near-linear-time inference for long contexts while retaining the expressiveness of full attention at regular intervals. Every 4th layer uses full softmax attention.

### Important Note: This is a Multimodal Model
Unlike Qwen2.5-32B which was text-only, Qwen3.5-27B has a built-in vision encoder. The `image-text-to-text` pipeline tag means this is a unified VLM. The model card states: "Early fusion training on multimodal tokens achieves cross-generational parity with Qwen3 and outperforms Qwen3-VL models."

---

## 2. Benchmark Scores (Language Benchmarks)

Compared models: GPT-5-mini, GPT-OSS-120B, Qwen3-235B-A22B, Qwen3.5-122B-A10B, **Qwen3.5-27B**, Qwen3.5-35B-A3B

| Benchmark | GPT-5-mini | Qwen3-235B | Qwen3.5-27B | Notes |
|---|---|---|---|---|
| **MMLU-Pro** | 83.7 | 84.4 | **86.1** | Beats GPT-5-mini |
| **MMLU-Redux** | 93.7 | 93.8 | **93.2** | Competitive |
| **C-Eval** | 82.2 | 92.1 | **90.5** | Strong Chinese |
| **SuperGPQA** | 58.6 | 64.9 | **65.6** | Beats Qwen3-235B |
| **IFEval** | 93.9 | 87.8 | **95.0** | Best in class |
| **IFBench** | 75.4 | 51.7 | **76.5** | Best in class |
| **GPQA Diamond** | 82.8 | 81.1 | **85.5** | Strong reasoning |
| **HMMT Feb 25** | 89.2 | 85.1 | **92.0** | Best in class |
| **SWE-bench Verified** | 72.0 | -- | **72.4** | Beats GPT-5-mini |
| **LiveCodeBench v6** | 80.5 | 75.1 | **80.7** | Beats GPT-5-mini |
| **LongBench v2** | 56.8 | 54.8 | **60.6** | Strong long-context |
| **BFCL-V4** (agent) | 55.5 | 54.8 | **68.5** | Massive improvement |
| **TAU2-Bench** (agent) | 69.8 | 58.5 | **79.0** | Massive improvement |
| **MMMLU** (multilingual) | 86.2 | 83.4 | **85.9** | Near GPT-5-mini |

### Vision-Language Benchmarks

| Benchmark | GPT-5-mini | Claude-Sonnet-4.5 | Qwen3.5-27B |
|---|---|---|---|
| MMMU | 79.0 | 79.6 | **82.3** |
| MMMU-Pro | 67.3 | 68.4 | **75.0** |
| MathVision | 71.9 | 71.1 | **86.0** |
| Mathvista (mini) | 79.1 | 79.8 | **87.8** |
| DynaMath | 81.4 | 78.8 | **87.7** |

### vs. Qwen2.5-32B (our current model)
Qwen3.5-27B benchmarks are not directly compared to Qwen2.5 in the model card. However, based on known Qwen2.5-32B scores and the Qwen3.5-27B results:
- **Qwen2.5-32B MMLU-Pro**: ~72-73 vs **Qwen3.5-27B: 86.1** (massive jump)
- **Qwen2.5-32B GPQA Diamond**: ~49 vs **Qwen3.5-27B: 85.5** (massive jump)
- Qwen3.5-27B generally competes with or beats models 4-8x its size (Qwen3-235B)
- The architectural changes (hybrid attention, DeltaNet) provide significant efficiency gains
- This is a generational leap -- not an incremental improvement

---

## 3. Instruct Variant

**No official `Qwen3.5-27B-Instruct` exists.** The HuggingFace page for `Qwen/Qwen3.5-27B-Instruct` returns a **404 error**.

The base `Qwen3.5-27B` IS already post-trained (instruction-tuned + RL). The model card states: "This repository contains model weights and configuration files for the **post-trained model**." The chat template with `<think>` tags is built in, confirming it is already instruction-following.

Community instruct variants exist (via distillation/fine-tuning):
- `DavidAU/Qwen3.5-27B-Claude-4.6-OS-INSTRUCT` (Claude distilled)
- `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` (reasoning distilled)

---

## 4. Quantization Formats Available

### MLX (Apple Silicon)
| Model | Bits | Downloads | Library |
|---|---|---|---|
| `mlx-community/Qwen3.5-27B-4bit` | 4-bit | 49,460 | transformers/mlx |
| `mlx-community/Qwen3.5-27B-8bit` | 8-bit | 7,759 | transformers/mlx |
| `mlx-community/Qwen3.5-27B-4bit-DWQ` | 4-bit DWQ | 651 | mlx |
| `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` | 4-bit | 31,826 | mlx |
| `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit` | 6-bit | 8,332 | mlx |

### GGUF
| Model | Notes |
|---|---|
| `unsloth/Qwen3.5-27B-GGUF` | Official-ish GGUF conversion |
| `bartowski/Qwen_Qwen3.5-27B-GGUF` | Community GGUF |
| `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF` | Distilled variant |

### AWQ
| Model | Notes |
|---|---|
| `QuantTrio/Qwen3.5-27B-AWQ` | AWQ quantized |
| `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4` | AWQ INT4 |
| `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8` | AWQ INT8 |

### GPTQ
| Model | Notes |
|---|---|
| `SamCrowdwave/Qwen3.5-27B-Instruct-abliterated-GPTQ-Int8` | GPTQ INT8 |

---

## 5. Memory Requirements

### Inference (estimated)

| Format | Size (GB) | RAM Needed |
|---|---|---|
| BF16 (full precision) | ~55.6 GB | ~58-60 GB |
| 8-bit quantized | ~28 GB | ~30-32 GB |
| 4-bit quantized | ~15-16 GB | ~17-19 GB |
| 3-bit quantized | ~12 GB | ~14 GB |

The model storage on HuggingFace is 55.6 GB (11 safetensor shards), confirming the BF16 size.

### LoRA Fine-Tuning (estimated)

| Configuration | RAM Needed | Notes |
|---|---|---|
| BF16 + LoRA (rank 16) | ~62-68 GB | Base model + LoRA adapters + optimizer states + activations |
| 4-bit QLoRA | ~22-28 GB | Quantized base + LoRA in BF16 |
| 8-bit + LoRA | ~38-45 GB | 8-bit base + LoRA adapters |

### GRPO Training (2 model copies: policy + reference)

| Configuration | RAM Needed | Notes |
|---|---|---|
| BF16 policy + BF16 ref | ~120-130 GB | Tight on 128GB |
| 4-bit policy + 4-bit ref | ~40-50 GB | Very feasible on 128GB |
| BF16 policy + 4-bit ref (common) | ~75-85 GB | Feasible on 128GB |
| 4-bit QLoRA policy + 4-bit ref | ~35-45 GB | Comfortable on 128GB |

---

## 6. mlx-lm Compatibility

### Current Status: UNCERTAIN / LIKELY INCOMPATIBLE for training

**Key concern: This is a NEW architecture (`qwen3_5`).**

The model uses `Qwen3_5ForConditionalGeneration` with a hybrid Gated DeltaNet + Full Attention architecture. This is NOT a standard transformer:
- Standard Qwen2.5 uses `Qwen2ForCausalLM` -- well-supported in mlx-lm
- Qwen3.5 uses `Qwen3_5ForConditionalGeneration` -- a **conditional generation** (multimodal) architecture with **linear attention layers** (Gated DeltaNet)

**mlx-lm issues:**
1. **Architecture support**: mlx-lm would need explicit support for the `qwen3_5` model type, including the Gated DeltaNet layers. The `mlx-community` quantized versions exist (suggesting inference works via some path), but training/fine-tuning support is a different matter.
2. **Multimodal**: The model is `image-text-to-text`, not pure `text-generation`. mlx-lm is designed for text-only causal LMs. The vision encoder adds complexity.
3. **Linear attention**: The Gated DeltaNet layers use a fundamentally different attention mechanism than standard softmax attention. LoRA applied to these layers would need custom implementation.
4. **Conditional generation class**: `AutoModelForImageTextToText` vs `AutoModelForCausalLM` -- these are different HuggingFace model classes.

**However**: The `mlx-community/Qwen3.5-27B-4bit` has 49K downloads and is tagged with `mlx`, suggesting inference works. Some community versions use the `mlx` library tag directly. The `Brooooooklyn/Qwen3.5-27B-unsloth-mlx` model is tagged with `mlx-node`.

**Bottom line**: Inference on MLX likely works via community conversions. LoRA training via `mlx-lm` is **highly uncertain** and would need testing. The hybrid DeltaNet architecture may not be supported for fine-tuning in current mlx-lm.

---

## 7. Compatibility Analysis for M4 Max 128GB

### SFT (Supervised Fine-Tuning)
- **4-bit QLoRA**: ~22-28 GB -- YES, fits easily
- **BF16 full LoRA**: ~62-68 GB -- YES, fits with headroom
- **Full fine-tuning**: ~110-130 GB -- TIGHT, likely OOM with optimizer states

### DPO (Direct Preference Optimization)
DPO needs 2 model copies (policy + reference):
- **4-bit both**: ~35-45 GB -- YES, comfortable
- **BF16 policy + 4-bit ref**: ~75-85 GB -- YES, feasible
- **BF16 both**: ~120-130 GB -- VERY TIGHT, likely OOM

### GRPO (Group Relative Policy Optimization)
GRPO needs 2 model copies + generation overhead:
- **4-bit QLoRA policy + 4-bit ref**: ~40-55 GB -- YES, feasible
- **BF16 policy + 4-bit ref**: ~80-95 GB -- POSSIBLE but tight
- **BF16 both**: NOT feasible on 128GB

### Comparison: 27B vs 32B (Qwen2.5)
| Factor | Qwen2.5-32B | Qwen3.5-27B |
|---|---|---|
| Parameters | 32.5B | 27.8B |
| BF16 size | ~65 GB | ~55.6 GB |
| 4-bit size | ~18-19 GB | ~15-16 GB |
| Architecture | Standard transformer | Hybrid DeltaNet + Attention |
| Context length | 128K (extendable) | 262K native (extendable to 1M) |
| mlx-lm training support | Well supported | Uncertain |
| Multimodal | No (text-only) | Yes (vision + text) |
| Memory for GRPO (4-bit both) | ~45-55 GB | ~40-50 GB |
| Benchmark quality | Good | Significantly better |

**Qwen3.5-27B is 15% smaller in parameters, meaning it uses less memory while being dramatically better on benchmarks.** However, the new architecture introduces compatibility risks.

---

## 8. mlx-community Versions

YES -- multiple mlx-community versions exist:

| Model | Quant | Downloads |
|---|---|---|
| `mlx-community/Qwen3.5-27B-4bit` | 4-bit | 49,460 |
| `mlx-community/Qwen3.5-27B-8bit` | 8-bit | 7,759 |
| `mlx-community/Qwen3.5-27B-4bit-DWQ` | 4-bit DWQ | 651 |
| `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` | 4-bit | 31,826 |
| `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit` | 6-bit | 8,332 |
| `mlx-community/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit` | 4-bit | 2,299 |

The distilled Claude Opus variants are particularly popular.

---

## 9. Known Issues and Limitations

1. **New architecture**: The Gated DeltaNet hybrid architecture is novel. Tooling (vLLM, SGLang, mlx-lm) may have varying levels of support. The model card mentions compatibility with "Hugging Face Transformers, vLLM, SGLang, KTransformers."

2. **No separate Instruct variant**: The base model is already post-trained, but there is no separate instruct model. For RL training, this means the base model already has instruction-following and reasoning (with `<think>` tags) baked in.

3. **Multimodal overhead**: The vision encoder adds parameters and complexity that may not be needed for text-only tasks like tax code analysis. However, the text capabilities are clearly competitive.

4. **Transformers version**: Requires `transformers >= 4.57.0.dev0` -- this is a development version, suggesting the architecture support is bleeding-edge.

5. **Training infrastructure**: The model card mentions "near-100% multimodal training efficiency compared to text-only training" -- this is about their training, not about fine-tuning on consumer hardware.

6. **`<think>` tag reasoning**: The chat template includes built-in chain-of-thought with `<think>` tags. This could be beneficial for RL training (the model already knows how to reason step-by-step) but might also add token overhead.

---

## 10. Recommendation for IRS Tax Code RL Project

### Pros
- Dramatically better benchmarks than Qwen2.5-32B (generational improvement)
- 15% fewer parameters (55.6 GB vs ~65 GB), making GRPO more feasible on 128GB
- 262K native context (vs 128K) -- useful for long tax code passages
- Built-in reasoning via `<think>` tags -- aligns with RL training objectives
- Apache 2.0 license -- no restrictions
- Active community with MLX quantizations available

### Cons
- **Multimodal architecture**: The vision encoder is unnecessary for text-only tax code work and adds complexity
- **mlx-lm compatibility**: UNCERTAIN. The Gated DeltaNet hybrid architecture may not be supported for LoRA training in mlx-lm. This is the biggest risk.
- **Bleeding-edge**: Requires dev version of transformers. Ecosystem support is still catching up.
- **No text-only variant**: Unlike Qwen2.5 which had separate text and VL models, Qwen3.5-27B is always multimodal

### Verdict
**Test before committing.** The benchmark improvements are compelling, but the new architecture poses real compatibility risks for mlx-lm LoRA training. Recommended approach:
1. Try loading `mlx-community/Qwen3.5-27B-4bit` in mlx-lm and run inference
2. Attempt a small LoRA training run to see if the architecture is supported
3. If mlx-lm does not support the architecture for training, consider using Unsloth or HuggingFace TRL directly (which would work on Mac via MPS backend, though slower)
4. Alternatively, wait for a text-only Qwen3.5 variant if one is released, or check if the `text_config` sub-model can be extracted separately

If mlx-lm support works, Qwen3.5-27B would be a clear upgrade over Qwen2.5-32B for this project.
