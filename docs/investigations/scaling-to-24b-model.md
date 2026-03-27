# Scaling to ~24B+ Parameter Model on Apple Silicon

**Date**: 2026-03-26
**Hardware**: MacBook Pro M4 Max, 128 GB unified memory, 16 cores (12P+4E), 40-core GPU
**Disk**: 226 GB available on /dev/disk3s5

---

## 1. Model Selection

### Available MLX Models in the ~24B-32B Range

| Model | Params | MLX Repo (4-bit) | Disk Size | RAM (4-bit) |
|-------|--------|-------------------|-----------|-------------|
| **Qwen2.5-32B-Instruct** | 32.5B | `mlx-community/Qwen2.5-32B-Instruct-4bit` | ~18.4 GB | ~20-22 GB |
| **Qwen2.5-14B-Instruct** | 14.8B | `mlx-community/Qwen2.5-14B-Instruct-4bit` | ~8.3 GB | ~10-12 GB |
| **Mistral-Small-3.2-24B-Instruct** | 24B | `mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit` | ~14 GB | ~16-18 GB |
| **Mistral-Small-24B-Instruct-2501** | 24B | `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` | ~14 GB | ~16-18 GB |

Also available in 8-bit and bf16 variants:
- `mlx-community/Qwen2.5-32B-Instruct-8bit` (~34 GB disk)
- `mlx-community/Qwen2.5-32B-bf16` (~65 GB disk)
- `mlx-community/Qwen2.5-14B-Instruct-8bit` (~15.7 GB disk)
- `mlx-community/Qwen2.5-14B-Instruct-bf16` (~30 GB disk)

### Why Stick with Qwen2.5 Family

Since the existing pipeline is built around Qwen2.5-3B-Instruct:
- Same tokenizer family and chat template -- minimal pipeline changes
- Same architecture (attention patterns, MLP structure) -- LoRA configs transfer cleanly
- Qwen2.5-32B outperforms Qwen2.5-14B on knowledge recall benchmarks (MMLU: 83.3 vs 79.7)
- The 32B model is closest to GPT-4o-mini-level factual knowledge, which is what we need for memorizing IRC section numbers and dollar amounts

---

## 2. Memory Analysis

### Hardware Budget

- **Total unified memory**: 128 GB
- **Usable for ML** (after OS/system): ~96-110 GB (Apple Silicon typically allows 75-85% for ML workloads)
- **M4 Max memory bandwidth**: 546 GB/s

### Estimated Memory for LoRA Training (4-bit Quantized Base)

**Formula**: Base model (4-bit) + LoRA adapters (bf16) + Optimizer states + Activations + KV cache

#### Qwen2.5-32B-Instruct-4bit, LoRA rank 32, batch_size=1, seq_len=2048

| Component | Estimated Memory |
|-----------|-----------------|
| Base model weights (4-bit) | ~18-20 GB |
| LoRA adapter weights (bf16, rank 32, 16 layers) | ~0.2 GB |
| Optimizer states (Adam, 2x adapter size) | ~0.4 GB |
| Activations + gradients (with grad checkpointing) | ~8-15 GB |
| KV cache (seq_len=2048) | ~2-4 GB |
| MLX framework overhead | ~2-3 GB |
| **Total estimated (SFT)** | **~35-45 GB** |

#### Qwen2.5-32B DPO (needs chosen + rejected forward passes)
| **Total estimated (DPO)** | **~50-65 GB** |

#### Qwen2.5-32B GRPO (group_size=4 completions per prompt)
| **Total estimated (GRPO, group_size=4)** | **~60-80 GB** |

### Verdict: 32B Fits Comfortably on 128 GB

With 128 GB unified memory, even GRPO with group_size=4 should fit. The 32B 4-bit model is well within budget. There is no need to settle for the 14B model.

For comparison, community reports show Qwen2.5-14B LoRA training peaks at ~33 GB on Apple Silicon. Scaling from 14B to 32B roughly doubles memory, putting us at ~60-70 GB for typical training -- well within the 128 GB envelope.

---

## 3. Training Feasibility and Time Estimates

### Training Speed Scaling

Apple Silicon LoRA training throughput scales roughly inversely with model size. Based on community benchmarks:

| Model Size | Approx. Training Speed (tok/s) | Time per 1000 SFT steps (batch=1, seq=2048) |
|------------|-------------------------------|---------------------------------------------|
| 3B (current) | ~200-300 tok/s | ~2-3 hours |
| 14B (4-bit) | ~60-100 tok/s | ~6-10 hours |
| 32B (4-bit) | ~30-50 tok/s | ~12-20 hours |

### Estimated Training Times for Full Pipeline

Using 4-bit Qwen2.5-32B-Instruct on M4 Max 128 GB:

| Stage | Steps | Estimated Time |
|-------|-------|---------------|
| **SFT** (1000 iters, batch=1-2, seq=2048) | 1000 | 12-24 hours |
| **DPO** (500 iters, batch=1, seq=1024) | 500 | 6-12 hours |
| **GRPO** (300 iters, batch=1, group_size=4) | 300 | 12-24 hours |
| **Total pipeline** | | **~30-60 hours (1.5-2.5 days)** |

This is roughly 5-10x slower than the 3B model, but entirely feasible as a multi-day run.

### Recommended Config Adjustments for 32B

```yaml
# SFT for 32B
batch_size: 1           # down from 4 (3B) -- memory constraint
max_seq_length: 2048    # keep the same
lora_layers: 16         # keep the same
lora_rank: 32           # keep the same (or try 64 for more capacity)
grad_checkpoint: true   # essential for 32B

# DPO for 32B
batch_size: 1           # down from 2
max_seq_length: 1024    # keep the same

# GRPO for 32B
group_size: 4           # keep the same (fits in 128 GB)
batch_size: 1           # keep at 1
max_new_tokens: 512     # keep the same
```

### LoRA Rank Considerations

- Rank 32 worked for 3B; for 32B the base model has much more capacity already
- Rank 32 is still reasonable -- the LoRA adapter is proportionally smaller relative to the full model
- Could experiment with rank 64 for the SFT stage to maximize knowledge injection, since the larger model has more dimensions to adapt
- Keep rank 32 for DPO/GRPO stages where we are fine-tuning behavior, not injecting knowledge

---

## 4. Disk Space Analysis

| Item | Size |
|------|------|
| Available disk space | 226 GB |
| Qwen2.5-32B-Instruct-4bit download | ~18.4 GB |
| Qwen2.5-32B-Instruct-8bit download | ~34 GB |
| LoRA adapters (all stages) | < 1 GB |
| HuggingFace cache overhead | ~5 GB |
| **Total needed (4-bit)** | **~25 GB** |
| **Remaining after download** | **~200 GB** |

Disk space is not a concern.

---

## 5. Recommendation

### Primary Recommendation: `mlx-community/Qwen2.5-32B-Instruct-4bit`

**Why this model:**

1. **Same family as current 3B model** -- minimal pipeline changes needed. Same tokenizer, same chat template, same architecture. Config files just need the model path updated and batch sizes reduced.

2. **32B has 10x the parameters of 3B** -- dramatically more capacity for memorizing specific IRC section numbers, dollar thresholds, and regulatory details. The 32B model scores 83.3 on MMLU vs 74.2 for 7B, showing superior factual knowledge retention.

3. **4-bit quantization keeps memory manageable** -- estimated ~35-45 GB peak for SFT, ~60-80 GB peak for GRPO. Well within the 128 GB budget.

4. **Qwen2.5-32B outperforms alternatives at this size class** -- beats Llama 3.1 33B equivalents and Mistral models on knowledge-intensive benchmarks.

5. **18.4 GB download** -- fast to acquire, fits easily on disk.

### Alternative: `mlx-community/Qwen2.5-14B-Instruct-4bit`

If 32B training proves too slow or memory-hungry in practice:
- 14B is a safe fallback (8.3 GB download, ~25-35 GB training memory)
- Still 4.7x larger than current 3B model
- Much faster training (~2x speed of 32B)
- Could serve as a stepping stone to validate pipeline changes before committing to 32B

### Pipeline Changes Needed

1. **Download model**: `mlx_lm.convert --hf-path Qwen/Qwen2.5-32B-Instruct -q --q-bits 4` or directly download `mlx-community/Qwen2.5-32B-Instruct-4bit`
2. **Update config files**: Change `model:` path, reduce `batch_size` to 1, keep everything else
3. **Consider mlx-tune**: The newer `mlx-tune` library (github.com/ARahim3/mlx-tune) provides native SFT + DPO + GRPO support with an Unsloth-compatible API, potentially simplifying the pipeline
4. **Test with a short SFT run first**: Run 50 steps to verify memory fits and estimate throughput before committing to the full pipeline

### Potential Blockers

- **None critical.** 128 GB unified memory on M4 Max is sufficient for 4-bit 32B LoRA training.
- **Training time**: ~30-60 hours total is long but manageable. Can be done over a weekend.
- **4-bit quantization during training**: Training on a quantized base model may slightly reduce final quality compared to bf16, but the trade-off is necessary and widely accepted in practice. The LoRA adapters themselves remain in bf16.

---

## 6. Sources

- [mlx-community/Qwen2.5-32B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-4bit)
- [mlx-community/Qwen2.5-32B-Instruct-8bit](https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-8bit)
- [mlx-community/Qwen2.5-14B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit)
- [mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit](https://huggingface.co/mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit)
- [mlx-tune - SFT/DPO/GRPO for Apple Silicon](https://github.com/ARahim3/mlx-tune)
- [MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO)
- [Qwen2.5 Technical Report](https://qwenlm.github.io/blog/qwen2.5-llm/)
- [Train LLM on Mac Studio using MLX](https://michalasobczak.pl/ai-ml/2025/10/train-llm-on-mac-studio-using-mlx-framework/)
- [Explore LLMs on Apple Silicon with MLX - WWDC25](https://developer.apple.com/videos/play/wwdc2025/298/)
- [LoRA Fine-Tuning on Apple Silicon MacBook](https://towardsdatascience.com/lora-fine-tuning-on-your-apple-silicon-macbook-432c7dab614a/)
