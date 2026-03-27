# Scaling the IRS Tax Code RL Pipeline to Larger Models

**Date:** 2026-03-26
**Current setup:** Qwen 2.5 3B Instruct, MLX LoRA, M4 Max 128GB, ~5 hours total pipeline

---

## 1. What Model Sizes Are Feasible on 128GB Apple Silicon?

### Memory Model for LoRA Training on MLX

MLX uses Apple's unified memory — the 128GB is shared between CPU and GPU. During LoRA training, memory usage comes from:

1. **Base model weights** (loaded, frozen)
2. **LoRA adapter weights** (small, trainable)
3. **Activations** (forward pass, kept for backward)
4. **Gradients** (backward pass)
5. **Optimizer states** (Adam: 2x the trainable params)

**Key insight:** With LoRA, only ~0.1-2% of parameters are trainable, so optimizer states are small. The dominant costs are base model weights and activations.

### Memory Estimates: bf16 (No Quantization)

| Model Size | Weights (bf16) | Activations (bs=4, 2048 ctx) | Activations (bs=1, 2048 ctx) | Peak Training Memory | Feasible? |
|-----------|---------------|-------------------------------|-------------------------------|---------------------|-----------|
| 3B        | ~6 GB         | ~4 GB                         | ~1.5 GB                       | ~12 GB              | Yes, comfortably |
| 7B        | ~14 GB        | ~8 GB                         | ~3 GB                         | ~25 GB              | Yes, batch_size=4 works |
| 14B       | ~28 GB        | ~16 GB                        | ~5 GB                         | ~50 GB (bs=2)       | Yes, batch_size=2 |
| 32B       | ~64 GB        | ~32 GB                        | ~10 GB                        | ~85 GB (bs=1)       | Tight, needs grad_checkpoint |
| 70B       | ~140 GB       | N/A                           | N/A                           | >140 GB             | No — weights alone exceed 128GB |

### Memory Estimates: QLoRA (4-bit Quantized Base + bf16 LoRA)

MLX supports quantized training natively via `mlx_lm.lora` with `--quantize` or loading a pre-quantized model. The base weights are stored in 4-bit; LoRA adapters remain in bf16/float32.

| Model Size | Weights (4-bit) | Activations (bs=2, 2048 ctx) | Peak Training Memory | Feasible? |
|-----------|----------------|-------------------------------|---------------------|-----------|
| 3B        | ~2 GB          | ~2 GB                         | ~6 GB               | Trivial |
| 7B        | ~4 GB          | ~4 GB                         | ~12 GB              | Easy, batch_size=4 |
| 14B       | ~8 GB          | ~8 GB                         | ~22 GB              | Easy, batch_size=4 |
| 32B       | ~18 GB         | ~16 GB                        | ~42 GB (bs=2)       | Yes, comfortable |
| 70B       | ~38 GB         | ~30 GB                        | ~80 GB (bs=1)       | Yes, batch_size=1 with grad_checkpoint |
| 72B       | ~40 GB         | ~30 GB                        | ~82 GB (bs=1)       | Yes, same as 70B |

### QLoRA Quality Tradeoffs

Quantizing the base model to 4-bit introduces some quality loss, but research consistently shows:

- **4-bit QLoRA matches full bf16 LoRA** within 1-2% on most benchmarks (Dettmers et al., 2023, "QLoRA" paper). The LoRA adapters themselves are trained in full precision, so the fine-tuned knowledge is high-fidelity.
- **8-bit is nearly lossless** — if memory allows, 8-bit quantized base + LoRA is practically identical to bf16.
- **For domain-specific fine-tuning** (like tax law), the fine-tuned layers dominate the output quality. The quantization of the base model matters less because you're overriding the model's behavior in the target domain.
- **Practical recommendation:** Use 4-bit QLoRA for 32B and 70B models. The quality gain from model size massively outweighs the small quantization penalty.

### Concrete Recommendations for M4 Max 128GB

| Model | Method | Batch Size | Estimated Training Time* | Recommendation |
|-------|--------|------------|-------------------------|----------------|
| **Qwen 2.5 7B** | bf16 LoRA | 4 | ~8-10 hours | Best next step. Easy, no quantization needed. |
| **Qwen 2.5 14B** | bf16 LoRA | 2 | ~18-24 hours | Good. Reduce batch_size to 2, enable grad_checkpoint. |
| **Qwen 2.5 14B** | 4-bit QLoRA | 4 | ~14-18 hours | Better throughput than bf16 at 14B. |
| **Qwen 2.5 32B** | 4-bit QLoRA | 1-2 | ~36-48 hours | Feasible but slow. Maximum local model. |
| **Qwen 2.5 72B** | 4-bit QLoRA | 1 | ~4-6 days | Technically possible, not practical for iteration. |

*Estimates based on scaling from the current 3B ~5 hour runtime. Training time scales roughly linearly with model size and inversely with batch size.

### MLX-Specific Notes

- `mlx_lm.lora` supports `--q-bits 4` to train with quantized base weights
- Pre-quantized MLX models (e.g., `mlx-community/Qwen2.5-32B-Instruct-4bit`) can be loaded directly
- Gradient checkpointing (`--grad-checkpoint`) is already enabled in your pipeline and is essential for 14B+
- MLX memory is lazy-allocated — the system won't OOM until the compute graph is evaluated, so you may need to watch for sudden crashes rather than gradual memory warnings

---

## 2. What Model Sizes Require Cloud GPUs?

### When to Move Off the Laptop

| Scenario | Stay Local | Go Cloud |
|----------|-----------|----------|
| 7B bf16 LoRA | Yes | No need |
| 14B bf16 LoRA | Yes (slower) | Optional for speed |
| 32B bf16 LoRA | No — too large | Yes |
| 32B QLoRA | Yes (slow) | Recommended for speed |
| 70B anything | Impractical | Yes |
| Full fine-tuning any size | No | Yes |
| Fast iteration (many experiments) | Only for 3B/7B | Yes for 14B+ |

**Rule of thumb:** Stay local for 7B and below. Use cloud for 32B+, or when you want fast iteration at 14B.

### Cloud GPU Options

#### Tier 1: Best Value for LoRA/QLoRA Training

| Provider | GPU | VRAM | Hourly Cost | Best For |
|----------|-----|------|-------------|----------|
| **RunPod** | A100 80GB | 80 GB | $1.64/hr | 70B QLoRA, 32B bf16 |
| **RunPod** | A6000 48GB | 48 GB | $0.76/hr | 14B-32B QLoRA |
| **vast.ai** | A100 80GB | 80 GB | $1.20-1.80/hr | Cheapest A100s (spot) |
| **vast.ai** | RTX 4090 24GB | 24 GB | $0.30-0.50/hr | 7B-14B QLoRA |
| **Lambda** | A100 80GB | 80 GB | $1.29/hr (reserved) | Reliable, good tooling |
| **Lambda** | H100 80GB | 80 GB | $2.49/hr | Fastest single-GPU |

#### Tier 2: Major Cloud Providers (Higher Cost, More Reliability)

| Provider | GPU | Hourly Cost | Notes |
|----------|-----|-------------|-------|
| **AWS** (p4d) | A100 40GB x8 | ~$32/hr | Overkill for LoRA, good for multi-GPU |
| **GCP** | A100 80GB | ~$3.67/hr | Reliable, easy setup |
| **Azure** | A100 80GB | ~$3.67/hr | Similar to GCP |

#### Cost Estimates for This Pipeline

Assuming the SFT(1000) -> DPO(500) -> GRPO(300) pipeline:

| Model | GPU | Estimated Time | Estimated Cost |
|-------|-----|---------------|----------------|
| **14B QLoRA** | RTX 4090 (vast.ai) | ~6-8 hours | $2-4 |
| **32B QLoRA** | A6000 (RunPod) | ~12-18 hours | $9-14 |
| **70B QLoRA** | A100 80GB (RunPod) | ~24-36 hours | $39-59 |
| **70B QLoRA** | H100 (Lambda) | ~12-18 hours | $30-45 |

### NVIDIA Frameworks (Replacing MLX for Cloud)

Your MLX scripts would need to be rewritten for NVIDIA GPUs. Here are the frameworks ranked by ease of migration from your current pipeline:

#### 1. Unsloth (Recommended for this project)
- **Why:** Drop-in replacement for the SFT+DPO+GRPO pipeline. 2-5x faster than stock HuggingFace. Native QLoRA support.
- **SFT:** `FastLanguageModel` + `SFTTrainer` from TRL
- **DPO:** `DPOTrainer` from TRL (Unsloth patches it automatically)
- **GRPO:** `GRPOTrainer` from TRL (supported since TRL 0.14+)
- **Migration effort:** Medium — rewrite configs, keep data format the same
- **Installation:** `pip install unsloth`

```python
# Example SFT with Unsloth (replaces your train_sft.py)
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # your existing JSONL loaded as HF Dataset
    args=SFTConfig(
        output_dir="outputs/sft",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        gradient_checkpointing=True,
        bf16=True,
    ),
)
trainer.train()
```

#### 2. TRL (HuggingFace's RL Training Library)
- Already in your `requirements.txt`
- Has `SFTTrainer`, `DPOTrainer`, `GRPOTrainer` built in
- Slower than Unsloth but more flexible
- Better documentation and community support

#### 3. Axolotl
- YAML-config-driven (similar to your current config approach)
- Supports SFT, DPO, RLHF out of the box
- Harder to customize reward functions (your `grpo_reward.py`)
- Best if you want a managed training pipeline

#### 4. LLaMA-Factory
- Also config-driven, supports many model families
- Good UI for experiment tracking
- Less suitable for custom GRPO reward functions

**Recommendation:** Use **Unsloth + TRL** for cloud training. It maps closest to your existing pipeline structure and supports all three stages (SFT, DPO, GRPO).

---

## 3. Model Candidates for Tax Law Fine-Tuning

### Qwen 2.5 Family (Recommended Primary Choice)

| Model | Parameters | Why Consider | Why Not |
|-------|-----------|--------------|---------|
| **Qwen 2.5 7B Instruct** | 7.6B | Direct upgrade from 3B. Same tokenizer/architecture. Zero migration friction. | Still relatively small. |
| **Qwen 2.5 14B Instruct** | 14.7B | Sweet spot for local training. Strong reasoning. | ~3-4x slower than 7B. |
| **Qwen 2.5 32B Instruct** | 32.5B | Excellent reasoning. Good at following complex instructions. | Requires QLoRA locally. |
| **Qwen 2.5 72B Instruct** | 72.7B | Best in the Qwen family. Near-frontier for open models. | Cloud-only for training. |

**Why Qwen is the best choice for this project:**
- Your data pipeline, tokenizer handling, and prompt formatting already work with Qwen
- Qwen 2.5 models have strong instruction-following, which matters for legal Q&A
- The Qwen 2.5 family was trained with substantial multilingual legal and regulatory text
- Continuity — you can compare results directly across sizes

### Llama 3.1/3.3 Family

| Model | Parameters | Strengths | Weaknesses |
|-------|-----------|-----------|------------|
| **Llama 3.1 8B Instruct** | 8B | Strong general capabilities. Huge community. | Different tokenizer — need to reformat data. |
| **Llama 3.3 70B Instruct** | 70B | Excellent reasoning and instruction following. | Cloud-only. Different prompt format. |

**Notes:**
- Llama 3.3 70B is competitive with Qwen 2.5 72B on most benchmarks
- Requires switching from ChatML to Llama prompt format
- Good alternative if Qwen results plateau

### Mistral/Mixtral

| Model | Parameters | Strengths | Weaknesses |
|-------|-----------|-----------|------------|
| **Mistral 7B Instruct v0.3** | 7B | Fast, efficient. Good sliding window attention. | Outperformed by Qwen 2.5 7B on most benchmarks. |
| **Mixtral 8x7B** | 46.7B (12.9B active) | MoE — fast inference. | Harder to LoRA train (which experts to target?). MoE + LoRA is less well-studied. |
| **Mistral Large 2** | 123B | Very capable. | Way too large for local. Expensive cloud. |

**Recommendation:** Skip Mistral/Mixtral. Qwen 2.5 and Llama 3 are both stronger at equivalent sizes.

### DeepSeek Models

| Model | Parameters | Strengths | Weaknesses |
|-------|-----------|-----------|------------|
| **DeepSeek-V3** | 671B (37B active) | MoE, very strong reasoning. | Too large. MoE LoRA is complicated. |
| **DeepSeek-R1-Distill-Qwen-32B** | 32B | Strong reasoning via R1 distillation. | Optimized for chain-of-thought, not direct answers. |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | R1 reasoning in small package. | CoT may produce overly verbose tax answers. |

**Notes on DeepSeek-R1 distilled models:** These are interesting because their chain-of-thought reasoning could help with complex tax scenarios (multi-step calculations, interacting provisions). However, the verbose thinking format may not be ideal for a tax Q&A system. Worth experimenting with the 7B distill as a comparison.

### Ranking for Tax Law Domain

**Best candidates, in order:**

1. **Qwen 2.5 14B Instruct** — Best balance of quality, local trainability, and migration effort. Train locally with QLoRA.
2. **Qwen 2.5 32B Instruct** — If 14B quality is insufficient. Train locally with 4-bit QLoRA (slow) or on cloud A6000.
3. **Qwen 2.5 7B Instruct** — Quick experiment to validate that scaling helps before committing to larger models.
4. **Qwen 2.5 72B Instruct** — Cloud training. Use when you need the best possible quality.
5. **Llama 3.3 70B Instruct** — Alternative to Qwen 72B if you hit quality ceilings.

### Why Base Models Matter Less Than Size for Legal Fine-Tuning

For domain-specific tasks like tax law:
- **Model size matters more than the base model choice** among top-tier options (Qwen, Llama, Mistral are all trained on similar data mixtures)
- **Instruction-tuned models are better starting points** than base models for this pipeline (your SFT data is instruction-formatted)
- **The fine-tuning data quality is the biggest lever** — the same pipeline on a 14B model with better data will beat a 72B model with the same data

---

## 4. Pipeline Changes for Larger Models

### Data: Mostly Keep It the Same

Your current data pipeline (parse_irc.py -> generate_training_data.py -> split_data.py) produces instruction-formatted JSONL. This works for any model size. However:

**For 7B-14B (same data, adjust format):**
- Keep the same data volume and format
- If switching from Qwen to Llama, update the chat template in data generation
- No need for more data — your existing training set is sufficient for LoRA

**For 32B-72B (consider enriching data):**
- Larger models can absorb more data without overfitting
- Consider increasing SFT data from current volume to 5,000-10,000 examples
- Add more complex multi-step tax scenarios to the GRPO prompts
- Add cross-reference questions (e.g., "How does Section 162 interact with Section 274?")
- DPO data should include harder preference pairs — the 72B base model will already get easy questions right

### Hyperparameter Adjustments

#### SFT Stage

| Parameter | 3B (current) | 7B | 14B | 32B | 70B+ |
|-----------|-------------|-----|-----|-----|------|
| `learning_rate` | 1e-5 | 1e-5 | 5e-6 | 2e-6 | 1e-6 |
| `batch_size` | 4 | 4 | 2 | 1 | 1 (or accumulate) |
| `lora_rank` | 32 | 32 | 64 | 64 | 64 |
| `lora_layers` | 16 | 16 | 24 | 32 | all |
| `iters` | 1000 | 1000 | 800 | 600 | 500 |
| `max_seq_length` | 2048 | 2048 | 2048 | 2048 | 2048 |

**Key principles:**
- **Lower learning rate for larger models.** Larger models need smaller updates to avoid catastrophic forgetting.
- **Increase LoRA rank for larger models.** More parameters = more capacity to adapt, but rank 64 is usually sufficient.
- **Apply LoRA to more layers for larger models.** For 32B+, apply to all attention layers.
- **Fewer iterations for larger models.** They converge faster because they start with better representations.

#### DPO Stage

| Parameter | 3B (current) | 7B | 14B | 32B+ |
|-----------|-------------|-----|-----|------|
| `learning_rate` | 5e-6 | 5e-6 | 2e-6 | 1e-6 |
| `beta` | 0.1 | 0.1 | 0.1 | 0.05 |
| `batch_size` | 2 | 2 | 1 | 1 |
| `iters` | 500 | 500 | 400 | 300 |

**Note on beta:** For larger models, consider lowering beta (0.05) because the base model is already closer to the preferred behavior. A lower beta lets the model deviate more from the reference during DPO.

#### GRPO Stage

| Parameter | 3B (current) | 7B | 14B | 32B+ |
|-----------|-------------|-----|-----|------|
| `learning_rate` | 1e-6 | 1e-6 | 5e-7 | 2e-7 |
| `group_size` | 4 | 4 | 4 | 2 |
| `epsilon_clip` | 0.2 | 0.2 | 0.15 | 0.1 |
| `kl_coeff` | 0.01 | 0.01 | 0.02 | 0.05 |
| `iters` | 300 | 300 | 200 | 150 |
| `temperature` | 0.8 | 0.8 | 0.7 | 0.7 |

**Key changes for GRPO at scale:**
- **Reduce group_size to 2 for 32B+** — generating 4 completions at 32B is very slow and memory-heavy
- **Tighter clipping (lower epsilon_clip)** — larger models need smaller policy updates
- **Higher KL coefficient** — penalize drift more aggressively since the base model is already good
- **Lower temperature** — larger models produce more coherent outputs already; high temperature adds noise

### Memory Optimization Techniques

#### For MLX (Local Training)

1. **Gradient checkpointing** — Already enabled in your pipeline (`--grad-checkpoint`). Essential for 14B+.
2. **Reduce max_seq_length** — For DPO/GRPO stages, 1024 tokens is usually sufficient for tax Q&A.
3. **Batch size 1 + gradient accumulation** — MLX doesn't have built-in gradient accumulation, but you can implement it manually:

```python
# Manual gradient accumulation for MLX
accumulated_grads = None
for micro_step in range(accumulation_steps):
    loss, grads = loss_and_grad(model, batch)
    if accumulated_grads is None:
        accumulated_grads = grads
    else:
        accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)
# Average and apply
accumulated_grads = tree_map(lambda g: g / accumulation_steps, accumulated_grads)
optimizer.update(model, accumulated_grads)
```

4. **Use pre-quantized models** — Load 4-bit models from `mlx-community` on HuggingFace rather than quantizing yourself.

#### For NVIDIA (Cloud Training)

1. **Gradient checkpointing** — Enabled by default in Unsloth/TRL
2. **DeepSpeed ZeRO Stage 2** — For multi-GPU setups. Shards optimizer states across GPUs.
3. **Flash Attention 2** — Automatically used by Unsloth. Reduces memory for long sequences.
4. **Gradient accumulation** — Native in TRL: `gradient_accumulation_steps=8` with `per_device_train_batch_size=1` gives effective batch of 8.

### Code Changes Required

#### Minimal Path: Stay on MLX, Swap Model

For 7B and 14B on the M4 Max, the changes are minimal:

1. **Download the new model:**
```bash
# For Qwen 2.5 14B with 4-bit quantization
pip install huggingface_hub
huggingface-cli download mlx-community/Qwen2.5-14B-Instruct-4bit --local-dir models/qwen25-14b-mlx
```

2. **Update config files** — change model paths and hyperparameters:
```yaml
# configs/sft_config.yaml
model: models/qwen25-14b-mlx
batch_size: 2
learning_rate: 5.0e-6
lora_rank: 64
lora_layers: 24
```

3. **Update script paths** — change `MODEL_MLX` and `MODEL_HF` in each training script, or better yet, make them config-driven.

#### Cloud Path: Rewrite for Unsloth/TRL

For 32B+ on cloud GPUs, you need new training scripts. The data format stays the same (JSONL), but the training code changes entirely. Your custom `grpo_reward.py` can be used directly with TRL's `GRPOTrainer` since it accepts a reward function.

```python
# Example: Using your existing reward function with TRL GRPOTrainer
from trl import GRPOTrainer, GRPOConfig
from scripts.grpo_reward import compute_reward

def reward_fn(prompts, completions, **kwargs):
    return [compute_reward(p, c) for p, c in zip(prompts, completions)]

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=GRPOConfig(
        output_dir="outputs/grpo",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=2e-7,
        num_generation_per_prompt=4,  # equivalent to group_size
    ),
    train_dataset=grpo_dataset,
)
```

---

## 5. Expected Quality Improvements

### Model Size vs. Tax Law Accuracy

Based on published scaling law research and domain-specific fine-tuning studies:

#### General Pattern

| Model Size | Base Model Quality (No Fine-Tuning) | After SFT+DPO+GRPO Pipeline |
|-----------|--------------------------------------|------------------------------|
| 3B (current) | Gets basic concepts, hallucinates details frequently | Improved citation accuracy, but still fabricates section numbers |
| 7B | Better structured answers, fewer gross errors | Significantly fewer hallucinated citations. Better at simple cross-references. |
| 14B | Competent on single-topic questions | Reliable for straightforward tax questions. Accurate citations for common sections. |
| 32B | Good reasoning, handles multi-step logic | Can handle interacting provisions (e.g., AMT + credits). Rare hallucinations on common topics. |
| 70B+ | Near-expert on common scenarios | Handles complex scenarios (corporate reorganizations, international tax). Hallucinations mostly limited to obscure provisions. |

#### Where Hallucinations Decrease

Research on legal domain LLMs (e.g., LegalBench, MMLU Professional Law) shows:

- **3B -> 7B:** ~15-20% reduction in factual errors. The model stops making up entirely fake section numbers but still confuses similar sections.
- **7B -> 14B:** ~25-30% reduction. The model starts correctly distinguishing between related but different provisions (e.g., Section 162 vs. 212).
- **14B -> 32B:** ~15-20% reduction. Diminishing returns on simple factual recall, but major improvement on **reasoning** tasks (multi-step calculations, applying exceptions).
- **32B -> 70B:** ~10-15% reduction. The improvement is mainly on **edge cases** and **interactions** between provisions.

**The biggest jump is 3B -> 14B.** Going from 3B to 14B roughly halves the hallucination rate on tax law questions.

#### Specific Improvements to Expect

**At 7B (your likely next step):**
- Correct standard deduction amounts by filing status
- Accurate citation of common IRC sections (1, 61, 63, 162, 170, 401, 501)
- Better at "does X qualify as Y" questions
- Still struggles with multi-year phase-outs and complex calculations

**At 14B:**
- Handles most individual tax scenarios correctly
- Can explain interactions between common provisions
- Better at business tax concepts (Section 199A, depreciation rules)
- Can distinguish between similar provisions (Section 351 vs 368)

**At 32B:**
- Handles corporate tax, partnership tax, international tax basics
- Can work through multi-step calculations with proper intermediate steps
- Understands temporal rules (when provisions took effect, sunset clauses)
- Reliable enough for a first-draft research tool

**At 70B:**
- Handles complex multi-entity scenarios
- Can reason about tax planning implications
- Understands regulatory hierarchy (statute vs. regulation vs. revenue ruling)
- Competitive with GPT-4 on common tax questions after domain fine-tuning

### The Data Quality Lever

It's worth emphasizing: **improving your training data at 3B may yield more benefit than scaling to 7B with the same data.**

Before scaling the model, consider:

1. **Audit your SFT data** — Are the reference answers actually correct? Have a tax professional spot-check 50 examples.
2. **Improve DPO pairs** — The quality of your preference pairs determines the ceiling of DPO. Generate harder pairs where both responses are plausible but one is legally precise.
3. **Expand GRPO prompts** — Add prompts that specifically test the failure modes you're seeing (e.g., section number hallucination, incorrect thresholds).
4. **Refine the reward function** — Your `grpo_reward.py` currently checks for citation presence and legal term usage. Consider adding:
   - Verification against a known list of valid IRC section numbers
   - Checking that cited sections are actually relevant to the question
   - Penalizing contradictions within the response

---

## 6. Recommended Scaling Path

### Phase 1: Quick Win (1-2 days)

**Goal:** Validate that scaling helps before investing in infrastructure.

1. Download `mlx-community/Qwen2.5-7B-Instruct-4bit` (or bf16 if you prefer)
2. Update config paths and hyperparameters per the table above
3. Run the full SFT -> DPO -> GRPO pipeline
4. Compare against 3B results using the same evaluation prompts
5. **Expected result:** Noticeable improvement in citation accuracy and fewer hallucinations

### Phase 2: Local Maximum (1 week)

**Goal:** Get the best model that trains comfortably on M4 Max.

1. Train Qwen 2.5 14B Instruct with 4-bit QLoRA
2. Increase LoRA rank to 64, apply to more layers
3. If data quality auditing reveals issues, fix data first
4. Run comprehensive evaluation
5. **Expected result:** The model becomes reliable for common individual tax questions

### Phase 3: Cloud Scale (2-3 weeks)

**Goal:** Push toward maximum quality.

1. Set up Unsloth + TRL on a RunPod A100
2. Port the training scripts (keep data pipeline, rewrite training loops)
3. Train Qwen 2.5 72B with the full pipeline
4. Evaluate against 14B to measure the improvement
5. **Expected result:** Handles complex tax scenarios, competitive with GPT-4 on domain questions

### Phase 4: Production (if needed)

**Goal:** Deploy the best model efficiently.

1. Export the best model (14B or 72B) with merged LoRA weights
2. Quantize to 4-bit GGUF for fast inference via Ollama
3. The 14B model can serve inference locally on M4 Max at ~30-40 tokens/sec
4. The 72B model needs a cloud GPU for serving, or can run on M4 Max at ~5-8 tokens/sec with 4-bit quantization

---

## Summary Decision Matrix

| If you want... | Do this |
|----------------|---------|
| Quick improvement, minimal effort | Train Qwen 2.5 7B on M4 Max with bf16 LoRA |
| Best local model | Train Qwen 2.5 14B on M4 Max with 4-bit QLoRA |
| Maximum quality, don't mind cloud cost | Train Qwen 2.5 72B on A100 with Unsloth |
| Fast iteration on experiments | Stay at 7B locally, run many variants |
| Best quality per dollar | Train 14B locally (free) then evaluate before spending on cloud |
