# RL and Post-Training for Small Language Models (1B-8B) - Research Report

**Date:** 2025-03-25
**Scope:** Practical approaches for fine-tuning and RL on small LLMs that can run on Ollama, with a focus on domain-specific knowledge injection (tax code).

---

## 1. RL Training Approaches for Small LLMs (2025-2026)

### Overview of Methods

| Method | Type | Requires Reward Model? | Memory Footprint | Best For |
|--------|------|----------------------|------------------|----------|
| **SFT** | Supervised | No | Low | Knowledge injection, format compliance |
| **DPO** | Offline preference | No (uses preference pairs) | Medium | Aligning style/tone without a reward model |
| **ORPO** | Combined SFT+preference | No | Medium | Single-stage alignment (newer, promising) |
| **GRPO** | RL (group relative) | Yes (can be rule-based) | Medium-High | Reasoning improvement, verifiable tasks |
| **PPO** | RL (on-policy) | Yes | High | Maximum control, but complex and resource-heavy |
| **SimPO** | Reference-free preference | No | Medium | Simpler DPO alternative, no reference model needed |
| **KTO** | Binary signal preference | No | Medium | When you only have thumbs-up/down data |

### What's Actually Working in 2025-2026

**GRPO (Group Relative Policy Optimization)** has emerged as the dominant RL method for small models after DeepSeek-R1's success. Key advantages:
- No separate reward model needed (uses group-based relative scoring)
- Works well with rule-based/verifiable rewards (exact match, code execution, regex checks)
- Significantly lower memory than PPO
- TRL has first-class support via `GRPOTrainer`
- The "DeepSeek-R1-Zero" approach showed even small models can develop chain-of-thought reasoning via GRPO alone

**DPO remains the pragmatic default** for preference alignment:
- Simple to implement (just needs preference pairs: chosen vs rejected)
- No reward model training step
- Well-supported across all frameworks
- Works well for style/format alignment after SFT

**ORPO** is gaining traction as a single-stage alternative that combines SFT and preference optimization in one pass, reducing training time.

**PPO is largely being abandoned** for small-model work due to:
- Requires training and hosting a separate reward model
- High memory overhead (4 models in memory: policy, reference, reward, value)
- Unstable training dynamics
- GRPO achieves comparable or better results with less complexity

### Recommendation for Tax Code Domain

For injecting IRS tax code knowledge into a small LLM:

1. **SFT first** (primary knowledge injection)
2. **DPO or GRPO second** (alignment/reasoning improvement)

More detail in Section 6.

---

## 2. Tools and Frameworks

### Tier 1: Production-Ready, Actively Maintained

#### **Unsloth**
- **Best for:** Local single-GPU fine-tuning of 1B-8B models
- **Key features:** 2x faster training, 60-80% less memory via kernel optimizations, QLoRA/LoRA with 4-bit quantization
- **Supports:** SFT, DPO, ORPO, GRPO (added late 2024/early 2025)
- **Models:** Llama 3.x, Mistral, Qwen 2.5, Gemma 2, Phi-3/4
- **Why it wins for local:** Specifically optimized for consumer GPUs. Can fine-tune 8B models on 16GB VRAM
- **Install:** `pip install unsloth`
- **Export:** Native GGUF export built-in (`model.save_pretrained_gguf()`)

#### **TRL (Transformer Reinforcement Learning by HuggingFace)**
- **Best for:** Reference implementation, full method coverage
- **Key features:** `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `ORPOTrainer`, `KTOTrainer`
- **Supports:** Every major alignment method
- **Integration:** Works with PEFT/LoRA, DeepSpeed, FSDP, quantization
- **Note:** Unsloth patches TRL trainers for speed, so you often use both together
- **Install:** `pip install trl`

#### **LLaMA-Factory**
- **Best for:** No-code/low-code fine-tuning with a web UI
- **Key features:** Web UI (LlamaBoard), 100+ model support, all major methods
- **Supports:** SFT, DPO, ORPO, PPO, KTO, GRPO
- **Unique:** Built-in dataset management, evaluation, and export
- **Good for:** Rapid experimentation without writing training scripts
- **Install:** `git clone https://github.com/hiyouga/LLaMA-Factory && pip install -e .`

#### **Axolotl**
- **Best for:** Config-driven training with YAML files
- **Key features:** Extensive YAML configuration, multi-GPU support, many dataset formats
- **Supports:** SFT, DPO, RLHF, LoRA, QLoRA
- **Strength:** Excellent for reproducible experiments
- **Note:** Heavier setup than Unsloth, more suited for multi-GPU

### Tier 2: Specialized / Advanced

#### **OpenRLHF**
- **Best for:** Distributed RL training (PPO, GRPO) across multiple GPUs/nodes
- **Not ideal for local single-GPU work** -- designed for Ray-based clusters
- **Use case:** If you scale up to cloud training later

#### **veRL (Volcano Engine RL)**
- **Best for:** Large-scale GRPO/PPO with efficient rollout generation
- **Similar to OpenRLHF** -- cluster-oriented

#### **torchtune (by Meta/PyTorch)**
- **Best for:** PyTorch-native fine-tuning with minimal abstractions
- **Supports:** SFT, DPO, LoRA, QLoRA
- **Strength:** Clean, educational codebase; good for understanding internals
- **Limitation:** Fewer RL methods than TRL

#### **MLX (Apple Silicon)**
- **Best for:** Fine-tuning directly on Mac with Apple Silicon
- **Framework:** `mlx-lm` for inference, `mlx` for training
- **Supports:** LoRA/QLoRA SFT
- **Limitation:** RL methods not yet well-supported
- **Note:** If you're on an M-series Mac, this is worth watching but not yet mature for RL

### Recommendation

**Unsloth + TRL** is the clear winner for local single-GPU fine-tuning:
- Fastest training speeds on consumer hardware
- Lowest memory usage
- Built-in GGUF export for Ollama
- Covers SFT, DPO, ORPO, GRPO

---

## 3. Most Practical Pipeline for Local Machine (No Cluster)

### Hardware Requirements

| Model Size | Minimum VRAM (QLoRA) | Recommended VRAM | Notes |
|-----------|----------------------|------------------|-------|
| 1B-3B | 8GB | 12GB+ | Comfortable on most modern GPUs |
| 7B-8B | 12GB | 16-24GB | RTX 3090/4090 or A5000 ideal |
| 7B-8B (full) | 40GB+ | 48GB+ | Only A6000/A100 or multi-GPU |

**For Mac (Apple Silicon):**
- M1 Pro/Max (32GB+): Can handle 7B QLoRA via MLX or Unsloth (with MPS backend)
- M2/M3 Ultra (64GB+): Can handle 8B comfortably
- Unified memory is your friend -- but training is slower than NVIDIA GPUs
- Unsloth has experimental Apple Silicon support; MLX is native

**For NVIDIA GPUs:**
- RTX 3090 (24GB) or RTX 4090 (24GB): Sweet spot for 7B-8B QLoRA
- RTX 4070/4080 (12-16GB): Fine for 1B-3B, tight for 7B

### The Pipeline

```
1. Prepare Dataset
   ├── Collect domain data (tax code sections, Q&A pairs, examples)
   ├── Format as JSONL (instruction/input/output or chat format)
   └── Split train/eval (90/10)

2. Download Base Model (HuggingFace weights)
   ├── Choose: Llama 3.2 3B, Qwen 2.5 7B, Mistral 7B v0.3, Phi-4
   └── Download: `huggingface-cli download <model-id>`

3. SFT with QLoRA (Unsloth)
   ├── Load 4-bit quantized model
   ├── Apply LoRA adapters (rank 16-64)
   ├── Train on domain data
   ├── Typical: 1-3 epochs, 2-8 hours on RTX 4090
   └── Save LoRA adapter + merge

4. (Optional) RL/Preference Alignment
   ├── Prepare preference dataset (chosen/rejected pairs)
   ├── Run DPO or GRPO on the SFT checkpoint
   └── Typical: 1 epoch, 1-4 hours

5. Export to GGUF
   ├── Unsloth: `model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")`
   ├── Or: llama.cpp `convert_hf_to_gguf.py` + `llama-quantize`
   └── Choose quantization: Q4_K_M (balanced), Q5_K_M (higher quality), Q8_0 (best quality)

6. Import to Ollama
   ├── Create Modelfile with: FROM ./model-q4_k_m.gguf
   ├── Add system prompt, parameters, template
   └── `ollama create my-tax-model -f Modelfile`

7. Test and Iterate
   ├── Run eval prompts through Ollama
   ├── Compare against base model
   └── Adjust data/hyperparameters and retrain
```

---

## 4. Model Formats: GGUF vs HuggingFace Weights

### Can You Fine-Tune GGUF Models Directly?

**No, not practically.** Here's why:

- GGUF is a **quantized inference format** designed for llama.cpp/Ollama
- Quantization destroys the gradient information needed for backpropagation
- Training requires full-precision (FP16/BF16) or carefully managed quantization (QLoRA)
- There is no mainstream framework that trains GGUF files directly

### What About llama.cpp's `finetune` Command?

- llama.cpp has an experimental `finetune` tool that works on GGUF
- It's extremely limited: only basic LoRA, no RL methods, no advanced features
- Quality is significantly worse than proper QLoRA via Unsloth/TRL
- **Not recommended** for serious work

### Required Workflow

You **must** start from HuggingFace-format weights:

```
HuggingFace Safetensors (FP16/BF16)
    ↓ (fine-tune with QLoRA)
Fine-tuned HF model (merged LoRA + base)
    ↓ (convert)
GGUF file (quantized for inference)
    ↓ (import)
Ollama model
```

### Important Notes

- **Ollama models are GGUF files** -- when you `ollama pull llama3.2`, it downloads a GGUF
- **You cannot fine-tune what Ollama gives you** -- you need the original HF weights
- **Most models on HuggingFace have both formats** -- look for the non-GGUF version
- **LoRA adapters are small** (50-500MB) vs full model weights (4-16GB), so you can experiment with many configurations cheaply

---

## 5. Recommended Workflow: HF Weights to Ollama

### Step-by-Step

#### Step 1: Choose Your Base Model

Best candidates for tax code domain (as of early 2026):

| Model | Size | Why |
|-------|------|-----|
| **Qwen 2.5 7B Instruct** | 7B | Excellent reasoning, strong at structured data |
| **Llama 3.2 3B Instruct** | 3B | Great if VRAM-constrained, surprisingly capable |
| **Mistral 7B v0.3 Instruct** | 7B | Good general capability, well-tested |
| **Phi-4** | 14B | If you have 24GB+ VRAM, exceptional reasoning for size |
| **Gemma 2 9B** | 9B | Strong at following instructions |

#### Step 2: Download HF Weights

```bash
# Install huggingface CLI
pip install huggingface-hub

# Login (needed for gated models like Llama)
huggingface-cli login

# Download model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen2.5-7b
```

#### Step 3: Fine-Tune with Unsloth

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_dataset

# Load model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # saves 60% memory
)

# Load your tax code dataset
dataset = load_dataset("json", data_files="tax_code_sft.jsonl")

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    max_seq_length=4096,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=10,
    output_dir="outputs",
)
trainer.train()
```

#### Step 4: Export to GGUF

```python
# Save merged model as GGUF (multiple quantization options)
model.save_pretrained_gguf(
    "tax-model-gguf",
    tokenizer,
    quantization_method="q4_k_m"  # Good balance of size/quality
)
# Also save q5_k_m and q8_0 for comparison
```

#### Step 5: Create Ollama Model

```dockerfile
# Modelfile
FROM ./tax-model-gguf/unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """You are a tax code expert assistant. You provide accurate information about IRS tax regulations, citing specific sections of the Internal Revenue Code when applicable. Always note when tax rules have specific effective dates or sunset provisions."""

PARAMETER temperature 0.3
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
```

```bash
ollama create tax-expert -f Modelfile
ollama run tax-expert
```

---

## 6. SFT vs RL for Domain-Specific Knowledge Injection (Tax Code)

### Short Answer

**SFT first, then optionally RL.** For domain knowledge injection, SFT is the primary tool. RL is for behavioral alignment, not knowledge.

### Why SFT Is Primary for Knowledge Injection

- **SFT directly teaches the model new factual content** -- tax code sections, regulations, interpretations
- **RL cannot inject knowledge** -- it can only steer the model toward using knowledge it already has (or learned via SFT) in better ways
- **SFT is simpler, faster, and more predictable** for factual domains
- **The model needs to "know" something before RL can help it "reason" about it**

### When RL Adds Value (After SFT)

RL/preference optimization is valuable for:

1. **Reducing hallucination:** DPO with (correct answer, hallucinated answer) pairs teaches the model to say "I'm not sure" rather than inventing tax rules
2. **Improving citation behavior:** GRPO with rule-based rewards for citing specific IRC sections
3. **Reasoning chains:** GRPO can improve step-by-step tax calculation reasoning
4. **Format compliance:** DPO for preferred output structure (e.g., always starting with the relevant IRC section)

### Recommended Training Pipeline for Tax Code

```
Phase 1: Data Preparation
├── Curate tax code sections as instruction/completion pairs
├── Create Q&A pairs from IRS publications
├── Format tax scenarios with step-by-step solutions
└── Target: 5,000-50,000 high-quality examples

Phase 2: SFT (Knowledge Injection)
├── QLoRA fine-tuning on curated dataset
├── 1-3 epochs
├── Evaluate on held-out tax questions
└── This is where 80% of the improvement happens

Phase 3: DPO/GRPO (Behavioral Alignment) -- Optional but recommended
├── Generate responses from SFT model to tax questions
├── Have expert rate or create preference pairs
├── Train DPO: (correct+cited answer) > (vague/wrong answer)
├── Or GRPO with rules: +reward for citing IRC sections, -reward for hedging on known facts
└── 1 epoch, lighter training

Phase 4: Export and Deploy
├── Merge LoRA → GGUF → Ollama
└── Continuous eval against tax question benchmark
```

### Data Format Examples

**SFT Training Data (JSONL):**
```json
{
  "instruction": "What is the standard deduction for a single filer in tax year 2025?",
  "input": "",
  "output": "For tax year 2025, the standard deduction for a single filer is $15,000. This is set by IRC Section 63(c) and is adjusted annually for inflation per IRC Section 63(c)(4). Note that taxpayers who are 65 or older or blind receive an additional standard deduction amount of $1,950 (single) or $1,550 (married) per IRC Section 63(f)."
}
```

**DPO Preference Data (JSONL):**
```json
{
  "prompt": "Can I deduct home office expenses?",
  "chosen": "Home office deductions are governed by IRC Section 280A. You may deduct home office expenses if you use a portion of your home regularly and exclusively as your principal place of business (Section 280A(c)(1)(A)). The simplified method allows a deduction of $5 per square foot, up to 300 square feet ($1,500 maximum). The regular method requires calculating actual expenses proportional to the business-use percentage of your home. Note: W-2 employees generally cannot claim this deduction after the Tax Cuts and Jobs Act of 2017 suspended the unreimbursed employee expense deduction through 2025.",
  "rejected": "Yes, you can probably deduct your home office. Just calculate the square footage and multiply by $5. Make sure you have a dedicated space."
}
```

---

## 7. Realistic Expectations for Local Fine-Tuning

### Training Time Estimates

| Scenario | GPU | Model | Dataset | Time |
|----------|-----|-------|---------|------|
| SFT QLoRA | RTX 4090 (24GB) | 7B | 10K examples | 1-2 hours |
| SFT QLoRA | RTX 4090 (24GB) | 7B | 50K examples | 4-8 hours |
| SFT QLoRA | RTX 3090 (24GB) | 7B | 10K examples | 2-4 hours |
| SFT QLoRA | M2 Max (32GB) | 7B | 10K examples | 4-8 hours (MLX) |
| DPO QLoRA | RTX 4090 (24GB) | 7B | 5K pairs | 1-3 hours |
| GRPO QLoRA | RTX 4090 (24GB) | 7B | 5K examples | 3-8 hours |
| SFT QLoRA | RTX 4090 (24GB) | 3B | 10K examples | 30-60 min |

### Quality Expectations

**What SFT on a 7B model can realistically achieve:**
- Accurate recall of specific tax rules and thresholds when trained on them
- Proper citation of IRC sections present in training data
- Improved formatting and structure for tax-related responses
- Good performance on "in-distribution" questions (similar to training data)

**What SFT on a 7B model cannot do:**
- Reliably generalize to novel tax scenarios not in training data
- Replace a tax professional for complex situations
- Maintain perfect accuracy across the entire tax code (too vast)
- Perform multi-step tax calculations with 100% accuracy (reasoning limitations)

**Realistic quality improvement:**
- Base model might score 30-50% on domain-specific tax questions
- After SFT: expect 60-80% accuracy on in-distribution questions
- After SFT + DPO: expect better calibration (knows when it doesn't know)
- A 7B model will never match GPT-4/Claude on complex reasoning, but can be very good at lookup-style questions and standard scenarios

### Cost Comparison

| Approach | Cost | Time to First Result |
|----------|------|---------------------|
| Local (existing GPU) | Electricity only | 2-8 hours |
| Cloud (1x A100 spot) | $1-3/hour | 1-4 hours |
| Cloud (Lambda/RunPod) | $1-2/hour | 1-4 hours |
| Unsloth Colab (free) | Free (limited) | 2-4 hours (with interruptions) |

### Common Pitfalls

1. **Too little data:** < 1,000 examples often leads to overfitting or minimal improvement
2. **Too many epochs:** More than 3 epochs on small datasets causes memorization
3. **Wrong learning rate:** Too high (> 5e-4) destroys base model capabilities; too low (< 1e-5) barely learns
4. **Ignoring eval:** Always hold out 10% for evaluation; watch for loss going up
5. **Bad data quality:** 1,000 expert-curated examples >> 50,000 noisy/generated examples
6. **Forgetting the template:** Must match the base model's chat template exactly or outputs will be garbled
7. **Skipping the merge:** LoRA adapters must be merged into base weights before GGUF conversion

### Iteration Cycle

A realistic development loop:

```
Week 1: Data preparation (most important and time-consuming step)
  - Collect/curate 5-10K training examples
  - Format and validate
  - Create eval set of 200-500 questions with known answers

Week 2: First training run
  - SFT with Unsloth, evaluate, identify gaps
  - Iterate on data quality based on failure modes

Week 3: Refinement
  - Add more examples for weak areas
  - Try DPO for hallucination reduction
  - A/B test different base models (Qwen vs Llama vs Mistral)

Week 4: Deployment
  - Export best model to GGUF
  - Set up Ollama with proper system prompt
  - Build evaluation harness for ongoing monitoring
```

---

## Summary and Recommendations

### For the Tax Code Project Specifically:

1. **Start with Qwen 2.5 7B Instruct** as the base model (best reasoning in the 7B class)
2. **Use Unsloth + TRL** for training (fastest, lowest memory, built-in GGUF export)
3. **Focus 80% of effort on data quality** -- curate high-quality tax Q&A pairs with IRC citations
4. **Do SFT first** with QLoRA (r=32, 4-bit) for 2-3 epochs
5. **Follow with DPO** using preference pairs to reduce hallucination
6. **Export to GGUF Q5_K_M** (good quality/size balance for tax accuracy)
7. **Import to Ollama** with a strong system prompt
8. **Expect 1-2 days** for the full pipeline from data to running model
9. **Budget 1-2 weeks** for data preparation (the bottleneck)

### Key Takeaway

The bottleneck is **data quality, not compute or algorithms**. A well-curated dataset of 10K tax code examples trained via straightforward SFT will outperform a poorly curated dataset of 100K examples trained with the fanciest RL method. Start simple, iterate on data, add RL methods only after SFT is working well.
