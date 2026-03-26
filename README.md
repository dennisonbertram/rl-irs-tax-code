# IRS Tax Code RL Training

Fine-tuning **Qwen 2.5 3B Instruct** on the IRS Tax Code (IRC Title 26 + 26 CFR Treasury Regulations) using a three-stage pipeline: **SFT → DPO → GRPO** — entirely on Apple Silicon with MLX.

The resulting model is exported to GGUF and served locally via [Ollama](https://ollama.com).

---

## Training Pipeline

```
IRC Title 26 XML          26 CFR XML
     │                       │
     └──────── parse ─────────┘
                │
         data/processed/
          ├── sft_train.jsonl     (instruction/response pairs)
          ├── dpo_train.jsonl     (chosen / rejected preference pairs)
          └── grpo_train.jsonl    (prompts with verifiable rewards)
                │
     ┌──────────┼──────────────┐
     │          │              │
  Stage 1    Stage 2        Stage 3
   SFT        DPO            GRPO
  (mlx_lm)  (TRL)          (TRL)
     │          │              │
     └──────────┴──────────────┘
                │
         outputs/final/
                │
         GGUF Q4_K_M → Ollama
```

### Stage 1 — Supervised Fine-Tuning (SFT)

- Framework: `mlx_lm.lora` (native Apple Silicon)
- LoRA rank 32, bf16, gradient checkpointing
- 1 000 gradient steps, batch size 4, lr 1e-5 with cosine decay
- Context window: 2 048 tokens

### Stage 2 — Direct Preference Optimization (DPO)

- Trains on chosen/rejected pairs derived from IRC sections
- Teaches the model to prefer precise, citation-grounded answers over vague ones

### Stage 3 — Group Relative Policy Optimization (GRPO)

- Rule-based reward signals: IRC section citation accuracy, answer completeness
- Reinforces factual grounding without a separate reward model

---

## Hardware Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| Chip | Apple Silicon (M1+) | Apple M4 Max |
| RAM | 32 GB unified | 128 GB unified |
| GPU cores | 16 Metal cores | 40-core GPU |
| Disk | 20 GB free | 41 GB free |

The pipeline is designed for **MLX on Apple Silicon**. It does not use CUDA or MPS via PyTorch — training runs natively via `mlx_lm`.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/dennisonbertram/rl-irs-tax-code.git
cd rl-irs-tax-code

# 2. Create a Python virtual environment (Python 3.11+ recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install mlx-lm          # Apple Silicon only

# 4. Install Ollama (for local inference)
brew install ollama
```

---

## Data Pipeline

The raw IRC and CFR XML files are **not included** in this repo (too large). Regenerate them:

```bash
# Download raw XML source data
bash scripts/download_data.sh

# Parse XML into section-level JSONL
python scripts/parse_irc.py
python scripts/parse_cfr.py

# Generate training datasets
python scripts/generate_training_data.py

# Split into train/eval sets
python scripts/split_data.py
```

This produces:
- `data/processed/sft_train.jsonl` (~52 MB) — instruction/response pairs
- `data/processed/dpo_train.jsonl` (~17 MB) — preference pairs
- `data/processed/grpo_train.jsonl` (~3 MB) — GRPO prompts
- `data/processed/train/` — train/valid splits for mlx_lm

---

## Training

### Download Base Model

```bash
# HuggingFace weights (mlx_lm will auto-convert on first run)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen2.5-3b-instruct
```

### Stage 1: SFT

```bash
python scripts/train_sft.py --iters 1000 --batch-size 4
# or via shell wrapper:
bash scripts/run_sft.sh
```

Adapters saved to `outputs/sft/adapters/`.

### Stage 2: DPO

```bash
python scripts/train_dpo.py
```

Adapters saved to `outputs/dpo/adapters/`.

### Stage 3: GRPO

```bash
python scripts/train_grpo.py
```

Adapters saved to `outputs/grpo/adapters/`.

### Evaluation

```bash
python scripts/evaluate.py
```

---

## Export to Ollama

```bash
# Fuse adapters, convert to GGUF Q4_K_M, and import into Ollama
python scripts/export_to_ollama.py

# Run the model locally
ollama run irs-tax-qwen
```

---

## HuggingFace Models

Trained adapters are published on HuggingFace:

- SFT adapters: [dennisonbertram/qwen2.5-3b-irs-sft](https://huggingface.co/dennisonbertram/qwen2.5-3b-irs-sft)
- GGUF (Ollama-ready): [dennisonbertram/qwen2.5-3b-irs-tax-gguf](https://huggingface.co/dennisonbertram/qwen2.5-3b-irs-tax-gguf)

---

## Project Structure

```
rl-irs-tax-code/
├── configs/
│   ├── sft_config.yaml         # MLX LoRA hyperparameters
│   ├── dpo_config.yaml         # DPO training config
│   └── grpo_config.yaml        # GRPO training config
├── scripts/
│   ├── parse_irc.py            # Parse IRC Title 26 XML
│   ├── parse_cfr.py            # Parse 26 CFR XML
│   ├── generate_training_data.py  # Build SFT/DPO/GRPO datasets
│   ├── split_data.py           # Train/eval split
│   ├── train_sft.py            # Stage 1 training (mlx_lm LoRA)
│   ├── train_dpo.py            # Stage 2 training (DPO via TRL)
│   ├── train_grpo.py           # Stage 3 training (GRPO via TRL)
│   ├── grpo_reward.py          # Reward functions for GRPO
│   ├── evaluate.py             # Evaluation harness
│   ├── export_to_ollama.py     # Fuse → GGUF → Ollama import
│   ├── setup.sh                # Environment setup helper
│   └── run_sft.sh              # SFT shell wrapper
├── docs/
│   ├── context/                # Background research
│   ├── investigations/         # Exploration notes
│   └── plans/                  # Execution plans
├── data/                       # Generated (not committed — see Data Pipeline)
│   ├── raw/                    # Downloaded XML source files
│   └── processed/              # Parsed JSONL datasets
├── models/                     # Downloaded model weights (not committed)
├── outputs/                    # Training checkpoints (not committed)
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Limitations

- **Apple Silicon only** — the MLX training path does not run on Linux/Windows or CUDA GPUs without modification. For CUDA, swap `mlx_lm` for `trl` with `SFTTrainer` on a standard GPU.
- **3B parameter model** — the model is small and will not match GPT-4-level tax analysis. It is trained to cite IRC sections accurately, not to give legal advice.
- **Not legal advice** — outputs should be verified against the official IRC and CFR before relying on them for any tax or legal purpose.
- **Data freshness** — the IRC and CFR XMLs used for training reflect a specific publication date. Tax law changes frequently.

---

## License

The training code in this repository is licensed under the **Apache 2.0 License**, matching the license of the base model (Qwen 2.5).

The IRS tax code text (IRC Title 26, 26 CFR) is a U.S. government work and is in the public domain.
