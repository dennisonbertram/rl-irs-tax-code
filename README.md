<p align="center">
  <img src="docs/assets/header.png" alt="Abstract watercolor in warm earth tones blending into soft blues" width="100%" />
</p>

# A Tax Code Tutor That Lives on Your Laptop

This project teaches a small AI model to read and understand the United States tax code — and runs it entirely on a personal computer, with no cloud, no subscription, and no data leaving the machine.

## What we're doing, in plain language

The U.S. tax code is enormous. The actual law (Title 26) plus the rules the Treasury writes to interpret it (26 CFR) run to thousands of pages of dense legal language. Most people — including most accountants — never read it directly. They read summaries of summaries.

We took a small, freely available AI model and **patiently taught it the source material itself**. Not blog posts about taxes. Not TurboTax help articles. The actual statutes and regulations, section by section.

The training happens in three rounds, each one sharpening a different skill:

1. **Reading** — we show the model the tax code and ask it to explain sections in its own words, then correct it when it's wrong.
2. **Judgment** — we show it pairs of answers (one good, one sloppy) and teach it to prefer the precise one.
3. **Citation** — we reward it for pointing back to the exact section of law it's relying on, the way a careful lawyer would.

The result is a model you can ask questions like *"what's the standard deduction for a single filer in 2024?"* or *"which IRC section governs §401(k) hardship withdrawals?"* — and it answers locally, on your machine, while citing its sources.

## Why this is interesting

- **Privacy.** Your tax questions never touch a server. The model runs offline.
- **Verifiability.** Because it cites IRC sections, you can check its work against the real law.
- **Small and fast.** It's a 3-billion-parameter model — small enough to fit on a laptop, fast enough to answer in seconds.
- **Reproducible.** Every step, from raw XML of the tax code to the final downloadable model, is in this repository.

## What it isn't

- **Not legal or tax advice.** It's a research project. Verify anything important against the real IRC and a qualified professional.
- **Not GPT-4.** A 3B model has limits. It's trained to be accurate about *citations* and *specific provisions*, not to do open-ended tax planning.
- **Not always current.** Tax law changes; the model knows what the source documents said on the date we trained it.

---

# Technical Details

Fine-tuning **Qwen 2.5 3B Instruct** on the IRS Tax Code (IRC Title 26 + 26 CFR Treasury Regulations) using a three-stage pipeline: **SFT → DPO → GRPO** — entirely on Apple Silicon with MLX.

The resulting model is exported to GGUF and served locally via [Ollama](https://ollama.com).

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

## Hardware Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| Chip | Apple Silicon (M1+) | Apple M4 Max |
| RAM | 32 GB unified | 128 GB unified |
| GPU cores | 16 Metal cores | 40-core GPU |
| Disk | 20 GB free | 41 GB free |

The pipeline is designed for **MLX on Apple Silicon**. It does not use CUDA or MPS via PyTorch — training runs natively via `mlx_lm`.

## Setup

```bash
git clone https://github.com/dennisonbertram/rl-irs-tax-code.git
cd rl-irs-tax-code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install mlx-lm          # Apple Silicon only
brew install ollama
```

## Data Pipeline

The raw IRC and CFR XML files are **not included** in this repo (too large). Regenerate them:

```bash
bash scripts/download_data.sh
python scripts/parse_irc.py
python scripts/parse_cfr.py
python scripts/generate_training_data.py
python scripts/split_data.py
```

Produces:
- `data/processed/sft_train.jsonl` (~52 MB) — instruction/response pairs
- `data/processed/dpo_train.jsonl` (~17 MB) — preference pairs
- `data/processed/grpo_train.jsonl` (~3 MB) — GRPO prompts
- `data/processed/train/` — train/valid splits for mlx_lm

## Training

```bash
# Download base model
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen2.5-3b-instruct

# Stage 1: SFT
python scripts/train_sft.py --iters 1000 --batch-size 4

# Stage 2: DPO
python scripts/train_dpo.py

# Stage 3: GRPO
python scripts/train_grpo.py

# Evaluate
python scripts/evaluate.py
```

## Export to Ollama

```bash
python scripts/export_to_ollama.py
ollama run irs-tax-qwen
```

## HuggingFace Models

| Version | GGUF | Adapters | Status |
|---------|------|----------|--------|
| v1 | [dennisonb/qwen25-tax-3b-GGUF](https://huggingface.co/dennisonb/qwen25-tax-3b-GGUF) | [dennisonb/qwen25-tax-3b-adapters](https://huggingface.co/dennisonb/qwen25-tax-3b-adapters) | Published |
| v2 | [dennisonb/qwen25-tax-3b-v2-GGUF](https://huggingface.co/dennisonb/qwen25-tax-3b-v2-GGUF) | [dennisonb/qwen25-tax-3b-v2-adapters](https://huggingface.co/dennisonb/qwen25-tax-3b-v2-adapters) | Uploading |
| v3 | [dennisonb/qwen25-tax-3b-v3-GGUF](https://huggingface.co/dennisonb/qwen25-tax-3b-v3-GGUF) | [dennisonb/qwen25-tax-3b-v3-adapters](https://huggingface.co/dennisonb/qwen25-tax-3b-v3-adapters) | Uploading |

## Project Structure

```
rl-irs-tax-code/
├── configs/                    # Training hyperparameters
├── scripts/                    # Parse, train, evaluate, export
├── docs/
│   ├── assets/                 # README header art
│   ├── context/                # Background research
│   └── investigations/         # Pipeline reviews + debug notes
├── data/                       # Generated, not committed
├── models/                     # Downloaded base weights, not committed
├── outputs/                    # Training adapters, not committed
├── requirements.txt
└── README.md
```

## Limitations

- **Apple Silicon only** — MLX training does not run on CUDA without rewriting to `trl` + `SFTTrainer`.
- **3B parameters** — small model; trained for citation accuracy, not open-ended tax analysis.
- **Not legal advice** — verify outputs against the official IRC and CFR before relying on them.
- **Data freshness** — reflects a specific publication date; tax law changes frequently.

## License

Training code: **Apache 2.0** (matching Qwen 2.5). The IRS tax code text is U.S. government public domain.
