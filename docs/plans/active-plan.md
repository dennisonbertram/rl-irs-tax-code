# IRS Tax Code RL Training - Execution Plan

## Goal
Fine-tune a small local LLM on IRS tax code data using SFT + RL (DPO/GRPO), deploy via Ollama.

## System Context
- Apple M4 Max, 128GB unified RAM, 40-core Metal GPU
- 41GB free disk (constraint — use 3B model to be safe)
- Ollama 0.18.2 installed, no models pulled
- Python 3.14, pip available, no ML packages installed

## Architecture Decision
- **Base model**: Qwen 2.5 3B Instruct (small enough for disk, strong reasoning)
- **Training framework**: Unsloth + TRL (QLoRA SFT, then DPO/GRPO for RL)
- **Alternative**: MLX-lm for SFT if Unsloth MPS support is problematic
- **Data**: IRC Title 26 XML + Treasury Regulations XML → parsed into JSONL
- **Export**: GGUF Q4_K_M → Ollama

## Phases

### Phase 1: Environment Setup [TASK-001]
- Install Python ML stack: torch, transformers, trl, peft, datasets, accelerate, unsloth, huggingface-hub
- Pull Qwen 2.5 3B via Ollama for baseline testing
- Download HuggingFace weights for Qwen 2.5 3B Instruct

### Phase 2: Data Acquisition [TASK-002]
- Download IRC Title 26 XML from uscode.house.gov
- Download 26 CFR XML from govinfo.gov
- Store in data/raw/

### Phase 3: Data Pipeline [TASK-003]
- Parse IRC XML into structured text (section-level chunks)
- Parse CFR XML into structured text
- Generate SFT training pairs (instruction/response format)
- Generate DPO preference pairs
- Generate GRPO prompts with verifiable reward signals
- Output: data/processed/sft_train.jsonl, data/processed/dpo_train.jsonl, data/processed/grpo_train.jsonl
- Split 90/10 train/eval

### Phase 4: SFT Training [TASK-004]
- QLoRA fine-tuning on SFT dataset
- Qwen 2.5 3B, r=32, 4-bit quantization
- 2-3 epochs, lr=2e-4
- Evaluate on held-out set
- Save checkpoint

### Phase 5: RL Training (DPO + GRPO) [TASK-005]
- DPO on preference pairs from SFT checkpoint
- GRPO with rule-based rewards (IRC section citation, factual accuracy)
- 1 epoch each
- Save final checkpoint

### Phase 6: Export & Deploy [TASK-006]
- Merge LoRA adapters
- Export to GGUF Q4_K_M
- Create Ollama Modelfile
- Import to Ollama
- Run evaluation prompts

## Task Dependencies
TASK-001 → TASK-002 (parallel possible)
TASK-001 + TASK-002 → TASK-003
TASK-003 → TASK-004
TASK-004 → TASK-005
TASK-005 → TASK-006
