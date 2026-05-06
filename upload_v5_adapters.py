"""
Upload v5 GRPO adapter (and SFT/DPO stages) to HuggingFace.

Repo: dennisonb/qwen25-tax-3b-v5-adapters
Structure mirrors v3: sft/, dpo/, grpo/ subdirectories.
"""

import os
import sys
from huggingface_hub import HfApi

TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
TOKEN = open(TOKEN_PATH).read().strip()

USERNAME = "dennisonb"
REPO = f"{USERNAME}/qwen25-tax-3b-v5-adapters"
BASE_DIR = "/Users/dennisonbertram/Develop/rl-irs-tax-code/outputs"

MODEL_CARD = """\
---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- lora
- irs
- tax
- legal
- fine-tuned
- sft
- dpo
- grpo
- rl
- mlx
---

# qwen25-tax-3b-v5-adapters

LoRA adapters for **Qwen 2.5 3B Instruct** fine-tuned on the IRS Tax Code (IRC Title 26 + 26 CFR Treasury Regulations) using a three-stage pipeline.

## v5 Training Details

| Stage | Dataset Size | Iterations | Notes |
|-------|-------------|------------|-------|
| SFT   | 99K examples | 1,500 iters | LoRA rank 32, lr 1e-5, cosine decay |
| DPO   | 23K pairs | 1,500 iters | Fixed length normalization bug from v4 |
| GRPO  | 99K prompts | 1,000 iters | Rule-based rewards: citation accuracy + completeness |

### v5 Improvements over v4
- **Fixed DPO length normalization bug** — DPO loss now correctly normalized per token
- **Inflation upsampling 20x** — IRC inflation adjustment sections upsampled to improve coverage
- **CFR data included** — 26 CFR Treasury Regulations added to all three training stages
- Larger SFT and GRPO datasets (99K vs ~50K in v4)
- More DPO iterations (1,500 vs 1,000 in v4)

## Base Model

- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Architecture**: Transformer, 3B parameters
- **Context window**: 2,048 tokens during training

## Adapter Files

```
sft/    — Stage 1 adapter (after supervised fine-tuning)
  adapter_config.json
  adapters.safetensors      (~102 MB)
  0000200_adapters.safetensors  (checkpoint at step 200)
  0000400_adapters.safetensors
  0000600_adapters.safetensors
  0000800_adapters.safetensors
  0001000_adapters.safetensors
  0001200_adapters.safetensors
  0001400_adapters.safetensors

dpo/    — Stage 2 adapter (after DPO on top of SFT)
  adapter_config.json
  adapters.safetensors      (~102 MB)
  adapters_best.safetensors (~102 MB, best checkpoint)

grpo/   — Stage 3 adapter (final, after GRPO on top of DPO)
  adapter_config.json
  adapters.safetensors      (~102 MB)  ← recommended
  adapters_best.safetensors (~102 MB, best checkpoint)
```

## Usage

These are MLX LoRA adapters. To use with `mlx_lm`:

```bash
pip install mlx-lm

# Inference with GRPO adapter (final stage)
python -m mlx_lm.generate \\
  --model Qwen/Qwen2.5-3B-Instruct \\
  --adapter-path grpo/ \\
  --prompt "What is the standard deduction for a single filer in 2024?"
```

## Training Hardware

Trained on Apple M4 Max (128 GB unified memory) using `mlx_lm`.

## Repository

Source code: https://github.com/dennisonbertram/rl-irs-tax-code
"""

api = HfApi()

print(f"Creating repo {REPO} ...")
api.create_repo(repo_id=REPO, repo_type="model", exist_ok=True, token=TOKEN)

print("Uploading model card (README.md) ...")
api.upload_file(
    path_or_fileobj=MODEL_CARD.encode(),
    path_in_repo="README.md",
    repo_id=REPO,
    token=TOKEN,
    commit_message="Add model card",
)

# Upload each stage
for stage in ["sft", "dpo", "grpo"]:
    folder = f"{BASE_DIR}/{stage}/adapters"
    print(f"\nUploading {stage}/ adapters from {folder} ...")
    api.upload_folder(
        folder_path=folder,
        path_in_repo=f"{stage}/",
        repo_id=REPO,
        token=TOKEN,
        commit_message=f"Upload {stage} adapters (v5)",
    )
    print(f"  Done: {stage}/")

print(f"\nAll uploads complete.")
print(f"Repo URL: https://huggingface.co/{REPO}")
