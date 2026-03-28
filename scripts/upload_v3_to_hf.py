"""
Upload v3 model artifacts to HuggingFace.

Uploads:
  - GGUF (q8_0 quantization) to dennisonb/qwen25-tax-3b-v3-GGUF
  - SFT, DPO, GRPO adapters to dennisonb/qwen25-tax-3b-v3-adapters
"""

import os
from huggingface_hub import HfApi, create_repo

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GGUF_PATH = os.path.join(BASE_DIR, "outputs", "final", "model-q8.gguf")
SFT_ADAPTERS = os.path.join(BASE_DIR, "outputs", "sft", "adapters")
DPO_ADAPTERS = os.path.join(BASE_DIR, "outputs", "dpo", "adapters")
GRPO_ADAPTERS = os.path.join(BASE_DIR, "outputs", "grpo", "adapters")

GGUF_REPO = "dennisonb/qwen25-tax-3b-v3-GGUF"
ADAPTERS_REPO = "dennisonb/qwen25-tax-3b-v3-adapters"

GGUF_README = """---
language:
- en
license: apache-2.0
tags:
- tax
- irs
- legal
- finance
- gguf
- qwen2.5
base_model: Qwen/Qwen2.5-3B-Instruct
---

# qwen25-tax-3b-v3 (GGUF)

A Q8_0 quantized GGUF of the v3 IRS Tax Code model.

## Model Description

**Base model**: Qwen/Qwen2.5-3B-Instruct
**Fine-tuning pipeline**: SFT → DPO → GRPO
**Training data**: IRC Title 26 (Internal Revenue Code), U.S. Code of Federal Regulations Title 26

This is **v3** of the IRS Tax Code RL project. The model was trained in three stages:

1. **SFT** (Supervised Fine-Tuning): Grounded question-answer pairs from IRC Title 26
2. **DPO** (Direct Preference Optimization): Preference data generated from SFT model outputs
3. **GRPO** (Group Relative Policy Optimization): RL fine-tuning with reward signals based on citation accuracy and answer correctness

## Files

| File | Description |
|------|-------------|
| `qwen25-tax-3b-v3-q8_0.gguf` | Q8_0 quantized GGUF, ~3.1GB |

## Usage (Ollama)

```bash
ollama run hf.co/dennisonb/qwen25-tax-3b-v3-GGUF
```

## Usage (llama.cpp)

```bash
./llama-cli -m qwen25-tax-3b-v3-q8_0.gguf -p "What is the standard deduction for 2024?"
```

## Intended Use

This model is intended for research and educational purposes related to U.S. tax law (IRC Title 26). It is NOT a substitute for professional tax advice.
"""

ADAPTERS_README = """---
language:
- en
license: apache-2.0
tags:
- tax
- irs
- legal
- finance
- lora
- peft
- qwen2.5
base_model: Qwen/Qwen2.5-3B-Instruct
---

# qwen25-tax-3b-v3 (LoRA Adapters)

LoRA adapter weights from all three training stages of the v3 IRS Tax Code RL model.

## Model Description

**Base model**: Qwen/Qwen2.5-3B-Instruct
**Fine-tuning pipeline**: SFT → DPO → GRPO
**Training data**: IRC Title 26 (Internal Revenue Code), U.S. Code of Federal Regulations Title 26

This is **v3** of the IRS Tax Code RL project.

## Training Stages

### 1. SFT (Supervised Fine-Tuning)
- Adapters in: `sft/`
- Grounded on IRC Title 26 question-answer pairs
- Multiple checkpoints retained (`0000200_adapters.safetensors` through `0001000_adapters.safetensors`)
- Final adapter: `sft/adapters.safetensors`

### 2. DPO (Direct Preference Optimization)
- Adapters in: `dpo/`
- Trained on preference data generated from SFT model outputs
- Best checkpoint: `dpo/adapters_best.safetensors`

### 3. GRPO (Group Relative Policy Optimization)
- Adapters in: `grpo/`
- RL fine-tuning with citation accuracy and answer correctness reward signals
- Best checkpoint: `grpo/adapters_best.safetensors`

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Load the final GRPO adapter (recommended)
model = PeftModel.from_pretrained(base, "dennisonb/qwen25-tax-3b-v3-adapters/grpo")
```

## Intended Use

This model is intended for research and educational purposes related to U.S. tax law (IRC Title 26). It is NOT a substitute for professional tax advice.
"""


def main():
    api = HfApi()

    print("Creating GGUF repo...")
    create_repo(GGUF_REPO, exist_ok=True, repo_type="model")
    print(f"  Repo ready: https://huggingface.co/{GGUF_REPO}")

    print("Uploading GGUF README...")
    api.upload_file(
        path_or_fileobj=GGUF_README.encode(),
        path_in_repo="README.md",
        repo_id=GGUF_REPO,
        commit_message="Add model card for v3 GGUF",
    )

    print(f"Uploading GGUF file ({GGUF_PATH})...")
    print("  This will take a while for the 3.1GB file...")
    api.upload_file(
        path_or_fileobj=GGUF_PATH,
        path_in_repo="qwen25-tax-3b-v3-q8_0.gguf",
        repo_id=GGUF_REPO,
        commit_message="Upload v3 Q8_0 GGUF",
    )
    print(f"  GGUF uploaded: https://huggingface.co/{GGUF_REPO}")

    print("\nCreating adapters repo...")
    create_repo(ADAPTERS_REPO, exist_ok=True, repo_type="model")
    print(f"  Repo ready: https://huggingface.co/{ADAPTERS_REPO}")

    print("Uploading adapters README...")
    api.upload_file(
        path_or_fileobj=ADAPTERS_README.encode(),
        path_in_repo="README.md",
        repo_id=ADAPTERS_REPO,
        commit_message="Add model card for v3 adapters",
    )

    print("Uploading SFT adapters...")
    api.upload_folder(
        folder_path=SFT_ADAPTERS,
        path_in_repo="sft",
        repo_id=ADAPTERS_REPO,
        commit_message="Upload v3 SFT adapters",
    )
    print("  SFT done.")

    print("Uploading DPO adapters...")
    api.upload_folder(
        folder_path=DPO_ADAPTERS,
        path_in_repo="dpo",
        repo_id=ADAPTERS_REPO,
        commit_message="Upload v3 DPO adapters",
    )
    print("  DPO done.")

    print("Uploading GRPO adapters...")
    api.upload_folder(
        folder_path=GRPO_ADAPTERS,
        path_in_repo="grpo",
        repo_id=ADAPTERS_REPO,
        commit_message="Upload v3 GRPO adapters",
    )
    print("  GRPO done.")

    print("\nAll uploads complete.")
    print(f"  GGUF repo:     https://huggingface.co/{GGUF_REPO}")
    print(f"  Adapters repo: https://huggingface.co/{ADAPTERS_REPO}")


if __name__ == "__main__":
    main()
