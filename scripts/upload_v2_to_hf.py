#!/usr/bin/env python3
"""
Upload v2 model adapters and GGUF to HuggingFace.

v2 adapter lineage:
  - SFT:  outputs/sft/adapters/  (rank=32, 106MB) — v2 re-trained SFT used as
          DPO initialization; same directory still has v2 weights since v3 SFT
          overwrote it (but DPO v2 log confirms it loaded from here)
  - DPO:  outputs/dpo/adapters_v2_backup/  (rank=16, 53MB)
  - GRPO: outputs/grpo/adapters_v2_backup/ (rank=32, 106MB; identical to run2)

GGUF:
  - Ollama model qwen25-tax-3b-v2 blob at
    ~/.ollama/models/blobs/sha256-0ba41839...  (3.28GB, q8_0)

Repos created:
  - dennisonb/qwen25-tax-3b-v2-adapters  (SFT + DPO + GRPO adapters)
  - dennisonb/qwen25-tax-3b-v2-GGUF      (q8_0 GGUF from Ollama)
"""

import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).parent.parent

SFT_ADAPTER_DIR   = PROJECT_ROOT / "outputs" / "sft" / "adapters"
DPO_ADAPTER_DIR   = PROJECT_ROOT / "outputs" / "dpo" / "adapters_v2_backup"
GRPO_ADAPTER_DIR  = PROJECT_ROOT / "outputs" / "grpo" / "adapters_v2_backup"
OLLAMA_BLOB       = Path.home() / ".ollama" / "models" / "blobs" / \
    "sha256-0ba41839a9b02c5bbf688b0e5676dc89caf9ea0fc6bde690f43b96da8f2bca0c"

HF_USER           = "dennisonb"
ADAPTERS_REPO     = f"{HF_USER}/qwen25-tax-3b-v2-adapters"
GGUF_REPO         = f"{HF_USER}/qwen25-tax-3b-v2-GGUF"


def upload_adapters(api: HfApi) -> None:
    print(f"\n{'='*60}")
    print(f"Creating / verifying repo: {ADAPTERS_REPO}")
    create_repo(ADAPTERS_REPO, repo_type="model", exist_ok=True)

    # SFT adapter
    print("\n--- Uploading SFT adapters (outputs/sft/adapters) ---")
    for f in SFT_ADAPTER_DIR.iterdir():
        if f.is_file():
            print(f"  Uploading sft/{f.name} ({f.stat().st_size // 1024 // 1024} MB)")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"sft/{f.name}",
                repo_id=ADAPTERS_REPO,
                repo_type="model",
            )

    # DPO adapter
    print("\n--- Uploading DPO adapters (adapters_v2_backup) ---")
    for f in DPO_ADAPTER_DIR.iterdir():
        if f.is_file():
            print(f"  Uploading dpo/{f.name} ({f.stat().st_size // 1024 // 1024} MB)")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"dpo/{f.name}",
                repo_id=ADAPTERS_REPO,
                repo_type="model",
            )

    # GRPO adapter
    print("\n--- Uploading GRPO adapters (adapters_v2_backup) ---")
    for f in GRPO_ADAPTER_DIR.iterdir():
        if f.is_file():
            print(f"  Uploading grpo/{f.name} ({f.stat().st_size // 1024 // 1024} MB)")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"grpo/{f.name}",
                repo_id=ADAPTERS_REPO,
                repo_type="model",
            )

    # Upload a README
    readme = _adapters_readme()
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=ADAPTERS_REPO,
        repo_type="model",
    )
    print(f"\nAdapters repo: https://huggingface.co/{ADAPTERS_REPO}")


def upload_gguf(api: HfApi) -> None:
    print(f"\n{'='*60}")
    print(f"Creating / verifying repo: {GGUF_REPO}")
    create_repo(GGUF_REPO, repo_type="model", exist_ok=True)

    if not OLLAMA_BLOB.exists():
        print(f"ERROR: Ollama blob not found at {OLLAMA_BLOB}")
        print("Skipping GGUF upload.")
        return

    size_gb = OLLAMA_BLOB.stat().st_size / 1024 / 1024 / 1024
    print(f"\n--- Uploading GGUF blob ({size_gb:.2f} GB) ---")
    print(f"  Source: {OLLAMA_BLOB}")
    print("  Destination: qwen25-tax-3b-v2-q8_0.gguf")
    print("  Note: This may take several minutes for a 3.3 GB file...")

    api.upload_file(
        path_or_fileobj=str(OLLAMA_BLOB),
        path_in_repo="qwen25-tax-3b-v2-q8_0.gguf",
        repo_id=GGUF_REPO,
        repo_type="model",
    )

    readme = _gguf_readme()
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=GGUF_REPO,
        repo_type="model",
    )
    print(f"\nGGUF repo: https://huggingface.co/{GGUF_REPO}")


def _adapters_readme() -> str:
    return """\
# qwen25-tax-3b-v2 — LoRA Adapters

LoRA adapters for the v2 IRS Tax Code fine-tune of
[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

## Training Pipeline

| Stage | Directory | LoRA Rank | Steps | Notes |
|-------|-----------|-----------|-------|-------|
| SFT   | `sft/`    | 32        | 1000  | Supervised fine-tune on IRC sections |
| DPO   | `dpo/`    | 16        | 500   | Direct preference optimization; init from SFT |
| GRPO  | `grpo/`   | 32        | 300   | Group relative policy optimization; init from DPO |

## Usage (mlx-lm)

```bash
pip install mlx-lm

# Generate with GRPO adapter (best quality):
mlx_lm.generate \\
  --model Qwen/Qwen2.5-3B-Instruct \\
  --adapter-path grpo/ \\
  --prompt "What is the standard deduction for 2024?"
```

## GGUF

See [dennisonb/qwen25-tax-3b-v2-GGUF](https://huggingface.co/dennisonb/qwen25-tax-3b-v2-GGUF)
for a ready-to-use GGUF (q8_0) for Ollama / llama.cpp.
"""


def _gguf_readme() -> str:
    return """\
# qwen25-tax-3b-v2 — GGUF (q8_0)

Quantized GGUF of the v2 IRS Tax Code fine-tune of
[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

The GGUF was produced by fusing all three training stages (SFT → DPO → GRPO)
and exporting via mlx_lm.fuse + llama.cpp convert_hf_to_gguf.

## File

| File | Quantization | Size |
|------|--------------|------|
| `qwen25-tax-3b-v2-q8_0.gguf` | Q8_0 | ~3.3 GB |

## Ollama

```bash
# Modelfile
cat > Modelfile << 'EOF'
FROM qwen25-tax-3b-v2-q8_0.gguf
SYSTEM "You are a tax law assistant trained on the Internal Revenue Code (Title 26) and Treasury Regulations (26 CFR). You answer questions about US federal tax law accurately, cite relevant IRC sections, and note important exceptions and limitations. You do not provide personalised tax advice; always recommend consulting a qualified tax professional for individual situations."
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
EOF

ollama create qwen25-tax-3b-v2 -f Modelfile
ollama run qwen25-tax-3b-v2
```

## LoRA Adapters

See [dennisonb/qwen25-tax-3b-v2-adapters](https://huggingface.co/dennisonb/qwen25-tax-3b-v2-adapters)
for the raw LoRA adapter weights (SFT / DPO / GRPO).
"""


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Upload v2 artifacts to HuggingFace")
    parser.add_argument("--skip-adapters", action="store_true", help="Skip adapter upload")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF upload")
    args = parser.parse_args()

    api = HfApi()
    me = api.whoami()["name"]
    print(f"Authenticated as: {me}")

    if not args.skip_adapters:
        upload_adapters(api)
    if not args.skip_gguf:
        upload_gguf(api)

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
