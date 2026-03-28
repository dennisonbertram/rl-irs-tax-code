# 32B SFT Training - Started

## Date: 2026-03-26

## Step 1: Inference Test - PASSED
- Model: `models/qwen25-32b-mlx` (Qwen2.5-32B-Instruct-4bit, ~18GB)
- Loaded successfully, generated coherent tax law response
- Response to "standard deduction for single filer in 2024": "$13,850" (slightly outdated - actual 2024 is $14,600, but coherent)

## Step 2: Training Configuration
- Config: `configs/mlx_lora_rank16_32b.yaml` (rank 16, alpha 32, dropout 0.05)
- Output: `outputs/32b/sft/adapters/`
- Hyperparameters:
  - Batch size: 1
  - LoRA layers: 8 (of 64 total)
  - Learning rate: 1e-5
  - Iters: 1000
  - Max seq length: 2048
  - Gradient checkpointing: enabled

## Step 3: Training Progress
- PID: 13079
- Trainable parameters: 16.777M / 32763.876M (0.051%)
- Peak memory: 19.543 GB
- Speed: ~0.8 it/sec, ~100 tokens/sec
- Loss trajectory (first 70 iters):
  - Iter 1 (val): 3.759
  - Iter 10: 3.501
  - Iter 20: 2.680
  - Iter 30: 1.935
  - Iter 40: 1.508
  - Iter 50: 1.469
  - Iter 60: 1.215
  - Iter 70: 1.027
- Estimated time: ~1000 iters / 0.8 it/sec = ~20 minutes total

## Files Created/Modified
- `configs/mlx_lora_rank16_32b.yaml` - new LoRA config for 32B
- `outputs/32b/sft/adapters/` - adapter output directory
- `outputs/32b/sft/train.log` - training log
- `outputs/32b/dpo/adapters/` - future DPO output
- `outputs/32b/grpo/adapters/` - future GRPO output
- `outputs/32b/final/` - final merged output
