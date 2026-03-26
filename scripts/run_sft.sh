#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_sft.sh — Shell wrapper for mlx_lm.lora SFT training.
#
# Prefer using train_sft.py for full pre-flight checks; this script is a
# thin wrapper for quick CLI invocation or CI pipelines.
#
# Usage:
#   bash scripts/run_sft.sh                   # default hyperparameters
#   ITERS=500 bash scripts/run_sft.sh         # override via env vars
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Model resolution: prefer MLX-converted, fall back to HuggingFace weights
# ---------------------------------------------------------------------------
MODEL_MLX="$PROJECT_ROOT/models/qwen25-3b-mlx"
MODEL_HF="$PROJECT_ROOT/models/qwen2.5-3b-instruct"

if [ -f "$MODEL_MLX/config.json" ]; then
    MODEL="$MODEL_MLX"
    echo "Using MLX model: $MODEL"
elif [ -f "$MODEL_HF/config.json" ]; then
    MODEL="$MODEL_HF"
    echo "Using HF model (will auto-convert): $MODEL"
else
    echo "ERROR: No model found at $MODEL_MLX or $MODEL_HF"
    exit 1
fi

# ---------------------------------------------------------------------------
# Hyperparameters — override with environment variables
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/processed/train}"
ADAPTER_PATH="${ADAPTER_PATH:-$PROJECT_ROOT/outputs/sft/adapters}"
ITERS="${ITERS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LORA_LAYERS="${LORA_LAYERS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
VAL_BATCHES="${VAL_BATCHES:-25}"
STEPS_PER_EVAL="${STEPS_PER_EVAL:-100}"
SAVE_EVERY="${SAVE_EVERY:-200}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "ERROR: $DATA_DIR/train.jsonl not found. Run data pipeline first."
    exit 1
fi
if [ ! -f "$DATA_DIR/valid.jsonl" ]; then
    echo "ERROR: $DATA_DIR/valid.jsonl not found. Run data pipeline first."
    exit 1
fi

mkdir -p "$ADAPTER_PATH"
mkdir -p "$PROJECT_ROOT/outputs/sft"

echo ""
echo "========================================================================"
echo "SFT TRAINING — mlx_lm.lora"
echo "  model:         $MODEL"
echo "  data:          $DATA_DIR"
echo "  adapter-path:  $ADAPTER_PATH"
echo "  iters:         $ITERS"
echo "  batch-size:    $BATCH_SIZE"
echo "  lora-layers:   $LORA_LAYERS"
echo "  learning-rate: $LEARNING_RATE"
echo "========================================================================"
echo ""

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
python3 -m mlx_lm.lora \
  --model "$MODEL" \
  --data "$DATA_DIR" \
  --train \
  --batch-size "$BATCH_SIZE" \
  --lora-layers "$LORA_LAYERS" \
  --iters "$ITERS" \
  --val-batches "$VAL_BATCHES" \
  --learning-rate "$LEARNING_RATE" \
  --steps-per-eval "$STEPS_PER_EVAL" \
  --adapter-path "$ADAPTER_PATH" \
  --save-every "$SAVE_EVERY" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --grad-checkpoint \
  2>&1 | tee "$PROJECT_ROOT/outputs/sft/train.log"

echo ""
echo "Training complete. Adapters at: $ADAPTER_PATH"
