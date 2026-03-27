#!/usr/bin/env bash
# Full retraining pipeline: SFT -> DPO -> GRPO (v2 grounded data)
# Run this AFTER SFT training completes (SFT is already running in background).
#
# Usage:
#   bash scripts/run_pipeline_v2.sh
#
# This script:
# 1. Waits for DPO data generation to finish
# 2. Converts grounded_dpo_full.jsonl -> data/processed/train/dpo.jsonl
# 3. Runs DPO training
# 4. Runs GRPO training

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python3.14"

echo "=== Pipeline v2 ==="
echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON"
echo ""

# ── Wait for DPO data generation ──────────────────────────────────────────────
DPO_FULL="$PROJECT_ROOT/data/processed/grounded_dpo_full.jsonl"
TARGET_LINES=1600  # expect 1,690 but stop waiting when close enough

echo "Waiting for DPO data generation to finish..."
while true; do
    if [ -f "$DPO_FULL" ]; then
        LINES=$(wc -l < "$DPO_FULL")
        if [ "$LINES" -ge "$TARGET_LINES" ]; then
            echo "DPO data ready: $LINES pairs"
            break
        fi
        echo "  DPO progress: $LINES / ~1690 pairs — waiting..."
    fi
    sleep 60
done

# ── Convert DPO data ───────────────────────────────────────────────────────────
echo ""
echo "Converting DPO data to training format..."
"$PYTHON" "$PROJECT_ROOT/scripts/prepare_dpo_training_data.py"

DPO_TRAIN="$PROJECT_ROOT/data/processed/train/dpo.jsonl"
echo "DPO training data: $(wc -l < "$DPO_TRAIN") pairs"

# ── DPO Training ──────────────────────────────────────────────────────────────
echo ""
echo "=== Stage 2: DPO Training ==="
"$PYTHON" "$PROJECT_ROOT/scripts/train_dpo.py" \
    --iters 500 \
    --learning-rate 5e-6 \
    2>&1 | tee "$PROJECT_ROOT/outputs/dpo/train_v2.log"

echo ""
echo "DPO training complete."

# ── GRPO Training ─────────────────────────────────────────────────────────────
echo ""
echo "=== Stage 3: GRPO Training ==="
"$PYTHON" "$PROJECT_ROOT/scripts/train_grpo.py" \
    --iters 300 \
    --group-size 4 \
    --learning-rate 1e-6 \
    --save-every 50 \
    --log-every 5 \
    2>&1 | tee "$PROJECT_ROOT/outputs/grpo/train_v2.log"

echo ""
echo "=== Pipeline v2 Complete ==="
echo "All three stages finished:"
echo "  SFT  -> outputs/sft/adapters/"
echo "  DPO  -> outputs/dpo/adapters/"
echo "  GRPO -> outputs/grpo/adapters/"
