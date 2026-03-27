#!/usr/bin/env bash
# Round 2 Training Pipeline
# Phases: DPO gen wait -> DPO train -> GRPO train -> export -> on-policy DPO gen
#         -> round2 DPO train -> round2 GRPO train -> export v3
#
# Usage: bash scripts/run_round2_pipeline.sh 2>&1 | tee outputs/pipeline_round2.log
set -euo pipefail

PROJ=/Users/dennisonbertram/Develop/rl-irs-tax-code
PYTHON="$PROJ/.venv/bin/python3.14"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_disk() {
    local avail
    avail=$(df -g /Users/dennisonbertram | awk 'NR==2 {print $4}')
    log "Disk space available: ${avail}GB"
    if [ "$avail" -lt 5 ]; then
        log "WARNING: Low disk space (${avail}GB). Cleaning old checkpoints..."
        # Remove intermediate checkpoint files but keep final adapters
        find "$PROJ/outputs" -name "[0-9]*_adapters.safetensors" -delete && log "Cleaned intermediate adapter checkpoints."
    fi
}

cd "$PROJ"
source .venv/bin/activate

mkdir -p outputs/dpo outputs/grpo outputs/final

# ============================================================
# Phase 2: Wait for DPO generation to finish
# ============================================================
log "=== Phase 2: Waiting for DPO generation to finish ==="

while pgrep -f "generate_dpo_from_sft" > /dev/null 2>&1; do
    COUNT=$(wc -l < data/processed/grounded_dpo_full.jsonl 2>/dev/null || echo 0)
    log "DPO gen still running... $COUNT pairs so far"
    sleep 30
done

COUNT=$(wc -l < data/processed/grounded_dpo_full.jsonl 2>/dev/null || echo 0)
log "DPO GENERATION COMPLETE: $COUNT pairs in data/processed/grounded_dpo_full.jsonl"

# ============================================================
# Phase 3: Prepare and Run DPO Training
# ============================================================
log "=== Phase 3: Preparing DPO training data ==="
check_disk

"$PYTHON" scripts/prepare_dpo_training_data.py
DPO_PAIRS=$(wc -l < data/processed/train/dpo.jsonl)
log "DPO training pairs: $DPO_PAIRS"

# Verify format
"$PYTHON" -c "
import json
with open('data/processed/train/dpo.jsonl') as f:
    first = json.loads(f.readline())
assert set(first.keys()) >= {'prompt','chosen','rejected'}, f'Bad keys: {first.keys()}'
print('DPO data format OK:', list(first.keys()))
"

log "Running DPO training (Round 1, 500 iters, lr=5e-6)..."
"$PYTHON" scripts/train_dpo.py --iters 500 --learning-rate 5e-6 2>&1 | tee outputs/dpo/train_v2.log
log "DPO training complete."

# ============================================================
# Phase 4: Run GRPO Training (Round 1)
# ============================================================
log "=== Phase 4: Running GRPO training (Round 1, 300 iters) ==="
check_disk

"$PYTHON" scripts/train_grpo.py --iters 300 --group-size 4 --learning-rate 1e-6 --save-every 50 --log-every 5 2>&1 | tee outputs/grpo/train_v2.log
log "GRPO training (Round 1) complete."

# ============================================================
# Phase 5: Export Round 1 model to Ollama
# ============================================================
log "=== Phase 5: Exporting Round 1 model (qwen25-tax-3b-v2) ==="
check_disk

"$PYTHON" scripts/export_to_ollama.py --adapter-path outputs/grpo/adapters --model-name qwen25-tax-3b-v2 2>&1 | tee outputs/export_v2.log || {
    log "WARNING: Export failed (possible disk space or missing GGUF tool). Continuing to Round 2."
}

# ============================================================
# Phase 6: Round 2 — On-Policy DPO Generation
# ============================================================
log "=== Phase 6: Generating on-policy DPO pairs ==="
check_disk

"$PYTHON" scripts/generate_onpolicy_dpo.py 2>&1 | tee outputs/onpolicy_dpo.log
ONPOLICY_COUNT=$(wc -l < data/processed/onpolicy_dpo.jsonl 2>/dev/null || echo 0)
log "On-policy DPO pairs generated: $ONPOLICY_COUNT"

# Combine with existing DPO data
log "Combining DPO datasets..."
"$PYTHON" -c "
import json, shutil
existing = [json.loads(l) for l in open('data/processed/train/dpo.jsonl')]
onpolicy_path = 'data/processed/onpolicy_dpo.jsonl'
try:
    onpolicy = [json.loads(l) for l in open(onpolicy_path)]
except FileNotFoundError:
    onpolicy = []
    print('WARNING: on-policy DPO file not found, using only existing data')
combined = existing + onpolicy
with open('data/processed/train/dpo_v2.jsonl', 'w') as f:
    for r in combined:
        f.write(json.dumps(r) + '\n')
print(f'Combined DPO: {len(existing)} existing + {len(onpolicy)} on-policy = {len(combined)} total')
shutil.copy('data/processed/train/dpo_v2.jsonl', 'data/processed/train/dpo.jsonl')
print('Updated data/processed/train/dpo.jsonl')
"

# ============================================================
# Round 2 DPO Training
# ============================================================
log "=== Round 2: DPO training (500 iters, lr=3e-6) ==="
check_disk

"$PYTHON" scripts/train_dpo.py --iters 500 --learning-rate 3e-6 2>&1 | tee outputs/dpo/train_v2_round2.log
log "Round 2 DPO training complete."

# ============================================================
# Round 2 GRPO Training
# ============================================================
log "=== Round 2: GRPO training (300 iters, lr=5e-7) ==="
check_disk

"$PYTHON" scripts/train_grpo.py --iters 300 --group-size 4 --learning-rate 5e-7 --save-every 50 --log-every 5 2>&1 | tee outputs/grpo/train_v2_round2.log
log "Round 2 GRPO training complete."

# ============================================================
# Export Round 2 model
# ============================================================
log "=== Exporting Round 2 model (qwen25-tax-3b-v3) ==="
check_disk

"$PYTHON" scripts/export_to_ollama.py --adapter-path outputs/grpo/adapters --model-name qwen25-tax-3b-v3 2>&1 | tee outputs/export_v3.log || {
    log "WARNING: v3 export failed. See outputs/export_v3.log for details."
}

# ============================================================
# Final Summary
# ============================================================
log "=== PIPELINE COMPLETE ==="
log "SFT adapters:          outputs/sft/adapters/"
log "DPO adapters (R1):     outputs/dpo/adapters/"
log "GRPO adapters (R1):    outputs/grpo/adapters/"
log "DPO training logs:     outputs/dpo/train_v2.log, outputs/dpo/train_v2_round2.log"
log "GRPO training logs:    outputs/grpo/train_v2.log, outputs/grpo/train_v2_round2.log"
log "On-policy DPO log:     outputs/onpolicy_dpo.log"
df -h /Users/dennisonbertram | tail -1
