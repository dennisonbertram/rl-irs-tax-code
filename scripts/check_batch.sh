#!/usr/bin/env bash
# Usage: ./scripts/check_batch.sh [BATCH_ID]
# Checks batch status and optionally downloads results when complete.

BATCH_ID="${1:-batch_69c556ddfa2481909df8005d1e454637}"
VENV_PYTHON="/Users/dennisonbertram/Develop/rl-irs-tax-code/.venv/bin/python3.14"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

$VENV_PYTHON - "$BATCH_ID" <<'PYEOF'
import sys, os
from openai import OpenAI
import datetime

batch_id = sys.argv[1]
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
b = client.batches.retrieve(batch_id)
counts = b.request_counts

now = datetime.datetime.now()
created = datetime.datetime.fromtimestamp(b.created_at)
elapsed = (now - created).total_seconds() / 60

print(f"Batch: {batch_id}")
print(f"Status: {b.status}")
print(f"Progress: {counts.completed}/{counts.total} completed, {counts.failed} failed")
print(f"Elapsed: {elapsed:.0f} minutes")

if b.status == "completed":
    print(f"\nOutput file: {b.output_file_id}")
    print(f"\nTo download results, run:")
    print(f"  .venv/bin/python3.14 scripts/generate_grounded_data.py \\")
    print(f"    --download-batch {batch_id} \\")
    print(f"    --output data/processed/grounded_sft_full.jsonl \\")
    print(f"    --dpo-output data/processed/grounded_dpo_full.jsonl")
elif b.status in ("failed", "expired", "cancelled"):
    print(f"\nBatch ended with status: {b.status}")
    if b.errors:
        for e in b.errors.data:
            print(f"  Error: {e}")
else:
    pct = (counts.completed / counts.total * 100) if counts.total > 0 else 0
    print(f"Still running ({pct:.0f}% complete)")
PYEOF
