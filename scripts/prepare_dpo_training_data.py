#!/usr/bin/env python3
"""
Convert grounded_dpo_full.jsonl to training format for train_dpo.py.

Strips metadata and writes only {prompt, chosen, rejected} records
to data/processed/train/dpo.jsonl.

Usage:
    python3.14 scripts/prepare_dpo_training_data.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DPO_SOURCE   = PROJECT_ROOT / "data" / "processed" / "grounded_dpo_full.jsonl"
DPO_TRAIN    = PROJECT_ROOT / "data" / "processed" / "train" / "dpo.jsonl"


def main():
    if not DPO_SOURCE.exists():
        print(f"ERROR: Source not found: {DPO_SOURCE}")
        return

    records = []
    with open(DPO_SOURCE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] Skipping malformed JSON line")
                continue
            # Validate required keys
            if all(k in rec for k in ("prompt", "chosen", "rejected")):
                records.append({
                    "prompt":   rec["prompt"],
                    "chosen":   rec["chosen"],
                    "rejected": rec["rejected"],
                })
            else:
                print(f"  [WARN] Skipping record with missing keys: {list(rec.keys())}")

    # Backup old dpo.jsonl if it exists
    if DPO_TRAIN.exists():
        backup = DPO_TRAIN.parent / "dpo_v1.jsonl"
        import shutil
        shutil.copy(DPO_TRAIN, backup)
        print(f"Backed up {DPO_TRAIN.name} -> {backup.name}")

    DPO_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(DPO_TRAIN, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Written {len(records)} DPO pairs to {DPO_TRAIN}")


if __name__ == "__main__":
    main()
