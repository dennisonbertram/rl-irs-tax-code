#!/usr/bin/env python3
"""
Split processed datasets 90/10 into train and eval splits.

Input files:
  - data/processed/sft_train.jsonl
  - data/processed/dpo_train.jsonl
  - data/processed/grpo_train.jsonl

Output directories:
  - data/processed/train/
  - data/processed/eval/
"""
import json
import math
import random
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data/processed"
TRAIN_DIR = PROCESSED_DIR / "train"
EVAL_DIR = PROCESSED_DIR / "eval"

DATASETS = [
    "sft_train.jsonl",
    "dpo_train.jsonl",
    "grpo_train.jsonl",
]

TRAIN_RATIO = 0.90

random.seed(42)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_and_write(dataset_name: str) -> dict:
    """Split a dataset and write train/eval files. Returns stats."""
    src = PROCESSED_DIR / dataset_name
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return {}

    records = load_jsonl(src)
    n = len(records)
    random.shuffle(records)

    split_idx = math.ceil(n * TRAIN_RATIO)
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    # Output filenames: sft_train.jsonl -> train/sft.jsonl + eval/sft.jsonl
    base_name = dataset_name.replace("_train", "")

    train_path = TRAIN_DIR / base_name
    eval_path = EVAL_DIR / base_name

    write_jsonl(train_records, train_path)
    write_jsonl(eval_records, eval_path)

    return {
        "dataset": dataset_name,
        "total": n,
        "train": len(train_records),
        "eval": len(eval_records),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
    }


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Splitting datasets 90/10 (train/eval)...\n")

    all_stats = []
    for ds in DATASETS:
        print(f"Processing {ds}...")
        stats = split_and_write(ds)
        if stats:
            all_stats.append(stats)
            print(f"  Total: {stats['total']:,}")
            print(f"  Train: {stats['train']:,} -> {stats['train_path']}")
            print(f"  Eval:  {stats['eval']:,} -> {stats['eval_path']}")
        print()

    # Summary
    print("=== Split Summary ===")
    for s in all_stats:
        pct = s["eval"] / s["total"] * 100 if s["total"] > 0 else 0
        print(
            f"  {s['dataset']:25s}  total={s['total']:6,}  "
            f"train={s['train']:6,}  eval={s['eval']:5,}  "
            f"({pct:.1f}% eval)"
        )

    # Verify samples
    print("\n=== Sample Verification ===")
    for s in all_stats:
        eval_path = Path(s["eval_path"])
        if eval_path.exists():
            records = load_jsonl(eval_path)
            if records:
                sample = records[0]
                print(f"\n{eval_path.name} eval sample:")
                # Show first key of the sample
                first_key = list(sample.keys())[0]
                val = sample[first_key]
                if isinstance(val, list):
                    print(f"  {first_key}: [{len(val)} items]")
                elif isinstance(val, str):
                    print(f"  {first_key}: {val[:80]}...")
                else:
                    print(f"  {first_key}: {val}")


if __name__ == "__main__":
    main()
