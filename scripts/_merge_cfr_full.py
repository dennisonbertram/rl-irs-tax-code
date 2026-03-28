#!/usr/bin/env python3
"""
_merge_cfr_full.py  (internal helper — called once, not part of public API)

Merges deduped CFR SFT data into ALL final training splits:
  - sft_train.jsonl  + train.jsonl  (compat copy)
  - sft_valid.jsonl  + valid.jsonl  (compat copy)
  - grpo_train.jsonl
  - grpo_valid.jsonl
  - MANIFEST.md  (appended)

Split: 90% CFR → train, 10% CFR → eval  (seed=42)
Re-shuffle: seed=42
GRPO format: {"prompt": <user content>, "expected_section": <source_section>}
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/Users/dennisonbertram/Develop/rl-irs-tax-code")
FINAL_DIR    = PROJECT_ROOT / "data" / "final"
CFR_DEDUPED  = PROJECT_ROOT / "data" / "processed" / "grounded_cfr_sft_deduped.jsonl"

SEED = 42


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def backup(path: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.stem}_pre_cfr_{ts}{path.suffix}")
    shutil.copy(path, dst)
    return dst


# ── CFR → GRPO record conversion ───────────────────────────────────────────────

def sft_to_grpo(rec: dict) -> dict:
    """Extract prompt and expected_section from a CFR SFT record."""
    messages = rec.get("messages", [])
    prompt = ""
    for msg in messages:
        if msg.get("role") == "user":
            prompt = msg.get("content", "")
            break
    source_section = rec.get("metadata", {}).get("source_section", "CFR")
    return {"prompt": prompt, "expected_section": source_section}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Full CFR Merge into Final Splits")
    print("=" * 60)

    # Load deduped CFR data
    print(f"\nLoading deduped CFR data from {CFR_DEDUPED}...")
    cfr_all = load_jsonl(CFR_DEDUPED)
    print(f"  Loaded {len(cfr_all):,} CFR records")

    # Split 90/10
    rng = random.Random(SEED)
    shuffled = list(cfr_all)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.90)
    cfr_train = shuffled[:split_idx]
    cfr_eval  = shuffled[split_idx:]
    print(f"  CFR train split: {len(cfr_train):,}")
    print(f"  CFR eval  split: {len(cfr_eval):,}")

    # ── SFT train ──────────────────────────────────────────────────────────────
    sft_train_path = FINAL_DIR / "sft_train.jsonl"
    print(f"\nLoading existing {sft_train_path.name}...")
    existing_sft_train = load_jsonl(sft_train_path)
    print(f"  Existing: {len(existing_sft_train):,}")

    combined_sft_train = existing_sft_train + cfr_train
    rng2 = random.Random(SEED)
    rng2.shuffle(combined_sft_train)
    print(f"  Combined + shuffled: {len(combined_sft_train):,}")

    bk = backup(sft_train_path)
    print(f"  Backed up to {bk.name}")
    n_sft_train = write_jsonl(sft_train_path, combined_sft_train)
    shutil.copy(sft_train_path, FINAL_DIR / "train.jsonl")
    print(f"  -> sft_train.jsonl: {n_sft_train:,}")
    print(f"  -> train.jsonl (copy)")

    # ── SFT valid ──────────────────────────────────────────────────────────────
    sft_valid_path = FINAL_DIR / "sft_valid.jsonl"
    print(f"\nLoading existing {sft_valid_path.name}...")
    existing_sft_valid = load_jsonl(sft_valid_path)
    print(f"  Existing: {len(existing_sft_valid):,}")

    combined_sft_valid = existing_sft_valid + cfr_eval
    rng3 = random.Random(SEED)
    rng3.shuffle(combined_sft_valid)
    print(f"  Combined + shuffled: {len(combined_sft_valid):,}")

    bk2 = backup(sft_valid_path)
    print(f"  Backed up to {bk2.name}")
    n_sft_valid = write_jsonl(sft_valid_path, combined_sft_valid)
    shutil.copy(sft_valid_path, FINAL_DIR / "valid.jsonl")
    print(f"  -> sft_valid.jsonl: {n_sft_valid:,}")
    print(f"  -> valid.jsonl (copy)")

    # ── GRPO train ─────────────────────────────────────────────────────────────
    grpo_train_path = FINAL_DIR / "grpo_train.jsonl"
    print(f"\nLoading existing {grpo_train_path.name}...")
    existing_grpo_train = load_jsonl(grpo_train_path)
    print(f"  Existing: {len(existing_grpo_train):,}")

    cfr_grpo_train = [sft_to_grpo(r) for r in cfr_train]
    combined_grpo_train = existing_grpo_train + cfr_grpo_train
    rng4 = random.Random(SEED)
    rng4.shuffle(combined_grpo_train)
    print(f"  Combined + shuffled: {len(combined_grpo_train):,}")

    bk3 = backup(grpo_train_path)
    print(f"  Backed up to {bk3.name}")
    n_grpo_train = write_jsonl(grpo_train_path, combined_grpo_train)
    print(f"  -> grpo_train.jsonl: {n_grpo_train:,}")

    # ── GRPO valid ─────────────────────────────────────────────────────────────
    grpo_valid_path = FINAL_DIR / "grpo_valid.jsonl"
    print(f"\nLoading existing {grpo_valid_path.name}...")
    existing_grpo_valid = load_jsonl(grpo_valid_path)
    print(f"  Existing: {len(existing_grpo_valid):,}")

    cfr_grpo_eval = [sft_to_grpo(r) for r in cfr_eval]
    combined_grpo_valid = existing_grpo_valid + cfr_grpo_eval
    rng5 = random.Random(SEED)
    rng5.shuffle(combined_grpo_valid)
    print(f"  Combined + shuffled: {len(combined_grpo_valid):,}")

    bk4 = backup(grpo_valid_path)
    print(f"  Backed up to {bk4.name}")
    n_grpo_valid = write_jsonl(grpo_valid_path, combined_grpo_valid)
    print(f"  -> grpo_valid.jsonl: {n_grpo_valid:,}")

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest_path = FINAL_DIR / "MANIFEST.md"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    addition = "\n".join([
        "",
        f"## CFR Merge — {now}",
        "",
        f"- Source: `{CFR_DEDUPED}`",
        f"- CFR records (deduped): {len(cfr_all):,}",
        f"- CFR train split (90%): {len(cfr_train):,}",
        f"- CFR eval  split (10%): {len(cfr_eval):,}",
        f"- Weight: 1x (no upsampling)",
        "",
        "### Updated File Sizes",
        "",
        "| File | Records |",
        "| ---- | ------: |",
        f"| `sft_train.jsonl` | {n_sft_train:,} |",
        f"| `sft_valid.jsonl` | {n_sft_valid:,} |",
        f"| `grpo_train.jsonl` | {n_grpo_train:,} |",
        f"| `grpo_valid.jsonl` | {n_grpo_valid:,} |",
        f"| `train.jsonl` | {n_sft_train:,} (copy of sft_train) |",
        f"| `valid.jsonl` | {n_sft_valid:,} (copy of sft_valid) |",
        "",
        "- Split seed: 42  |  Shuffle seed: 42",
        "- Dedup: 283 duplicates removed, 6,118 disclaimers added",
        "",
    ])
    with open(manifest_path, "a") as fh:
        fh.write(addition)
    print(f"\n  Updated {manifest_path.name}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Merge complete — final counts")
    print("=" * 60)
    print(f"  sft_train.jsonl:  {n_sft_train:,}")
    print(f"  sft_valid.jsonl:  {n_sft_valid:,}")
    print(f"  grpo_train.jsonl: {n_grpo_train:,}")
    print(f"  grpo_valid.jsonl: {n_grpo_valid:,}")
    print(f"  train.jsonl:      {n_sft_train:,} (compat copy)")
    print(f"  valid.jsonl:      {n_sft_valid:,} (compat copy)")
    print("=" * 60)


if __name__ == "__main__":
    main()
