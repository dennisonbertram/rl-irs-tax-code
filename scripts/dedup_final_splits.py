#!/usr/bin/env python3
"""
dedup_final_splits.py

Fix duplicate records and train/eval leakage in final training splits.

Actions:
  1. Deduplicate sft_train.jsonl (preserve intentional 5x inflation_adjusted records)
  2. Deduplicate grpo_train.jsonl
  3. Remove leaking records from sft_valid, dpo_valid, grpo_valid
  4. Update compatibility copies: train.jsonl, valid.jsonl
  5. Report statistics

Usage:
    python scripts/dedup_final_splits.py
"""

import json
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "final"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_sft_question(rec: dict) -> str | None:
    for msg in rec.get("messages", []):
        if msg.get("role") == "user":
            return msg["content"]
    return None


def dedup_sft_train(path: Path) -> tuple[list[dict], int]:
    """
    Deduplicate SFT train records by question text.

    Inflation-adjusted records (metadata.category == 'inflation_adjusted') are
    excluded from dedup because their repetition is intentional (5x upsample).
    For all other records, keep the first occurrence.

    Returns (deduped_records, n_removed).
    """
    records = load_jsonl(path)
    before = len(records)

    seen_non_inflation: set[str] = set()
    result = []

    for rec in records:
        is_inflation = rec.get("metadata", {}).get("category") == "inflation_adjusted"
        q = get_sft_question(rec)

        if is_inflation:
            # Always keep — intentional duplicate
            result.append(rec)
        else:
            if q is None or q not in seen_non_inflation:
                result.append(rec)
                if q is not None:
                    seen_non_inflation.add(q)
            # else: duplicate non-inflation record — skip

    n_removed = before - len(result)
    return result, n_removed


def dedup_grpo_train(path: Path) -> tuple[list[dict], int]:
    """
    Deduplicate GRPO train records by prompt text (keep first occurrence).

    Returns (deduped_records, n_removed).
    """
    records = load_jsonl(path)
    before = len(records)

    seen: set[str] = set()
    result = []

    for rec in records:
        prompt = rec.get("prompt", "")
        if prompt not in seen:
            result.append(rec)
            seen.add(prompt)

    n_removed = before - len(result)
    return result, n_removed


def remove_leaking_records_sft(train_path: Path, valid_path: Path) -> tuple[list[dict], int]:
    """
    Remove records from valid whose question also appears in train.
    Returns (cleaned_valid_records, n_removed).
    """
    train_qs: set[str] = set()
    for rec in load_jsonl(train_path):
        q = get_sft_question(rec)
        if q:
            train_qs.add(q)

    valid_records = load_jsonl(valid_path)
    before = len(valid_records)

    result = []
    for rec in valid_records:
        q = get_sft_question(rec)
        if q not in train_qs:
            result.append(rec)

    n_removed = before - len(result)
    return result, n_removed


def remove_leaking_records_prompt(train_path: Path, valid_path: Path) -> tuple[list[dict], int]:
    """
    Remove records from valid whose 'prompt' also appears in train.
    Returns (cleaned_valid_records, n_removed).
    """
    train_prompts: set[str] = set()
    for rec in load_jsonl(train_path):
        p = rec.get("prompt", "")
        if p:
            train_prompts.add(p)

    valid_records = load_jsonl(valid_path)
    before = len(valid_records)

    result = []
    for rec in valid_records:
        p = rec.get("prompt", "")
        if p not in train_prompts:
            result.append(rec)

    n_removed = before - len(result)
    return result, n_removed


def verify_inflation_count(records: list[dict]) -> int:
    return sum(
        1 for rec in records
        if rec.get("metadata", {}).get("category") == "inflation_adjusted"
    )


def verify_no_duplicates_sft(records: list[dict]) -> tuple[int, int]:
    """Return (n_dup_questions, n_dup_records) for non-inflation records."""
    from collections import Counter
    qs = [
        get_sft_question(rec)
        for rec in records
        if rec.get("metadata", {}).get("category") != "inflation_adjusted"
        and get_sft_question(rec) is not None
    ]
    counter = Counter(qs)
    dup_qs = sum(1 for c in counter.values() if c > 1)
    dup_recs = sum(c - 1 for c in counter.values() if c > 1)
    return dup_qs, dup_recs


def verify_no_duplicates_prompt(records: list[dict]) -> tuple[int, int]:
    """Return (n_dup_prompts, n_dup_records)."""
    from collections import Counter
    prompts = [rec.get("prompt", "") for rec in records]
    counter = Counter(prompts)
    dup_qs = sum(1 for c in counter.values() if c > 1)
    dup_recs = sum(c - 1 for c in counter.values() if c > 1)
    return dup_qs, dup_recs


def verify_no_leakage_sft(train_records: list[dict], valid_records: list[dict]) -> int:
    train_qs = {get_sft_question(r) for r in train_records if get_sft_question(r)}
    return sum(1 for r in valid_records if get_sft_question(r) in train_qs)


def verify_no_leakage_prompt(train_records: list[dict], valid_records: list[dict]) -> int:
    train_prompts = {r.get("prompt", "") for r in train_records}
    return sum(1 for r in valid_records if r.get("prompt", "") in train_prompts)


def main() -> None:
    print("=" * 60)
    print("dedup_final_splits.py")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Deduplicate sft_train.jsonl                                       #
    # ------------------------------------------------------------------ #
    sft_train_path = DATA_DIR / "sft_train.jsonl"
    print(f"\n[1] Deduplicating {sft_train_path.name}")
    sft_train_orig_count = sum(1 for _ in open(sft_train_path))
    sft_train_records, sft_train_removed = dedup_sft_train(sft_train_path)
    print(f"    Before : {sft_train_orig_count:,}")
    print(f"    Removed: {sft_train_removed:,}")
    print(f"    After  : {len(sft_train_records):,}")
    inflation_after = verify_inflation_count(sft_train_records)
    print(f"    Inflation-adjusted records retained: {inflation_after} (expected 350)")
    assert inflation_after == 350, f"Expected 350 inflation records, got {inflation_after}"

    # ------------------------------------------------------------------ #
    # 2. Deduplicate grpo_train.jsonl                                      #
    # ------------------------------------------------------------------ #
    grpo_train_path = DATA_DIR / "grpo_train.jsonl"
    print(f"\n[2] Deduplicating {grpo_train_path.name}")
    grpo_train_orig_count = sum(1 for _ in open(grpo_train_path))
    grpo_train_records, grpo_train_removed = dedup_grpo_train(grpo_train_path)
    print(f"    Before : {grpo_train_orig_count:,}")
    print(f"    Removed: {grpo_train_removed:,}")
    print(f"    After  : {len(grpo_train_records):,}")

    # ------------------------------------------------------------------ #
    # 3a. Fix SFT leakage: write sft_train first, then filter valid        #
    # ------------------------------------------------------------------ #
    print(f"\n[3a] Writing deduped sft_train.jsonl")
    write_jsonl(sft_train_path, sft_train_records)
    print(f"     Written {len(sft_train_records):,} records")

    sft_valid_path = DATA_DIR / "sft_valid.jsonl"
    print(f"\n[3b] Removing SFT leakage from {sft_valid_path.name}")
    sft_valid_orig_count = sum(1 for _ in open(sft_valid_path))
    sft_valid_records, sft_valid_removed = remove_leaking_records_sft(sft_train_path, sft_valid_path)
    print(f"    Before : {sft_valid_orig_count:,}")
    print(f"    Removed: {sft_valid_removed:,}")
    print(f"    After  : {len(sft_valid_records):,}")
    write_jsonl(sft_valid_path, sft_valid_records)

    # ------------------------------------------------------------------ #
    # 3c. Fix DPO leakage                                                  #
    # ------------------------------------------------------------------ #
    dpo_train_path = DATA_DIR / "dpo_train.jsonl"
    dpo_valid_path = DATA_DIR / "dpo_valid.jsonl"
    print(f"\n[3c] Removing DPO leakage from {dpo_valid_path.name}")
    dpo_valid_orig_count = sum(1 for _ in open(dpo_valid_path))
    dpo_valid_records, dpo_valid_removed = remove_leaking_records_prompt(dpo_train_path, dpo_valid_path)
    print(f"    Before : {dpo_valid_orig_count:,}")
    print(f"    Removed: {dpo_valid_removed:,}")
    print(f"    After  : {len(dpo_valid_records):,}")
    write_jsonl(dpo_valid_path, dpo_valid_records)

    # ------------------------------------------------------------------ #
    # 3d. Fix GRPO leakage: write grpo_train first, then filter valid      #
    # ------------------------------------------------------------------ #
    print(f"\n[3d] Writing deduped grpo_train.jsonl")
    write_jsonl(grpo_train_path, grpo_train_records)
    print(f"     Written {len(grpo_train_records):,} records")

    grpo_valid_path = DATA_DIR / "grpo_valid.jsonl"
    print(f"\n[3e] Removing GRPO leakage from {grpo_valid_path.name}")
    grpo_valid_orig_count = sum(1 for _ in open(grpo_valid_path))
    grpo_valid_records, grpo_valid_removed = remove_leaking_records_prompt(grpo_train_path, grpo_valid_path)
    print(f"    Before : {grpo_valid_orig_count:,}")
    print(f"    Removed: {grpo_valid_removed:,}")
    print(f"    After  : {len(grpo_valid_records):,}")
    write_jsonl(grpo_valid_path, grpo_valid_records)

    # ------------------------------------------------------------------ #
    # 4. Update compatibility copies                                        #
    # ------------------------------------------------------------------ #
    print(f"\n[4] Updating compatibility copies")
    shutil.copy2(sft_train_path, DATA_DIR / "train.jsonl")
    print(f"    train.jsonl <- sft_train.jsonl ({len(sft_train_records):,} records)")
    shutil.copy2(sft_valid_path, DATA_DIR / "valid.jsonl")
    print(f"    valid.jsonl <- sft_valid.jsonl ({len(sft_valid_records):,} records)")

    # ------------------------------------------------------------------ #
    # 5. Verification                                                       #
    # ------------------------------------------------------------------ #
    print(f"\n[5] Verification")

    # Re-load from disk for safety
    sft_train_final = load_jsonl(sft_train_path)
    grpo_train_final = load_jsonl(grpo_train_path)
    sft_valid_final = load_jsonl(sft_valid_path)
    dpo_train_final = load_jsonl(dpo_train_path)
    dpo_valid_final = load_jsonl(dpo_valid_path)
    grpo_valid_final = load_jsonl(grpo_valid_path)

    dup_qs, dup_recs = verify_no_duplicates_sft(sft_train_final)
    print(f"    SFT train non-inflation dups remaining: {dup_qs} questions, {dup_recs} records")
    assert dup_recs == 0, f"SFT train still has {dup_recs} duplicate records!"

    dup_qs, dup_recs = verify_no_duplicates_prompt(grpo_train_final)
    print(f"    GRPO train dups remaining: {dup_qs} questions, {dup_recs} records")
    assert dup_recs == 0, f"GRPO train still has {dup_recs} duplicate records!"

    sft_leak = verify_no_leakage_sft(sft_train_final, sft_valid_final)
    print(f"    SFT leakage remaining: {sft_leak}")
    assert sft_leak == 0, f"SFT still has {sft_leak} leaking records!"

    dpo_leak = verify_no_leakage_prompt(dpo_train_final, dpo_valid_final)
    print(f"    DPO leakage remaining: {dpo_leak}")
    assert dpo_leak == 0, f"DPO still has {dpo_leak} leaking records!"

    grpo_leak = verify_no_leakage_prompt(grpo_train_final, grpo_valid_final)
    print(f"    GRPO leakage remaining: {grpo_leak}")
    assert grpo_leak == 0, f"GRPO still has {grpo_leak} leaking records!"

    infl = verify_inflation_count(sft_train_final)
    print(f"    Inflation records in sft_train: {infl} (expected 350)")
    assert infl == 350, f"Expected 350 inflation records, got {infl}!"

    # ------------------------------------------------------------------ #
    # 6. Summary                                                            #
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'File':<25} {'Before':>8} {'After':>8} {'Removed':>8}")
    print("-" * 55)
    print(f"{'sft_train.jsonl':<25} {sft_train_orig_count:>8,} {len(sft_train_final):>8,} {sft_train_orig_count - len(sft_train_final):>8,}")
    print(f"{'sft_valid.jsonl':<25} {sft_valid_orig_count:>8,} {len(sft_valid_final):>8,} {sft_valid_orig_count - len(sft_valid_final):>8,}")
    print(f"{'dpo_valid.jsonl':<25} {dpo_valid_orig_count:>8,} {len(dpo_valid_final):>8,} {dpo_valid_orig_count - len(dpo_valid_final):>8,}")
    print(f"{'grpo_train.jsonl':<25} {grpo_train_orig_count:>8,} {len(grpo_train_final):>8,} {grpo_train_orig_count - len(grpo_train_final):>8,}")
    print(f"{'grpo_valid.jsonl':<25} {grpo_valid_orig_count:>8,} {len(grpo_valid_final):>8,} {grpo_valid_orig_count - len(grpo_valid_final):>8,}")
    print(f"\nAll checks passed. Files written to {DATA_DIR}")

    return {
        "sft_train": {"before": sft_train_orig_count, "after": len(sft_train_final)},
        "sft_valid": {"before": sft_valid_orig_count, "after": len(sft_valid_final)},
        "dpo_valid": {"before": dpo_valid_orig_count, "after": len(dpo_valid_final)},
        "grpo_train": {"before": grpo_train_orig_count, "after": len(grpo_train_final)},
        "grpo_valid": {"before": grpo_valid_orig_count, "after": len(grpo_valid_final)},
    }


if __name__ == "__main__":
    main()
