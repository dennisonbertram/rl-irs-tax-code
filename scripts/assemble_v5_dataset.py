#!/usr/bin/env python3
"""
Assemble v5 training dataset from all available SFT and DPO sources.

Applies inflation upsampling, deduplication, train/valid splits,
and outputs GRPO prompts alongside SFT/DPO splits.
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

SFT_SOURCES = [
    {
        "path": "data/processed/grounded_sft_full.jsonl",
        "name": "grounded_sft_full",
        "description": "IRC grounded",
        "expected_count": 16909,
        "is_inflation": False,
    },
    {
        "path": "data/processed/grounded_cfr_sft_deduped.jsonl",
        "name": "grounded_cfr_sft_deduped",
        "description": "CFR grounded",
        "expected_count": 45855,
        "is_inflation": False,
    },
    {
        "path": "data/processed/bulk_sft_full.jsonl",
        "name": "bulk_sft_full",
        "description": "Tavily bulk batch",
        "expected_count": 15398,
        "is_inflation": False,
    },
    {
        "path": "data/processed/tavily_sft_full.jsonl",
        "name": "tavily_sft_full",
        "description": "Tavily original",
        "expected_count": 4077,
        "is_inflation": False,
    },
    {
        "path": "data/processed/inflation_sft_v2.jsonl",
        "name": "inflation_sft_v2",
        "description": "inflation batch",
        "expected_count": 1359,
        "is_inflation": True,
    },
    {
        "path": "data/processed/inflation_adjusted_sft.jsonl",
        "name": "inflation_adjusted_sft",
        "description": "inflation v1",
        "expected_count": 70,
        "is_inflation": True,
    },
]

DPO_SOURCES = [
    {
        "path": "data/processed/grounded_dpo_full.jsonl",
        "name": "grounded_dpo_full",
        "description": "IRC grounded",
        "expected_count": 1719,
        "is_inflation": False,
    },
    {
        "path": "data/processed/bulk_dpo_full.jsonl",
        "name": "bulk_dpo_full",
        "description": "Tavily bulk batch",
        "expected_count": 7610,
        "is_inflation": False,
    },
    {
        "path": "data/processed/inflation_dpo_v2.jsonl",
        "name": "inflation_dpo_v2",
        "description": "inflation batch",
        "expected_count": 810,
        "is_inflation": True,
    },
    {
        "path": "data/processed/inflation_adjusted_dpo.jsonl",
        "name": "inflation_adjusted_dpo",
        "description": "inflation v1",
        "expected_count": 17,
        "is_inflation": True,
    },
    {
        "path": "data/processed/onpolicy_dpo_v2.jsonl",
        "name": "onpolicy_dpo_v2",
        "description": "on-policy",
        "expected_count": 86,
        "is_inflation": False,
    },
    {
        "path": "data/processed/tavily_dpo_full.jsonl",
        "name": "tavily_dpo_full",
        "description": "Tavily original",
        "expected_count": 56,
        "is_inflation": False,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed line {line_no} in {path}: {e}")
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_sft_user_message(record: dict) -> str | None:
    """Extract the first user message text from an SFT record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def validate_sft(record: dict) -> bool:
    """Validate an SFT record has required structure."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = {m.get("role") for m in messages}
    return "user" in roles and "assistant" in roles


def validate_dpo(record: dict) -> bool:
    """Validate a DPO record has required keys."""
    return all(k in record for k in ("prompt", "chosen", "rejected"))


# ---------------------------------------------------------------------------
# Core assembly logic
# ---------------------------------------------------------------------------

def load_and_validate_sft(sources: list[dict], base_dir: str) -> tuple[dict, list[dict]]:
    """
    Load all SFT sources.
    Returns (source_counts dict, list of (record, source_name, is_inflation)).
    """
    source_counts = {}
    all_records = []  # list of (record, source_name, is_inflation)

    for src in sources:
        path = os.path.join(base_dir, src["path"])
        if not os.path.exists(path):
            print(f"  ERROR: missing file {path}")
            source_counts[src["name"]] = {"loaded": 0, "valid": 0, "missing": True}
            continue

        raw = load_jsonl(path)
        valid = [r for r in raw if validate_sft(r)]
        invalid = len(raw) - len(valid)

        print(
            f"  {src['name']}: loaded {len(raw):,}  valid {len(valid):,}"
            + (f"  ({invalid} invalid skipped)" if invalid else "")
        )

        source_counts[src["name"]] = {
            "loaded": len(raw),
            "valid": len(valid),
            "is_inflation": src["is_inflation"],
            "description": src["description"],
        }
        for r in valid:
            all_records.append((r, src["name"], src["is_inflation"]))

    return source_counts, all_records


def load_and_validate_dpo(sources: list[dict], base_dir: str) -> tuple[dict, list[dict]]:
    """Load all DPO sources."""
    source_counts = {}
    all_records = []

    for src in sources:
        path = os.path.join(base_dir, src["path"])
        if not os.path.exists(path):
            print(f"  ERROR: missing file {path}")
            source_counts[src["name"]] = {"loaded": 0, "valid": 0, "missing": True}
            continue

        raw = load_jsonl(path)
        valid = [r for r in raw if validate_dpo(r)]
        invalid = len(raw) - len(valid)

        print(
            f"  {src['name']}: loaded {len(raw):,}  valid {len(valid):,}"
            + (f"  ({invalid} invalid skipped)" if invalid else "")
        )

        source_counts[src["name"]] = {
            "loaded": len(raw),
            "valid": len(valid),
            "is_inflation": src["is_inflation"],
            "description": src["description"],
        }
        for r in valid:
            all_records.append((r, src["name"], src["is_inflation"]))

    return source_counts, all_records


def deduplicate_sft(records: list[tuple]) -> tuple[list[tuple], dict]:
    """Deduplicate SFT records by user message text. Keep first occurrence."""
    seen = set()
    deduped = []
    dup_count = 0

    for record, source, is_inflation in records:
        key = get_sft_user_message(record)
        if key is None:
            continue
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append((record, source, is_inflation))

    stats = {"before": len(records), "after": len(deduped), "removed": dup_count}
    return deduped, stats


def _dpo_field_key(value) -> str:
    """Convert a DPO field value (str, list of messages, or other) to a hashable string key."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # List of message dicts (chat format) — serialize to stable string
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


# Keep the old name as an alias for backward compatibility
_dpo_prompt_key = _dpo_field_key


def deduplicate_dpo(records: list[tuple]) -> tuple[list[tuple], dict]:
    """Deduplicate DPO records by (prompt, chosen, rejected) tuple.

    Fix 4-B (review item 4-B MEDIUM): Previously deduped by prompt text only,
    which discarded alternative hard negatives sharing the same prompt but with
    different chosen/rejected pairs.  Now we keep all unique (prompt, chosen,
    rejected) triples so contrastive signal is preserved.

    Fix (Bug 2): chosen/rejected fields can be lists of message dicts, which are
    unhashable and cannot be used as dict/set keys directly. _dpo_field_key()
    serialises any list to a stable JSON string before hashing.
    """
    seen = set()
    deduped = []
    dup_count = 0

    for record, source, is_inflation in records:
        prompt_key = _dpo_field_key(record.get("prompt", ""))
        chosen_key = _dpo_field_key(record.get("chosen", ""))
        rejected_key = _dpo_field_key(record.get("rejected", ""))
        # Deduplicate on the full (prompt, chosen, rejected) triple
        key = (prompt_key, chosen_key, rejected_key)
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append((record, source, is_inflation))

    stats = {"before": len(records), "after": len(deduped), "removed": dup_count}
    return deduped, stats


def apply_inflation_upsampling(records: list[tuple], multiplier: int) -> list[tuple]:
    """
    Duplicate all inflation records `multiplier` times.
    The original copies are already present so we add (multiplier - 1) extra copies.
    """
    upsampled = []
    extra_count = 0

    for record, source, is_inflation in records:
        upsampled.append((record, source, is_inflation))
        if is_inflation:
            for _ in range(multiplier - 1):
                upsampled.append((record, source, is_inflation))
                extra_count += 1

    return upsampled, extra_count


def train_valid_split(records: list[tuple], ratio: float, seed: int) -> tuple[list, list]:
    """Shuffle and split records into train/valid."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def make_grpo_record(sft_record: dict) -> dict:
    """Extract just the user message as a GRPO prompt record."""
    for msg in sft_record.get("messages", []):
        if msg.get("role") == "user":
            return {"messages": [{"role": "user", "content": msg["content"]}]}
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble v5 training dataset from all available SFT and DPO sources."
    )
    parser.add_argument(
        "--output-dir",
        default="data/v5/",
        help="Output directory for assembled dataset (default: data/v5/)",
    )
    parser.add_argument(
        "--inflation-multiplier",
        type=int,
        default=20,
        # Fix 4-C (review item 4-C LOW): 20x is aggressive (inflation records will
        # dominate >35% of tokens).  Monitor class imbalance and reduce if the model
        # over-indexes on inflation scenarios.  Default kept at 20 for compatibility.
        help="Inflation upsampling multiplier (default: 20, NOTE: 20x is aggressive — monitor class balance)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
        help="Train/valid split ratio (default: 0.9)",
    )
    args = parser.parse_args()

    # Resolve paths relative to script location's parent (repo root)
    script_dir = Path(__file__).resolve().parent
    base_dir = str(script_dir.parent)
    output_dir = Path(base_dir) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v5 Dataset Assembly")
    print("=" * 70)
    print(f"  Output dir       : {output_dir}")
    print(f"  Inflation mult   : {args.inflation_multiplier}x")
    print(f"  Random seed      : {args.seed}")
    print(f"  Train/valid split: {args.split_ratio:.0%} / {1-args.split_ratio:.0%}")
    print()

    report = {
        "config": {
            "output_dir": str(output_dir),
            "inflation_multiplier": args.inflation_multiplier,
            "seed": args.seed,
            "split_ratio": args.split_ratio,
        },
        "sft": {},
        "dpo": {},
    }

    # ------------------------------------------------------------------
    # SFT pipeline
    # Fix 1-C (review item 1-C HIGH): Perform train/valid split FIRST on
    # the deduplicated base records, THEN upsample inflation records within
    # each partition separately.  Previously upsampling happened before the
    # split, which allowed copies of the same record to appear in both train
    # and valid sets, making validation non-independent (data leakage).
    # ------------------------------------------------------------------
    print("--- SFT Sources ---")
    sft_source_counts, sft_records = load_and_validate_sft(SFT_SOURCES, base_dir)
    print(f"  Total raw SFT records: {len(sft_records):,}")
    print()

    print("--- SFT Deduplication ---")
    sft_deduped, sft_dedup_stats = deduplicate_sft(sft_records)
    print(
        f"  Before: {sft_dedup_stats['before']:,}  "
        f"After: {sft_dedup_stats['after']:,}  "
        f"Removed: {sft_dedup_stats['removed']:,}"
    )
    print()

    # Split on BASE records first (before upsampling) to prevent leakage
    print("--- SFT Train/Valid Split (on base records, before upsampling) ---")
    sft_train_base, sft_valid_base = train_valid_split(
        sft_deduped, args.split_ratio, args.seed
    )
    print(
        f"  Base train: {len(sft_train_base):,}  "
        f"Base valid: {len(sft_valid_base):,}  "
        f"Ratio: {len(sft_train_base)/len(sft_deduped):.3f}"
    )
    print()

    # Now upsample inflation records WITHIN each partition
    print("--- SFT Inflation Upsampling (within each partition) ---")
    inflation_sft_base = sum(1 for _, _, is_inf in sft_deduped if is_inf)
    sft_train_tuples, sft_train_extra = apply_inflation_upsampling(
        sft_train_base, args.inflation_multiplier
    )
    sft_valid_tuples, sft_valid_extra = apply_inflation_upsampling(
        sft_valid_base, args.inflation_multiplier
    )
    sft_extra = sft_train_extra + sft_valid_extra
    sft_total = len(sft_train_tuples) + len(sft_valid_tuples)
    print(
        f"  Inflation records (base): {inflation_sft_base:,}  "
        f"Extra copies added (total): {sft_extra:,}  "
        f"Total SFT after upsampling: {sft_total:,}"
    )
    print(
        f"  Train: {len(sft_train_tuples):,}  "
        f"Valid: {len(sft_valid_tuples):,}"
    )
    print()

    # Extract just records (drop source/is_inflation metadata for output)
    sft_train = [r for r, _, _ in sft_train_tuples]
    sft_valid = [r for r, _, _ in sft_valid_tuples]
    sft_upsampled_total = sft_total  # used in report below

    # ------------------------------------------------------------------
    # DPO pipeline
    # Fix 1-C (review item 1-C HIGH): Same split-before-upsample fix as SFT.
    # ------------------------------------------------------------------
    print("--- DPO Sources ---")
    dpo_source_counts, dpo_records = load_and_validate_dpo(DPO_SOURCES, base_dir)
    print(f"  Total raw DPO records: {len(dpo_records):,}")
    print()

    print("--- DPO Deduplication ---")
    dpo_deduped, dpo_dedup_stats = deduplicate_dpo(dpo_records)
    print(
        f"  Before: {dpo_dedup_stats['before']:,}  "
        f"After: {dpo_dedup_stats['after']:,}  "
        f"Removed: {dpo_dedup_stats['removed']:,}"
    )
    print()

    # Split on BASE records first (before upsampling) to prevent leakage
    print("--- DPO Train/Valid Split (on base records, before upsampling) ---")
    dpo_train_base, dpo_valid_base = train_valid_split(
        dpo_deduped, args.split_ratio, args.seed
    )
    print(
        f"  Base train: {len(dpo_train_base):,}  "
        f"Base valid: {len(dpo_valid_base):,}  "
        f"Ratio: {len(dpo_train_base)/len(dpo_deduped):.3f}"
    )
    print()

    # Upsample inflation records WITHIN each partition
    print("--- DPO Inflation Upsampling (within each partition) ---")
    inflation_dpo_base = sum(1 for _, _, is_inf in dpo_deduped if is_inf)
    dpo_train_tuples, dpo_train_extra = apply_inflation_upsampling(
        dpo_train_base, args.inflation_multiplier
    )
    dpo_valid_tuples, dpo_valid_extra = apply_inflation_upsampling(
        dpo_valid_base, args.inflation_multiplier
    )
    dpo_extra = dpo_train_extra + dpo_valid_extra
    dpo_total = len(dpo_train_tuples) + len(dpo_valid_tuples)
    print(
        f"  Inflation records (base): {inflation_dpo_base:,}  "
        f"Extra copies added (total): {dpo_extra:,}  "
        f"Total DPO after upsampling: {dpo_total:,}"
    )
    print(
        f"  Train: {len(dpo_train_tuples):,}  "
        f"Valid: {len(dpo_valid_tuples):,}"
    )
    print()

    dpo_train = [r for r, _, _ in dpo_train_tuples]
    dpo_valid = [r for r, _, _ in dpo_valid_tuples]
    dpo_upsampled_total = dpo_total  # used in report below

    # ------------------------------------------------------------------
    # GRPO prompts
    # ------------------------------------------------------------------
    print("--- GRPO Prompts ---")
    grpo_train = [g for r in sft_train if (g := make_grpo_record(r)) is not None]
    grpo_valid = [g for r in sft_valid if (g := make_grpo_record(r)) is not None]
    print(f"  GRPO train: {len(grpo_train):,}  GRPO valid: {len(grpo_valid):,}")
    print()

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    print("--- Writing Output Files ---")
    files = {
        "sft_train.jsonl": sft_train,
        "sft_valid.jsonl": sft_valid,
        "dpo_train.jsonl": dpo_train,
        "dpo_valid.jsonl": dpo_valid,
        "grpo_train.jsonl": grpo_train,
        "grpo_valid.jsonl": grpo_valid,
    }

    for filename, records in files.items():
        out_path = output_dir / filename
        write_jsonl(records, str(out_path))
        print(f"  Wrote {len(records):,} records -> {out_path}")

    # Compatibility copies
    shutil.copy(output_dir / "sft_train.jsonl", output_dir / "train.jsonl")
    shutil.copy(output_dir / "sft_valid.jsonl", output_dir / "valid.jsonl")
    print(f"  Copied sft_train.jsonl -> train.jsonl (compatibility)")
    print(f"  Copied sft_valid.jsonl -> valid.jsonl (compatibility)")
    print()

    # ------------------------------------------------------------------
    # Assembly report
    # ------------------------------------------------------------------
    report["sft"] = {
        "sources": sft_source_counts,
        "deduplication": sft_dedup_stats,
        "inflation_base_records": inflation_sft_base,
        "extra_inflation_copies": sft_extra,
        "total_after_upsampling": sft_upsampled_total,
        "train_count": len(sft_train),
        "valid_count": len(sft_valid),
        "actual_split_ratio": len(sft_train) / sft_upsampled_total,
    }
    report["dpo"] = {
        "sources": dpo_source_counts,
        "deduplication": dpo_dedup_stats,
        "inflation_base_records": inflation_dpo_base,
        "extra_inflation_copies": dpo_extra,
        "total_after_upsampling": dpo_upsampled_total,
        "train_count": len(dpo_train),
        "valid_count": len(dpo_valid),
        "actual_split_ratio": len(dpo_train) / dpo_upsampled_total,
    }
    report["grpo"] = {
        "train_count": len(grpo_train),
        "valid_count": len(grpo_valid),
    }
    report["output_files"] = {
        "sft_train.jsonl": len(sft_train),
        "sft_valid.jsonl": len(sft_valid),
        "dpo_train.jsonl": len(dpo_train),
        "dpo_valid.jsonl": len(dpo_valid),
        "grpo_train.jsonl": len(grpo_train),
        "grpo_valid.jsonl": len(grpo_valid),
        "train.jsonl": len(sft_train),
        "valid.jsonl": len(sft_valid),
    }

    report_path = output_dir / "assembly_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Assembly report -> {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Assembly Complete — Summary")
    print("=" * 70)
    print(f"  SFT total (after dedup + upsample) : {sft_upsampled_total:,}")
    print(f"    train  : {len(sft_train):,}")
    print(f"    valid  : {len(sft_valid):,}")
    print(f"  DPO total (after dedup + upsample) : {dpo_upsampled_total:,}")
    print(f"    train  : {len(dpo_train):,}")
    print(f"    valid  : {len(dpo_valid):,}")
    print(f"  GRPO train : {len(grpo_train):,}")
    print(f"  GRPO valid : {len(grpo_valid):,}")
    print()


if __name__ == "__main__":
    main()
