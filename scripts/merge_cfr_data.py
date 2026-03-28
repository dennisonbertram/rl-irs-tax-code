#!/usr/bin/env python3
"""
merge_cfr_data.py

Merges CFR (Code of Federal Regulations) data into the existing final splits.
Run this after CFR data generation completes.

Usage:
  python scripts/merge_cfr_data.py [--cfr-sft PATH] [--cfr-weight FLOAT] [--seed INT]

Defaults:
  --cfr-sft    data/processed/grounded_cfr_sft_full.jsonl
  --cfr-weight 1.0  (1x — no upsampling by default)
  --seed       42
"""

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/dennisonbertram/Develop/rl-irs-tax-code")
DATA_ROOT    = PROJECT_ROOT / "data"
FINAL_DIR    = DATA_ROOT / "final"

DEFAULT_CFR_SFT = DATA_ROOT / "processed" / "grounded_cfr_sft_full.jsonl"


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> int:
    """Write records to a JSONL file; returns record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def apply_weight(records: list[dict], weight: float) -> list[dict]:
    """
    Apply importance weight to a list of records by upsampling.
    Weight < 1.0 is treated as 1x (no downsampling).
    Weight 2.5 → each record appears 2 or 3 times (probabilistic rounding).
    """
    if weight <= 1.0:
        return list(records)

    result = []
    rng = random.Random(999)  # stable secondary seed
    for rec in records:
        # Always include at least floor(weight) copies
        floor_copies = int(weight)
        result.extend([rec] * floor_copies)
        # Probabilistically add one more copy for fractional part
        frac = weight - floor_copies
        if frac > 0 and rng.random() < frac:
            result.append(rec)
    return result


def validate_sft_record(rec: dict) -> bool:
    """Check that a record has the required SFT format."""
    if "messages" not in rec:
        return False
    msgs = rec["messages"]
    roles = [m.get("role") for m in msgs]
    return "user" in roles and "assistant" in roles


def update_manifest(cfr_count_raw: int, cfr_count_up: int, weight: float,
                    new_train_count: int, cfr_path: Path) -> None:
    """Append a CFR merge section to MANIFEST.md."""
    manifest_path = FINAL_DIR / "MANIFEST.md"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    addition = [
        "",
        f"## CFR Merge — {now}",
        "",
        f"- Source: `{cfr_path}`",
        f"- CFR records (raw): {cfr_count_raw:,}",
        f"- CFR weight: {weight}x → {cfr_count_up:,} records after weighting",
        f"- New sft_train.jsonl size: {new_train_count:,} records",
        "- Re-shuffled with seed=42",
        "",
    ]

    with open(manifest_path, "a") as fh:
        fh.write("\n".join(addition))

    print(f"  Updated {manifest_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge CFR data into final training splits")
    parser.add_argument(
        "--cfr-sft",
        type=Path,
        default=DEFAULT_CFR_SFT,
        help=f"Path to grounded CFR SFT JSONL (default: {DEFAULT_CFR_SFT})",
    )
    parser.add_argument(
        "--cfr-weight",
        type=float,
        default=1.0,
        help="Importance weight for CFR records (default: 1.0 = no upsampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42)",
    )
    args = parser.parse_args()

    # Validate input
    if not args.cfr_sft.exists():
        print(f"ERROR: CFR SFT file not found: {args.cfr_sft}")
        print("       Re-run once CFR data generation is complete.")
        raise SystemExit(1)

    if not (FINAL_DIR / "sft_train.jsonl").exists():
        print(f"ERROR: Final directory missing. Run assemble_final_splits.py first.")
        raise SystemExit(1)

    print("=" * 60)
    print("Merging CFR data into final SFT split")
    print("=" * 60)

    # Load CFR data
    print(f"\nLoading CFR SFT from {args.cfr_sft}...")
    cfr_records = load_jsonl(args.cfr_sft)
    print(f"  Loaded {len(cfr_records):,} CFR records")

    # Validate format
    invalid = [i for i, r in enumerate(cfr_records) if not validate_sft_record(r)]
    if invalid:
        print(f"  WARNING: {len(invalid)} records failed SFT format validation (indices: {invalid[:10]}...)")
        cfr_records = [r for i, r in enumerate(cfr_records) if i not in invalid]
        print(f"  Filtered to {len(cfr_records):,} valid records")

    # Apply importance weighting
    cfr_weighted = apply_weight(cfr_records, args.cfr_weight)
    print(f"  After {args.cfr_weight}x weighting: {len(cfr_weighted):,} records")

    # Load existing SFT train
    print("\nLoading existing sft_train.jsonl...")
    existing = load_jsonl(FINAL_DIR / "sft_train.jsonl")
    print(f"  Existing records: {len(existing):,}")

    # Combine and shuffle
    combined = existing + cfr_weighted
    rng = random.Random(args.seed)
    rng.shuffle(combined)
    print(f"  Combined + shuffled: {len(combined):,} records")

    # Backup existing file before overwrite
    backup_path = FINAL_DIR / f"sft_train_pre_cfr_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    shutil.copy(FINAL_DIR / "sft_train.jsonl", backup_path)
    print(f"\n  Backed up original to {backup_path.name}")

    # Write updated files
    n = write_jsonl(FINAL_DIR / "sft_train.jsonl", combined)
    shutil.copy(FINAL_DIR / "sft_train.jsonl", FINAL_DIR / "train.jsonl")
    print(f"  -> Wrote sft_train.jsonl: {n:,} records")
    print(f"  -> Updated train.jsonl (copy)")

    # Update manifest
    update_manifest(
        cfr_count_raw=len(cfr_records),
        cfr_count_up=len(cfr_weighted),
        weight=args.cfr_weight,
        new_train_count=n,
        cfr_path=args.cfr_sft,
    )

    print()
    print("=" * 60)
    print(f"CFR merge complete.")
    print(f"  Records added (after weighting): {len(cfr_weighted):,}")
    print(f"  New sft_train.jsonl size:        {n:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
