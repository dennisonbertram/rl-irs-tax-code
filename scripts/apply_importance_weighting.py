#!/usr/bin/env python3
"""
apply_importance_weighting.py

Apply section-importance weighting to an IRC training JSONL file.

Tier 1 (high traffic):  repeat 3x
Tier 2 (moderate):      keep as-is (1x)
Tier 3 (low traffic):   keep 30% (random, seed=42)

Usage:
    python scripts/apply_importance_weighting.py \
        --input  data/train_v2/sft.jsonl \
        --output data/train_v2/sft_weighted.jsonl \
        --tiers  data/reference/section_tiers.json

The script detects whether the file is SFT format (has "messages" + "metadata")
or GRPO format (has "prompt" + "expected_section") and extracts the section
number accordingly.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Section-number extraction
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"IRC\s*[§§]?\s*(?:Section\s+)?([0-9]+[A-Za-z]*)", re.IGNORECASE)


def extract_section_number(record: dict) -> str | None:
    """
    Return the bare section identifier (e.g. "401", "199A", "408A") from a
    record, or None if it cannot be determined.

    Handles both SFT format (metadata.source_section) and GRPO format
    (expected_section).
    """
    raw: str | None = None

    if "metadata" in record and isinstance(record["metadata"], dict):
        raw = record["metadata"].get("source_section")
    elif "expected_section" in record:
        raw = record["expected_section"]

    if not raw:
        return None

    m = _SECTION_RE.search(raw)
    if m:
        return m.group(1).upper()

    # Fallback: strip any leading non-digit characters after the §
    cleaned = re.sub(r"[^\d]", "", raw)
    return cleaned if cleaned else None


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def load_tiers(tiers_path: str) -> dict:
    with open(tiers_path) as f:
        return json.load(f)


def build_tier1_set(tiers: dict) -> set[str]:
    return {s.upper() for s in tiers.get("tier1", [])}


def build_tier3_ranges(tiers: dict) -> list[tuple[int, int]]:
    return [(lo, hi) for lo, hi in tiers.get("tier3_ranges", [])]


def classify_section(section: str | None, tier1: set, tier3_ranges: list) -> int:
    """Return 1, 2, or 3."""
    if section is None:
        return 2  # default to moderate if unknown

    # Tier 1 check (exact string match including alpha suffix like "408A")
    if section.upper() in tier1:
        return 1

    # Numeric portion for range check
    numeric_match = re.match(r"(\d+)", section)
    if numeric_match:
        num = int(numeric_match.group(1))
        for lo, hi in tier3_ranges:
            if lo <= num <= hi:
                return 3

    return 2


# ---------------------------------------------------------------------------
# Weighting logic
# ---------------------------------------------------------------------------

def apply_weights(
    records: list[dict],
    tier1: set,
    tier3_ranges: list,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """
    Returns (weighted_records, stats_dict).

    Tier 1 → 3 copies
    Tier 2 → 1 copy
    Tier 3 → ~30% kept (random)
    """
    rng = random.Random(seed)

    counts_before = {1: 0, 2: 0, 3: 0}
    counts_after  = {1: 0, 2: 0, 3: 0}
    weighted: list[dict] = []

    for rec in records:
        section = extract_section_number(rec)
        tier = classify_section(section, tier1, tier3_ranges)
        counts_before[tier] += 1

        if tier == 1:
            weighted.extend([rec, rec, rec])
            counts_after[tier] += 3
        elif tier == 2:
            weighted.append(rec)
            counts_after[tier] += 1
        else:  # tier 3
            if rng.random() < 0.30:
                weighted.append(rec)
                counts_after[tier] += 1

    stats = {
        "total_before": len(records),
        "total_after":  len(weighted),
        "before_by_tier": counts_before,
        "after_by_tier":  counts_after,
    }
    return weighted, stats


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed line {line_no}: {e}", file=sys.stderr)
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Statistics reporting
# ---------------------------------------------------------------------------

def print_stats(stats: dict, input_path: str, output_path: str) -> None:
    before = stats["before_by_tier"]
    after  = stats["after_by_tier"]
    total_before = stats["total_before"]
    total_after  = stats["total_after"]

    print(f"\n{'='*60}")
    print(f"  Importance Weighting Statistics")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"{'='*60}")
    print(f"  {'Tier':<10} {'Before':>10} {'%':>7}   {'After':>10} {'%':>7}   {'Multiplier':>12}")
    print(f"  {'-'*62}")

    for tier, label, mult in [
        (1, "Tier 1 (3x)", "3x"),
        (2, "Tier 2 (1x)", "1x"),
        (3, "Tier 3 (.3x)", "~0.3x"),
    ]:
        b = before[tier]
        a = after[tier]
        pct_b = 100.0 * b / total_before if total_before else 0
        pct_a = 100.0 * a / total_after  if total_after  else 0
        print(f"  {label:<10} {b:>10,} {pct_b:>6.1f}%   {a:>10,} {pct_a:>6.1f}%   {mult:>12}")

    print(f"  {'-'*62}")
    print(f"  {'TOTAL':<10} {total_before:>10,} {'100.0%':>7}   {total_after:>10,} {'100.0%':>7}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply section-importance weighting to a training JSONL file."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input JSONL file (sft.jsonl or grpo.jsonl)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to write the weighted JSONL output",
    )
    parser.add_argument(
        "--tiers", "-t",
        default="data/reference/section_tiers.json",
        help="Path to section_tiers.json (default: data/reference/section_tiers.json)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for tier-3 downsampling (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading tiers from: {args.tiers}")
    tiers = load_tiers(args.tiers)
    tier1 = build_tier1_set(tiers)
    tier3_ranges = build_tier3_ranges(tiers)
    print(f"  Tier 1 sections: {len(tier1)}")
    print(f"  Tier 3 ranges  : {tier3_ranges}")

    print(f"\nReading input: {args.input}")
    records = read_jsonl(args.input)
    print(f"  Read {len(records):,} records")

    print("\nApplying weights...")
    weighted, stats = apply_weights(records, tier1, tier3_ranges, seed=args.seed)

    print_stats(stats, args.input, args.output)

    print(f"Writing output: {args.output}")
    write_jsonl(weighted, args.output)
    print(f"  Wrote {len(weighted):,} records")


if __name__ == "__main__":
    main()
