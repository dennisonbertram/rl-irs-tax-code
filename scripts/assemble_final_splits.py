#!/usr/bin/env python3
"""
assemble_final_splits.py

Assembles final training-ready splits from all improved data sources.
Combines weighted IRC SFT + inflation-adjusted data with 5x upsampling for
inflation records (which correct the biggest hallucination issues).

Outputs:
  data/final/sft_train.jsonl   - Combined weighted + upsampled inflation SFT
  data/final/sft_valid.jsonl   - Eval SFT split
  data/final/dpo_train.jsonl   - Combined weighted + upsampled inflation DPO
  data/final/dpo_valid.jsonl   - Eval DPO split
  data/final/grpo_train.jsonl  - Weighted GRPO (as-is)
  data/final/grpo_valid.jsonl  - Eval GRPO split
  data/final/train.jsonl       - Copy of sft_train.jsonl (train_sft.py compat)
  data/final/valid.jsonl       - Copy of sft_valid.jsonl (train_sft.py compat)
  data/final/MANIFEST.md       - Complete statistics
"""

import json
import random
import shutil
import os
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/dennisonbertram/Develop/rl-irs-tax-code")
DATA_ROOT    = PROJECT_ROOT / "data"

SOURCES = {
    "weighted_sft":  DATA_ROOT / "train_v2" / "sft_weighted.jsonl",
    "inflation_sft": DATA_ROOT / "processed" / "inflation_adjusted_sft.jsonl",
    "grounded_dpo":  DATA_ROOT / "train_v2" / "dpo.jsonl",
    "inflation_dpo": DATA_ROOT / "processed" / "inflation_adjusted_dpo.jsonl",
    "weighted_grpo": DATA_ROOT / "train_v2" / "grpo_weighted.jsonl",
    "eval_sft":      DATA_ROOT / "eval_v2" / "sft.jsonl",
    "eval_dpo":      DATA_ROOT / "eval_v2" / "dpo.jsonl",
    "eval_grpo":     DATA_ROOT / "eval_v2" / "grpo.jsonl",
}

OUTPUT_DIR = DATA_ROOT / "final"

INFLATION_UPSAMPLE = 5   # 5x upsampling for inflation-adjusted records
SEED = 42


# ── Utility helpers ────────────────────────────────────────────────────────────

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


def normalize_inflation_dpo(record: dict) -> dict:
    """
    Normalize inflation DPO records to match the grounded DPO string format.

    Inflation DPO uses:
      prompt:   list of {role, content} messages
      chosen:   list of {role, content} messages  (just assistant turn)
      rejected: list of {role, content} messages

    Target format (grounded DPO style):
      prompt:   str  (user question only)
      chosen:   str  (assistant answer)
      rejected: str  (assistant answer)
      metadata: dict (preserved)
    """
    prompt_raw = record["prompt"]
    chosen_raw = record["chosen"]
    rejected_raw = record["rejected"]

    # Extract the user question from the prompt message list
    if isinstance(prompt_raw, list):
        user_msgs = [m["content"] for m in prompt_raw if m["role"] == "user"]
        prompt_str = user_msgs[-1] if user_msgs else ""
    else:
        prompt_str = prompt_raw

    # Extract assistant content from chosen/rejected
    def extract_content(field):
        if isinstance(field, list):
            parts = [m["content"] for m in field if m["role"] == "assistant"]
            return " ".join(parts) if parts else str(field)
        return field

    return {
        "prompt":   prompt_str,
        "chosen":   extract_content(chosen_raw),
        "rejected": extract_content(rejected_raw),
        "metadata": record.get("metadata", {"source": "inflation_adjusted"}),
    }


def upsample(records: list[dict], factor: int) -> list[dict]:
    """Return records repeated `factor` times."""
    return records * factor


# ── Assembly logic ─────────────────────────────────────────────────────────────

def assemble_sft(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Build final SFT train and valid splits."""
    print("Loading SFT sources...")
    weighted = load_jsonl(SOURCES["weighted_sft"])
    inflation = load_jsonl(SOURCES["inflation_sft"])
    eval_sft  = load_jsonl(SOURCES["eval_sft"])

    print(f"  Weighted SFT:   {len(weighted):>6,} records")
    print(f"  Inflation SFT:  {len(inflation):>6,} records (before upsampling)")

    inflation_up = upsample(inflation, INFLATION_UPSAMPLE)
    print(f"  Inflation SFT:  {len(inflation_up):>6,} records (after {INFLATION_UPSAMPLE}x upsampling)")

    # Combine and shuffle
    combined = weighted + inflation_up
    rng.shuffle(combined)

    print(f"  Train total:    {len(combined):>6,} records")
    print(f"  Valid total:    {len(eval_sft):>6,} records")

    return combined, eval_sft


def assemble_dpo(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Build final DPO train and valid splits."""
    print("Loading DPO sources...")
    grounded  = load_jsonl(SOURCES["grounded_dpo"])
    inflation  = load_jsonl(SOURCES["inflation_dpo"])
    eval_dpo   = load_jsonl(SOURCES["eval_dpo"])

    print(f"  Grounded DPO:   {len(grounded):>6,} records")
    print(f"  Inflation DPO:  {len(inflation):>6,} records (before upsampling)")

    # Normalize inflation DPO to match grounded format
    inflation_norm = [normalize_inflation_dpo(r) for r in inflation]
    inflation_up   = upsample(inflation_norm, INFLATION_UPSAMPLE)
    print(f"  Inflation DPO:  {len(inflation_up):>6,} records (after {INFLATION_UPSAMPLE}x upsampling)")

    combined = grounded + inflation_up
    rng.shuffle(combined)

    print(f"  Train total:    {len(combined):>6,} records")
    print(f"  Valid total:    {len(eval_dpo):>6,} records")

    return combined, eval_dpo


def assemble_grpo() -> tuple[list[dict], list[dict]]:
    """Build final GRPO train and valid splits."""
    print("Loading GRPO sources...")
    weighted  = load_jsonl(SOURCES["weighted_grpo"])
    eval_grpo = load_jsonl(SOURCES["eval_grpo"])

    print(f"  Weighted GRPO:  {len(weighted):>6,} records")
    print(f"  Valid total:    {len(eval_grpo):>6,} records")

    return weighted, eval_grpo


# ── Manifest writer ────────────────────────────────────────────────────────────

def write_manifest(stats: dict) -> None:
    """Write MANIFEST.md into the output directory."""
    path = OUTPUT_DIR / "MANIFEST.md"
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Final Training Splits — Manifest",
        "",
        f"Generated: {now}",
        f"Shuffle seed: {SEED}",
        f"Inflation upsample factor: {INFLATION_UPSAMPLE}x",
        "",
        "## File Summary",
        "",
        "| File | Records | Description |",
        "| ---- | ------: | ----------- |",
    ]

    for fname, info in stats.items():
        lines.append(f"| `{fname}` | {info['count']:,} | {info['desc']} |")

    lines += [
        "",
        "## Source Breakdown",
        "",
        "### SFT Train",
        f"- Weighted IRC SFT (`train_v2/sft_weighted.jsonl`): {stats['sft_train.jsonl']['sources']['weighted']:,} records (Tier 1 sections 3x upsampled at source)",
        f"- Inflation-adjusted SFT (`processed/inflation_adjusted_sft.jsonl`): {stats['sft_train.jsonl']['sources']['inflation_raw']:,} records × {INFLATION_UPSAMPLE}x = {stats['sft_train.jsonl']['sources']['inflation_up']:,} records",
        "",
        "### DPO Train",
        f"- Grounded IRC DPO (`train_v2/dpo.jsonl`): {stats['dpo_train.jsonl']['sources']['grounded']:,} records",
        f"- Inflation-adjusted DPO (`processed/inflation_adjusted_dpo.jsonl`): {stats['dpo_train.jsonl']['sources']['inflation_raw']:,} records × {INFLATION_UPSAMPLE}x = {stats['dpo_train.jsonl']['sources']['inflation_up']:,} records",
        "  - Format normalized: prompt/chosen/rejected extracted from message lists",
        "",
        "### GRPO Train",
        f"- Weighted IRC GRPO (`train_v2/grpo_weighted.jsonl`): {stats['grpo_train.jsonl']['sources']['weighted']:,} records (used as-is)",
        "",
        "### Validation Splits",
        "- SFT valid: `eval_v2/sft.jsonl` (no data leakage from train set)",
        "- DPO valid: `eval_v2/dpo.jsonl` (no data leakage from train set)",
        "- GRPO valid: `eval_v2/grpo.jsonl` (no data leakage from train set)",
        "",
        "## Format Specifications",
        "",
        "| Split | Required Keys | Notes |",
        "| ----- | ------------- | ----- |",
        "| SFT   | `messages` (system/user/assistant) | Also exported as train.jsonl / valid.jsonl |",
        "| DPO   | `prompt`, `chosen`, `rejected` | prompt is plain string |",
        "| GRPO  | `prompt`, `expected_section` | prompt-only; expected_section for reward shaping |",
        "",
        "## Compatibility",
        "",
        "- `train.jsonl` / `valid.jsonl` are copies of SFT files for `train_sft.py` compatibility",
        "- CFR data can be merged later via `scripts/merge_cfr_data.py`",
        "",
        "## Notes",
        "",
        "- Inflation-adjusted records receive 5x upsampling to correct high-frequency hallucinations",
        "  on standard deduction amounts, contribution limits, and other indexed figures.",
        "- All splits were shuffled with seed=42 for reproducibility.",
        "- Eval splits are held out and NOT included in training data.",
    ]

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  Wrote {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Assembling final training splits")
    print("=" * 60)

    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {}

    # ── SFT ─────────────────────────────────────────────────────────────────
    print()
    sft_train, sft_valid = assemble_sft(rng)
    n = write_jsonl(OUTPUT_DIR / "sft_train.jsonl", sft_train)
    weighted_count = len(load_jsonl(SOURCES["weighted_sft"]))
    inflation_count = len(load_jsonl(SOURCES["inflation_sft"]))
    stats["sft_train.jsonl"] = {
        "count": n,
        "desc": "Weighted IRC SFT + 5x inflation SFT, shuffled",
        "sources": {
            "weighted": weighted_count,
            "inflation_raw": inflation_count,
            "inflation_up": inflation_count * INFLATION_UPSAMPLE,
        }
    }
    print(f"  -> Wrote sft_train.jsonl: {n:,} records")

    n = write_jsonl(OUTPUT_DIR / "sft_valid.jsonl", sft_valid)
    stats["sft_valid.jsonl"] = {
        "count": n,
        "desc": "Eval SFT split (eval_v2/sft.jsonl)",
        "sources": {}
    }
    print(f"  -> Wrote sft_valid.jsonl: {n:,} records")

    # ── DPO ─────────────────────────────────────────────────────────────────
    print()
    dpo_train, dpo_valid = assemble_dpo(rng)
    n = write_jsonl(OUTPUT_DIR / "dpo_train.jsonl", dpo_train)
    grounded_count = len(load_jsonl(SOURCES["grounded_dpo"]))
    inflation_dpo_count = len(load_jsonl(SOURCES["inflation_dpo"]))
    stats["dpo_train.jsonl"] = {
        "count": n,
        "desc": "Grounded IRC DPO + 5x inflation DPO (normalized), shuffled",
        "sources": {
            "grounded": grounded_count,
            "inflation_raw": inflation_dpo_count,
            "inflation_up": inflation_dpo_count * INFLATION_UPSAMPLE,
        }
    }
    print(f"  -> Wrote dpo_train.jsonl: {n:,} records")

    n = write_jsonl(OUTPUT_DIR / "dpo_valid.jsonl", dpo_valid)
    stats["dpo_valid.jsonl"] = {
        "count": n,
        "desc": "Eval DPO split (eval_v2/dpo.jsonl)",
        "sources": {}
    }
    print(f"  -> Wrote dpo_valid.jsonl: {n:,} records")

    # ── GRPO ────────────────────────────────────────────────────────────────
    print()
    grpo_train, grpo_valid = assemble_grpo()
    n = write_jsonl(OUTPUT_DIR / "grpo_train.jsonl", grpo_train)
    stats["grpo_train.jsonl"] = {
        "count": n,
        "desc": "Weighted IRC GRPO (as-is, train_v2/grpo_weighted.jsonl)",
        "sources": {"weighted": n}
    }
    print(f"  -> Wrote grpo_train.jsonl: {n:,} records")

    n = write_jsonl(OUTPUT_DIR / "grpo_valid.jsonl", grpo_valid)
    stats["grpo_valid.jsonl"] = {
        "count": n,
        "desc": "Eval GRPO split (eval_v2/grpo.jsonl)",
        "sources": {}
    }
    print(f"  -> Wrote grpo_valid.jsonl: {n:,} records")

    # ── Compatibility copies for train_sft.py ────────────────────────────────
    print()
    print("Writing train_sft.py compatibility files...")
    shutil.copy(OUTPUT_DIR / "sft_train.jsonl", OUTPUT_DIR / "train.jsonl")
    shutil.copy(OUTPUT_DIR / "sft_valid.jsonl", OUTPUT_DIR / "valid.jsonl")
    train_count = len(load_jsonl(OUTPUT_DIR / "train.jsonl"))
    valid_count  = len(load_jsonl(OUTPUT_DIR / "valid.jsonl"))
    stats["train.jsonl"] = {
        "count": train_count,
        "desc": "Copy of sft_train.jsonl (train_sft.py compatibility)",
        "sources": {}
    }
    stats["valid.jsonl"] = {
        "count": valid_count,
        "desc": "Copy of sft_valid.jsonl (train_sft.py compatibility)",
        "sources": {}
    }
    print(f"  -> Wrote train.jsonl: {train_count:,} records")
    print(f"  -> Wrote valid.jsonl: {valid_count:,} records")

    # ── Manifest ─────────────────────────────────────────────────────────────
    print()
    print("Writing MANIFEST.md...")
    write_manifest(stats)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Assembly complete. Final split sizes:")
    for fname, info in stats.items():
        print(f"  {fname:<22} {info['count']:>7,} records")
    print("=" * 60)


if __name__ == "__main__":
    main()
