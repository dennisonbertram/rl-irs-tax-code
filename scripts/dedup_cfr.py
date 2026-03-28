#!/usr/bin/env python3
"""
dedup_cfr.py

Cleans CFR SFT data by:
  1. Removing records with exact duplicate assistant responses (keep first)
  2. Removing records with exact duplicate user questions (keep first)
  3. Appending a professional disclaimer to assistant responses that are missing it
  4. Writing cleaned output to data/processed/grounded_cfr_sft_deduped.jsonl

Usage:
  python scripts/dedup_cfr.py [--input PATH] [--output PATH]
"""

import argparse
import json
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/Users/dennisonbertram/Develop/rl-irs-tax-code")
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "grounded_cfr_sft_full.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "grounded_cfr_sft_deduped.jsonl"

DISCLAIMER = (
    "This information is for educational purposes. "
    "Consult a qualified tax professional for advice specific to your situation."
)

# Phrases that indicate a disclaimer is already present (case-insensitive substrings)
DISCLAIMER_MARKERS = [
    "educational purposes",
    "consult a qualified tax professional",
    "qualified tax professional",
    "for personalized advice",
    "not constitute legal",
    "not constitute tax",
    "consult a tax professional",
    "consult with a tax",
    "professional advice",
]


# ── Utilities ──────────────────────────────────────────────────────────────────

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


def get_message(messages: list[dict], role: str) -> str | None:
    """Return the content of the first message with the given role."""
    for msg in messages:
        if msg.get("role") == role:
            return msg.get("content", "")
    return None


def has_disclaimer(text: str) -> bool:
    """Return True if the text already contains a disclaimer-like phrase."""
    lower = text.lower()
    return any(marker in lower for marker in DISCLAIMER_MARKERS)


def append_disclaimer(messages: list[dict]) -> tuple[list[dict], bool]:
    """
    Append disclaimer to assistant message if missing.
    Returns (modified messages list, was_modified flag).
    """
    modified = False
    result = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if not has_disclaimer(content):
                msg = dict(msg)  # shallow copy
                msg["content"] = content.rstrip() + " " + DISCLAIMER
                modified = True
        result.append(msg)
    return result, modified


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deduplicate CFR SFT data")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        raise SystemExit(1)

    print("=" * 60)
    print("CFR SFT Deduplication")
    print("=" * 60)

    # Load
    print(f"\nLoading {args.input}...")
    records = load_jsonl(args.input)
    records_in = len(records)
    print(f"  Records in: {records_in:,}")

    # Deduplicate on exact duplicate assistant answers (keep first)
    seen_answers: set[str] = set()
    deduped_answers: list[dict] = []
    dup_answer_count = 0
    for rec in records:
        answer = get_message(rec.get("messages", []), "assistant") or ""
        if answer in seen_answers:
            dup_answer_count += 1
        else:
            seen_answers.add(answer)
            deduped_answers.append(rec)

    print(f"  Duplicate answers removed: {dup_answer_count:,}")
    print(f"  After answer dedup: {len(deduped_answers):,}")

    # Deduplicate on exact duplicate user questions (keep first)
    seen_questions: set[str] = set()
    deduped_questions: list[dict] = []
    dup_question_count = 0
    for rec in deduped_answers:
        question = get_message(rec.get("messages", []), "user") or ""
        if question in seen_questions:
            dup_question_count += 1
        else:
            seen_questions.add(question)
            deduped_questions.append(rec)

    print(f"  Duplicate questions removed: {dup_question_count:,}")
    print(f"  After question dedup: {len(deduped_questions):,}")

    total_dupes = dup_answer_count + dup_question_count

    # Add disclaimers where missing
    disclaimer_added = 0
    final_records: list[dict] = []
    for rec in deduped_questions:
        messages = rec.get("messages", [])
        new_messages, was_modified = append_disclaimer(messages)
        if was_modified:
            rec = dict(rec)  # shallow copy of record
            rec["messages"] = new_messages
            disclaimer_added += 1
        final_records.append(rec)

    print(f"  Disclaimers added: {disclaimer_added:,}")

    # Write output
    print(f"\nWriting {args.output}...")
    records_out = write_jsonl(args.output, final_records)
    print(f"  Records out: {records_out:,}")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Records in:              {records_in:,}")
    print(f"  Duplicate answers removed: {dup_answer_count:,}")
    print(f"  Duplicate questions removed: {dup_question_count:,}")
    print(f"  Total duplicates removed: {total_dupes:,}")
    print(f"  Disclaimers added:        {disclaimer_added:,}")
    print(f"  Records out:             {records_out:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
