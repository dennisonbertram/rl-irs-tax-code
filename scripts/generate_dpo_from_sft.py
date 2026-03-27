#!/usr/bin/env python3
"""
Generate DPO pairs from existing grounded_sft_full.jsonl.

Reads all 16,909 SFT pairs, samples ~1,700 (10%), and calls GPT-4o-mini
to produce hard-negative (subtly corrupted) rejected answers.

Output: data/processed/grounded_dpo_full.jsonl (overwrites existing 152-pair version)

Usage:
    python3.14 scripts/generate_dpo_from_sft.py

Resume (skip already-generated pairs by appending):
    python3.14 scripts/generate_dpo_from_sft.py --resume
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SFT_IN       = PROJECT_ROOT / "data" / "processed" / "grounded_sft_full.jsonl"
DPO_OUT      = PROJECT_ROOT / "data" / "processed" / "grounded_dpo_full.jsonl"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL            = "gpt-4o-mini"
DPO_FRACTION     = 0.10          # 10% of SFT pairs
RATE_LIMIT_SLEEP = 0.5           # seconds between API calls
RANDOM_SEED      = 42

# GPT-4o-mini pricing (per 1M tokens, 2025)
PRICE_PER_1M_INPUT  = 0.15
PRICE_PER_1M_OUTPUT = 0.60

DPO_CORRUPTION_PROMPT = """\
You are generating hard-negative training examples for a DPO dataset.

Below is a correct answer to a tax law question. Your task is to introduce EXACTLY ONE subtle error to create a rejected version. The error should be realistic — the kind a non-expert might make — but clearly wrong to someone who reads the statute carefully.

Acceptable error types (pick one):
- Change a specific subsection letter/number (e.g., §401(a) → §401(b))
- Change a dollar threshold by a plausible amount (e.g., $5,000 → $10,000, or remove it entirely)
- Remove or invert an important exception or limitation
- Change a percentage or time limit
- Omit a required condition that changes the meaning
- Attribute the rule to the wrong IRC section number

Do NOT make the error obvious. The rejected answer should read fluently and sound authoritative.

Question: {question}

Correct answer:
{correct_answer}

Return ONLY a JSON object with:
- "rejected_answer": the subtly corrupted version
- "error_description": a brief note describing what you changed (for our records)
"""


def load_sft_pairs(path: Path) -> list[tuple[str, str, str]]:
    """Load SFT records and return list of (question, answer, section_num) tuples."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            if len(msgs) < 3:
                continue
            question = msgs[1]["content"]
            answer   = msgs[2]["content"]
            sec_meta = rec.get("metadata", {}).get("source_section", "")
            sec_num  = sec_meta.replace("IRC §", "").strip()
            pairs.append((question, answer, sec_num))
    return pairs


def select_dpo_candidates(
    all_pairs: list[tuple[str, str, str]],
    n: int,
) -> list[tuple[str, str, str]]:
    """Stratified sample of n pairs spread across all sections."""
    if len(all_pairs) <= n:
        return all_pairs[:]

    shuffled = list(all_pairs)
    random.shuffle(shuffled)

    by_section: dict[str, list] = {}
    for q, a, sec in shuffled:
        by_section.setdefault(sec, []).append((q, a, sec))

    candidates = []
    sec_list = list(by_section.values())
    idx = 0
    while len(candidates) < n:
        bucket = sec_list[idx % len(sec_list)]
        if bucket:
            candidates.append(bucket.pop(0))
        idx += 1
        if all(len(b) == 0 for b in sec_list):
            break

    return candidates[:n]


def call_openai_for_dpo(
    client: OpenAI,
    question: str,
    correct_answer: str,
) -> tuple[str, str, dict]:
    """
    Generate a hard-negative rejected answer via GPT-4o-mini.
    Returns (rejected_answer, error_description, usage_dict).
    """
    prompt = DPO_CORRUPTION_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a data generation assistant. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
        max_tokens=1024,
    )

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        rejected   = parsed.get("rejected_answer", "")
        error_desc = parsed.get("error_description", "")
    except json.JSONDecodeError:
        rejected   = ""
        error_desc = "parse error"

    return rejected, error_desc, usage


def make_dpo_record(
    question: str,
    chosen_answer: str,
    rejected_answer: str,
    section_num: str,
    error_description: str,
) -> dict:
    return {
        "prompt":   question,
        "chosen":   chosen_answer,
        "rejected": rejected_answer,
        "metadata": {
            "source_section":  f"IRC §{section_num}",
            "error_introduced": error_description,
        },
    }


def compute_cost(total_input: int, total_output: int) -> float:
    return (
        total_input  / 1_000_000 * PRICE_PER_1M_INPUT
        + total_output / 1_000_000 * PRICE_PER_1M_OUTPUT
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO pairs from existing grounded_sft_full.jsonl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing DPO file instead of overwriting (skip already-done count)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=DPO_FRACTION,
        help=f"Fraction of SFT pairs to use (default: {DPO_FRACTION})",
    )
    parser.add_argument(
        "--sft-input",
        default=str(SFT_IN),
        help=f"SFT input file (default: {SFT_IN})",
    )
    parser.add_argument(
        "--dpo-output",
        default=str(DPO_OUT),
        help=f"DPO output file (default: {DPO_OUT})",
    )
    args = parser.parse_args()

    sft_path = Path(args.sft_input)
    dpo_path = Path(args.dpo_output)

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Make sure it is exported in ~/.zshrc."
        )
    client = OpenAI(api_key=api_key)

    # ── Load SFT data ─────────────────────────────────────────────────────────
    print(f"Loading SFT pairs from {sft_path}...")
    all_pairs = load_sft_pairs(sft_path)
    print(f"  Loaded {len(all_pairs):,} SFT pairs")

    random.seed(RANDOM_SEED)
    target_n = max(1, int(len(all_pairs) * args.fraction))
    print(f"  Target DPO pairs: {target_n:,} ({args.fraction:.0%} of {len(all_pairs):,})")

    # ── Resume: skip already-done ──────────────────────────────────────────────
    already_done = 0
    if args.resume and dpo_path.exists():
        with open(dpo_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done += 1
        print(f"  Resume mode: {already_done} pairs already generated, skipping those candidates")

    candidates = select_dpo_candidates(all_pairs, target_n)
    # Skip already-done candidates when resuming
    candidates = candidates[already_done:]
    remaining = len(candidates)
    print(f"  Candidates to process: {remaining:,}")

    # Cost estimate
    est_input  = remaining * 400    # ~400 tokens per prompt
    est_output = remaining * 200    # ~200 tokens per response
    est_cost   = compute_cost(est_input, est_output)
    est_min    = remaining * (RATE_LIMIT_SLEEP + 0.5) / 60
    print(f"\n  Estimated cost: ${est_cost:.2f}")
    print(f"  Estimated time: ~{est_min:.0f} minutes")
    print()

    # ── Generate DPO pairs ─────────────────────────────────────────────────────
    write_mode = "a" if args.resume else "w"
    dpo_path.parent.mkdir(parents=True, exist_ok=True)

    total_generated = already_done
    total_failed    = 0
    total_input_tok = 0
    total_output_tok = 0

    with open(dpo_path, write_mode, encoding="utf-8") as dpo_file:
        for j, (question, chosen_answer, sec_num) in enumerate(candidates, 1):
            global_j = already_done + j
            try:
                rejected, error_desc, usage = call_openai_for_dpo(client, question, chosen_answer)
                total_input_tok  += usage["prompt_tokens"]
                total_output_tok += usage["completion_tokens"]
            except Exception as e:
                print(f"  [ERROR] pair {global_j}: {e}")
                total_failed += 1
                time.sleep(RATE_LIMIT_SLEEP)
                continue

            if rejected:
                rec = make_dpo_record(question, chosen_answer, rejected, sec_num, error_desc)
                dpo_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_generated += 1
            else:
                total_failed += 1

            # Progress report every 100 pairs
            if j % 100 == 0 or j == remaining:
                cost = compute_cost(total_input_tok, total_output_tok)
                print(
                    f"  DPO {global_j}/{target_n} | "
                    f"generated: {total_generated} | "
                    f"failed: {total_failed} | "
                    f"cost: ${cost:.4f}"
                )
                dpo_file.flush()

            if j < remaining:
                time.sleep(RATE_LIMIT_SLEEP)

    final_cost = compute_cost(total_input_tok, total_output_tok)
    print(f"\nDone.")
    print(f"  Total generated: {total_generated}")
    print(f"  Total failed:    {total_failed}")
    print(f"  Total API cost:  ${final_cost:.4f}")
    print(f"  Written to:      {dpo_path}")


if __name__ == "__main__":
    main()
