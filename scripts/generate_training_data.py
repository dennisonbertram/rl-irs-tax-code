#!/usr/bin/env python3
"""
Generate SFT, DPO, and GRPO training datasets from parsed IRC and CFR sections.

Sources:
  - data/processed/irc_sections.jsonl
  - data/processed/cfr_sections.jsonl

Outputs:
  - data/processed/sft_train.jsonl   (chat format for mlx-lm)
  - data/processed/dpo_train.jsonl   (preference pairs)
  - data/processed/grpo_train.jsonl  (prompts only)
"""
import json
import random
import textwrap
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data/processed"
IRC_JSONL = PROCESSED_DIR / "irc_sections.jsonl"
CFR_JSONL = PROCESSED_DIR / "cfr_sections.jsonl"
SFT_OUT = PROCESSED_DIR / "sft_train.jsonl"
DPO_OUT = PROCESSED_DIR / "dpo_train.jsonl"
GRPO_OUT = PROCESSED_DIR / "grpo_train.jsonl"

SYSTEM_PROMPT = (
    "You are a tax law expert specializing in the Internal Revenue Code (IRC) "
    "and Code of Federal Regulations (CFR) Title 26. Always cite specific IRC sections "
    "or CFR sections when answering questions. Provide accurate, detailed explanations."
)

random.seed(42)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate text to max_chars, ending at a sentence boundary if possible."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to end at last period
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return truncated[: last_period + 1]
    return truncated.rstrip() + "..."


def generate_irc_questions(section: dict) -> list[tuple[str, str]]:
    """Generate question-answer pairs for an IRC section."""
    sec_num = section["section"]
    heading = section["heading"]
    text = section["text"]

    # Skip sections with very short text
    if len(text) < 100:
        return []

    truncated_text = truncate(text, 1800)
    pairs = []

    # Q1: Definition/general question
    if heading:
        q1 = f"What does IRC Section {sec_num} say about {heading.lower()}?"
        a1 = (
            f"Under IRC Section {sec_num} ({heading}), {truncated_text}"
        )
        pairs.append((q1, a1))

    # Q2: What is defined / what does it cover
    q2 = f"What is covered under Internal Revenue Code Section {sec_num}?"
    a2 = (
        f"IRC Section {sec_num}"
        + (f", titled '{heading}'," if heading else "")
        + f" covers the following: {truncated_text}"
    )
    pairs.append((q2, a2))

    # Q3: Tax treatment question (for substantive sections)
    if len(text) > 300:
        q3 = f"How does IRC Section {sec_num} apply to tax calculations or treatment?"
        a3 = (
            f"IRC Section {sec_num} provides the following rules and requirements: "
            f"{truncated_text}"
        )
        pairs.append((q3, a3))

    # Q4: Cite and explain
    if heading and len(text) > 500:
        q4 = f"Please cite and explain the provisions of IRC § {sec_num}."
        a4 = (
            f"IRC § {sec_num} — {heading}\n\n"
            f"{truncated_text}"
        )
        pairs.append((q4, a4))

    # Q5: Practical application
    if len(text) > 400:
        q5 = f"What are the key provisions of IRC Section {sec_num} that a taxpayer should know?"
        a5 = (
            f"The key provisions of IRC Section {sec_num}"
            + (f" ({heading})" if heading else "")
            + f" that taxpayers should understand include: {truncated_text}"
        )
        pairs.append((q5, a5))

    # Shuffle and return up to 5
    random.shuffle(pairs)
    return pairs[:5]


def generate_cfr_questions(section: dict) -> list[tuple[str, str]]:
    """Generate question-answer pairs for a CFR section."""
    sec_num = section["section"]
    heading = section["heading"]
    text = section["text"]

    if len(text) < 100:
        return []

    truncated_text = truncate(text, 1800)
    pairs = []

    # Q1: What does the regulation say
    q1 = f"What does CFR Section {sec_num} of Title 26 provide regarding {heading.lower() if heading else 'this topic'}?"
    a1 = (
        f"26 CFR § {sec_num}"
        + (f" ({heading})" if heading else "")
        + f" provides: {truncated_text}"
    )
    pairs.append((q1, a1))

    # Q2: How does regulation apply
    if len(text) > 300:
        q2 = f"How does Treasury Regulation § {sec_num} apply under the Internal Revenue Code?"
        a2 = (
            f"Treasury Regulation § {sec_num}"
            + (f", titled '{heading}'," if heading else "")
            + f" applies as follows: {truncated_text}"
        )
        pairs.append((q2, a2))

    # Q3: Key rules
    if len(text) > 400:
        q3 = f"What are the key rules established by 26 CFR § {sec_num}?"
        a3 = (
            f"26 CFR § {sec_num}"
            + (f" ({heading})" if heading else "")
            + f" establishes the following rules: {truncated_text}"
        )
        pairs.append((q3, a3))

    random.shuffle(pairs)
    return pairs[:3]


def make_sft_record(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def make_dpo_record(question: str, chosen_answer: str, rejected_answer: str) -> dict:
    return {
        "prompt": question,
        "chosen": chosen_answer,
        "rejected": rejected_answer,
    }


def make_rejected_answer(question: str, sec_ref: str) -> str:
    """Create a vague, citation-free rejected answer."""
    vague_templates = [
        "This depends on various factors and you should consult a tax professional.",
        "The tax rules in this area are complex and vary by situation.",
        "There are several considerations that may apply depending on your specific circumstances.",
        "Tax law in this area requires careful analysis of the facts and circumstances.",
        "The answer to this question depends on many factors including your income, filing status, and specific transactions.",
    ]
    return random.choice(vague_templates)


def generate_grpo_prompts(section: dict) -> list[str]:
    """Generate GRPO-style prompts from a section."""
    sec_num = section["section"]
    heading = section["heading"]
    source = section["source"]
    prefix = "IRC" if source == "IRC" else "CFR"
    sec_label = f"IRC § {sec_num}" if source == "IRC" else f"26 CFR § {sec_num}"

    prompts = []

    if heading:
        prompts.append(f"Explain the tax treatment of {heading.lower()} under {sec_label}.")
        prompts.append(
            f"What are the requirements and limitations established by {sec_label} "
            f"regarding {heading.lower()}?"
        )
    prompts.append(
        f"Provide a detailed explanation of the rules under {sec_label} "
        f"and how they apply to taxpayers."
    )
    if source == "IRC":
        prompts.append(
            f"A taxpayer is asking about {sec_label}. "
            f"What should they know about this provision?"
        )

    return prompts


def main():
    print("Loading parsed sections...")

    irc_sections = load_jsonl(IRC_JSONL)
    cfr_sections = load_jsonl(CFR_JSONL)
    print(f"  IRC sections: {len(irc_sections)}")
    print(f"  CFR sections: {len(cfr_sections)}")

    all_sections = irc_sections + cfr_sections

    # ── SFT ──────────────────────────────────────────────────────────────────
    print("\nGenerating SFT data...")
    sft_records = []

    for sec in irc_sections:
        pairs = generate_irc_questions(sec)
        for q, a in pairs:
            sft_records.append(make_sft_record(q, a))

    for sec in cfr_sections:
        pairs = generate_cfr_questions(sec)
        for q, a in pairs:
            sft_records.append(make_sft_record(q, a))

    random.shuffle(sft_records)
    print(f"  Generated {len(sft_records)} SFT examples")

    with open(SFT_OUT, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written to {SFT_OUT}")

    # ── DPO ──────────────────────────────────────────────────────────────────
    print("\nGenerating DPO data...")
    dpo_records = []

    for sec in irc_sections:
        sec_ref = f"IRC Section {sec['section']}"
        pairs = generate_irc_questions(sec)
        # Use first pair as chosen, generate rejected for same question
        for q, chosen in pairs[:2]:
            rejected = make_rejected_answer(q, sec_ref)
            dpo_records.append(make_dpo_record(q, chosen, rejected))

    for sec in cfr_sections:
        sec_ref = f"26 CFR § {sec['section']}"
        pairs = generate_cfr_questions(sec)
        for q, chosen in pairs[:1]:
            rejected = make_rejected_answer(q, sec_ref)
            dpo_records.append(make_dpo_record(q, chosen, rejected))

    random.shuffle(dpo_records)
    print(f"  Generated {len(dpo_records)} DPO pairs")

    with open(DPO_OUT, "w", encoding="utf-8") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written to {DPO_OUT}")

    # ── GRPO ─────────────────────────────────────────────────────────────────
    print("\nGenerating GRPO prompts...")
    grpo_records = []

    for sec in all_sections:
        prompts = generate_grpo_prompts(sec)
        for p in prompts:
            grpo_records.append({"prompt": p})

    random.shuffle(grpo_records)
    print(f"  Generated {len(grpo_records)} GRPO prompts")

    with open(GRPO_OUT, "w", encoding="utf-8") as f:
        for rec in grpo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written to {GRPO_OUT}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  SFT examples:   {len(sft_records):,}")
    print(f"  DPO pairs:      {len(dpo_records):,}")
    print(f"  GRPO prompts:   {len(grpo_records):,}")

    # Print samples
    if sft_records:
        sample = sft_records[0]
        print(f"\nSFT sample:")
        print(f"  user: {sample['messages'][1]['content'][:100]}...")
        print(f"  assistant: {sample['messages'][2]['content'][:100]}...")

    if dpo_records:
        sample = dpo_records[0]
        print(f"\nDPO sample:")
        print(f"  prompt: {sample['prompt'][:100]}...")
        print(f"  chosen: {sample['chosen'][:80]}...")
        print(f"  rejected: {sample['rejected'][:80]}...")

    if grpo_records:
        sample = grpo_records[0]
        print(f"\nGRPO sample:")
        print(f"  prompt: {sample['prompt'][:100]}...")


if __name__ == "__main__":
    main()
