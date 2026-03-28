#!/usr/bin/env python3
"""
Generate RAG-grounded SFT training data from CFR (Treasury Regulation) sections
using GPT-4o-mini.

Reads:
  - data/processed/cfr_sections.jsonl  (6,149 Treasury Regulation sections)

Outputs:
  - data/processed/grounded_cfr_sft_full.jsonl   (full run, ~55K pairs)
  - data/processed/grounded_cfr_sft_test.jsonl   (test output via --limit)

Usage:
  # Test run with 10 sections
  python3 scripts/generate_cfr_grounded_data.py --limit 10

  # Full run
  python3 scripts/generate_cfr_grounded_data.py

  # Resume an interrupted run
  python3 scripts/generate_cfr_grounded_data.py --resume

  # Full run with custom output path
  python3 scripts/generate_cfr_grounded_data.py --output data/processed/grounded_cfr_sft_full.jsonl

Cost estimate (full 6,149 sections, 9 pairs each, GPT-4o-mini):
  - Direct API:  ~$18.21
  - Batch API:   ~$9.10 (50% off, async, 24h window)

Key adaptations from the IRC pipeline:
  1. Prompts reference "Treasury Regulation" / "Treas. Reg." instead of "IRC Section"
  2. Citation validation adapted for CFR section numbering (e.g., 1.162-5)
  3. Cross-reference extraction adapted for CFR-style refs (§1.162-5, Reg. §1.162-5)
  4. Source section metadata set to "CFR §X.X" format
"""

import argparse
import json
import os
import re
import random
import time
from pathlib import Path

from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CFR_JSONL = PROCESSED_DIR / "cfr_sections.jsonl"
SFT_OUT_TEST = PROCESSED_DIR / "grounded_cfr_sft_test.jsonl"
SFT_OUT_FULL = PROCESSED_DIR / "grounded_cfr_sft_full.jsonl"
PROGRESS_FILE = PROCESSED_DIR / ".cfr_generation_progress.json"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
PAIRS_PER_SECTION = 9
MAX_SECTION_CHARS = 6000       # Cap section text sent to API
RANDOM_SEED = 42
PROGRESS_SAVE_INTERVAL = 50    # Save progress every N sections
RATE_LIMIT_SLEEP = 0.5         # Seconds between API calls

# GPT-4o-mini pricing (per 1M tokens) as of 2025
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60
BATCH_PRICE_PER_1M_INPUT = 0.075
BATCH_PRICE_PER_1M_OUTPUT = 0.30

random.seed(RANDOM_SEED)

SYSTEM_PROMPT = (
    "You are a tax law expert specializing in Treasury Regulations (CFR Title 26). "
    "Always cite specific Treasury Regulation sections when answering questions. "
    "Provide accurate, detailed explanations grounded strictly in the regulatory text."
)

GENERATION_PROMPT = """\
You are generating training data for a tax law AI assistant.

Below is the EXACT text of Treasury Regulation (Treas. Reg.) §{section_number}: {heading}

---
{full_section_text}
---

Related Treasury Regulation sections referenced in the text above (for context only — \
do NOT generate questions about them):
{related_sections_text}

Generate {n} diverse question-answer pairs about this Treasury Regulation section.

RULES:
1. Every answer MUST be grounded in the text above. Do not add information not present \
in the source text.
2. Directly quote or closely paraphrase the regulatory language.
3. Always cite the specific subsection (e.g., "Under Treas. Reg. §1.162-5(a)..." not \
just "Under Treas. Reg. §1.162-5...").
4. Include important exceptions, limitations, and cross-references mentioned in the text.
5. Vary question types: definitional ("What is..."), procedural ("How does..."), \
conditional ("When can..."), comparative ("What is the difference between..."), \
edge case ("Does X apply if...").
6. If dollar amounts, dates, percentages, or thresholds appear in the text, include \
them accurately.
7. End each answer with: "For personalized advice, consult a qualified tax professional."
8. IMPORTANT: All questions and answers MUST be about Treasury Regulation §{section_number} \
specifically. Do not generate questions about the related sections provided as context — \
they are only for cross-reference understanding.
9. When citing dollar amounts from the regulation, note that these are the regulatory \
base amounts and may be subject to inflation adjustments. Include a note like \
"subject to annual inflation adjustment" when relevant.
10. If the regulatory text references effective dates or amendments, note them in the answer.

Return ONLY a JSON array of objects with "question" and "answer" fields — no other text.\
"""


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cfr_sections(path: Path) -> dict[str, dict]:
    """Load all CFR sections indexed by section number."""
    sections: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("source") == "CFR":
                    sections[d["section"]] = d
    return sections


# ── Cross-reference extraction ────────────────────────────────────────────────

def extract_cfr_cross_refs(text: str) -> list[str]:
    """
    Find CFR section references like 'section 1.162-5', '§1.162-5',
    'Reg. §1.162-5', 'Treas. Reg. §1.162-5'.
    Returns a list of CFR section number strings.
    """
    patterns = [
        # §1.162-5, § 1.162-5, §1.1(h)-1
        r'§\s*(\d+\.\d+[\w()\-]*)',
        # Reg. §1.162-5, Treas. Reg. §1.162-5
        r'Reg\.\s*§\s*(\d+\.\d+[\w()\-]*)',
        # section 1.162-5
        r'\bsections?\s+(\d+\.\d+[\w()\-]*)',
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw = m.group(1).rstrip(".,;)")
            found.add(raw)
    return list(found)


def build_cfr_related_context(
    section: dict,
    all_sections: dict[str, dict],
    max_related: int = 3,
    max_chars_each: int = 800,
) -> str:
    """Build a related-sections context block by following CFR cross-references."""
    text = section.get("text", "")
    refs = extract_cfr_cross_refs(text)

    # Remove self-reference
    refs = [r for r in refs if r != section["section"]]

    # Only keep refs we actually have data for
    available = [r for r in refs if r in all_sections]

    seen = set()
    related_texts = []
    for ref in available:
        if ref in seen or len(related_texts) >= max_related:
            break
        seen.add(ref)
        sec = all_sections[ref]
        snippet = sec["text"][:max_chars_each]
        related_texts.append(
            f"Treas. Reg. §{ref} — {sec['heading']}:\n{snippet}"
            + ("..." if len(sec["text"]) > max_chars_each else "")
        )

    if not related_texts:
        return "(No cross-referenced Treasury Regulation sections found in the dataset.)"
    return "\n\n".join(related_texts)


# ── Citation validation ───────────────────────────────────────────────────────

def extract_primary_cfr_citation(answer_text: str) -> str | None:
    """
    Extract the first CFR/Treasury Regulation section number cited in the answer.
    Returns the section number string (e.g. "1.162-5") or None if none found.
    """
    patterns = [
        r'Treas\.?\s*Reg\.?\s*§\s*(\d+\.\d+[\w()\-]*)',
        r'Reg\.?\s*§\s*(\d+\.\d+[\w()\-]*)',
        r'§\s*(\d+\.\d+[\w()\-]*)',
        r'\bsections?\s+(\d+\.\d+[\w()\-]*)',
    ]
    for pat in patterns:
        m = re.search(pat, answer_text, re.IGNORECASE)
        if m:
            raw = m.group(1).rstrip(".,;)")
            return raw
    return None


def cfr_base_section(s: str) -> str:
    """
    Return the base CFR section number for loose comparison.
    Handles:
      "1.162-5(a)"   -> "1.162-5"   (subsection paren stripped)
      "1.0-1(b"      -> "1.0-1"     (truncated/dangling paren stripped)
      "1.1(h)-1"     -> "1.1(h)-1"  (paren IS part of section id, kept)
      "1.1(h)-1(a)"  -> "1.1(h)-1"  (section-id paren kept, subsection stripped)
    Strategy: parens followed by a hyphen-digit are part of the section id;
    parens after the final hyphen-digit sequence are subsection indicators.
    """
    m = re.match(r'(\d+\.\d+(?:\([a-zA-Z0-9]+\))?(?:-\d+[A-Za-z]*)?)(?:\(.*)?$', s)
    return m.group(1) if m else s


def validate_cfr_citation(answer: str, source_section: str) -> bool:
    """
    Return True if the answer's primary citation matches the source_section,
    or if no CFR citation is found (benefit of the doubt).
    Return False if a citation IS found and clearly points to a different section.
    """
    primary = extract_primary_cfr_citation(answer)
    if primary is None:
        return True
    return cfr_base_section(primary) == cfr_base_section(source_section)


# ── Prompt construction ───────────────────────────────────────────────────────

def build_cfr_generation_prompt(
    section: dict,
    all_sections: dict[str, dict],
    n: int,
) -> str:
    """Build the generation prompt for a given CFR section."""
    sec_num = section["section"]
    sec_text = section["text"][:MAX_SECTION_CHARS]
    if len(section["text"]) > MAX_SECTION_CHARS:
        sec_text += "\n[Text truncated for length — additional provisions exist]"

    related = build_cfr_related_context(section, all_sections)

    return GENERATION_PROMPT.format(
        section_number=sec_num,
        heading=section["heading"],
        full_section_text=sec_text,
        related_sections_text=related,
        n=n,
    )


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress(progress_file: Path) -> dict:
    """Load progress tracking file, returning empty state if not found."""
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed_sections": [],
        "failed_sections": [],
        "total_pairs": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_discarded": 0,
    }


def save_progress(progress_file: Path, progress: dict) -> None:
    """Save progress tracking file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# ── OpenAI calls ──────────────────────────────────────────────────────────────

def parse_pairs_from_raw(raw: str, sec_num: str) -> tuple[list[dict], int]:
    """
    Parse and validate Q&A pairs from raw JSON response.
    Returns (valid_pairs, discarded_count).
    Applies citation validation to discard cross-section leaks.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            pairs = parsed
        elif isinstance(parsed, dict):
            pairs = []
            for v in parsed.values():
                if isinstance(v, list):
                    pairs = v
                    break
        else:
            pairs = []
    except json.JSONDecodeError:
        print(f"  [WARN] JSON parse failed for §{sec_num}, raw: {raw[:200]}")
        pairs = []

    shaped_pairs = []
    for p in pairs:
        if isinstance(p, dict) and "question" in p and "answer" in p:
            shaped_pairs.append({"question": str(p["question"]), "answer": str(p["answer"])})

    valid_pairs = []
    discarded = 0
    for p in shaped_pairs:
        if validate_cfr_citation(p["answer"], sec_num):
            valid_pairs.append(p)
        else:
            primary = extract_primary_cfr_citation(p["answer"])
            print(
                f"  [DISCARD] §{sec_num}: answer cites §{primary} instead — "
                f"Q: {p['question'][:80]}"
            )
            discarded += 1

    return valid_pairs, discarded


def call_openai_for_pairs(
    client: OpenAI,
    section: dict,
    all_sections: dict[str, dict],
    n: int = PAIRS_PER_SECTION,
) -> tuple[list[dict], dict, int]:
    """
    Call GPT-4o-mini to generate n Q&A pairs for a CFR section.
    Returns (list of {"question": ..., "answer": ...}, usage dict, discarded_count).
    """
    sec_num = section["section"]
    prompt = build_cfr_generation_prompt(section, all_sections, n)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a legal data generation assistant. Always return valid JSON arrays.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=8192,
    )

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    finish_reason = response.choices[0].finish_reason
    raw = response.choices[0].message.content

    if finish_reason == "length":
        print(f"  [WARN] §{sec_num} response was truncated (hit max_tokens)")

    valid_pairs, discarded = parse_pairs_from_raw(raw, sec_num)
    return valid_pairs, usage, discarded


# ── Record formatter ──────────────────────────────────────────────────────────

def make_sft_record(question: str, answer: str, section_num: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source_section": f"CFR §{section_num}",
            "grounded": True,
            "source": "CFR",
        },
    }


# ── Cost estimation ───────────────────────────────────────────────────────────

def compute_cost(total_input_tokens: int, total_output_tokens: int, batch: bool = False) -> float:
    price_in = BATCH_PRICE_PER_1M_INPUT if batch else PRICE_PER_1M_INPUT
    price_out = BATCH_PRICE_PER_1M_OUTPUT if batch else PRICE_PER_1M_OUTPUT
    return (
        total_input_tokens / 1_000_000 * price_in
        + total_output_tokens / 1_000_000 * price_out
    )


# ── Main generation loop ──────────────────────────────────────────────────────

def run_generation(
    client: OpenAI,
    all_sections: dict[str, dict],
    sft_out: Path,
    resume: bool,
    limit: int | None,
    pairs_per_section: int,
) -> None:
    """
    Process CFR sections using direct API calls with rate limiting and resume support.
    """
    progress = load_progress(PROGRESS_FILE) if resume else {
        "completed_sections": [],
        "failed_sections": [],
        "total_pairs": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_discarded": 0,
    }

    completed_set = set(progress["completed_sections"])

    # Build list of sections to process
    sections_list = [
        sec for sec_num, sec in all_sections.items()
        if sec_num not in completed_set
    ]

    # Apply limit if specified
    if limit is not None:
        sections_list = sections_list[:limit]

    total_sections = len(all_sections)
    already_done = len(completed_set)
    to_process = len(sections_list)

    print(f"\nCFR Grounded Data Generation")
    print(f"  Model: {MODEL}")
    print(f"  Total CFR sections in dataset: {total_sections:,}")
    print(f"  Already completed: {already_done:,}")
    print(f"  Sections to process this run: {to_process:,}")
    print(f"  Pairs per section: {pairs_per_section}")
    print(f"  Expected new pairs: ~{to_process * pairs_per_section:,}")
    print(f"  Output: {sft_out}")

    if to_process == 0:
        print("\n  Nothing to do — all sections already completed.")
        if resume:
            print("  Remove progress file to start fresh: rm " + str(PROGRESS_FILE))
        return

    # Load existing SFT records if resuming
    if resume and sft_out.exists():
        existing_count = sum(1 for _ in open(sft_out, encoding="utf-8"))
        print(f"  Resuming: {existing_count} existing records in {sft_out}")

    write_mode = "a" if (resume and sft_out.exists()) else "w"
    sft_out.parent.mkdir(parents=True, exist_ok=True)
    sft_file = open(sft_out, write_mode, encoding="utf-8")

    total_input_tokens = progress["total_input_tokens"]
    total_output_tokens = progress["total_output_tokens"]
    total_discarded = progress["total_discarded"]
    new_pairs_count = 0

    try:
        for i, section in enumerate(sections_list, 1):
            sec_num = section["section"]
            heading = section.get("heading", "")[:60]
            global_idx = already_done + i

            print(f"\n  [{i}/{to_process}] §{sec_num} — {heading}")

            try:
                pairs, usage, discarded = call_openai_for_pairs(
                    client, section, all_sections, n=pairs_per_section
                )
            except Exception as e:
                print(f"  [ERROR] §{sec_num}: {e}")
                progress["failed_sections"].append(sec_num)
                save_progress(PROGRESS_FILE, progress)
                continue

            total_input_tokens += usage["prompt_tokens"]
            total_output_tokens += usage["completion_tokens"]
            total_discarded += discarded
            new_pairs_count += len(pairs)

            kept = len(pairs)
            discard_msg = f" | discarded: {discarded}" if discarded > 0 else ""
            print(f"    Got {kept} valid pairs{discard_msg} | tokens: {usage['total_tokens']}")

            for p in pairs:
                rec = make_sft_record(p["question"], p["answer"], sec_num)
                sft_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

            progress["completed_sections"].append(sec_num)
            progress["total_pairs"] = progress["total_pairs"] + len(pairs)
            progress["total_input_tokens"] = total_input_tokens
            progress["total_output_tokens"] = total_output_tokens
            progress["total_discarded"] = total_discarded

            # Progress report every 10 sections
            if i % 10 == 0 or i == to_process:
                cost = compute_cost(total_input_tokens, total_output_tokens)
                print(
                    f"\n  [PROGRESS] {global_idx}/{total_sections} total | "
                    f"{progress['total_pairs']} pairs | "
                    f"${cost:.4f} spent"
                )

            # Checkpoint every PROGRESS_SAVE_INTERVAL sections
            if i % PROGRESS_SAVE_INTERVAL == 0:
                sft_file.flush()
                save_progress(PROGRESS_FILE, progress)
                print(f"  [CHECKPOINT] Progress saved at section {global_idx}/{total_sections}")

            # Rate limiting
            if i < to_process:
                time.sleep(RATE_LIMIT_SLEEP)

    finally:
        sft_file.flush()
        sft_file.close()
        save_progress(PROGRESS_FILE, progress)

    # ── Final summary ──────────────────────────────────────────────────────────
    total_cost = compute_cost(total_input_tokens, total_output_tokens)
    total_pairs = progress["total_pairs"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Sections processed this run:  {len(progress['completed_sections']) - already_done}")
    print(f"  Sections failed:              {len(progress['failed_sections'])}")
    print(f"  New SFT pairs generated:      {new_pairs_count}")
    print(f"  Total pairs (cumulative):     {total_pairs}")
    print(f"  Pairs discarded (cite check): {total_discarded}")
    print(f"  Total input tokens:           {total_input_tokens:,}")
    print(f"  Total output tokens:          {total_output_tokens:,}")
    print(f"  Total tokens:                 {total_input_tokens + total_output_tokens:,}")
    print(f"  Estimated cost (this run):    ${total_cost:.4f}")
    print(f"  Written to: {sft_out}")

    if progress["failed_sections"]:
        print(f"\n  Failed sections: {progress['failed_sections'][:20]}")
        print(f"  Re-run with --resume to retry failed sections.")

    # Show example pairs
    print("\n" + "=" * 60)
    print("EXAMPLE PAIRS (last batch)")
    print("=" * 60)
    try:
        with open(sft_out, "r", encoding="utf-8") as f:
            all_recs = [json.loads(l) for l in f if l.strip()]
        for j, rec in enumerate(all_recs[-3:], 1):
            msgs = rec["messages"]
            q = msgs[1]["content"]
            a = msgs[2]["content"]
            sec_meta = rec.get("metadata", {}).get("source_section", "")
            print(f"\n--- Example {j} [{sec_meta}] ---")
            print(f"Q: {q}")
            print(f"A: {a[:600]}{'...' if len(a) > 600 else ''}")
    except Exception:
        pass


# ── Cost estimate (dry run) ───────────────────────────────────────────────────

def print_cost_estimate(all_sections: dict[str, dict], pairs_per_section: int) -> None:
    """Print a cost estimate without making any API calls."""
    overhead_tokens = 800
    total_input = 0
    for s in all_sections.values():
        text_chars = min(len(s.get("text", "")), MAX_SECTION_CHARS)
        text_tokens = text_chars / 4
        total_input += text_tokens + overhead_tokens

    total_output = len(all_sections) * pairs_per_section * 500

    cost_direct = compute_cost(int(total_input), int(total_output), batch=False)
    cost_batch = compute_cost(int(total_input), int(total_output), batch=True)

    print(f"\nCost Estimate — {len(all_sections):,} CFR sections × {pairs_per_section} pairs each")
    print(f"  Estimated input tokens:   {total_input:,.0f}")
    print(f"  Estimated output tokens:  {total_output:,}")
    print(f"  Expected SFT pairs:       {len(all_sections) * pairs_per_section:,}")
    print(f"  Direct API cost:          ${cost_direct:.2f}")
    print(f"  Batch API cost (50% off): ${cost_batch:.2f}  [use --prepare-batch]")


# ── Batch API support ─────────────────────────────────────────────────────────

def prepare_batch_file(
    all_sections: dict[str, dict],
    batch_output: Path,
    n: int = PAIRS_PER_SECTION,
    limit: int | None = None,
) -> int:
    """
    Write a JSONL batch input file for the OpenAI Batch API.
    Returns the number of requests written.
    """
    sections_list = list(all_sections.values())
    if limit is not None:
        sections_list = sections_list[:limit]

    batch_output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(batch_output, "w", encoding="utf-8") as f:
        for section in sections_list:
            sec_num = section["section"]
            prompt = build_cfr_generation_prompt(section, all_sections, n)
            request = {
                "custom_id": f"cfr-{sec_num}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a legal data generation assistant. Always return valid JSON arrays.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7,
                    "max_tokens": 8192,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
    return count


def download_batch_results(
    client: OpenAI,
    output_file_id: str,
    sft_out: Path,
    all_sections: dict[str, dict],
) -> tuple[int, int]:
    """
    Download batch results, parse Q&A pairs, write SFT JSONL.
    Returns (sft_count, discarded_count).
    """
    print(f"\nDownloading batch results (file_id={output_file_id})...")
    content = client.files.content(output_file_id)
    raw_lines = content.text.strip().split("\n")
    print(f"  Got {len(raw_lines)} result lines")

    sft_records = []
    total_discarded = 0

    for line in raw_lines:
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue

        custom_id = result.get("custom_id", "")
        sec_num = custom_id.replace("cfr-", "") if custom_id.startswith("cfr-") else None
        if not sec_num:
            continue

        error = result.get("error")
        if error:
            print(f"  [ERROR] §{sec_num}: {error}")
            continue

        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            continue

        raw_content = choices[0].get("message", {}).get("content", "")
        finish_reason = choices[0].get("finish_reason", "")
        if finish_reason == "length":
            print(f"  [WARN] §{sec_num} response was truncated")

        pairs, discarded = parse_pairs_from_raw(raw_content, sec_num)
        total_discarded += discarded

        for p in pairs:
            rec = make_sft_record(p["question"], p["answer"], sec_num)
            sft_records.append(rec)

    sft_out.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_out, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written {len(sft_records)} SFT records to {sft_out}")
    return len(sft_records), total_discarded


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate grounded SFT training data from CFR (Treasury Regulation) sections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run: 10 sections
  python3 scripts/generate_cfr_grounded_data.py --limit 10

  # Full run: all 6,149 sections (~$18 direct, ~$9 batch)
  python3 scripts/generate_cfr_grounded_data.py

  # Resume an interrupted full run
  python3 scripts/generate_cfr_grounded_data.py --resume

  # Print cost estimate only (no API calls)
  python3 scripts/generate_cfr_grounded_data.py --estimate-cost

  # Prepare batch input file for OpenAI Batch API (50% cheaper, async)
  python3 scripts/generate_cfr_grounded_data.py --prepare-batch

  # Download completed batch results
  python3 scripts/generate_cfr_grounded_data.py --download-batch BATCH_FILE_ID \\
      --output data/processed/grounded_cfr_sft_full.jsonl
""",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N sections (for testing). Default: all 6,149.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file path. Defaults to grounded_cfr_sft_test.jsonl (if --limit) "
             "or grounded_cfr_sft_full.jsonl (full run).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved progress checkpoint.",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Print cost estimate for the full run and exit (no API calls).",
    )
    parser.add_argument(
        "--prepare-batch",
        action="store_true",
        help="Write a batch input JSONL file for the OpenAI Batch API instead of calling directly.",
    )
    parser.add_argument(
        "--batch-output",
        type=Path,
        default=PROCESSED_DIR / "cfr_batch_input.jsonl",
        help="Path for batch input file (used with --prepare-batch).",
    )
    parser.add_argument(
        "--download-batch",
        metavar="FILE_ID",
        help="Download and process completed batch results by output file ID.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CFR_JSONL,
        help=f"Path to CFR sections JSONL file. Default: {CFR_JSONL}",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        sft_out = args.output
    elif args.limit is not None:
        sft_out = SFT_OUT_TEST
    else:
        sft_out = SFT_OUT_FULL

    # Load CFR sections
    print(f"Loading CFR sections from {args.input}...")
    all_sections = load_cfr_sections(args.input)
    print(f"  Loaded {len(all_sections):,} CFR sections")

    if not all_sections:
        print("[ERROR] No CFR sections found. Check the input file.")
        return

    # ── Estimate cost only ─────────────────────────────────────────────────────
    if args.estimate_cost:
        print_cost_estimate(all_sections, PAIRS_PER_SECTION)
        return

    # ── Prepare batch file ─────────────────────────────────────────────────────
    if args.prepare_batch:
        print(f"\nPreparing batch input file: {args.batch_output}")
        count = prepare_batch_file(all_sections, args.batch_output, limit=args.limit)
        print(f"  Written {count:,} requests to {args.batch_output}")
        print_cost_estimate(
            {k: v for k, v in list(all_sections.items())[:count]},
            PAIRS_PER_SECTION,
        )
        print(
            f"\nNext step: submit with\n"
            f"  python3 -c \"\nimport openai, json\n"
            f"client = openai.OpenAI()\n"
            f"with open('{args.batch_output}', 'rb') as f:\n"
            f"    upload = client.files.create(file=f, purpose='batch')\n"
            f"batch = client.batches.create(\n"
            f"    input_file_id=upload.id,\n"
            f"    endpoint='/v1/chat/completions',\n"
            f"    completion_window='24h'\n"
            f")\nprint(batch.id)\n\""
        )
        return

    # ── Download batch results ─────────────────────────────────────────────────
    if args.download_batch:
        client = OpenAI()
        sft_count, discarded = download_batch_results(
            client, args.download_batch, sft_out, all_sections
        )
        print(f"\nBatch download complete: {sft_count:,} pairs, {discarded} discarded")
        return

    # ── Direct API generation ──────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key)

    run_generation(
        client=client,
        all_sections=all_sections,
        sft_out=sft_out,
        resume=args.resume,
        limit=args.limit,
        pairs_per_section=PAIRS_PER_SECTION,
    )


if __name__ == "__main__":
    main()
