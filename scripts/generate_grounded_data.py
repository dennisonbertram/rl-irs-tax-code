#!/usr/bin/env python3
"""
Generate RAG-grounded SFT and DPO training data from IRC sections using GPT-4o-mini.

Outputs:
  - data/processed/grounded_sft_test.jsonl   (~100 pairs, chat format)
  - data/processed/grounded_dpo_test.jsonl   (~10 DPO pairs with hard-negatives)

Full-run outputs:
  - data/processed/grounded_sft_full.jsonl   (~18-20K pairs from all 2,113 sections)
  - data/processed/grounded_dpo_full.jsonl   (~2K DPO pairs)

v2 fixes applied:
  1. Cross-section metadata leak — post-generation validation discards pairs whose
     primary citation does not match the source section.
  2. TCJA-modified sections — relevant sections get an amendment notice injected
     into the generation prompt.
  3. Inflation-adjusted dollar amounts — prompt-level instruction added.
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
IRC_JSONL = PROCESSED_DIR / "irc_sections.jsonl"
SFT_OUT = PROCESSED_DIR / "grounded_sft_test.jsonl"
DPO_OUT = PROCESSED_DIR / "grounded_dpo_test.jsonl"
SFT_OUT_V2 = PROCESSED_DIR / "grounded_sft_test_v2.jsonl"
DPO_OUT_V2 = PROCESSED_DIR / "grounded_dpo_test_v2.jsonl"
SFT_OUT_FULL = PROCESSED_DIR / "grounded_sft_full.jsonl"
DPO_OUT_FULL = PROCESSED_DIR / "grounded_dpo_full.jsonl"
PROGRESS_FILE = PROCESSED_DIR / ".generation_progress.json"
BATCH_INPUT_DEFAULT = PROCESSED_DIR / "batch_input.jsonl"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
PAIRS_PER_SECTION = 8          # 8 * ~13 sections ≈ 104 pairs (test)
PAIRS_PER_SECTION_FULL = 9     # 9 * 2113 ≈ 19K pairs (full run)
DPO_PAIRS_TO_GENERATE = 10     # for test mode
DPO_FRACTION_FULL = 0.10       # 10% of SFT pairs become DPO in full mode
MAX_SECTION_CHARS = 6000       # Cap section text sent to API
RANDOM_SEED = 42
PROGRESS_SAVE_INTERVAL = 50    # Save progress every N sections
RATE_LIMIT_SLEEP = 0.5         # Seconds between API calls

# Target sections: mix of common, important, and obscure
TARGET_SECTIONS = [
    "1",     # Tax imposed (income tax brackets)
    "61",    # Gross income defined
    "162",   # Trade or business expenses
    "170",   # Charitable contributions
    "179",   # Election to expense depreciable assets
    "401",   # Qualified pension plans / 401k
    "1031",  # Like-kind exchanges
    "6662",  # Accuracy-related penalty
    "21",    # Child and dependent care expenses
    "132",   # Certain fringe benefits (obscure-ish)
    "469",   # Passive activity losses (important, complex)
    "1221",  # Capital asset defined
    "7701",  # Definitions (obscure but foundational)
]

SYSTEM_PROMPT = (
    "You are a tax law expert specializing in the Internal Revenue Code (IRC). "
    "Always cite specific IRC sections and subsections when answering questions. "
    "Provide accurate, detailed explanations grounded strictly in the statutory text."
)

random.seed(RANDOM_SEED)


# ── Fix 2: TCJA-modified section registry ─────────────────────────────────────

# Maps section number → description of the TCJA change.
# Used to inject an amendment notice into the generation prompt.
TCJA_AMENDMENTS: dict[str, str] = {
    # Charitable deduction — cash gifts to public charities raised from 50% to 60% AGI limit
    "170": (
        "The Tax Cuts and Jobs Act of 2017 (P.L. 115-97) raised the AGI percentage limit "
        "for cash contributions to public charities from 50% to 60% under §170(b)(1)(G). "
        "This change applies to tax years 2018 through 2025."
    ),
    # Like-kind exchanges — now limited to real property only
    "1031": (
        "The Tax Cuts and Jobs Act of 2017 (P.L. 115-97) restricted §1031 like-kind exchanges "
        "to real property only. Personal property (equipment, vehicles, artwork, collectibles, "
        "intangibles) no longer qualifies for tax-deferred exchange treatment. "
        "Applicable to exchanges completed after December 31, 2017."
    ),
    # Qualified business income deduction — entirely new
    "199A": (
        "IRC §199A was created by the Tax Cuts and Jobs Act of 2017 (P.L. 115-97). "
        "It provides a 20% deduction for qualified business income from pass-through entities "
        "for tax years 2018 through 2025."
    ),
    # State and local tax deduction capped
    "164": (
        "The Tax Cuts and Jobs Act of 2017 (P.L. 115-97) added §164(b)(6), which caps the "
        "itemized deduction for state and local taxes (SALT) at $10,000 ($5,000 for married "
        "filing separately) for tax years 2018 through 2025."
    ),
    # Mortgage interest — acquisition debt limit reduced
    "163": (
        "The Tax Cuts and Jobs Act of 2017 (P.L. 115-97) reduced the mortgage interest "
        "deduction limit under §163(h)(3) from $1,000,000 to $750,000 of acquisition "
        "indebtedness for loans taken out after December 15, 2017. "
        "The $1,000,000 limit still applies to loans taken out on or before that date."
    ),
    # Personal exemption suspended
    "151": (
        "The Tax Cuts and Jobs Act of 2017 (P.L. 115-97) suspended the personal exemption "
        "deduction under §151 for tax years 2018 through 2025 (the exemption amount is $0)."
    ),
    # Corporate tax rate — flat 21%
    "11": (
        "The Tax Cuts and Jobs Act of 2017 (P.L. 115-97) replaced the graduated corporate "
        "tax rate structure under §11 with a flat 21% rate, effective for tax years beginning "
        "after December 31, 2017."
    ),
    # Individual rate brackets modified
    "1": (
        "The Tax Cuts and Jobs Act of 2017 (P.L. 115-97) modified the individual income tax "
        "rate brackets under §1(j), creating rates of 10%, 12%, 22%, 24%, 32%, 35%, and 37% "
        "for tax years 2018 through 2025, replacing the prior brackets of 10%, 15%, 25%, "
        "28%, 33%, 35%, and 39.6%."
    ),
    # Standard deduction nearly doubled
    "63": (
        "The Tax Cuts and Jobs Act of 2017 (P.L. 115-97) nearly doubled the standard "
        "deduction under §63(c) for tax years 2018 through 2025. The 2018 amounts were "
        "$12,000 (single), $24,000 (married filing jointly), and $18,000 (head of household), "
        "subject to annual inflation adjustment."
    ),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_irc_sections(path: Path) -> dict[str, dict]:
    """Load all IRC sections indexed by section number."""
    sections: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("source") == "IRC":
                    sections[d["section"]] = d
    return sections


# ── Cross-reference extraction ────────────────────────────────────────────────

def extract_cross_refs(text: str) -> list[str]:
    """
    Find section references like 'section 414', 'sections 401 and 403', '§ 72', etc.
    Returns a list of section number strings.
    """
    # Matches: "section 401", "§401", "§ 401", "sections 414 and 416"
    patterns = [
        r'§\s*(\d+[A-Za-z]?(?:\(\w+\))*)',
        r'\bsections?\s+(\d+[A-Za-z]?)',
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            # Grab only the numeric+letter part (no subsection parens)
            raw = m.group(1)
            num = re.match(r'(\d+[A-Za-z]?)', raw)
            if num:
                found.add(num.group(1))
    return list(found)


def build_related_context(
    section: dict,
    all_sections: dict[str, dict],
    max_related: int = 3,
    max_chars_each: int = 800,
) -> str:
    """Build a related-sections context block by following cross-references."""
    text = section.get("text", "")
    refs = extract_cross_refs(text)

    # Remove self-reference
    refs = [r for r in refs if r != section["section"]]

    # Only keep refs we actually have data for
    available = [r for r in refs if r in all_sections]

    # Deduplicate, limit count
    seen = set()
    related_texts = []
    for ref in available:
        if ref in seen or len(related_texts) >= max_related:
            break
        seen.add(ref)
        sec = all_sections[ref]
        snippet = sec["text"][:max_chars_each]
        related_texts.append(
            f"IRC §{ref} — {sec['heading']}:\n{snippet}"
            + ("..." if len(sec["text"]) > max_chars_each else "")
        )

    if not related_texts:
        return "(No cross-referenced sections found in the dataset.)"
    return "\n\n".join(related_texts)


# ── Fix 1: Cross-section citation validation ──────────────────────────────────

def extract_primary_citation(answer_text: str) -> str | None:
    """
    Extract the first IRC section number cited in the answer.
    Returns the section number string (e.g. "1031") or None if none found.
    We look for patterns like '§1031', 'IRC §1031', 'section 1031'.
    We return only the base section number without subsection letters/parens.
    """
    patterns = [
        r'IRC\s*§\s*(\d+[A-Za-z]?)',
        r'§\s*(\d+[A-Za-z]?)',
        r'\bsections?\s+(\d+[A-Za-z]?)',
        r'\bI\.R\.C\.\s*§\s*(\d+[A-Za-z]?)',
    ]
    for pat in patterns:
        m = re.search(pat, answer_text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            num = re.match(r'(\d+[A-Za-z]?)', raw)
            if num:
                return num.group(1)
    return None


def validate_citation_matches_source(answer: str, source_section: str) -> bool:
    """
    Return True if the answer's primary citation matches the source_section,
    or if no citation can be found (give benefit of the doubt on citation-free answers).
    Return False if a citation IS found and it clearly points to a different section.
    """
    primary = extract_primary_citation(answer)
    if primary is None:
        # No citation found — cannot confirm a leak, keep the pair
        return True
    # Normalize: strip any trailing letter suffix for the comparison
    # so that §1031A vs §1031 does not false-positive discard
    def base_num(s: str) -> str:
        m = re.match(r'(\d+)', s)
        return m.group(1) if m else s

    return base_num(primary) == base_num(source_section)


# ── Prompt construction ───────────────────────────────────────────────────────

# Fix 2 placeholder is injected conditionally below when the section is TCJA-modified.
# Fix 3 inflation note is always included.

GENERATION_PROMPT = """\
You are generating training data for a tax law AI assistant.

Below is the EXACT text of IRC Section {section_number}: {heading}

---
{full_section_text}
---

Related sections referenced in the text above (for context only — do NOT generate \
questions about them):
{related_sections_text}

{tcja_note}\
Generate {n} diverse question-answer pairs about this section.

RULES:
1. Every answer MUST be grounded in the text above. Do not add information not present in the source text.
2. Directly quote or closely paraphrase the statute language.
3. Always cite the specific subsection (e.g., "Under IRC §401(a)(4)..." not just "Under IRC §401...").
4. Include important exceptions, limitations, and cross-references mentioned in the text.
5. Vary question types: definitional ("What is..."), procedural ("How does..."), conditional ("When can..."), comparative ("What is the difference between..."), edge case ("Does X apply if...").
6. If dollar amounts, dates, percentages, or thresholds appear in the text, include them accurately.
7. End each answer with: "For personalized advice, consult a qualified tax professional."
8. IMPORTANT: All questions and answers MUST be about Section {section_number} specifically. \
Do not generate questions about the related sections provided as context — they are only for \
cross-reference understanding.
9. When citing dollar amounts from the statute, note that these are the statutory base amounts \
and may be adjusted annually for inflation under IRC §1(f). Include a note like \
"subject to annual inflation adjustment" when relevant.
10. If the statute text references effective dates or amendments, note them in the answer.

Return ONLY a JSON array of objects with "question" and "answer" fields — no other text.
"""

TCJA_NOTE_TEMPLATE = """\
AMENDMENT NOTICE — THIS SECTION WAS MODIFIED BY THE TAX CUTS AND JOBS ACT (TCJA):
{description}
Ensure all answers reflect current law as amended by the TCJA.

"""


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


# ── Prompt builder (reusable for both direct API and batch) ───────────────────

def build_generation_prompt(
    section: dict,
    all_sections: dict[str, dict],
    n: int,
) -> str:
    """Build the generation prompt for a given section."""
    sec_num = section["section"]
    sec_text = section["text"][:MAX_SECTION_CHARS]
    if len(section["text"]) > MAX_SECTION_CHARS:
        sec_text += "\n[Text truncated for length — additional provisions exist]"

    related = build_related_context(section, all_sections)

    tcja_description = TCJA_AMENDMENTS.get(sec_num, "")
    if tcja_description:
        tcja_note = TCJA_NOTE_TEMPLATE.format(description=tcja_description)
    else:
        tcja_note = ""

    return GENERATION_PROMPT.format(
        section_number=sec_num,
        heading=section["heading"],
        full_section_text=sec_text,
        related_sections_text=related,
        tcja_note=tcja_note,
        n=n,
    )


# ── OpenAI calls ──────────────────────────────────────────────────────────────

def parse_pairs_from_raw(raw: str, sec_num: str) -> tuple[list[dict], int]:
    """
    Parse and validate Q&A pairs from raw JSON response.
    Returns (valid_pairs, discarded_count).
    Applies Fix 1 (cross-section citation validation).
    """
    try:
        parsed = json.loads(raw)
        # Handle {"pairs": [...]}, {"data": [...]}, {"questions": [...]}, or bare list
        if isinstance(parsed, list):
            pairs = parsed
        elif isinstance(parsed, dict):
            # Find the first list value regardless of key name
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

    # Validate shape
    shaped_pairs = []
    for p in pairs:
        if isinstance(p, dict) and "question" in p and "answer" in p:
            shaped_pairs.append({"question": str(p["question"]), "answer": str(p["answer"])})

    # Fix 1: discard pairs whose primary citation is for a different section
    valid_pairs = []
    discarded = 0
    for p in shaped_pairs:
        if validate_citation_matches_source(p["answer"], sec_num):
            valid_pairs.append(p)
        else:
            primary = extract_primary_citation(p["answer"])
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
    Call GPT-4o-mini to generate n Q&A pairs for a section.
    Returns (list of {"question": ..., "answer": ...}, usage dict, discarded_count).
    Applies Fix 1 (cross-section citation validation) before returning.
    """
    sec_num = section["section"]
    prompt = build_generation_prompt(section, all_sections, n)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a legal data generation assistant. Always return valid JSON arrays."},
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


def call_openai_for_dpo(
    client: OpenAI,
    question: str,
    correct_answer: str,
) -> tuple[str, str, dict]:
    """
    Generate a hard-negative rejected answer for DPO.
    Returns (rejected_answer, error_description, usage).
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
        rejected = parsed.get("rejected_answer", "")
        error_desc = parsed.get("error_description", "")
    except json.JSONDecodeError:
        rejected = ""
        error_desc = "parse error"

    return rejected, error_desc, usage


# ── Record formatters ─────────────────────────────────────────────────────────

def make_sft_record(question: str, answer: str, section_num: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source_section": f"IRC §{section_num}",
            "grounded": True,
            "tcja_modified": section_num in TCJA_AMENDMENTS,
        },
    }


def make_dpo_record(
    question: str,
    chosen_answer: str,
    rejected_answer: str,
    section_num: str,
    error_description: str,
) -> dict:
    return {
        "prompt": question,
        "chosen": chosen_answer,
        "rejected": rejected_answer,
        "metadata": {
            "source_section": f"IRC §{section_num}",
            "error_introduced": error_description,
        },
    }


# ── Cost estimation ───────────────────────────────────────────────────────────

# GPT-4o-mini pricing (per 1M tokens) as of 2025
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60

# Batch API is 50% cheaper
BATCH_PRICE_PER_1M_INPUT = 0.075
BATCH_PRICE_PER_1M_OUTPUT = 0.30


def compute_cost(total_input_tokens: int, total_output_tokens: int, batch: bool = False) -> float:
    price_in = BATCH_PRICE_PER_1M_INPUT if batch else PRICE_PER_1M_INPUT
    price_out = BATCH_PRICE_PER_1M_OUTPUT if batch else PRICE_PER_1M_OUTPUT
    return (
        total_input_tokens / 1_000_000 * price_in
        + total_output_tokens / 1_000_000 * price_out
    )


# ── DPO candidate selection ───────────────────────────────────────────────────

def select_dpo_candidates(
    all_pairs_flat: list[tuple[str, str, str]],
    n: int,
) -> list[tuple[str, str, str]]:
    """
    Stratified sample of n pairs from all_pairs_flat for DPO generation.
    Spreads selection across all sections.
    """
    if len(all_pairs_flat) <= n:
        return all_pairs_flat[:]

    random.shuffle(all_pairs_flat)
    by_section: dict[str, list] = {}
    for q, a, sec in all_pairs_flat:
        by_section.setdefault(sec, []).append((q, a, sec))

    candidates = []
    sec_list = list(by_section.values())
    idx = 0
    while len(candidates) < n:
        bucket = sec_list[idx % len(sec_list)]
        if bucket:
            candidates.append(bucket.pop(0))
        idx += 1
        # Safety: break if all buckets exhausted
        if all(len(b) == 0 for b in sec_list):
            break

    return candidates[:n]


# ── Batch API support ─────────────────────────────────────────────────────────

def prepare_batch_file(
    all_sections: dict[str, dict],
    batch_output: Path,
    n: int = PAIRS_PER_SECTION_FULL,
) -> int:
    """
    Write a JSONL batch input file for the OpenAI Batch API.
    Each line is a request object with a custom_id for tracking.
    Returns the number of requests written.
    """
    batch_output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(batch_output, "w", encoding="utf-8") as f:
        for sec_num, section in all_sections.items():
            prompt = build_generation_prompt(section, all_sections, n)
            request = {
                "custom_id": f"irc-{sec_num}",
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


def submit_batch(client: OpenAI, batch_input: Path) -> str:
    """Upload batch input file and submit batch job. Returns batch ID."""
    print(f"  Uploading batch file: {batch_input} ({batch_input.stat().st_size / 1024:.1f} KB)...")
    with open(batch_input, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    file_id = upload.id
    print(f"  File uploaded: {file_id}")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "IRC full dataset generation — all 2113 sections"},
    )
    print(f"  Batch submitted: {batch.id} (status: {batch.status})")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str, poll_interval: int = 60) -> str | None:
    """
    Poll batch status until complete.
    Returns output_file_id on success, None on failure.
    """
    import sys

    print(f"\nPolling batch {batch_id} every {poll_interval}s...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        print(
            f"  Status: {status} | "
            f"total={counts.total} completed={counts.completed} failed={counts.failed}",
            flush=True,
        )
        if status == "completed":
            return batch.output_file_id
        elif status in ("failed", "expired", "cancelled"):
            print(f"  [ERROR] Batch ended with status: {status}")
            return None
        time.sleep(poll_interval)


def download_batch_results(
    client: OpenAI,
    output_file_id: str,
    sft_out: Path,
    dpo_out: Path,
    all_sections: dict[str, dict],
    dpo_count: int,
) -> tuple[int, int, int]:
    """
    Download batch results, parse Q&A pairs, write SFT JSONL.
    Then generate DPO pairs from a stratified sample.
    Returns (sft_count, dpo_count_actual, discarded_count).
    """
    print(f"\nDownloading batch results (file_id={output_file_id})...")
    content = client.files.content(output_file_id)
    raw_lines = content.text.strip().split("\n")
    print(f"  Got {len(raw_lines)} result lines")

    sft_records = []
    all_pairs_flat: list[tuple[str, str, str]] = []
    total_discarded = 0

    for line in raw_lines:
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue

        custom_id = result.get("custom_id", "")
        sec_num = custom_id.replace("irc-", "") if custom_id.startswith("irc-") else None
        if not sec_num:
            continue

        # Check for API-level error
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
            all_pairs_flat.append((p["question"], p["answer"], sec_num))

    # Write SFT
    sft_out.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_out, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written {len(sft_records)} SFT records to {sft_out}")

    # DPO note: batch API was used for SFT; DPO needs a separate direct API call
    print(f"\n  [NOTE] DPO generation requires direct API calls (not included in batch results).")
    print(f"  Run with --generate-dpo-from {sft_out} to generate DPO pairs separately.")

    return len(sft_records), 0, total_discarded


# ── Full run: direct API, all sections ───────────────────────────────────────

def run_full_direct(
    client: OpenAI,
    all_sections: dict[str, dict],
    sft_out: Path,
    dpo_out: Path,
    resume: bool,
    pairs_per_section: int,
) -> None:
    """
    Process all sections using direct API calls with rate limiting and resume support.
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
    sections_to_process = [
        sec for sec_num, sec in all_sections.items()
        if sec_num not in completed_set
    ]

    total_sections = len(all_sections)
    already_done = len(completed_set)

    if resume and already_done > 0:
        print(f"  Resuming: {already_done} sections already completed, {len(sections_to_process)} remaining")

    # Load existing SFT records if resuming
    sft_records_existing = []
    all_pairs_flat_existing: list[tuple[str, str, str]] = []
    if resume and sft_out.exists():
        with open(sft_out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    sft_records_existing.append(rec)
                    msgs = rec.get("messages", [])
                    if len(msgs) >= 3:
                        sec_meta = rec.get("metadata", {}).get("source_section", "")
                        sec_num = sec_meta.replace("IRC §", "")
                        all_pairs_flat_existing.append((msgs[1]["content"], msgs[2]["content"], sec_num))
        print(f"  Loaded {len(sft_records_existing)} existing SFT records")

    # Open SFT output in append mode if resuming, write mode otherwise
    write_mode = "a" if (resume and sft_out.exists()) else "w"
    sft_out.parent.mkdir(parents=True, exist_ok=True)
    sft_file = open(sft_out, write_mode, encoding="utf-8")

    new_pairs_flat: list[tuple[str, str, str]] = []
    total_input_tokens = progress["total_input_tokens"]
    total_output_tokens = progress["total_output_tokens"]
    total_discarded = progress["total_discarded"]
    new_pairs_count = 0

    try:
        for i, section in enumerate(sections_to_process, 1):
            sec_num = section["section"]
            global_idx = already_done + i

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

            for p in pairs:
                rec = make_sft_record(p["question"], p["answer"], sec_num)
                sft_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new_pairs_flat.append((p["question"], p["answer"], sec_num))

            progress["completed_sections"].append(sec_num)
            progress["total_pairs"] = (progress["total_pairs"] + len(pairs))
            progress["total_input_tokens"] = total_input_tokens
            progress["total_output_tokens"] = total_output_tokens
            progress["total_discarded"] = total_discarded

            # Progress report every 10 sections
            if global_idx % 10 == 0 or i == len(sections_to_process):
                cost = compute_cost(total_input_tokens, total_output_tokens)
                total_pairs_so_far = progress["total_pairs"]
                print(
                    f"  Processed {global_idx}/{total_sections} sections | "
                    f"{total_pairs_so_far} pairs | "
                    f"${cost:.4f} spent"
                )

            # Save progress every PROGRESS_SAVE_INTERVAL sections
            if i % PROGRESS_SAVE_INTERVAL == 0:
                sft_file.flush()
                save_progress(PROGRESS_FILE, progress)
                print(f"  [CHECKPOINT] Progress saved at section {global_idx}/{total_sections}")

            # Rate limiting
            if i < len(sections_to_process):
                time.sleep(RATE_LIMIT_SLEEP)

    finally:
        sft_file.flush()
        sft_file.close()
        save_progress(PROGRESS_FILE, progress)

    total_sft = progress["total_pairs"]
    print(f"\n  SFT generation complete: {total_sft} pairs")
    print(f"  Failed sections: {len(progress['failed_sections'])}")
    print(f"  Written to {sft_out}")

    # ── DPO generation ────────────────────────────────────────────────────────
    all_pairs_flat = all_pairs_flat_existing + new_pairs_flat
    dpo_count = max(1, int(len(all_pairs_flat) * DPO_FRACTION_FULL))
    print(f"\nGenerating ~{dpo_count} DPO pairs ({DPO_FRACTION_FULL:.0%} of {len(all_pairs_flat)} SFT pairs)...")

    dpo_candidates = select_dpo_candidates(all_pairs_flat, dpo_count)

    dpo_records = []
    dpo_out.parent.mkdir(parents=True, exist_ok=True)
    with open(dpo_out, "w", encoding="utf-8") as dpo_file:
        for j, (question, chosen_answer, sec_num) in enumerate(dpo_candidates, 1):
            try:
                rejected, error_desc, usage = call_openai_for_dpo(client, question, chosen_answer)
                total_input_tokens += usage["prompt_tokens"]
                total_output_tokens += usage["completion_tokens"]

                if rejected:
                    rec = make_dpo_record(question, chosen_answer, rejected, sec_num, error_desc)
                    dpo_records.append(rec)
                    dpo_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if j % 100 == 0 or j == len(dpo_candidates):
                        print(f"  DPO {j}/{len(dpo_candidates)} | {len(dpo_records)} generated")
                else:
                    print(f"  [WARN] §{sec_num} DPO: empty rejected answer — skipping")

                if j < len(dpo_candidates):
                    time.sleep(0.3)
            except Exception as e:
                print(f"  [ERROR] DPO §{sec_num}: {e}")
                continue

    print(f"  Written {len(dpo_records)} DPO pairs to {dpo_out}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_cost = compute_cost(total_input_tokens, total_output_tokens)
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Sections processed:     {len(progress['completed_sections'])}")
    print(f"  Sections failed:        {len(progress['failed_sections'])}")
    print(f"  SFT pairs generated:    {total_sft}")
    print(f"  DPO pairs generated:    {len(dpo_records)}")
    print(f"  Pairs discarded (Fix 1 — cross-section): {total_discarded}")
    print(f"  Total input tokens:     {total_input_tokens:,}")
    print(f"  Total output tokens:    {total_output_tokens:,}")
    print(f"  Total tokens:           {total_input_tokens + total_output_tokens:,}")
    print(f"  Estimated cost:         ${total_cost:.4f}")
    if progress["failed_sections"]:
        print(f"\n  Failed sections: {progress['failed_sections'][:20]}")
        print(f"  Re-run with --resume to retry failed sections.")


# ── Test run (original behavior) ─────────────────────────────────────────────

def run_test(
    client: OpenAI,
    all_sections: dict[str, dict],
    sft_out: Path,
    dpo_out: Path,
) -> None:
    """Original test run over TARGET_SECTIONS only."""

    # Resolve target sections — skip any not in dataset
    target_sections = []
    for sec_num in TARGET_SECTIONS:
        if sec_num in all_sections:
            target_sections.append(all_sections[sec_num])
        else:
            print(f"  [WARN] Section §{sec_num} not found — skipping")

    print(f"\nGenerating SFT pairs for {len(target_sections)} sections ({PAIRS_PER_SECTION} pairs each)...")
    print(f"  Model: {MODEL}")
    print(f"  Estimated total pairs: ~{len(target_sections) * PAIRS_PER_SECTION}")
    print(f"  TCJA-modified sections in target list: "
          f"{[s for s in TARGET_SECTIONS if s in TCJA_AMENDMENTS]}")

    sft_records = []
    all_pairs_flat: list[tuple[str, str, str]] = []  # (question, answer, section_num)

    total_input_tokens = 0
    total_output_tokens = 0
    total_discarded = 0

    # Track TCJA section pairs for example output
    tcja_example_pairs: dict[str, list[dict]] = {}

    for i, section in enumerate(target_sections, 1):
        sec_num = section["section"]
        heading = section["heading"]
        is_tcja = sec_num in TCJA_AMENDMENTS
        tcja_tag = " [TCJA-MODIFIED]" if is_tcja else ""
        print(f"\n  [{i}/{len(target_sections)}] §{sec_num} — {heading[:60]}{tcja_tag}")

        pairs, usage, discarded = call_openai_for_pairs(
            client, section, all_sections, n=PAIRS_PER_SECTION
        )

        total_input_tokens += usage["prompt_tokens"]
        total_output_tokens += usage["completion_tokens"]
        total_discarded += discarded

        kept = len(pairs)
        discard_msg = f" | discarded: {discarded}" if discarded > 0 else ""
        print(f"    Got {kept} valid pairs{discard_msg} | tokens: {usage['total_tokens']}")

        for p in pairs:
            rec = make_sft_record(p["question"], p["answer"], sec_num)
            sft_records.append(rec)
            all_pairs_flat.append((p["question"], p["answer"], sec_num))
            if is_tcja:
                tcja_example_pairs.setdefault(sec_num, []).append(p)

        # Small delay to avoid rate limits
        if i < len(target_sections):
            time.sleep(0.5)

    print(f"\n  Total SFT records: {len(sft_records)}")
    print(f"  Total discarded (cross-section leak): {total_discarded}")

    # Write SFT output
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(sft_out, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written to {sft_out}")

    # ── DPO generation ────────────────────────────────────────────────────────
    print(f"\nGenerating {DPO_PAIRS_TO_GENERATE} DPO hard-negative pairs...")

    dpo_candidates = select_dpo_candidates(all_pairs_flat, DPO_PAIRS_TO_GENERATE)

    dpo_records = []
    for i, (question, chosen_answer, sec_num) in enumerate(
        dpo_candidates[:DPO_PAIRS_TO_GENERATE], 1
    ):
        print(f"  [{i}/{DPO_PAIRS_TO_GENERATE}] §{sec_num} — corrupting answer...")
        rejected, error_desc, usage = call_openai_for_dpo(client, question, chosen_answer)

        total_input_tokens += usage["prompt_tokens"]
        total_output_tokens += usage["completion_tokens"]

        if rejected:
            rec = make_dpo_record(question, chosen_answer, rejected, sec_num, error_desc)
            dpo_records.append(rec)
            print(f"    Error introduced: {error_desc[:80]}")
        else:
            print(f"    [WARN] Empty rejected answer — skipping")

        if i < DPO_PAIRS_TO_GENERATE:
            time.sleep(0.3)

    with open(dpo_out, "w", encoding="utf-8") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written {len(dpo_records)} DPO pairs to {dpo_out}")

    # ── Cost summary ──────────────────────────────────────────────────────────
    total_cost = compute_cost(total_input_tokens, total_output_tokens)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  SFT pairs generated:    {len(sft_records)}")
    print(f"  DPO pairs generated:    {len(dpo_records)}")
    print(f"  Pairs discarded (Fix 1 — cross-section): {total_discarded}")
    print(f"  Total input tokens:     {total_input_tokens:,}")
    print(f"  Total output tokens:    {total_output_tokens:,}")
    print(f"  Total tokens:           {total_input_tokens + total_output_tokens:,}")
    print(f"  Estimated cost:         ${total_cost:.4f}")

    # ── TCJA example pairs ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TCJA-MODIFIED SECTION EXAMPLES")
    print("=" * 60)
    shown = 0
    for sec_num in ["170", "1031", "1", "163"]:
        examples = tcja_example_pairs.get(sec_num, [])
        for p in examples[:2]:
            if shown >= 4:
                break
            print(f"\n--- §{sec_num} example ---")
            print(f"Q: {p['question']}")
            print(f"A: {p['answer'][:700]}{'...' if len(p['answer']) > 700 else ''}")
            shown += 1
        if shown >= 4:
            break

    # ── Print example pairs ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXAMPLE PAIRS (first 4 SFT records)")
    print("=" * 60)
    for i, rec in enumerate(sft_records[:4], 1):
        msgs = rec["messages"]
        q = msgs[1]["content"]
        a = msgs[2]["content"]
        section_meta = rec.get("metadata", {}).get("source_section", "")
        print(f"\n--- Example {i} [{section_meta}] ---")
        print(f"Q: {q}")
        print(f"A: {a[:600]}{'...' if len(a) > 600 else ''}")

    if dpo_records:
        print("\n" + "=" * 60)
        print("EXAMPLE DPO PAIR (first record)")
        print("=" * 60)
        rec = dpo_records[0]
        print(f"Q:        {rec['prompt'][:200]}")
        print(f"Chosen:   {rec['chosen'][:300]}...")
        print(f"Rejected: {rec['rejected'][:300]}...")
        print(f"Error:    {rec['metadata']['error_introduced']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate grounded SFT/DPO training data from all IRC sections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run: 13 sections, ~100 SFT pairs + 10 DPO pairs
  python3.14 scripts/generate_grounded_data.py

  # Full direct run: all 2113 sections, ~19K SFT + ~2K DPO
  python3.14 scripts/generate_grounded_data.py --all-sections --output data/processed/grounded_sft_full.jsonl

  # Resume an interrupted full run
  python3.14 scripts/generate_grounded_data.py --all-sections --output data/processed/grounded_sft_full.jsonl --resume

  # Prepare batch input file for OpenAI Batch API (50% cheaper, async)
  python3.14 scripts/generate_grounded_data.py --all-sections --prepare-batch --batch-output data/processed/batch_input.jsonl

  # Submit + poll a batch (requires batch input file to exist)
  python3.14 scripts/generate_grounded_data.py --submit-batch --batch-input data/processed/batch_input.jsonl

  # Download completed batch results
  python3.14 scripts/generate_grounded_data.py --download-batch BATCH_ID_HERE --output data/processed/grounded_sft_full.jsonl
""",
    )

    # Mode flags
    parser.add_argument(
        "--all-sections",
        action="store_true",
        help="Process all 2,113 IRC sections (vs. test set of 13)",
    )
    parser.add_argument(
        "--prepare-batch",
        action="store_true",
        help="Write a batch input JSONL file for the OpenAI Batch API instead of calling directly",
    )
    parser.add_argument(
        "--submit-batch",
        action="store_true",
        help="Upload and submit a batch input file to the OpenAI Batch API",
    )
    parser.add_argument(
        "--poll-batch",
        metavar="BATCH_ID",
        help="Poll an existing batch by ID until it completes",
    )
    parser.add_argument(
        "--download-batch",
        metavar="BATCH_ID",
        help="Download results for a completed batch by ID",
    )

    # Output paths
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="SFT output path (default: data/processed/grounded_sft_full.jsonl for --all-sections)",
    )
    parser.add_argument(
        "--dpo-output",
        metavar="PATH",
        help="DPO output path (default: data/processed/grounded_dpo_full.jsonl for --all-sections)",
    )
    parser.add_argument(
        "--batch-output",
        metavar="PATH",
        default=str(BATCH_INPUT_DEFAULT),
        help="Batch input JSONL output path (default: data/processed/batch_input.jsonl)",
    )
    parser.add_argument(
        "--batch-input",
        metavar="PATH",
        default=str(BATCH_INPUT_DEFAULT),
        help="Batch input JSONL file to submit (default: data/processed/batch_input.jsonl)",
    )

    # Behavior flags
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted full run using progress file",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Write output to _v2 files (grounded_sft_test_v2.jsonl / grounded_dpo_test_v2.jsonl)",
    )
    parser.add_argument(
        "--pairs-per-section",
        type=int,
        default=None,
        help=f"Override pairs per section (default: {PAIRS_PER_SECTION_FULL} for full, {PAIRS_PER_SECTION} for test)",
    )

    args = parser.parse_args()

    # ── Resolve output paths ───────────────────────────────────────────────────
    if args.all_sections:
        sft_out = Path(args.output) if args.output else SFT_OUT_FULL
        dpo_out = Path(args.dpo_output) if args.dpo_output else DPO_OUT_FULL
        pairs_per_section = args.pairs_per_section or PAIRS_PER_SECTION_FULL
    else:
        sft_out = SFT_OUT_V2 if args.v2 else SFT_OUT
        dpo_out = DPO_OUT_V2 if args.v2 else DPO_OUT
        pairs_per_section = args.pairs_per_section or PAIRS_PER_SECTION

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found in environment. "
            "Make sure it is exported in ~/.zshrc and the shell is sourced."
        )

    client = OpenAI(api_key=api_key)

    # ── Load IRC data ─────────────────────────────────────────────────────────
    print("Loading IRC sections...")
    all_sections = load_irc_sections(IRC_JSONL)
    print(f"  Loaded {len(all_sections)} IRC sections")

    # ── Dispatch mode ─────────────────────────────────────────────────────────

    if args.prepare_batch:
        # Write batch input file only — no API calls
        batch_output = Path(args.batch_output)
        print(f"\nPreparing batch input file for all {len(all_sections)} sections...")
        print(f"  Pairs per section: {pairs_per_section}")
        count = prepare_batch_file(all_sections, batch_output, n=pairs_per_section)
        size_mb = batch_output.stat().st_size / (1024 * 1024)
        print(f"  Written {count} requests to {batch_output} ({size_mb:.1f} MB)")
        print(f"\nNext step:")
        print(f"  python3.14 scripts/generate_grounded_data.py --submit-batch --batch-input {batch_output}")

    elif args.submit_batch:
        batch_input = Path(args.batch_input)
        if not batch_input.exists():
            raise FileNotFoundError(
                f"Batch input file not found: {batch_input}\n"
                f"Run with --prepare-batch first."
            )
        print(f"\nSubmitting batch from {batch_input}...")
        batch_id = submit_batch(client, batch_input)
        print(f"\nBatch submitted successfully!")
        print(f"  Batch ID: {batch_id}")
        print(f"\nTo check status / download results:")
        print(f"  python3.14 scripts/generate_grounded_data.py --poll-batch {batch_id} --output {sft_out}")

    elif args.poll_batch:
        batch_id = args.poll_batch
        output_file_id = poll_batch(client, batch_id)
        if output_file_id:
            print(f"\nBatch complete! Output file ID: {output_file_id}")
            print(f"\nDownloading and processing results...")
            download_batch_results(client, output_file_id, sft_out, dpo_out, all_sections, 0)
        else:
            print("Batch did not complete successfully.")

    elif args.download_batch:
        batch_id = args.download_batch
        print(f"\nRetrieving batch {batch_id}...")
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            print(f"  [ERROR] Batch status is '{batch.status}', not 'completed'")
            return
        output_file_id = batch.output_file_id
        print(f"  Output file ID: {output_file_id}")
        dpo_target = int(batch.request_counts.completed * pairs_per_section * DPO_FRACTION_FULL)
        download_batch_results(client, output_file_id, sft_out, dpo_out, all_sections, dpo_target)

    elif args.all_sections:
        print(f"\nFull run: processing all {len(all_sections)} IRC sections")
        print(f"  Model: {MODEL}")
        print(f"  Pairs per section: {pairs_per_section}")
        print(f"  Target SFT pairs: ~{len(all_sections) * pairs_per_section:,}")
        print(f"  SFT output: {sft_out}")
        print(f"  DPO output: {dpo_out}")
        print(f"  Resume mode: {args.resume}")
        if args.resume:
            progress = load_progress(PROGRESS_FILE)
            print(f"  Progress file: {PROGRESS_FILE}")
            print(f"  Already completed: {len(progress['completed_sections'])} sections")

        # Rough cost estimate
        # ~1500 input tokens + ~2000 output tokens per section call
        est_input = len(all_sections) * 1500
        est_output = len(all_sections) * pairs_per_section * 220
        est_cost = compute_cost(est_input, est_output)
        print(f"\n  Estimated cost (direct API): ${est_cost:.2f}")
        print(f"  Estimated time: ~{len(all_sections) * (RATE_LIMIT_SLEEP + 2) / 3600:.1f} hours")
        print(f"  (Use --prepare-batch + --submit-batch for 50% cheaper async processing)")
        print()

        run_full_direct(
            client=client,
            all_sections=all_sections,
            sft_out=sft_out,
            dpo_out=dpo_out,
            resume=args.resume,
            pairs_per_section=pairs_per_section,
        )

    else:
        # Original test run
        print(f"\nTest run: {len(TARGET_SECTIONS)} target sections")
        run_test(client, all_sections, sft_out, dpo_out)


if __name__ == "__main__":
    main()
