#!/usr/bin/env python3
"""
Bulk training data generator using OpenAI Batch API.

Reads all available content sources:
  - data/raw/tavily/extracted_content/*.json  (255 IRS pages)
  - data/raw/tavily/search_results/*.json     (216 topic search results)
  - data/processed/irc_sections.jsonl         (IRC sections for additional Q&A)
  - data/reference/inflation_adjusted_amounts.json

Prepares batch files, submits to OpenAI Batch API, polls for completion,
downloads and processes results.

Modes:
  python3 scripts/bulk_generate_training_data.py --prepare
      Load content, build batch JSONL files, report counts — no API calls.

  python3 scripts/bulk_generate_training_data.py --submit
      Upload batch files and submit to OpenAI Batch API.

  python3 scripts/bulk_generate_training_data.py --status
      Check status of all submitted batches.

  python3 scripts/bulk_generate_training_data.py --wait
      Poll every 5 min until all batches complete, then download + process.

  python3 scripts/bulk_generate_training_data.py --download
      Download and process completed batches (idempotent).

  python3 scripts/bulk_generate_training_data.py --all
      Prepare + submit + wait + download in one shot.

  python3 scripts/bulk_generate_training_data.py --prepare --limit 5
      Test: only 5 chunks per source type.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
BATCH_DIR = DATA / "batch"
PROCESSED_DIR = DATA / "processed"
RAW_TAVILY = DATA / "raw" / "tavily"
REFERENCE_DIR = DATA / "reference"

SFT_BATCH_FILE = BATCH_DIR / "sft_batch_requests.jsonl"
DPO_BATCH_FILE = BATCH_DIR / "dpo_batch_requests.jsonl"
INFLATION_BATCH_FILE = BATCH_DIR / "inflation_batch_requests.jsonl"
BATCH_IDS_FILE = BATCH_DIR / "batch_ids.json"

BULK_SFT_OUT = PROCESSED_DIR / "bulk_sft_full.jsonl"
BULK_DPO_OUT = PROCESSED_DIR / "bulk_dpo_full.jsonl"
INFLATION_SFT_OUT = PROCESSED_DIR / "inflation_sft_v2.jsonl"
INFLATION_DPO_OUT = PROCESSED_DIR / "inflation_dpo_v2.jsonl"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
CHUNK_SIZE = 12000     # ~3000 tokens
CHUNK_OVERLAP = 500
MAX_RAW_CONTENT = 24000  # Cap raw_content from search results to 2 chunks max
SFT_PAIRS_PER_CHUNK = 10
DPO_PAIRS_PER_CHUNK = 5
INFLATION_SFT_PER_AMOUNT = 5
INFLATION_DPO_PER_AMOUNT = 3
MIN_ANSWER_LENGTH = 200
POLL_INTERVAL_SECONDS = 300  # 5 minutes
RANDOM_SEED = 42
MAX_TOKENS_SFT = 2048
MAX_TOKENS_DPO = 2560
MAX_TOKENS_INFLATION = 2048

# Batch API pricing (gpt-4o-mini)
BATCH_PRICE_INPUT_PER_M = 0.075   # $0.075/M tokens
BATCH_PRICE_OUTPUT_PER_M = 0.300  # $0.30/M tokens
AVG_INPUT_TOKENS = 800
AVG_OUTPUT_TOKENS_SFT = 1200
AVG_OUTPUT_TOKENS_DPO = 1500
AVG_OUTPUT_TOKENS_INFLATION = 1000

random.seed(RANDOM_SEED)

# ── Prompts ───────────────────────────────────────────────────────────────────
SFT_SYSTEM_PROMPT = (
    "You are a tax law expert specializing in U.S. federal tax law, IRS regulations, "
    "and the Internal Revenue Code. Always cite specific IRC sections, Treasury Regulations, "
    "or IRS publications when answering. Provide accurate, detailed explanations."
)

SFT_GENERATION_PROMPT = """\
You are generating high-quality training data for a U.S. tax law AI assistant.

Below is content from an IRS resource or official tax document. Generate exactly {n_pairs} \
diverse question-and-answer pairs based ONLY on the content provided.

REQUIREMENTS:
1. Cover these question types (distribute evenly): factual recall, procedural, eligibility, \
calculation, comparison, edge cases
2. Every answer MUST cite specific IRC sections (e.g., IRC § 401(k)), Treasury Regulations \
(e.g., Treas. Reg. § 1.401(k)-1), or IRS Publications (e.g., IRS Pub. 590-A)
3. Every answer mentioning dollar amounts MUST specify the tax year (e.g., "For tax year 2024...")
4. Every answer MUST be at least 200 characters
5. Every answer MUST end with: "Consult a qualified tax professional for advice specific to \
your situation."
6. Questions must be specific and practical — not generic or vague
7. Do NOT make up information not supported by the content below

SOURCE CONTENT:
{content}

Respond ONLY with valid JSON in this exact format:
{{
  "pairs": [
    {{
      "question": "...",
      "answer": "...",
      "source_section": "IRC § XXX or Pub. XXX or Treas. Reg. § X.XXX",
      "question_type": "factual|procedural|eligibility|calculation|comparison|edge_case"
    }}
  ]
}}"""

DPO_GENERATION_PROMPT = """\
You are generating preference training data for a U.S. tax law AI assistant.

Below is content from an IRS resource. Generate exactly {n_pairs} preference pairs based \
on this content.

Each pair needs:
- question: A realistic tax question
- chosen: A high-quality answer with SPECIFIC citations (IRC §, Treas. Reg. §, or IRS Pub.), \
dollar amounts with tax years, and key exceptions/limitations. Must end with professional \
disclaimer.
- rejected: A FLAWED version of the answer. Rotate through these error types:
  {error_types_list}
- error_type: Which error the rejected answer contains

CRITICAL: The rejected answer must look plausible but contain a clear, specific error. \
Do not make rejected answers obviously wrong — make them subtly incorrect.

SOURCE CONTENT:
{content}

Respond ONLY with valid JSON:
{{
  "pairs": [
    {{
      "question": "...",
      "chosen": "...",
      "rejected": "...",
      "error_type": "wrong_dollar_amount|wrong_section_citation|vague_non_answer|outdated_pre_tcja|missing_exception"
    }}
  ]
}}"""

DPO_ERROR_TYPES = [
    "wrong_dollar_amount (e.g., $135,000 instead of $14,600 — subtle but wrong)",
    "wrong_section_citation (e.g., cites IRC § 401 instead of § 408)",
    "vague_non_answer (says 'consult a professional' without any substantive information)",
    "outdated_pre_tcja (presents pre-2018 TCJA rules as current law, e.g., old standard deductions)",
    "missing_exception (omits a critical limitation, phase-out, or special rule)",
]

INFLATION_SFT_PROMPT = """\
You are generating training data to teach a tax AI assistant current-year U.S. tax figures.

The following is a specific U.S. federal tax amount for {tax_year}:
  Category: {category}
  Subcategory: {subcategory}
  Amount: {amount_display}
  Source: {source}

Generate exactly {n_pairs} diverse questions and answers about this specific figure, \
from different angles (e.g., "What is the limit?", "How does this compare to prior year?", \
"Who qualifies?", "What if I'm over the limit?").

REQUIREMENTS:
- Every answer must state the tax year explicitly: "For tax year {tax_year}..."
- Every answer must cite the authoritative source (Revenue Procedure, IRS Notice, or IRC section)
- Every answer must be at least 200 characters
- End every answer with: "Consult a qualified tax professional for advice specific to your situation."
- Make questions varied in phrasing and angle

Respond ONLY with valid JSON:
{{
  "pairs": [
    {{
      "question": "...",
      "answer": "...",
      "question_type": "factual|procedural|eligibility|calculation|comparison|edge_case"
    }}
  ]
}}"""

INFLATION_DPO_PROMPT = """\
You are generating preference data to correct common mistakes in tax AI assistants.

The CORRECT figure is:
  Category: {category}
  Subcategory: {subcategory}
  Tax Year: {tax_year}
  Correct Amount: {amount_display}
  Source: {source}

Generate exactly {n_pairs} preference pairs where the rejected answer uses a common \
WRONG amount for this figure.

For each pair:
- chosen: correct answer citing the right amount ({amount_display}) and tax year {tax_year}
- rejected: plausible-looking answer with a wrong amount (use prior-year figures or \
common misconceptions — make it subtly wrong, not obviously absurd)

Respond ONLY with valid JSON:
{{
  "pairs": [
    {{
      "question": "...",
      "chosen": "...",
      "rejected": "...",
      "error_type": "wrong_dollar_amount"
    }}
  ]
}}"""

# ── Content loading ───────────────────────────────────────────────────────────

def load_extracted_pages(limit: int | None = None) -> list[dict]:
    """Load all extracted IRS pages from data/raw/tavily/extracted_content/."""
    pages = []
    content_dir = RAW_TAVILY / "extracted_content"
    files = sorted(content_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            content = d.get("content", "")
            if content and len(content) > 200:
                pages.append({
                    "text": content,
                    "url": d.get("url", ""),
                    "title": d.get("title", f.stem),
                    "source_type": "irs_extracted",
                    "source_file": f.name,
                })
        except (json.JSONDecodeError, OSError):
            pass
    print(f"  Loaded {len(pages)} extracted IRS pages")
    return pages


def load_search_results(limit: int | None = None) -> list[dict]:
    """Load all search result files from data/raw/tavily/search_results/."""
    results = []
    search_dir = RAW_TAVILY / "search_results"
    files = sorted(search_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            topic = d.get("topic", f.stem)
            for result in d.get("results", []):
                # Use raw_content if available (longer), else content
                # Cap raw_content to avoid memory/chunk explosion (some files are 190K+ chars)
                text = result.get("raw_content") or result.get("content", "")
                text = text[:MAX_RAW_CONTENT]
                if text and len(text) > 200:
                    results.append({
                        "text": text,
                        "url": result.get("url", ""),
                        "title": result.get("title", topic),
                        "topic": topic,
                        "source_type": "search_result",
                        "source_file": f.name,
                    })
        except (json.JSONDecodeError, OSError):
            pass
    print(f"  Loaded {len(results)} search result documents")
    return results


def load_irc_sections(limit: int | None = None) -> list[dict]:
    """Load IRC sections from data/processed/irc_sections.jsonl."""
    sections = []
    irc_path = PROCESSED_DIR / "irc_sections.jsonl"
    if not irc_path.exists():
        print("  [WARN] irc_sections.jsonl not found, skipping")
        return sections
    with open(irc_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                text = d.get("text", "")
                if text and len(text) > 200:
                    sections.append({
                        "text": text,
                        "section": d.get("section", ""),
                        "heading": d.get("heading", ""),
                        "source_type": "irc_section",
                        "source_file": "irc_sections.jsonl",
                    })
            except json.JSONDecodeError:
                pass
    print(f"  Loaded {len(sections)} IRC sections")
    return sections


def load_inflation_amounts() -> list[dict]:
    """
    Load all dollar amounts from inflation_adjusted_amounts.json.
    Returns list of dicts with tax_year, category, subcategory, amount, source.
    """
    inflation_path = REFERENCE_DIR / "inflation_adjusted_amounts.json"
    if not inflation_path.exists():
        print("  [WARN] inflation_adjusted_amounts.json not found, skipping")
        return []

    with open(inflation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources_list = data.get("metadata", {}).get("sources", data.get("sources", []))

    amounts = []

    def walk_amounts(obj: Any, tax_year: str, category: str, subcategory: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk_amounts(v, tax_year, category, f"{subcategory}.{k}" if subcategory else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk_amounts(v, tax_year, category, f"{subcategory}[{i}]")
        elif isinstance(obj, (int, float)) and obj > 100:
            # It's a dollar amount worth generating training data for
            amounts.append({
                "tax_year": tax_year,
                "category": category,
                "subcategory": subcategory,
                "amount": obj,
                "amount_display": f"${obj:,.0f}",
                "source": _get_source_for_category(category, tax_year, sources_list),
            })

    for year, categories in data.get("tax_years", {}).items():
        for cat_name, cat_data in categories.items():
            walk_amounts(cat_data, year, cat_name, "")

    print(f"  Loaded {len(amounts)} inflation-adjusted dollar amounts across all tax years")
    return amounts


def _get_source_for_category(category: str, tax_year: str, sources_list: list) -> str:
    """Return the most relevant source string for a given category."""
    year_suffix = tax_year[-2:]  # e.g. "24" from "2024"
    # Match sources containing the year
    for s in sources_list:
        if f"TY{tax_year}" in s or f"TY20{year_suffix}" in s:
            if "retirement" in category.lower() and "Notice" in s:
                return s
            if "hsa" in category.lower() and "HSA" in s:
                return s
    # Fall back to first source with the year
    for s in sources_list:
        if tax_year in s or f"20{year_suffix}" in s:
            return s
    return f"IRS Revenue Procedure (TY{tax_year})"


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at paragraph boundary
        if end < len(text):
            break_pos = text.rfind("\n\n", start, end)
            if break_pos == -1 or break_pos <= start:
                break_pos = text.rfind("\n", start, end)
            if break_pos == -1 or break_pos <= start:
                break_pos = text.rfind(". ", start, end)
            if break_pos > start:
                end = break_pos + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= len(text) - overlap:
            break
    return [c for c in chunks if c]


# ── Batch request builders ────────────────────────────────────────────────────

def make_sft_request(custom_id: str, content: str, n_pairs: int = SFT_PAIRS_PER_CHUNK) -> dict:
    """Build a single OpenAI Batch API request for SFT generation."""
    prompt = SFT_GENERATION_PROMPT.format(content=content[:CHUNK_SIZE], n_pairs=n_pairs)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS_SFT,
            "temperature": 0.7,
        },
    }


def make_dpo_request(custom_id: str, content: str, n_pairs: int = DPO_PAIRS_PER_CHUNK) -> dict:
    """Build a single OpenAI Batch API request for DPO generation."""
    error_types_list = "\n  ".join(f"- {e}" for e in DPO_ERROR_TYPES)
    prompt = DPO_GENERATION_PROMPT.format(
        content=content[:CHUNK_SIZE],
        n_pairs=n_pairs,
        error_types_list=error_types_list,
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS_DPO,
            "temperature": 0.7,
        },
    }


def make_inflation_sft_request(custom_id: str, amount_info: dict) -> dict:
    """Build a batch request for inflation-focused SFT generation."""
    prompt = INFLATION_SFT_PROMPT.format(
        tax_year=amount_info["tax_year"],
        category=amount_info["category"].replace("_", " ").title(),
        subcategory=amount_info["subcategory"].replace("_", " ").replace(".", " > "),
        amount_display=amount_info["amount_display"],
        source=amount_info["source"],
        n_pairs=INFLATION_SFT_PER_AMOUNT,
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS_INFLATION,
            "temperature": 0.6,
        },
    }


def make_inflation_dpo_request(custom_id: str, amount_info: dict) -> dict:
    """Build a batch request for inflation-focused DPO generation."""
    prompt = INFLATION_DPO_PROMPT.format(
        tax_year=amount_info["tax_year"],
        category=amount_info["category"].replace("_", " ").title(),
        subcategory=amount_info["subcategory"].replace("_", " ").replace(".", " > "),
        amount_display=amount_info["amount_display"],
        source=amount_info["source"],
        n_pairs=INFLATION_DPO_PER_AMOUNT,
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS_INFLATION,
            "temperature": 0.6,
        },
    }


# ── Batch file preparation ────────────────────────────────────────────────────

def prepare_batch_files(limit: int | None = None) -> dict:
    """
    Load all content, chunk it, build batch JSONL files.
    Returns statistics dict.
    """
    print("\n" + "=" * 60)
    print("PREPARING BATCH FILES")
    print("=" * 60)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load content ─────────────────────────────────────────────────────────
    print("\nLoading content sources...")
    extracted = load_extracted_pages(limit=limit)
    search = load_search_results(limit=limit)
    irc = load_irc_sections(limit=limit)
    inflation_amounts = load_inflation_amounts()

    # ── Build SFT batch requests ──────────────────────────────────────────────
    print("\nBuilding SFT batch requests...")
    sft_requests = []

    def add_sft_chunks(docs: list[dict], prefix: str) -> int:
        total_chunks = 0
        for i, doc in enumerate(docs):
            chunks = chunk_text(doc["text"])
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) < 200:
                    continue
                cid = f"sft-{prefix}-{i}-c{j}"
                req = make_sft_request(cid, chunk)
                # Embed source metadata in custom_id-accessible way via a metadata field
                req["_meta"] = {
                    "source_type": doc.get("source_type", prefix),
                    "source_file": doc.get("source_file", ""),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "chunk_index": j,
                }
                sft_requests.append(req)
                total_chunks += 1
        return total_chunks

    ext_chunks = add_sft_chunks(extracted, "ext")
    srch_chunks = add_sft_chunks(search, "srch")
    irc_chunks = add_sft_chunks(irc, "irc")

    print(f"  SFT: {ext_chunks} chunks from extracted, {srch_chunks} from search, {irc_chunks} from IRC")
    print(f"  SFT total requests: {len(sft_requests):,}")

    # Write SFT batch file (strip _meta — not part of API spec)
    with open(SFT_BATCH_FILE, "w", encoding="utf-8") as f:
        for req in sft_requests:
            api_req = {k: v for k, v in req.items() if k != "_meta"}
            f.write(json.dumps(api_req, ensure_ascii=False) + "\n")
    print(f"  Written: {SFT_BATCH_FILE}")

    # ── Write SFT metadata sidecar ────────────────────────────────────────────
    sft_meta_file = BATCH_DIR / "sft_batch_meta.jsonl"
    with open(sft_meta_file, "w", encoding="utf-8") as f:
        for req in sft_requests:
            entry = {"custom_id": req["custom_id"], **(req.get("_meta") or {})}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Build DPO batch requests ──────────────────────────────────────────────
    print("\nBuilding DPO batch requests...")
    dpo_requests = []

    def add_dpo_chunks(docs: list[dict], prefix: str) -> int:
        total_chunks = 0
        for i, doc in enumerate(docs):
            chunks = chunk_text(doc["text"])
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) < 300:
                    continue
                cid = f"dpo-{prefix}-{i}-c{j}"
                req = make_dpo_request(cid, chunk)
                req["_meta"] = {
                    "source_type": doc.get("source_type", prefix),
                    "source_file": doc.get("source_file", ""),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "chunk_index": j,
                }
                dpo_requests.append(req)
                total_chunks += 1
        return total_chunks

    dpo_ext = add_dpo_chunks(extracted, "ext")
    dpo_srch = add_dpo_chunks(search, "srch")
    print(f"  DPO: {dpo_ext} chunks from extracted, {dpo_srch} from search")
    print(f"  DPO total requests: {len(dpo_requests):,}")

    with open(DPO_BATCH_FILE, "w", encoding="utf-8") as f:
        for req in dpo_requests:
            api_req = {k: v for k, v in req.items() if k != "_meta"}
            f.write(json.dumps(api_req, ensure_ascii=False) + "\n")
    print(f"  Written: {DPO_BATCH_FILE}")

    dpo_meta_file = BATCH_DIR / "dpo_batch_meta.jsonl"
    with open(dpo_meta_file, "w", encoding="utf-8") as f:
        for req in dpo_requests:
            entry = {"custom_id": req["custom_id"], **(req.get("_meta") or {})}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Build inflation batch requests ────────────────────────────────────────
    print("\nBuilding inflation batch requests...")
    inflation_requests = []
    inflation_amounts_used = inflation_amounts
    if limit:
        inflation_amounts_used = inflation_amounts[:limit]

    for i, amt in enumerate(inflation_amounts_used):
        safe_cat = re.sub(r"[^a-zA-Z0-9_]", "_", amt["category"])
        safe_sub = re.sub(r"[^a-zA-Z0-9_]", "_", amt["subcategory"])[:30]
        yr = amt["tax_year"]

        # SFT
        sft_cid = f"inf-sft-{yr}-{safe_cat}-{safe_sub}-{i}"
        inflation_requests.append({
            "type": "sft",
            **make_inflation_sft_request(sft_cid, amt),
            "_meta": {**amt, "request_type": "inflation_sft"},
        })

        # DPO
        dpo_cid = f"inf-dpo-{yr}-{safe_cat}-{safe_sub}-{i}"
        inflation_requests.append({
            "type": "dpo",
            **make_inflation_dpo_request(dpo_cid, amt),
            "_meta": {**amt, "request_type": "inflation_dpo"},
        })

    print(f"  Inflation requests: {len(inflation_requests):,} ({len(inflation_amounts_used)} amounts × 2)")

    with open(INFLATION_BATCH_FILE, "w", encoding="utf-8") as f:
        for req in inflation_requests:
            api_req = {k: v for k, v in req.items() if k not in ("_meta", "type")}
            f.write(json.dumps(api_req, ensure_ascii=False) + "\n")
    print(f"  Written: {INFLATION_BATCH_FILE}")

    inflation_meta_file = BATCH_DIR / "inflation_batch_meta.jsonl"
    with open(inflation_meta_file, "w", encoding="utf-8") as f:
        for req in inflation_requests:
            entry = {"custom_id": req["custom_id"], "type": req["type"], **(req.get("_meta") or {})}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Cost estimate ─────────────────────────────────────────────────────────
    total_sft = len(sft_requests)
    total_dpo = len(dpo_requests)
    total_inf = len(inflation_requests)
    total_reqs = total_sft + total_dpo + total_inf

    est_input_tokens = total_reqs * AVG_INPUT_TOKENS
    est_sft_output = (total_sft + total_inf // 2) * AVG_OUTPUT_TOKENS_SFT
    est_dpo_output = (total_dpo + total_inf // 2) * AVG_OUTPUT_TOKENS_DPO
    est_output_tokens = est_sft_output + est_dpo_output

    est_input_cost = est_input_tokens / 1_000_000 * BATCH_PRICE_INPUT_PER_M
    est_output_cost = est_output_tokens / 1_000_000 * BATCH_PRICE_OUTPUT_PER_M
    est_total_cost = est_input_cost + est_output_cost

    est_sft_pairs = total_sft * SFT_PAIRS_PER_CHUNK
    est_dpo_pairs = total_dpo * DPO_PAIRS_PER_CHUNK
    est_inf_sft_pairs = (total_inf // 2) * INFLATION_SFT_PER_AMOUNT
    est_inf_dpo_pairs = (total_inf // 2) * INFLATION_DPO_PER_AMOUNT

    stats = {
        "sft_requests": total_sft,
        "dpo_requests": total_dpo,
        "inflation_requests": total_inf,
        "total_requests": total_reqs,
        "estimated_sft_pairs": est_sft_pairs,
        "estimated_dpo_pairs": est_dpo_pairs,
        "estimated_inflation_sft_pairs": est_inf_sft_pairs,
        "estimated_inflation_dpo_pairs": est_inf_dpo_pairs,
        "estimated_total_pairs": est_sft_pairs + est_dpo_pairs + est_inf_sft_pairs + est_inf_dpo_pairs,
        "estimated_cost_usd": round(est_total_cost, 2),
    }

    print("\n" + "=" * 60)
    print("BATCH PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  SFT requests:        {total_sft:,}")
    print(f"  DPO requests:        {total_dpo:,}")
    print(f"  Inflation requests:  {total_inf:,}")
    print(f"  TOTAL requests:      {total_reqs:,}")
    print(f"\n  Estimated pairs:")
    print(f"    SFT:               ~{est_sft_pairs:,}")
    print(f"    DPO:               ~{est_dpo_pairs:,}")
    print(f"    Inflation SFT:     ~{est_inf_sft_pairs:,}")
    print(f"    Inflation DPO:     ~{est_inf_dpo_pairs:,}")
    print(f"    TOTAL:             ~{est_sft_pairs + est_dpo_pairs + est_inf_sft_pairs + est_inf_dpo_pairs:,}")
    print(f"\n  Estimated cost:      ~${est_total_cost:.2f} (Batch API, 50% off)")
    print("=" * 60)

    return stats


# ── Batch submission ──────────────────────────────────────────────────────────

def submit_batches(client: OpenAI) -> dict:
    """Upload batch files and submit to OpenAI Batch API. Returns batch ID dict."""
    print("\n" + "=" * 60)
    print("SUBMITTING BATCHES TO OPENAI")
    print("=" * 60)

    batch_map = {
        "sft": SFT_BATCH_FILE,
        "dpo": DPO_BATCH_FILE,
        "inflation": INFLATION_BATCH_FILE,
    }

    batch_ids = {}
    for name, batch_file in batch_map.items():
        if not batch_file.exists():
            print(f"  [SKIP] {name}: file not found ({batch_file})")
            continue

        file_size_mb = batch_file.stat().st_size / 1_048_576
        line_count = sum(1 for _ in open(batch_file, encoding="utf-8"))
        print(f"\n  Uploading {name} ({file_size_mb:.1f} MB, {line_count:,} requests)...")

        with open(batch_file, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
        print(f"    File ID: {uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_ids[name] = {
            "batch_id": batch.id,
            "file_id": uploaded.id,
            "status": batch.status,
            "request_count": line_count,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(f"    Batch ID: {batch.id}")
        print(f"    Status:   {batch.status}")

    # Save batch IDs
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(BATCH_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(batch_ids, f, indent=2)
    print(f"\n  Saved batch IDs to: {BATCH_IDS_FILE}")

    print("\n" + "=" * 60)
    print("SUBMISSION SUMMARY")
    print("=" * 60)
    for name, info in batch_ids.items():
        print(f"  {name}: {info['batch_id']} ({info['request_count']:,} requests)")

    return batch_ids


# ── Status checking ───────────────────────────────────────────────────────────

def load_batch_ids() -> dict:
    """Load batch IDs from saved file."""
    if not BATCH_IDS_FILE.exists():
        print(f"[ERROR] {BATCH_IDS_FILE} not found. Run --submit first.")
        sys.exit(1)
    with open(BATCH_IDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def check_batch_status(client: OpenAI, batch_ids: dict) -> dict:
    """Check status of all batches. Returns dict of name -> status info."""
    statuses = {}
    for name, info in batch_ids.items():
        bid = info["batch_id"]
        try:
            batch = client.batches.retrieve(bid)
            counts = batch.request_counts
            statuses[name] = {
                "batch_id": bid,
                "status": batch.status,
                "completed": counts.completed if counts else 0,
                "failed": counts.failed if counts else 0,
                "total": counts.total if counts else 0,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }
        except Exception as e:
            statuses[name] = {"batch_id": bid, "status": "error", "error": str(e)}
    return statuses


def print_batch_statuses(statuses: dict) -> None:
    for name, s in statuses.items():
        total = s.get("total") or 1
        completed = s.get("completed", 0)
        pct = completed / total * 100 if total else 0
        print(
            f"  {name:12s}: {s['status']:15s} | "
            f"{completed:>6,}/{s.get('total', 0):>6,} ({pct:5.1f}%)"
            + (f" | output={s['output_file_id']}" if s.get("output_file_id") else "")
        )


def all_complete(statuses: dict) -> bool:
    terminal = {"completed", "failed", "expired", "cancelled"}
    return all(s.get("status") in terminal for s in statuses.values())


# ── Result download and processing ───────────────────────────────────────────

def load_meta_sidecar(meta_file: Path) -> dict:
    """Load metadata sidecar into dict keyed by custom_id."""
    meta = {}
    if not meta_file.exists():
        return meta
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    meta[d["custom_id"]] = d
                except (json.JSONDecodeError, KeyError):
                    pass
    return meta


def parse_pairs_from_response(raw_content: str) -> list[dict]:
    """Parse GPT response JSON into list of pair dicts."""
    if not raw_content:
        return []
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        m = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list) and v:
                return v
    return []


def has_citation(text: str) -> bool:
    """Check whether text contains an IRC / Reg / Pub citation."""
    patterns = [
        r'IRC\s*§',
        r'I\.R\.C\.\s*§',
        r'Section\s+\d{2,4}',
        r'§\s*\d{2,4}',
        r'Treas\.?\s*Reg\.?',
        r'Revenue\s+Procedure',
        r'Rev\.\s*Proc\.',
        r'IRS\s+Pub(lication)?\.?\s*\d',
        r'Notice\s+\d{4}-\d+',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def make_sft_record(question: str, answer: str, meta: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source_type": meta.get("source_type", "bulk"),
            "source_file": meta.get("source_file", ""),
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "question_type": meta.get("question_type", ""),
            "source_section": meta.get("source_section", ""),
            "grounded": True,
        },
    }


def make_dpo_record(question: str, chosen: str, rejected: str, error_type: str, meta: dict) -> dict:
    return {
        "prompt": question,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "source_type": meta.get("source_type", "bulk"),
            "source_file": meta.get("source_file", ""),
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "error_type": error_type,
            "grounded": True,
        },
    }


def process_sft_results(client: OpenAI, output_file_id: str, out_path: Path, meta_dict: dict) -> tuple[int, int, int]:
    """Download and process SFT batch results. Returns (written, discarded, errors)."""
    print(f"\n  Downloading SFT results (file_id={output_file_id})...")
    content = client.files.content(output_file_id)
    lines = content.text.strip().split("\n")
    print(f"  Got {len(lines):,} result lines")

    records = []
    discarded = 0
    errors = 0
    seen_questions: set[str] = set()

    for line in lines:
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue

        if result.get("error"):
            errors += 1
            continue

        custom_id = result.get("custom_id", "")
        choices = result.get("response", {}).get("body", {}).get("choices", [])
        if not choices:
            errors += 1
            continue

        raw = choices[0].get("message", {}).get("content", "")
        pairs = parse_pairs_from_response(raw)
        src_meta = meta_dict.get(custom_id, {})

        for p in pairs:
            q = str(p.get("question", "")).strip()
            a = str(p.get("answer", "")).strip()

            if not q or not a:
                discarded += 1
                continue
            if len(a) < MIN_ANSWER_LENGTH:
                discarded += 1
                continue
            if not has_citation(a):
                discarded += 1
                continue
            q_lower = q.lower()
            if q_lower in seen_questions:
                discarded += 1
                continue
            seen_questions.add(q_lower)

            pair_meta = {
                **src_meta,
                "question_type": p.get("question_type", ""),
                "source_section": p.get("source_section", ""),
            }
            records.append(make_sft_record(q, a, pair_meta))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Written {len(records):,} SFT records to {out_path}")
    print(f"  Discarded: {discarded} | Errors: {errors}")
    return len(records), discarded, errors


def process_dpo_results(client: OpenAI, output_file_id: str, out_path: Path, meta_dict: dict) -> tuple[int, int, int]:
    """Download and process DPO batch results. Returns (written, discarded, errors)."""
    print(f"\n  Downloading DPO results (file_id={output_file_id})...")
    content = client.files.content(output_file_id)
    lines = content.text.strip().split("\n")
    print(f"  Got {len(lines):,} result lines")

    records = []
    discarded = 0
    errors = 0
    seen_questions: set[str] = set()

    for line in lines:
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue

        if result.get("error"):
            errors += 1
            continue

        custom_id = result.get("custom_id", "")
        choices = result.get("response", {}).get("body", {}).get("choices", [])
        if not choices:
            errors += 1
            continue

        raw = choices[0].get("message", {}).get("content", "")
        pairs = parse_pairs_from_response(raw)
        src_meta = meta_dict.get(custom_id, {})

        for p in pairs:
            q = str(p.get("question", "")).strip()
            chosen = str(p.get("chosen", "")).strip()
            rejected = str(p.get("rejected", "")).strip()
            error_type = str(p.get("error_type", "unknown")).strip()

            if not q or not chosen or not rejected:
                discarded += 1
                continue
            if len(chosen) < MIN_ANSWER_LENGTH or len(rejected) < MIN_ANSWER_LENGTH:
                discarded += 1
                continue
            if not has_citation(chosen):
                discarded += 1
                continue
            q_lower = q.lower()
            if q_lower in seen_questions:
                discarded += 1
                continue
            seen_questions.add(q_lower)

            records.append(make_dpo_record(q, chosen, rejected, error_type, src_meta))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Written {len(records):,} DPO records to {out_path}")
    print(f"  Discarded: {discarded} | Errors: {errors}")
    return len(records), discarded, errors


def process_inflation_results(
    client: OpenAI,
    output_file_id: str,
    sft_out: Path,
    dpo_out: Path,
    meta_dict: dict,
) -> tuple[int, int, int, int]:
    """Download and process inflation batch results. Returns (sft_written, dpo_written, discarded, errors)."""
    print(f"\n  Downloading inflation results (file_id={output_file_id})...")
    content = client.files.content(output_file_id)
    lines = content.text.strip().split("\n")
    print(f"  Got {len(lines):,} result lines")

    sft_records = []
    dpo_records = []
    discarded = 0
    errors = 0
    seen_questions: set[str] = set()

    for line in lines:
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue

        if result.get("error"):
            errors += 1
            continue

        custom_id = result.get("custom_id", "")
        choices = result.get("response", {}).get("body", {}).get("choices", [])
        if not choices:
            errors += 1
            continue

        raw = choices[0].get("message", {}).get("content", "")
        pairs = parse_pairs_from_response(raw)
        src_meta = meta_dict.get(custom_id, {})
        request_type = src_meta.get("request_type", "inflation_sft")

        for p in pairs:
            q = str(p.get("question", "")).strip()

            if request_type == "inflation_sft":
                a = str(p.get("answer", "")).strip()
                if not q or not a or len(a) < MIN_ANSWER_LENGTH:
                    discarded += 1
                    continue
                q_lower = q.lower()
                if q_lower in seen_questions:
                    discarded += 1
                    continue
                seen_questions.add(q_lower)
                pair_meta = {
                    **src_meta,
                    "source_type": "inflation",
                    "question_type": p.get("question_type", ""),
                }
                sft_records.append(make_sft_record(q, a, pair_meta))

            else:  # inflation_dpo
                chosen = str(p.get("chosen", "")).strip()
                rejected = str(p.get("rejected", "")).strip()
                if not q or not chosen or not rejected:
                    discarded += 1
                    continue
                if len(chosen) < MIN_ANSWER_LENGTH or len(rejected) < MIN_ANSWER_LENGTH:
                    discarded += 1
                    continue
                q_lower = q.lower()
                if q_lower in seen_questions:
                    discarded += 1
                    continue
                seen_questions.add(q_lower)
                dpo_records.append(make_dpo_record(q, chosen, rejected, "wrong_dollar_amount", src_meta))

    sft_out.parent.mkdir(parents=True, exist_ok=True)
    dpo_out.parent.mkdir(parents=True, exist_ok=True)

    with open(sft_out, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(dpo_out, "w", encoding="utf-8") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Inflation SFT: {len(sft_records):,} records -> {sft_out}")
    print(f"  Inflation DPO: {len(dpo_records):,} records -> {dpo_out}")
    print(f"  Discarded: {discarded} | Errors: {errors}")
    return len(sft_records), len(dpo_records), discarded, errors


def download_and_process_all(client: OpenAI) -> None:
    """Download and process results for all completed batches."""
    print("\n" + "=" * 60)
    print("DOWNLOADING AND PROCESSING RESULTS")
    print("=" * 60)

    batch_ids = load_batch_ids()
    statuses = check_batch_status(client, batch_ids)

    sft_meta = load_meta_sidecar(BATCH_DIR / "sft_batch_meta.jsonl")
    dpo_meta = load_meta_sidecar(BATCH_DIR / "dpo_batch_meta.jsonl")
    inflation_meta = load_meta_sidecar(BATCH_DIR / "inflation_batch_meta.jsonl")

    total_results = {}

    for name, s in statuses.items():
        if s["status"] != "completed":
            print(f"  [SKIP] {name}: status={s['status']} (not completed)")
            continue
        if not s.get("output_file_id"):
            print(f"  [SKIP] {name}: no output file")
            continue

        if name == "sft":
            written, disc, errs = process_sft_results(
                client, s["output_file_id"], BULK_SFT_OUT, sft_meta
            )
            total_results["sft"] = {"written": written, "discarded": disc, "errors": errs}

        elif name == "dpo":
            written, disc, errs = process_dpo_results(
                client, s["output_file_id"], BULK_DPO_OUT, dpo_meta
            )
            total_results["dpo"] = {"written": written, "discarded": disc, "errors": errs}

        elif name == "inflation":
            sft_w, dpo_w, disc, errs = process_inflation_results(
                client, s["output_file_id"], INFLATION_SFT_OUT, INFLATION_DPO_OUT, inflation_meta
            )
            total_results["inflation_sft"] = {"written": sft_w}
            total_results["inflation_dpo"] = {"written": dpo_w, "discarded": disc, "errors": errs}

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    grand_total = 0
    for name, r in total_results.items():
        w = r.get("written", 0)
        grand_total += w
        print(f"  {name:20s}: {w:>8,} pairs written")
    print(f"  {'TOTAL':20s}: {grand_total:>8,} pairs")
    print("=" * 60)
    print("\nOutput files:")
    for p in [BULK_SFT_OUT, BULK_DPO_OUT, INFLATION_SFT_OUT, INFLATION_DPO_OUT]:
        if p.exists():
            lines = sum(1 for _ in open(p))
            size_mb = p.stat().st_size / 1_048_576
            print(f"  {p}: {lines:,} lines ({size_mb:.1f} MB)")


# ── Polling ───────────────────────────────────────────────────────────────────

def wait_for_completion(client: OpenAI) -> None:
    """Poll every 5 minutes until all batches complete, then process."""
    batch_ids = load_batch_ids()
    print(f"\nMonitoring {len(batch_ids)} batch(es). Polling every {POLL_INTERVAL_SECONDS}s...")

    while True:
        statuses = check_batch_status(client, batch_ids)
        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}]")
        print_batch_statuses(statuses)

        if all_complete(statuses):
            print("\nAll batches have reached terminal state.")
            download_and_process_all(client)
            break

        # Check for failures
        for name, s in statuses.items():
            if s.get("status") in ("failed", "expired", "cancelled"):
                print(f"  [WARN] {name} ended with status: {s['status']}")

        time.sleep(POLL_INTERVAL_SECONDS)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Bulk training data generator via OpenAI Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test batch file generation with 5 chunks per source
  python3 scripts/bulk_generate_training_data.py --prepare --limit 5

  # Full preparation (no API calls)
  python3 scripts/bulk_generate_training_data.py --prepare

  # Submit to Batch API
  python3 scripts/bulk_generate_training_data.py --submit

  # Check status
  python3 scripts/bulk_generate_training_data.py --status

  # Wait for completion and download
  python3 scripts/bulk_generate_training_data.py --wait

  # Download already-completed batches
  python3 scripts/bulk_generate_training_data.py --download

  # Full pipeline end-to-end
  python3 scripts/bulk_generate_training_data.py --all
""",
    )
    parser.add_argument("--prepare", action="store_true", help="Prepare batch JSONL files")
    parser.add_argument("--submit", action="store_true", help="Submit batches to OpenAI")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--wait", action="store_true", help="Poll until done, then process")
    parser.add_argument("--download", action="store_true", help="Download and process completed batches")
    parser.add_argument("--all", action="store_true", help="Full pipeline: prepare + submit + wait + download")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit docs/chunks per source (for testing)")

    args = parser.parse_args()

    if not any([args.prepare, args.submit, args.status, args.wait, args.download, args.all]):
        parser.print_help()
        return

    # Modes that don't need OpenAI client
    if args.prepare or args.all:
        prepare_batch_files(limit=args.limit)
        if args.prepare and not args.all:
            return

    # Modes that need OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if args.submit or args.all:
        submit_batches(client)
        if args.submit and not args.all:
            return

    if args.status:
        batch_ids = load_batch_ids()
        statuses = check_batch_status(client, batch_ids)
        print("\nBatch status:")
        print_batch_statuses(statuses)
        return

    if args.wait or args.all:
        wait_for_completion(client)
        return

    if args.download:
        download_and_process_all(client)
        return


if __name__ == "__main__":
    main()
