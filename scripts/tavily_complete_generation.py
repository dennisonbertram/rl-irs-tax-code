#!/usr/bin/env python3
"""
Resume/complete Tavily training data generation.

What this script does:
  1. Identifies extracted content files not yet processed for SFT/DPO
  2. Generates SFT pairs from remaining extracted files
  3. Generates DPO pairs from ALL extracted + search content
  4. Consolidates into:
       data/processed/tavily_sft_full.jsonl  (appended)
       data/processed/tavily_dpo_full.jsonl  (new full file)

Run:
  python scripts/tavily_complete_generation.py
  python scripts/tavily_complete_generation.py --test       # 5 files only
  python scripts/tavily_complete_generation.py --skip-search  # extracted only
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ── Env & paths ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TAVILY_DIR = BASE_DIR / "data" / "raw" / "tavily"
EXTRACTED_DIR = TAVILY_DIR / "extracted_content"
SEARCH_DIR = TAVILY_DIR / "search_results"
TRAINING_DIR = TAVILY_DIR / "training_pairs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

for d in [TRAINING_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tavily_complete")

# ── Prompts (copied from harvester for self-containment) ──────────────────────

GENERATION_SYSTEM_PROMPT = (
    "You are a tax law expert specializing in the Internal Revenue Code (IRC). "
    "Always cite specific IRC sections and subsections when answering questions. "
    "Provide accurate, detailed explanations grounded strictly in the statutory text "
    "and official IRS guidance."
)

QA_GENERATION_PROMPT = """\
Below is official IRS content extracted from: {source_url}

--- BEGIN IRS CONTENT ---
{content}
--- END IRS CONTENT ---

Generate {n_pairs} diverse question-answer pairs that a tax professional or taxpayer might ask.
Include a mix of these question types:
- Factual recall (e.g., "What is the X limit for 2024?")
- Procedural (e.g., "How do you file for X?")
- Eligibility (e.g., "Who qualifies for X?")
- Calculation (e.g., "How is X calculated?")
- Comparison (e.g., "What is the difference between X and Y?")
- Edge cases (e.g., "What happens if X?")

Requirements:
- Questions must be specific and realistic — not generic
- Answers MUST be grounded strictly in the content above — do not hallucinate
- Every answer MUST cite the specific IRC section numbers or IRS publication referenced in the content
- Every answer MUST include specific dollar amounts, percentages, or dates when present in the content
- Answers must be at least 150 characters long
- Return a JSON array of objects, each with keys: "question" and "answer"

IMPORTANT: Return your answer as a JSON object with a single key \
"pairs" whose value is the array. Example: {{"pairs": [...]}}"""

DPO_GENERATION_PROMPT = """\
Below is official IRS content extracted from: {source_url}

--- BEGIN IRS CONTENT ---
{content}
--- END IRS CONTENT ---

Generate {n_pairs} preference pairs for DPO training on this tax content.
Each pair should represent a question where one answer is clearly better.

For each pair:
- "question": A specific tax question answerable from the content
- "chosen": A high-quality answer that cites the specific IRC section, includes exact numbers/limits, explains the rule precisely, and references the source
- "rejected": A poor answer that is either vague (no specific citations, no dollar amounts), subtly incorrect, or gives boilerplate advice without substance

Return as JSON object: {{"pairs": [...]}}
Each element must have keys: "question", "chosen", "rejected".
Return ONLY valid JSON. No markdown fences, no extra text."""

# ── Chunking constants ─────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4
CHUNK_TOKENS = 3000
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN  # ~12000 chars


# ── Helper functions ───────────────────────────────────────────────────────────

def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_gpt(client, prompt: str, system: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as exc:
        log.warning("GPT call failed: %s", exc)
        return None


def parse_llm_json(text: str):
    if not text or not text.strip():
        return None
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def _unwrap_pairs(parsed) -> list:
    """Unwrap a JSON object wrapper into a list of pairs."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("pairs", "qa_pairs", "dpo_pairs", "preference_pairs", "questions", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # Fall back to first list-valued key
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return []


def _validate_sft_pair(question: str, answer: str, source_url: str) -> bool:
    if not question or not answer:
        return False
    if len(answer) < 150:
        return False
    has_citation = (
        "irs.gov" in answer.lower()
        or source_url in answer
        or "irc" in answer.lower()
        or "section" in answer.lower()
        or "§" in answer
        or re.search(r'\$[\d,]+', answer)
        or re.search(r'\d+(\.\d+)?%', answer)
    )
    return bool(has_citation)


def chunk_content(content: str, chunk_size: int = CHUNK_CHARS, overlap: int = 500) -> list[str]:
    if len(content) <= chunk_size:
        return [content]
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        if end < len(content):
            break_pos = content.rfind("\n\n", start + chunk_size - 500, end)
            if break_pos == -1:
                break_pos = content.rfind("\n", start + chunk_size - 200, end)
            if break_pos != -1:
                end = break_pos
        chunks.append(content[start:end].strip())
        start = end - overlap
        if start >= len(content):
            break
    return [c for c in chunks if len(c) > 200]


def content_to_sft_pairs(content: str, source_url: str, source_type: str, client, n_pairs: int = 8) -> list[dict]:
    chunks = chunk_content(content)
    pairs_per_chunk = max(3, min(10, n_pairs // max(1, len(chunks))))
    all_pairs = []

    for chunk_idx, chunk in enumerate(chunks):
        prompt = QA_GENERATION_PROMPT.format(
            source_url=source_url,
            content=chunk,
            n_pairs=pairs_per_chunk,
        )
        raw = _call_gpt(client, prompt, GENERATION_SYSTEM_PROMPT)
        if not raw:
            continue

        parsed = parse_llm_json(raw)
        items = _unwrap_pairs(parsed)

        for pair in items:
            if not isinstance(pair, dict):
                continue
            question = pair.get("question", "").strip()
            answer = pair.get("answer", "").strip()
            if not _validate_sft_pair(question, answer, source_url):
                continue
            all_pairs.append({
                "messages": [
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source_url": source_url,
                    "source_type": source_type,
                    "grounded": True,
                    "citation_validated": True,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                },
            })

    return all_pairs


def content_to_dpo_pairs(content: str, source_url: str, source_type: str, client, n_pairs: int = 4) -> list[dict]:
    chunk = content[:CHUNK_CHARS] if len(content) > CHUNK_CHARS else content

    prompt = DPO_GENERATION_PROMPT.format(
        source_url=source_url,
        content=chunk,
        n_pairs=n_pairs,
    )
    raw = _call_gpt(client, prompt, GENERATION_SYSTEM_PROMPT)
    if not raw:
        return []

    parsed = parse_llm_json(raw)
    items = _unwrap_pairs(parsed)

    dpo_pairs = []
    for pair in items:
        if not isinstance(pair, dict):
            continue
        question = pair.get("question", "").strip()
        chosen = pair.get("chosen", "").strip()
        rejected = pair.get("rejected", "").strip()
        if not question or not chosen or not rejected:
            continue
        if len(chosen) < 150:
            continue
        if chosen == rejected:
            continue
        dpo_pairs.append({
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "source_url": source_url,
                "source_type": source_type,
            },
        })

    return dpo_pairs


def _append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


# ── Main logic ─────────────────────────────────────────────────────────────────

def get_already_processed_urls() -> set[str]:
    """Return set of source URLs already present in the extracted SFT intermediate file."""
    sft_path = TRAINING_DIR / "tavily_sft_extracted.jsonl"
    urls = set()
    if sft_path.exists():
        with sft_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    url = rec.get("metadata", {}).get("source_url", "")
                    if url:
                        urls.add(url)
                except Exception:
                    pass
    return urls


def process_extracted_content(client, test_mode: bool = False) -> tuple[list[dict], list[dict]]:
    """
    Process extracted content files not yet turned into SFT/DPO pairs.
    Returns (new_sft_pairs, new_dpo_pairs).
    """
    already_processed = get_already_processed_urls()
    log.info("Already processed URLs (from extracted SFT file): %d", len(already_processed))

    all_files = sorted(EXTRACTED_DIR.glob("*.json"))
    remaining = []
    for fpath in all_files:
        try:
            rec = json.loads(fpath.read_text())
            url = rec.get("url", "")
            if url not in already_processed:
                remaining.append(fpath)
        except Exception:
            pass

    if test_mode:
        remaining = remaining[:5]
        log.info("TEST MODE: processing only %d extracted files", len(remaining))
    else:
        log.info("Remaining extracted files to process: %d / %d total",
                 len(remaining), len(all_files))

    new_sft: list[dict] = []
    new_dpo: list[dict] = []

    for idx, fpath in enumerate(remaining, 1):
        try:
            record = json.loads(fpath.read_text())
        except Exception as exc:
            log.warning("Could not read %s: %s", fpath, exc)
            continue

        content = record.get("content", "")
        url = record.get("url", str(fpath))

        if len(content) < 200:
            log.debug("Skipping %s — content too short (%d chars)", url, len(content))
            continue

        sft_pairs = content_to_sft_pairs(
            content=content,
            source_url=url,
            source_type="irs_publication",
            client=client,
            n_pairs=10,
        )
        new_sft.extend(sft_pairs)

        dpo_pairs = content_to_dpo_pairs(
            content=content,
            source_url=url,
            source_type="irs_publication",
            client=client,
            n_pairs=5,
        )
        new_dpo.extend(dpo_pairs)

        log.info("[%d/%d extracted] %s  SFT+%d DPO+%d  (running: %d SFT, %d DPO)",
                 idx, len(remaining), fpath.name[:60],
                 len(sft_pairs), len(dpo_pairs),
                 len(new_sft), len(new_dpo))

        # Checkpoint: append immediately so we don't lose work on crash
        if sft_pairs:
            _append_jsonl(TRAINING_DIR / "tavily_sft_extracted.jsonl", sft_pairs)
        if dpo_pairs:
            _append_jsonl(TRAINING_DIR / "tavily_dpo_extracted.jsonl", dpo_pairs)

        time.sleep(0.3)

    return new_sft, new_dpo


def get_search_dpo_processed_urls() -> set[str]:
    """Return set of source URLs already in the search DPO intermediate file."""
    dpo_search_path = TRAINING_DIR / "tavily_dpo_search.jsonl"
    urls = set()
    if dpo_search_path.exists():
        with dpo_search_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    url = rec.get("metadata", {}).get("source_url", "")
                    if url:
                        urls.add(url)
                except Exception:
                    pass
    return urls


def process_search_dpo(client, test_mode: bool = False) -> list[dict]:
    """
    Generate DPO pairs from search result files.
    Only processes files whose primary URL hasn't been DPO-processed yet.
    Returns new DPO pairs.
    """
    already_done = get_search_dpo_processed_urls()
    log.info("Already processed search URLs for DPO: %d", len(already_done))

    search_files = sorted(SEARCH_DIR.glob("*.json"))
    if test_mode:
        search_files = search_files[:5]
        log.info("TEST MODE: processing only %d search files for DPO", len(search_files))

    new_dpo: list[dict] = []

    for idx, sfpath in enumerate(search_files, 1):
        try:
            data = json.loads(sfpath.read_text())
        except Exception as exc:
            log.warning("Could not read %s: %s", sfpath, exc)
            continue

        topic = data.get("topic", str(sfpath.stem))
        results = data.get("results", [])

        # Sort by score and take top-5
        results_sorted = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        top_results = results_sorted[:5]

        combined_parts = []
        primary_url = ""
        for item_idx, item in enumerate(top_results):
            url = item.get("url", "")
            content = item.get("raw_content") or item.get("content", "")
            if len(content) < 300:
                continue
            if not primary_url:
                primary_url = url
            snippet = content[:2000]
            combined_parts.append(f"--- Source {item_idx+1}: {url} ---\n{snippet}")

        if not combined_parts or not primary_url:
            continue

        # Skip if already processed
        if primary_url in already_done:
            continue

        combined_content = "\n\n".join(combined_parts)

        dpo_pairs = content_to_dpo_pairs(
            content=combined_content,
            source_url=primary_url,
            source_type="tavily_search",
            client=client,
            n_pairs=5,
        )
        new_dpo.extend(dpo_pairs)

        log.info("[%d/%d search DPO] '%s' → +%d DPO (running total: %d)",
                 idx, len(search_files), topic, len(dpo_pairs), len(new_dpo))

        # Checkpoint
        if dpo_pairs:
            _append_jsonl(TRAINING_DIR / "tavily_dpo_search.jsonl", dpo_pairs)

        time.sleep(0.3)

    return new_dpo


def consolidate_all(test_mode: bool = False) -> tuple[int, int]:
    """
    Read all intermediate files and consolidate into final processed files.
    - tavily_sft_full.jsonl: deduplicated union of all SFT sources
    - tavily_dpo_full.jsonl: deduplicated union of all DPO sources
    Returns (total_sft, total_dpo).
    """
    log.info("Consolidating all training pairs...")

    # ── SFT ──
    sft_sources = [
        TRAINING_DIR / "tavily_sft_extracted.jsonl",
        TRAINING_DIR / "tavily_sft_search.jsonl",
        TRAINING_DIR / "tavily_sft_pairs.jsonl",  # original combined file if exists
    ]

    sft_all: list[dict] = []
    sft_seen: set[str] = set()

    for src in sft_sources:
        if not src.exists():
            continue
        count_before = len(sft_all)
        with src.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msgs = rec.get("messages", [])
                # Dedup key: question text
                question = next((m["content"] for m in msgs if m["role"] == "user"), "")
                key = question.strip().lower()[:200]
                if key and key not in sft_seen:
                    sft_seen.add(key)
                    sft_all.append(rec)
        log.info("  SFT from %s: +%d (deduped)", src.name, len(sft_all) - count_before)

    sft_out = PROCESSED_DIR / "tavily_sft_full.jsonl"
    _write_jsonl(sft_out, sft_all)

    # ── DPO ──
    dpo_sources = [
        TRAINING_DIR / "tavily_dpo_extracted.jsonl",
        TRAINING_DIR / "tavily_dpo_search.jsonl",
    ]

    dpo_all: list[dict] = []
    dpo_seen: set[str] = set()

    for src in dpo_sources:
        if not src.exists():
            continue
        count_before = len(dpo_all)
        with src.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # Dedup key: prompt text
                key = rec.get("prompt", "").strip().lower()[:200]
                if key and key not in dpo_seen:
                    dpo_seen.add(key)
                    dpo_all.append(rec)
        log.info("  DPO from %s: +%d (deduped)", src.name, len(dpo_all) - count_before)

    dpo_out = PROCESSED_DIR / "tavily_dpo_full.jsonl"
    _write_jsonl(dpo_out, dpo_all)

    return len(sft_all), len(dpo_all)


def print_sample(sft_all: list[dict], dpo_all: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("SAMPLE SFT PAIR")
    print("=" * 60)
    for rec in sft_all[-3:]:
        msgs = rec.get("messages", [])
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        src = rec.get("metadata", {}).get("source_url", "")
        print(f"  Q: {q[:120]}")
        print(f"  A: {a[:200]}...")
        print(f"  Source: {src}")
        print()

    print("\n" + "=" * 60)
    print("SAMPLE DPO PAIR")
    print("=" * 60)
    for rec in dpo_all[-2:]:
        print(f"  Q: {rec.get('prompt','')[:120]}")
        print(f"  CHOSEN: {rec.get('chosen','')[:200]}...")
        print(f"  REJECTED: {rec.get('rejected','')[:150]}...")
        src = rec.get("metadata", {}).get("source_url", "")
        print(f"  Source: {src}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Complete Tavily training data generation")
    parser.add_argument("--test", action="store_true", help="Process only 5 files per category")
    parser.add_argument("--skip-search", action="store_true", help="Skip search DPO generation")
    parser.add_argument("--consolidate-only", action="store_true",
                        help="Skip generation, just consolidate existing intermediates")
    args = parser.parse_args()

    if not args.consolidate_only:
        try:
            client = _get_openai_client()
        except RuntimeError as exc:
            log.error("Cannot generate: %s", exc)
            sys.exit(1)

        # Step 1: process remaining extracted content files
        log.info("=== STEP 1: Extracted content (SFT + DPO) ===")
        new_ext_sft, new_ext_dpo = process_extracted_content(client, test_mode=args.test)
        log.info("Extracted step done: +%d SFT, +%d DPO new pairs", len(new_ext_sft), len(new_ext_dpo))

        # Step 2: generate DPO from search results (SFT already done by harvester)
        if not args.skip_search:
            log.info("=== STEP 2: Search results DPO ===")
            new_search_dpo = process_search_dpo(client, test_mode=args.test)
            log.info("Search DPO step done: +%d new DPO pairs", len(new_search_dpo))

    # Step 3: consolidate
    log.info("=== STEP 3: Consolidate ===")
    total_sft, total_dpo = consolidate_all(test_mode=args.test)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL COUNTS")
    print("=" * 60)
    print(f"  SFT pairs total: {total_sft}")
    print(f"  DPO pairs total: {total_dpo}")
    print(f"  SFT output: {PROCESSED_DIR / 'tavily_sft_full.jsonl'}")
    print(f"  DPO output: {PROCESSED_DIR / 'tavily_dpo_full.jsonl'}")
    print("=" * 60)

    # Print samples from final files
    sft_sample = []
    dpo_sample = []
    sft_path = PROCESSED_DIR / "tavily_sft_full.jsonl"
    dpo_path = PROCESSED_DIR / "tavily_dpo_full.jsonl"

    if sft_path.exists():
        with sft_path.open() as f:
            lines = f.readlines()
            for line in lines[-3:]:
                try:
                    sft_sample.append(json.loads(line))
                except Exception:
                    pass

    if dpo_path.exists():
        with dpo_path.open() as f:
            lines = f.readlines()
            for line in lines[-2:]:
                try:
                    dpo_sample.append(json.loads(line))
                except Exception:
                    pass

    if sft_sample or dpo_sample:
        print_sample(sft_sample, dpo_sample)


if __name__ == "__main__":
    main()
