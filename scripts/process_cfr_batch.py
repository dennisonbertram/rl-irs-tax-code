#!/usr/bin/env python3
"""
Poll OpenAI Batch API for CFR grounded data generation completion,
download results, validate quality, and write final SFT JSONL.

Usage:
  # Poll status
  python3 scripts/process_cfr_batch.py --status

  # Poll until complete and download
  python3 scripts/process_cfr_batch.py --wait

  # Download results if already complete (provide output_file_id)
  python3 scripts/process_cfr_batch.py --download-file-id <FILE_ID>

  # Run quality validation on existing output
  python3 scripts/process_cfr_batch.py --validate
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

from openai import OpenAI

# ── Constants ──────────────────────────────────────────────────────────────────
BATCH_ID = "batch_69c80b2cb0208190aa5187070f1aafe0"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SFT_OUT = PROCESSED_DIR / "grounded_cfr_sft_full.jsonl"
CFR_JSONL = PROCESSED_DIR / "cfr_sections.jsonl"

POLL_INTERVAL_SECONDS = 300   # 5 minutes
MIN_ANSWER_LENGTH = 100
DISCLAIMER_PHRASE = "consult a qualified tax professional"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ── Citation helpers (mirrors generate_cfr_grounded_data.py) ─────────────────

def extract_primary_cfr_citation(answer_text: str) -> str | None:
    patterns = [
        r'Treas\.?\s*Reg\.?\s*§\s*(\d+\.\d+[\w()\-]*)',
        r'Reg\.?\s*§\s*(\d+\.\d+[\w()\-]*)',
        r'§\s*(\d+\.\d+[\w()\-]*)',
        r'\bsections?\s+(\d+\.\d+[\w()\-]*)',
    ]
    for pat in patterns:
        m = re.search(pat, answer_text, re.IGNORECASE)
        if m:
            return m.group(1).rstrip(".,;)")
    return None


def cfr_base_section(s: str) -> str:
    m = re.match(r'(\d+\.\d+(?:\([a-zA-Z0-9]+\))?(?:-\d+[A-Za-z]*)?)(?:\(.*)?$', s)
    return m.group(1) if m else s


def validate_cfr_citation(answer: str, source_section: str) -> bool:
    primary = extract_primary_cfr_citation(answer)
    if primary is None:
        return True
    return cfr_base_section(primary) == cfr_base_section(source_section)


# ── Status check ──────────────────────────────────────────────────────────────

def check_status(client: OpenAI, batch_id: str) -> dict:
    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    return {
        "status": batch.status,
        "completed": counts.completed,
        "failed": counts.failed,
        "total": counts.total,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def print_status(info: dict) -> None:
    total = info["total"] or 1
    pct = info["completed"] / total * 100
    print(
        f"  Status: {info['status']} | "
        f"{info['completed']:,}/{info['total']:,} completed ({pct:.1f}%) | "
        f"{info['failed']} failed"
    )
    if info["output_file_id"]:
        print(f"  Output file: {info['output_file_id']}")
    if info["error_file_id"]:
        print(f"  Error file: {info['error_file_id']}")


# ── Download and parse ─────────────────────────────────────────────────────────

def load_cfr_sections(path: Path) -> dict[str, dict]:
    sections: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("source") == "CFR":
                    sections[d["section"]] = d
    return sections


def parse_pairs_from_raw(raw: str | None, sec_num: str) -> tuple[list[dict], int]:
    if not raw:
        return [], 0
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
            discarded += 1

    return valid_pairs, discarded


SYSTEM_PROMPT = (
    "You are a tax law expert specializing in Treasury Regulations (CFR Title 26). "
    "Always cite specific Treasury Regulation sections when answering questions. "
    "Provide accurate, detailed explanations grounded strictly in the regulatory text."
)


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


def download_and_process(client: OpenAI, output_file_id: str, sft_out: Path) -> tuple[int, int, int]:
    """
    Download batch results and write SFT JSONL.
    Returns (total_records, discarded_count, error_count).
    """
    print(f"\nDownloading batch results (file_id={output_file_id})...")
    content = client.files.content(output_file_id)
    raw_lines = content.text.strip().split("\n")
    print(f"  Got {len(raw_lines)} result lines")

    sft_records = []
    total_discarded = 0
    error_count = 0

    for line in raw_lines:
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            error_count += 1
            continue

        custom_id = result.get("custom_id", "")
        sec_num = custom_id.replace("cfr-", "") if custom_id.startswith("cfr-") else None
        if not sec_num:
            error_count += 1
            continue

        error = result.get("error")
        if error:
            print(f"  [ERROR] §{sec_num}: {error}")
            error_count += 1
            continue

        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            error_count += 1
            continue

        raw_content = choices[0].get("message", {}).get("content", "")
        finish_reason = choices[0].get("finish_reason", "")
        if finish_reason == "length":
            print(f"  [WARN] §{sec_num} response was truncated")

        pairs, discarded = parse_pairs_from_raw(raw_content, sec_num)
        total_discarded += discarded

        for p in pairs:
            # Apply length filter
            if len(p["answer"]) < MIN_ANSWER_LENGTH:
                total_discarded += 1
                continue
            rec = make_sft_record(p["question"], p["answer"], sec_num)
            sft_records.append(rec)

    sft_out.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_out, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Written {len(sft_records):,} SFT records to {sft_out}")
    print(f"  Discarded {total_discarded} pairs (citation mismatch or too short)")
    print(f"  Errors: {error_count}")
    return len(sft_records), total_discarded, error_count


# ── Quality validation ─────────────────────────────────────────────────────────

def run_quality_validation(sft_out: Path, sample_size: int = 30) -> None:
    """
    Load the SFT output file, run quality checks on a random sample,
    check for duplicates, and report statistics.
    """
    print(f"\n{'='*60}")
    print("QUALITY VALIDATION")
    print(f"{'='*60}")

    if not sft_out.exists():
        print(f"[ERROR] Output file not found: {sft_out}")
        return

    records = []
    with open(sft_out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total = len(records)
    print(f"\nTotal SFT records: {total:,}")

    if total == 0:
        print("[ERROR] No records found.")
        return

    # ── Metadata completeness ───────────────────────────────────────────────
    meta_ok = 0
    grounded_true = 0
    source_cfr = 0
    for rec in records:
        meta = rec.get("metadata", {})
        if meta.get("source_section", "").startswith("CFR §"):
            meta_ok += 1
        if meta.get("grounded") is True:
            grounded_true += 1
        if meta.get("source") == "CFR":
            source_cfr += 1

    print(f"\nMetadata completeness:")
    print(f"  source_section present & correct: {meta_ok}/{total} ({meta_ok/total*100:.1f}%)")
    print(f"  grounded=True:                    {grounded_true}/{total} ({grounded_true/total*100:.1f}%)")
    print(f"  source='CFR':                     {source_cfr}/{total} ({source_cfr/total*100:.1f}%)")

    # ── Answer length stats ─────────────────────────────────────────────────
    answer_lengths = []
    for rec in records:
        msgs = rec.get("messages", [])
        if len(msgs) >= 3:
            answer_lengths.append(len(msgs[2]["content"]))

    if answer_lengths:
        avg_len = sum(answer_lengths) / len(answer_lengths)
        min_len = min(answer_lengths)
        max_len = max(answer_lengths)
        print(f"\nAnswer length stats:")
        print(f"  Average: {avg_len:.0f} chars")
        print(f"  Min:     {min_len} chars")
        print(f"  Max:     {max_len} chars")
        short_count = sum(1 for l in answer_lengths if l < MIN_ANSWER_LENGTH)
        print(f"  Too short (<{MIN_ANSWER_LENGTH} chars): {short_count}")

    # ── Duplicate check ─────────────────────────────────────────────────────
    questions = []
    answers = []
    for rec in records:
        msgs = rec.get("messages", [])
        if len(msgs) >= 3:
            questions.append(msgs[1]["content"])
            answers.append(msgs[2]["content"])

    unique_questions = len(set(questions))
    unique_answers = len(set(answers))
    dup_q = total - unique_questions
    dup_a = total - unique_answers
    print(f"\nDuplicate check:")
    print(f"  Unique questions: {unique_questions:,} ({unique_questions/total*100:.1f}%)")
    print(f"  Unique answers:   {unique_answers:,} ({unique_answers/total*100:.1f}%)")
    if dup_q > 0:
        print(f"  [WARN] {dup_q} duplicate questions found")
    if dup_a > 0:
        print(f"  [WARN] {dup_a} duplicate answers found")

    # ── Citation validation on all records ──────────────────────────────────
    cite_ok = 0
    cite_missing = 0
    cite_wrong = 0
    for rec in records:
        meta = rec.get("metadata", {})
        sec_raw = meta.get("source_section", "").replace("CFR §", "")
        msgs = rec.get("messages", [])
        if len(msgs) < 3:
            continue
        answer = msgs[2]["content"]
        primary = extract_primary_cfr_citation(answer)
        if primary is None:
            cite_missing += 1
        elif cfr_base_section(primary) == cfr_base_section(sec_raw):
            cite_ok += 1
        else:
            cite_wrong += 1

    print(f"\nCitation validation (all records):")
    print(f"  Correct citations:  {cite_ok:,} ({cite_ok/total*100:.1f}%)")
    print(f"  No citation found:  {cite_missing:,} ({cite_missing/total*100:.1f}%)")
    print(f"  Wrong citation:     {cite_wrong:,} ({cite_wrong/total*100:.1f}%)")

    # ── Disclaimer check ────────────────────────────────────────────────────
    has_disclaimer = 0
    for rec in records:
        msgs = rec.get("messages", [])
        if len(msgs) >= 3:
            if DISCLAIMER_PHRASE.lower() in msgs[2]["content"].lower():
                has_disclaimer += 1

    print(f"\nDisclaimer check:")
    print(f"  Has professional disclaimer: {has_disclaimer:,}/{total} ({has_disclaimer/total*100:.1f}%)")

    # ── Sample of 30 random pairs ───────────────────────────────────────────
    sample_recs = random.sample(records, min(sample_size, total))
    print(f"\n{'='*60}")
    print(f"RANDOM SAMPLE QUALITY CHECK ({min(sample_size, total)} pairs)")
    print(f"{'='*60}")

    issues_found = 0
    for i, rec in enumerate(sample_recs, 1):
        msgs = rec.get("messages", [])
        meta = rec.get("metadata", {})
        if len(msgs) < 3:
            continue

        q = msgs[1]["content"]
        a = msgs[2]["content"]
        sec = meta.get("source_section", "?")
        sec_num = sec.replace("CFR §", "")

        # Individual checks
        is_generic = len(q) < 30 or q.lower().startswith("what does this")
        too_short = len(a) < MIN_ANSWER_LENGTH
        no_disclaimer = DISCLAIMER_PHRASE.lower() not in a.lower()
        primary = extract_primary_cfr_citation(a)
        wrong_cite = (
            primary is not None
            and cfr_base_section(primary) != cfr_base_section(sec_num)
        )

        issues = []
        if is_generic:
            issues.append("GENERIC_Q")
        if too_short:
            issues.append(f"TOO_SHORT({len(a)})")
        if no_disclaimer:
            issues.append("NO_DISCLAIMER")
        if wrong_cite:
            issues.append(f"WRONG_CITE(cites §{primary})")

        status = "[PASS]" if not issues else f"[ISSUE: {', '.join(issues)}]"
        if issues:
            issues_found += 1

        print(f"\n  [{i:02d}] {sec} {status}")
        print(f"       Q: {q[:120]}")
        print(f"       A: {a[:200]}{'...' if len(a) > 200 else ''}")
        if issues:
            print(f"       ^ Issues: {issues}")

    print(f"\n{'='*60}")
    print(f"SAMPLE SUMMARY: {issues_found} issues found in {min(sample_size, total)} sampled pairs")
    pass_rate = (min(sample_size, total) - issues_found) / min(sample_size, total) * 100
    print(f"Pass rate: {pass_rate:.1f}%")
    print(f"{'='*60}")

    # ── Sections coverage ───────────────────────────────────────────────────
    sections_covered = set()
    for rec in records:
        meta = rec.get("metadata", {})
        sections_covered.add(meta.get("source_section", ""))

    print(f"\nSections coverage:")
    print(f"  Unique sections in output: {len(sections_covered):,} / 6,149 total")
    print(f"  Coverage: {len(sections_covered)/6149*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor and process CFR Batch API job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/process_cfr_batch.py --status
  python3 scripts/process_cfr_batch.py --wait
  python3 scripts/process_cfr_batch.py --download-file-id file-XXXXX
  python3 scripts/process_cfr_batch.py --validate
"""
    )
    parser.add_argument("--status", action="store_true", help="Check current batch status")
    parser.add_argument("--wait", action="store_true",
                        help="Poll until complete, then download and process")
    parser.add_argument("--download-file-id", metavar="FILE_ID",
                        help="Download results using a specific output file ID")
    parser.add_argument("--validate", action="store_true",
                        help="Run quality validation on the existing output file")
    parser.add_argument("--batch-id", default=BATCH_ID,
                        help=f"Batch ID to monitor (default: {BATCH_ID})")
    parser.add_argument("--output", type=Path, default=SFT_OUT,
                        help=f"Output JSONL path (default: {SFT_OUT})")
    parser.add_argument("--sample-size", type=int, default=30,
                        help="Number of random samples for quality check (default: 30)")

    args = parser.parse_args()

    # ── Validate only ─────────────────────────────────────────────────────
    if args.validate:
        run_quality_validation(args.output, sample_size=args.sample_size)
        return

    # ── Download by file ID ───────────────────────────────────────────────
    if args.download_file_id:
        client = OpenAI()
        total, discarded, errors = download_and_process(client, args.download_file_id, args.output)
        print(f"\nDownload complete: {total:,} pairs, {discarded} discarded, {errors} errors")
        run_quality_validation(args.output, sample_size=args.sample_size)
        return

    client = OpenAI()

    # ── Status check ──────────────────────────────────────────────────────
    if args.status:
        info = check_status(client, args.batch_id)
        print_status(info)
        return

    # ── Wait for completion ───────────────────────────────────────────────
    if args.wait:
        print(f"Monitoring batch: {args.batch_id}")
        print(f"Polling every {POLL_INTERVAL_SECONDS}s (5 min)...")
        while True:
            info = check_status(client, args.batch_id)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] ", end="")
            print_status(info)

            if info["status"] == "completed":
                print("\nBatch completed!")
                if info["output_file_id"]:
                    total, discarded, errors = download_and_process(
                        client, info["output_file_id"], args.output
                    )
                    print(f"\nFinal: {total:,} SFT pairs, {discarded} discarded, {errors} errors")
                    run_quality_validation(args.output, sample_size=args.sample_size)
                else:
                    print("[ERROR] No output file available despite 'completed' status.")
                break
            elif info["status"] in ("failed", "expired", "cancelled"):
                print(f"\nBatch ended with status: {info['status']}")
                if info["error_file_id"]:
                    print(f"Error file ID: {info['error_file_id']}")
                break

            time.sleep(POLL_INTERVAL_SECONDS)
        return

    # ── Default: print status ──────────────────────────────────────────────
    info = check_status(client, args.batch_id)
    print_status(info)
    print("\nUse --wait to poll until complete, or --validate to check existing output.")


if __name__ == "__main__":
    main()
