#!/usr/bin/env python3
"""
Wait for remaining batches (sft, inflation) to complete, then download and process all.
Usage: python3 scripts/wait_and_download.py
"""
import json, os, re, time, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

from openai import OpenAI
client = OpenAI()

BATCH_DIR = ROOT / "data" / "batch"
PROCESSED_DIR = ROOT / "data" / "processed"
POLL_INTERVAL = 180  # 3 minutes

SYSTEM_PROMPT = (
    "You are an expert on U.S. federal tax law. Provide accurate, well-cited answers "
    "grounded in the Internal Revenue Code and Treasury Regulations."
)


def parse_llm_json(text):
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
    return None


def check_all_statuses(ids):
    statuses = {}
    for name, bid in ids.items():
        b = client.batches.retrieve(bid)
        rc = b.request_counts
        statuses[name] = {
            "batch_id": bid,
            "status": b.status,
            "completed": rc.completed if rc else 0,
            "failed": rc.failed if rc else 0,
            "total": rc.total if rc else 0,
            "output_file_id": b.output_file_id,
        }
    return statuses


def download_batch(output_file_id, dest_path):
    content = client.files.content(output_file_id)
    dest_path.write_bytes(content.content)
    lines = sum(1 for _ in open(dest_path))
    print(f"    Downloaded {dest_path.stat().st_size / 1e6:.1f} MB, {lines} lines -> {dest_path.name}")
    return lines


def process_sft_results(results_path):
    pairs = []
    errors = 0
    with open(results_path) as f:
        for line in f:
            try:
                result = json.loads(line)
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                parsed = parse_llm_json(content)
                if not parsed or "pairs" not in parsed:
                    errors += 1
                    continue
                source_id = result.get("custom_id", "unknown")
                for p in parsed["pairs"]:
                    q = str(p.get("question", "")).strip()
                    a = str(p.get("answer", "")).strip()
                    if not q or len(a) < 150:
                        continue
                    pairs.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ],
                        "metadata": {
                            "source": "tavily_batch",
                            "batch_id": source_id,
                            "grounded": True,
                        },
                    })
            except Exception:
                errors += 1

    # Dedup
    seen = set()
    deduped = []
    for p in pairs:
        q = p["messages"][1]["content"]
        if q not in seen:
            seen.add(q)
            deduped.append(p)

    out_path = PROCESSED_DIR / "bulk_sft_full.jsonl"
    with open(out_path, "w") as f:
        for p in deduped:
            f.write(json.dumps(p) + "\n")

    print(f"    SFT: {len(deduped)} pairs written (from {len(pairs)} raw, {errors} errors) -> {out_path.name}")
    return len(deduped)


def process_inflation_results(results_path):
    # Load metadata sidecar for type info
    meta_path = BATCH_DIR / "inflation_batch_meta.jsonl"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        meta[d["custom_id"]] = d
                    except Exception:
                        pass

    sft_pairs = []
    dpo_pairs = []
    errors = 0

    with open(results_path) as f:
        for line in f:
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id", "")
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                parsed = parse_llm_json(content)
                if not parsed or "pairs" not in parsed:
                    errors += 1
                    continue

                req_type = meta.get(custom_id, {}).get("type", "sft")
                amt_meta = meta.get(custom_id, {})

                for p in parsed["pairs"]:
                    q = str(p.get("question", "")).strip()

                    if req_type == "sft":
                        a = str(p.get("answer", "")).strip()
                        if not q or len(a) < 150:
                            continue
                        sft_pairs.append({
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": a},
                            ],
                            "metadata": {
                                "source": "inflation_batch",
                                "tax_year": amt_meta.get("tax_year", ""),
                                "category": amt_meta.get("category", ""),
                                "grounded": True,
                            },
                        })
                    else:  # dpo
                        chosen = str(p.get("chosen", "")).strip()
                        rejected = str(p.get("rejected", "")).strip()
                        # No minimum length for inflation DPO — model returns concise dollar amounts
                        if not q or not chosen or not rejected:
                            continue
                        dpo_pairs.append({
                            "prompt": q,
                            "chosen": chosen,
                            "rejected": rejected,
                            "metadata": {
                                "source": "inflation_batch",
                                "tax_year": amt_meta.get("tax_year", ""),
                                "category": amt_meta.get("category", ""),
                                "error_type": "wrong_dollar_amount",
                                "grounded": True,
                            },
                        })
            except Exception:
                errors += 1

    # Dedup
    def dedup_sft(pairs):
        seen = set()
        out = []
        for p in pairs:
            q = p["messages"][1]["content"]
            if q not in seen:
                seen.add(q)
                out.append(p)
        return out

    def dedup_dpo(pairs):
        seen = set()
        out = []
        for p in pairs:
            if p["prompt"] not in seen:
                seen.add(p["prompt"])
                out.append(p)
        return out

    sft_deduped = dedup_sft(sft_pairs)
    dpo_deduped = dedup_dpo(dpo_pairs)

    sft_out = PROCESSED_DIR / "inflation_sft_v2.jsonl"
    dpo_out = PROCESSED_DIR / "inflation_dpo_v2.jsonl"

    with open(sft_out, "w") as f:
        for p in sft_deduped:
            f.write(json.dumps(p) + "\n")
    with open(dpo_out, "w") as f:
        for p in dpo_deduped:
            f.write(json.dumps(p) + "\n")

    print(f"    Inflation SFT: {len(sft_deduped)} pairs -> {sft_out.name}")
    print(f"    Inflation DPO: {len(dpo_deduped)} pairs -> {dpo_out.name}")
    print(f"    Errors: {errors}")
    return len(sft_deduped), len(dpo_deduped)


def main():
    ids = json.loads((BATCH_DIR / "batch_ids.json").read_text())
    print(f"Monitoring {len(ids)} batch(es): {list(ids.keys())}")
    print(f"Poll interval: {POLL_INTERVAL}s\n")

    downloaded = set()
    results = {}

    while True:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        statuses = check_all_statuses(ids)
        print(f"[{ts}]")
        for name, s in statuses.items():
            pct = s["completed"] / s["total"] * 100 if s["total"] else 0
            print(f"  {name:12s}: {s['status']:15s} {s['completed']:>5}/{s['total']:<5} ({pct:.0f}%)")

        # Download any newly completed batches
        for name, s in statuses.items():
            if s["status"] == "completed" and name not in downloaded:
                print(f"\n  [{name}] COMPLETED — downloading...")
                if name == "sft":
                    raw_path = BATCH_DIR / "sft_results.jsonl"
                    download_batch(s["output_file_id"], raw_path)
                    n = process_sft_results(raw_path)
                    results["sft"] = n
                elif name == "dpo":
                    # Already done, but process if not yet
                    raw_path = BATCH_DIR / "dpo_results.jsonl"
                    if not (PROCESSED_DIR / "bulk_dpo_full.jsonl").exists():
                        download_batch(s["output_file_id"], raw_path)
                elif name == "inflation":
                    raw_path = BATCH_DIR / "inflation_results.jsonl"
                    if not raw_path.exists():
                        download_batch(s["output_file_id"], raw_path)
                    sft_n, dpo_n = process_inflation_results(raw_path)
                    results["inflation_sft"] = sft_n
                    results["inflation_dpo"] = dpo_n
                downloaded.add(name)

        terminal = {"completed", "failed", "expired", "cancelled"}
        if all(s["status"] in terminal for s in statuses.values()):
            print("\nAll batches in terminal state.")
            break

        print(f"  Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)

    # Final summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    total = 0
    for name, n in results.items():
        print(f"  {name:20s}: {n:>6,} pairs")
        total += n

    # Add DPO from earlier
    dpo_path = PROCESSED_DIR / "bulk_dpo_full.jsonl"
    if dpo_path.exists():
        dpo_count = sum(1 for _ in open(dpo_path))
        print(f"  {'dpo (pre-loaded)':20s}: {dpo_count:>6,} pairs")
        total += dpo_count

    print(f"  {'TOTAL':20s}: {total:>6,} pairs")
    print("\nOutput files:")
    for p in [
        PROCESSED_DIR / "bulk_sft_full.jsonl",
        PROCESSED_DIR / "bulk_dpo_full.jsonl",
        PROCESSED_DIR / "inflation_sft_v2.jsonl",
        PROCESSED_DIR / "inflation_dpo_v2.jsonl",
    ]:
        if p.exists():
            lines = sum(1 for _ in open(p))
            size_mb = p.stat().st_size / 1_048_576
            print(f"  {p.name}: {lines:,} lines ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
