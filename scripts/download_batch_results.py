#!/usr/bin/env python3
"""Check batch status, download results, and convert to training format."""
import json, re, sys, time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

from openai import OpenAI
client = OpenAI()

BATCH_DIR = ROOT / "data" / "batch"
PROCESSED_DIR = ROOT / "data" / "processed"

def check_status():
    """Check and print status of all batches."""
    ids = json.loads((BATCH_DIR / "batch_ids.json").read_text())
    for name, bid in ids.items():
        b = client.batches.retrieve(bid)
        rc = b.request_counts
        print(f"  {name}: {b.status} — {rc.completed}/{rc.total} completed, {rc.failed} failed")
    return ids

def download_results(batch_id, output_path):
    """Download batch results to a file."""
    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        print(f"  Batch {batch_id} not complete yet: {b.status}")
        return False
    content = client.files.content(b.output_file_id)
    output_path.write_bytes(content.content)
    print(f"  Downloaded to {output_path}")
    return True

def parse_llm_json(text):
    """Parse JSON from LLM response, handle markdown fences."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    try:
        return json.loads(text)
    except:
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
    return None

def process_sft_results(results_path, output_path):
    """Convert SFT batch results to training format."""
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
                    q = p.get("question", "")
                    a = p.get("answer", "")
                    if len(a) < 150 or not q:
                        continue
                    pairs.append({
                        "messages": [
                            {"role": "system", "content": "You are an expert on U.S. federal tax law. Provide accurate, well-cited answers grounded in the Internal Revenue Code and Treasury Regulations."},
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a}
                        ],
                        "metadata": {
                            "source": "tavily_batch",
                            "batch_id": source_id,
                            "grounded": True
                        }
                    })
            except Exception as e:
                errors += 1

    # Dedup by question
    seen = set()
    deduped = []
    for p in pairs:
        q = p["messages"][1]["content"]
        if q not in seen:
            seen.add(q)
            deduped.append(p)

    with open(output_path, "w") as f:
        for p in deduped:
            f.write(json.dumps(p) + "\n")

    print(f"  SFT: {len(deduped)} pairs (from {len(pairs)} raw, {errors} errors)")
    return len(deduped)

def process_dpo_results(results_path, output_path):
    """Convert DPO batch results to training format."""
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
                    q = p.get("question", "")
                    chosen = p.get("chosen", "")
                    rejected = p.get("rejected", "")
                    if not q or not chosen or not rejected or len(chosen) < 100:
                        continue
                    pairs.append({
                        "prompt": q,
                        "chosen": chosen,
                        "rejected": rejected,
                        "metadata": {
                            "source": "tavily_batch",
                            "batch_id": source_id,
                            "error_type": p.get("error_type", "unknown"),
                            "grounded": True
                        }
                    })
            except Exception as e:
                errors += 1

    # Dedup by question
    seen = set()
    deduped = []
    for p in pairs:
        if p["prompt"] not in seen:
            seen.add(p["prompt"])
            deduped.append(p)

    with open(output_path, "w") as f:
        for p in deduped:
            f.write(json.dumps(p) + "\n")

    print(f"  DPO: {len(deduped)} pairs (from {len(pairs)} raw, {errors} errors)")
    return len(deduped)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"

    if mode == "status":
        print("Batch status:")
        check_status()

    elif mode == "wait":
        print("Waiting for batches to complete...")
        ids = json.loads((BATCH_DIR / "batch_ids.json").read_text())
        while True:
            all_done = True
            for name, bid in ids.items():
                b = client.batches.retrieve(bid)
                rc = b.request_counts
                print(f"  {name}: {b.status} — {rc.completed}/{rc.total}")
                if b.status not in ("completed", "failed", "cancelled"):
                    all_done = False
            if all_done:
                print("All batches done!")
                break
            print("  Waiting 3 minutes...")
            time.sleep(180)

    elif mode == "download":
        print("Downloading and processing results...")
        ids = json.loads((BATCH_DIR / "batch_ids.json").read_text())

        sft_raw = BATCH_DIR / "sft_results.jsonl"
        dpo_raw = BATCH_DIR / "dpo_results.jsonl"

        if "sft" in ids:
            download_results(ids["sft"], sft_raw)
            process_sft_results(sft_raw, PROCESSED_DIR / "bulk_sft_full.jsonl")

        if "dpo" in ids:
            download_results(ids["dpo"], dpo_raw)
            process_dpo_results(dpo_raw, PROCESSED_DIR / "bulk_dpo_full.jsonl")

    else:
        print(f"Usage: {sys.argv[0]} [status|wait|download]")

if __name__ == "__main__":
    main()
