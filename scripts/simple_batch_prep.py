#!/usr/bin/env python3
"""Prepare and submit OpenAI batch requests for training data generation."""
import json, os, glob, hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BATCH_DIR = Path(__file__).parent.parent / "data" / "batch"
BATCH_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = "You are a tax law expert. Generate training data from the provided content."

def chunk_text(text, max_chars=10000):
    """Split text into chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_chars])
        text = text[max_chars:]
    return chunks

def make_sft_request(content, source_id, idx):
    """Create one batch request for SFT pair generation."""
    return {
        "custom_id": f"sft-{source_id}-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "max_tokens": 4000,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"""Generate 8 diverse tax law Q&A pairs from this IRS content.

Requirements:
- Include factual, procedural, eligibility, and calculation questions
- Every answer MUST cite specific IRC sections or IRS regulations
- Include specific dollar amounts with the applicable tax year when relevant
- Minimum 200 character answers
- End each answer with "Consult a qualified tax professional for advice specific to your situation."

Return JSON: {{"pairs": [{{"question": "...", "answer": "...", "source": "..."}}]}}

Content:
{content[:8000]}"""}
            ]
        }
    }

def make_dpo_request(content, source_id, idx):
    """Create one batch request for DPO pair generation."""
    return {
        "custom_id": f"dpo-{source_id}-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "max_tokens": 4000,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"""Generate 4 preference pairs from this IRS content. Each pair has:
- "question": a tax law question
- "chosen": a detailed, correct answer with specific IRC citations and dollar amounts
- "rejected": a plausible but WRONG answer (wrong amount, wrong section, vague, or outdated info)

Return JSON: {{"pairs": [{{"question": "...", "chosen": "...", "rejected": "...", "error_type": "..."}}]}}

Content:
{content[:8000]}"""}
            ]
        }
    }

# Collect all requests
sft_requests = []
dpo_requests = []

# Process extracted content (255 files)
print("Processing extracted content...")
for filepath in sorted(glob.glob(str(Path(__file__).parent.parent / "data/raw/tavily/extracted_content/*.json"))):
    try:
        with open(filepath) as f:
            data = json.load(f)
        content = data.get("raw_content", data.get("content", data.get("text", "")))
        if not content or len(content) < 200:
            continue
        source_id = hashlib.md5(filepath.encode()).hexdigest()[:8]
        for i, chunk in enumerate(chunk_text(content)):
            sft_requests.append(make_sft_request(chunk, source_id, i))
            dpo_requests.append(make_dpo_request(chunk, source_id, i))
    except Exception as e:
        print(f"  Skip {filepath}: {e}")

print(f"  Extracted: {len(sft_requests)} SFT + {len(dpo_requests)} DPO requests")

# Process search results (216 files)
print("Processing search results...")
search_sft_start = len(sft_requests)
for filepath in sorted(glob.glob(str(Path(__file__).parent.parent / "data/raw/tavily/search_results/*.json"))):
    try:
        with open(filepath) as f:
            data = json.load(f)
        results = data.get("results", [])
        # Combine top results into one content block per topic
        combined = ""
        for r in results[:10]:
            rc = r.get("raw_content", r.get("content", ""))
            if rc:
                combined += f"\n\n--- Source: {r.get('url', 'unknown')} ---\n{rc[:3000]}"
        if len(combined) < 200:
            continue
        source_id = hashlib.md5(filepath.encode()).hexdigest()[:8]
        for i, chunk in enumerate(chunk_text(combined)):
            sft_requests.append(make_sft_request(chunk, source_id, i))
            dpo_requests.append(make_dpo_request(chunk, source_id, i))
    except Exception as e:
        print(f"  Skip {filepath}: {e}")

print(f"  Search: {len(sft_requests) - search_sft_start} SFT + {len(dpo_requests) - search_sft_start} DPO requests")

# Process inflation data
print("Processing inflation data...")
inflation_path = Path(__file__).parent.parent / "data/reference/inflation_adjusted_amounts.json"
if inflation_path.exists():
    with open(inflation_path) as f:
        inflation = json.load(f)
    # Create one big content block from all inflation data
    inflation_text = json.dumps(inflation.get("tax_years", inflation), indent=2)
    for i, chunk in enumerate(chunk_text(inflation_text)):
        source_id = f"inflation-{i}"
        sft_requests.append(make_sft_request(
            f"IRS inflation-adjusted tax amounts:\n{chunk}\n\nGenerate questions about SPECIFIC dollar amounts for SPECIFIC tax years. Every answer must state the exact dollar amount and the tax year it applies to.",
            source_id, i
        ))
        dpo_requests.append(make_dpo_request(
            f"IRS inflation-adjusted tax amounts:\n{chunk}\n\nFor rejected answers, use WRONG dollar amounts that are common hallucinations (e.g., $135,000 instead of $14,600 for standard deduction).",
            source_id, i
        ))

print(f"\nTotal: {len(sft_requests)} SFT + {len(dpo_requests)} DPO requests")

# Write batch files
sft_path = BATCH_DIR / "sft_requests.jsonl"
dpo_path = BATCH_DIR / "dpo_requests.jsonl"

with open(sft_path, "w") as f:
    for req in sft_requests:
        f.write(json.dumps(req) + "\n")

with open(dpo_path, "w") as f:
    for req in dpo_requests:
        f.write(json.dumps(req) + "\n")

print(f"\nWritten: {sft_path} ({len(sft_requests)} requests)")
print(f"Written: {dpo_path} ({len(dpo_requests)} requests)")

# Submit batches
from openai import OpenAI
client = OpenAI()

batch_ids = {}
for name, path in [("sft", sft_path), ("dpo", dpo_path)]:
    print(f"\nSubmitting {name}...")
    file = client.files.create(file=open(path, "rb"), purpose="batch")
    print(f"  File uploaded: {file.id}")
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_ids[name] = batch.id
    print(f"  Batch created: {batch.id}")

# Save batch IDs
ids_path = BATCH_DIR / "batch_ids.json"
with open(ids_path, "w") as f:
    json.dump(batch_ids, f, indent=2)
print(f"\nBatch IDs saved to {ids_path}")
print(json.dumps(batch_ids, indent=2))
