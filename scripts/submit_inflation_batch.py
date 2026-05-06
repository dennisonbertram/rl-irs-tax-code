#!/usr/bin/env python3
"""Prepare and submit the inflation-specific batch requests."""
import json, os, re
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

from openai import OpenAI
client = OpenAI()

BATCH_DIR = ROOT / "data" / "batch"
REFERENCE_DIR = ROOT / "data" / "reference"

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a tax law expert specializing in U.S. federal tax law, IRS regulations, "
    "and the Internal Revenue Code. Always cite specific IRC sections, Treasury Regulations, "
    "or IRS publications when answering. Provide accurate, detailed explanations."
)

INFLATION_SFT_PROMPT = """\
You are generating training data to teach a tax AI assistant current-year U.S. tax figures.

The following is a specific U.S. federal tax amount for {tax_year}:
  Category: {category}
  Subcategory: {subcategory}
  Amount: {amount_display}
  Source: {source}

Generate exactly 5 diverse questions and answers about this specific figure, \
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

Generate exactly 3 preference pairs where the rejected answer uses a common \
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


def get_source_for_category(category, tax_year, sources_list):
    """Return best source string for a category."""
    year_suffix = tax_year[-2:]
    for s in sources_list:
        if f"TY{tax_year}" in s or f"TY20{year_suffix}" in s:
            if "retirement" in category.lower() and "Notice" in s:
                return s
            if "hsa" in category.lower() and "HSA" in s:
                return s
    for s in sources_list:
        if tax_year in s or f"20{year_suffix}" in s:
            return s
    return f"IRS Revenue Procedure (TY{tax_year})"


def walk_amounts(obj, tax_year, category, subcategory, amounts, sources_list):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_amounts(v, tax_year, category,
                         f"{subcategory}.{k}" if subcategory else k,
                         amounts, sources_list)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_amounts(v, tax_year, category, f"{subcategory}[{i}]",
                         amounts, sources_list)
    elif isinstance(obj, (int, float)) and obj > 100:
        amounts.append({
            "tax_year": tax_year,
            "category": category,
            "subcategory": subcategory,
            "amount": obj,
            "amount_display": f"${obj:,.0f}",
            "source": get_source_for_category(category, tax_year, sources_list),
        })


def load_inflation_amounts():
    path = REFERENCE_DIR / "inflation_adjusted_amounts.json"
    data = json.load(open(path))
    sources_list = data.get("metadata", {}).get("sources", data.get("sources", []))
    amounts = []
    for year, categories in data.get("tax_years", {}).items():
        for cat_name, cat_data in categories.items():
            walk_amounts(cat_data, year, cat_name, "", amounts, sources_list)
    return amounts


def make_inflation_sft_request(custom_id, amt):
    prompt = INFLATION_SFT_PROMPT.format(
        tax_year=amt["tax_year"],
        category=amt["category"].replace("_", " ").title(),
        subcategory=amt["subcategory"].replace("_", " ").replace(".", " > "),
        amount_display=amt["amount_display"],
        source=amt["source"],
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 2048,
            "temperature": 0.6,
        },
    }


def make_inflation_dpo_request(custom_id, amt):
    prompt = INFLATION_DPO_PROMPT.format(
        tax_year=amt["tax_year"],
        category=amt["category"].replace("_", " ").title(),
        subcategory=amt["subcategory"].replace("_", " ").replace(".", " > "),
        amount_display=amt["amount_display"],
        source=amt["source"],
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 2048,
            "temperature": 0.6,
        },
    }


def main():
    print("Loading inflation amounts...")
    amounts = load_inflation_amounts()
    print(f"  Found {len(amounts)} dollar amounts across all tax years")

    requests = []
    meta = []
    for i, amt in enumerate(amounts):
        safe_cat = re.sub(r"[^a-zA-Z0-9_]", "_", amt["category"])
        safe_sub = re.sub(r"[^a-zA-Z0-9_]", "_", amt["subcategory"])[:30]
        yr = amt["tax_year"]

        sft_cid = f"inf-sft-{yr}-{safe_cat}-{safe_sub}-{i}"
        requests.append(make_inflation_sft_request(sft_cid, amt))
        meta.append({"custom_id": sft_cid, "type": "sft", **amt})

        dpo_cid = f"inf-dpo-{yr}-{safe_cat}-{safe_sub}-{i}"
        requests.append(make_inflation_dpo_request(dpo_cid, amt))
        meta.append({"custom_id": dpo_cid, "type": "dpo", **amt})

    print(f"  Built {len(requests)} inflation requests ({len(amounts)} amounts × 2)")

    # Write batch file
    inf_batch_path = BATCH_DIR / "inflation_requests.jsonl"
    with open(inf_batch_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"  Written: {inf_batch_path}")

    # Write metadata sidecar
    inf_meta_path = BATCH_DIR / "inflation_batch_meta.jsonl"
    with open(inf_meta_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  Written: {inf_meta_path}")

    # Submit
    print("\nSubmitting inflation batch to OpenAI...")
    with open(inf_batch_path, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    print(f"  File uploaded: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch created: {batch.id}")
    print(f"  Status: {batch.status}")

    # Update batch_ids.json
    ids_path = BATCH_DIR / "batch_ids.json"
    existing_ids = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    existing_ids["inflation"] = batch.id
    ids_path.write_text(json.dumps(existing_ids, indent=2))
    print(f"\nUpdated batch_ids.json:")
    print(ids_path.read_text())


if __name__ == "__main__":
    main()
