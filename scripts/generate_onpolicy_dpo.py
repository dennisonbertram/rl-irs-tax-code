#!/usr/bin/env python3
"""Generate on-policy DPO pairs by querying the trained model via Ollama API.

Samples questions from grounded SFT data, runs them through the current model,
and keeps cases where the model clearly errs (wrong section, wrong numbers,
missing key facts). These on-policy errors become 'rejected' responses for DPO.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).parent.parent
SFT_DATA = PROJECT_ROOT / "data" / "processed" / "grounded_sft_full.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "onpolicy_dpo_v2.jsonl"

SYSTEM_PROMPT = (
    "You are a tax law expert specializing in the Internal Revenue Code (IRC). "
    "Always cite specific IRC sections and subsections when answering questions. "
    "Provide accurate, detailed explanations grounded strictly in the statutory text."
)


def query_ollama(prompt: str, model: str, base_url: str, timeout: int = 120) -> str:
    """Query Ollama HTTP API and return the generated text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        },
    }).encode("utf-8")

    req = Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    resp = urlopen(req, timeout=timeout)
    data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def extract_irc_sections(text: str) -> set:
    """Extract IRC section references from text (e.g., 'Section 162', '§ 401(k)')."""
    patterns = [
        r'[Ss]ection\s+(\d+[A-Za-z]?)',
        r'§\s*(\d+[A-Za-z]?)',
        r'IRC\s+(\d+[A-Za-z]?)',
        r'I\.R\.C\.\s*§?\s*(\d+[A-Za-z]?)',
    ]
    sections = set()
    for pat in patterns:
        sections.update(re.findall(pat, text))
    return sections


def extract_numbers(text: str) -> set:
    """Extract dollar amounts and percentages from text."""
    amounts = set()
    # Dollar amounts
    amounts.update(re.findall(r'\$[\d,]+(?:\.\d+)?', text))
    # Percentages
    amounts.update(re.findall(r'\d+(?:\.\d+)?%', text))
    return amounts


def is_meaningfully_wrong(model_answer: str, correct_answer: str) -> tuple[bool, str]:
    """Determine if model answer is meaningfully wrong and categorize the error.

    Returns (is_wrong, error_type) where error_type is one of:
    - 'wrong_section': cites different/wrong IRC sections
    - 'wrong_number': wrong dollar amounts or percentages
    - 'missing_key_fact': much shorter or missing critical content
    - 'gibberish': repetitive or incoherent output
    - '' if not meaningfully wrong
    """
    if len(model_answer) < 30:
        return True, "gibberish"

    # Check for repetitive/degenerate output
    words = model_answer.split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return True, "gibberish"

    # Check for wrong IRC sections
    ref_sections = extract_irc_sections(correct_answer)
    model_sections = extract_irc_sections(model_answer)
    if ref_sections and model_sections:
        if not ref_sections.intersection(model_sections):
            return True, "wrong_section"

    # Check for wrong numbers
    ref_numbers = extract_numbers(correct_answer)
    model_numbers = extract_numbers(model_answer)
    if ref_numbers and model_numbers:
        if not ref_numbers.intersection(model_numbers):
            return True, "wrong_number"

    # Check for missing key facts (model answer much shorter)
    if len(model_answer) < len(correct_answer) * 0.3 and len(correct_answer) > 100:
        return True, "missing_key_fact"

    # Check for high textual divergence using simple word overlap
    ref_words = set(correct_answer.lower().split())
    model_words = set(model_answer.lower().split())
    if ref_words:
        overlap = len(ref_words.intersection(model_words)) / len(ref_words)
        if overlap < 0.15:
            return True, "missing_key_fact"

    return False, ""


def main():
    parser = argparse.ArgumentParser(description="Generate on-policy DPO pairs via Ollama")
    parser.add_argument("--model", default="qwen25-tax-3b-v2", help="Ollama model name")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of questions to sample")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = Path(args.output)

    # Verify Ollama is reachable
    try:
        resp = urlopen(f"{args.ollama_url}/api/tags", timeout=5)
        models = json.loads(resp.read().decode())
        available = [m["name"] for m in models.get("models", [])]
        if not any(args.model in m for m in available):
            print(f"WARNING: Model '{args.model}' not found in Ollama. Available: {available}")
            print("Proceeding anyway in case the name resolves...")
        else:
            print(f"Ollama reachable. Model '{args.model}' available.")
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {args.ollama_url}: {e}")
        sys.exit(1)

    # Load SFT data
    if not SFT_DATA.exists():
        print(f"ERROR: SFT data not found at {SFT_DATA}")
        sys.exit(1)

    with open(SFT_DATA) as f:
        sft_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(sft_data)} SFT records.")

    # Sample
    random.seed(args.seed)
    samples = random.sample(sft_data, min(args.sample_size, len(sft_data)))
    print(f"Sampled {len(samples)} records for on-policy generation.\n")

    pairs = []
    error_counts = {"wrong_section": 0, "wrong_number": 0, "missing_key_fact": 0, "gibberish": 0}
    skipped = 0
    errors = 0
    start = time.time()

    for i, rec in enumerate(samples):
        messages = rec.get("messages", [])
        if len(messages) < 3:
            skipped += 1
            continue

        question = messages[1].get("content", "") if messages[1].get("role") == "user" else ""
        correct_answer = messages[2].get("content", "") if messages[2].get("role") == "assistant" else ""

        if not question or not correct_answer:
            skipped += 1
            continue

        # Build ChatML prompt
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            model_answer = query_ollama(prompt, args.model, args.ollama_url)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [Error on sample {i}]: {e}")
            if errors == 5:
                print("  (suppressing further error messages)")
            continue

        # Check if meaningfully wrong
        is_wrong, error_type = is_meaningfully_wrong(model_answer, correct_answer)

        if is_wrong and error_type:
            error_counts[error_type] += 1
            pairs.append({
                "prompt": question,
                "chosen": correct_answer,
                "rejected": model_answer,
                "error_type": error_type,
            })

        # Progress
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(samples) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(samples)}] "
                f"pairs={len(pairs)} | "
                f"wrong_section={error_counts['wrong_section']} "
                f"wrong_number={error_counts['wrong_number']} "
                f"missing_fact={error_counts['missing_key_fact']} "
                f"gibberish={error_counts['gibberish']} | "
                f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining",
                flush=True,
            )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    elapsed_total = time.time() - start
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Questions sampled:    {len(samples)}")
    print(f"Skipped (bad format): {skipped}")
    print(f"API errors:           {errors}")
    print(f"Model got it wrong:   {len(pairs)} ({100*len(pairs)/max(1,len(samples)-skipped-errors):.1f}%)")
    print(f"")
    print(f"Error breakdown:")
    for etype, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:20s}: {count}")
    print(f"")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"Saved {len(pairs)} DPO pairs to: {output_path}")


if __name__ == "__main__":
    main()
