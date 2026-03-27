#!/usr/bin/env python3
"""
Formal evaluation comparing qwen25-tax-3b (v1) and qwen25-tax-3b-v2 via Ollama.

Samples 50 random questions from grounded_sft_full.jsonl, queries both models,
and scores responses on citation accuracy, key fact match, and hallucination.
"""

import json
import random
import re
import time
import sys
import requests
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "grounded_sft_full.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "evaluation"
OUTPUT_PATH = OUTPUT_DIR / "v1_vs_v2_eval.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = {
    "v1": "qwen25-tax-3b",
    "v2": "qwen25-tax-3b-v2",
}
SAMPLE_SIZE = 50
SEED = 42
TIMEOUT = 120  # seconds per request


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_samples(path: Path, n: int, seed: int):
    """Load all entries, sample n with seed."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])
            metadata = obj.get("metadata", {})

            user_msg = None
            assistant_msg = None
            system_msg = None
            for m in messages:
                if m["role"] == "user":
                    user_msg = m["content"]
                elif m["role"] == "assistant":
                    assistant_msg = m["content"]
                elif m["role"] == "system":
                    system_msg = m["content"]

            if user_msg and assistant_msg:
                entries.append({
                    "question": user_msg,
                    "reference_answer": assistant_msg,
                    "system_prompt": system_msg or "",
                    "source_section": metadata.get("source_section", ""),
                })

    random.seed(seed)
    return random.sample(entries, min(n, len(entries)))


# ---------------------------------------------------------------------------
# Ollama query
# ---------------------------------------------------------------------------
def query_model(model: str, system_prompt: str, question: str) -> str:
    """Query an Ollama model and return the response text."""
    payload = {
        "model": model,
        "prompt": question,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR] {e}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def extract_irc_sections(text: str) -> set:
    """Extract IRC section references like §1, §61, §162(a), IRC Section 401(k), etc."""
    patterns = [
        r'§\s*(\d+[A-Za-z]?)(?:\([a-z0-9]+\))*',
        r'[Ss]ection\s+(\d+[A-Za-z]?)(?:\([a-z0-9]+\))*',
        r'IRC\s*§?\s*(\d+[A-Za-z]?)(?:\([a-z0-9]+\))*',
    ]
    sections = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            sections.add(m.group(1))
    return sections


def extract_key_numbers(text: str) -> set:
    """Extract dollar amounts, percentages, and other key numeric values."""
    numbers = set()
    # Dollar amounts: $1,000 or $1000 or $1,000,000
    for m in re.finditer(r'\$[\d,]+(?:\.\d+)?', text):
        numbers.add(m.group().replace(',', ''))
    # Percentages
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text):
        numbers.add(m.group(1) + '%')
    # Year references (4-digit)
    for m in re.finditer(r'\b((?:19|20)\d{2})\b', text):
        numbers.add(m.group(1))
    return numbers


def score_citation_accuracy(response: str, reference: str, source_section: str) -> float:
    """Score whether the response cites the correct IRC section(s)."""
    ref_sections = extract_irc_sections(reference)
    # Also consider the metadata source_section
    if source_section:
        # e.g. "IRC §61" -> "61"
        sec_match = re.search(r'(\d+[A-Za-z]?)', source_section)
        if sec_match:
            ref_sections.add(sec_match.group(1))

    if not ref_sections:
        # No sections to check against -- skip (neutral score)
        return 1.0

    resp_sections = extract_irc_sections(response)
    if not resp_sections:
        return 0.0

    # What fraction of reference sections appear in response?
    hits = ref_sections & resp_sections
    return len(hits) / len(ref_sections)


def score_key_fact_match(response: str, reference: str) -> float:
    """Score whether key numbers/thresholds from the reference appear in the response."""
    ref_numbers = extract_key_numbers(reference)
    if not ref_numbers:
        return 1.0  # Nothing to check

    resp_numbers = extract_key_numbers(response)
    if not resp_numbers:
        return 0.0

    hits = ref_numbers & resp_numbers
    return len(hits) / len(ref_numbers)


def score_no_hallucination(response: str, reference: str) -> float:
    """
    Heuristic hallucination check: penalize if the response cites IRC sections
    NOT mentioned in the reference (potential hallucination).
    This is a rough proxy -- a real check would need an LLM judge.
    """
    ref_sections = extract_irc_sections(reference)
    resp_sections = extract_irc_sections(response)

    if not resp_sections:
        return 0.5  # Can't tell -- neutral

    if not ref_sections:
        return 0.5  # No reference sections to compare against

    # Extra sections cited that aren't in reference
    extra = resp_sections - ref_sections
    if not extra:
        return 1.0

    # Penalize proportionally but cap at 0.0
    penalty = len(extra) / len(resp_sections)
    return max(0.0, 1.0 - penalty)


def score_response(response: str, reference: str, source_section: str) -> dict:
    """Compute all scores for a single response."""
    return {
        "citation_accuracy": score_citation_accuracy(response, reference, source_section),
        "key_fact_match": score_key_fact_match(response, reference),
        "no_hallucination": score_no_hallucination(response, reference),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading samples from {DATA_PATH}...")
    samples = load_samples(DATA_PATH, SAMPLE_SIZE, SEED)
    print(f"Loaded {len(samples)} samples.")

    results = []
    model_scores = {name: defaultdict(list) for name in MODELS}

    for i, sample in enumerate(samples):
        q = sample["question"]
        ref = sample["reference_answer"]
        sys_prompt = sample["system_prompt"]
        src = sample["source_section"]

        print(f"\n[{i+1}/{len(samples)}] {q[:80]}...")

        entry = {
            "question": q,
            "reference_answer": ref,
            "source_section": src,
            "responses": {},
            "scores": {},
        }

        for label, model_name in MODELS.items():
            t0 = time.time()
            response = query_model(model_name, sys_prompt, q)
            elapsed = time.time() - t0

            scores = score_response(response, ref, src)

            entry["responses"][label] = response
            entry["scores"][label] = scores
            entry["scores"][label]["latency_s"] = round(elapsed, 2)

            for k, v in scores.items():
                model_scores[label][k].append(v)
            model_scores[label]["latency_s"].append(elapsed)

            status = "OK" if not response.startswith("[ERROR]") else "ERR"
            print(f"  {label} ({model_name}): {status} | cite={scores['citation_accuracy']:.2f} fact={scores['key_fact_match']:.2f} no_hall={scores['no_hallucination']:.2f} | {elapsed:.1f}s")

        results.append(entry)

    # Aggregate
    summary = {}
    for label in MODELS:
        s = model_scores[label]
        summary[label] = {
            "model": MODELS[label],
            "n_samples": len(samples),
            "avg_citation_accuracy": round(sum(s["citation_accuracy"]) / len(s["citation_accuracy"]), 4),
            "avg_key_fact_match": round(sum(s["key_fact_match"]) / len(s["key_fact_match"]), 4),
            "avg_no_hallucination": round(sum(s["no_hallucination"]) / len(s["no_hallucination"]), 4),
            "avg_latency_s": round(sum(s["latency_s"]) / len(s["latency_s"]), 2),
        }
        # Composite score (equal weight)
        summary[label]["composite"] = round(
            (summary[label]["avg_citation_accuracy"]
             + summary[label]["avg_key_fact_match"]
             + summary[label]["avg_no_hallucination"]) / 3, 4
        )

    output = {
        "metadata": {
            "seed": SEED,
            "n_samples": len(samples),
            "models": MODELS,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": summary,
        "details": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'v1 (qwen25-tax-3b)':<22} {'v2 (qwen25-tax-3b-v2)':<22}")
    print("-" * 69)
    for metric in ["avg_citation_accuracy", "avg_key_fact_match", "avg_no_hallucination", "composite", "avg_latency_s"]:
        v1_val = summary["v1"][metric]
        v2_val = summary["v2"][metric]
        if metric == "avg_latency_s":
            print(f"{metric:<25} {v1_val:<22.2f} {v2_val:<22.2f}")
        else:
            marker = ""
            if metric != "avg_latency_s":
                if v2_val > v1_val:
                    marker = " (+)"
                elif v2_val < v1_val:
                    marker = " (-)"
            print(f"{metric:<25} {v1_val:<22.4f} {v2_val:.4f}{marker}")
    print("=" * 80)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
