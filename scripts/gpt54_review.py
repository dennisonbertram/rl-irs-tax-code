#!/usr/bin/env python3
"""Send training pipeline review package to GPT-5.4 (o3) for expert review."""

import os
import sys
from openai import OpenAI

REVIEW_PACKAGE = "/Users/dennisonbertram/Develop/rl-irs-tax-code/docs/investigations/training-pipeline-review-package.md"
OUTPUT_PATH = "/Users/dennisonbertram/Develop/rl-irs-tax-code/docs/investigations/gpt54-training-review.md"

SYSTEM_PROMPT = """You are a senior ML engineer specializing in LLM fine-tuning, RLHF, and training pipeline debugging. You are reviewing a complete training pipeline for fine-tuning Qwen 2.5 3B on IRS tax code data using a 3-stage approach (SFT → DPO → GRPO) with MLX on Apple Silicon.

Review the entire codebase for:
1. BUGS — Any remaining code bugs, silent failures, incorrect API usage, or logic errors
2. TRAINING METHODOLOGY — Is the 3-stage pipeline sound? Are hyperparameters reasonable? Any anti-patterns?
3. REWARD FUNCTION — Is the GRPO reward function well-designed? Will it produce the desired behavior?
4. DATA PIPELINE — Any issues with how data is assembled, deduplicated, split, or weighted?
5. ADAPTER MANAGEMENT — Are LoRA adapters being correctly chained across stages? Any fusion/loading issues?
6. QUANTIZATION — The GGUF Q8_0 conversion is losing fine-tuned numeric precision. What's the best approach?
7. EVALUATION — Is the eval methodology sound? Are we measuring the right things?
8. ARCHITECTURE DECISIONS — Any fundamental design choices that should be reconsidered?

For each issue found, rate severity as CRITICAL/HIGH/MEDIUM/LOW and provide a specific fix recommendation.

Be thorough, adversarial, and specific. We already found and fixed two critical bugs (DPO length normalization, GRPO adapter load order). What else are we missing?"""

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print("Reading review package...")
    with open(REVIEW_PACKAGE, "r") as f:
        review_content = f.read()

    print(f"Review package: {len(review_content)} characters")

    client = OpenAI(api_key=api_key)

    print("Sending to GPT-5.4 (o3) for review... this may take a few minutes.")

    response = client.chat.completions.create(
        model="o3",
        max_completion_tokens=16000,
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please review this complete training pipeline codebase and provide your expert analysis:\n\n{review_content}"}
        ],
    )

    result = response.choices[0].message.content

    # Print full response
    print("\n" + "=" * 80)
    print("GPT-5.4 (o3) EXPERT REVIEW")
    print("=" * 80 + "\n")
    print(result)

    # Save to file
    with open(OUTPUT_PATH, "w") as f:
        f.write("# GPT-5.4 (o3) Expert Review: IRS Tax Code Training Pipeline\n\n")
        f.write(f"*Generated: 2026-03-29*\n\n")
        f.write(result)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Tokens used: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")

if __name__ == "__main__":
    main()
