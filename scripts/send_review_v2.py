#!/usr/bin/env python3
"""Send v2 review package to GPT-5.4 (o3) for re-review."""

import os
import sys
from pathlib import Path

def main():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment")
        sys.exit(1)

    review_package_path = Path(__file__).parent.parent / "docs" / "investigations" / "training-pipeline-review-package-v2.md"
    output_path = Path(__file__).parent.parent / "docs" / "investigations" / "gpt54-training-review-v2.md"

    print(f"Reading review package from: {review_package_path}")
    content = review_package_path.read_text()
    print(f"Review package size: {len(content):,} characters")

    system_prompt = """You are a senior ML engineer specializing in LLM fine-tuning. This is your SECOND review of this codebase. Your first review identified 25+ issues across 8 categories. All have been addressed.

Your job now:
1. VERIFY each fix was correctly implemented — check for off-by-one errors, incorrect API usage, or incomplete fixes
2. FIND any NEW issues introduced by the fixes
3. CHECK for any issues from the first review that were NOT actually fixed or were fixed incorrectly
4. LOOK for any issues you MISSED in the first review
5. EVALUATE overall pipeline readiness — is this ready for a production training run?

For each finding, rate as CRITICAL/HIGH/MEDIUM/LOW.

Be thorough and adversarial. If everything looks good, say so — but don't rubber-stamp it."""

    client = OpenAI(api_key=api_key)

    print("Sending to o3 for review...")
    print("(This may take a few minutes)")

    response = client.chat.completions.create(
        model="o3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=16000,
    )

    review_text = response.choices[0].message.content

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "# GPT-5.4 (o3) Second Review: IRS Tax Code Training Pipeline\n\n*Generated: 2026-03-29*\n\n"
    output_path.write_text(header + review_text)

    print(f"\nReview saved to: {output_path}")
    print(f"Token usage: {response.usage}")
    print("\n" + "=" * 70)
    print("REVIEW RESPONSE:")
    print("=" * 70 + "\n")
    print(review_text)

if __name__ == "__main__":
    main()
