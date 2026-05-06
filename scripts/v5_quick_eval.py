#!/usr/bin/env python3
"""
Quick manual evaluation of v5 model — 5 regression-test questions.
Checks if standard deduction and Section 179 hallucinations from v4 are fixed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "qwen25-3b-mlx"
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "grpo" / "adapters"

QUESTIONS = [
    "What is the standard deduction for 2024?",
    "What is the maximum Section 179 expense deduction?",
    "Who qualifies as a qualifying individual under Section 21?",
    "What is the penalty for substantial understatement under Section 6662?",
    "How does Section 7701 define a partnership?",
]


def main():
    print("Loading model + v5 GRPO adapter...")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Adapter: {ADAPTER_PATH}")

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(MODEL_PATH), adapter_path=str(ADAPTER_PATH))
    sampler = make_sampler(temp=0.3)

    print("\n" + "=" * 70)
    print("v5 GRPO Model — Regression Test (5 questions)")
    print("=" * 70)

    results = []
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 60)

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful tax law assistant. Answer questions about "
                        "US federal tax law accurately, citing relevant IRC sections."
                    ),
                },
                {"role": "user", "content": question},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = question
        else:
            prompt = question

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            sampler=sampler,
            verbose=False,
        )

        # Strip echoed prompt if present
        if response.startswith(prompt):
            response = response[len(prompt):]
        response = response.strip()

        print(f"A{i}: {response}")
        results.append((question, response))

    return results


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 70)
    print("Done.")
