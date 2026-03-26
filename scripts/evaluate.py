#!/usr/bin/env python3
"""
Evaluation Script — Tax Law LLM.

Runs a suite of 25 tax-law questions, scores responses, and compares the
fine-tuned model against the baseline (no adapter).

Scoring:
    - IRC section citation presence (+0.4)
    - Factual keyword coverage       (+0.4)
    - Response length plausibility   (+0.2)

Results written to: outputs/eval_results.json

Usage:
    python scripts/evaluate.py [--adapter-path outputs/grpo/adapters]
    python scripts/evaluate.py --baseline-only   # evaluate base model only
    python scripts/evaluate.py --max-tokens 512
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_MLX = PROJECT_ROOT / "models" / "qwen25-3b-mlx"
MODEL_HF = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct"
EVAL_RESULTS = PROJECT_ROOT / "outputs" / "eval_results.json"

ADAPTER_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "grpo" / "adapters",
    PROJECT_ROOT / "outputs" / "dpo" / "adapters",
    PROJECT_ROOT / "outputs" / "sft" / "adapters",
]


# ---------------------------------------------------------------------------
# Evaluation questions with expected signals
# ---------------------------------------------------------------------------
# Each entry: (question, [expected_irc_sections], [expected_keywords])
EVAL_QUESTIONS: list[tuple[str, list[str], list[str]]] = [
    # Section 61 — Gross income
    (
        "What is included in gross income under IRC Section 61? Give examples.",
        ["61"],
        ["compensation", "wages", "interest", "dividends", "rents", "royalties",
         "gains", "income from whatever source"],
    ),
    # Section 63 — Standard deduction
    (
        "What is the standard deduction and how does it work under IRC Section 63?",
        ["63"],
        ["standard deduction", "itemized", "filing status", "adjusted gross income"],
    ),
    # Section 1 — Capital gains tax rates
    (
        "Explain the capital gains tax rates under IRC Section 1. "
        "What is the rate for long-term capital gains?",
        ["1", "1(h)"],
        ["long-term", "short-term", "0%", "15%", "20%", "holding period"],
    ),
    # Section 162 — Business deductions
    (
        "What ordinary and necessary business expenses are deductible "
        "under IRC Section 162?",
        ["162"],
        ["ordinary", "necessary", "trade or business", "deductible", "expense"],
    ),
    # Section 170 — Charitable contributions
    (
        "What are the rules for deducting charitable contributions under "
        "IRC Section 170? What is the AGI limitation?",
        ["170"],
        ["charitable", "contribution", "501(c)(3)", "60%", "AGI", "deduction limit"],
    ),
    # Section 280A — Home office
    (
        "When can a taxpayer deduct home office expenses under IRC Section 280A? "
        "What is the exclusive use requirement?",
        ["280A"],
        ["exclusive use", "regular basis", "principal place", "home office",
         "trade or business"],
    ),
    # Section 401 — 401(k) qualified plans
    (
        "What are the contribution limits and basic rules for 401(k) plans "
        "under IRC Section 401?",
        ["401", "401(k)"],
        ["elective deferral", "contribution limit", "employer match",
         "vesting", "qualified plan"],
    ),
    # Section 408 — IRAs
    (
        "What is the difference between a traditional IRA and a Roth IRA "
        "under IRC Sections 408 and 408A?",
        ["408", "408A"],
        ["traditional IRA", "Roth IRA", "deductible", "tax-free", "contribution limit",
         "income limit", "distribution"],
    ),
    # Section 501 — Tax-exempt organizations
    (
        "What types of organizations qualify for tax exemption under "
        "IRC Section 501(c)(3)? What is the inurement prohibition?",
        ["501", "501(c)(3)"],
        ["charitable", "religious", "educational", "scientific", "inurement",
         "private benefit", "public charity"],
    ),
    # Section 1031 — Like-kind exchanges
    (
        "How does a like-kind exchange work under IRC Section 1031? "
        "What property qualifies?",
        ["1031"],
        ["like-kind", "real property", "boot", "exchange", "defer", "gain",
         "qualified intermediary"],
    ),
    # Section 179 — Expensing election
    (
        "What is the Section 179 expensing election and what are its limits?",
        ["179"],
        ["expensing", "first-year", "deduction", "phase-out", "business use",
         "tangible personal property"],
    ),
    # Section 168 — MACRS / bonus depreciation
    (
        "How does bonus depreciation work under IRC Section 168(k)? "
        "What is the phase-down schedule?",
        ["168", "168(k)"],
        ["bonus depreciation", "first year", "MACRS", "placed in service",
         "phase-down", "100%", "80%"],
    ),
    # Section 267 — Related party losses
    (
        "What are the restrictions on deducting losses between related parties "
        "under IRC Section 267?",
        ["267"],
        ["related party", "loss disallowance", "constructive ownership",
         "family member", "controlled", "deferral"],
    ),
    # Section 469 — Passive activity losses
    (
        "What are the passive activity loss rules under IRC Section 469? "
        "What is the $25,000 rental exception?",
        ["469"],
        ["passive activity", "material participation", "rental", "$25,000",
         "active participation", "suspended losses"],
    ),
    # Section 121 — Home sale exclusion
    (
        "How does the home sale exclusion work under IRC Section 121? "
        "What is the dollar limit for a married couple?",
        ["121"],
        ["exclusion", "principal residence", "$250,000", "$500,000", "married",
         "2 out of 5 years", "ownership", "use"],
    ),
    # Section 1014 — Step-up in basis
    (
        "Explain the step-up in basis at death under IRC Section 1014. "
        "How does it affect inherited property?",
        ["1014"],
        ["step-up", "fair market value", "date of death", "inherited",
         "basis", "capital gains"],
    ),
    # Section 2503 — Annual gift exclusion
    (
        "What is the annual gift tax exclusion under IRC Section 2503? "
        "How much can a person give tax-free per year per recipient?",
        ["2503", "2501"],
        ["annual exclusion", "$18,000", "$17,000", "gift tax", "per recipient",
         "present interest"],
    ),
    # Section 6662 — Accuracy penalties
    (
        "What penalties apply for substantial understatements of tax under "
        "IRC Section 6662?",
        ["6662"],
        ["substantial understatement", "20%", "accuracy-related", "negligence",
         "reasonable cause", "substantial authority"],
    ),
    # Section 72 — Annuities / early withdrawal
    (
        "What is the 10% early withdrawal penalty for retirement accounts "
        "under IRC Section 72(t)? What are the exceptions?",
        ["72", "72(t)"],
        ["10%", "early withdrawal", "59½", "exception", "substantially equal",
         "disability", "death"],
    ),
    # Section 199A — QBI deduction
    (
        "How does the qualified business income deduction work under "
        "IRC Section 199A for pass-through entities?",
        ["199A"],
        ["qualified business income", "20%", "pass-through", "W-2 wages",
         "specified service", "SSTB", "threshold"],
    ),
    # Section 163 — Interest deduction
    (
        "What types of interest are deductible under IRC Section 163? "
        "What are the limitations on investment interest and mortgage interest?",
        ["163", "163(h)"],
        ["mortgage interest", "qualified residence", "investment interest",
         "business interest", "limitation", "deductible"],
    ),
    # Section 104 — Damages exclusion
    (
        "Are personal injury lawsuit damages taxable? What does IRC Section 104 say?",
        ["104"],
        ["personal physical injury", "physical sickness", "excludable",
         "damages", "compensatory", "punitive", "taxable"],
    ),
    # Section 2056 — Marital deduction
    (
        "What is the unlimited marital deduction for estate tax purposes "
        "under IRC Section 2056?",
        ["2056", "2001"],
        ["marital deduction", "unlimited", "U.S. citizen", "surviving spouse",
         "estate tax", "QTIP", "qualified terminable interest"],
    ),
    # Section 1221 — Capital asset definition
    (
        "What is a capital asset under IRC Section 1221? "
        "What property is excluded from capital asset treatment?",
        ["1221"],
        ["capital asset", "inventory", "accounts receivable", "depreciable property",
         "real property used in trade", "copyrights", "exclusion"],
    ),
    # Section 83 — Property transferred for services
    (
        "How are restricted stock units (RSUs) and stock options taxed under "
        "IRC Section 83? What is the Section 83(b) election?",
        ["83", "83(b)"],
        ["substantial risk of forfeiture", "83(b) election", "vesting",
         "fair market value", "ordinary income", "RSU", "stock option"],
    ),
]

# Total: 25 questions


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_response(
    response: str,
    expected_sections: list[str],
    expected_keywords: list[str],
) -> dict[str, Any]:
    """
    Return a score dict with component scores and overall [0, 1].

    Components:
        citation_score  (0 or 0.4)   — any expected section cited
        keyword_score   (0–0.4)      — fraction of keywords present
        length_score    (0 or 0.2)   — response is 50–2000 chars
    """
    response_lower = response.lower()

    # Citation score
    cited = any(
        re.search(rf"\b{re.escape(sec)}\b", response, re.IGNORECASE)
        for sec in expected_sections
    )
    citation_score = 0.4 if cited else 0.0

    # Keyword coverage
    matched_keywords = [
        kw for kw in expected_keywords
        if kw.lower() in response_lower
    ]
    kw_fraction = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0.0
    keyword_score = round(kw_fraction * 0.4, 4)

    # Length score
    length = len(response.strip())
    length_score = 0.2 if 50 <= length <= 2000 else 0.0

    overall = citation_score + keyword_score + length_score

    return {
        "citation_score": citation_score,
        "keyword_score": keyword_score,
        "length_score": length_score,
        "overall": round(overall, 4),
        "cited_sections": cited,
        "matched_keywords": matched_keywords,
        "response_length": length,
    }


# ---------------------------------------------------------------------------
# Model loading and generation
# ---------------------------------------------------------------------------

def resolve_model_path() -> Path:
    if MODEL_MLX.exists() and (MODEL_MLX / "config.json").exists():
        return MODEL_MLX
    if MODEL_HF.exists() and (MODEL_HF / "config.json").exists():
        return MODEL_HF
    print(f"ERROR: No model found at {MODEL_MLX} or {MODEL_HF}")
    sys.exit(1)


def resolve_adapter(override: str | None) -> Path | None:
    if override:
        p = Path(override)
        if not (p / "adapter_config.json").exists():
            print(f"WARNING: No adapter_config.json at {p}. Evaluating base model.")
            return None
        return p
    for candidate in ADAPTER_CANDIDATES:
        if (candidate / "adapter_config.json").exists():
            return candidate
    return None


def load_model(model_path: Path, adapter_path: Path | None):
    from mlx_lm import load as mlx_load

    if adapter_path:
        model, tokenizer = mlx_load(str(model_path), adapter_path=str(adapter_path))
    else:
        model, tokenizer = mlx_load(str(model_path))
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_tokens: int,
    temperature: float = 0.3,
) -> str:
    from mlx_lm.utils import generate

    # Format as chat using the model's chat template if available
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
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )
    # Strip echoed prompt if present
    if response.startswith(prompt):
        response = response[len(prompt):]
    return response.strip()


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    label: str,
    max_tokens: int,
) -> list[dict]:
    results = []
    total = len(EVAL_QUESTIONS)
    print(f"\nEvaluating: {label} ({total} questions)")
    print("-" * 60)

    for idx, (question, sections, keywords) in enumerate(EVAL_QUESTIONS, 1):
        print(f"  [{idx:2d}/{total}] {question[:70]}...", end=" ", flush=True)
        t0 = time.time()
        response = generate_answer(model, tokenizer, question, max_tokens)
        elapsed = time.time() - t0
        score = score_response(response, sections, keywords)
        print(f"score={score['overall']:.2f} ({elapsed:.1f}s)")
        results.append({
            "idx": idx,
            "question": question,
            "expected_sections": sections,
            "response": response,
            "score": score,
            "elapsed_s": round(elapsed, 2),
        })
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def summarise(results: list[dict]) -> dict:
    overall_scores = [r["score"]["overall"] for r in results]
    citation_scores = [r["score"]["citation_score"] for r in results]
    keyword_scores = [r["score"]["keyword_score"] for r in results]
    n = len(results)
    return {
        "n_questions": n,
        "mean_overall": round(sum(overall_scores) / n, 4),
        "mean_citation": round(sum(citation_scores) / n, 4),
        "mean_keyword": round(sum(keyword_scores) / n, 4),
        "pct_cited_section": round(
            sum(1 for s in citation_scores if s > 0) / n * 100, 1
        ),
    }


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{label} Summary:")
    print(f"  Mean overall score:  {summary['mean_overall']:.3f} / 1.000")
    print(f"  Mean citation score: {summary['mean_citation']:.3f} / 0.400")
    print(f"  Mean keyword score:  {summary['mean_keyword']:.3f} / 0.400")
    print(f"  Questions citing IRC section: {summary['pct_cited_section']:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned tax law LLM")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (default: auto-detect best available)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate only the baseline model (no adapter)",
    )
    parser.add_argument(
        "--finetuned-only",
        action="store_true",
        help="Evaluate only the fine-tuned model (skip baseline)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EVAL_RESULTS),
        help=f"Output JSON path (default: {EVAL_RESULTS})",
    )
    args = parser.parse_args()

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("ERROR: mlx_lm not found. Install with: pip install mlx-lm")
        sys.exit(1)

    model_path = resolve_model_path()
    adapter_path = None if args.baseline_only else resolve_adapter(args.adapter_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "meta": {
            "model_path": str(model_path),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "n_questions": len(EVAL_QUESTIONS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # --- Baseline ---
    if not args.finetuned_only:
        print(f"\nLoading baseline model from {model_path} ...")
        baseline_model, baseline_tokenizer = load_model(model_path, adapter_path=None)
        baseline_results = evaluate_model(
            baseline_model, baseline_tokenizer, "Baseline (no adapter)", args.max_tokens
        )
        baseline_summary = summarise(baseline_results)
        print_summary("Baseline", baseline_summary)
        all_results["baseline"] = {
            "summary": baseline_summary,
            "questions": baseline_results,
        }
        del baseline_model  # free memory before loading fine-tuned

    # --- Fine-tuned ---
    if not args.baseline_only and adapter_path is not None:
        print(f"\nLoading fine-tuned model (adapter: {adapter_path}) ...")
        ft_model, ft_tokenizer = load_model(model_path, adapter_path)
        ft_results = evaluate_model(
            ft_model, ft_tokenizer, f"Fine-tuned ({adapter_path.parent.name})", args.max_tokens
        )
        ft_summary = summarise(ft_results)
        print_summary("Fine-tuned", ft_summary)
        all_results["finetuned"] = {
            "adapter_path": str(adapter_path),
            "summary": ft_summary,
            "questions": ft_results,
        }

        # Delta comparison
        if "baseline" in all_results:
            delta = round(
                ft_summary["mean_overall"] - baseline_summary["mean_overall"], 4
            )
            all_results["delta_overall"] = delta
            print(f"\nImprovement over baseline: {delta:+.4f}")

    elif not args.baseline_only and adapter_path is None:
        print(
            "\nNo adapter found. Run SFT/DPO/GRPO training first to evaluate "
            "the fine-tuned model."
        )

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
