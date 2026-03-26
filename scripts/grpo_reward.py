#!/usr/bin/env python3
"""
GRPO reward function for tax law responses.

Rewards higher-quality responses that:
1. Cite specific IRC/CFR sections
2. Are sufficiently detailed
3. Avoid vague non-answers
4. Use precise legal language
"""
import re
from typing import Optional


# ── Citation patterns ─────────────────────────────────────────────────────────

IRC_CITATION_PATTERN = re.compile(
    r"(?:IRC|I\.R\.C\.|Internal Revenue Code)\s*(?:Section|§|Sec\.?)\s*(\d+\w*)",
    re.IGNORECASE,
)
CFR_CITATION_PATTERN = re.compile(
    r"(?:26\s*C\.?F\.?R\.?|Treasury\s*Reg(?:ulation)?s?)\s*[§\s]*(\d+[\.\w\-]+)",
    re.IGNORECASE,
)
SECTION_PATTERN = re.compile(
    r"[§\s](?:section\s+)?(\d+[A-Za-z]?(?:\(\w+\))*)",
    re.IGNORECASE,
)

# Vague non-answer phrases that indicate low quality
VAGUE_PHRASES = [
    "consult a tax professional",
    "depends on your circumstances",
    "complex and vary",
    "facts and circumstances",
    "i cannot provide",
    "i am not able to",
    "please seek professional advice",
    "this is not legal advice",
    "i'm not able to give",
    "you should talk to",
]

# Precision legal language that indicates quality
LEGAL_PRECISION_TERMS = [
    "taxable income",
    "gross income",
    "adjusted gross income",
    "deduction",
    "exclusion",
    "credit",
    "basis",
    "recognition",
    "realization",
    "ordinary income",
    "capital gain",
    "tax liability",
    "filing status",
    "taxpayer",
    "fiscal year",
    "taxable year",
    "withholding",
    "estimated tax",
    "penalty",
    "interest",
    "statute of limitations",
]


def count_citations(response: str) -> int:
    """Count the number of specific legal citations in a response."""
    irc_cites = len(IRC_CITATION_PATTERN.findall(response))
    cfr_cites = len(CFR_CITATION_PATTERN.findall(response))
    # Also count generic § references
    sec_refs = len(SECTION_PATTERN.findall(response))
    return irc_cites + cfr_cites + max(0, sec_refs - irc_cites - cfr_cites)


def has_vague_language(response: str) -> bool:
    """Check if response contains vague non-answer language."""
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in VAGUE_PHRASES)


def count_legal_terms(response: str) -> int:
    """Count precision legal terms used."""
    response_lower = response.lower()
    return sum(1 for term in LEGAL_PRECISION_TERMS if term in response_lower)


def compute_reward(
    prompt: str,
    response: str,
    reference: Optional[str] = None,
) -> float:
    """
    Compute a scalar reward for a tax law response.

    Returns a float in [0.0, 1.0].

    Scoring breakdown:
    - Citation score:       0.40 (up to 4 citations for full score)
    - Length/detail score:  0.25 (200-1500 chars is ideal range)
    - Precision score:      0.20 (legal term usage)
    - Penalty for vague:   -0.30 (applied if vague language detected)
    - Bonus for reference:  0.15 (if response overlaps with reference text)
    """
    if not response or not response.strip():
        return 0.0

    # 1. Citation score (0.0 - 0.40)
    n_citations = count_citations(response)
    citation_score = min(n_citations / 4.0, 1.0) * 0.40

    # 2. Length/detail score (0.0 - 0.25)
    response_len = len(response)
    if response_len < 50:
        length_score = 0.0
    elif response_len < 200:
        length_score = (response_len - 50) / 150 * 0.15
    elif response_len <= 1500:
        length_score = 0.25
    elif response_len <= 3000:
        # Slightly penalize very long responses
        length_score = 0.25 - (response_len - 1500) / 1500 * 0.05
    else:
        length_score = 0.20

    # 3. Precision legal language score (0.0 - 0.20)
    n_terms = count_legal_terms(response)
    precision_score = min(n_terms / 5.0, 1.0) * 0.20

    # 4. Vague language penalty (-0.30)
    vague_penalty = -0.30 if has_vague_language(response) else 0.0

    # 5. Reference overlap bonus (0.0 - 0.15)
    reference_bonus = 0.0
    if reference:
        # Compute simple word overlap
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        if reference_words:
            overlap = len(response_words & reference_words) / len(reference_words)
            reference_bonus = min(overlap, 1.0) * 0.15

    total = citation_score + length_score + precision_score + vague_penalty + reference_bonus
    # Clamp to [0, 1]
    return max(0.0, min(1.0, total))


def batch_reward(
    prompts: list[str],
    responses: list[str],
    references: Optional[list[str]] = None,
) -> list[float]:
    """
    Compute rewards for a batch of (prompt, response) pairs.

    Args:
        prompts: List of input prompts
        responses: List of model responses
        references: Optional list of reference answers

    Returns:
        List of float rewards in [0.0, 1.0]
    """
    if references is None:
        references = [None] * len(prompts)

    return [
        compute_reward(p, r, ref)
        for p, r, ref in zip(prompts, responses, references)
    ]


if __name__ == "__main__":
    # Test the reward function
    print("Testing GRPO reward function...\n")

    test_cases = [
        {
            "name": "High quality with citations",
            "prompt": "What does IRC Section 63 say about taxable income?",
            "response": (
                "Under IRC Section 63, taxable income is defined as gross income minus "
                "the deductions allowed under Chapter 1. For individuals who do not "
                "itemize deductions, taxable income equals adjusted gross income (AGI) "
                "minus the standard deduction under IRC § 63(c) and personal exemptions. "
                "For itemizing taxpayers, taxable income equals AGI minus itemized "
                "deductions under IRC § 63(d). The standard deduction amounts vary by "
                "filing status and are adjusted annually for inflation under IRC § 63(c)(4). "
                "Special rules apply to dependents under IRC § 63(c)(5), limiting their "
                "standard deduction. See also 26 CFR § 1.63-1 for Treasury Regulations."
            ),
        },
        {
            "name": "Vague non-answer",
            "prompt": "What does IRC Section 63 say about taxable income?",
            "response": (
                "This is a complex area of tax law that depends on your circumstances. "
                "You should consult a tax professional for advice specific to your situation."
            ),
        },
        {
            "name": "Moderate quality, no citations",
            "prompt": "What does IRC Section 63 say about taxable income?",
            "response": (
                "Taxable income is generally defined as gross income minus allowable "
                "deductions. For most individual taxpayers, this means starting with "
                "adjusted gross income and then subtracting either the standard deduction "
                "or itemized deductions. The standard deduction amount depends on filing "
                "status and is adjusted each year for inflation."
            ),
        },
        {
            "name": "Empty response",
            "prompt": "What is IRC Section 1?",
            "response": "",
        },
    ]

    for tc in test_cases:
        reward = compute_reward(tc["prompt"], tc["response"])
        print(f"Test: {tc['name']}")
        print(f"  Citations found: {count_citations(tc['response'])}")
        print(f"  Legal terms: {count_legal_terms(tc['response'])}")
        print(f"  Vague: {has_vague_language(tc['response'])}")
        print(f"  Reward: {reward:.3f}")
        print()
