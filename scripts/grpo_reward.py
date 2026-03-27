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


def extract_section_number(section_str: str) -> Optional[str]:
    """Extract the numeric section from strings like 'IRC §1' or 'IRC §179'."""
    m = re.search(r'(\d+[A-Za-z]?)', section_str)
    return m.group(1) if m else None


def extract_cited_sections(response: str) -> list[str]:
    """Extract all section numbers cited in a response."""
    sections = set()
    for pattern in [IRC_CITATION_PATTERN, SECTION_PATTERN]:
        for match in pattern.findall(response):
            # Normalize: strip subsection references like (a)(1)
            base = re.match(r'(\d+[A-Za-z]?)', match)
            if base:
                sections.add(base.group(1))
    return list(sections)


def citation_accuracy_score(response: str, expected_section: Optional[str]) -> float:
    """
    Check if the model cites the correct IRC section.
    Returns 1.0 if the expected section is among cited sections,
    0.5 if the model cites some section but not the expected one,
    0.0 if no sections are cited at all.
    """
    if not expected_section:
        return 0.5  # No expected section to verify against; neutral

    expected_num = extract_section_number(expected_section)
    if not expected_num:
        return 0.5  # Can't parse expected section; neutral

    cited = extract_cited_sections(response)
    if not cited:
        return 0.0  # No citations at all

    if expected_num in cited:
        return 1.0  # Correct section cited

    return 0.2  # Cited sections but wrong ones


def compute_reward(
    prompt: str,
    response: str,
    reference: Optional[str] = None,
    expected_section: Optional[str] = None,
) -> float:
    """
    Compute a scalar reward for a tax law response.

    Returns a float in [0.0, 1.0].

    Scoring breakdown (v3 — prioritizes citation accuracy):
    - Citation format score:    0.30 (up to 4 citations for full score)
    - Citation accuracy score:  0.40 (correct section cited)
    - Length/detail score:      0.30 (200-1500 chars is ideal range)
    - Penalty for vague:       -0.30 (applied if vague language detected)
    """
    if not response or not response.strip():
        return 0.0

    # 1. Citation format score (0.0 - 0.30)
    n_citations = count_citations(response)
    citation_format_score = min(n_citations / 4.0, 1.0) * 0.30

    # 2. Citation accuracy score (0.0 - 0.40)  [NEW in v3]
    accuracy = citation_accuracy_score(response, expected_section)
    citation_accuracy = accuracy * 0.40

    # 3. Length/detail score (0.0 - 0.30)
    response_len = len(response)
    if response_len < 50:
        length_score = 0.0
    elif response_len < 200:
        length_score = (response_len - 50) / 150 * 0.20
    elif response_len <= 1500:
        length_score = 0.30
    elif response_len <= 3000:
        length_score = 0.30 - (response_len - 1500) / 1500 * 0.05
    else:
        length_score = 0.25

    # 4. Vague language penalty (-0.30)
    vague_penalty = -0.30 if has_vague_language(response) else 0.0

    total = citation_format_score + citation_accuracy + length_score + vague_penalty
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
