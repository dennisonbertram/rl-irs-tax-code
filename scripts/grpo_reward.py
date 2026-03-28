#!/usr/bin/env python3
"""
GRPO reward function for tax law responses.

Rewards higher-quality responses that:
1. Cite specific IRC/CFR sections                (citation_format)
2. Cite the *correct* section for the question  (citation_accuracy)
3. Reproduce key factual numbers from reference  (factual_accuracy)  [NEW v4]
4. Are sufficiently detailed                     (length)
5. Avoid vague non-answers                       (vague_penalty)

Weight breakdown (v4):
    factual_accuracy  = 0.30
    citation_accuracy = 0.25
    citation_format   = 0.20
    length            = 0.15
    vague_penalty     = 0.10  (applied as a deduction)
"""
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Shared utilities (canonical citation regex lives here)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from citation_utils import (  # noqa: E402
    count_citations,
    extract_irc_sections,
    extract_numbers,
    extract_section_number,
    IRC_CITATION_PATTERN,
    CFR_CITATION_PATTERN,
)

# ---------------------------------------------------------------------------
# Vague non-answer phrases
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_vague_language(response: str) -> bool:
    """Check if response contains vague non-answer language."""
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in VAGUE_PHRASES)


def count_legal_terms(response: str) -> int:
    """Count precision legal terms used."""
    response_lower = response.lower()
    return sum(1 for term in LEGAL_PRECISION_TERMS if term in response_lower)


def extract_cited_sections(response: str) -> list[str]:
    """
    Return a list of base IRC section numbers cited in *response*.

    Uses the canonical IRC_CITATION_PATTERN from citation_utils.
    """
    return list(extract_irc_sections(response))


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def citation_accuracy_score(response: str, expected_section: Optional[str]) -> float:
    """
    Check if the model cites the correct IRC section.

    Returns:
        1.0  — expected section is among cited sections
        0.5  — no expected section provided (neutral; cannot verify)
        0.2  — model cited *some* sections but none match the expected one
        0.0  — model cited no sections at all
    """
    if not expected_section:
        return 0.5  # No ground truth; neutral

    expected_num = extract_section_number(expected_section)
    if not expected_num:
        return 0.5  # Cannot parse expected section; neutral

    cited = extract_cited_sections(response)
    if not cited:
        return 0.0  # No citations whatsoever

    if expected_num in cited:
        return 1.0  # Correct section cited

    return 0.2  # Wrong sections cited


def factual_accuracy_score(response: str, reference: Optional[str]) -> float:
    """
    Measure how many key numbers from *reference* appear in *response*.

    Key numbers = dollar amounts ($25,000) and percentages (20%).

    Returns:
        float in [0.0, 1.0] — fraction of reference numbers present in response.
        0.5 if reference is absent or contains no numbers (neutral; cannot verify).
    """
    if not reference or not reference.strip():
        return 0.5  # No reference; neutral

    ref_numbers = extract_numbers(reference)
    if not ref_numbers:
        return 0.5  # Reference has no numbers to verify against; neutral

    resp_numbers = extract_numbers(response)
    if not resp_numbers:
        return 0.0  # Reference has numbers but response has none

    matched = ref_numbers.intersection(resp_numbers)
    return len(matched) / len(ref_numbers)


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    prompt: str,
    response: str,
    reference: Optional[str] = None,
    expected_section: Optional[str] = None,
) -> float:
    """
    Compute a scalar reward for a tax law response.

    Returns a float clamped to [0.0, 1.0].

    Weight breakdown (v4):
        factual_accuracy  = 0.30  (key numbers from reference present in response)
        citation_accuracy = 0.25  (correct IRC section cited)
        citation_format   = 0.20  (citations present, up to 4 for full score)
        length            = 0.15  (200–1500 chars ideal)
        vague_penalty     = 0.10  (deducted if vague language detected)

    Args:
        prompt:           The user question (unused in scoring currently but
                          kept for API symmetry).
        response:         The model's answer.
        reference:        Gold-standard reference answer; used for factual
                          accuracy (number overlap).
        expected_section: The IRC section the question is about (e.g. "179").
                          Used for citation accuracy scoring.
    """
    if not response or not response.strip():
        return 0.0

    # 1. Factual accuracy (0.0 – 0.30)
    factual = factual_accuracy_score(response, reference)
    factual_score = factual * 0.30

    # 2. Citation accuracy (0.0 – 0.25)
    accuracy = citation_accuracy_score(response, expected_section)
    citation_accuracy = accuracy * 0.25

    # 3. Citation format (0.0 – 0.20)
    n_citations = count_citations(response)
    citation_format_score = min(n_citations / 4.0, 1.0) * 0.20

    # 4. Length / detail (0.0 – 0.15)
    response_len = len(response)
    if response_len < 50:
        length_score = 0.0
    elif response_len < 200:
        length_score = (response_len - 50) / 150 * 0.10
    elif response_len <= 1500:
        length_score = 0.15
    elif response_len <= 3000:
        length_score = 0.15 - (response_len - 1500) / 1500 * 0.05
    else:
        length_score = 0.10

    # 5. Vague language penalty (-0.10)
    vague_penalty = -0.10 if has_vague_language(response) else 0.0

    total = factual_score + citation_accuracy + citation_format_score + length_score + vague_penalty
    return max(0.0, min(1.0, total))


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def batch_reward(
    prompts: list[str],
    responses: list[str],
    references: Optional[list[str]] = None,
    expected_sections: Optional[list[Optional[str]]] = None,
) -> list[float]:
    """
    Compute rewards for a batch of (prompt, response) pairs.

    Args:
        prompts:           List of input prompts.
        responses:         List of model responses.
        references:        Optional list of reference answers (for factual
                           accuracy).  Defaults to all-None.
        expected_sections: Optional list of expected IRC section strings (for
                           citation accuracy).  Defaults to all-None.

    Returns:
        List of float rewards in [0.0, 1.0].

    Note:
        Previously *expected_sections* was missing from this function, so
        citation_accuracy_score always returned the neutral 0.5.  This is
        now fixed.
    """
    n = len(prompts)
    if references is None:
        references = [None] * n
    if expected_sections is None:
        expected_sections = [None] * n

    return [
        compute_reward(p, r, ref, sec)
        for p, r, ref, sec in zip(prompts, responses, references, expected_sections)
    ]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing GRPO reward function (v4)...\n")

    REFERENCE_179 = (
        "Under IRC Section 179, a taxpayer may elect to expense the cost of qualifying "
        "depreciable property placed in service during the tax year. For 2023, the "
        "maximum deduction is $1,160,000, subject to a phase-out when qualifying "
        "property exceeds $2,890,000. The property must be used more than 50% for "
        "business purposes."
    )

    test_cases = [
        {
            "name": "High quality — correct section + all numbers",
            "prompt": "What is the Section 179 expensing limit?",
            "response": (
                "IRC Section 179 allows taxpayers to immediately expense qualifying "
                "depreciable property. The 2023 deduction limit is $1,160,000. "
                "This limit phases out dollar-for-dollar when total property placed "
                "in service exceeds $2,890,000. Property must exceed 50% business use. "
                "See also 26 CFR § 1.179-1 for Treasury Regulation details."
            ),
            "reference": REFERENCE_179,
            "expected_section": "179",
        },
        {
            "name": "Wrong numbers — correct section cited",
            "prompt": "What is the Section 179 expensing limit?",
            "response": (
                "Under IRC Section 179 you can deduct up to $500,000 of equipment "
                "costs, with a phase-out starting at $2,000,000."
            ),
            "reference": REFERENCE_179,
            "expected_section": "179",
        },
        {
            "name": "Vague non-answer",
            "prompt": "What is the Section 179 expensing limit?",
            "response": (
                "This is a complex area of tax law that depends on your circumstances. "
                "You should consult a tax professional for advice specific to your situation."
            ),
            "reference": REFERENCE_179,
            "expected_section": "179",
        },
        {
            "name": "Moderate quality — no citations, some numbers",
            "prompt": "What is the Section 179 expensing limit?",
            "response": (
                "Businesses can expense up to $1,160,000 of qualifying property in the "
                "year it is placed in service. A phase-out applies above $2,890,000."
            ),
            "reference": REFERENCE_179,
            "expected_section": "179",
        },
        {
            "name": "Empty response",
            "prompt": "What is IRC Section 1?",
            "response": "",
            "reference": None,
            "expected_section": "1",
        },
    ]

    for tc in test_cases:
        reward = compute_reward(
            tc["prompt"], tc["response"],
            reference=tc.get("reference"),
            expected_section=tc.get("expected_section"),
        )
        print(f"Test: {tc['name']}")
        print(f"  Citations found:    {count_citations(tc['response'])}")
        print(f"  Cited sections:     {extract_cited_sections(tc['response'])}")
        print(f"  Numbers in resp:    {extract_numbers(tc['response'])}")
        print(f"  Vague:             {has_vague_language(tc['response'])}")
        print(f"  Factual accuracy:  {factual_accuracy_score(tc['response'], tc.get('reference')):.3f}")
        print(f"  Citation accuracy: {citation_accuracy_score(tc['response'], tc.get('expected_section')):.3f}")
        print(f"  Reward:            {reward:.3f}")
        print()
