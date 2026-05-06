#!/usr/bin/env python3
"""
Shared citation utilities for the IRS tax-code RL project.

All scripts that need to detect or extract IRC/CFR section citations
should import from this module rather than defining their own patterns.
This ensures consistent behaviour across grpo_reward.py, evaluate.py,
and generate_onpolicy_dpo.py.
"""
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Canonical citation regex
# ---------------------------------------------------------------------------
# Matches all of the following (case-insensitive):
#   Section 179          →  bare "Section N"
#   §179 / § 179         →  bare section-sign with optional space
#   IRC §179 / IRC 179   →  "IRC" prefix
#   I.R.C. §179          →  dotted abbreviation
#   I.R.C. § 179(d)(1)  →  with subsections
#   26 U.S.C. §179       →  title-26 USC citation
#   Internal Revenue Code Section 179
#   Sec. 179             →  abbreviated "Sec."
#   CFR / Treasury Regs citations are handled by CFR_CITATION_PATTERN below
#
# Capturing group 1: the numeric section (digits + optional trailing letter),
#                    e.g.  "179", "168", "199A", "408A".
# Capturing group 2 (non-captured internally): optional subsection string,
#                    e.g.  "(d)(1)", "(k)", "(t)".
# The public API always returns the base section number only (group 1).

_IRC_PREFIXES = r"""
    (?:
        (?:IRC|I\.R\.C\.)                       # IRC or I.R.C.
        |
        (?:26\s+U\.S\.C\.)                      # 26 U.S.C.
        |
        (?:Internal\s+Revenue\s+Code)           # spelled out
    )
    \s*
    (?:Section|Sec\.?|§)?                       # optional "Section"/"Sec."/"§"
    \s*
"""

_SECTION_KEYWORD = r"""
    (?:Section|Sec\.?)\s+                       # bare "Section N" or "Sec. N"
"""

_SECTION_SIGN = r"""
    (?<!C\.F\.R\.\s)(?<!C\.F\.R\.)             # Fix 1-I (review item 1-I LOW): negative
    (?<!CFR\s)(?<!CFR)                          # lookbehind so that CFR/C.F.R. section
    §\s*                                        # signs are NOT counted as IRC citations
"""

IRC_CITATION_PATTERN = re.compile(
    rf"""
    (?:
        {_IRC_PREFIXES}
        |
        {_SECTION_KEYWORD}
        |
        {_SECTION_SIGN}
    )
    (\d+[A-Za-z]?)                              # base section number (group 1)
    (?:\([^\)]*\))*                             # optional subsection(s) like (d)(1)
    """,
    re.IGNORECASE | re.VERBOSE,
)

CFR_CITATION_PATTERN = re.compile(
    r"(?:26\s*C\.?F\.?R\.?|Treasury\s*Reg(?:ulation)?s?)\s*[§\s]*(\d+[\.\w\-]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_irc_sections(text: str) -> set[str]:
    """
    Return the set of base IRC section numbers cited in *text*.

    Examples
    --------
    >>> extract_irc_sections("Under IRC §179 and Section 168(k)...")
    {'179', '168'}
    >>> extract_irc_sections("See 26 U.S.C. §1031 and I.R.C. Section 408A")
    {'1031', '408A'}
    """
    sections: set[str] = set()
    for m in IRC_CITATION_PATTERN.finditer(text):
        sections.add(m.group(1))
    return sections


def count_citations(text: str) -> int:
    """
    Count total IRC + CFR citations in *text*.

    IRC citations detected via IRC_CITATION_PATTERN; CFR citations via
    CFR_CITATION_PATTERN.  Overlapping matches are not double-counted.
    """
    irc_count = len(IRC_CITATION_PATTERN.findall(text))
    cfr_count = len(CFR_CITATION_PATTERN.findall(text))
    return irc_count + cfr_count


def extract_section_number(section_str: str) -> Optional[str]:
    """
    Extract the base section number from a free-form string.

    Useful for parsing values like "IRC §179" or "Section 408A" that
    are stored in training-data metadata fields.

    Returns the first match or *None* if no number is found.
    """
    m = re.search(r"(\d+[A-Za-z]?)", section_str)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Number / amount extraction (shared between reward and DPO generation)
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> set[str]:
    """
    Extract dollar amounts and percentages from *text*.

    Returns normalised strings, e.g. "$25,000", "20%", "10%".
    """
    amounts: set[str] = set()
    # Dollar amounts: $1,000  $1,000.50  $1000
    amounts.update(re.findall(r'\$[\d,]+(?:\.\d+)?', text))
    # Percentages: 20%  10.5%  59½% is unusual but guard against it
    amounts.update(re.findall(r'\d+(?:\.\d+)?%', text))
    return amounts
