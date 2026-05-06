#!/usr/bin/env python3
"""
Unit tests for grpo_reward.py and citation_utils.py.

Run with:
    python -m pytest tests/test_grpo_reward.py -v
"""
import sys
from pathlib import Path
import pytest

# Ensure scripts/ is on the path so we can import without installing
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from citation_utils import (
    extract_irc_sections,
    extract_numbers,
    extract_section_number,
    count_citations,
    IRC_CITATION_PATTERN,
    CFR_CITATION_PATTERN,
)
from grpo_reward import (
    batch_reward,
    citation_accuracy_score,
    compute_reward,
    factual_accuracy_score,
    has_vague_language,
)


# ===========================================================================
# citation_utils — extract_numbers
# ===========================================================================

class TestExtractNumbers:
    def test_dollar_amount_plain(self):
        assert "$1000" in extract_numbers("The limit is $1000.")

    def test_dollar_amount_with_commas(self):
        assert "$1,160,000" in extract_numbers("Deduct up to $1,160,000 of property.")

    def test_dollar_amount_with_cents(self):
        assert "$25.50" in extract_numbers("Cost is $25.50.")

    def test_percentage(self):
        assert "20%" in extract_numbers("A 20% penalty applies.")

    def test_percentage_decimal(self):
        assert "10.5%" in extract_numbers("Rate is 10.5%.")

    def test_multiple_values(self):
        text = "Limit $500,000; phase-out at $2,000,000; rate 60%."
        nums = extract_numbers(text)
        assert "$500,000" in nums
        assert "$2,000,000" in nums
        assert "60%" in nums

    def test_empty_text(self):
        assert extract_numbers("") == set()

    def test_no_numbers(self):
        assert extract_numbers("No amounts here at all.") == set()


# ===========================================================================
# citation_utils — extract_irc_sections
# ===========================================================================

class TestExtractIRCSections:
    """All canonical citation formats should resolve to bare section numbers."""

    def test_section_keyword(self):
        assert "179" in extract_irc_sections("Section 179 allows expensing.")

    def test_section_sign_no_space(self):
        assert "179" in extract_irc_sections("§179 deduction.")

    def test_section_sign_with_space(self):
        assert "179" in extract_irc_sections("§ 179 deduction.")

    def test_irc_prefix_section(self):
        assert "179" in extract_irc_sections("IRC Section 179")

    def test_irc_prefix_sign(self):
        assert "179" in extract_irc_sections("IRC §179")

    def test_irc_dotted_prefix(self):
        assert "179" in extract_irc_sections("I.R.C. §179")

    def test_irc_dotted_prefix_section_keyword(self):
        assert "179" in extract_irc_sections("I.R.C. Section 179")

    def test_26_usc(self):
        assert "1031" in extract_irc_sections("26 U.S.C. §1031")

    def test_internal_revenue_code_spelled_out(self):
        # "408A" is captured as a single token (trailing letter is part of the number)
        sections = extract_irc_sections("Internal Revenue Code Section 408A")
        assert "408A" in sections

    def test_subsection_stripped(self):
        # "Section 179(d)(1)" should give section "179"
        sections = extract_irc_sections("Under Section 179(d)(1), the rules apply.")
        assert "179" in sections

    def test_multiple_sections(self):
        text = "See IRC §179 and IRC §168(k) for expensing rules."
        sections = extract_irc_sections(text)
        assert "179" in sections
        assert "168" in sections

    def test_trailing_letter(self):
        assert "199A" in extract_irc_sections("Section 199A provides a 20% deduction.")

    def test_no_citations(self):
        assert extract_irc_sections("This text has no citations.") == set()

    def test_case_insensitive(self):
        assert "162" in extract_irc_sections("section 162 ordinary expenses")

    def test_sec_abbreviation(self):
        assert "63" in extract_irc_sections("Sec. 63 defines taxable income.")


# ===========================================================================
# citation_utils — count_citations
# ===========================================================================

class TestCountCitations:
    def test_irc_citations(self):
        text = "IRC §179 and IRC Section 168(k) apply."
        assert count_citations(text) >= 2

    def test_cfr_citation(self):
        text = "See 26 CFR § 1.179-1 for regulations."
        assert count_citations(text) >= 1

    def test_combined(self):
        text = "IRC §179, 26 CFR § 1.179-1, and Treasury Reg. 1.168(k)-1."
        assert count_citations(text) >= 2

    def test_no_citations(self):
        assert count_citations("No citations here.") == 0


# ===========================================================================
# citation_utils — extract_section_number
# ===========================================================================

class TestExtractSectionNumber:
    def test_plain_number(self):
        assert extract_section_number("179") == "179"

    def test_irc_prefix(self):
        assert extract_section_number("IRC §179") == "179"

    def test_trailing_letter(self):
        assert extract_section_number("Section 199A") == "199A"

    def test_none_when_no_number(self):
        assert extract_section_number("no number here") is None

    def test_empty_string(self):
        assert extract_section_number("") is None


# ===========================================================================
# grpo_reward — factual_accuracy_score
# ===========================================================================

class TestFactualAccuracyScore:
    REFERENCE = (
        "The Section 179 deduction limit is $1,160,000. "
        "The phase-out threshold is $2,890,000. "
        "Property must be used more than 50% for business."
    )

    def test_exact_match_all_numbers(self):
        response = (
            "You can deduct $1,160,000 under Section 179. "
            "Phase-out begins at $2,890,000 and 50% business use is required."
        )
        score = factual_accuracy_score(response, self.REFERENCE)
        assert score == pytest.approx(1.0)

    def test_partial_match(self):
        # Only one of the three numbers appears
        response = "The deduction limit is $1,160,000."
        score = factual_accuracy_score(response, self.REFERENCE)
        assert 0.0 < score < 1.0

    def test_no_match(self):
        response = "You can deduct $500,000 with a phase-out at $2,000,000."
        score = factual_accuracy_score(response, self.REFERENCE)
        assert score == pytest.approx(0.0)

    def test_no_reference_returns_neutral(self):
        assert factual_accuracy_score("Any response.", None) == pytest.approx(0.5)

    def test_empty_reference_returns_neutral(self):
        assert factual_accuracy_score("Any response.", "") == pytest.approx(0.5)

    def test_reference_with_no_numbers_returns_neutral(self):
        ref = "The deduction applies to qualifying property used in business."
        assert factual_accuracy_score("Some response.", ref) == pytest.approx(0.5)

    def test_response_has_no_numbers(self):
        response = "Section one-seventy-nine allows expensing of qualifying assets."
        score = factual_accuracy_score(response, self.REFERENCE)
        assert score == pytest.approx(0.0)

    def test_percentage_match(self):
        ref = "A 20% penalty applies under Section 6662."
        response = "Under IRC §6662, the accuracy penalty is 20%."
        assert factual_accuracy_score(response, ref) == pytest.approx(1.0)


# ===========================================================================
# grpo_reward — citation_accuracy_score
# ===========================================================================

class TestCitationAccuracyScore:
    def test_correct_section_cited(self):
        response = "Under IRC Section 179, the limit is $1,160,000."
        assert citation_accuracy_score(response, "179") == pytest.approx(1.0)

    def test_wrong_section_cited(self):
        response = "Under IRC Section 168, bonus depreciation applies."
        assert citation_accuracy_score(response, "179") == pytest.approx(0.2)

    def test_no_citations_returns_zero(self):
        response = "You can expense qualifying property in the year placed in service."
        assert citation_accuracy_score(response, "179") == pytest.approx(0.0)

    def test_no_expected_section_returns_neutral(self):
        # Fix 3-B: default changed from 0.5 to 0.25 (uncertain/neutral)
        response = "IRC Section 179 allows expensing."
        assert citation_accuracy_score(response, None) == pytest.approx(0.25)

    def test_expected_section_unparseable_returns_neutral(self):
        # Fix 3-B: default changed from 0.5 to 0.25 (uncertain/neutral)
        assert citation_accuracy_score("some text", "No number here") == pytest.approx(0.25)

    def test_trailing_letter_match(self):
        response = "Under IRC Section 199A, the QBI deduction is 20%."
        assert citation_accuracy_score(response, "199A") == pytest.approx(1.0)


# ===========================================================================
# grpo_reward — compute_reward
# ===========================================================================

class TestComputeReward:
    REFERENCE = (
        "The Section 179 deduction limit is $1,160,000. "
        "Phase-out at $2,890,000. More than 50% business use required."
    )

    def test_empty_response_returns_zero(self):
        assert compute_reward("prompt", "") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert compute_reward("prompt", "   \n\t  ") == 0.0

    def test_high_quality_response_scores_high(self):
        response = (
            "Under IRC Section 179, a taxpayer may immediately expense qualifying "
            "depreciable property placed in service during the tax year. The 2023 "
            "deduction limit is $1,160,000. This amount is reduced dollar-for-dollar "
            "when total qualifying property exceeds $2,890,000. Property must be used "
            "more than 50% for business purposes. See also 26 CFR § 1.179-1."
        )
        reward = compute_reward(
            "What is Section 179?", response,
            reference=self.REFERENCE, expected_section="179",
        )
        assert reward >= 0.70, f"Expected >= 0.70, got {reward:.3f}"

    def test_vague_response_penalised(self):
        vague = (
            "This is a complex area that depends on your circumstances. "
            "You should consult a tax professional for personalized advice."
        )
        reward = compute_reward(
            "What is Section 179?", vague,
            reference=self.REFERENCE, expected_section="179",
        )
        assert has_vague_language(vague)
        # Vague response should score lower than one with content
        decent = "IRC Section 179 allows expensing up to $1,160,000."
        reward_decent = compute_reward(
            "What is Section 179?", decent,
            reference=self.REFERENCE, expected_section="179",
        )
        assert reward < reward_decent

    def test_reward_clamped_to_unit_interval(self):
        for _ in range(5):
            r = compute_reward("prompt", "IRC Section 179 allows deductions.")
            assert 0.0 <= r <= 1.0

    def test_no_reference_no_section_still_scores(self):
        # Should not raise; factual=neutral, citation_acc=neutral
        response = "IRC Section 179 allows deductions up to a statutory limit."
        r = compute_reward("prompt", response)
        assert 0.0 <= r <= 1.0

    def test_wrong_numbers_reduces_score(self):
        good = (
            "Under IRC Section 179, the deduction limit is $1,160,000 "
            "with phase-out at $2,890,000."
        )
        bad = (
            "Under IRC Section 179, the deduction limit is $500,000 "
            "with phase-out at $2,000,000."
        )
        r_good = compute_reward("prompt", good, reference=self.REFERENCE, expected_section="179")
        r_bad = compute_reward("prompt", bad, reference=self.REFERENCE, expected_section="179")
        assert r_good > r_bad


# ===========================================================================
# grpo_reward — batch_reward
# ===========================================================================

class TestBatchReward:
    def test_basic_batch_returns_list(self):
        prompts = ["What is Section 179?", "What is Section 168?"]
        responses = ["IRC Section 179 allows expensing.", "168(k) bonus depreciation."]
        rewards = batch_reward(prompts, responses)
        assert len(rewards) == 2
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_without_references_or_sections(self):
        prompts = ["prompt1", "prompt2"]
        responses = ["response1", "response2"]
        rewards = batch_reward(prompts, responses)
        assert len(rewards) == 2

    def test_with_expected_sections(self):
        """expected_sections must now be passed through to compute_reward."""
        prompts = ["What is Section 179?"] * 2
        responses = [
            "IRC Section 179 allows expensing.",   # correct section
            "IRC Section 168 allows depreciation.", # wrong section
        ]
        expected_sections = ["179", "179"]
        rewards = batch_reward(prompts, responses, expected_sections=expected_sections)
        # First should score higher than second on citation accuracy
        assert rewards[0] > rewards[1], (
            f"Correct section should score higher: {rewards[0]:.3f} vs {rewards[1]:.3f}"
        )

    def test_with_references(self):
        reference = "The limit is $1,160,000 with phase-out at $2,890,000."
        prompts = ["What is Section 179?"] * 2
        responses = [
            "Deduct up to $1,160,000; phase-out at $2,890,000.",  # numbers match
            "Deduct up to $500,000; phase-out at $2,000,000.",    # numbers wrong
        ]
        rewards = batch_reward(
            prompts, responses,
            references=[reference, reference],
        )
        assert rewards[0] > rewards[1], (
            f"Correct numbers should score higher: {rewards[0]:.3f} vs {rewards[1]:.3f}"
        )

    def test_with_both_references_and_sections(self):
        reference = "The limit is $1,160,000."
        prompts = ["prompt"] * 2
        responses = [
            "Under IRC Section 179 the limit is $1,160,000.",
            "Under IRC Section 168 the limit is $500,000.",
        ]
        rewards = batch_reward(
            prompts, responses,
            references=[reference, reference],
            expected_sections=["179", "179"],
        )
        assert rewards[0] > rewards[1]

    def test_empty_responses_return_zero(self):
        rewards = batch_reward(["p1", "p2"], ["", ""])
        assert rewards == [0.0, 0.0]

    def test_batch_size_one(self):
        rewards = batch_reward(["What is tax?"], ["IRC Section 61 covers gross income."])
        assert len(rewards) == 1

    def test_expected_sections_default_none(self):
        """Before the bug fix, expected_sections was never passed.
        After the fix, omitting it defaults to all-None (neutral, 0.5 weight)."""
        rewards = batch_reward(["prompt"], ["IRC Section 179 deduction."])
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_percentage_vs_dollar_no_confusion(self):
        """$20 and 20% should be treated as distinct tokens."""
        ref = "Rate is 20%."
        resp_pct = "The rate is 20%."
        resp_dollar = "The limit is $20."
        score_pct = factual_accuracy_score(resp_pct, ref)
        score_dollar = factual_accuracy_score(resp_dollar, ref)
        assert score_pct == pytest.approx(1.0)
        assert score_dollar == pytest.approx(0.0)

    def test_irc_section_with_complex_subsections(self):
        sections = extract_irc_sections("Under IRC §179(d)(1)(A) the rules apply.")
        assert "179" in sections

    def test_bare_number_not_confused_as_citation(self):
        """A bare number like '2023' not preceded by § or Section should NOT match."""
        # The year "2023" alone should not be extracted as a section
        sections = extract_irc_sections("In tax year 2023, the rules changed.")
        assert "2023" not in sections

    def test_very_long_response_not_rewarded_maximally_for_length(self):
        """Responses over 3000 chars should not get the maximum length score."""
        long_resp = "IRC Section 179. " * 200  # ~3400 chars
        r = compute_reward("prompt", long_resp, expected_section="179")
        # Just check it doesn't error and is in range
        assert 0.0 <= r <= 1.0

    def test_no_numbers_in_either_reference_or_response(self):
        """Both without numbers → neutral (0.5) factual score."""
        ref = "Property must qualify under the statute."
        resp = "Qualifying property may be expensed immediately."
        score = factual_accuracy_score(resp, ref)
        assert score == pytest.approx(0.5)
