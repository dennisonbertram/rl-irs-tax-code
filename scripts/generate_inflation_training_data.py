#!/usr/bin/env python3
"""
Generate inflation-adjusted SFT and DPO training examples from the
IRS Revenue Procedure reference data.

Outputs:
  data/processed/inflation_adjusted_sft.jsonl
  data/processed/inflation_adjusted_dpo.jsonl
"""

import json
import random
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
REF_FILE  = REPO_ROOT / "data" / "reference" / "inflation_adjusted_amounts.json"
OUT_SFT   = REPO_ROOT / "data" / "processed" / "inflation_adjusted_sft.jsonl"
OUT_DPO   = REPO_ROOT / "data" / "processed" / "inflation_adjusted_dpo.jsonl"

SYSTEM_PROMPT = (
    "You are a tax law expert specializing in the Internal Revenue Code (IRC). "
    "Always cite specific IRC sections and subsections when answering questions. "
    "Provide accurate, detailed explanations grounded strictly in the statutory text."
)

CAVEAT = "Always verify current figures with the applicable IRS Revenue Procedure, as amounts are indexed for inflation annually under IRC §1(f) and may be affected by subsequent legislation."

# ── Source-citation map ───────────────────────────────────────────────────────
REV_PROC = {
    "2023": {
        "general":     "Rev. Proc. 2022-38",
        "retirement":  "IRS Notice 2022-55",
        "hsa":         "Rev. Proc. 2022-24",
    },
    "2024": {
        "general":     "Rev. Proc. 2023-34",
        "retirement":  "IRS Notice 2023-75",
        "hsa":         "Rev. Proc. 2023-23",
    },
    "2025": {
        "general":     "Rev. Proc. 2024-40",
        "retirement":  "IRS Notice 2024-80",
        "hsa":         "Rev. Proc. 2024-25",
    },
}

def fmt(n):
    """Format an integer as a dollar string."""
    return f"${n:,}"

def source_for(year, category="general"):
    return REV_PROC[str(year)].get(category, REV_PROC[str(year)]["general"])

def make_sft(question, answer, year, category, source_proc):
    return {
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": question},
            {"role": "assistant","content": answer},
        ],
        "metadata": {
            "source": "IRS Revenue Procedure",
            "grounded": True,
            "source_section": source_proc,
            "category": "inflation_adjusted",
            "tax_year": year,
            "topic": category,
        },
    }

def make_dpo(question, chosen, rejected, year, category, source_proc):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "chosen":   [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "metadata": {
            "source": "IRS Revenue Procedure",
            "grounded": True,
            "source_section": source_proc,
            "category": "inflation_adjusted",
            "tax_year": year,
            "topic": category,
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
# SFT GENERATORS  (one function per question-type family)
# ══════════════════════════════════════════════════════════════════════════════

def gen_standard_deduction(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["standard_deduction"]
        src = source_for(yr)
        yr_int = int(yr)

        # a) Direct – single filer
        q = f"What is the standard deduction for single filers in {yr}?"
        a = (
            f"For tax year {yr}, the standard deduction for single filers is {fmt(d['single'])}. "
            f"This amount is set forth in IRC §63(c)(2) and adjusted annually for inflation under IRC §63(c)(4) "
            f"and IRC §1(f). The figure for {yr} is published in {src}. "
            f"{CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "standard_deduction", src))

        # a) Direct – married filing jointly
        q = f"What is the standard deduction for married couples filing jointly in {yr}?"
        a = (
            f"For tax year {yr}, the standard deduction for married couples filing jointly is "
            f"{fmt(d['married_filing_jointly'])} under IRC §63(c)(2)(A), as inflation-adjusted per "
            f"IRC §1(f) and published in {src}. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "standard_deduction", src))

        # c) Multi-filing-status
        q = f"What are the standard deduction amounts for each filing status in {yr}?"
        a = (
            f"Under IRC §63(c)(2) and IRC §1(f), the inflation-adjusted standard deduction amounts "
            f"for tax year {yr} (per {src}) are:\n"
            f"  • Single: {fmt(d['single'])}\n"
            f"  • Married Filing Jointly: {fmt(d['married_filing_jointly'])}\n"
            f"  • Married Filing Separately: {fmt(d['married_filing_separately'])}\n"
            f"  • Head of Household: {fmt(d['head_of_household'])}\n"
            f"  • Additional deduction for age 65+ or blind (married): {fmt(d['additional_age_65_or_blind']['married'])}\n"
            f"  • Additional deduction for age 65+ or blind (unmarried): {fmt(d['additional_age_65_or_blind']['unmarried'])}\n"
            f"{CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "standard_deduction", src))

    # b) Comparative 2023-2024
    if "2023" in years and "2024" in years:
        d23, d24 = data["2023"]["standard_deduction"], data["2024"]["standard_deduction"]
        q = "How did the standard deduction change from 2023 to 2024 for single filers?"
        a = (
            f"The standard deduction for single filers increased from {fmt(d23['single'])} in 2023 to "
            f"{fmt(d24['single'])} in 2024 — a {fmt(d24['single'] - d23['single'])} increase. "
            f"This adjustment reflects inflation indexing under IRC §63(c)(4) and IRC §1(f). "
            f"The 2023 amount was published in Rev. Proc. 2022-38 and the 2024 amount in Rev. Proc. 2023-34. "
            f"{CAVEAT}"
        )
        examples.append(make_sft(q, a, 2024, "standard_deduction", "Rev. Proc. 2023-34"))

    if "2024" in years and "2025" in years:
        d24, d25 = data["2024"]["standard_deduction"], data["2025"]["standard_deduction"]
        q = "How did the standard deduction change from 2024 to 2025 for married filing jointly?"
        a = (
            f"The standard deduction for married couples filing jointly increased from {fmt(d24['married_filing_jointly'])} "
            f"in 2024 to {fmt(d25['married_filing_jointly'])} in 2025 — a {fmt(d25['married_filing_jointly'] - d24['married_filing_jointly'])} "
            f"increase due to inflation adjustments under IRC §63(c)(4) and IRC §1(f). "
            f"Sources: Rev. Proc. 2023-34 (2024) and Rev. Proc. 2024-40 (2025). {CAVEAT}"
        )
        examples.append(make_sft(q, a, 2025, "standard_deduction", "Rev. Proc. 2024-40"))

    return examples


def gen_retirement(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["retirement_contribution_limits"]
        src = source_for(yr, "retirement")
        yr_int = int(yr)

        # a) Direct – 401(k)
        q = f"What is the 401(k) elective deferral contribution limit for {yr}?"
        a = (
            f"For tax year {yr}, the elective deferral limit for 401(k), 403(b), and 457 plans is "
            f"{fmt(d['401k_403b_457_elective_deferral'])} under IRC §402(g)(1), adjusted annually for "
            f"inflation per IRC §402(g)(4). Employees age 50 or older may contribute an additional "
            f"{fmt(d['401k_catch_up_age_50_plus'])} catch-up contribution under IRC §414(v). "
            f"These limits are published in {src}. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "retirement_contribution_limits", src))

        # d) Practical – 401(k) at age 55
        q = f"How much can I contribute to my 401(k) in {yr} if I'm 55 years old?"
        a = (
            f"In {yr}, if you are age 55 (i.e., age 50 or older), you may contribute:\n"
            f"  • Regular elective deferral: {fmt(d['401k_403b_457_elective_deferral'])} under IRC §402(g)(1)\n"
            f"  • Age-50+ catch-up contribution: {fmt(d['401k_catch_up_age_50_plus'])} under IRC §414(v)\n"
            f"  • Total potential contribution: {fmt(d['401k_403b_457_elective_deferral'] + d['401k_catch_up_age_50_plus'])}\n"
        )
        if yr == "2025" and "401k_super_catch_up_age_60_to_63" in d:
            a += (
                f"Note: In {yr}, taxpayers aged 60–63 may use a SECURE 2.0 'super catch-up' of "
                f"{fmt(d['401k_super_catch_up_age_60_to_63'])} instead of the standard catch-up. "
                f"At age 55 you use the standard catch-up amount above.\n"
            )
        a += (
            f"These limits are published in {src} and are indexed under IRC §1(f). "
            f"{CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "retirement_contribution_limits", src))

        # a) Direct – IRA
        q = f"What is the IRA contribution limit for {yr}?"
        a = (
            f"For tax year {yr}, the contribution limit for traditional and Roth IRAs is "
            f"{fmt(d['traditional_roth_ira'])} under IRC §219(b)(5)(A), as adjusted for inflation "
            f"per IRC §219(b)(5)(C). Individuals age 50 or older may contribute an additional "
            f"{fmt(d['ira_catch_up_age_50_plus'])} catch-up under IRC §219(b)(5)(B). "
            f"Source: {src}. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "retirement_contribution_limits", src))

    # b) Comparative 401(k) 2023-2024
    if "2023" in years and "2024" in years:
        d23 = data["2023"]["retirement_contribution_limits"]
        d24 = data["2024"]["retirement_contribution_limits"]
        q = "How did the 401(k) contribution limit change from 2023 to 2024?"
        a = (
            f"The 401(k) elective deferral limit under IRC §402(g)(1) increased from "
            f"{fmt(d23['401k_403b_457_elective_deferral'])} in 2023 to "
            f"{fmt(d24['401k_403b_457_elective_deferral'])} in 2024 — an increase of "
            f"{fmt(d24['401k_403b_457_elective_deferral'] - d23['401k_403b_457_elective_deferral'])}. "
            f"The catch-up limit for ages 50+ remained at {fmt(d24['401k_catch_up_age_50_plus'])}. "
            f"Sources: IRS Notice 2022-55 (2023) and IRS Notice 2023-75 (2024). {CAVEAT}"
        )
        examples.append(make_sft(q, a, 2024, "retirement_contribution_limits", "IRS Notice 2023-75"))

    if "2024" in years and "2025" in years:
        d24 = data["2024"]["retirement_contribution_limits"]
        d25 = data["2025"]["retirement_contribution_limits"]
        q = "How did the 401(k) contribution limit change from 2024 to 2025?"
        a = (
            f"The 401(k) elective deferral limit under IRC §402(g)(1) increased from "
            f"{fmt(d24['401k_403b_457_elective_deferral'])} in 2024 to "
            f"{fmt(d25['401k_403b_457_elective_deferral'])} in 2025 — an increase of "
            f"{fmt(d25['401k_403b_457_elective_deferral'] - d24['401k_403b_457_elective_deferral'])}. "
            f"Additionally, SECURE 2.0 (IRC §414(v)(2)(E)) introduced a super catch-up for ages 60–63 "
            f"of {fmt(d25['401k_super_catch_up_age_60_to_63'])} effective in 2025. "
            f"Sources: IRS Notice 2023-75 (2024) and IRS Notice 2024-80 (2025). {CAVEAT}"
        )
        examples.append(make_sft(q, a, 2025, "retirement_contribution_limits", "IRS Notice 2024-80"))

    return examples


def gen_hsa(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["hsa_contribution_limits"]
        src = source_for(yr, "hsa")
        yr_int = int(yr)

        # a) Direct
        q = f"What are the HSA contribution limits for {yr}?"
        a = (
            f"For tax year {yr}, the HSA contribution limits under IRC §223(b)(2), as published in "
            f"{src}, are:\n"
            f"  • Self-only HDHP coverage: {fmt(d['self_only'])}\n"
            f"  • Family HDHP coverage: {fmt(d['family'])}\n"
            f"  • Additional catch-up contribution (age 55+): {fmt(d['catch_up_age_55_plus'])} under IRC §223(b)(3)\n"
            f"These limits are indexed for inflation under IRC §223(g). {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "hsa_contribution_limits", src))

        # d) Practical – age 57 with family plan
        q = f"How much can I contribute to my HSA in {yr} if I'm 57 years old with family coverage?"
        a = (
            f"In {yr}, with family HDHP coverage at age 57 (age 55 or older), you may contribute:\n"
            f"  • Family limit: {fmt(d['family'])} under IRC §223(b)(2)(B)\n"
            f"  • Age-55+ catch-up: {fmt(d['catch_up_age_55_plus'])} under IRC §223(b)(3)\n"
            f"  • Total: {fmt(d['family'] + d['catch_up_age_55_plus'])}\n"
            f"Source: {src}. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "hsa_contribution_limits", src))

    return examples


def gen_section_179(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["section_179_expensing"]
        src = source_for(yr)
        yr_int = int(yr)

        # e) Cross-reference: limit + phaseout
        q = f"What is the Section 179 expensing limit for {yr} and what is the phaseout threshold?"
        a = (
            f"For tax year {yr}, under IRC §179(b):\n"
            f"  • Maximum deduction: {fmt(d['maximum_deduction'])} (IRC §179(b)(1))\n"
            f"  • Investment-based phaseout begins at: {fmt(d['phaseout_threshold'])} (IRC §179(b)(2)); "
            f"the deduction is reduced dollar-for-dollar by the amount by which qualified property placed "
            f"in service exceeds this threshold.\n"
            f"  • SUV expensing limit: {fmt(d['suv_limit'])} (IRC §179(b)(5))\n"
            f"These amounts are indexed for inflation under IRC §179(b)(6) and published in {src}. "
            f"{CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "section_179_expensing", src))

        # a) Direct – max deduction only
        q = f"What is the maximum Section 179 deduction allowed in {yr}?"
        a = (
            f"For tax year {yr}, the maximum amount a taxpayer may elect to expense under IRC §179(b)(1) "
            f"is {fmt(d['maximum_deduction'])}. This limit is adjusted annually for inflation under "
            f"IRC §179(b)(6) and was published in {src}. The deduction phases out dollar-for-dollar "
            f"when total qualified property placed in service exceeds {fmt(d['phaseout_threshold'])}. "
            f"{CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "section_179_expensing", src))

    # b) Comparative
    if "2023" in years and "2024" in years:
        d23, d24 = data["2023"]["section_179_expensing"], data["2024"]["section_179_expensing"]
        q = "How did the Section 179 expensing limit change from 2023 to 2024?"
        a = (
            f"The IRC §179(b)(1) maximum deduction increased from {fmt(d23['maximum_deduction'])} in 2023 "
            f"to {fmt(d24['maximum_deduction'])} in 2024 — an increase of "
            f"{fmt(d24['maximum_deduction'] - d23['maximum_deduction'])}. The phaseout threshold "
            f"also rose from {fmt(d23['phaseout_threshold'])} to {fmt(d24['phaseout_threshold'])}. "
            f"Sources: Rev. Proc. 2022-38 (2023) and Rev. Proc. 2023-34 (2024). {CAVEAT}"
        )
        examples.append(make_sft(q, a, 2024, "section_179_expensing", "Rev. Proc. 2023-34"))

    return examples


def gen_estate_gift(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["estate_and_gift_tax"]
        src = source_for(yr)
        yr_int = int(yr)

        # a) Direct – annual gift exclusion
        q = f"What is the annual gift tax exclusion for {yr}?"
        a = (
            f"For {yr}, the annual gift tax exclusion under IRC §2503(b)(1) is {fmt(d['annual_gift_tax_exclusion'])} "
            f"per recipient, as adjusted for inflation per IRC §2503(b)(2) and published in {src}. "
            f"Gifts at or below this amount are excluded from gift tax and do not reduce the donor's "
            f"applicable exclusion amount. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "estate_and_gift_tax", src))

        # a) Direct – basic exclusion
        q = f"What is the federal estate and gift tax basic exclusion amount for {yr}?"
        a = (
            f"For {yr}, the basic exclusion amount under IRC §2010(c)(3)(A), as inflation-adjusted per "
            f"IRC §2010(c)(3)(B) and published in {src}, is {fmt(d['basic_exclusion_amount'])}. "
            f"This amount is unified across estate and gift tax under IRC §2505. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "estate_and_gift_tax", src))

    # b) Comparative annual gift exclusion
    if "2023" in years and "2024" in years:
        d23, d24 = data["2023"]["estate_and_gift_tax"], data["2024"]["estate_and_gift_tax"]
        q = "How did the annual gift tax exclusion change from 2023 to 2024?"
        a = (
            f"The annual gift tax exclusion under IRC §2503(b)(1) increased from {fmt(d23['annual_gift_tax_exclusion'])} "
            f"in 2023 to {fmt(d24['annual_gift_tax_exclusion'])} in 2024 — an increase of "
            f"{fmt(d24['annual_gift_tax_exclusion'] - d23['annual_gift_tax_exclusion'])}. "
            f"Sources: Rev. Proc. 2022-38 (2023) and Rev. Proc. 2023-34 (2024). {CAVEAT}"
        )
        examples.append(make_sft(q, a, 2024, "estate_and_gift_tax", "Rev. Proc. 2023-34"))

    if "2024" in years and "2025" in years:
        d24, d25 = data["2024"]["estate_and_gift_tax"], data["2025"]["estate_and_gift_tax"]
        q = "How did the annual gift tax exclusion change from 2024 to 2025?"
        a = (
            f"The annual gift tax exclusion under IRC §2503(b)(1) increased from {fmt(d24['annual_gift_tax_exclusion'])} "
            f"in 2024 to {fmt(d25['annual_gift_tax_exclusion'])} in 2025 — an increase of "
            f"{fmt(d25['annual_gift_tax_exclusion'] - d24['annual_gift_tax_exclusion'])}. "
            f"Sources: Rev. Proc. 2023-34 (2024) and Rev. Proc. 2024-40 (2025). {CAVEAT}"
        )
        examples.append(make_sft(q, a, 2025, "estate_and_gift_tax", "Rev. Proc. 2024-40"))

    return examples


def gen_amt(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["amt"]
        src = source_for(yr)
        yr_int = int(yr)

        # a) Direct
        q = f"What is the AMT exemption amount for single filers in {yr}?"
        a = (
            f"For tax year {yr}, the Alternative Minimum Tax (AMT) exemption under IRC §55(d)(1)(B) "
            f"for single filers is {fmt(d['exemption_single'])}, as indexed for inflation under "
            f"IRC §55(d)(4) and published in {src}. The phaseout of the exemption begins at "
            f"{fmt(d['phaseout_single'])} of AMTI under IRC §55(d)(3). {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "amt", src))

        # c) Multi-status
        q = f"What are the AMT exemption amounts for all filing statuses in {yr}?"
        a = (
            f"For tax year {yr}, the AMT exemption amounts under IRC §55(d)(1), as published in {src}, are:\n"
            f"  • Single / Head of Household: {fmt(d['exemption_single'])}\n"
            f"  • Married Filing Jointly / Surviving Spouse: {fmt(d['exemption_married_filing_jointly'])}\n"
            f"  • Married Filing Separately: {fmt(d['exemption_married_filing_separately'])}\n"
            f"Phaseout thresholds begin at {fmt(d['phaseout_single'])} (single) and "
            f"{fmt(d['phaseout_married_filing_jointly'])} (MFJ). "
            f"These are indexed under IRC §55(d)(4). {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "amt", src))

    return examples


def gen_eitc(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["earned_income_credit"]
        src = source_for(yr)
        yr_int = int(yr)

        # a) Direct – maximum credit
        q = f"What is the maximum Earned Income Tax Credit (EITC) for a family with 3 or more children in {yr}?"
        a = (
            f"For tax year {yr}, the maximum EITC for a taxpayer with 3 or more qualifying children "
            f"is {fmt(d['max_credit_3_plus_children'])} under IRC §32, as adjusted for inflation "
            f"per IRC §32(j) and published in {src}. "
            f"The credit phases out beginning at earned income of {fmt(d['income_limit_single_3_plus_children'])} "
            f"for single filers and {fmt(d['income_limit_mfj_3_plus_children'])} for married filing jointly. "
            f"{CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "earned_income_credit", src))

        # c) Multi-child breakdown
        q = f"What are the EITC maximum credit amounts by number of children for {yr}?"
        a = (
            f"For tax year {yr}, the maximum Earned Income Tax Credit amounts under IRC §32, "
            f"as published in {src}, are:\n"
            f"  • No qualifying children: {fmt(d['max_credit_0_children'])}\n"
            f"  • 1 qualifying child: {fmt(d['max_credit_1_child'])}\n"
            f"  • 2 qualifying children: {fmt(d['max_credit_2_children'])}\n"
            f"  • 3 or more qualifying children: {fmt(d['max_credit_3_plus_children'])}\n"
            f"These amounts are indexed for inflation under IRC §32(j). {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "earned_income_credit", src))

    return examples


def gen_capital_gains(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["capital_gains_brackets"]
        src = source_for(yr)
        yr_int = int(yr)

        # a) Direct – single
        q = f"What is the 0% long-term capital gains tax bracket threshold for single filers in {yr}?"
        a = (
            f"For tax year {yr}, single filers pay 0% on long-term capital gains and qualified dividends "
            f"on income up to {fmt(d['single']['zero_percent_up_to'])} under IRC §1(h)(1)(B), as "
            f"inflation-adjusted per IRC §1(f) and published in {src}. "
            f"The 15% rate applies up to {fmt(d['single']['fifteen_percent_up_to'])}, above which the 20% rate applies. "
            f"{CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "capital_gains_brackets", src))

        # c) Multi-status
        q = f"What are the long-term capital gains tax brackets for all filing statuses in {yr}?"
        a = (
            f"For tax year {yr}, the long-term capital gains rate brackets under IRC §1(h)(1), "
            f"as published in {src}, are:\n"
            f"  Single: 0% up to {fmt(d['single']['zero_percent_up_to'])}; 15% up to {fmt(d['single']['fifteen_percent_up_to'])}; 20% above\n"
            f"  MFJ: 0% up to {fmt(d['married_filing_jointly']['zero_percent_up_to'])}; 15% up to {fmt(d['married_filing_jointly']['fifteen_percent_up_to'])}; 20% above\n"
            f"  MFS: 0% up to {fmt(d['married_filing_separately']['zero_percent_up_to'])}; 15% up to {fmt(d['married_filing_separately']['fifteen_percent_up_to'])}; 20% above\n"
            f"  HoH: 0% up to {fmt(d['head_of_household']['zero_percent_up_to'])}; 15% up to {fmt(d['head_of_household']['fifteen_percent_up_to'])}; 20% above\n"
            f"These thresholds are indexed for inflation under IRC §1(f). {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "capital_gains_brackets", src))

    return examples


def gen_foreign_income_exclusion(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["foreign_earned_income_exclusion"]
        src = source_for(yr)
        yr_int = int(yr)

        q = f"What is the foreign earned income exclusion amount for {yr}?"
        a = (
            f"For tax year {yr}, the foreign earned income exclusion under IRC §911(b)(2)(D)(i) is "
            f"{fmt(d)}, as adjusted for inflation under IRC §911(b)(2)(D)(ii) and published in {src}. "
            f"Qualifying individuals who meet the bona fide residence or physical presence test may "
            f"exclude this amount of foreign earned income from gross income. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "foreign_earned_income_exclusion", src))

    return examples


def gen_qbi(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["qbi_deduction_199a"]
        src = source_for(yr)
        yr_int = int(yr)

        # e) Cross-reference threshold + phaseout
        q = f"What is the QBI deduction threshold for single filers in {yr} and how does the phaseout work?"
        a = (
            f"For tax year {yr}, the qualified business income (QBI) deduction under IRC §199A(e)(2) "
            f"has a taxable income threshold of {fmt(d['threshold_single'])} for single filers "
            f"({fmt(d['threshold_married_filing_jointly'])} for MFJ), as published in {src} and "
            f"indexed per IRC §199A(e)(2)(B). "
            f"Above the threshold, the W-2 wage and capital limitations phase in over a range of "
            f"{fmt(d['phaseout_range'])} (single) / {fmt(d['phaseout_range_mfj'])} (MFJ). "
            f"Specified service trade or business (SSTB) owners lose the deduction entirely above "
            f"the top of the phaseout range. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "qbi_deduction_199a", src))

    return examples


def gen_social_security(data, years):
    examples = []
    for yr in years:
        d   = data[yr]["social_security"]
        src = source_for(yr)
        yr_int = int(yr)

        q = f"What is the Social Security wage base for {yr}?"
        a = (
            f"For {yr}, the Social Security taxable wage base is {fmt(d['wage_base'])} under "
            f"IRC §3121(a)(1) and 42 U.S.C. §430, as adjusted by the SSA COLA announcement. "
            f"Wages up to this amount are subject to the 6.2% employee Social Security tax "
            f"(and matching 6.2% employer tax). The COLA for {yr} was {d['cola_percentage']}%. {CAVEAT}"
        )
        examples.append(make_sft(q, a, yr_int, "social_security", src))

    return examples


# ══════════════════════════════════════════════════════════════════════════════
# DPO GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def gen_dpo_standard_deduction(data, years):
    pairs = []

    # DPO 1 – correct 2024 vs. stale statutory base
    if "2024" in years:
        d = data["2024"]["standard_deduction"]
        src = source_for("2024")
        q = "What is the standard deduction for single filers in 2024?"
        chosen = (
            f"For tax year 2024, the standard deduction for single filers is {fmt(d['single'])} "
            f"under IRC §63(c)(2), as inflation-adjusted per IRC §1(f) and published in Rev. Proc. 2023-34. "
            f"{CAVEAT}"
        )
        rejected = (
            "The standard deduction for single filers is $3,000 — this is the long-standing amount "
            "established under the original statutory text of IRC §63(c)(2)."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2024, "standard_deduction", src))

    # DPO 2 – correct 2025 vs. prior-year figure
    if "2025" in years:
        d25 = data["2025"]["standard_deduction"]
        d24 = data["2024"]["standard_deduction"]
        src = source_for("2025")
        q = "What is the standard deduction for a married couple filing jointly in 2025?"
        chosen = (
            f"For tax year 2025, the standard deduction for married couples filing jointly is "
            f"{fmt(d25['married_filing_jointly'])} under IRC §63(c)(2)(A), as published in "
            f"Rev. Proc. 2024-40 and indexed under IRC §1(f). {CAVEAT}"
        )
        rejected = (
            f"For married couples filing jointly, the standard deduction is {fmt(d24['married_filing_jointly'])}."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "standard_deduction", src))

    # DPO 3 – correct 2023 vs. completely wrong number
    if "2023" in years:
        d = data["2023"]["standard_deduction"]
        src = source_for("2023")
        q = "What is the standard deduction for head of household filers in 2023?"
        chosen = (
            f"For tax year 2023, the standard deduction for head of household filers is "
            f"{fmt(d['head_of_household'])} under IRC §63(c)(2)(B), as published in "
            f"Rev. Proc. 2022-38 and indexed for inflation under IRC §1(f). {CAVEAT}"
        )
        rejected = (
            "The standard deduction for head of household filers is $18,000 in 2023."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2023, "standard_deduction", src))

    return pairs


def gen_dpo_retirement(data, years):
    pairs = []

    # DPO 1 – correct 2024 401(k) vs. wrong year's amount used
    if "2024" in years and "2023" in years:
        d24 = data["2024"]["retirement_contribution_limits"]
        d23 = data["2023"]["retirement_contribution_limits"]
        src = source_for("2024", "retirement")
        q = "What is the 401(k) contribution limit for 2024?"
        chosen = (
            f"For tax year 2024, the 401(k) elective deferral limit under IRC §402(g)(1) is "
            f"{fmt(d24['401k_403b_457_elective_deferral'])}, with an additional "
            f"{fmt(d24['401k_catch_up_age_50_plus'])} catch-up for ages 50+. "
            f"Source: IRS Notice 2023-75. {CAVEAT}"
        )
        rejected = (
            f"The 401(k) contribution limit for 2024 is {fmt(d23['401k_403b_457_elective_deferral'])} "
            f"(plus {fmt(d23['401k_catch_up_age_50_plus'])} catch-up for ages 50+)."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2024, "retirement_contribution_limits", src))

    # DPO 2 – correct 2025 IRA vs. completely wrong number
    if "2025" in years:
        d25 = data["2025"]["retirement_contribution_limits"]
        src = source_for("2025", "retirement")
        q = "What is the IRA contribution limit for 2025?"
        chosen = (
            f"For tax year 2025, the IRA contribution limit under IRC §219(b)(5)(A) is "
            f"{fmt(d25['traditional_roth_ira'])}, with a {fmt(d25['ira_catch_up_age_50_plus'])} "
            f"catch-up for individuals age 50 or older. Source: IRS Notice 2024-80. {CAVEAT}"
        )
        rejected = (
            "The IRA contribution limit for 2025 is $5,500. This is the long-standing limit before "
            "the TCJA adjustments."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "retirement_contribution_limits", src))

    # DPO 3 – 2025 super catch-up correct vs. confused with regular catch-up
    if "2025" in years:
        d25 = data["2025"]["retirement_contribution_limits"]
        src = source_for("2025", "retirement")
        q = "I'm 62 years old. How much can I contribute to my 401(k) in 2025?"
        chosen = (
            f"In 2025, at age 62 (within the SECURE 2.0 super catch-up window of ages 60-63), you may contribute:\n"
            f"  • Regular deferral: {fmt(d25['401k_403b_457_elective_deferral'])} under IRC §402(g)(1)\n"
            f"  • Super catch-up (ages 60-63): {fmt(d25['401k_super_catch_up_age_60_to_63'])} under IRC §414(v)(2)(E)\n"
            f"  • Total: {fmt(d25['401k_403b_457_elective_deferral'] + d25['401k_super_catch_up_age_60_to_63'])}\n"
            f"Source: IRS Notice 2024-80. {CAVEAT}"
        )
        rejected = (
            f"At age 62 in 2025, you can contribute {fmt(d25['401k_403b_457_elective_deferral'])} plus "
            f"the standard {fmt(d25['401k_catch_up_age_50_plus'])} catch-up, for a total of "
            f"{fmt(d25['401k_403b_457_elective_deferral'] + d25['401k_catch_up_age_50_plus'])}. "
            f"There is no special treatment for ages 60-63."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "retirement_contribution_limits", src))

    return pairs


def gen_dpo_hsa(data, years):
    pairs = []

    # DPO 1 – correct 2024 vs. prior year
    if "2024" in years and "2023" in years:
        d24 = data["2024"]["hsa_contribution_limits"]
        d23 = data["2023"]["hsa_contribution_limits"]
        src = source_for("2024", "hsa")
        q = "What is the HSA contribution limit for self-only coverage in 2024?"
        chosen = (
            f"For tax year 2024, the HSA contribution limit for self-only HDHP coverage under "
            f"IRC §223(b)(2)(A) is {fmt(d24['self_only'])}, as published in Rev. Proc. 2023-23. "
            f"{CAVEAT}"
        )
        rejected = (
            f"The HSA contribution limit for self-only coverage in 2024 is {fmt(d23['self_only'])}."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2024, "hsa_contribution_limits", src))

    # DPO 2 – correct 2025 family vs. fabricated number
    if "2025" in years:
        d25 = data["2025"]["hsa_contribution_limits"]
        src = source_for("2025", "hsa")
        q = "What is the HSA family coverage contribution limit for 2025?"
        chosen = (
            f"For tax year 2025, the HSA contribution limit for family HDHP coverage under "
            f"IRC §223(b)(2)(B) is {fmt(d25['family'])}, as published in Rev. Proc. 2024-25 "
            f"and indexed under IRC §223(g). {CAVEAT}"
        )
        rejected = (
            "The HSA family contribution limit for 2025 is $9,000."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "hsa_contribution_limits", src))

    return pairs


def gen_dpo_section_179(data, years):
    pairs = []

    # DPO 1 – correct 2024 vs. wrong order-of-magnitude
    if "2024" in years:
        d24 = data["2024"]["section_179_expensing"]
        src = source_for("2024")
        q = "What is the Section 179 expensing limit for 2024?"
        chosen = (
            f"For tax year 2024, the maximum IRC §179(b)(1) deduction is {fmt(d24['maximum_deduction'])}, "
            f"with a phaseout threshold of {fmt(d24['phaseout_threshold'])}, as published in "
            f"Rev. Proc. 2023-34. {CAVEAT}"
        )
        rejected = (
            "The Section 179 limit for 2024 is $500,000, which has been the standard limit for several years."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2024, "section_179_expensing", src))

    # DPO 2 – correct 2025 vs. stale 2023 figure
    if "2025" in years and "2023" in years:
        d25 = data["2025"]["section_179_expensing"]
        d23 = data["2023"]["section_179_expensing"]
        src = source_for("2025")
        q = "What is the Section 179 expensing limit for 2025?"
        chosen = (
            f"For tax year 2025, the maximum IRC §179(b)(1) deduction is {fmt(d25['maximum_deduction'])}, "
            f"published in Rev. Proc. 2024-40. The phaseout threshold is {fmt(d25['phaseout_threshold'])}. "
            f"{CAVEAT}"
        )
        rejected = (
            f"The Section 179 limit for 2025 is {fmt(d23['maximum_deduction'])}, the same as it has been."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "section_179_expensing", src))

    return pairs


def gen_dpo_estate_gift(data, years):
    pairs = []

    # DPO 1 – correct 2025 gift exclusion vs. stale 2023 amount
    if "2025" in years and "2023" in years:
        d25 = data["2025"]["estate_and_gift_tax"]
        d23 = data["2023"]["estate_and_gift_tax"]
        src = source_for("2025")
        q = "What is the annual gift tax exclusion for 2025?"
        chosen = (
            f"For 2025, the annual gift tax exclusion under IRC §2503(b)(1) is "
            f"{fmt(d25['annual_gift_tax_exclusion'])} per recipient, published in Rev. Proc. 2024-40. "
            f"{CAVEAT}"
        )
        rejected = (
            f"The annual gift tax exclusion is {fmt(d23['annual_gift_tax_exclusion'])} per recipient. "
            f"This has been the limit for several years."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "estate_and_gift_tax", src))

    # DPO 2 – correct 2024 basic exclusion vs. completely wrong
    if "2024" in years:
        d24 = data["2024"]["estate_and_gift_tax"]
        src = source_for("2024")
        q = "What is the estate tax basic exclusion amount for 2024?"
        chosen = (
            f"For 2024, the federal estate and gift tax basic exclusion amount under IRC §2010(c)(3)(A) "
            f"is {fmt(d24['basic_exclusion_amount'])}, published in Rev. Proc. 2023-34. {CAVEAT}"
        )
        rejected = (
            "The federal estate tax basic exclusion amount for 2024 is $5,490,000, consistent with "
            "pre-TCJA historical levels."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2024, "estate_and_gift_tax", src))

    return pairs


def gen_dpo_capital_gains(data, years):
    pairs = []

    # DPO 1 – correct 2024 vs. prior year
    if "2024" in years and "2023" in years:
        d24 = data["2024"]["capital_gains_brackets"]
        d23 = data["2023"]["capital_gains_brackets"]
        src = source_for("2024")
        q = "What is the 0% long-term capital gains threshold for a single filer in 2024?"
        chosen = (
            f"For tax year 2024, the 0% long-term capital gains rate applies to income up to "
            f"{fmt(d24['single']['zero_percent_up_to'])} for single filers under IRC §1(h), "
            f"published in Rev. Proc. 2023-34. {CAVEAT}"
        )
        rejected = (
            f"For single filers in 2024, the 0% long-term capital gains rate applies up to "
            f"{fmt(d23['single']['zero_percent_up_to'])}."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2024, "capital_gains_brackets", src))

    # DPO 2 – correct 2025 MFJ vs. fabricated number
    if "2025" in years:
        d25 = data["2025"]["capital_gains_brackets"]
        src = source_for("2025")
        q = "At what income level does the 20% capital gains rate kick in for married filing jointly in 2025?"
        chosen = (
            f"For tax year 2025, married filing jointly taxpayers pay the 20% long-term capital gains "
            f"rate on income above {fmt(d25['married_filing_jointly']['fifteen_percent_up_to'])} under "
            f"IRC §1(h)(1)(D), published in Rev. Proc. 2024-40. {CAVEAT}"
        )
        rejected = (
            "For married filing jointly in 2025, the 20% capital gains rate applies to income above $500,000."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "capital_gains_brackets", src))

    return pairs


def gen_dpo_eitc(data, years):
    pairs = []

    # DPO 1 – correct 2025 max credit vs. wrong year
    if "2025" in years and "2023" in years:
        d25 = data["2025"]["earned_income_credit"]
        d23 = data["2023"]["earned_income_credit"]
        src = source_for("2025")
        q = "What is the maximum EITC for a family with 3 or more children in 2025?"
        chosen = (
            f"For tax year 2025, the maximum Earned Income Tax Credit for a family with 3 or more "
            f"qualifying children is {fmt(d25['max_credit_3_plus_children'])} under IRC §32, "
            f"published in Rev. Proc. 2024-40. {CAVEAT}"
        )
        rejected = (
            f"The maximum EITC for a family with 3 or more children is {fmt(d23['max_credit_3_plus_children'])}."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "earned_income_credit", src))

    return pairs


def gen_dpo_amt(data, years):
    pairs = []

    # DPO 1 – correct 2025 single exemption vs. prior year
    if "2025" in years and "2024" in years:
        d25 = data["2025"]["amt"]
        d24 = data["2024"]["amt"]
        src = source_for("2025")
        q = "What is the AMT exemption for a single filer in 2025?"
        chosen = (
            f"For tax year 2025, the AMT exemption for single filers under IRC §55(d)(1)(B) is "
            f"{fmt(d25['exemption_single'])}, as published in Rev. Proc. 2024-40. "
            f"The phaseout begins at {fmt(d25['phaseout_single'])} of AMTI. {CAVEAT}"
        )
        rejected = (
            f"The AMT exemption for a single filer in 2025 is {fmt(d24['exemption_single'])}."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "amt", src))

    return pairs


def gen_dpo_foreign_income(data, years):
    pairs = []

    # DPO 1 – correct 2025 vs. wrong number
    if "2025" in years:
        d25 = data["2025"]["foreign_earned_income_exclusion"]
        src = source_for("2025")
        q = "What is the foreign earned income exclusion for 2025?"
        chosen = (
            f"For tax year 2025, the foreign earned income exclusion under IRC §911(b)(2)(D)(i) is "
            f"{fmt(d25)}, published in Rev. Proc. 2024-40. {CAVEAT}"
        )
        rejected = (
            "The foreign earned income exclusion for 2025 is $107,600, the figure from several years ago."
        )
        pairs.append(make_dpo(q, chosen, rejected, 2025, "foreign_earned_income_exclusion", src))

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    with open(REF_FILE) as f:
        ref = json.load(f)

    data  = ref["tax_years"]
    years = list(data.keys())   # ["2023", "2024", "2025"]

    # ── Generate SFT examples ─────────────────────────────────────────────────
    sft_examples = []
    sft_examples += gen_standard_deduction(data, years)
    sft_examples += gen_retirement(data, years)
    sft_examples += gen_hsa(data, years)
    sft_examples += gen_section_179(data, years)
    sft_examples += gen_estate_gift(data, years)
    sft_examples += gen_amt(data, years)
    sft_examples += gen_eitc(data, years)
    sft_examples += gen_capital_gains(data, years)
    sft_examples += gen_foreign_income_exclusion(data, years)
    sft_examples += gen_qbi(data, years)
    sft_examples += gen_social_security(data, years)

    # ── Generate DPO pairs ────────────────────────────────────────────────────
    dpo_pairs = []
    dpo_pairs += gen_dpo_standard_deduction(data, years)
    dpo_pairs += gen_dpo_retirement(data, years)
    dpo_pairs += gen_dpo_hsa(data, years)
    dpo_pairs += gen_dpo_section_179(data, years)
    dpo_pairs += gen_dpo_estate_gift(data, years)
    dpo_pairs += gen_dpo_capital_gains(data, years)
    dpo_pairs += gen_dpo_eitc(data, years)
    dpo_pairs += gen_dpo_amt(data, years)
    dpo_pairs += gen_dpo_foreign_income(data, years)

    # ── Write outputs ─────────────────────────────────────────────────────────
    OUT_SFT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_SFT, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    with open(OUT_DPO, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    # ── Statistics ────────────────────────────────────────────────────────────
    print(f"SFT examples written : {len(sft_examples):>4}  → {OUT_SFT}")
    print(f"DPO pairs written    : {len(dpo_pairs):>4}  → {OUT_DPO}")

    # Per-topic breakdown for SFT
    from collections import Counter
    topic_counts = Counter(ex["metadata"]["topic"] for ex in sft_examples)
    print("\nSFT breakdown by topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic:<40} {count}")

    topic_counts_dpo = Counter(p["metadata"]["topic"] for p in dpo_pairs)
    print("\nDPO breakdown by topic:")
    for topic, count in sorted(topic_counts_dpo.items()):
        print(f"  {topic:<40} {count}")

    return sft_examples, dpo_pairs


if __name__ == "__main__":
    main()
