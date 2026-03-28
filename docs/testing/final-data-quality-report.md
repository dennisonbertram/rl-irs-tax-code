# Final Training Data Quality Report

**Overall Verdict: FAIL** (4/7 checks passing)

Generated: 2026-03-27

---

## Dataset Summary

| File | Records | Bad Lines |
|------|---------|----------|
| sft_train | 59,341 | 0 |
| sft_valid | 6,277 | 0 |
| dpo_train | 1,708 | 0 |
| dpo_valid | 181 | 0 |
| grpo_train | 58,991 | 0 |
| grpo_valid | 6,277 | 0 |

## Check 1: Structural Validation

**Result: PASS**

| File | Structural Errors | Empty Fields | Missing Metadata | Status |
|------|------------------:|-------------:|-----------------:|--------|
| sft_train | 0 | 0 | 0 | PASS |
| sft_valid | 0 | 0 | 0 | PASS |
| dpo_train | 0 | 0 | 0 | PASS |
| dpo_valid | 0 | 0 | 0 | PASS |
| grpo_train | 0 | 0 | 0 | PASS |
| grpo_valid | 0 | 0 | 0 | PASS |

## Check 2: Content Quality Sampling

**Result: PASS** — Quality Score: 91.0%

Sample size: 50 random SFT train records

- PASS: 41
- MARGINAL: 9
- FAIL: 0

**Non-passing examples:**

- [MARGINAL] `What is a pre-change separate attribute of a new loss member?` — no tax concept in answer
- [MARGINAL] `What was the effect of the amendment by Pub. L. 106–519 on IRC Section 903?` — no tax concept in answer
- [MARGINAL] `What is the method of collection for liabilities under IRC Section 6901?` — no tax concept in answer
- [MARGINAL] `What are the requirements for renewing a grant under Treas. Reg. §53.4945-4?` — no tax concept in answer
- [MARGINAL] `What is the definition of 'C-CPI-U' as used in this section?` — no tax concept in answer
- [MARGINAL] `What must be done if the property pledged as security is insufficient to satisfy` — no tax concept in answer
- [MARGINAL] `What is the effective date of the regulations in Treas. Reg. §1.175-3?` — no tax concept in answer
- [MARGINAL] `What is the significance of the 16 FR 9499 reference in Treas. Reg. §301.7621-1?` — no tax concept in answer
- [MARGINAL] `How does a corporation qualify as a terminal railroad corporation under Treas. R` — no tax concept in answer

## Check 3: Source Distribution Analysis

**Result: PASS**

### Source Counts

| Source | Count |
|--------|------:|
| CFR | 41,269 |
| IRC | 17,722 |
| IRS Revenue Procedure | 350 |

### Tier Counts

| Tier | Count |
|------|------:|
| Tier 1 | 38,379 |
| Tier 2 | 19,921 |
| Tier 3 | 1,041 |

### Tier 1 Coverage

- Tier 1 sections in reference: 337
- Tier 1 sections covered: 329
- Coverage: 97.6%

**Tier 1 sections with 0 coverage (first 30):** 1245, 129, 151, 2032A, 280F, 543, 62, 7443

### Top 20 Most Represented Sections

| Section | Count |
|---------|------:|
| Rev. Proc. 2023-34 | 95 |
| Rev. Proc. 2024-40 | 90 |
| Rev. Proc. 2022-38 | 80 |
| IRC §311 | 30 |
| IRC §707 | 30 |
| IRC §2512 | 30 |
| IRC §104 | 30 |
| IRC §7463 | 30 |
| IRC §1256 | 30 |
| IRC §7805 | 30 |
| IRC §7431 | 30 |
| IRC §691 | 30 |
| IRC §677 | 30 |
| IRC §79 | 30 |
| IRC §514 | 27 |
| IRC §471 | 27 |
| IRC §55 | 27 |
| IRC §1035 | 27 |
| IRC §1 | 27 |
| IRC §6046A | 27 |

### Bottom 20 Least Represented Sections

| Section | Count |
|---------|------:|
| IRC §5351 | 1 |
| IRC §4051 | 1 |
| IRC §4225 | 1 |
| IRC §5101 | 1 |
| CFR §1.25E-2 | 1 |
| CFR §1.45U-1-1.45U-2 | 1 |
| IRC §5854 | 1 |
| IRC §1358 | 1 |
| CFR §1.6038A-7 | 1 |
| IRC §5353 | 1 |
| CFR §1.547-3 | 1 |
| CFR §1.987-3 | 1 |
| CFR §1.170A-2 | 1 |
| CFR §1.168A-1 | 1 |
| IRC §5662 | 1 |
| IRC §4701 | 1 |
| CFR §1.401(l)-0 | 1 |
| IRC §5223 | 1 |
| IRC §9005 | 1 |
| IRC §5684 | 1 |

## Check 4: Inflation Amount Verification

**Result: PASS**

- Inflation-adjusted records: 350
- Sample checked: 10
- Mismatches: 0

| Question | Amounts Found | Valid Matches | Status |
|----------|:-------------|:-------------|--------|
| How did the Section 179 expensing limit change from 2023 to 2024? | [1160000, 1220000, 60000, 2890000, 3050000] | [1160000, 1220000, 2890000, 3050000] | PASS |
| What are the long-term capital gains tax brackets for all filing statuses in 202 | [48350, 533400, 96700, 600050, 48350, 300025, 64750, 566700] | [48350, 533400, 96700, 600050, 48350, 300025, 64750, 566700] | PASS |
| What is the maximum Section 179 deduction allowed in 2023? | [1160000, 2890000] | [1160000, 2890000] | PASS |
| What is the federal estate and gift tax basic exclusion amount for 2023? | [12920000] | [12920000] | PASS |
| What are the EITC maximum credit amounts by number of children for 2023? | [600, 3995, 6604, 7430] | [600, 3995, 6604, 7430] | PASS |
| What are the AMT exemption amounts for all filing statuses in 2024? | [85700, 133300, 66650, 609350, 1218700] | [85700, 133300, 66650, 609350, 1218700] | PASS |
| What are the long-term capital gains tax brackets for all filing statuses in 202 | [44625, 492300, 89250, 553850, 44625, 276900, 59750, 523050] | [44625, 492300, 89250, 553850, 44625, 276900, 59750, 523050] | PASS |
| How did the annual gift tax exclusion change from 2023 to 2024? | [17000, 18000, 1000] | [17000, 18000, 1000] | PASS |
| What is the federal estate and gift tax basic exclusion amount for 2024? | [13610000] | [13610000] | PASS |
| What is the standard deduction for single filers in 2025? | [15000] | [15000] | PASS |

## Check 5: Train/Eval Leakage

**Result: FAIL**

| Split | Train | Valid | Leaks | Status |
|-------|------:|------:|------:|--------|
| sft | 59,341 | 6,277 | 1 | FAIL |
| dpo | 1,708 | 181 | 6 | FAIL |
| grpo | 58,991 | 6,277 | 1 | FAIL |

**sft leaking questions (first 3):**
- `What happens if a tax return preparer fails to furnish a copy of the return to the taxpayer?`

**dpo leaking questions (first 3):**
- `What limitations apply to carryforwards according to IRC Section 39?`
- `Does IRC Section 190 apply to expenditures incurred after a certain date?`
- `What constitutes a parachute payment according to IRC Section 280G?`

**grpo leaking questions (first 3):**
- `What happens if a tax return preparer fails to furnish a copy of the return to the taxpayer?`

## Check 6: Duplicate Analysis

**Result: FAIL**

- Total SFT train records: 59,341
- Exact duplicate records: 5,252 (8.85%)
- Near-duplicate records (same first 50 chars): 8,241 (13.89%)

**Example duplicate questions:**
- `What exceptions exist for debt-financed property under IRC Section 514?`
- `Is there a provision for recognizing gain on the contribution of property to a partnership?`
- `What is the general rule regarding gain or loss for corporations under IRC Section 361?`
- `Does the rehabilitation credit apply to buildings used for tax-exempt purposes?`
- `What amendments have been made to IRC Section 7454 regarding foundation managers?`

## Check 7: DPO Quality Check

**Result: WARN**

- Sample size: 20
- Records with issues: 14

**Pairs with issues:**

- `How does the penalty under IRC Section 7268 compare to other tax-related penalti` — chosen and rejected nearly identical
- `What does 'compounded daily' mean in the context of IRC Section 6622?` — chosen and rejected nearly identical
- `When is it permissible to recover distilled spirits from denatured distilled spi` — chosen and rejected nearly identical
- `What is the effective date of the repeal of IRC § 51A?` — chosen and rejected nearly identical
- `What does the term 'noncompliance period' refer to in IRC Section 9707?` — neither chosen nor rejected has legal citation
- `What adjustments must be made when determining total excess distributions?` — chosen much shorter than rejected (may be lower quality)
- `What are the implications of the effective date noted in IRC Section 6039L?` — chosen and rejected nearly identical
- `What general rule applies to group health plans under IRC Section 9815?` — chosen and rejected nearly identical
- `What amendment was made to IRC Section 5503 in 1976?` — chosen and rejected nearly identical
- `What exclusions from gross income are provided under IRC Section 1385?` — chosen and rejected nearly identical
- `What historical context is provided in IRC Section 515?` — chosen and rejected nearly identical
- `How does the treatment of renewal options affect a lessee's tax deductions?` — chosen much shorter than rejected (may be lower quality)
- `How does a taxpayer make an election under Section 179C?` — neither chosen nor rejected has legal citation
- `What conditions may the Secretary impose for registration under IRC Section 4101` — chosen and rejected nearly identical, neither chosen nor rejected has legal citation

## Summary and Recommendations

### Check Results Summary

- PASS: 1. Structural Validation
- PASS: 2. Content Quality Sampling
- PASS: 3. Source Distribution
- PASS: 4. Inflation Amount Verification
- WARN/FAIL: 5. Train/Eval Leakage
- WARN/FAIL: 6. Duplicate Analysis
- WARN/FAIL: 7. DPO Quality

**Overall: FAIL**

### Old Data vs New Data Comparison

| Metric | Old (v1) | New (v3) |
|--------|:--------:|:--------:|
| SFT Train Records | 27,600 | 59,341 |
| SFT Valid Records | ~3,000 | 6,277 |
| Generation Method | Template-based | Grounded IRC + CFR |
| DPO Pairs | unknown | 1,708 train / 181 valid |
| GRPO Records | unknown | 58,991 train / 6,277 valid |
| Inflation Records | 0 | 350 (5x weighted) |
| CFR Coverage | None | ~41,269 records |

### Recommendations

- **Content Quality**: Score 91.0% passes the 70% threshold. Note: 'consult a professional' postscript appears frequently but doesn't render answers invalid.
- **Coverage Gaps**: 8 Tier 1 sections have zero coverage (e.g., 1245, 129, 151, 2032A, 280F). Consider augmenting with targeted generation for these high-priority sections.
- **Leakage**: 8 questions appear in both train and valid splits. Re-split or remove overlapping records.
- **Duplicates**: 8.85% exact duplicate rate exceeds 5% threshold. Run deduplication before final training.
- **DPO**: 70% of sampled pairs have quality warnings. Review DPO pairs where chosen/rejected are too similar.

---

*Report generated by `scripts/validate_final_data.py`*
