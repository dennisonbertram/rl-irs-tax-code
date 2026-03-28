# Active Plan: Training Data Quality Improvement

**Goal:** Fix critical training data issues that cause factual hallucinations and reward function failures.

**Milestone:** M-002 — Accurate Training Data v2

## Wave 1 (Parallel — No Dependencies)

### TASK-001: Audit and Clean Training Splits
- Inventory all files in `data/train/`, `data/eval/`, `data/processed/`
- Determine which are old template-based vs new grounded data
- Remove/isolate old low-quality data
- Create clean splits using ONLY grounded data
- **Files:** `data/train/`, `data/eval/`

### TASK-002: Research Current-Year Inflation-Adjusted Amounts
- Find IRS Revenue Procedures for tax year 2024 (Rev. Proc. 2023-34) and 2025 (Rev. Proc. 2024-40)
- Compile a reference JSON dataset of key inflation-adjusted figures:
  - Standard deduction (all filing statuses)
  - §179 expensing limit and phaseout
  - Tax bracket thresholds
  - Contribution limits (401k, IRA, HSA)
  - Estate/gift tax exclusions
  - AMT exemptions
  - Earned income credit thresholds
- **Output:** `data/reference/inflation_adjusted_amounts.json`

### TASK-003: Fix GRPO Reward Function
- Add factual accuracy component: extract numbers from reference answer, check against model response
- Fix `batch_reward` to pass `expected_section`
- Unify citation regex across `grpo_reward.py`, `evaluate.py`, and `generate_onpolicy_dpo.py`
- **Files:** `scripts/grpo_reward.py`, `scripts/evaluate.py`

## Wave 2 (After Wave 1)

### TASK-004: Generate Grounded Data from CFR Sections
- Extend `generate_grounded_data.py` to process CFR sections (currently IRC-only)
- Generate Q&A pairs from the 6,149 CFR regulation sections
- Apply same citation validation and quality checks
- **Depends on:** TASK-001 (clean data pipeline understanding)
- **Note:** Requires OpenAI API calls — cost estimate needed before execution
- **Files:** `scripts/generate_grounded_data.py`, `data/processed/`

### TASK-005: Inject Inflation-Adjusted Amounts
- Using the reference data from TASK-002, create supplementary training examples
- Add current-year dollar amounts for high-impact sections
- Generate SFT pairs that teach the model current amounts with proper caveats
- **Depends on:** TASK-001, TASK-002

### TASK-006: Section Importance Weighting
- Define tier system: Tier 1 (high-traffic), Tier 2 (moderate), Tier 3 (low)
- Tier 1 sections: 1, 11, 21, 24, 25A, 61, 63, 67, 68, 83, 101, 104, 121, 125, 132, 162, 163, 164, 165, 167, 168, 170, 179, 197, 199A, 213, 217, 219, 263, 267, 351, 368, 401, 402, 403, 408, 409A, 414, 415, 421, 422, 453, 469, 501, 509, 512, 1001, 1014, 1031, 1033, 1202, 1221, 1231, 1245, 7701
- Implement upsampling for Tier 1, normal for Tier 2, downsampling for Tier 3
- **Depends on:** TASK-001

## Wave 3 (Final Assembly)

### TASK-007: Assemble Final Clean Training Splits
- Combine: clean grounded IRC data + CFR data + inflation-adjusted supplements
- Apply importance weighting
- Generate final `train/sft.jsonl`, `train/dpo.jsonl`, `train/grpo.jsonl`
- Split train/eval (90/10)
- **Depends on:** TASK-004, TASK-005, TASK-006

### TASK-008: Validate Data Quality
- Sample and manually verify 50 examples from new training data
- Run statistics: topic coverage, section distribution, answer length
- Compare old vs new data volumes and quality metrics
- **Depends on:** TASK-007
