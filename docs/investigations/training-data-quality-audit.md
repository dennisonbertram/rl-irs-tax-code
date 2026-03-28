# Training Data Quality Audit

**Date:** 2026-03-27
**Scope:** Full audit of training data pipeline, sources, known quality issues, and data volumes.

---

## 1. Data Generation Pipeline Overview

### How Training Data Is Generated

All training data is **LLM-generated** (GPT-4o-mini) from parsed IRC/CFR source text. There is no manually curated data.

**Pipeline stages:**

1. **Raw Source Parsing:** XML files (`data/raw/irc/usc26.xml` at 53MB, `data/raw/cfr/cfr_title26.xml` at 84MB) are parsed into JSONL section files:
   - `data/processed/irc_sections.jsonl` -- 2,113 IRC sections
   - `data/processed/cfr_sections.jsonl` -- 6,149 CFR sections
   - Total: 8,262 parsed sections

2. **Grounded SFT Generation** (`scripts/generate_grounded_data.py`):
   - Feeds each IRC section's text + cross-referenced sections into GPT-4o-mini
   - Asks for 9 diverse Q&A pairs per section
   - Applies cross-section citation validation (Fix 1) to discard answers citing wrong sections
   - Injects TCJA amendment notices for 9 modified sections (Fix 2)
   - Adds inflation adjustment notes (Fix 3)
   - Output: `data/processed/grounded_sft_full.jsonl`

3. **DPO Data Generation** (two sources):
   - **Synthetic hard negatives:** GPT-4o-mini takes correct answers and introduces exactly one subtle error (wrong subsection, wrong dollar amount, inverted exception). Output: `data/processed/grounded_dpo_full.jsonl`
   - **On-policy DPO** (`scripts/generate_onpolicy_dpo.py`): Queries the current fine-tuned model via Ollama, compares its answers to ground truth, keeps cases where the model is meaningfully wrong. Output: `data/processed/onpolicy_dpo_v2.jsonl`

4. **GRPO Data:** Prompts-only format used for reward-based training. The reward function (`scripts/grpo_reward.py`) scores responses at training time.

### Older (Non-Grounded) Data

An older pipeline (`scripts/generate_training_data.py`, referenced in docs) produced the original data using template-based generation -- mechanical paraphrases of truncated source text with vague template rejections for DPO. This data is still present:
- `data/processed/sft_train.jsonl` -- 27,600 examples (no metadata, no grounding verification)
- `data/processed/dpo_train.jsonl` -- 10,181 pairs (rejected answers are generic "consult a professional" templates, not hard negatives)
- `data/processed/grpo_train.jsonl` -- 26,899 prompts only

---

## 2. Data Volumes

| File | Lines | Description |
|------|-------|-------------|
| **Grounded SFT (new)** | | |
| `grounded_sft_full.jsonl` | 16,909 | GPT-4o-mini generated, citation-validated |
| `grounded_dpo_full.jsonl` | 1,719 | Hard-negative DPO pairs |
| `onpolicy_dpo_v2.jsonl` | 86 | On-policy model errors |
| **Old (template-based)** | | |
| `sft_train.jsonl` | 27,600 | No metadata, no grounding |
| `dpo_train.jsonl` | 10,181 | Vague template rejections |
| `grpo_train.jsonl` | 26,899 | Prompts only |
| **Active training splits** | | |
| `train/sft.jsonl` | 24,840 | |
| `train/dpo.jsonl` / `dpo_v1.jsonl` / `dpo_v2.jsonl` | 1,311 / 9,163 / 1,225 | Multiple DPO versions |
| `train/grpo.jsonl` | 16,909 | |
| `train/train.jsonl` | 15,218 | Combined training set |
| **Eval splits** | | |
| `eval/sft.jsonl` | 2,760 | |
| `eval/dpo.jsonl` | 1,018 | |
| `eval/grpo.jsonl` | 2,689 | |

**Key observation:** It is unclear which combination of old vs. new data was used for the most recent training runs. The `train/` directory contains files that may mix both generations.

---

## 3. Data Sources

### IRC (Internal Revenue Code)
- **Source file:** `data/raw/irc/usc26.xml` (53MB) -- Title 26 of the US Code
- **Parsed into:** 2,113 sections in `irc_sections.jsonl`
- **Coverage:** All of Title 26 including obscure provisions (alcohol/tobacco taxes, repealed sections)

### CFR (Code of Federal Regulations)
- **Source file:** `data/raw/cfr/cfr_title26.xml` (84MB) -- Title 26 Treasury Regulations
- **Parsed into:** 6,149 sections in `cfr_sections.jsonl`
- **Usage:** The old template-based pipeline includes CFR sections. The new grounded pipeline (`generate_grounded_data.py`) **only processes IRC sections** (filters for `source == "IRC"`), meaning 6,149 CFR sections are unused in the grounded data.

### What Is Missing
- **IRS Revenue Procedures and Revenue Rulings** -- contain inflation-adjusted amounts (current standard deduction, contribution limits, etc.)
- **IRS Publications** (Pub 17, Pub 590-A, etc.) -- contain practical interpretations
- **Tax Court decisions** -- interpretive authority
- **TCJA full text** -- only 9 sections have manual TCJA annotations

---

## 4. Known Factual Errors and Quality Issues

### 4.1 The Standard Deduction Hallucination ($135,000)

**Status:** This is a **model generation issue, NOT a training data issue.**

Evidence:
- The grounded SFT data for Section 63 contains 604 examples. Sampled examples correctly state the statutory definition ("gross income minus the standard deduction") without specific dollar amounts.
- The IRC source text for Section 63 contains base statutory amounts ($3,000 / $4,400 / $6,000 for pre-TCJA, and references to "200 percent of the dollar amount" formulas) -- NOT the inflation-adjusted amounts taxpayers actually use.
- The TCJA annotation for Section 63 correctly notes 2018 amounts of $12,000/$24,000/$18,000.
- The $135,000 figure does not appear anywhere in the training data.
- **Root cause:** The base model (Qwen 2.5 3B) hallucinates this number from its pretraining data. The fine-tuning does not reliably correct it because the training data uses statutory base amounts and formulas, not the specific inflation-adjusted figures that users ask about.

**Impact:** HIGH. Users asking "What is the standard deduction?" get a wildly wrong number. The training data cannot fix this because the IRC source text doesn't contain current-year inflation-adjusted amounts.

### 4.2 Section 179 Hallucination ($11,500)

The v3 model outputs $11,500 as the Section 179 maximum expense instead of the correct ~$1.16M (inflation-adjusted) or $2,500,000 (statutory base per the IRC text).

- The training data correctly contains $2,500,000 from the IRC source text
- The $11,500 figure appears to be a model hallucination, possibly from the v2/v3 reward optimization over-indexing on citation format at the expense of factual accuracy
- v1 said $500,000 (old statutory amount), v2 said $1,080,000 (correct inflation-adjusted), v3 regressed to $11,500

**Impact:** HIGH. Demonstrates that GRPO training can **degrade** factual accuracy while improving citation formatting.

### 4.3 Citation Accuracy Metric Is Broken

**All three model versions score 0-4% on citation accuracy in formal evaluation**, despite the models clearly citing sections in their responses.

Root cause: The evaluation regex in `scripts/evaluate.py` uses `re.search(rf"\b{re.escape(sec)}\b", response)` which looks for bare numbers like `179` as word boundaries. However, the formal evaluation script at `outputs/evaluation/formal_50q_eval.txt` shows slightly different numbers (2-4%), suggesting a separate eval script was used. The GRPO reward function's citation regex is different from the eval script's regex.

**Impact:** MEDIUM. Cannot reliably measure whether training is improving citation behavior. The reward function (0.40 weight on citation accuracy) and the evaluation script use different citation detection logic, making it impossible to correlate training rewards with eval scores.

### 4.4 Grounded SFT Data Quality (Validated)

Per the manual validation report (`docs/testing/grounded_data_quality_validation.md`):
- **24 SFT pairs manually validated:** 75% fully grounded, 25% minor issues, 0% hallucinations
- **10 DPO pairs validated:** 100% correct chosen answers, 100% correct error injection
- **Common issues in the 25% with minor problems:**
  1. TCJA/post-2017 omissions (Section 170: 50% vs 60% charitable deduction)
  2. Dropped cross-references (Section 1221 missing Section 167 ref)
  3. One source section mismatch (Section 1092 answer generated from Section 1221 context)
  4. Statutory base amounts vs. inflation-adjusted amounts

### 4.5 Old Template-Based Data Quality

The old `sft_train.jsonl` (27,600 examples) has these known problems per `docs/investigations/rag-grounded-training-data-research.md`:
- **No citation grounding** -- answers cite sections without verifying subsection existence
- **Mechanical paraphrases** -- truncated source text wrapped in boilerplate, not genuine Q&A
- **Template DPO rejections** -- "consult a tax professional" instead of realistic hard negatives
- **No metadata** -- no `source_section` or grounding flags
- **Estimated 15-30% hallucination rate** (vs 0% in grounded data)
- Teaches the model to **regurgitate** rather than **reason**

### 4.6 Reward Function Design Issues

The GRPO reward function (`scripts/grpo_reward.py`) scores on:
- Citation format (0.30) -- counts citations regardless of correctness
- Citation accuracy (0.40) -- checks if expected section number is cited
- Length/detail (0.30) -- 200-1500 chars ideal
- Vague language penalty (-0.30)

**Problems:**
1. **No factual accuracy component.** The reward function cannot verify that dollar amounts, percentages, or dates are correct. A response citing the right section with completely wrong numbers scores identically to a correct response.
2. **Citation accuracy only checks section numbers**, not subsections. Citing "Section 179" when the answer is about Section 179 gets full marks regardless of whether the content is right.
3. **Length reward incentivizes verbosity** over precision.
4. **`batch_reward` does not pass `expected_section`**, so when called from training, citation accuracy always returns 0.5 (neutral) -- the 0.40 weight on citation accuracy is partially wasted.

### 4.7 Topic Coverage Gaps

The grounded data covers all 2,113 IRC sections but with equal weight. This means:
- **Obscure provisions overrepresented:** Alcohol/tobacco tax (Sections 5001-5872), repealed sections, and procedural sections get the same 9 Q&A pairs as critical sections like 401(k), 1031, or 199A
- **High-traffic practical topics underrepresented:** Standard deduction mechanics, W-2 income, filing status rules, child tax credit, earned income credit
- **CFR regulations completely absent** from grounded data (6,149 sections parsed but not used)
- **No IRS Publication content** -- practical guidance and current-year numbers are missing entirely

---

## 5. Evaluation Results Summary

### 5-Question Qualitative Eval (v1 vs v2 vs v3)

| Question | v1 | v2 | v3 |
|----------|----|----|-----|
| Standard deduction | 0 ($135K wrong) | 1 ($13,850) | 1 ($13,850) |
| Section 179 max | 0.5 ($500K outdated) | 1 ($1.08M) | 0 ($11,500 hallucinated) |
| Section 21 qualifying individual | 0 | 0.5 | 0.5 |
| Section 6662 penalty | 1 | 1 | 1 |
| Section 7701 partnership | 1 | 1 | 1 |
| **Total** | **2.5/5** | **4.5/5** | **3.5/5** |

### 50-Question Formal Eval

| Model | Citation Accuracy | Fact Match |
|-------|------------------|-----------|
| v1 | 4.1% (2/49) | 100% (49/49) |
| v2 | 2.0% (1/50) | 96.0% (48/50) |
| v3 | 4.2% (2/48) | 95.8% (46/48) |

**Key findings:**
- Citation accuracy metric is unreliable (broken regex)
- Fact match **decreased** from v1 to v3, suggesting GRPO training trades factual accuracy for citation formatting
- v2 appears to be the best model overall despite v3 having more training

---

## 6. Root Cause Analysis

### Why the model hallucinates dollar amounts

The IRC statute contains **base statutory amounts** (e.g., $3,000 standard deduction, $2,500,000 Section 179 limit) that are adjusted annually for inflation. The training data faithfully reproduces these base amounts. But users ask about **current-year amounts** (e.g., "$14,600 standard deduction for 2024"). The model has no training signal for current amounts, so it falls back to pretraining knowledge, which may be wrong or outdated.

### Why citation accuracy appears broken

Three different citation detection systems exist:
1. `evaluate.py`: Uses `re.search(rf"\b{re.escape(sec)}\b", response)` -- matches bare numbers
2. `grpo_reward.py`: Uses `IRC_CITATION_PATTERN` and `SECTION_PATTERN` -- more sophisticated
3. `generate_onpolicy_dpo.py`: Uses yet another pattern set

These are inconsistent, leading to conflicting accuracy measurements.

### Why GRPO training degrades factual accuracy

The reward function (0.40 citation accuracy + 0.30 citation format + 0.30 length) contains **no factual verification**. GRPO optimizes for what is rewarded: citation presence and response length. A confidently wrong answer with good citations scores higher than a short correct answer without citations. The training literally incentivizes hallucination-with-citations over accuracy-without-citations.

---

## 7. Priority Issues (Ranked)

1. **CRITICAL: Reward function has no factual accuracy signal.** GRPO training actively degrades factual accuracy. The $11,500 Section 179 hallucination in v3 is direct evidence.

2. **CRITICAL: Inflation-adjusted dollar amounts are absent from training data.** Users ask about current-year numbers. The IRC source text only has base statutory amounts. IRS Revenue Procedures with current amounts are not in the data pipeline.

3. **HIGH: Old template-based data may still be mixed into training.** The `train/` directory contains files from both old and new pipelines. If old low-quality data is being used, it undermines the grounded data improvements.

4. **HIGH: CFR regulations (6,149 sections) are parsed but unused.** Treasury Regulations provide interpretive guidance that would improve answer quality, especially for complex topics.

5. **MEDIUM: Citation accuracy metric is inconsistent across scripts.** Three different regex systems make it impossible to track improvement reliably.

6. **MEDIUM: `batch_reward` in GRPO does not pass `expected_section`.** The 0.40-weight citation accuracy component defaults to 0.5 (neutral) during actual training, meaning citation accuracy is not being rewarded as intended.

7. **LOW: Equal coverage across all 2,113 sections.** Obscure provisions (alcohol taxes, repealed sections) get the same training weight as high-impact sections.

---

## 8. Recommendations

1. **Add factual verification to the reward function.** Extract dollar amounts, percentages, and dates from the reference answer and check them against the model's response. Even a simple number-overlap score would prevent gross hallucinations like $11,500 for Section 179.

2. **Incorporate IRS Revenue Procedures** for current-year inflation-adjusted amounts. At minimum, add a supplementary dataset with current standard deduction, contribution limits, bracket thresholds, etc.

3. **Audit the `train/` directory** to determine exactly which data files are being used for each training stage. Remove or isolate old template-based data.

4. **Unify citation detection** across reward function, evaluation, and DPO scripts.

5. **Fix `batch_reward` to pass `expected_section`** so the citation accuracy component actually works during GRPO training.

6. **Weight training examples by section importance.** Give more training signal to high-traffic sections (61, 63, 162, 170, 179, 401, 1031) and less to obscure provisions.

7. **Generate grounded data from CFR sections** to cover Treasury Regulation interpretive guidance.
