# Final Training Splits — Manifest

Generated: 2026-03-28 17:10 UTC
Shuffle seed: 42
Inflation upsample factor: 5x

## File Summary

| File | Records | Description |
| ---- | ------: | ----------- |
| `sft_train.jsonl` | 18,072 | Weighted IRC SFT + 5x inflation SFT, shuffled |
| `sft_valid.jsonl` | 1,691 | Eval SFT split (eval_v2/sft.jsonl) |
| `dpo_train.jsonl` | 1,708 | Grounded IRC DPO + 5x inflation DPO (normalized), shuffled |
| `dpo_valid.jsonl` | 181 | Eval DPO split (eval_v2/dpo.jsonl) |
| `grpo_train.jsonl` | 17,722 | Weighted IRC GRPO (as-is, train_v2/grpo_weighted.jsonl) |
| `grpo_valid.jsonl` | 1,691 | Eval GRPO split (eval_v2/grpo.jsonl) |
| `train.jsonl` | 18,072 | Copy of sft_train.jsonl (train_sft.py compatibility) |
| `valid.jsonl` | 1,691 | Copy of sft_valid.jsonl (train_sft.py compatibility) |

## Source Breakdown

### SFT Train
- Weighted IRC SFT (`train_v2/sft_weighted.jsonl`): 17,722 records (Tier 1 sections 3x upsampled at source)
- Inflation-adjusted SFT (`processed/inflation_adjusted_sft.jsonl`): 70 records × 5x = 350 records

### DPO Train
- Grounded IRC DPO (`train_v2/dpo.jsonl`): 1,623 records
- Inflation-adjusted DPO (`processed/inflation_adjusted_dpo.jsonl`): 17 records × 5x = 85 records
  - Format normalized: prompt/chosen/rejected extracted from message lists

### GRPO Train
- Weighted IRC GRPO (`train_v2/grpo_weighted.jsonl`): 17,722 records (used as-is)

### Validation Splits
- SFT valid: `eval_v2/sft.jsonl` (no data leakage from train set)
- DPO valid: `eval_v2/dpo.jsonl` (no data leakage from train set)
- GRPO valid: `eval_v2/grpo.jsonl` (no data leakage from train set)

## Format Specifications

| Split | Required Keys | Notes |
| ----- | ------------- | ----- |
| SFT   | `messages` (system/user/assistant) | Also exported as train.jsonl / valid.jsonl |
| DPO   | `prompt`, `chosen`, `rejected` | prompt is plain string |
| GRPO  | `prompt`, `expected_section` | prompt-only; expected_section for reward shaping |

## Compatibility

- `train.jsonl` / `valid.jsonl` are copies of SFT files for `train_sft.py` compatibility
- CFR data can be merged later via `scripts/merge_cfr_data.py`

## Notes

- Inflation-adjusted records receive 5x upsampling to correct high-frequency hallucinations
  on standard deduction amounts, contribution limits, and other indexed figures.
- All splits were shuffled with seed=42 for reproducibility.
- Eval splits are held out and NOT included in training data.

## CFR Merge — 2026-03-28 17:48 UTC

- Source: `/Users/dennisonbertram/Develop/rl-irs-tax-code/data/processed/grounded_cfr_sft_deduped.jsonl`
- CFR records (deduped): 45,855
- CFR train split (90%): 41,269
- CFR eval  split (10%): 4,586
- Weight: 1x (no upsampling)

### Updated File Sizes

| File | Records |
| ---- | ------: |
| `sft_train.jsonl` | 59,341 |
| `sft_valid.jsonl` | 6,277 |
| `grpo_train.jsonl` | 58,991 |
| `grpo_valid.jsonl` | 6,277 |
| `train.jsonl` | 59,341 (copy of sft_train) |
| `valid.jsonl` | 6,277 (copy of sft_valid) |

- Split seed: 42  |  Shuffle seed: 42
- Dedup: 283 duplicates removed, 6,118 disclaimers added

## Dedup + Leakage Fix — 2026-03-27

Run by `scripts/dedup_final_splits.py`.

### Changes

| File | Before | After | Removed | Reason |
| ---- | -----: | ----: | ------: | ------ |
| `sft_train.jsonl` | 59,341 | 54,369 | 4,972 | Exact-duplicate questions (non-inflation) |
| `sft_valid.jsonl` | 6,277 | 6,276 | 1 | Train/eval leakage |
| `dpo_valid.jsonl` | 181 | 175 | 6 | Train/eval leakage |
| `grpo_train.jsonl` | 58,991 | 54,019 | 4,972 | Exact-duplicate prompts |
| `grpo_valid.jsonl` | 6,277 | 6,276 | 1 | Train/eval leakage |
| `train.jsonl` | 59,341 | 54,369 | — | Compatibility copy of sft_train.jsonl |
| `valid.jsonl` | 6,277 | 6,276 | — | Compatibility copy of sft_valid.jsonl |

### Invariants Preserved

- `inflation_adjusted` records in `sft_train.jsonl`: **350** (70 unique × 5x — intentional upsample retained)
- `dpo_train.jsonl` and `dpo_valid.jsonl` (train side) unchanged
- All verification assertions passed (zero remaining duplicates, zero remaining leakage)
