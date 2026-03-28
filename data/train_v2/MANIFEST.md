# Training Data Manifest — v2 (Grounded Only)

**Created**: 2026-03-27
**Random seed**: 42 (deterministic, reproducible)
**Policy**: All splits contain ONLY citation-validated, grounded data. No template-generated data.

---

## Why v2?

The existing `data/processed/train/` splits were found to contain mixed or exclusively
template-generated data with an estimated 15-30% hallucination rate:

| File | Lines | Source | Status |
|---|---|---|---|
| `processed/train/sft.jsonl` | 24,840 | `sft_train.jsonl` (old) | OLD — 100% template, no metadata, hallucinated |
| `processed/train/train.jsonl` | 15,218 | `grounded_sft_full.jsonl` | GROUNDED — subset |
| `processed/train/valid.jsonl` | 1,691 | `grounded_sft_full.jsonl` | GROUNDED — subset |
| `processed/train/dpo.jsonl` | 1,311 | `grounded_dpo_full.jsonl` | GROUNDED — 90% subset |
| `processed/train/dpo_v2.jsonl` | 1,225 | `grounded_dpo_full.jsonl` | GROUNDED — 90% subset |
| `processed/train/grpo.jsonl` | 16,909 | `grounded_sft_full.jsonl` | GROUNDED — prompts only |
| `processed/eval/sft.jsonl` | 2,760 | `sft_train.jsonl` (old) | OLD — 100% template |
| `processed/eval/dpo.jsonl` | 1,018 | `dpo_train.jsonl` (old) | OLD — template, no metadata |
| `processed/eval/grpo.jsonl` | 2,689 | `grpo_train.jsonl` (old) | OLD — template prompts |

The training scripts (`train_sft.py`, `train_dpo.py`, `train_grpo.py`) load from
`data/processed/train/` by default. `train_sft.py` specifically loads `train.jsonl` +
`valid.jsonl` (the grounded subset). `train_dpo.py` loads `dpo.jsonl` (grounded).
`train_grpo.py` loads `grpo.jsonl` (grounded prompts).

However, `processed/train/sft.jsonl` (the large 24,840-line file) and all `eval/` files
are old template data. This manifest documents the clean v2 splits created to replace them.

---

## Grounded Source Files (Ground Truth)

| File | Records | Description |
|---|---|---|
| `processed/grounded_sft_full.jsonl` | 16,909 | SFT data grounded in IRC statutory text. Each record has `metadata.source_section`, `metadata.grounded=true`. Answers are derived directly from IRC text, no hallucination. |
| `processed/grounded_dpo_full.jsonl` | 1,718* | DPO preference pairs grounded in IRC. Chosen = correct statutory text. Rejected = subtly wrong version with a specific introduced error documented in `metadata.error_introduced`. |
| `processed/onpolicy_dpo_v2.jsonl` | 86 | On-policy DPO pairs: chosen = correct grounded answer, rejected = actual model output (hallucinated). High signal for correcting specific model failure modes. |

*1 record skipped due to malformed JSON (line break in middle of record).

---

## v2 Splits

### `train_v2/sft.jsonl` — SFT Training Set

- **Source**: `grounded_sft_full.jsonl`
- **Split**: 90% train
- **Records**: 15,218
- **Format**: `{"messages": [...], "metadata": {"source_section": "IRC §X", "grounded": true, ...}}`
- **Validation**: 100% have `metadata.grounded=true`; 0 overlap with eval set

### `eval_v2/sft.jsonl` — SFT Evaluation Set

- **Source**: `grounded_sft_full.jsonl`
- **Split**: 10% held-out
- **Records**: 1,691
- **Format**: same as train
- **Validation**: 0 overlap with train set (confirmed)

### `train_v2/dpo.jsonl` — DPO Training Set

- **Source**: `grounded_dpo_full.jsonl` (1,718) + `onpolicy_dpo_v2.jsonl` (86) = 1,804 combined, shuffled, 90% split
- **Records**: 1,623
- **Format**: `{"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}`
- **Note**: `grounded_dpo_full` records include `metadata.error_introduced` describing the synthetic error. `onpolicy_dpo_v2` records have `chosen` = correct answer, `rejected` = model hallucination.

### `eval_v2/dpo.jsonl` — DPO Evaluation Set

- **Source**: same combined pool, 10% held-out
- **Records**: 181
- **Format**: same as train

### `train_v2/grpo.jsonl` — GRPO Training Prompts

- **Source**: Prompts extracted from `grounded_sft_full.jsonl` (user turn of each conversation)
- **Records**: 15,218
- **Format**: `{"prompt": "...", "expected_section": "IRC §X"}`
- **Note**: `expected_section` enables citation-accuracy reward scoring in `grpo_reward.py`. These are the same questions as the SFT train set, which is intentional — GRPO uses them as RL exploration prompts, not to copy answers.

### `eval_v2/grpo.jsonl` — GRPO Evaluation Prompts

- **Source**: prompts from the 10% held-out SFT eval records
- **Records**: 1,691
- **Format**: same as train

---

## Counts Summary

| File | Records | Notes |
|---|---|---|
| `train_v2/sft.jsonl` | 15,218 | |
| `train_v2/dpo.jsonl` | 1,623 | |
| `train_v2/grpo.jsonl` | 15,218 | |
| `eval_v2/sft.jsonl` | 1,691 | |
| `eval_v2/dpo.jsonl` | 181 | |
| `eval_v2/grpo.jsonl` | 1,691 | |

---

## Training Script Integration

The training scripts load from `data/processed/train/` by default. To use these v2 splits,
pass the data path explicitly:

```bash
# SFT
python scripts/train_sft.py
# (train_sft.py reads from data/processed/train/train.jsonl + valid.jsonl by default)
# Override: copy train_v2/sft.jsonl to processed/train/train.jsonl (after backing up)
# or update DATA_DIR in train_sft.py to point to data/train_v2/

# DPO
python scripts/train_dpo.py --data data/train_v2/dpo.jsonl

# GRPO
python scripts/train_grpo.py --data data/train_v2/grpo.jsonl
```

**Note**: `train_sft.py` does not accept a `--data` flag — it hardcodes `data/processed/train/`
and requires `train.jsonl` + `valid.jsonl` filenames. To use v2 SFT splits, either:
1. Update `DATA_DIR` in `train_sft.py` (not allowed per task contract), or
2. Symlink/copy: `cp data/train_v2/sft.jsonl data/processed/train/train.jsonl`
   and `cp data/eval_v2/sft.jsonl data/processed/train/valid.jsonl`

---

## Backup Location

All previous `data/processed/train/` files are backed up at `data/train_old_backup/`:

| Backup File | Original | Lines |
|---|---|---|
| `sft_old.jsonl` | `processed/train/sft.jsonl` | 24,840 — OLD template data |
| `train_old.jsonl` | `processed/train/train.jsonl` | 15,218 — grounded subset |
| `valid_old.jsonl` | `processed/train/valid.jsonl` | 1,691 — grounded subset |
| `dpo_old.jsonl` | `processed/train/dpo.jsonl` | 1,311 — grounded subset |
| `dpo_v1_old.jsonl` | `processed/train/dpo_v1.jsonl` | 9,163 — OLD template DPO |
| `dpo_v2_old.jsonl` | `processed/train/dpo_v2.jsonl` | 1,225 — grounded subset |
| `grpo_old.jsonl` | `processed/train/grpo.jsonl` | 16,909 — grounded prompts |
| `grpo_v2_backup_old.jsonl` | `processed/train/grpo_v2_backup.jsonl` | 16,909 — grounded prompts |
| `train_v1_old.jsonl` | `processed/train/train_v1.jsonl` | 24,840 — OLD template data |
| `valid_v1_old.jsonl` | `processed/train/valid_v1.jsonl` | 2,760 — OLD template data |
