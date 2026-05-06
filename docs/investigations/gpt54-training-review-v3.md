# GPT-5.4 (o3) Third Review: IRS Tax Code Training Pipeline

*Generated: 2026-03-29*

TRAINING PIPELINE REVIEW ‑ ROUND 3  
Qwen-2.5-3B tax-law fine-tuning on MLX

============================================================
SUMMARY
• 2 new HIGH-severity flaws were introduced.
• 2 medium / 2 low issues also detected.
• Several previous fixes verified as correct.

PIPELINE IS **NOT** PRODUCTION-READY.

============================================================
DETAILED FINDINGS
------------------------------------------------------------
1. HIGH — tree_flatten_items fallback logic breaks on MLX < 0.17  
   File(s): train_dpo.py save_lora_weights(), train_grpo.py save_lora_weights()  
   Code path when `mlx.utils.tree_flatten_items` is missing:
```python
from mlx.utils import tree_flatten as _flatten
all_params = _flatten(model.parameters())      # returns (leaves, treedef)
lora_params = {k: v for k, v in all_params if "lora" in k}
```
   • On MLX <= 0.16 `tree_flatten` returns **tuple(leaves, treedef)** where
     each leaf is a value, not a (key,value) pair.  
   • The dict-comp therefore raises “not enough values to unpack”.  
   • Result: checkpoints are never written → training cannot resume / export.

   Fix: emulate old behaviour:
```python
try:
    from mlx.utils import tree_flatten_items as flatten_items
    items = flatten_items(model.parameters())           # [(k,v), …]
except ImportError:
    from mlx.utils import tree_flatten                 # (<leaves>, treedef)
    leaves, _ = tree_flatten(model.parameters())
    items = [(p.name, p) for p in leaves]              # or walk with mx.utils.tree_map
```

------------------------------------------------------------
2. HIGH — GRPO adapter_config.json still written with wrong values  
   Focus fix #3 required rank/scale/dropout to mirror live training.  
   Sample file in review package:

```json
"lora_parameters": { "rank": 32, "dropout": 0.0, "scale": 1.0 }
"num_layers": 16
```
   • Code in train_grpo.py hard-codes `dropout: 0.05` and `scale: lora_scale`
     (20.0 when starting from SFT) but actual file shows 0 / 1.0.  
   • File also contains 30+ spurious keys (batch_size, grad_checkpoint …)
     indicating it is **still the original mlx_lm.lora config**, so the
     new overwrite never occurred.  
   • Down-stream scripts (export_to_ollama, evaluation) read stale scale=1.0
     and incorrectly scale LoRA deltas.

   Likely cause:
   • `save_lora_weights()` writes `*.safetensors`; if this I/O fails
     (see bug #1) the subsequent write to adapter_config is skipped by the
     Python interpreter and old file persists.  Once bug #1 is fixed, verify
     that the new minimal JSON is actually flushed.

------------------------------------------------------------
3. MEDIUM — evaluation subsection stripping misses “§” prefixed strings  
   _base_section() uses `re.match(r"(\d+[A-Za-z]?)", s)` which fails when
   the string starts with "§" (e.g. "§ 179").  These appear in several
   datasets.  False negative lowers citation score.

   Fix: `re.search`, or strip leading non-digit chars before regex.

------------------------------------------------------------
4. MEDIUM — GRPO/DPO save_lora_weights silently saves full parameters
   if no “lora” substring is found in keys.  On newest MLX the LoRA keys are
   named `lora_A.weight`, `lora_B.weight`, **but** when LoRA layers are
   fused-to-linear (resume run) the keys change to plain weights without “lora”.
   Current guard:

```python
if not lora_params:
    print("WARNING: No LoRA parameters found, saving all trainable params")
```
   This inflates checkpoint from 7 MB → 6 GB and breaks low-disk systems.

   Mitigation: raise hard error instead, or at least detect fused state and
   skip save.

------------------------------------------------------------
5. LOW — sequence_log_prob in train_dpo.py constructs mx.arange(T)
   where T = `shift_labels.shape[1]`.  If sequence length is 1, `shift_logits`
   is (B,0,V) and indexing with arange(0) triggers “Index arrays must be
   non-empty” on MLX nightly.  Handle edge case by early-return 0-tensor.

------------------------------------------------------------
6. LOW — assemble_v5_dataset: make_grpo_record() may return None,
   but list comprehension in GRPO prompt generation uses walrus operator
   inside comprehension (`g := make_grpo_record(r)`), which is Python 3.8+
   only.  Pipeline README states 3.7 support.  Clarify min Python version
   or rewrite.

------------------------------------------------------------
VERIFIED FIXES FROM PREVIOUS ROUNDS
✓ GRPO value_and_grad closure now zero-arg — no double-binding.  
✓ DPO batch_iterator uses epoch-based seed variation.  
✓ Evaluation expected-section stripping works for “1(h)”, “168(k)”.  
✓ SFT --lora-rank flag correctly writes generated YAML.  
✓ Data-path and inflation split-before-upsample changes intact.

============================================================
ACTION ITEMS
1. Rewrite save_lora_weights() fallback as described; test on MLX 0.16 & 0.17.
2. After #1, confirm GRPO adapter_config.json is overwritten; ensure it
   contains only:
   {num_layers, lora_parameters{rank,scale,dropout}, training, group_size,
    eps_clip, step}.  Remove legacy keys.
3. Extend evaluation _base_section() to strip any leading non-digit chars.
4. Decide policy when LoRA params not found (error vs. large save).
5. Guard sequence_log_prob zero-length path.
6. Update docs or code for Python 3.7 vs walrus operator.

Fix #1 and #2 are blocking; others can be iterative.

============================================================
END OF REVIEW
