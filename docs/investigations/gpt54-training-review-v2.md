# GPT-5.4 (o3) Second Review: IRS Tax Code Training Pipeline

*Generated: 2026-03-29*

SECOND REVIEW FINDINGS  
Legend – impact on a *production* training run  
CRITICAL   = will crash / corrupt model / waste $  
HIGH      = severe quality or safety degradation, very hard to catch later  
MEDIUM    = noticeable quality / reproducibility issue, but training will run  
LOW       = cosmetic / minor / future-tech-debt

────────────────────────────────────────────────────────
A. REGRESSIONS / INCORRECT-OR-INCOMPLETE FIXES
────────────────────────────────────────────────────────
1. GRPO training crashes at first backward pass ‑- wrong
   value_and_grad invocation                                    CRITICAL
   train_grpo.py lines ~730:
       loss, grads = nn.value_and_grad(policy_model, loss_fn)(policy_model)
   • value_and_grad(model, fn) already “binds” *model*; the
     returned callable expects **zero** args here.
   • The extra policy_model argument raises:
       TypeError: … takes 0 positional arguments but 1 given

2. save_lora_weights still not version-safe                       HIGH
   (train_dpo.py & train_grpo.py)
   • Branch 1 (`tree_flatten`) is picked up on mlx ≥0.17, but
     it now returns *(leaves, treedef)*, **not** (key, value)
     pairs – the dict-comp `for k, v in all_params` crashes:
        ValueError: too many values to unpack
   • Branch 2 (tree_flatten_items) is never reached on
     current mlx releases, so this will trigger at the first
     adapter checkpoint save.

3. GRPO adapter metadata is never written                         HIGH
   • train_grpo.py writes *.safetensors* but **does not write**
     adapter_config.json after training.
   • Down-stream scripts (export_to_ollama, future finetuning,
     evaluation) therefore read a stale file (example shows
     scale=1.0, dropout=0.0) ⇒ scale / dropout mismatch,
     incorrect LoRA fan-in during fuse or further RL.

4. DPO adapter config still top-level scale/rank                  MEDIUM
   Intended fix was nested `"lora_parameters":{...}` but the
   sample adapter_config.json shows:
        "scale": 1.0,
        "dropout": 0.0
   train_grpo.py has a fallback so it *runs*, but the file
   format is now inconsistent across stages.

5. tree_flatten guard uses ImportError, not AttributeError        MEDIUM
   On mlx <0.17 tree_flatten exists but behaviour is old-style;
   the try/except never fires. Safer pattern:
       try:  from mlx.utils import tree_flatten_items
       except ImportError: from mlx.utils import tree_flatten

────────────────────────────────────────────────────────
B. NEW ISSUES INTRODUCED BY THE FIXES
────────────────────────────────────────────────────────
6. Deterministic but **identical** epoch shuffles in DPO          LOW
   batch_iterator recreates RNG with the same seed every time
   StopIteration is hit ⇒ order of batches repeats exactly.
   Not fatal, but hurts generalisation.

7. Evaluation false negatives for sections with subsections       LOW
   score_response compares expected list entry "1(h)" to
   extract_irc_sections() output "1". Mismatch under-scores the
   model.  (Pre-existing but still unfixed.)

────────────────────────────────────────────────────────
C. FIRST-ROUND ISSUES THAT ARE *STILL* UNADDRESSED
────────────────────────────────────────────────────────
8. train_sft CLI flag --lora-rank is accepted but silently
   ignored (not forwarded to mlx_lm.lora)                         LOW

9. assemble_v5_dataset copies SFT files to train.jsonl/valid.jsonl
   but **train_sft default path is data/processed/train/**        LOW
   Out-of-the-box run will complain “missing train.jsonl”.

────────────────────────────────────────────────────────
D. PIPELINE READINESS
────────────────────────────────────────────────────────
• Two CRITICAL runtime errors (items 1 & 2) will break GRPO and
  DPO training as soon as an adapter checkpoint is attempted.
• Metadata write gap (item 3) will silently poison any later
  stage that relies on LoRA scale / dropout.
• Therefore the end-to-end pipeline is **NOT production-ready**.

Fix the three high-impact issues first, rerun a short smoke
training (SFT→DPO→GRPO 1–2 steps each), and verify that:
  – checkpoints save successfully
  – adapter_config.json is emitted with correct nested
    lora_parameters {rank, scale, dropout} and num_layers
  – export_to_ollama reads the same values

If those pass, the remaining MEDIUM/LOW items can be deferred.