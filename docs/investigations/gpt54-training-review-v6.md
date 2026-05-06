# GPT-5.4 (o3) Sixth Review: IRS Tax Code Training Pipeline

*Generated: 2026-03-29*

CRITICAL
1.  Dataset path mismatch breaks DPO / GRPO stages out-of-the-box  
   • assemble_v5_dataset.py now writes v5 files:
        data/v5/dpo_train.jsonl, dpo_valid.jsonl,  
        data/v5/grpo_train.jsonl, grpo_valid.jsonl …  
   • scripts/train_dpo.py still hard-codes  
        DPO_DATA = data/processed/train/dpo.jsonl  
   • scripts/train_grpo.py still hard-codes  
        GRPO_DATA = data/processed/train/grpo.jsonl  

   Unless the user passes ­--data each time, both training scripts abort at
   check_data(), so the documented “SFT → DPO → GRPO” pipeline cannot be run
   end-to-end after the v5 dataset assembly.  This is a pipeline-blocking issue.

HIGH
2.  LoRA “already applied” guard is ineffective → double-conversion risk  
   Functions _apply_lora_if_needed() in both train_dpo.py and train_grpo.py test

        first_layer = next(iter(mdl.model.layers), None)
        if first_layer is not None and hasattr(first_layer, "lora_A"):

   LoRA attributes live on the replaced **Linear** sub-modules, not on the
   top-level transformer block stored in model.layers.  After resuming training
   from a checkpoint that already contains LoRA layers, this guard returns
   False; linear_to_lora_layers() is invoked a second time and attempts to wrap
   LoRALinear objects again.  Depending on mlx_lm’s implementation this either

   • raises an exception (duplicate attribute), or  
   • silently nests wrappers, doubling parameter count and breaking weight
     loading.

   The intended safety check therefore fails; resuming or re-using a model that
   already has LoRA can crash or corrupt the model.

PIPELINE IS **NOT** PRODUCTION-READY until the above CRITICAL/HIGH issues are fixed.
