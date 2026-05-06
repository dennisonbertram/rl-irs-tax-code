# GPT-5.4 (o3) Fourth Review: IRS Tax Code Training Pipeline

*Generated: 2026-03-29*

Training Pipeline Review – Round 4  
MLX 0.31.1 / Python 3.14  

================================================================
CRITICAL FINDINGS
================================================================
1. save_lora_weights ─ _get_model_items fallback is broken  
   • Location: scripts/train_dpo.py & scripts/train_grpo.py  
   • Details:  
     ```
     result = fn(model.parameters())
     for k,v in result:      # assumes result is iterable of pairs
     ```  
     mlx.utils.tree_flatten **returns a 2-tuple** `(leaves, treedef)`.  
     When tree_flatten_items is absent (or renamed again) the fallback
     path will iterate over the tuple → the second element is a
     PyTreeDef, causing “too many values to unpack” or a silent skip of
     LoRA parameters.  Requirement 1 (“handle all MLX versions”) is
     therefore NOT met.  
   • Fix:  
     ```
     flat = result[0] if isinstance(result, tuple) else result
     ```  
     plus a type-check that each element is a 2-tuple (k,v).

2. Same bug in save_lora_weights helper in **two** scripts → must be
   corrected everywhere (train_dpo.py, train_grpo.py, any copies).

================================================================
HIGH FINDINGS
================================================================
None (other than propagation of the above CRITICAL bug).

================================================================
MEDIUM FINDINGS
================================================================
1. GRPO training – recompiles every step  
   ```
   loss, grads = nn.value_and_grad(policy_model, loss_fn)()
   ```  
   value_and_grad is JIT-compiled; recreating it each iteration incurs
   unnecessary compile overhead (~0.3-0.5 s/step on M-chips) and memory
   churn.  Define it **once** outside the loop, or at least cache it.

2. GRPO training – no gradient clipping  
   DPO loop clips (`clip_grad_norm`); GRPO loop does not.  With group
   sampling (temperature 0.8) occasional very long completions explode
   the KL term, producing NaNs after a few hundred steps.  Recommend
   identical clipping (`max_norm=1.0`).

================================================================
LOW FINDINGS
================================================================
1. Repeated copy-paste of _get_model_items / save_lora_weights across
   files – easy to drift again.  Consider a single utils module.

2. tokenizer.pad_token_id missing → hard-code 0.  For some Qwen
   variants id 0 is actually “<unk>”.  Safer: fall back to
   `tokenizer.eos_token_id`.

3. outputs/sft/adapters/adapter_config.json still shows
   `"num_layers": 16`; script default is now 24.  Probably an old run
   but will mis-lead downstream code that trusts the file.

4. Modelfile example in outputs/ still references `model-q8.gguf`
   whereas export script now produces `model-q6_k.gguf`.  Cosmetic but
   confusing for users.

================================================================
CONCLUSION
================================================================
Because of the CRITICAL bug in _get_model_items fallback, the pipeline
will crash (or worse, silently save full-model 6 GB tensors) on MLX
versions that lack `tree_flatten_items`.  The pipeline is therefore **NOT
production-ready** in its current state.

Fix the flatten fallback in every copy of save_lora_weights; after that,
rerun quick integration tests on an MLX build **without**
`tree_flatten_items` to confirm robustness.
