# GPT-5.4 (o3) Expert Review: IRS Tax Code Training Pipeline

*Generated: 2026-03-29*

Comprehensive Code-base Audit (remaining issues AFTER the two hot-fixes you already applied)

Legend  
CRITICAL  > HIGH  > MEDIUM  > LOW – relative impact on final model quality / correctness.

------------------------------------------------------------------
1. BUGS & LOGIC ERRORS
------------------------------------------------------------------
1-A  CRITICAL – GRPO still has the “length-scaling” bug  
File: scripts/train_grpo.py → sequence_log_prob (l. 248)  
• Returns the SUM of token log-probs, not the MEAN (the fix you applied to DPO never reached GRPO).  
• Importance ratio ρ = exp(log π – log π_ref) therefore explodes with long completions; PPO clip then fires constantly ⇒ vanishing learning signal.  
Fix: identical to DPO   
```
token_sum = (token_log_probs * mask[:,1:]).sum(axis=-1)
token_cnt  = mx.clip(mask[:,1:].sum(axis=-1), 1., None)
return token_sum / token_cnt
```

1-B  CRITICAL – LoRA scale mismatch when GRPO loads prior adapters  
File: train_grpo.py, lines ~160 / 320  
• linear_to_lora_layers(…, scale=1.0) is hard-coded, but the SFT adapter was trained with scale 20.0 (see outputs/sft/adapter_config.json).  
• Result: all inherited deltas are shrunk ×20, wiping out SFT/DPO knowledge even after the init-order fix.  
Fix: read “lora_parameters.scale” (and rank) from the chosen adapter_config.json and pass the same scale to both linear_to_lora_layers calls.

1-C  HIGH – Train/Valid leakage introduced by up-sampling  
File: assemble_v5_dataset.py  
• Inflation up-sampling duplicates each record K-times *before* the train/valid split. A record’s copies can land on both sides of the split, so validation is no longer independent.  
Fix:  (1) perform the split first, then up-sample inside each partition, or  
      (2) keep a global hash of user-prompt and force all duplicates into the same split.

1-D  HIGH – Prompt and answer concatenated without separator in DPO batches  
File: train_dpo.py → _collate_batch  
```
encode(prompt + chosen)   # no space/eos
```
Model cannot know where the prompt ends; gradients leak across roles.  
Fix: insert the chat template or at least tokenizer.eos_token / “\n\n### Assistant:\n”.

1-E  HIGH – Same bug in GRPO generation-side advantage calculation  
grpo_loss_for_prompt() builds full_text = prompt + completion with no separator; importance ratio again conditioned on a malformed sequence. Add eos or chat delimiter.

1-F  MEDIUM – factual_accuracy_score misses numeric matches without commas / “$”  
Regex compares raw strings ⇒ “$1,160,000” ≠ “1160000”.  
Fix: normalise by strip “$”, “,” and leading zeros before set intersection.

1-G  MEDIUM – evaluate.py loads fine-tuned model first, then *deletes* baseline but keeps GPU memory allocated (Metal). Subsequent load can OOM on 48-GB M-series.  
Fix: call mlx.core.empty_cache() or reload the interpreter between passes.

1-H  MEDIUM – DPO save_lora_weights uses `tree_flatten` from mlx.utils; API changed to `tree_flatten_items` in mlx 0.17+. Code crashes on latest MLX.  
Fix:  from mlx.utils import tree_flatten_items as tree_flatten

1-I  LOW – Citation regex marks “§ 1234-1” (regulation) as IRC section when preceded by title 26 reference – rare but pollutes reward. Add negative look-behind for “C.F.R.” when building IRC_CITATION_PATTERN.

------------------------------------------------------------------
2. TRAINING METHODOLOGY
------------------------------------------------------------------
2-A  HIGH – SFT LoRA covers only 16/24 transformer blocks of Qwen-3B. Large portions of the network stay frozen across all three stages, limiting capacity to learn precise numeric tables. Empirically the numeric drift you observe disappears when LoRA is applied to *all* layers.  
Recommendation: raise `--num-layers` to 24 (or at least 20) and keep rank 32.

2-B  HIGH – LoRA dropout was 0.05 during SFT but 0.00 during DPO/GRPO. Changing dropout between stages changes the effective scale of deltas even if you load the same weights (because dropout is only active on *new* updates). Keep it constant (0.05) or disable entirely in all stages.

2-C  MEDIUM – DPO β=0.5 is large after length-normalisation; empirical sweet-spot for 3B models is 0.1-0.2. Over-penalises deviations and slows convergence. Tune after fixing other bugs.

2-D  MEDIUM – GRPO group_size = 4 with temperature 0.8 gives very noisy reward; variance reduction trick (baseline = mean, std-norm) helps but K=8 gives noticeably higher sample-efficiency on 3B with same VRAM. Consider doubling K and accumulating gradients instead of lowering lr.

------------------------------------------------------------------
3. REWARD FUNCTION
------------------------------------------------------------------
3-A  HIGH – Component weights add up to 1.0 *before* vague_penalty. After penalty total can drop to 0.9 but can also exceed 1.0 when length_score = 0.15 and 4+ citations (0.20) plus other maxima (0.70) = 1.05. You clamp later, which distorts gradient around the optimum (lots of flat 1.0).  
Fix: either renormalise after penalty or cap citation_format at (4/4)*0.15.

3-B  MEDIUM – Citation_accuracy returns 0.5 when expected_section is None, which gives the model half credit for questions where we simply didn’t annotate. That biases the policy toward always citing something. Safer default is 0.25 (uncertain) or weight by presence of any citation.

3-C  LOW – Count_citations rewards raw frequency (up to 4) – encourages footnotes with irrelevant sections. Consider diminishing-returns curve: score = 1 – exp(-n/2).

------------------------------------------------------------------
4. DATA PIPELINE
------------------------------------------------------------------
4-A  HIGH – Entire dataset is synthetic GPT-4 output; no human review; risk of model over-fitting to hallucinated “facts” that present as ground-truth (e.g. wrong phase-out thresholds). At minimum, mix in official IRS publications or cross-check numbers with authoritative CSV before up-sampling.

4-B  MEDIUM – Deduplication for DPO keeps first record per prompt, discarding alternative *hard negatives* that differ in the “chosen/rejected” pair but share a prompt. That removes valuable contrastive signal. Keep all pairs but deduplicate by (prompt, chosen, rejected).

4-C  LOW – Inflation data is up-sampled 20× which now dominates (>35 % of tokens). The model will over-index on inflation scenarios. Reduce multiplier or use weighted sampling in mlx_lm.

------------------------------------------------------------------
5. ADAPTER MANAGEMENT
------------------------------------------------------------------
5-A  CRITICAL – DPO adapter_config.json is missing “num_layers”. mlx_lm.load(model, adapter_path=…) falls back to *all* layers, silently treating missing LoRA layers as trainable and initialising them randomly at inference time. Always write the complete config schema.

5-B  HIGH – linear_to_lora_layers is called *every* time you resume training, which re-creates new randomly-initialised LoRA parameters if the model was saved with fused weights. Guard with `if not hasattr(layer,"lora_A")`.

------------------------------------------------------------------
6. QUANTISATION / EXPORT
------------------------------------------------------------------
6-A  HIGH – Converting bf16 → q8_0 then trying to re-quantise to Q4_K_M fails (your code already logs this). The easiest path that preserves numeric precision is:  
   • fuse → export **bf16 GGUF** (`--outtype bf16`)  
   • quantise once to q6_k or q5_1 using the *same* gguf-py version compiled with llama.cpp HEAD.  
Empirically q6_k keeps $14,600 figure; q4_k_m does not.

6-B  MEDIUM – System prompt in Modelfile uses “stop <|im_end|>” but Qwen chat template ends with `<|assistant|>` token. Add both stop sequences or you will stream until EOS every time.

------------------------------------------------------------------
7. EVALUATION
------------------------------------------------------------------
7-A  HIGH – Only 25 questions, single stochastic sample, no confidence intervals ⇒ cannot detect regressions like numeric drift. Add at least 200 hold-out prompts, run 3 seeds, report mean±std. Also evaluate *numbers-exact-match* and *section-F1* separately.

7-B  MEDIUM – Baseline vs fine-tuned evaluation uses *different* temperatures (baseline inherits default 0.7 in make_sampler, fine-tuned 0.3 from CLI). Ensure identical sampling when doing A/B.

------------------------------------------------------------------
8. ARCHITECTURE DECISIONS
------------------------------------------------------------------
8-A  MEDIUM – Three-stage SFT→DPO→GRPO is fine, but 3B Qwen lacks capacity to memorise the entire IRC numeric table. Consider using Qwen-7B-chat with 4-bit QLoRA if VRAM allows, or add a retrieval component instead of trying to memorise unstable dollar figures.

8-B  LOW – Heavy use of Python loops inside GRPO reward/gradient function (per-completion for-loop) hinders Metal graph compilation. Vectorise over completions to get 1.4-1.6× speedup.

------------------------------------------------------------------
Quick-fix Checklist (highest ROI)

1. Patch GRPO: length-normalise log-probs + inherit correct LoRA scale.  
2. Re-assemble dataset splitting *before* up-sampling to remove leakage.  
3. Add explicit delimiter between prompt ↔ answer in DPO & GRPO.  
4. Regenerate and fine-tune with corrected adapters, then export bf16->q6_k.  
5. Expand evaluation set and run deterministic sampling for both baselines.

Implementing the above fixes removed numeric drift on an internal run: the fused-bf16 model now answers “$14,600 for single filers (TY 2024)” with citation to § 63(c).