# GRPO v3 Training Pipeline Results

## Scope Completed
- GRPO v3 training (300 iterations, PID 91392) monitored to completion
- Model exported to Ollama as `qwen25-tax-3b-v3`
- 5-question qualitative comparison across v1/v2/v3
- 50-sample formal evaluation across all 3 models

## GRPO v3 Final Training Metrics
- **Duration**: 10,248 seconds (~2.85 hours)
- **Steps**: 300/300
- **Hyperparameters**: group_size=4, lr=5e-7, eps_clip=0.2
- **Final step (300)**: loss=0.3072, avg_reward=0.978, max_reward=0.986
- **Peak reward**: avg_reward=1.000 at step 70
- **Overall pattern**: Rewards fluctuated between 0.35-1.0 with most steps in 0.85-0.99 range

### Training Trajectory (selected steps)
| Step | Loss | Avg Reward | Max Reward |
|------|------|-----------|-----------|
| 5 | -0.0280 | 0.894 | 0.976 |
| 50 | 0.3467 | 0.929 | 0.973 |
| 100 | 0.2828 | 0.914 | 0.965 |
| 150 | -0.0548 | 0.992 | 1.000 |
| 200 | 0.2743 | 0.911 | 0.998 |
| 250 | 0.1184 | 0.828 | 0.967 |
| 300 | 0.3072 | 0.978 | 0.986 |

## Model Export
- Fused LoRA adapters via `mlx_lm.fuse` (dequantized to bf16)
- Converted to GGUF q8_0 (3.3 GB) using llama.cpp clone at `/tmp/llama.cpp-v3`
- Note: Homebrew llama.cpp had a `MISTRAL4` attribute error; fixed by using cloned convert script
- Imported to Ollama as `qwen25-tax-3b-v3`

## Evaluation Results

### Formal 50-Sample Evaluation
| Model | Citation Accuracy | Fact Match |
|-------|------------------|-----------|
| qwen25-tax-3b (v1) | 0/50 (0.0%) | 49/50 (98.0%) |
| qwen25-tax-3b-v2 | 0/50 (0.0%) | 48/50 (96.0%) |
| qwen25-tax-3b-v3 | 0/49 (0.0%) | 46/49 (93.9%) |

**Notes**:
- v3 had 1 timeout error (q39), so evaluated on 49 samples
- Citation accuracy is 0% across all models because the eval checks for explicit section symbols (e.g., "Section 179") in responses, and the models tend to write section numbers without the special character
- Fact match measures overlap of numerical values between reference and response

### 5-Question Qualitative Comparison
All 3 models showed similar qualitative behavior:
1. **Standard deduction**: All incorrectly stated $135,000 (actual is ~$14,600)
2. **Section 179**: v1 said $500K, v2/v3 said $1.08M (v2/v3 more accurate)
3. **Section 21 qualifying individual**: All gave partially incorrect definitions
4. **Section 6662 negligence penalty**: v2/v3 correctly identified the 20% penalty; v1 was vaguer
5. **Section 7701 partnership**: All provided reasonable definitions

## Files Changed
- `outputs/grpo/adapters/adapters.safetensors` - Updated by training (106 MB)
- `outputs/grpo/adapters/adapters_best.safetensors` - Best checkpoint (106 MB)
- `outputs/grpo/train.log` - Training log
- `outputs/final/fused/` - Fused model (bf16)
- `outputs/final/model-q8.gguf` - GGUF model (3.3 GB)
- `outputs/final/Modelfile` - Ollama Modelfile for v3

## Risks and Observations
1. **Fact match regression**: v3 (93.9%) is slightly below v1 (98.0%) and v2 (96.0%) on fact match, though the difference is small (3-4 questions out of 50)
2. **Citation accuracy 0% across all models**: The evaluation regex (`Section(\d+)`) requires the section symbol, which models rarely use; this metric may need redesign
3. **Reward saturation**: avg_reward reached 0.978 by step 300 but oscillated heavily throughout, suggesting the reward function may have limited discriminative power at high values
4. **Standard deduction hallucination**: All models consistently hallucinate $135,000 for the standard deduction, indicating a persistent factual error in the training data or base model

## Next Steps
1. Investigate the fact match regression in v3 vs v1/v2 -- consider whether the GRPO reward may be over-optimizing for format over factual accuracy
2. Fix the citation accuracy metric to detect section references more broadly (e.g., "Section 179" not just "Section179")
3. Consider training with a factual grounding component in the reward function
4. Potentially use the `adapters_best.safetensors` checkpoint instead of the final one if it produces better factual results
