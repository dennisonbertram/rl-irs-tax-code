# Round 3 Pipeline: GRPO v3 Training, Export, and Evaluation

## Training Summary

**GRPO v3 Training completed successfully.**
- Duration: 10,248 seconds (~2h 51m)
- Iterations: 300
- Group size: 4
- Learning rate: 5e-7
- Starting adapter: DPO v3 (outputs/dpo/adapters)
- Final loss: 0.3072
- Final avg_reward: 0.978
- Final max_reward: 0.986

### Reward Function
The reward function already had citation accuracy checking (v3 scoring):
- Citation format: 0.30 weight (up to 4 citations)
- Citation accuracy: 0.40 weight (correct section match against expected_section)
- Length/detail: 0.30 weight
- Vague language penalty: -0.30

### Training Stability
Loss stabilized around 0.1-0.35 after initial oscillation (steps 1-30 had some spikes up to 8.25).
Rewards remained strong throughout: avg typically 0.85-1.00, with occasional dips to 0.35-0.56 on harder prompts.

## Export
Model exported as `qwen25-tax-3b-v3` to Ollama using q8_0 quantization (3.3 GB).

## 5-Question Evaluation: v1 vs v2 vs v3

### Q1: Standard deduction for single filer in 2024?
| Model | Answer | Correct? | Citations? |
|-------|--------|----------|------------|
| v1 | $135,000 (wildly wrong) | NO | No IRC citations |
| v2 | $13,850 | YES | No IRC citations |
| v3 | $13,850 | YES | No IRC citations |

### Q2: IRC Section 179 maximum expense amount?
| Model | Answer | Correct? | Citations? |
|-------|--------|----------|------------|
| v1 | $500,000, $2.5M phase-out | Outdated but structurally correct | IRC Section 179 |
| v2 | $1,080,000 (2023+) | YES (current) | IRC Section 179 |
| v3 | $11,500 (wrong) | NO | IRC Section 179 |

### Q3: Qualifying individual under IRC Section 21?
| Model | Answer | Correct? | Citations? |
|-------|--------|----------|------------|
| v1 | Confused with "qualifying child"; age 27 cutoff wrong | NO | IRC Section 21 |
| v2 | Confused with "qualified child"; mixes Sec 21 and 152 | PARTIAL | IRC Section 21 |
| v3 | Correctly redirects to dependent rules under Sec 152 | PARTIAL | IRC Sections 21, 152 |

### Q4: Penalty for negligence under IRC Section 6662?
| Model | Answer | Correct? | Citations? |
|-------|--------|----------|------------|
| v1 | 20% accuracy-related penalty | YES | IRC Section 6662 |
| v2 | 20% accuracy-related penalty | YES | IRC Section 6662 |
| v3 | 20% penalty, adds AGI context | YES (with extra detail) | IRC Section 6662(a) |

### Q5: Partnership definition under IRC Section 7701?
| Model | Answer | Correct? | Citations? |
|-------|--------|----------|------------|
| v1 | Two or more persons, trade or business | YES | IRC Section 7701 |
| v2 | Two or more persons, co-owners, trade or business | YES | IRC Section 7701(a)(7) |
| v3 | Two or more persons, for profit, trade or business | YES | IRC Section 7701(a)(7) |

## Scorecard Summary

| Question | v1 | v2 | v3 |
|----------|----|----|-----|
| Q1 (Std deduction) | 0 | 1 | 1 |
| Q2 (Sec 179) | 0.5 | 1 | 0 |
| Q3 (Sec 21) | 0 | 0.5 | 0.5 |
| Q4 (Sec 6662) | 1 | 1 | 1 |
| Q5 (Sec 7701) | 1 | 1 | 1 |
| **Total** | **2.5/5** | **4.5/5** | **3.5/5** |

## Key Observations

1. **v2 is the best model overall** on this 5-question benchmark (4.5/5)
2. **v3 improved over v1** significantly (3.5 vs 2.5) but regressed from v2 on the Section 179 question
3. **v3 provides more specific subsection citations** (e.g., 7701(a)(7), 6662(a)) compared to v1
4. **v3 hallucinated on Section 179** claiming $11,500 max expense instead of the correct ~$1.16M
5. **All models correctly handle well-known provisions** (Section 6662, 7701)
6. **Citation quality improved across versions** - v3 cites specific subsections more frequently

## Risks
- v3 Section 179 hallucination suggests possible overfitting on some training examples
- The reward function heavily weights citation accuracy (0.40) which may encourage citing sections even when the factual content is wrong
- Small 5-question eval may not be representative; recommend larger benchmark

## Next Steps
- Consider using the `adapters_best.safetensors` checkpoint instead of final (it was saved at an earlier step with potentially better generalization)
- Expand evaluation to 20+ questions covering more IRC sections
- Consider blending v2 and v3 approaches or using v2 as the production model until v3 improves
