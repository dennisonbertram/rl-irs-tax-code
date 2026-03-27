# RAG-Grounded Training Data Generation Research

**Date:** 2026-03-26
**Goal:** Replace hallucination-prone synthetic training data with RAG-grounded, citation-verified data for tax law LLM fine-tuning.

---

## Current State Analysis

The existing `scripts/generate_training_data.py` uses **template-based generation** -- it takes the raw section text, wraps it in boilerplate questions ("What does IRC Section X say about Y?"), and uses the truncated source text directly as the answer. This produces:

- **27,600 SFT examples** -- but they're mechanical paraphrases, not genuine Q&A
- **10,181 DPO pairs** -- rejected answers are random vague templates ("consult a tax professional"), not hard negatives
- **26,899 GRPO prompts** -- prompts only, no reference answers for verification

The fundamental problem: the training data doesn't teach the model to **reason** about tax law -- it teaches it to regurgitate truncated text. When the fine-tuned model encounters questions that don't match these templates, it hallucinates.

---

## 1. Kimi K2.5

### What It Is
Kimi K2.5 is Moonshot AI's flagship model (released January 2026), built on a 1 trillion parameter Mixture-of-Experts architecture that activates 32B parameters per request. It features a 256K token context window and an "agent swarm" paradigm that can coordinate up to 100 parallel sub-agents.

### API Access
- **OpenAI-compatible API** at `https://api.moonshot.ai/v1`
- Model identifier: `kimi-k2.5`
- Uses standard OpenAI SDK -- just change the base URL and API key
- Sign up at https://platform.moonshot.ai
- Also available via OpenRouter at `moonshotai/kimi-k2.5`

### Pricing (Extremely Competitive)
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| **Kimi K2.5** | **$0.60** | **$2.50** |
| GPT-4o | $2.50 | $10.00 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Opus 4.6 | $15.00 | $75.00 |

Kimi K2.5 includes **automatic context caching** that reduces input costs by 75% (no configuration needed). This is significant for RAG workflows where you repeatedly send the same context chunks.

### Suitability for Legal/Tax Text
- 256K context window can fit entire IRC chapters (most chapters are under 200K tokens)
- Moonshot markets it for "automated contract review" and "complex patent analysis"
- Open-source under Modified MIT license (weights on HuggingFace at `moonshotai/Kimi-K2.5`)
- **No specific legal benchmarks published** -- this is the risk. It's primarily benchmarked on coding and math tasks.

### Verdict on Kimi K2.5
**Good for bulk generation at low cost, but not the best choice as the sole generator for high-stakes legal text.** The 4-17x cost advantage is real and matters at 25K pairs. However, it lacks proven legal reasoning benchmarks. Best used as part of a multi-model pipeline (generate with Kimi, validate with Claude/GPT-4o).

---

## 2. RAG-Grounded Data Generation Architecture

### Recommended Architecture

```
IRC/CFR Sections (8,262 total)
        |
        v
[Embedding + Vector Store]  <-- text-embedding-3-large
        |
        v
[Question Generator]        <-- Generates diverse questions per section
        |
        v
[Context Retriever]         <-- Retrieves top-k relevant sections (not just the one)
        |                       This captures cross-references!
        v
[Answer Generator]          <-- Must quote source text, cite sections
        |
        v
[Verifier/Filter]           <-- Checks citations exist, quotes are real
        |
        v
[Training Data]             <-- SFT, DPO, GRPO with verified answers
```

### Embedding Model Selection

**Recommendation: OpenAI `text-embedding-3-large`**

| Model | Dims | MTEB Score | Cost per 1M tokens | Notes |
|-------|------|------------|---------------------|-------|
| text-embedding-3-large | 3072 | 64.6 | $0.13 | Best for legal precision |
| text-embedding-3-small | 1536 | 62.3 | $0.02 | Good enough for 95% of cases |
| LexLM-Embed (OpenLegal) | varies | N/A | Free (local) | Specialized for legal, but smaller community |

For 8,262 sections averaging ~2K tokens each = ~16.5M tokens. Embedding cost: **~$2.15 with text-embedding-3-large**.

### Vector Store

**Use ChromaDB or FAISS locally.** No need for a hosted solution at this scale.

```python
# Simple setup -- no framework needed
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.PersistentClient(path="./data/vectorstore")
collection = chroma.get_or_create_collection("tax_code")

# Embed and store all sections
for section in all_sections:
    embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=section["text"][:8000]  # chunk if needed
    ).data[0].embedding

    collection.add(
        ids=[f"{section['source']}_{section['section']}"],
        embeddings=[embedding],
        documents=[section["text"]],
        metadatas=[{"source": section["source"], "section": section["section"], "heading": section["heading"]}]
    )
```

### Framework Choice: Raw API Calls Over LlamaIndex/LangChain

For this specific use case, **raw API calls with a thin wrapper** are better than LlamaIndex/LangChain because:
1. You're doing batch generation, not interactive chat
2. You need fine-grained control over the prompt for each training pair
3. The retrieval logic is straightforward (embed query, get top-k)
4. Frameworks add complexity and debugging overhead without proportional benefit
5. You need structured output (JSON) which the APIs handle natively

### The Critical Insight: Cross-Reference Retrieval

The IRC is full of cross-references ("as defined in section 7703", "see section 6013"). When generating an answer about Section 1 (Tax imposed), you should retrieve:
- Section 1 itself (the target)
- Section 7703 (definition of "married individual")
- Section 2 (surviving spouse definition)
- Section 63 (taxable income definition)

This means **retrieval should use both the question AND the source section text** to capture cross-references.

---

## 3. Alternative Approaches Worth Considering

### A. Claude with Long Context (200K-1M tokens)

**Promising for chapter-level generation.** Claude can now take up to 1M tokens in context. The IRC has ~90 subtitles/chapters. You could stuff an entire chapter into context and ask Claude to generate Q&A pairs grounded in that chapter.

**Pros:**
- No embedding/retrieval pipeline needed
- Claude sees ALL cross-references within the chapter naturally
- Better coherence -- answers can reference multiple related sections

**Cons:**
- Cost: a 200K token input to Claude Sonnet 4.6 = $0.60 per call. For ~90 chapters x multiple calls = expensive
- Latency: 30-60 seconds per large-context call
- Still can't see cross-references ACROSS chapters

**Verdict:** Use this for the most important IRC chapters (1-199, which cover income tax fundamentals). Use RAG for the rest.

### B. GPT-4o with Structured Outputs

GPT-4o's structured output mode guarantees valid JSON conforming to a schema. This is extremely useful for consistent training data formatting.

```python
from openai import OpenAI

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "tax_qa_pair",
            "schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "cited_sections": {"type": "array", "items": {"type": "string"}},
                    "direct_quotes": {"type": "array", "items": {"type": "string"}},
                    "difficulty": {"enum": ["basic", "intermediate", "advanced"]},
                    "topic_tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["question", "answer", "cited_sections", "direct_quotes", "difficulty"]
            }
        }
    },
    messages=[...]
)
```

**Verdict:** Use GPT-4o structured outputs as the **output format enforcer**, regardless of which model generates the core content.

### C. Self-Instruct / Evol-Instruct

**Evol-Instruct** (from WizardLM) iteratively makes questions harder:
1. Start with simple question: "What is taxable income?"
2. Add constraints: "How is taxable income calculated for a surviving spouse filing jointly who has capital gains from a partnership?"
3. Add complexity: "Compare the treatment under IRC 1(a) vs 1(d) for a married individual who files separately when their spouse has itemized deductions under IRC 63(d)."

This is excellent for generating the **question diversity** your current template approach lacks.

### D. Multi-Model Cross-Validation

Generate with Model A, validate with Model B:
1. Kimi K2.5 generates answer (cheap, high volume)
2. Claude verifies citations actually exist in the source text
3. GPT-4o checks the answer is factually consistent with the source

Any answer that fails validation gets flagged or regenerated. This is more expensive but catches hallucinations.

### E. Constitutional AI Style: Generate-Critique-Refine

1. **Generate:** Produce initial answer
2. **Critique:** "Does this answer cite specific IRC sections? Are the citations accurate? Does it contain any statements not supported by the source text?"
3. **Refine:** Fix issues identified in critique
4. **Repeat** if needed

This can be done with a single model doing all three steps. Budget 3x the tokens but get much higher quality.

---

## 4. DPO Hard Negative Generation

### The Problem with Current DPO Data
Current rejected answers are random vague templates. This teaches the model nothing useful -- the difference between chosen/rejected is too obvious. The model learns "don't be vague" rather than "be precisely correct."

### Best Practice: Hard Negatives via Perturbation

Based on recent research (arxiv:2512.19728 -- "Hard Negative Sample-Augmented DPO Post-Training"), the most effective approach creates negatives that are **structurally similar but contain specific errors**.

#### Perturbation Types for Tax Law

| Perturbation Type | Example | What It Teaches |
|-------------------|---------|-----------------|
| **Wrong section number** | "Under IRC 162" -> "Under IRC 163" | Precise citation matters |
| **Outdated figures** | "standard deduction of $14,600" -> "$12,400" (old amount) | Currency of information |
| **Missing exceptions** | Omit "except as provided in subsection (d)" | Completeness matters |
| **Wrong filing status** | Apply married-filing-jointly rules to single filer | Attention to conditions |
| **Swapped definitions** | Confuse "gross income" with "adjusted gross income" | Precision of terms |
| **Incorrect cross-reference** | "as defined in section 152" -> "as defined in section 151" | Accuracy of references |
| **Partial truth** | State the general rule but omit a critical limitation | Thoroughness |

#### Implementation Strategy

```python
def generate_hard_negative(correct_answer: str, section_text: str) -> str:
    """Use an LLM to create a subtle perturbation of a correct answer."""
    prompt = f"""Given this correct answer about tax law:

{correct_answer}

Create a subtly INCORRECT version that:
1. Changes exactly ONE factual detail (wrong section number, wrong dollar amount,
   wrong filing status, or omits a critical exception)
2. Maintains the same professional tone and structure
3. Would be difficult for a non-expert to distinguish from the correct answer
4. The error should be the type a poorly-trained model would actually make

Source text for reference:
{section_text}

Return ONLY the incorrect version, nothing else."""

    return call_llm(prompt)
```

#### On-Policy Hard Negatives (Best for GRPO/DPO Iteration)

After initial SFT training:
1. Run the current model on GRPO prompts
2. Collect responses that score < 0.5 on your reward function
3. Pair them as rejected against verified correct answers as chosen
4. This creates DPO pairs calibrated to your model's actual failure modes

---

## 5. Practical Recommendation

### Recommended Workflow

```
Phase 1: Embed & Index (one-time, ~$2)
  - Embed all 8,262 sections with text-embedding-3-large
  - Store in ChromaDB locally

Phase 2: Question Generation (~$15-25)
  - Use GPT-4o-mini to generate 5-10 diverse questions per section
  - Use Evol-Instruct to create difficulty variants
  - Target: ~50K raw questions

Phase 3: RAG-Grounded Answer Generation (~$80-150)
  - For each question, retrieve top-5 relevant sections
  - Use Kimi K2.5 for bulk generation (cheap)
  - Use structured output schema to enforce citations
  - Target: ~50K raw Q&A pairs

Phase 4: Verification & Filtering (~$30-50)
  - Use Claude Sonnet 4.6 to verify a 20% sample
  - Automated checks: do cited sections exist? Are quotes real?
  - Filter to ~25K verified pairs

Phase 5: DPO Pair Construction (~$20-30)
  - Take verified correct answers
  - Generate hard negatives via perturbation (use GPT-4o-mini, cheapest)
  - Create ~10K hard DPO pairs

Phase 6: GRPO Reference Answers (~$10-15)
  - For GRPO prompts, attach the source section text as reference
  - Upgrade reward function to use exact text matching against source
```

### Cost Breakdown

| Phase | Model | Tokens (est.) | Cost |
|-------|-------|---------------|------|
| Embedding | text-embedding-3-large | 16.5M | $2 |
| Question gen | GPT-4o-mini ($0.15/$0.60 per 1M) | ~25M in / ~25M out | $19 |
| Answer gen (Kimi K2.5) | Kimi K2.5 ($0.60/$2.50 per 1M) | ~100M in / ~50M out | $185 |
| Answer gen (alt: GPT-4o) | GPT-4o ($2.50/$10 per 1M) | ~100M in / ~50M out | $750 |
| Verification | Claude Sonnet 4.6 (20% sample) | ~20M in / ~5M out | $135 |
| DPO negatives | GPT-4o-mini | ~15M in / ~10M out | $8 |
| **Total (Kimi path)** | | | **~$350** |
| **Total (GPT-4o path)** | | | **~$915** |

### Model Selection Recommendation

**Use a two-model pipeline:**

1. **Kimi K2.5 as the bulk generator** -- 4x cheaper than GPT-4o, 256K context, good enough for initial generation
2. **Claude Sonnet 4.6 as the verifier** -- best at instruction-following for verification tasks, catches subtle errors

Do NOT use Claude Opus for generation -- it's 25x the cost of Kimi for marginal quality improvement on this task. Save Opus for designing the prompts and evaluation criteria.

### Single Most Impactful Change

Before investing in any of the above, **upgrade the reward function** in `scripts/grpo_reward.py` to do **exact text matching against the source section**:

```python
def compute_grounded_reward(prompt, response, source_sections):
    """Reward based on verifiable grounding in source text."""
    # 1. Extract claimed citations from response
    claimed_sections = extract_citations(response)

    # 2. Check each citation exists in our corpus
    valid_citations = [s for s in claimed_sections if s in source_sections]
    citation_accuracy = len(valid_citations) / max(len(claimed_sections), 1)

    # 3. Check quoted text actually appears in source
    quotes = extract_quotes(response)  # text in quotation marks
    verified_quotes = 0
    for quote in quotes:
        for sec_id in valid_citations:
            if quote.lower() in source_sections[sec_id].lower():
                verified_quotes += 1
                break
    quote_accuracy = verified_quotes / max(len(quotes), 1)

    # 4. Penalize claims not grounded in source
    # ... (use NLI model or simple overlap check)

    return 0.4 * citation_accuracy + 0.4 * quote_accuracy + 0.2 * length_score
```

This single change -- making the reward function verify against actual source text -- is what will most reduce hallucinations during GRPO training.

---

## Key Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Primary generator | Kimi K2.5 | 4x cheaper, 256K context, sufficient quality |
| Verifier | Claude Sonnet 4.6 | Best instruction-following for verification |
| Embedding | text-embedding-3-large | Best quality for legal text at $0.13/1M tokens |
| Vector store | ChromaDB (local) | 8K sections is tiny, no need for hosted DB |
| Framework | Raw API calls | More control, less debugging, simpler pipeline |
| DPO approach | Perturbation + on-policy | Hard negatives that match actual model failures |
| Question diversity | Evol-Instruct | Creates difficulty gradients from basic to advanced |

---

## Sources

- [Kimi K2.5 Quickstart](https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart)
- [Kimi K2.5 on OpenRouter](https://openrouter.ai/moonshotai/kimi-k2.5)
- [Kimi K2.5 Pricing](https://www.nxcode.io/resources/news/kimi-k2-5-pricing-plans-api-costs-2026)
- [Kimi K2.5 Analysis](https://artificialanalysis.ai/models/kimi-k2-5)
- [Kimi K2.5 on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2.5)
- [OpenAI Embedding Models](https://platform.openai.com/docs/guides/embeddings)
- [Best Embedding Models 2026](https://elephas.app/blog/best-embedding-models)
- [Hard Negative DPO (arxiv:2512.19728)](https://arxiv.org/abs/2512.19728)
- [Long Context RAG Performance](https://www.databricks.com/blog/long-context-rag-performance-llms)
- [1M Token Context vs RAG](https://www.mindstudio.ai/blog/1m-token-context-window-vs-rag-claude)
- [RAG vs Long Context Tradeoffs](https://redis.io/blog/rag-vs-large-context-window-ai-apps/)
- [OpenAI API Pricing](https://openai.com/api/pricing/)
- [LLM Synthetic Data Research](https://github.com/pengr/LLM-Synthetic-Data)
- [Red Hat SDG Hub for RAG Evaluation](https://developers.redhat.com/articles/2026/02/23/synthetic-data-rag-evaluation-why-your-rag-system-needs-better-testing)
