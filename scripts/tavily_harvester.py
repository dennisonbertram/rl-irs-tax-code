#!/usr/bin/env python3
"""
Tavily-powered IRS content harvesting toolkit.

Commands:
  map           -- discover IRS publication/form/bulletin URLs via Tavily Map
  extract       -- batch-extract full content from discovered URLs
  search TOPIC  -- search a specific tax topic
  bulk-search   -- search all ~200 predefined tax topics in parallel
  generate      -- convert extracted content into SFT training pairs via GPT-4o-mini
  pipeline      -- run the full pipeline end-to-end
  --estimate    -- estimate Tavily credit cost without running anything
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient, AsyncTavilyClient

# ── Env & paths ───────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

BASE_DIR = Path(__file__).parent.parent
TAVILY_DIR = BASE_DIR / "data" / "raw" / "tavily"
EXTRACTED_DIR = TAVILY_DIR / "extracted_content"
SEARCH_DIR = TAVILY_DIR / "search_results"
TRAINING_DIR = TAVILY_DIR / "training_pairs"
URLS_FILE = TAVILY_DIR / "urls_discovered.json"
PROGRESS_FILE = TAVILY_DIR / ".progress.json"

for d in [TAVILY_DIR, EXTRACTED_DIR, SEARCH_DIR, TRAINING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tavily_harvester")

# ── Rate limiting ─────────────────────────────────────────────────────────────

RATE_LIMIT_RPS = 100 / 60  # 100 req/min → ~1.67 req/s
_last_request_time: float = 0.0


def _rate_sleep() -> None:
    global _last_request_time
    now = time.monotonic()
    gap = 1.0 / RATE_LIMIT_RPS
    wait = _last_request_time + gap - now
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"extracted_urls": [], "searched_topics": []}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ── Predefined tax topics ─────────────────────────────────────────────────────

TAX_TOPICS: list[str] = [
    # Individual income tax — brackets and rates
    "federal income tax brackets 2024",
    "marginal tax rates individuals",
    "standard deduction amount 2024",
    "personal exemption IRC section 151",
    "alternative minimum tax AMT individuals",
    "net investment income tax 3.8 percent",
    "qualified business income deduction 199A",
    "filing status definitions head of household",
    "married filing jointly versus separately",
    "earned income tax credit EITC eligibility",
    "child tax credit 2024 phase out",
    "child and dependent care credit IRC 21",
    "American opportunity tax credit education",
    "lifetime learning credit",
    "adoption tax credit",
    "saver's credit retirement contributions",
    "premium tax credit ACA health insurance",
    "recovery rebate credit stimulus payments",
    # Itemized deductions
    "itemized deductions schedule A",
    "SALT deduction cap $10000 TCJA",
    "mortgage interest deduction IRC 163",
    "home equity loan interest deductibility",
    "charitable contribution deduction IRC 170",
    "cash charitable contributions 60 percent AGI limit",
    "non-cash charitable contributions appraisal",
    "qualified conservation easement donation",
    "casualty loss deduction federally declared disaster",
    "medical expense deduction 7.5 percent AGI",
    "miscellaneous itemized deductions suspended TCJA",
    # Business taxation
    "trade or business expenses IRC 162",
    "ordinary and necessary business expense test",
    "home office deduction IRC 280A",
    "vehicle business use depreciation",
    "meals entertainment deduction 50 percent",
    "travel expense deduction away from home",
    "Section 179 expensing election limit 2024",
    "bonus depreciation 100 percent phase down",
    "MACRS depreciation schedules",
    "accelerated cost recovery system",
    "amortization of intangibles IRC 197",
    "startup costs organizational expenses IRC 195",
    "net operating loss NOL carryforward TCJA",
    "business interest expense limitation IRC 163j",
    "excess business loss limitation IRC 461l",
    "self-employment tax deduction IRC 164",
    "self-employed health insurance deduction",
    "qualified opportunity zone investment IRC 1400Z",
    # Retirement accounts
    "401k contribution limits 2024",
    "IRA traditional contribution deduction limits",
    "Roth IRA income limits 2024",
    "Roth IRA contribution limits",
    "SEP IRA contribution limits self-employed",
    "SIMPLE IRA contribution limits",
    "403b retirement plan rules",
    "457 deferred compensation plan",
    "solo 401k plan rules",
    "defined benefit pension plan funding",
    "required minimum distributions RMD age 73",
    "early withdrawal penalty IRC 72t exceptions",
    "substantially equal periodic payments 72t",
    "Roth conversion rules tax consequences",
    "backdoor Roth IRA strategy",
    "rollover IRA rules 60-day rule",
    "qualified charitable distribution IRA age 70.5",
    "inherited IRA 10-year rule SECURE Act",
    "SECURE 2.0 Act retirement provisions 2024",
    "designated Roth account 401k in-plan conversion",
    # Capital gains
    "short-term capital gains tax rate",
    "long-term capital gains tax rates 0 15 20 percent",
    "capital loss carryover rules",
    "wash sale rule IRC 1091",
    "qualified small business stock exclusion IRC 1202",
    "collectibles capital gains 28 percent rate",
    "unrecaptured depreciation 1250 gain 25 percent",
    "installment sale reporting IRC 453",
    "like-kind exchange IRC 1031 real property",
    "1031 exchange boot recognized gain",
    "1033 involuntary conversion replacement property",
    "primary residence exclusion IRC 121 $250000",
    "home sale exclusion married $500000",
    "capital gains qualified dividends preferential rates",
    "dividend taxation qualified versus ordinary",
    # Estate and gift tax
    "estate tax exemption 2024 unified credit",
    "gift tax annual exclusion $18000 2024",
    "lifetime gift tax exemption",
    "estate tax rate schedule IRC 2001",
    "marital deduction unlimited IRC 2056",
    "charitable deduction estate tax IRC 2055",
    "stepped-up basis death IRC 1014",
    "gift tax basis carryover IRC 1015",
    "generation skipping transfer tax GST",
    "qualified personal residence trust QPRT",
    "grantor retained annuity trust GRAT",
    "irrevocable life insurance trust ILIT",
    "qualified tuition program 529 plan gift tax",
    # International tax
    "foreign earned income exclusion FEIE Form 2555",
    "foreign housing exclusion allowance",
    "foreign tax credit Form 1116 IRC 901",
    "foreign tax credit limitation basket rules",
    "FATCA foreign account tax compliance Act",
    "FBAR FinCEN 114 foreign bank account reporting",
    "passive foreign investment company PFIC rules",
    "controlled foreign corporation CFC Subpart F",
    "global intangible low-taxed income GILTI",
    "foreign derived intangible income FDII deduction",
    "base erosion anti-abuse tax BEAT",
    "treaty benefits tax treaty residency",
    "permanent establishment business profits treaty",
    "transfer pricing related party transactions IRC 482",
    "section 965 repatriation toll tax",
    # Employment taxes
    "FICA social security tax rate 6.2 percent",
    "Medicare tax rate 1.45 percent additional 0.9",
    "FUTA federal unemployment tax IRC 3301",
    "employee versus independent contractor test",
    "worker classification Form SS-8",
    "payroll tax deposits schedule",
    "backup withholding 24 percent",
    "supplemental wage withholding rates",
    "fringe benefits taxation IRC 132",
    "employer provided health insurance exclusion IRC 106",
    "group term life insurance imputed income $50000",
    "dependent care flexible spending account FSA",
    "health savings account HSA contribution limits",
    "commuter benefits transit parking exclusion",
    "stock options ISO versus NSO taxation",
    "restricted stock units RSU income recognition",
    "section 83b election property transferred services",
    # Tax-exempt organizations
    "501c3 public charity exemption requirements",
    "501c4 social welfare organization",
    "private foundation rules restrictions",
    "unrelated business income tax UBIT IRC 511",
    "unrelated debt-financed income",
    "excess benefit transactions IRC 4958",
    "private foundation self-dealing rules IRC 4941",
    "required distributions private foundation IRC 4942",
    "Form 990 annual information return",
    "nonprofit political activity lobbying limits",
    # Real estate
    "real estate professional status passive activity",
    "passive activity loss rules IRC 469",
    "material participation tests",
    "rental real estate $25000 allowance",
    "at-risk rules IRC 465",
    "real estate depreciation 27.5 years residential",
    "commercial property depreciation 39 years",
    "cost segregation study accelerated depreciation",
    "qualified improvement property 15-year MACRS",
    "real estate dealer versus investor",
    "foreclosure deed in lieu cancellation of debt",
    "cancellation of debt income IRC 108 exclusions",
    "insolvency exclusion canceled debt",
    "mortgage forgiveness debt relief act",
    # Partnership and S-Corp
    "partnership taxation pass-through IRC 701",
    "partnership basis outside inside basis",
    "partnership special allocations substantial economic effect",
    "partnership guaranteed payments IRC 707",
    "IRC 754 election step-up in basis",
    "partnership disguised sale rules IRC 707",
    "S-corporation election requirements IRC 1362",
    "S-corp reasonable compensation shareholder officer",
    "S-corp basis loss limitations",
    "S-corp fringe benefit treatment 2-percent shareholder",
    "built-in gains tax IRC 1374",
    "excess net passive income tax S-corp",
    # Penalties and compliance
    "failure to file penalty IRC 6651",
    "failure to pay penalty",
    "accuracy-related penalty IRC 6662",
    "fraud penalty 75 percent IRC 6663",
    "estimated tax penalty underpayment IRC 6654",
    "interest on underpayment IRC 6601",
    "foreign information return penalties IRC 6038",
    "FBAR civil penalty willful non-willful",
    "statute of limitations assessment IRC 6501",
    "offer in compromise IRC 7122",
    "currently not collectible status",
    "installment agreement payment plan",
    "innocent spouse relief IRC 6015",
    "tax levy seizure IRC 6331",
    "tax lien federal IRC 6321",
    # Recent legislation
    "Tax Cuts and Jobs Act TCJA 2017 summary",
    "CARES Act 2020 tax provisions",
    "American Rescue Plan 2021 tax changes",
    "Inflation Reduction Act 2022 clean energy credits",
    "electric vehicle tax credit IRC 30D",
    "used electric vehicle credit IRC 25E",
    "residential clean energy credit 30 percent IRC 25D",
    "energy efficient home improvement credit IRC 25C",
    "clean electricity production credit IRC 45Y",
    "SECURE Act 2019 provisions",
    "SECURE 2.0 Act 2022 catch-up contributions",
    "corporate alternative minimum tax CAMT 15 percent",
    "stock buyback excise tax 1 percent",
    "research and development credit IRC 41 amortization",
    # Tax Court and IRS guidance
    "Tax Court regular versus small case procedure",
    "IRS revenue rulings authority",
    "IRS private letter ruling process",
    "Treasury regulations proposed final temporary",
    "IRS notices announcements guidance priority plan",
    "Chief Counsel advice memoranda",
    "IRS audit selection DIF score",
    "correspondence audit versus field audit",
    "Appeals office tax dispute resolution",
    "Tax Court Cohan rule substantiation",
    "economic substance doctrine codified IRC 7701o",
    "sham transaction doctrine substance over form",
    # Tax planning strategies
    "bunching charitable deductions donor advised fund",
    "tax loss harvesting capital gains offset",
    "asset location taxable versus tax-advantaged accounts",
    "qualified opportunity zone deferral exclusion",
    "charitable remainder trust CRT annuity unitrust",
    "charitable lead trust CLT",
    "family limited partnership FLP valuation discount",
    "dynasty trust generation skipping",
    "income shifting family members kiddie tax",
    "kiddie tax unearned income minor IRC 1(g)",
    "deferred compensation NQDC section 409A",
    "nonqualified deferred compensation 409A requirements",
]

# ── IRS seed URLs for mapping ─────────────────────────────────────────────────

IRS_MAP_ROOTS = [
    "https://www.irs.gov/publications",
    "https://www.irs.gov/forms-instructions",
    "https://www.irs.gov/irb",
    "https://www.irs.gov/credits-deductions",
    "https://www.irs.gov/businesses",
    "https://www.irs.gov/retirement-plans",
]

# ── 1. Site mapping ────────────────────────────────────────────────────────────

def map_irs_publications() -> list[str]:
    """Use Tavily Map API to discover IRS content URLs."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set in .env")

    client = TavilyClient(api_key=TAVILY_API_KEY)
    all_urls: set[str] = set()

    for root in IRS_MAP_ROOTS:
        log.info("Mapping %s ...", root)
        try:
            _rate_sleep()
            result = client.map(
                url=root,
                max_depth=2,
                limit=100,
                include_subdomains=False,
            )
            urls = result if isinstance(result, list) else result.get("urls", result.get("results", []))
            log.info("  Found %d URLs under %s", len(urls), root)
            all_urls.update(urls)
        except Exception as exc:
            log.warning("Map failed for %s: %s", root, exc)

    # Filter to IRS domain only, drop PDFs larger than plausible page (keep pdfs)
    filtered = [u for u in all_urls if "irs.gov" in u]
    log.info("Total unique IRS URLs discovered: %d", len(filtered))

    URLS_FILE.write_text(json.dumps(sorted(filtered), indent=2))
    log.info("Saved to %s", URLS_FILE)
    return sorted(filtered)


# ── 2. Batch content extraction ───────────────────────────────────────────────

async def extract_urls(
    urls: list[str],
    batch_size: int = 20,
    max_retries: int = 2,
) -> list[dict]:
    """Extract full content from URLs in batches of 20 using AsyncTavilyClient."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set in .env")

    progress = load_progress()
    already_done: set[str] = set(progress.get("extracted_urls", []))

    pending = [u for u in urls if u not in already_done]
    log.info("Extracting %d URLs (%d already done, %d total)",
             len(pending), len(already_done), len(urls))

    results: list[dict] = []
    client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        log.info("Extracting batch %d-%d of %d ...",
                 batch_start + 1, batch_start + len(batch), len(pending))

        for attempt in range(1, max_retries + 2):
            try:
                response = await client.extract(
                    urls=batch,
                    extract_depth="advanced",
                    format="markdown",
                    include_images=False,
                )
                extracted = response if isinstance(response, list) else response.get("results", [])
                break
            except Exception as exc:
                if attempt <= max_retries:
                    wait = 2 ** attempt
                    log.warning("Extraction attempt %d failed: %s — retrying in %ds", attempt, exc, wait)
                    await asyncio.sleep(wait)
                else:
                    log.error("Extraction permanently failed for batch starting %s: %s", batch[0], exc)
                    extracted = []

        for item in extracted:
            url = item.get("url", "")
            record = {
                "url": url,
                "title": item.get("title", ""),
                "content": item.get("raw_content") or item.get("content", ""),
                "metadata": {
                    "source": "tavily_extract",
                    "extract_depth": "advanced",
                },
            }
            results.append(record)

            # Persist one file per URL
            safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_")[:200]
            out_path = EXTRACTED_DIR / f"{safe_name}.json"
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

            already_done.add(url)

        progress["extracted_urls"] = list(already_done)
        save_progress(progress)
        log.info("Progress: %d/%d URLs extracted", len(already_done), len(urls))

        # Brief pause between batches
        await asyncio.sleep(0.6)

    return results


# ── 3. IRS-specific search ────────────────────────────────────────────────────

async def search_irs_topic(topic: str, num_results: int = 20) -> list[dict]:
    """Search for IRS guidance on a specific tax topic."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set in .env")

    client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
    try:
        response = await client.search(
            query=f"IRS tax law {topic}",
            search_depth="advanced",
            max_results=num_results,
            include_domains=["irs.gov", "law.cornell.edu", "taxcourt.gov", "federalregister.gov"],
            include_raw_content=True,
        )
        results = response.get("results", [])
        log.info("Topic '%s': %d results", topic, len(results))
        return results
    except Exception as exc:
        log.error("Search failed for topic '%s': %s", topic, exc)
        return []


# ── 4. Bulk topic search ──────────────────────────────────────────────────────

async def bulk_search_tax_topics(
    topics: list[str] | None = None,
    max_concurrent: int = 5,
) -> list[dict]:
    """Search many tax topics with bounded concurrency and resume support."""
    if topics is None:
        topics = TAX_TOPICS

    progress = load_progress()
    done_topics: set[str] = set(progress.get("searched_topics", []))

    pending = [t for t in topics if t not in done_topics]
    log.info("Bulk search: %d topics pending (%d already done)", len(pending), len(done_topics))

    semaphore = asyncio.Semaphore(max_concurrent)
    all_results: list[dict] = []

    async def _search_one(topic: str) -> list[dict]:
        async with semaphore:
            results = await search_irs_topic(topic)
            # Save per-topic file
            safe = topic.replace(" ", "_").replace("/", "-")[:120]
            out_path = SEARCH_DIR / f"{safe}.json"
            out_path.write_text(json.dumps({"topic": topic, "results": results}, indent=2, ensure_ascii=False))
            done_topics.add(topic)
            progress["searched_topics"] = list(done_topics)
            save_progress(progress)
            # Brief sleep to respect rate limit
            await asyncio.sleep(0.65)
            return results

    tasks = [_search_one(t) for t in pending]
    for coro in asyncio.as_completed(tasks):
        res = await coro
        all_results.extend(res)

    log.info("Bulk search complete. Total results: %d", len(all_results))
    return all_results


# ── 5. Training data generator ────────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = (
    "You are a tax law expert specializing in the Internal Revenue Code (IRC). "
    "Always cite specific IRC sections and subsections when answering questions. "
    "Provide accurate, detailed explanations grounded strictly in the statutory text "
    "and official IRS guidance."
)

QA_GENERATION_PROMPT = """\
Below is official IRS content extracted from: {source_url}

--- BEGIN IRS CONTENT ---
{content}
--- END IRS CONTENT ---

Generate {n_pairs} diverse question-answer pairs that a tax professional or taxpayer might ask.
Include a mix of these question types:
- Factual recall (e.g., "What is the X limit for 2024?")
- Procedural (e.g., "How do you file for X?")
- Eligibility (e.g., "Who qualifies for X?")
- Calculation (e.g., "How is X calculated?")
- Comparison (e.g., "What is the difference between X and Y?")
- Edge cases (e.g., "What happens if X?")

Requirements:
- Questions must be specific and realistic — not generic
- Answers MUST be grounded strictly in the content above — do not hallucinate
- Every answer MUST cite the specific IRC section numbers or IRS publication referenced in the content
- Every answer MUST include specific dollar amounts, percentages, or dates when present in the content
- Answers must be at least 150 characters long
- Return a JSON array of objects, each with keys: "question" and "answer"

Return ONLY a valid JSON array. No other text, no markdown fences."""

DPO_GENERATION_PROMPT = """\
Below is official IRS content extracted from: {source_url}

--- BEGIN IRS CONTENT ---
{content}
--- END IRS CONTENT ---

Generate {n_pairs} preference pairs for DPO training on this tax content.
Each pair should represent a question where one answer is clearly better.

For each pair:
- "question": A specific tax question answerable from the content
- "chosen": A high-quality answer that cites the specific IRC section, includes exact numbers/limits, explains the rule precisely, and references the source
- "rejected": A poor answer that is either vague (no specific citations, no dollar amounts), subtly incorrect, or gives boilerplate advice without substance

Return a JSON array of objects with keys: "question", "chosen", "rejected".
Return ONLY valid JSON. No markdown fences, no extra text."""


# ── Token-based chunking ──────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4  # rough estimate: 1 token ≈ 4 characters
CHUNK_TOKENS = 3000
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN  # ~12000 characters


def chunk_content(content: str, chunk_size: int = CHUNK_CHARS, overlap: int = 500) -> list[str]:
    """Split content into overlapping chunks for better coverage of long documents."""
    if len(content) <= chunk_size:
        return [content]

    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        # Try to break at a paragraph or sentence boundary
        if end < len(content):
            # Look for paragraph break within the last 500 chars of the chunk
            break_pos = content.rfind("\n\n", start + chunk_size - 500, end)
            if break_pos == -1:
                break_pos = content.rfind("\n", start + chunk_size - 200, end)
            if break_pos != -1:
                end = break_pos
        chunks.append(content[start:end].strip())
        start = end - overlap  # overlap for context continuity
        if start >= len(content):
            break

    return [c for c in chunks if len(c) > 200]


# ── Robust JSON parsing ───────────────────────────────────────────────────────

def parse_llm_json(text: str) -> Any:
    """Parse JSON from LLM response, handling markdown fences and other wrapping."""
    if not text or not text.strip():
        return None
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object in the text
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


# ── Core generation functions ─────────────────────────────────────────────────

def _get_openai_client():
    """Return an OpenAI client or raise if unavailable."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_gpt(client, prompt: str, system: str) -> str | None:
    """Call GPT-4o-mini with json_object response format. Returns raw text or None."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as exc:
        log.warning("GPT call failed: %s", exc)
        return None


def _validate_sft_pair(question: str, answer: str, source_url: str) -> bool:
    """Return True if the pair meets quality requirements."""
    if not question or not answer:
        return False
    if len(answer) < 150:
        return False
    # Must have some specificity — citation, number, or section reference
    has_citation = (
        "irs.gov" in answer.lower()
        or source_url in answer
        or "irc" in answer.lower()
        or "section" in answer.lower()
        or "§" in answer
        or re.search(r'\$[\d,]+', answer)  # dollar amount
        or re.search(r'\d+(\.\d+)?%', answer)  # percentage
    )
    return bool(has_citation)


def content_to_training_pairs(
    content: str,
    source_url: str,
    source_type: str = "irs_publication",
    n_pairs: int = 8,
    client=None,
) -> list[dict]:
    """Convert extracted IRS content into SFT training pairs using GPT-4o-mini.

    Handles chunking for long documents and robust JSON parsing.
    """
    if client is None:
        try:
            client = _get_openai_client()
        except RuntimeError as exc:
            log.error("%s", exc)
            return []

    # Chunk long content for better coverage
    chunks = chunk_content(content)
    if len(chunks) > 1:
        log.debug("Content chunked into %d pieces for %s", len(chunks), source_url)

    all_pairs: list[dict] = []

    # Distribute n_pairs across chunks (at least 3 per chunk, max 10)
    pairs_per_chunk = max(3, min(10, n_pairs // max(1, len(chunks))))

    for chunk_idx, chunk in enumerate(chunks):
        prompt = QA_GENERATION_PROMPT.format(
            source_url=source_url,
            content=chunk,
            n_pairs=pairs_per_chunk,
        )

        # json_object mode requires a JSON *object* wrapper, not bare array.
        # We ask for {"pairs": [...]} and unwrap.
        wrapped_prompt = (
            prompt.rstrip()
            + "\n\nIMPORTANT: Return your answer as a JSON object with a single key "
            '"pairs" whose value is the array. Example: {"pairs": [...]}'
        )

        raw = _call_gpt(client, wrapped_prompt, GENERATION_SYSTEM_PROMPT)
        if not raw:
            continue

        parsed = parse_llm_json(raw)
        if isinstance(parsed, dict):
            # Unwrap the wrapper object — handle various key names
            for key in ("pairs", "qa_pairs", "questions", "results", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            else:
                # If no known key, try the first list-valued key
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    parsed = []
        if not isinstance(parsed, list):
            log.warning("Unexpected JSON shape from GPT for %s chunk %d", source_url, chunk_idx)
            continue

        for pair in parsed:
            if not isinstance(pair, dict):
                continue
            question = pair.get("question", "").strip()
            answer = pair.get("answer", "").strip()
            if not _validate_sft_pair(question, answer, source_url):
                continue
            has_citation = _validate_sft_pair(question, answer, source_url)
            all_pairs.append({
                "messages": [
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source_url": source_url,
                    "source_type": source_type,
                    "grounded": True,
                    "citation_validated": has_citation,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                },
            })

    return all_pairs


def content_to_dpo_pairs(
    content: str,
    source_url: str,
    source_type: str = "irs_publication",
    n_pairs: int = 4,
    client=None,
) -> list[dict]:
    """Generate DPO preference pairs from IRS content."""
    if client is None:
        try:
            client = _get_openai_client()
        except RuntimeError as exc:
            log.error("%s", exc)
            return []

    # Use first chunk only for DPO (more focused)
    chunk = content[:CHUNK_CHARS] if len(content) > CHUNK_CHARS else content

    prompt = DPO_GENERATION_PROMPT.format(
        source_url=source_url,
        content=chunk,
        n_pairs=n_pairs,
    )
    wrapped_prompt = (
        prompt.rstrip()
        + "\n\nReturn as JSON object: {\"pairs\": [...]}"
    )

    raw = _call_gpt(client, wrapped_prompt, GENERATION_SYSTEM_PROMPT)
    if not raw:
        return []

    parsed = parse_llm_json(raw)
    if isinstance(parsed, dict):
        for key in ("pairs", "dpo_pairs", "preference_pairs", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
            else:
                parsed = []
    if not isinstance(parsed, list):
        return []

    dpo_pairs = []
    for pair in parsed:
        if not isinstance(pair, dict):
            continue
        question = pair.get("question", "").strip()
        chosen = pair.get("chosen", "").strip()
        rejected = pair.get("rejected", "").strip()
        if not question or not chosen or not rejected:
            continue
        if len(chosen) < 150:
            continue
        if chosen == rejected:
            continue
        dpo_pairs.append({
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "source_url": source_url,
                "source_type": source_type,
            },
        })

    return dpo_pairs


# ── 6. Full pipeline helpers ──────────────────────────────────────────────────

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


async def run_generate(max_pages: int | None = None, test_mode: bool = False) -> None:
    """Generate SFT and DPO training pairs from extracted content + search results.

    Writes intermediate files per source type, then consolidates into:
      data/processed/tavily_sft_full.jsonl
      data/processed/tavily_dpo_full.jsonl
    """
    try:
        client = _get_openai_client()
    except RuntimeError as exc:
        log.error("Cannot generate: %s", exc)
        return

    # ── A. Extracted content files ──────────────────────────────────────────

    files = sorted(EXTRACTED_DIR.glob("*.json"))
    if max_pages:
        files = files[:max_pages]
    if test_mode:
        files = files[:5]
        log.info("TEST MODE: processing only 5 extracted files")

    log.info("Generating training pairs from %d extracted pages ...", len(files))

    extracted_sft: list[dict] = []
    extracted_dpo: list[dict] = []
    processed = 0

    for fpath in files:
        try:
            record = json.loads(fpath.read_text())
        except Exception:
            continue
        content = record.get("content", "")
        url = record.get("url", str(fpath))

        if len(content) < 200:
            log.debug("Skipping %s — content too short (%d chars)", url, len(content))
            continue

        sft_pairs = content_to_training_pairs(
            content=content,
            source_url=url,
            source_type="irs_publication",
            n_pairs=10,
            client=client,
        )
        extracted_sft.extend(sft_pairs)

        dpo_pairs = content_to_dpo_pairs(
            content=content,
            source_url=url,
            source_type="irs_publication",
            n_pairs=3,
            client=client,
        )
        extracted_dpo.extend(dpo_pairs)

        processed += 1
        log.info("[%d/%d extracted] %s → %d SFT, %d DPO from %s",
                 processed, len(files), fpath.name, len(sft_pairs), len(dpo_pairs), url)
        time.sleep(0.3)

    # Save extracted pairs intermediate
    _write_jsonl(TRAINING_DIR / "tavily_sft_extracted.jsonl", extracted_sft)
    _write_jsonl(TRAINING_DIR / "tavily_dpo_extracted.jsonl", extracted_dpo)
    log.info("Extracted content: %d SFT, %d DPO pairs", len(extracted_sft), len(extracted_dpo))

    # ── B. Search result files ───────────────────────────────────────────────

    search_files = sorted(SEARCH_DIR.glob("*.json"))
    if max_pages:
        search_files = search_files[:max_pages]
    if test_mode:
        search_files = search_files[:5]
        log.info("TEST MODE: processing only 5 search files")

    log.info("Generating training pairs from %d search result files ...", len(search_files))

    search_sft: list[dict] = []
    search_dpo: list[dict] = []
    s_processed = 0

    for sfpath in search_files:
        try:
            data = json.loads(sfpath.read_text())
        except Exception:
            continue
        topic = data.get("topic", str(sfpath.stem))
        results = data.get("results", [])

        topic_sft_count = 0
        topic_dpo_count = 0

        # Sort by score descending and take top-5 to limit GPT calls
        results_sorted = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        top_results = results_sorted[:5]

        # Aggregate top results into one combined context (one GPT call per topic)
        combined_parts = []
        primary_url = ""
        for idx, item in enumerate(top_results):
            url = item.get("url", "")
            content = item.get("raw_content") or item.get("content", "")
            if len(content) < 300:
                continue
            if not primary_url:
                primary_url = url
            # Take up to 2000 chars per result to keep total context manageable
            snippet = content[:2000]
            combined_parts.append(f"--- Source {idx+1}: {url} ---\n{snippet}")

        if not combined_parts or not primary_url:
            s_processed += 1
            continue

        combined_content = "\n\n".join(combined_parts)

        sft_pairs = content_to_training_pairs(
            content=combined_content,
            source_url=primary_url,
            source_type="tavily_search",
            n_pairs=10,
            client=client,
        )
        search_sft.extend(sft_pairs)
        topic_sft_count += len(sft_pairs)

        # DPO from combined context
        dpo_pairs = content_to_dpo_pairs(
            content=combined_content,
            source_url=primary_url,
            source_type="tavily_search",
            n_pairs=3,
            client=client,
        )
        search_dpo.extend(dpo_pairs)
        topic_dpo_count += len(dpo_pairs)

        time.sleep(0.3)

        s_processed += 1
        log.info("[%d/%d search] topic '%s' → %d SFT, %d DPO (cumulative: %d SFT, %d DPO)",
                 s_processed, len(search_files), topic,
                 topic_sft_count, topic_dpo_count,
                 len(search_sft), len(search_dpo))

    # Save search pairs intermediate
    _write_jsonl(TRAINING_DIR / "tavily_sft_search.jsonl", search_sft)
    _write_jsonl(TRAINING_DIR / "tavily_dpo_search.jsonl", search_dpo)
    log.info("Search results: %d SFT, %d DPO pairs", len(search_sft), len(search_dpo))

    # ── C. Consolidate into data/processed/ ─────────────────────────────────

    all_sft = extracted_sft + search_sft
    all_dpo = extracted_dpo + search_dpo

    sft_out = PROCESSED_DIR / "tavily_sft_full.jsonl"
    dpo_out = PROCESSED_DIR / "tavily_dpo_full.jsonl"
    _write_jsonl(sft_out, all_sft)
    _write_jsonl(dpo_out, all_dpo)

    log.info("=== Generation complete ===")
    log.info("  SFT pairs: %d extracted + %d search = %d total → %s",
             len(extracted_sft), len(search_sft), len(all_sft), sft_out)
    log.info("  DPO pairs: %d extracted + %d search = %d total → %s",
             len(extracted_dpo), len(search_dpo), len(all_dpo), dpo_out)

    # Print summary statistics
    _print_stats(all_sft, all_dpo)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as newline-delimited JSON."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def _print_stats(sft_pairs: list[dict], dpo_pairs: list[dict]) -> None:
    """Print quality statistics about generated pairs."""
    if not sft_pairs and not dpo_pairs:
        log.info("No pairs generated.")
        return

    print("\n" + "=" * 60)
    print("GENERATION STATISTICS")
    print("=" * 60)

    # SFT stats
    print(f"\nSFT pairs total: {len(sft_pairs)}")
    by_source: dict[str, int] = {}
    cited = 0
    for pair in sft_pairs:
        st = pair.get("metadata", {}).get("source_type", "unknown")
        by_source[st] = by_source.get(st, 0) + 1
        if pair.get("metadata", {}).get("citation_validated"):
            cited += 1
    for src, cnt in sorted(by_source.items()):
        print(f"  {src}: {cnt}")
    print(f"  Citation-validated: {cited}/{len(sft_pairs)} ({100*cited//max(1,len(sft_pairs))}%)")

    # DPO stats
    print(f"\nDPO pairs total: {len(dpo_pairs)}")
    dpo_by_source: dict[str, int] = {}
    for pair in dpo_pairs:
        st = pair.get("metadata", {}).get("source_type", "unknown")
        dpo_by_source[st] = dpo_by_source.get(st, 0) + 1
    for src, cnt in sorted(dpo_by_source.items()):
        print(f"  {src}: {cnt}")

    # Sample pairs
    if sft_pairs:
        print("\n--- SAMPLE SFT PAIRS ---")
        for i, pair in enumerate(sft_pairs[:3]):
            msgs = pair.get("messages", [])
            q = next((m["content"] for m in msgs if m["role"] == "user"), "")
            a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            print(f"\n[SFT {i+1}] Q: {q[:120]}...")
            print(f"         A: {a[:200]}...")

    if dpo_pairs:
        print("\n--- SAMPLE DPO PAIRS ---")
        for i, pair in enumerate(dpo_pairs[:2]):
            print(f"\n[DPO {i+1}] Q: {pair.get('prompt','')[:120]}...")
            print(f"         Chosen: {pair.get('chosen','')[:150]}...")
            print(f"         Rejected: {pair.get('rejected','')[:100]}...")

    print("=" * 60 + "\n")


async def run_pipeline() -> None:
    """Run the full pipeline: map -> extract -> bulk-search -> generate."""
    log.info("=== Starting full pipeline ===")

    log.info("--- Step 1: Map IRS URLs ---")
    urls = map_irs_publications()

    log.info("--- Step 2: Extract content from %d URLs ---", len(urls))
    await extract_urls(urls)

    log.info("--- Step 3: Bulk search tax topics ---")
    await bulk_search_tax_topics()

    log.info("--- Step 4: Generate training pairs ---")
    await run_generate()

    log.info("=== Pipeline complete ===")


# ── 7. Credit estimator ───────────────────────────────────────────────────────

def estimate_credits() -> None:
    """Print a dry-run estimate of Tavily API credit usage."""
    n_map_roots = len(IRS_MAP_ROOTS)
    estimated_urls = n_map_roots * 80  # ~80 URLs per root
    n_topics = len(TAX_TOPICS)
    n_extract_batches = (estimated_urls + 19) // 20

    # Tavily pricing (approximate as of 2024):
    # Map: 1 credit per map call
    # Extract: 1 credit per URL extracted (advanced = 2 credits each)
    # Search: 1 credit per search (advanced = 2 credits)

    map_credits = n_map_roots * 1
    extract_credits = estimated_urls * 2  # advanced depth
    search_credits = n_topics * 2  # advanced search

    total = map_credits + extract_credits + search_credits

    print("\n=== Tavily Credit Estimate ===")
    print(f"  Map calls:          {n_map_roots:>6}  →  {map_credits:>6} credits")
    print(f"  URLs to extract:    {estimated_urls:>6}  →  {extract_credits:>6} credits  (advanced)")
    print(f"  Topic searches:     {n_topics:>6}  →  {search_credits:>6} credits  (advanced)")
    print(f"  {'─' * 40}")
    print(f"  TOTAL ESTIMATED:    {'':>6}     {total:>6} credits")
    print()
    print(f"  Predefined topics:  {n_topics}")
    print(f"  Map seed URLs:      {n_map_roots}")
    print(f"  Extract batches:    {n_extract_batches}  (batch_size=20)")
    print()
    print("  Note: Extract credit cost depends on content length.")
    print("  Advanced extract charges 2 credits per URL.")
    print("  Advanced search charges 2 credits per query.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tavily-powered IRS content harvesting toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["map", "extract", "search", "bulk-search", "generate", "pipeline"],
        help="Command to run",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Tax topic for 'search' command",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate credits needed without running",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of pages to process in 'generate' (for testing)",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=20,
        help="Number of search results per topic (default: 20)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 5 files from each source to verify generation works",
    )

    args = parser.parse_args()

    if args.estimate:
        estimate_credits()
        return

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # generate command does not need TAVILY_API_KEY
    if args.command not in ("generate",) and not TAVILY_API_KEY:
        log.error("TAVILY_API_KEY not found. Check your .env file.")
        sys.exit(1)

    if args.command == "map":
        urls = map_irs_publications()
        print(f"\nDiscovered {len(urls)} IRS URLs. Saved to {URLS_FILE}")

    elif args.command == "extract":
        if not URLS_FILE.exists():
            log.error("No URLs file found. Run 'map' first.")
            sys.exit(1)
        urls = json.loads(URLS_FILE.read_text())
        asyncio.run(extract_urls(urls))

    elif args.command == "search":
        if not args.topic:
            log.error("Provide a topic: python tavily_harvester.py search 'capital gains'")
            sys.exit(1)
        results = asyncio.run(search_irs_topic(args.topic, num_results=args.num_results))
        out = SEARCH_DIR / f"{args.topic.replace(' ', '_')[:80]}.json"
        out.write_text(json.dumps({"topic": args.topic, "results": results}, indent=2))
        print(f"\n{len(results)} results saved to {out}")

    elif args.command == "bulk-search":
        asyncio.run(bulk_search_tax_topics())

    elif args.command == "generate":
        asyncio.run(run_generate(max_pages=args.max_pages, test_mode=args.test))

    elif args.command == "pipeline":
        asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
