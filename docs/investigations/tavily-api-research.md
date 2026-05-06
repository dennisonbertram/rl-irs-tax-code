# Tavily API Research Report

**Date**: 2026-03-27
**Purpose**: Evaluate Tavily API for bulk IRS tax content gathering

---

## 1. API Overview

Tavily is a search and content extraction API designed specifically for RAG and agent workflows. It provides five core endpoints:

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `/search` | Web search with relevance ranking | GA |
| `/extract` | Raw content extraction from URLs | GA |
| `/map` | Website structure discovery | GA |
| `/crawl` | Multi-page traversal + extraction | Invite-only |
| `/research` | Automated research reports | GA |

Base URL: `https://api.tavily.com/`
Auth: Bearer token (`Authorization: Bearer tvly-YOUR_API_KEY`)

---

## 2. Search Endpoint (`POST /search`)

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query (max 400 chars) |
| `search_depth` | enum | `basic` | `ultra-fast`, `fast`, `basic`, `advanced` |
| `max_results` | int | 5 | Max 20 results per request |
| `topic` | enum | `general` | `general`, `news`, `finance` |
| `include_raw_content` | bool/str | false | `true`/`markdown`/`text` for full page content |
| `include_answer` | bool/str | false | LLM-generated answer (`basic` or `advanced`) |
| `include_domains` | array | [] | Restrict to specific domains (max 300) |
| `exclude_domains` | array | [] | Exclude domains (max 150) |
| `time_range` | enum | none | `day`, `week`, `month`, `year` |
| `start_date`/`end_date` | string | none | `YYYY-MM-DD` format |
| `country` | enum | none | Boost results from specific country |
| `exact_match` | bool | false | Only exact phrase matches |
| `chunks_per_source` | int | 3 | 1-3, only with `advanced` depth |
| `include_images` | bool | false | Include query-related images |
| `include_favicon` | bool | false | Include favicon URLs |
| `auto_parameters` | bool | false | Auto-configure based on query intent (2 credits) |

### Response Structure

```json
{
  "query": "string",
  "answer": "string (if requested)",
  "results": [
    {
      "title": "string",
      "url": "string",
      "content": "string (snippet)",
      "score": 0.95,
      "raw_content": "string (full page if requested)"
    }
  ],
  "response_time": 1.5,
  "usage": { "credits": 1 }
}
```

### Search Depth Comparison

| Depth | Credits | Latency | Use Case |
|-------|---------|---------|----------|
| `ultra-fast` | 1 | Lowest | Real-time apps |
| `fast` | 1 | Low | Quick lookups with reranking |
| `basic` | 1 | Medium | General purpose (default) |
| `advanced` | 2 | Higher | Detailed, high-relevance results |

---

## 3. Extract Endpoint (`POST /extract`)

Retrieves cleaned content from specified URLs. This is the primary tool for pulling full-text IRS documents.

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | string/array | required | Single URL or up to 20 URLs |
| `extract_depth` | enum | `basic` | `basic` or `advanced` |
| `format` | enum | `markdown` | `markdown` or `text` |
| `query` | string | none | User intent for reranking content chunks |
| `chunks_per_source` | int | 3 | 1-5, only with `query` set |
| `include_images` | bool | false | Include image URLs |
| `timeout` | float | 10s/30s | 1.0-60.0 seconds |

### Response Structure

```json
{
  "results": [
    {
      "url": "https://www.irs.gov/publications/p17",
      "raw_content": "full extracted content in markdown...",
      "images": []
    }
  ],
  "failed_results": [
    {
      "url": "https://failed-url.com",
      "error": "reason"
    }
  ],
  "response_time": 3.2,
  "usage": { "credits": 1 }
}
```

### Extract Depth Comparison

| Depth | Credits | Timeout | Capability |
|-------|---------|---------|------------|
| `basic` | 1 per 5 URLs | ~10s | Standard HTML extraction |
| `advanced` | 2 per 5 URLs | ~30s | Tables, embedded content, complex layouts |

**Critical for IRS**: Use `advanced` extract depth since IRS publications contain complex tables and nested structures.

---

## 4. Map Endpoint (`POST /map`)

Discovers all URLs on a website from a starting point. Useful for discovering all IRS publication pages.

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | Starting URL |
| `max_depth` | int | 1 | How far to traverse from base |
| `max_breadth` | int | 20 | Max links per page level |
| `limit` | int | 50 | Max total URLs to discover |
| `instructions` | string | none | Natural language focus guidance |
| `select_paths` | array | none | Regex patterns for URL paths |
| `exclude_paths` | array | none | Regex to exclude paths |
| `allow_external` | bool | true | Include external domain links |

**Cost**: 1 credit per 10 pages (2 credits with instructions)

---

## 5. Crawl Endpoint (`POST /crawl`)

Combines Map + Extract: traverses a site and extracts content from discovered pages.

**Status**: Currently invite-only (apply at crawl.tavily.com)

### Key Parameters

Same as Map, plus all Extract parameters (`extract_depth`, `format`, `include_images`).

**Cost**: Map credits + Extract credits combined. Example: crawling 10 pages with basic extraction = 3 credits (1 mapping + 2 extraction).

---

## 6. Python SDK

### Installation

```bash
pip install tavily-python
```

### Synchronous Client

```python
from tavily import TavilyClient

client = TavilyClient(api_key="tvly-YOUR_API_KEY")

# Basic search
response = client.search("IRS Publication 17 tax filing requirements")

# Search restricted to IRS.gov
response = client.search(
    query="retirement account contribution limits 2025",
    include_domains=["irs.gov"],
    search_depth="advanced",
    include_raw_content="markdown",
    max_results=10
)

# Extract content from specific URLs
response = client.extract(
    urls=[
        "https://www.irs.gov/publications/p17",
        "https://www.irs.gov/publications/p501",
        "https://www.irs.gov/publications/p590a"
    ],
    extract_depth="advanced",
    format="markdown"
)

# Get RAG-ready context
context = client.get_search_context(query="IRS rules for capital gains")

# Direct Q&A
answer = client.qna_search(query="What is the standard deduction for 2025?")

# Map IRS publications structure
response = client.map(
    url="https://www.irs.gov/publications",
    max_depth=2,
    limit=100,
    select_paths=["/publications/p.*"]
)
```

### Async Client (for Batch/Parallel Operations)

```python
import asyncio
from tavily import AsyncTavilyClient

client = AsyncTavilyClient(api_key="tvly-YOUR_API_KEY")

async def batch_search(queries: list[str]):
    """Run multiple searches in parallel."""
    responses = await asyncio.gather(
        *(client.search(q, include_domains=["irs.gov"], max_results=10) for q in queries),
        return_exceptions=True
    )
    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"Query '{queries[i]}' failed: {response}")
        else:
            results.append(response)
    return results

async def batch_extract(url_batches: list[list[str]]):
    """Extract content from multiple URL batches in parallel.
    Each batch can contain up to 20 URLs.
    """
    responses = await asyncio.gather(
        *(client.extract(urls=batch, extract_depth="advanced", format="markdown")
          for batch in url_batches),
        return_exceptions=True
    )
    return responses

# Example usage
queries = [
    "IRS filing requirements single taxpayer",
    "IRS capital gains tax rates",
    "IRS retirement contribution limits",
    "IRS standard deduction amounts",
    "IRS earned income credit eligibility",
]
results = asyncio.run(batch_search(queries))
```

### Custom Session Injection

```python
import requests

session = requests.Session()
session.headers["Authorization"] = "Bearer token"
client = TavilyClient(session=session, api_base_url="https://gateway.com/tavily")
```

---

## 7. Pricing and Rate Limits

### Plans

| Plan | Monthly Credits | Price | Per-Credit Cost |
|------|----------------|-------|-----------------|
| Researcher (Free) | 1,000 | $0 | Free |
| Project | 4,000 | $30/mo | $0.0075 |
| Bootstrap | 15,000 | $100/mo | $0.0067 |
| Startup | 38,000 | $220/mo | $0.0058 |
| Growth | 100,000 | $500/mo | $0.005 |
| Pay-as-you-go | Unlimited | $0.008/credit | $0.008 |
| Enterprise | Custom | Custom | Custom |

**Credits do NOT roll over** between months.

### Credit Consumption

| Operation | Credits |
|-----------|---------|
| Search (basic/fast/ultra-fast) | 1 |
| Search (advanced) | 2 |
| Extract basic (per 5 URLs) | 1 |
| Extract advanced (per 5 URLs) | 2 |
| Map (per 10 pages) | 1 |
| Map with instructions (per 10 pages) | 2 |
| Crawl | Map + Extract combined |
| Research (pro) | 15-250 |
| Research (mini) | 4-110 |
| Auto-parameters | 2 |
| Failed extractions/maps | 0 (no charge) |

### Rate Limits

| Environment | Requests/Minute |
|-------------|-----------------|
| Development | 100 |
| Production | 1,000 |

### Error Codes

| Code | Meaning |
|------|---------|
| 429 | Rate limit exceeded |
| 432 | Plan usage exceeded |
| 433 | Pay-as-you-go limit exceeded |

---

## 8. Recommended Approach for Bulk IRS Content Gathering

### Strategy: Two-Phase Search + Extract Pipeline

**Phase 1: Discovery (Search + Map)**
1. Use `map()` on `https://www.irs.gov/publications` to discover all publication URLs
2. Use `map()` on `https://www.irs.gov/internal-revenue-bulletins` for IRBs
3. Use `search()` with `include_domains=["irs.gov"]` for targeted content discovery
4. Collect and deduplicate all discovered URLs

**Phase 2: Extraction (Batch Extract)**
1. Group URLs into batches of 20 (max per extract request)
2. Use `AsyncTavilyClient` to run extract batches in parallel
3. Use `extract_depth="advanced"` for tables and complex content
4. Use `format="markdown"` for structured output
5. Handle failed extractions with retry logic

### Cost Estimation for IRS Content Gathering

Assuming we target ~500 IRS pages:

| Phase | Operation | Credits |
|-------|-----------|---------|
| Discovery | 50 searches (basic) | 50 |
| Discovery | Map IRS.gov (500 pages) | 50 |
| Extraction | 500 pages advanced (100 batches of 5) | 200 |
| **Total** | | **~300 credits** |

This fits within the free tier (1,000 credits/month) for initial exploration.

### Implementation Pattern

```python
import asyncio
import json
from tavily import TavilyClient, AsyncTavilyClient

TAVILY_API_KEY = "tvly-YOUR_KEY"
sync_client = TavilyClient(api_key=TAVILY_API_KEY)
async_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# Phase 1: Discover IRS publication URLs
pub_map = sync_client.map(
    url="https://www.irs.gov/publications",
    max_depth=2,
    limit=200,
    select_paths=["/publications/p\\d+"]
)
pub_urls = [item["url"] for item in pub_map.get("results", [])]

# Phase 2: Batch extract content
async def extract_all(urls, batch_size=20):
    batches = [urls[i:i+batch_size] for i in range(0, len(urls), batch_size)]
    all_results = []

    for i in range(0, len(batches), 5):  # 5 concurrent batches
        chunk = batches[i:i+5]
        responses = await asyncio.gather(
            *(async_client.extract(
                urls=batch,
                extract_depth="advanced",
                format="markdown"
            ) for batch in chunk),
            return_exceptions=True
        )
        for resp in responses:
            if not isinstance(resp, Exception):
                all_results.extend(resp.get("results", []))

        await asyncio.sleep(1)  # Rate limit courtesy

    return all_results

results = asyncio.run(extract_all(pub_urls))

# Save results
for result in results:
    filename = result["url"].split("/")[-1] + ".md"
    with open(f"data/irs_publications/{filename}", "w") as f:
        f.write(result["raw_content"])
```

### Rate Limit Management

- Limit to ~5 concurrent extract requests to stay under 100 req/min (dev) or 1000 req/min (prod)
- Add 1-second delay between batch waves
- Implement exponential backoff on 429 errors
- Track credit usage with `include_usage=True`

---

## 9. High-Value IRS Data Sources to Target

### Tier 1: Core Tax Publications (HTML on irs.gov)

These are the primary reference documents for individual and business taxes. All available at `https://www.irs.gov/publications/p{number}`.

| Pub # | Title | Topic |
|-------|-------|-------|
| 17 | Your Federal Income Tax | Comprehensive individual tax guide |
| 501 | Dependents, Standard Deduction, Filing Info | Filing status, exemptions |
| 502 | Medical and Dental Expenses | Itemized deductions - medical |
| 503 | Child and Dependent Care Expenses | Care credits |
| 504 | Divorced or Separated Individuals | Filing after divorce |
| 505 | Tax Withholding and Estimated Tax | W-4, quarterly payments |
| 525 | Taxable and Nontaxable Income | Income inclusion/exclusion |
| 526 | Charitable Contributions | Donation deductions |
| 527 | Residential Rental Property | Rental income/deductions |
| 529 | Miscellaneous Deductions | Other deductions |
| 535 | Business Expenses | Business deductions |
| 544 | Sales and Other Dispositions of Assets | Capital gains, basis |
| 550 | Investment Income and Expenses | Interest, dividends, gains |
| 551 | Basis of Assets | Cost basis rules |
| 554 | Tax Guide for Seniors | Senior-specific rules |
| 559 | Survivors, Executors, and Administrators | Estate tax |
| 590-A | Contributions to IRAs | IRA contribution rules |
| 590-B | Distributions from IRAs | IRA withdrawal rules |
| 596 | Earned Income Credit | EITC eligibility and calculation |
| 936 | Home Mortgage Interest Deduction | Mortgage interest rules |
| 946 | How to Depreciate Property | Depreciation methods |
| 969 | Health Savings Accounts | HSA rules |
| 970 | Tax Benefits for Education | Education credits/deductions |

### Tier 2: Business and Entity Publications

| Pub # | Title |
|-------|-------|
| 15 | Employer's Tax Guide (Circular E) |
| 15-A | Employer's Supplemental Tax Guide |
| 334 | Tax Guide for Small Business |
| 541 | Partnerships |
| 542 | Corporations |
| 557 | Tax-Exempt Status for Your Organization |
| 583 | Starting a Business and Keeping Records |
| 925 | Passive Activity and At-Risk Rules |

### Tier 3: IRS Administrative Guidance

| Source | URL Pattern | Description |
|--------|------------|-------------|
| Internal Revenue Bulletins | `irs.gov/internal-revenue-bulletins` | Weekly compendium of official guidance |
| Revenue Rulings | Published within IRBs | IRS interpretation of tax law applied to facts |
| Revenue Procedures | Published within IRBs | Administrative procedures and filing instructions |
| Treasury Decisions | Published within IRBs | Finalized or temporary Treasury regulations |
| IRS Notices | Published within IRBs | Public guidance on tax issues |
| Announcements | Published within IRBs | Public statements of policy |

### Tier 4: Forms and Instructions

| Source | URL Pattern |
|--------|------------|
| Form 1040 Instructions | `irs.gov/pub/irs-pdf/i1040gi.pdf` |
| Schedule A Instructions | `irs.gov/pub/irs-pdf/i1040sa.pdf` |
| Schedule C Instructions | `irs.gov/pub/irs-pdf/i1040sc.pdf` |
| Schedule D Instructions | `irs.gov/pub/irs-pdf/i1040sd.pdf` |
| Schedule E Instructions | `irs.gov/pub/irs-pdf/i1040se.pdf` |
| All forms index | `irs.gov/forms-instructions` |

### Tier 5: External Authoritative Sources

| Source | URL | Content Type |
|--------|-----|--------------|
| Tax Court Opinions | `ustaxcourt.gov/ustc/opinions-orders` | Case law |
| Cornell Law - IRC | `law.cornell.edu/uscode/text/26` | Internal Revenue Code text |
| eCFR Title 26 | `ecfr.gov/current/title-26` | Treasury Regulations |
| Tax Foundation | `taxfoundation.org` | Policy analysis |
| Joint Committee on Taxation | `jct.gov` | Legislative analysis |
| Congressional Research Service | `crsreports.congress.gov` | Tax policy research |
| AICPA Tax Section | `aicpa.org/tax` | Professional guidance |
| Tax Policy Center | `taxpolicycenter.org` | Economic analysis |

---

## 10. Key Recommendations

### For This Project

1. **Start with Free Tier**: 1,000 credits/month is sufficient for initial content gathering and testing the pipeline.

2. **Use Extract over Search for known URLs**: Since we know IRS publication URLs follow predictable patterns (`irs.gov/publications/p{number}`), skip search and go directly to extract for known documents.

3. **Use Search for discovery**: Use search with `include_domains=["irs.gov"]` to find Revenue Rulings, Procedures, and other guidance documents that don't follow predictable URL patterns.

4. **Advanced extract depth is essential**: IRS documents contain complex tables (tax brackets, phase-out ranges, etc.) that basic extraction may miss.

5. **Markdown format preferred**: Maintains document structure (headings, tables, lists) which is critical for tax content understanding.

6. **Map first, then extract**: Use the Map endpoint to discover all reachable pages under `/publications/` and `/internal-revenue-bulletins/` before extracting.

7. **Budget for Bootstrap plan ($100/mo)**: For serious bulk gathering of 500+ pages with advanced extraction, expect to need ~15,000 credits. The Bootstrap plan at $100/month provides this.

8. **Async is mandatory for bulk work**: Use `AsyncTavilyClient` with `asyncio.gather()` for all batch operations to stay within rate limits while maximizing throughput.

9. **Note on PDFs**: Tavily Extract works on HTML pages. For IRS PDF documents (forms, instructions), you may need a separate PDF extraction pipeline. The HTML versions of publications at `irs.gov/publications/p{number}` are preferable.

10. **Consider Crawl access**: Apply for Crawl API access at `crawl.tavily.com` -- it would simplify the Map+Extract two-step process into a single operation for IRS.gov.

---

## Sources

- [Tavily API Documentation](https://docs.tavily.com/documentation/api-reference/introduction)
- [Tavily Search Endpoint](https://docs.tavily.com/documentation/api-reference/endpoint/search) (note: not reproduced, just referenced)
- [Tavily Extract Endpoint](https://docs.tavily.com/documentation/api-reference/endpoint/extract)
- [Tavily Credits & Pricing](https://docs.tavily.com/documentation/api-credits)
- [Tavily Python SDK (GitHub)](https://github.com/tavily-ai/tavily-python)
- [Tavily Python SDK Quick Start](https://docs.tavily.com/sdk/python/quick-start)
- [Tavily Best Practices - Search](https://docs.tavily.com/documentation/best-practices/best-practices-search)
- [Tavily Rate Limits](https://help.tavily.com/articles/3240802908-rate-limits)
- [IRS Publications Index](https://www.irs.gov/publications)
- [IRS Internal Revenue Bulletins](https://www.irs.gov/internal-revenue-bulletins)
