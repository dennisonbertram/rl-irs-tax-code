# IRS Tax Code (Internal Revenue Code) - Machine-Readable Sources

**Date:** 2026-03-25
**Purpose:** Identify sources for obtaining the full Internal Revenue Code and supplementary IRS materials in machine-readable formats suitable for LLM training.

---

## 1. Internal Revenue Code (Title 26 of the U.S. Code)

### Primary Source: Office of the Law Revision Counsel (OLRC)

**Download page:** https://uscode.house.gov/download/download.shtml

This is the authoritative source. Title 26 is available in **four formats**:

| Format | URL Pattern | Notes |
|--------|------------|-------|
| **XML** | `https://uscode.house.gov/download/releasepoints/us/pl/119/73not60/xml_usc26@119-73not60.zip` | Best for structured parsing; uses USLM schema |
| **XHTML** | `https://uscode.house.gov/download/releasepoints/us/pl/119/73not60/htm_usc26@119-73not60.zip` | HTML-based, easier to read |
| **PCC** (text) | `https://uscode.house.gov/download/releasepoints/us/pl/119/73not60/pcc_usc26@119-73not60.zip` | GPO photocomposition codes; less useful |
| **PDF** | `https://uscode.house.gov/download/releasepoints/us/pl/119/73not60/pdf_usc26@119-73not60.zip` | Hardest to parse |

**Currency:** Files are current through Public Law 119-73 (01/23/2026), except PL 119-60.

> **Recommendation:** Use the **XML format**. It has structured tags for sections, subsections, paragraphs, and cross-references, making it ideal for preprocessing into training data.

### Secondary Sources

- **Cornell LII:** https://www.law.cornell.edu/uscode/text/26 — browsable HTML, no bulk download
- **GovInfo:** https://www.govinfo.gov/app/details/USCODE-2009-title26 — older editions available; XML bulk data exists at https://www.govinfo.gov/bulkdata/USCODE/
- **Justia:** https://law.justia.com/codes/us/title-26/ — browsable, no bulk download

---

## 2. Treasury Regulations (26 CFR) - Supplements the Code

The Code of Federal Regulations Title 26 contains the Treasury Regulations that implement the IRC. These are often more detailed than the statute itself.

### eCFR (Electronic Code of Federal Regulations)

- **Browse:** https://www.ecfr.gov/current/title-26
- **Bulk XML download:** https://www.govinfo.gov/bulkdata/ECFR/title-26/ECFR-title26.xml
- **API documentation:** https://www.ecfr.gov/developers/documentation/api/v1
- **GPO bulk data repo:** https://github.com/usgpo/bulk-data/blob/main/ECFR-XML-User-Guide.md

### Annual CFR (Archived editions)

- **GovInfo bulk data:** https://www.govinfo.gov/app/collection/cfr/
- **XML by year/title:** e.g., `https://www.govinfo.gov/bulkdata/CFR/2020/title-26/CFR-2020-title26-vol4.xml`
- Title 26 CFR spans **multiple volumes** (approximately 20 volumes)

> **Note:** The XML format of CFR is not yet an "official" legal format. Only PDF and Text have official legal status, but XML is best for machine processing.

---

## 3. IRS Publications, Instructions, and Guidance

### IRS Publications (PDF)

- **Index page:** https://www.irs.gov/forms-instructions-and-publications
- **Publications listing:** https://www.irs.gov/publications
- **Browser-friendly (HTML) versions:** https://www.irs.gov/forms-pubs/browser-friendly
- **Direct PDF access pattern:** `https://www.irs.gov/pub/irs-pdf/p{number}.pdf`
  - e.g., Publication 17: https://www.irs.gov/pub/irs-pdf/p17.pdf

Key publications for tax law understanding:
- **Pub 17** — Your Federal Income Tax (comprehensive individual guide)
- **Pub 334** — Tax Guide for Small Business
- **Pub 550** — Investment Income and Expenses
- **Pub 544** — Sales and Other Dispositions of Assets
- **Pub 535** — Business Expenses

### Internal Revenue Bulletins (IRBs)

- **URL:** https://www.irs.gov/internal-revenue-bulletins
- Contains Revenue Rulings, Revenue Procedures, Notices, Announcements
- Available in HTML and PDF on IRS.gov

### IRS FOIA Library

- **URL:** https://www.irs.gov/privacy-disclosure/foia-library
- Contains: Private Letter Rulings (PLRs), Technical Advice Memoranda (TAMs), Chief Counsel Advice (CCA)
- These provide interpretive guidance beyond the statute and regulations

---

## 4. Size Estimates

| Source | Approximate Size |
|--------|-----------------|
| **IRC (Title 26 statute only)** | ~2,652 pages (GPO print); ~1-1.2 million words |
| **IRC (digital file)** | ~3.4 million words (includes notes, amendments, cross-refs) |
| **26 CFR (Treasury Regulations)** | ~20 volumes; estimated 5-10 million words |
| **IRC + Regulations combined** | ~4 million words (National Taxpayer Advocate estimate) |
| **IRS Publications** | Hundreds of publications; estimated additional millions of words |
| **XML download of Title 26** | ZIP file likely 20-50 MB compressed |

For LLM training context: the core IRC + regulations would be roughly **10-20 million tokens** (using ~0.75 words per token). Including publications and guidance could push this to **50-100 million tokens**.

---

## 5. Existing Datasets on HuggingFace

| Dataset | URL | Contents | Relevance |
|---------|-----|----------|-----------|
| **pile-of-law/pile-of-law** | https://huggingface.co/datasets/pile-of-law/pile-of-law | Large legal corpus including IRS legal advice memos, US Code, CFR | **High** — includes IRC and regulatory text |
| **quotientai/irs_form_instruction_qa_pairs** | https://huggingface.co/datasets/quotientai/irs_form_instruction_qa_pairs | QA pairs from IRS form instructions | **Medium** — good for fine-tuning |
| **TrevorJS/irs-forms** | https://huggingface.co/datasets/TrevorJS/irs-forms | IRS forms data | **Low** — forms, not code text |
| **lexlms/lex_files_preprocessed** | https://huggingface.co/datasets/lexlms/lex_files_preprocessed | Multi-jurisdictional legal corpus including US law | **Medium** — broader legal text |
| **nguha/legalbench** | https://huggingface.co/datasets/nguha/legalbench | 162 legal reasoning evaluation tasks | **Low** — eval benchmark, not training data |
| **FiscalNote/billsum** | https://huggingface.co/datasets/FiscalNote/billsum | US Congressional bill summaries | **Low** — bills, not tax code |

### Other Relevant Sources

- **Tax Foundation data:** https://github.com/TaxFoundation/data — tax policy datasets and studies
- **Caselaw Access Project:** https://huggingface.co/datasets/common-pile/caselaw_access_project — 40M pages of court decisions (includes tax cases)

---

## 6. Preprocessing Recommendations for LLM Training

### For XML sources (IRC and CFR):

1. **Parse XML structure** — Extract section numbers, headings, and body text while preserving hierarchy
2. **Strip markup** — Remove XML tags but retain section numbering (e.g., "Section 401(k)")
3. **Resolve cross-references** — The IRC is heavily cross-referenced; consider inlining or annotating references
4. **Handle amendments** — Remove historical amendment notes unless you want temporal context
5. **Chunk by section** — Natural chunking boundary is at the section level (e.g., IRC Sec. 162)
6. **Preserve hierarchy** — Maintain subtitle > chapter > subchapter > part > section > subsection structure as metadata

### For PDF publications:

1. **PDF extraction** — Use tools like `pdfplumber`, `PyMuPDF`, or `marker-pdf` for text extraction
2. **Table handling** — Tax publications contain many tables (tax brackets, etc.); extract as structured data
3. **Example cleanup** — Remove page headers/footers, form references
4. **Deduplication** — Many publications overlap in content

### General preprocessing:

1. **Deduplication** — Between IRC, regulations, and publications there is significant overlap
2. **Quality filtering** — Remove boilerplate, table of contents, index pages
3. **Metadata tagging** — Tag each chunk with source (IRC/CFR/Pub), section number, topic, year
4. **Version tracking** — Tax law changes annually; tag with effective date/year
5. **Tokenization check** — Legal text has unusual tokens (section symbols, paragraph markers); verify tokenizer handles them

### Suggested pipeline:

```
1. Download XML from uscode.house.gov (IRC) and govinfo.gov (CFR)
2. Parse with lxml/BeautifulSoup
3. Extract structured text with section metadata
4. Download IRS publications as PDF, extract with marker-pdf
5. Combine all sources, deduplicate
6. Create train/val splits (by section or by source)
7. Output as JSONL with fields: {text, source, section, title, year}
```

---

## 7. Legal and Licensing Considerations

- **U.S. Code and CFR are public domain** — No copyright restrictions on federal law
- **IRS publications are U.S. government works** — Public domain under 17 USC 105
- **Pile of Law dataset** — Licensed CC BY-NC-SA 4.0 (non-commercial)
- **No restrictions on using the raw government sources** for any purpose including commercial LLM training

---

## Quick Start: Minimum Viable Download

To get started immediately with the core tax code text:

```bash
# Download IRC (Title 26) in XML
curl -o irc_title26.zip "https://uscode.house.gov/download/releasepoints/us/pl/119/73not60/xml_usc26@119-73not60.zip"
unzip irc_title26.zip -d irc_xml/

# Download Treasury Regulations (26 CFR) in XML
curl -o cfr_title26.xml "https://www.govinfo.gov/bulkdata/ECFR/title-26/ECFR-title26.xml"
```

These two downloads give you the complete statutory tax code and its implementing regulations in machine-readable XML.
