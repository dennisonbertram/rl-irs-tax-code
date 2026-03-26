#!/usr/bin/env python3
"""
Parse CFR XML (eCFR DLPSTEXTCLASS schema) into clean text sections.
Output: data/processed/cfr_sections.jsonl

CFR structure uses DIV8 TYPE="SECTION" as the section-level element.
Each section has:
  - N attribute: section number (e.g., "§ 1.0-1")
  - HEAD child: section heading
  - P children: paragraph text
"""
import json
import re
from pathlib import Path

from lxml import etree

CFR_XML = Path(__file__).parent.parent / "data/raw/cfr/cfr_title26.xml"
OUTPUT = Path(__file__).parent.parent / "data/processed/cfr_sections.jsonl"


def clean_text(text: str) -> str:
    """Collapse whitespace and strip."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def extract_text_recursive(elem) -> str:
    """Extract all visible text from an element, recursively."""
    parts = []
    tag = elem.tag if isinstance(elem.tag, str) else ""
    local = tag.split("}")[-1] if "}" in tag else tag

    # Skip processing instructions and comments
    if not tag:
        return ""

    if elem.text:
        parts.append(elem.text)
    for child in elem:
        child_text = extract_text_recursive(child)
        if child_text:
            parts.append(child_text)
        if child.tail:
            parts.append(child.tail)

    return " ".join(p for p in parts if p.strip())


def parse_cfr_section(sec_elem) -> dict | None:
    """Parse a single DIV8 TYPE='SECTION' element."""
    section_num_raw = sec_elem.get("N", "").strip()
    node = sec_elem.get("NODE", "").strip()

    # Get heading
    head_elem = sec_elem.find("HEAD")
    if head_elem is None:
        return None
    heading_raw = clean_text(extract_text_recursive(head_elem))

    # Clean up the section number from heading (e.g., "§ 1.0-1   Internal Revenue Code...")
    # Extract section number from heading
    m = re.match(r"^§\s*([\w.()\-]+)\s+(.*)", heading_raw)
    if m:
        section_num = m.group(1)
        heading = clean_text(m.group(2))
    else:
        section_num = clean_text(section_num_raw).lstrip("§").strip()
        heading = heading_raw

    # Collect all text except the HEAD element
    text_parts = []
    for child in sec_elem:
        if child.tag == "HEAD":
            continue
        part = clean_text(extract_text_recursive(child))
        if part:
            text_parts.append(part)

    full_text = " ".join(text_parts)

    if not full_text and not heading:
        return None

    # Use section_num from N attribute if cleaner
    if not section_num:
        section_num = clean_text(section_num_raw).lstrip("§").strip()

    return {
        "section": section_num,
        "heading": heading,
        "text": full_text,
        "source": "CFR",
        "node": node,
    }


def main():
    print(f"Parsing {CFR_XML} ...")
    # Use iterparse for large file efficiency
    tree = etree.parse(str(CFR_XML))
    root = tree.getroot()

    # Find all DIV8 TYPE="SECTION" elements
    sections = [
        elem for elem in root.iter("DIV8") if elem.get("TYPE") == "SECTION"
    ]
    print(f"Found {len(sections)} DIV8 SECTION elements")

    records = []
    skipped = 0
    for sec in sections:
        rec = parse_cfr_section(sec)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    print(f"Parsed {len(records)} CFR sections ({skipped} skipped)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written to {OUTPUT}")

    # Print samples
    if records:
        sample = records[0]
        print(f"\nSample — Section {sample['section']}: {sample['heading']}")
        print(f"  Text preview: {sample['text'][:200]}...")

        # Print a few more section numbers to verify
        print("\nFirst 10 section numbers:")
        for r in records[:10]:
            print(f"  {r['section']}: {r['heading'][:60]}")


if __name__ == "__main__":
    main()
