#!/usr/bin/env python3
"""
Parse IRC XML (USLM schema) into clean text sections.
Output: data/processed/irc_sections.jsonl
"""
import json
import re
import sys
from pathlib import Path

from lxml import etree

IRC_XML = Path(__file__).parent.parent / "data/raw/irc/usc26.xml"
OUTPUT = Path(__file__).parent.parent / "data/processed/irc_sections.jsonl"

NS = {"uslm": "http://xml.house.gov/schemas/uslm/1.0"}
USLM_NS = "http://xml.house.gov/schemas/uslm/1.0"


def clean_text(text: str) -> str:
    """Collapse whitespace and strip."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def extract_text_recursive(elem) -> str:
    """Extract all text from an element and its descendants, skipping table elements."""
    # Skip HTML table elements (embedded xhtml tables)
    local = etree.QName(elem.tag).localname if "}" in str(elem.tag) else elem.tag
    if local in ("table", "thead", "tbody", "tr", "th", "td", "colgroup", "col"):
        return ""

    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(extract_text_recursive(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(p for p in parts if p.strip())


def parse_section(sec_elem) -> dict | None:
    """Parse a single <section> element into a dict."""
    identifier = sec_elem.get("identifier", "")
    # Match /us/usc/t26/sXXX — only top-level sections, not subsections
    m = re.match(r"^/us/usc/t26/s(\w+)$", identifier)
    if not m:
        return None

    section_num = m.group(1)

    # Extract heading
    heading_elem = sec_elem.find(f"{{{USLM_NS}}}heading")
    heading = clean_text(heading_elem.text if heading_elem is not None else "")

    # Extract all text content from the section
    full_text = clean_text(extract_text_recursive(sec_elem))

    # Remove the section number prefix if it appears at the start
    # e.g. "§ 63. Taxable income defined" type prefix
    full_text = re.sub(r"^§\s*\w+\.\s*", "", full_text)

    if not full_text:
        return None

    return {
        "section": section_num,
        "heading": heading,
        "text": full_text,
        "source": "IRC",
    }


def main():
    print(f"Parsing {IRC_XML} ...")
    tree = etree.parse(str(IRC_XML))
    root = tree.getroot()

    # Find all top-level section elements
    sections = root.findall(f".//{{{USLM_NS}}}section")
    print(f"Found {len(sections)} <section> elements")

    records = []
    skipped = 0
    for sec in sections:
        rec = parse_section(sec)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    print(f"Parsed {len(records)} top-level IRC sections ({skipped} skipped)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written to {OUTPUT}")

    # Print a sample
    if records:
        sample = records[0]
        print(f"\nSample — Section {sample['section']}: {sample['heading']}")
        print(f"  Text preview: {sample['text'][:200]}...")


if __name__ == "__main__":
    main()
