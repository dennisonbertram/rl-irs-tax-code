#!/usr/bin/env python3
"""
Comprehensive quality audit of final assembled training data.
Runs 7 validation checks and writes a report to docs/testing/final-data-quality-report.md
"""

import json
import os
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path("/Users/dennisonbertram/Develop/rl-irs-tax-code")
DATA_FINAL = PROJECT / "data" / "final"
DATA_REF   = PROJECT / "data" / "reference"
DOCS_DIR   = PROJECT / "docs" / "testing"

FILES = {
    "sft_train":  DATA_FINAL / "sft_train.jsonl",
    "sft_valid":  DATA_FINAL / "sft_valid.jsonl",
    "dpo_train":  DATA_FINAL / "dpo_train.jsonl",
    "dpo_valid":  DATA_FINAL / "dpo_valid.jsonl",
    "grpo_train": DATA_FINAL / "grpo_train.jsonl",
    "grpo_valid": DATA_FINAL / "grpo_valid.jsonl",
}

SECTION_TIERS_PATH  = DATA_REF / "section_tiers.json"
INFLATION_DATA_PATH = DATA_REF / "inflation_adjusted_amounts.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Load a .jsonl file, returning (records, bad_line_count)."""
    records = []
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad += 1
                print(f"  [WARN] Bad JSON at {path.name}:{i}: {exc}")
    return records, bad


def section_to_number(sec_str):
    """Extract leading integer from a section string like 'IRC §179A' or '1012'."""
    m = re.search(r"(\d+)", str(sec_str))
    return int(m.group(1)) if m else None


def get_tier(sec_str, tier1_set, tier3_ranges):
    """Return 1, 2, or 3 for a given section string."""
    # Normalise: strip IRC §, CFR §, whitespace
    clean = re.sub(r"(IRC|CFR|Treas\.?|Reg\.?|§|\s)", "", str(sec_str)).strip()
    # Match against tier-1 list
    m = re.match(r"^(\d+[A-Za-z]?)", clean)
    bare = m.group(1) if m else clean
    if bare in tier1_set:
        return 1
    num = section_to_number(clean)
    if num is not None:
        for r in tier3_ranges:
            if r["start"] <= num <= r["end"]:
                return 3
    return 2


def extract_dollar_amounts(text):
    """Return a list of numeric dollar values found in text."""
    # Matches $1,000 / $1,000,000 / $23,000 etc.
    raw = re.findall(r"\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)", text)
    results = []
    for r in raw:
        try:
            results.append(int(r.replace(",", "").split(".")[0]))
        except ValueError:
            pass
    return results


def levenshtein_ratio(s1, s2):
    """Cheap Levenshtein distance ratio using DP."""
    if s1 == s2:
        return 0.0
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 1.0
    # Only bother if lengths are close
    if abs(n - m) / max(n, m) > 0.3:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m] / max(n, m)


# ── Section 1: Structural Validation ──────────────────────────────────────────

def check_structural(records, file_key, file_type):
    """Validate structural requirements for each file type."""
    errors = []
    empty_fields = 0
    missing_meta = 0

    for i, rec in enumerate(records):
        if file_type == "sft":
            msgs = rec.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                errors.append(f"  Record {i}: missing/malformed 'messages'")
                continue
            roles = [m.get("role") for m in msgs]
            if "system" not in roles:
                errors.append(f"  Record {i}: no 'system' role")
            if "user" not in roles:
                errors.append(f"  Record {i}: no 'user' role")
            if "assistant" not in roles:
                errors.append(f"  Record {i}: no 'assistant' role")
            for m in msgs:
                if not m.get("content", "").strip():
                    empty_fields += 1
            if not rec.get("metadata"):
                missing_meta += 1

        elif file_type == "dpo":
            for field in ("prompt", "chosen", "rejected"):
                val = rec.get(field)
                if not isinstance(val, str) or not val.strip():
                    errors.append(f"  Record {i}: empty/missing '{field}'")
                    empty_fields += 1

        elif file_type == "grpo":
            for field in ("prompt", "expected_section"):
                val = rec.get(field)
                if not isinstance(val, str) or not val.strip():
                    errors.append(f"  Record {i}: empty/missing '{field}'")
                    empty_fields += 1

    return {
        "structural_errors": errors[:20],  # cap at 20 for display
        "structural_error_count": len(errors),
        "empty_field_count": empty_fields,
        "missing_meta_count": missing_meta,
        "pass": len(errors) == 0 and empty_fields == 0,
    }


# ── Section 2: Content Quality Sampling ───────────────────────────────────────

BOILERPLATE_ONLY = re.compile(
    r"^(consult a (qualified )?(tax )?professional\.?|please consult|i (am|'m) not a).*$",
    re.IGNORECASE | re.DOTALL,
)

TAX_CONCEPT_RE = re.compile(
    r"(IRC|§|section \d|26 U\.S\.C|Treas|deduct|credit|income|capital gain|"
    r"depreciat|amortiz|basis|gross income|taxable|exclusion|\$[0-9]|percent|%|"
    r"Form \d|Schedule [A-Z]|withhold|return|election|treatment)",
    re.IGNORECASE,
)

QUESTION_RE = re.compile(
    r"(\?$|^(what|how|when|why|who|which|can|does|is|are|do|will|may|should|"
    r"explain|describe|define|list|identify|summarize)[\s,])",
    re.IGNORECASE,
)


def rate_sft_record(rec):
    msgs = rec.get("messages", [])
    question = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    answer   = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")

    issues = []

    # Answer length
    if len(answer) < 100:
        issues.append("answer < 100 chars")

    # Not purely boilerplate
    if BOILERPLATE_ONLY.match(answer.strip()):
        issues.append("boilerplate-only answer")

    # Contains tax concept
    if not TAX_CONCEPT_RE.search(answer):
        issues.append("no tax concept in answer")

    # Question is real question
    if not QUESTION_RE.search(question.strip()):
        issues.append("question not clearly interrogative")

    if not issues:
        return "PASS", issues
    elif len(issues) == 1:
        return "MARGINAL", issues
    else:
        return "FAIL", issues


def check_content_quality(records, n=50):
    random.seed(42)
    sample = random.sample(records, min(n, len(records)))
    results = {"PASS": 0, "MARGINAL": 0, "FAIL": 0}
    fail_examples = []

    for rec in sample:
        rating, issues = rate_sft_record(rec)
        results[rating] += 1
        if rating != "PASS":
            msgs = rec.get("messages", [])
            q = next((m["content"][:80] for m in msgs if m.get("role") == "user"), "")
            fail_examples.append({"rating": rating, "issues": issues, "question": q})

    total = len(sample)
    quality_score = (results["PASS"] * 1.0 + results["MARGINAL"] * 0.5) / total * 100

    return {
        "sample_size": total,
        "pass_count": results["PASS"],
        "marginal_count": results["MARGINAL"],
        "fail_count": results["FAIL"],
        "quality_score_pct": round(quality_score, 1),
        "fail_examples": fail_examples[:10],
        "pass": quality_score >= 70,
    }


# ── Section 3: Source Distribution Analysis ───────────────────────────────────

def check_source_distribution(sft_train, tier1_set, tier3_ranges):
    source_counts = Counter()
    tier_counts   = Counter()
    section_counts = Counter()

    for rec in sft_train:
        meta = rec.get("metadata", {})

        # Source
        src = meta.get("source", "")
        if not src:
            # Infer from source_section
            sec = meta.get("source_section", "")
            if "CFR" in sec:
                src = "CFR"
            elif "IRC" in sec or sec.startswith("§"):
                src = "IRC"
            elif meta.get("category") == "inflation_adjusted":
                src = "inflation_adjusted"
            else:
                src = "unknown"
        source_counts[src] += 1

        # Tier
        sec_str = meta.get("source_section", "")
        tier = get_tier(sec_str, tier1_set, tier3_ranges)
        tier_counts[tier] += 1

        # Section
        section_counts[sec_str] += 1

    # Tier 1 coverage
    tier1_covered = set()
    for rec in sft_train:
        sec_str = rec.get("metadata", {}).get("source_section", "")
        clean = re.sub(r"(IRC|CFR|Treas\.?|Reg\.?|§|\s)", "", str(sec_str)).strip()
        m = re.match(r"^(\d+[A-Za-z]?)", clean)
        bare = m.group(1) if m else clean
        if bare in tier1_set:
            tier1_covered.add(bare)

    tier1_missing = sorted(tier1_set - tier1_covered)

    return {
        "source_counts": dict(source_counts.most_common(20)),
        "tier_counts": dict(tier_counts),
        "top_20_sections": section_counts.most_common(20),
        "bottom_20_sections": section_counts.most_common()[:-21:-1],
        "tier1_total": len(tier1_set),
        "tier1_covered": len(tier1_covered),
        "tier1_missing": tier1_missing[:30],
        "pass": len(tier1_missing) <= 20,
    }


# ── Section 4: Inflation Amount Verification ──────────────────────────────────

def _flatten_inflation(data, year=None, prefix=""):
    """Recursively yield (description, value) from inflation_adjusted_amounts.json."""
    for k, v in data.items():
        if isinstance(v, dict):
            yield from _flatten_inflation(v, year, prefix + k + ".")
        elif isinstance(v, (int, float)):
            yield (prefix + k, int(v))


def check_inflation_amounts(sft_train, inflation_data):
    # Collect inflation records
    inflation_records = [
        r for r in sft_train
        if r.get("metadata", {}).get("category") == "inflation_adjusted"
    ]

    if not inflation_records:
        return {"inflation_record_count": 0, "pass": True, "note": "no inflation records found"}

    # Build a flat set of valid dollar values across all years
    valid_amounts = set()
    tax_years = inflation_data.get("tax_years", {})
    for year_data in tax_years.values():
        for _, val in _flatten_inflation(year_data):
            valid_amounts.add(val)

    random.seed(42)
    sample = random.sample(inflation_records, min(10, len(inflation_records)))
    checked = []
    mismatches = 0

    for rec in sample:
        msgs = rec.get("messages", [])
        answer = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        amounts = extract_dollar_amounts(answer)
        meta = rec.get("metadata", {})
        year = str(meta.get("tax_year", ""))
        year_amounts = set()
        if year and year in tax_years:
            for _, v in _flatten_inflation(tax_years[year]):
                year_amounts.add(v)

        found_valid = []
        found_invalid = []
        for amt in amounts:
            if amt in valid_amounts:
                found_valid.append(amt)
            else:
                found_invalid.append(amt)

        status = "PASS" if not amounts or found_valid else "UNVERIFIED"
        if found_invalid and not found_valid:
            status = "MISMATCH"
            mismatches += 1

        msgs_user = next((m["content"][:80] for m in msgs if m.get("role") == "user"), "")
        checked.append({
            "question": msgs_user,
            "amounts_found": amounts,
            "valid_matches": found_valid,
            "unverified": found_invalid,
            "status": status,
        })

    return {
        "inflation_record_count": len(inflation_records),
        "sample_checked": len(sample),
        "mismatches": mismatches,
        "checks": checked,
        "pass": mismatches == 0,
    }


# ── Section 5: Train/Eval Leakage Check ───────────────────────────────────────

def get_question(rec, file_type):
    if file_type == "sft":
        msgs = rec.get("messages", [])
        return next((m["content"] for m in msgs if m.get("role") == "user"), "")
    elif file_type in ("dpo", "grpo"):
        return rec.get("prompt", "")
    return ""


def check_leakage(train_records, valid_records, file_type):
    train_qs = set(get_question(r, file_type) for r in train_records)
    leaks = []
    for r in valid_records:
        q = get_question(r, file_type)
        if q and q in train_qs:
            leaks.append(q[:120])
    return {
        "train_count": len(train_records),
        "valid_count": len(valid_records),
        "leak_count": len(leaks),
        "examples": leaks[:5],
        "pass": len(leaks) == 0,
    }


# ── Section 6: Duplicate Analysis ─────────────────────────────────────────────

def check_duplicates(records, file_type):
    questions = [get_question(r, file_type) for r in records]
    exact_counts = Counter(questions)
    exact_dupes = {q: c for q, c in exact_counts.items() if c > 1}
    exact_dupe_records = sum(c - 1 for c in exact_dupes.values())

    # Near-duplicate: same first 50 chars
    prefix_counts = Counter(q[:50] for q in questions if q)
    near_dupe_prefixes = {p: c for p, c in prefix_counts.items() if c > 1}
    near_dupe_records = sum(c - 1 for c in near_dupe_prefixes.values())

    total = len(questions)
    exact_rate = exact_dupe_records / total * 100 if total else 0
    near_rate  = near_dupe_records / total * 100 if total else 0

    return {
        "total_records": total,
        "exact_duplicate_pairs": len(exact_dupes),
        "exact_duplicate_records": exact_dupe_records,
        "exact_rate_pct": round(exact_rate, 2),
        "near_duplicate_records": near_dupe_records,
        "near_rate_pct": round(near_rate, 2),
        "examples": list(exact_dupes.keys())[:5],
        "pass": exact_rate < 5.0 and near_rate < 10.0,
    }


# ── Section 7: DPO Quality Check ──────────────────────────────────────────────

def check_dpo_quality(dpo_records, n=20):
    random.seed(42)
    sample = random.sample(dpo_records, min(n, len(dpo_records)))
    results = []
    issues_found = 0

    for rec in sample:
        chosen  = rec.get("chosen", "")
        rejected = rec.get("rejected", "")
        prompt  = rec.get("prompt", "")
        issues  = []

        # chosen and rejected must differ substantially
        if chosen.strip() == rejected.strip():
            issues.append("chosen == rejected (identical)")
        elif levenshtein_ratio(chosen, rejected) < 0.05:
            issues.append("chosen and rejected nearly identical")

        # chosen should be longer or at least as detailed as rejected
        if len(chosen) < len(rejected) * 0.5:
            issues.append("chosen much shorter than rejected (may be lower quality)")

        # rejected must not be garbage (must be non-trivial)
        if len(rejected.strip()) < 20:
            issues.append("rejected is too short (likely garbage)")

        # chosen should look more authoritative (has section refs)
        chosen_has_ref  = bool(re.search(r"(§|IRC|Treas\.|CFR|IRS Notice|Rev\. Proc\.)", chosen))
        rejected_has_ref = bool(re.search(r"(§|IRC|Treas\.|CFR|IRS Notice|Rev\. Proc\.)", rejected))
        if not chosen_has_ref and not rejected_has_ref:
            issues.append("neither chosen nor rejected has legal citation")

        if issues:
            issues_found += 1
        results.append({
            "prompt": prompt[:80],
            "chosen_len": len(chosen),
            "rejected_len": len(rejected),
            "issues": issues,
            "status": "PASS" if not issues else "WARN",
        })

    return {
        "sample_size": len(sample),
        "issues_found": issues_found,
        "details": results,
        "pass": issues_found / len(sample) < 0.25 if sample else True,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FINAL TRAINING DATA QUALITY AUDIT")
    print("=" * 70)

    # Load reference data
    print("\nLoading reference data...")
    with open(SECTION_TIERS_PATH) as f:
        tiers_data = json.load(f)
    tier1_set    = set(tiers_data.get("tier1", []))
    tier3_ranges = tiers_data.get("_tier3_ranges", [])

    with open(INFLATION_DATA_PATH) as f:
        inflation_data = json.load(f)

    # Load all datasets
    print("Loading datasets...")
    loaded = {}
    bad_lines = {}
    for key, path in FILES.items():
        recs, bad = load_jsonl(path)
        loaded[key]    = recs
        bad_lines[key] = bad
        print(f"  {key}: {len(recs):,} records  (bad lines: {bad})")

    # ── Check 1: Structural ────────────────────────────────────────────────────
    print("\n[1] Structural Validation...")
    structural = {}
    type_map = {
        "sft_train": "sft", "sft_valid": "sft",
        "dpo_train": "dpo", "dpo_valid": "dpo",
        "grpo_train": "grpo", "grpo_valid": "grpo",
    }
    for key, recs in loaded.items():
        res = check_structural(recs, key, type_map[key])
        structural[key] = res
        status = "PASS" if res["pass"] else "FAIL"
        print(f"  {key}: {status}  (errors={res['structural_error_count']}, empty={res['empty_field_count']})")

    # ── Check 2: Content Quality ───────────────────────────────────────────────
    print("\n[2] Content Quality Sampling (50 random SFT train records)...")
    quality = check_content_quality(loaded["sft_train"], n=50)
    status = "PASS" if quality["pass"] else "FAIL"
    print(f"  Quality score: {quality['quality_score_pct']}%  ({quality['pass_count']} PASS, "
          f"{quality['marginal_count']} MARGINAL, {quality['fail_count']} FAIL)  -> {status}")

    # ── Check 3: Source Distribution ──────────────────────────────────────────
    print("\n[3] Source Distribution Analysis...")
    dist = check_source_distribution(loaded["sft_train"], tier1_set, tier3_ranges)
    print(f"  Source counts: {dist['source_counts']}")
    print(f"  Tier counts: {dist['tier_counts']}")
    print(f"  Tier 1 coverage: {dist['tier1_covered']}/{dist['tier1_total']} sections")
    print(f"  Tier 1 missing (first 10): {dist['tier1_missing'][:10]}")
    status = "PASS" if dist["pass"] else "WARN"
    print(f"  Status: {status}")

    # ── Check 4: Inflation Amounts ─────────────────────────────────────────────
    print("\n[4] Inflation Amount Verification...")
    inflation = check_inflation_amounts(loaded["sft_train"], inflation_data)
    status = "PASS" if inflation["pass"] else "FAIL"
    print(f"  Inflation records: {inflation['inflation_record_count']}")
    print(f"  Checked {inflation.get('sample_checked', 0)}, mismatches: {inflation.get('mismatches', 0)}  -> {status}")

    # ── Check 5: Leakage ──────────────────────────────────────────────────────
    print("\n[5] Train/Eval Leakage Check...")
    leakage = {}
    for ft in ("sft", "dpo", "grpo"):
        res = check_leakage(loaded[f"{ft}_train"], loaded[f"{ft}_valid"], ft)
        leakage[ft] = res
        status = "PASS" if res["pass"] else "FAIL"
        print(f"  {ft}: {res['leak_count']} leaks  -> {status}")

    # ── Check 6: Duplicates ────────────────────────────────────────────────────
    print("\n[6] Duplicate Analysis (SFT train)...")
    dupes = check_duplicates(loaded["sft_train"], "sft")
    status = "PASS" if dupes["pass"] else "FAIL"
    print(f"  Exact dupes: {dupes['exact_duplicate_records']} ({dupes['exact_rate_pct']}%)")
    print(f"  Near dupes:  {dupes['near_duplicate_records']} ({dupes['near_rate_pct']}%)")
    print(f"  Status: {status}")

    # ── Check 7: DPO Quality ──────────────────────────────────────────────────
    print("\n[7] DPO Quality Check (20 random pairs)...")
    dpo_q = check_dpo_quality(loaded["dpo_train"], n=20)
    status = "PASS" if dpo_q["pass"] else "WARN"
    print(f"  Issues found in {dpo_q['issues_found']}/{dpo_q['sample_size']} records  -> {status}")

    # ── Write Report ───────────────────────────────────────────────────────────
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_DIR / "final-data-quality-report.md"

    overall_checks = [
        all(v["pass"] for v in structural.values()),
        quality["pass"],
        dist["pass"],
        inflation["pass"],
        all(v["pass"] for v in leakage.values()),
        dupes["pass"],
        dpo_q["pass"],
    ]
    overall = "PASS" if all(overall_checks) else ("WARN" if sum(overall_checks) >= 5 else "FAIL")

    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write("# Final Training Data Quality Report\n\n")
        rpt.write(f"**Overall Verdict: {overall}** ({sum(overall_checks)}/7 checks passing)\n\n")
        rpt.write(f"Generated: 2026-03-27\n\n")
        rpt.write("---\n\n")

        # Dataset summary
        rpt.write("## Dataset Summary\n\n")
        rpt.write("| File | Records | Bad Lines |\n")
        rpt.write("|------|---------|----------|\n")
        for key in FILES:
            rpt.write(f"| {key} | {len(loaded[key]):,} | {bad_lines[key]} |\n")
        rpt.write("\n")

        # Check 1
        rpt.write("## Check 1: Structural Validation\n\n")
        overall_struct = all(v["pass"] for v in structural.values())
        rpt.write(f"**Result: {'PASS' if overall_struct else 'FAIL'}**\n\n")
        rpt.write("| File | Structural Errors | Empty Fields | Missing Metadata | Status |\n")
        rpt.write("|------|------------------:|-------------:|-----------------:|--------|\n")
        for key, res in structural.items():
            s = "PASS" if res["pass"] else "FAIL"
            rpt.write(f"| {key} | {res['structural_error_count']} | {res['empty_field_count']} | {res['missing_meta_count']} | {s} |\n")
        # Show any error samples
        for key, res in structural.items():
            if res["structural_errors"]:
                rpt.write(f"\n**{key} errors (first 5):**\n")
                for e in res["structural_errors"][:5]:
                    rpt.write(f"- {e}\n")
        rpt.write("\n")

        # Check 2
        rpt.write("## Check 2: Content Quality Sampling\n\n")
        rpt.write(f"**Result: {'PASS' if quality['pass'] else 'FAIL'}** — Quality Score: {quality['quality_score_pct']}%\n\n")
        rpt.write(f"Sample size: {quality['sample_size']} random SFT train records\n\n")
        rpt.write(f"- PASS: {quality['pass_count']}\n")
        rpt.write(f"- MARGINAL: {quality['marginal_count']}\n")
        rpt.write(f"- FAIL: {quality['fail_count']}\n\n")
        if quality["fail_examples"]:
            rpt.write("**Non-passing examples:**\n\n")
            for ex in quality["fail_examples"][:10]:
                rpt.write(f"- [{ex['rating']}] `{ex['question']}` — {', '.join(ex['issues'])}\n")
        rpt.write("\n")

        # Check 3
        rpt.write("## Check 3: Source Distribution Analysis\n\n")
        rpt.write(f"**Result: {'PASS' if dist['pass'] else 'WARN'}**\n\n")
        rpt.write("### Source Counts\n\n")
        rpt.write("| Source | Count |\n|--------|------:|\n")
        for src, cnt in dist["source_counts"].items():
            rpt.write(f"| {src} | {cnt:,} |\n")
        rpt.write("\n### Tier Counts\n\n")
        rpt.write("| Tier | Count |\n|------|------:|\n")
        for tier in sorted(dist["tier_counts"]):
            rpt.write(f"| Tier {tier} | {dist['tier_counts'][tier]:,} |\n")
        rpt.write(f"\n### Tier 1 Coverage\n\n")
        rpt.write(f"- Tier 1 sections in reference: {dist['tier1_total']}\n")
        rpt.write(f"- Tier 1 sections covered: {dist['tier1_covered']}\n")
        coverage_pct = round(dist['tier1_covered'] / dist['tier1_total'] * 100, 1) if dist['tier1_total'] else 0
        rpt.write(f"- Coverage: {coverage_pct}%\n\n")
        if dist["tier1_missing"]:
            rpt.write(f"**Tier 1 sections with 0 coverage (first 30):** {', '.join(dist['tier1_missing'][:30])}\n\n")
        rpt.write("### Top 20 Most Represented Sections\n\n")
        rpt.write("| Section | Count |\n|---------|------:|\n")
        for sec, cnt in dist["top_20_sections"]:
            rpt.write(f"| {sec} | {cnt} |\n")
        rpt.write("\n### Bottom 20 Least Represented Sections\n\n")
        rpt.write("| Section | Count |\n|---------|------:|\n")
        for sec, cnt in dist["bottom_20_sections"]:
            rpt.write(f"| {sec} | {cnt} |\n")
        rpt.write("\n")

        # Check 4
        rpt.write("## Check 4: Inflation Amount Verification\n\n")
        rpt.write(f"**Result: {'PASS' if inflation['pass'] else 'FAIL'}**\n\n")
        rpt.write(f"- Inflation-adjusted records: {inflation['inflation_record_count']}\n")
        rpt.write(f"- Sample checked: {inflation.get('sample_checked', 0)}\n")
        rpt.write(f"- Mismatches: {inflation.get('mismatches', 0)}\n\n")
        checks = inflation.get("checks", [])
        if checks:
            rpt.write("| Question | Amounts Found | Valid Matches | Status |\n")
            rpt.write("|----------|:-------------|:-------------|--------|\n")
            for c in checks:
                rpt.write(f"| {c['question']} | {c['amounts_found']} | {c['valid_matches']} | {c['status']} |\n")
        rpt.write("\n")

        # Check 5
        rpt.write("## Check 5: Train/Eval Leakage\n\n")
        all_pass_leak = all(v["pass"] for v in leakage.values())
        rpt.write(f"**Result: {'PASS' if all_pass_leak else 'FAIL'}**\n\n")
        rpt.write("| Split | Train | Valid | Leaks | Status |\n")
        rpt.write("|-------|------:|------:|------:|--------|\n")
        for ft, res in leakage.items():
            s = "PASS" if res["pass"] else "FAIL"
            rpt.write(f"| {ft} | {res['train_count']:,} | {res['valid_count']:,} | {res['leak_count']} | {s} |\n")
        for ft, res in leakage.items():
            if res["examples"]:
                rpt.write(f"\n**{ft} leaking questions (first 3):**\n")
                for ex in res["examples"][:3]:
                    rpt.write(f"- `{ex[:100]}`\n")
        rpt.write("\n")

        # Check 6
        rpt.write("## Check 6: Duplicate Analysis\n\n")
        rpt.write(f"**Result: {'PASS' if dupes['pass'] else 'FAIL'}**\n\n")
        rpt.write(f"- Total SFT train records: {dupes['total_records']:,}\n")
        rpt.write(f"- Exact duplicate records: {dupes['exact_duplicate_records']:,} ({dupes['exact_rate_pct']}%)\n")
        rpt.write(f"- Near-duplicate records (same first 50 chars): {dupes['near_duplicate_records']:,} ({dupes['near_rate_pct']}%)\n\n")
        if dupes["examples"]:
            rpt.write("**Example duplicate questions:**\n")
            for ex in dupes["examples"][:5]:
                rpt.write(f"- `{ex[:100]}`\n")
        rpt.write("\n")

        # Check 7
        rpt.write("## Check 7: DPO Quality Check\n\n")
        rpt.write(f"**Result: {'PASS' if dpo_q['pass'] else 'WARN'}**\n\n")
        rpt.write(f"- Sample size: {dpo_q['sample_size']}\n")
        rpt.write(f"- Records with issues: {dpo_q['issues_found']}\n\n")
        warn_examples = [d for d in dpo_q["details"] if d["status"] == "WARN"]
        if warn_examples:
            rpt.write("**Pairs with issues:**\n\n")
            for ex in warn_examples:
                rpt.write(f"- `{ex['prompt']}` — {', '.join(ex['issues'])}\n")
        rpt.write("\n")

        # Summary and comparison
        rpt.write("## Summary and Recommendations\n\n")
        rpt.write("### Check Results Summary\n\n")
        check_names = [
            "1. Structural Validation",
            "2. Content Quality Sampling",
            "3. Source Distribution",
            "4. Inflation Amount Verification",
            "5. Train/Eval Leakage",
            "6. Duplicate Analysis",
            "7. DPO Quality",
        ]
        for name, passed in zip(check_names, overall_checks):
            icon = "PASS" if passed else "WARN/FAIL"
            rpt.write(f"- {icon}: {name}\n")
        rpt.write(f"\n**Overall: {overall}**\n\n")

        # Old vs new comparison
        rpt.write("### Old Data vs New Data Comparison\n\n")
        rpt.write("| Metric | Old (v1) | New (v3) |\n")
        rpt.write("|--------|:--------:|:--------:|\n")
        rpt.write("| SFT Train Records | 27,600 | 59,341 |\n")
        rpt.write("| SFT Valid Records | ~3,000 | 6,277 |\n")
        rpt.write("| Generation Method | Template-based | Grounded IRC + CFR |\n")
        rpt.write("| DPO Pairs | unknown | 1,708 train / 181 valid |\n")
        rpt.write("| GRPO Records | unknown | 58,991 train / 6,277 valid |\n")
        rpt.write("| Inflation Records | 0 | 350 (5x weighted) |\n")
        rpt.write("| CFR Coverage | None | ~41,269 records |\n\n")

        rpt.write("### Recommendations\n\n")

        recs_list = []

        # Structural issues
        total_struct_errors = sum(v["structural_error_count"] for v in structural.values())
        if total_struct_errors > 0:
            recs_list.append(
                f"**Structural**: {total_struct_errors} structural errors detected. "
                "Investigate records missing required fields before training."
            )

        # Quality
        if not quality["pass"]:
            recs_list.append(
                f"**Content Quality**: Score {quality['quality_score_pct']}% is below 70% threshold. "
                "Review boilerplate-heavy or insufficiently substantive answers."
            )
        else:
            recs_list.append(
                f"**Content Quality**: Score {quality['quality_score_pct']}% passes the 70% threshold. "
                "Note: 'consult a professional' postscript appears frequently but doesn't render answers invalid."
            )

        # Missing tier 1 sections
        if dist["tier1_missing"]:
            recs_list.append(
                f"**Coverage Gaps**: {len(dist['tier1_missing'])} Tier 1 sections have zero coverage "
                f"(e.g., {', '.join(dist['tier1_missing'][:5])}). "
                "Consider augmenting with targeted generation for these high-priority sections."
            )

        # Leakage
        total_leaks = sum(v["leak_count"] for v in leakage.values())
        if total_leaks > 0:
            recs_list.append(
                f"**Leakage**: {total_leaks} questions appear in both train and valid splits. "
                "Re-split or remove overlapping records."
            )
        else:
            recs_list.append("**Leakage**: No train/valid leakage detected — splits are clean.")

        # Duplicates
        if not dupes["pass"]:
            recs_list.append(
                f"**Duplicates**: {dupes['exact_rate_pct']}% exact duplicate rate exceeds 5% threshold. "
                "Run deduplication before final training."
            )
        else:
            recs_list.append(
                f"**Duplicates**: {dupes['exact_rate_pct']}% exact duplicate rate is within acceptable range."
            )

        # DPO
        dpo_warn_rate = dpo_q['issues_found'] / dpo_q['sample_size'] * 100 if dpo_q['sample_size'] else 0
        if not dpo_q["pass"]:
            recs_list.append(
                f"**DPO**: {dpo_warn_rate:.0f}% of sampled pairs have quality warnings. "
                "Review DPO pairs where chosen/rejected are too similar."
            )
        else:
            recs_list.append(
                f"**DPO**: {dpo_warn_rate:.0f}% of sampled pairs have minor warnings — "
                "quality is acceptable for training."
            )

        for r in recs_list:
            rpt.write(f"- {r}\n")

        rpt.write("\n---\n\n*Report generated by `scripts/validate_final_data.py`*\n")

    print(f"\nReport written to: {report_path}")
    print(f"\nOverall verdict: {overall} ({sum(overall_checks)}/7 checks passing)")

    # Return structured summary for stdout
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    return overall


if __name__ == "__main__":
    main()
