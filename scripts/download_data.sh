#!/usr/bin/env bash
# download_data.sh
# Downloads IRS Internal Revenue Code (Title 26) and Treasury Regulations (26 CFR)
# in machine-readable XML format for the rl-irs-tax-code project.
#
# Usage: bash scripts/download_data.sh
#
# IRC source: https://uscode.house.gov/download/download.shtml
# CFR source: https://www.govinfo.gov/bulkdata/ECFR/title-26/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

IRC_DIR="$PROJECT_ROOT/data/raw/irc"
CFR_DIR="$PROJECT_ROOT/data/raw/cfr"
PROCESSED_DIR="$PROJECT_ROOT/data/processed"

echo "=== rl-irs-tax-code Data Download Script ==="
echo "Project root: $PROJECT_ROOT"

# Create directory structure
mkdir -p "$IRC_DIR" "$CFR_DIR" "$PROCESSED_DIR"

# ---------------------------------------------------------------------------
# IRC Title 26 (Internal Revenue Code) XML
# Source: Office of the Law Revision Counsel (OLRC), uscode.house.gov
# Format: USLM XML schema (http://xml.house.gov/schemas/uslm/1.0)
# Current release point: 119-73not60 (as of early 2026)
# ---------------------------------------------------------------------------
echo ""
echo "--- Downloading IRC Title 26 XML ---"

IRC_URL_PRIMARY="https://uscode.house.gov/download/releasepoints/us/pl/119/73not60/xml_usc26@119-73not60.zip"
IRC_URL_FALLBACK1="https://uscode.house.gov/download/releasepoints/us/pl/118/200/xml_usc26@118-200.zip"
IRC_ZIP="$IRC_DIR/irc_title26.zip"

download_irc() {
    local url="$1"
    echo "Trying: $url"
    if curl -L --max-time 180 -o "$IRC_ZIP" "$url"; then
        echo "Download succeeded from: $url"
        return 0
    else
        echo "Download failed from: $url"
        return 1
    fi
}

if [ ! -f "$IRC_DIR/usc26.xml" ]; then
    download_irc "$IRC_URL_PRIMARY" \
        || download_irc "$IRC_URL_FALLBACK1" \
        || { echo "ERROR: All IRC download attempts failed. Check https://uscode.house.gov/download/download.shtml for the current release point."; exit 1; }

    echo "Extracting IRC XML..."
    cd "$IRC_DIR" && unzip -o irc_title26.zip
    echo "IRC XML extracted: $(ls -lh "$IRC_DIR/usc26.xml")"
else
    echo "IRC XML already exists, skipping download."
fi

IRC_SECTION_COUNT=$(grep -c '<section ' "$IRC_DIR/usc26.xml" 2>/dev/null || echo "unknown")
echo "IRC sections found: $IRC_SECTION_COUNT"

# ---------------------------------------------------------------------------
# Treasury Regulations (26 CFR) XML
# Source: GovInfo.gov eCFR bulk data
# Format: DLPSTEXTCLASS XML (ECFR schema)
# Amendment date: Dec. 18, 2025
# ---------------------------------------------------------------------------
echo ""
echo "--- Downloading 26 CFR (Treasury Regulations) XML ---"

CFR_URL="https://www.govinfo.gov/bulkdata/ECFR/title-26/ECFR-title26.xml"
CFR_FILE="$CFR_DIR/cfr_title26.xml"

if [ ! -f "$CFR_FILE" ]; then
    echo "Trying: $CFR_URL"
    if curl -L --max-time 600 -o "$CFR_FILE" "$CFR_URL"; then
        echo "CFR download succeeded."
        echo "CFR file size: $(ls -lh "$CFR_FILE")"
    else
        echo "WARNING: Single-file CFR download failed. Falling back to per-volume downloads..."
        for vol in $(seq 1 22); do
            VOL_URL="https://www.govinfo.gov/bulkdata/CFR/2024/title-26/CFR-2024-title26-vol${vol}.xml"
            VOL_FILE="$CFR_DIR/cfr-2024-title26-vol${vol}.xml"
            echo "  Volume $vol: $VOL_URL"
            curl -L --max-time 300 -o "$VOL_FILE" "$VOL_URL" 2>/dev/null \
                && echo "    -> saved: $(ls -lh "$VOL_FILE")" \
                || echo "    -> skipped (not found)"
        done
    fi
else
    echo "CFR XML already exists, skipping download."
fi

CFR_SECTION_COUNT=$(grep -c '<DIV8\|<SECTION' "$CFR_FILE" 2>/dev/null || echo "unknown")
echo "CFR sections/divisions found: $CFR_SECTION_COUNT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Download Summary ==="
echo "IRC Title 26 XML:  $IRC_DIR/usc26.xml"
ls -lh "$IRC_DIR/usc26.xml" 2>/dev/null || echo "  (not found)"
echo "26 CFR XML:        $CFR_FILE"
ls -lh "$CFR_FILE" 2>/dev/null || echo "  (not found)"
echo ""
echo "Done."
