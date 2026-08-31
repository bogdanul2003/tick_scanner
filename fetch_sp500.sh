#!/usr/bin/env bash
# ==============================================================================
# Fetch S&P 500 Symbols from Wikipedia
#
# Description:
#   Downloads the S&P 500 constituent table from Wikipedia using curl,
#   parses the ticker symbols, and writes them to an output file (one per line).
#
# Usage:
#   ./fetch_sp500.sh [OUTPUT_FILE]
#
# Example:
#   ./fetch_sp500.sh sp500_symbols.txt
# ==============================================================================

set -euo pipefail

# Output file defaults to sp500_symbols.txt if not specified
OUTPUT_FILE="${1:-sp500_symbols.txt}"
URL="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

echo "Fetching S&P 500 companies list from Wikipedia..."

# 1. Fetch the HTML page using curl
# 2. Extract the table with id="constituents"
# 3. Match the first <td> of each row containing the ticker link
# 4. Extract the raw ticker text
# 5. Clean up any trailing/leading whitespace and save to output file
curl -s -L -A "$USER_AGENT" "$URL" \
  | sed -n '/id="constituents"/,/<\/table>/p' \
  | grep -E '^\s*<td[^>]*><a[^>]*class="external text"[^>]*>[A-Za-z0-9.-]+</a>' \
  | sed -E 's/.*<a[^>]*>([^<]+)<\/a>.*/\1/' \
  | tr -d '\r' \
  | grep -v '^$' > "$OUTPUT_FILE"

# Verify output
COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')

if [ "$COUNT" -eq 0 ]; then
  echo "Error: No symbols found. Please check your network connection or the Wikipedia page structure." >&2
  exit 1
fi

echo "Successfully saved $COUNT S&P 500 symbols to '$OUTPUT_FILE'."
