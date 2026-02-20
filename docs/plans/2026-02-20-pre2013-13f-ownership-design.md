# Pre-2013 13F Ownership via EFTS Legacy Text Parsing — Design

## Problem

SEC bulk 13F data sets start at Q2 2013. For GME (IPO Feb 2002), this means
11 years of institutional ownership data are missing. The filings exist on
EDGAR and are searchable via EFTS back to ~2001, but they use heterogeneous
ASCII/SGML/TSV formats — not the standardized XML the current parser expects.

## Approach

Extend the existing EFTS fallback path with a legacy text parser. No new
dependencies, no new data sources — just a new `_parse_13f_text()` method
that kicks in when `_parse_13f_xml()` returns empty results.

**Scope:** GME-focused. The text parser extracts holdings by scanning for
lines containing the target CUSIP. Tolerates some missing quarters where
the filing format is too exotic to parse.

## Architecture

### Data flow

```
fetch_ownership_snapshots("GME", num_quarters=100)
  │
  ├─ Q1 2026 → Q2 2013: _fetch_from_bulk()  [unchanged]
  │
  └─ Q1 2013 → Q1 2003: _fetch_from_efts()  [extended]
       │
       ├─ EFTS query: CUSIP "36467W109", form 13F-HR
       ├─ For each hit: download infotable document
       ├─ Try _parse_13f_xml() first  [existing]
       └─ If empty → _parse_13f_text() fallback  [NEW]
```

### New method: `_parse_13f_text()`

Scans text content line-by-line for rows containing the target CUSIP.
Extracts shares and value (in $1000s) from numeric tokens on matching
lines. Returns `list[InstitutionalHolding]`.

Does NOT attempt to parse every column or detect the full table format.
Only needs the CUSIP match, plus the two key numeric fields.

### Changes to existing code

1. **`_fetch_from_efts()`** — after `_parse_13f_xml()` returns empty,
   call `_parse_13f_text()` on the same content as fallback.

2. **`_find_infotable_url()`** — also match `.txt` files when no `.xml`
   is found. Pre-2013 filings store the information table as plain text.

3. **`ownership_start_year`** in `SymbolInfo` — lower floor from 2013
   to 2003 (EFTS coverage starts ~2001; reliable 13F data ~2003).

4. **`fetch_ownership_snapshots()`** — update `min_year, min_quarter`
   floor from `(2013, 2)` to `(2003, 1)`.

## Files changed

| File | Change |
|------|--------|
| `src/stockdownloader/data/sec_ownership_client.py` | Add `_parse_13f_text()`, modify EFTS fallback, modify `_find_infotable_url()`, lower quarter floor |
| `src/stockdownloader/model/symbol_info.py` | Lower `ownership_start_year` floor from 2013 to 2003 |
| `tests/data/test_sec_ownership_client.py` | Add `TestLegacyTextParsing` tests |
| `tests/model/test_symbol_info.py` | Update `ownership_start_year` tests |

## Error handling

- Text parsing yields zero holdings → log warning, return `None` for that quarter
- Malformed lines → skip (log at DEBUG)
- EFTS returns 0 hits → normal, no institutions held that symbol for that quarter

## Constraints

- SEC rate limit: 10 req/sec (existing `_rate_limit()` handles this)
- EFTS coverage: ~2001 onward. Pre-2001 filings may not be indexed.
- Format heterogeneity: some quarters may fail to parse. Acceptable for GME-focused scope.
