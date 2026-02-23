# Data Package Restructure — Design

**Date:** 2026-02-23
**Scope:** Break up 3 monster files in `src/stockdownloader/data/`, extract shared SEC helpers, fix inheritance inconsistency.

## Problem

The data package has three oversized files that each cram multiple responsibilities into a single module:

1. **`sec_insider_client.py` (1,616 lines)** — bulk ZIP downloading, TSV parsing, individual EDGAR filing parsing, 13D/13G EFTS search, CIK resolution, HTTP helpers, caching, module-level parsers
2. **`sec_ownership_client.py` (1,373 lines)** — bulk 13F ZIP downloading, XML/text infotable parsing, EFTS fallback, HTTP helpers, URL construction, caching
3. **`regsho_threshold_client.py` (1,028 lines)** — 4 separate exchange source fetchers (NYSE with TLS impersonation, OCC, Nasdaq, CBOE), response parsing, orchestration, caching

### Concrete Duplication

| Duplicated Code | sec_insider_client.py | sec_ownership_client.py |
|---|---|---|
| `_download_with_retry()` | Line 1262 | Line 782 |
| `_fetch_url_text()` | Line 1286 | Line 803 |
| `_fetch_efts_page()` | Line 1302 | Line 824 |
| `_prev_quarter()` | Line 1483 | Line 1369 |

Both import `SplitAdjustment` and `_KNOWN_SPLITS` from `sec_ftd_client` — the split data is defined in FTD but used by insider + ownership.

### Inheritance Inconsistency

- `SecInsiderClient(BaseDataClient)` — properly extends base
- `SecOwnershipClient` — reimplements its own `requests.Session`, rate limiting, and caching instead of extending `BaseDataClient`

### Missing Export

`SecInsiderClient` is NOT in `data/__init__.py` despite being used by scripts and pipeline.

## Approach: Extract + Split

Same pattern as the util/ restructure — extract shared code into common modules, split each monster file by responsibility, atomic moves with no backward-compat shims.

## New Layout

```
data/
├── sec_common.py               ← NEW: shared SEC HTTP helpers + quarter math
│                                   _download_with_retry(), _fetch_url_text(),
│                                   _fetch_efts_page(), _prev_quarter()
│
├── sec_insider_client.py        ← SLIM: orchestration + caching (~600 lines)
├── sec_insider_parsers.py       ← NEW: TSV bulk parsing, XML/HTML Form 3/4/5
│                                   parsing, 13D/13G content extraction (~500 lines)
│
├── sec_ownership_client.py      ← SLIM: orchestration + caching, extend BaseDataClient (~500 lines)
├── sec_ownership_parsers.py     ← NEW: 13F XML/text parsing, EFTS response parsing (~300 lines)
│
├── regsho_threshold_client.py   ← SLIM: orchestrator (~200 lines)
├── regsho_sources.py            ← NEW: 4 exchange fetchers (NYSE, OCC, Nasdaq, CBOE) (~600 lines)
├── regsho_parsers.py            ← NEW: shared parsing helpers (~200 lines)
│
├── base_client.py               ← UNCHANGED (294 lines)
├── sec_edgar_client.py          ← UNCHANGED
├── sec_ftd_client.py            ← UNCHANGED (keeps _KNOWN_SPLITS, SplitAdjustment)
├── yahoo_*.py                   ← UNCHANGED (well-organized)
├── finra_*.py                   ← UNCHANGED
├── ... (all other clients)      ← UNCHANGED
└── __init__.py                  ← MODIFY: add SecInsiderClient export
```

## Detailed Design

### 1. `sec_common.py` — Shared SEC HTTP Helpers

Standalone functions (not a class) used by both SEC clients:

```python
# HTTP helpers
def download_with_retry(session, url, dest_path, ...)  # was _download_with_retry
def fetch_url_text(session, url, ...)                   # was _fetch_url_text
def fetch_efts_page(session, url, ...)                  # was _fetch_efts_page

# Quarter math
def prev_quarter(year, quarter) -> tuple[int, int]      # was _prev_quarter (duplicated)

# Shared constants
SEC_RATE_LIMIT_DELAY = 0.11
EFTS_BASE_URL = "https://efts.sec.gov/LATEST/search-index"
```

Functions stay private-prefixed if only used internally by the SEC clients. Made public (no underscore) since they're now module-level exports.

### 2. `sec_insider_parsers.py` — Insider Filing Parsers

Extract all parsing logic from `sec_insider_client.py`:

- `parse_bulk_zip()` — parse Form 3/4/5 bulk ZIP TSV files (SUBMISSION, REPORTING_OWNER, NON_DERIVATIVE_TRANSACTION)
- `parse_form345_xml()` — parse individual Form 3/4/5 XML filings
- `parse_form345_html()` — parse individual Form 3/4/5 HTML filings
- `extract_13d_13g_data()` — parse 13D/13G filing content for share counts
- `normalize_date()` — date normalization helper
- `xml_text()` — XML text extraction helper
- `apply_split_to_transaction()` — split-adjust insider transactions

### 3. `sec_insider_client.py` — Slim Orchestrator

Keeps only orchestration + caching:

- `SecInsiderClient(BaseDataClient)` — unchanged inheritance
- `fetch_insider_transactions()` — calls parsers + sec_common helpers
- `fetch_beneficial_owners()` — calls parsers + sec_common helpers
- `fetch_insider_snapshot()` — aggregation
- `_resolve_cik()` — CIK resolution (client-specific, uses session)
- Cache load/save methods

### 4. `sec_ownership_parsers.py` — Ownership Filing Parsers

Extract parsing logic from `sec_ownership_client.py`:

- `parse_bulk_zip()` — parse 13F bulk ZIP TSV files
- `parse_13f_xml()` — parse XML infotable format
- `parse_13f_text()` — parse text infotable format
- `find_infotable_url()` — locate infotable within filing
- `get_xml_text()` — XML text extraction
- `filing_date_to_quarter_end()` — date mapping
- `apply_split_to_snapshot()` — split-adjust ownership data
- `quarter_range()` — quarter enumeration helper
- `bulk_zip_urls()` — URL construction for bulk data sets

### 5. `sec_ownership_client.py` — Slim Orchestrator, Extend BaseDataClient

- **Refactor to extend `BaseDataClient`** instead of reimplementing session/rate-limiting
- `SecOwnershipClient(BaseDataClient)` — use base class session, rate limiting, caching
- `fetch_ownership_snapshots()` — orchestration, calls parsers + sec_common
- Cache load/save methods

### 6. `regsho_sources.py` — Exchange Source Fetchers

Extract the 4 source fetchers, each self-contained:

- `query_nyse()` — NYSE with TLS impersonation (curl_cffi)
- `query_occ()` — OCC combined endpoint
- `query_nasdaq()` — Nasdaq threshold list
- `query_cboe()` — CBOE threshold list
- Per-source rate limiting stays with each fetcher

### 7. `regsho_parsers.py` — Shared RegSHO Parsing

- Pipe-delimited parsing (OCC, NYSE)
- CSV parsing (Nasdaq)
- HTML table parsing (CBOE)
- ThresholdRecord construction helpers

### 8. `regsho_threshold_client.py` — Slim Orchestrator

- `RegShoThresholdClient(BaseDataClient)` — unchanged inheritance
- `fetch_threshold_status()` — calls sources in priority order
- `fetch_threshold_targeted()` — targeted date range fetching
- `is_currently_on_threshold()` — convenience method
- Cache load/save

### 9. `data/__init__.py` — Add Missing Export

Add `SecInsiderClient` to imports and `__all__`.

## Migration Strategy

Atomic moves — same as util/ restructure:
1. Create new parser/helper files
2. Update client to import from new files
3. Remove extracted code from client
4. Update callers if import paths changed
5. Delete nothing extra (clients keep same names, just get slimmer)

Since client class names and their import paths don't change (`from stockdownloader.data.sec_insider_client import SecInsiderClient`), callers outside the data package need NO changes. Only internal wiring changes.

## Files Changed

| File | Action | Notes |
|------|--------|-------|
| `data/sec_common.py` | **NEW** | Shared SEC HTTP + quarter helpers |
| `data/sec_insider_parsers.py` | **NEW** | Insider filing parsers |
| `data/sec_insider_client.py` | **MODIFY** | Slim down, import from parsers + common |
| `data/sec_ownership_parsers.py` | **NEW** | Ownership filing parsers |
| `data/sec_ownership_client.py` | **MODIFY** | Slim down, extend BaseDataClient |
| `data/regsho_sources.py` | **NEW** | 4 exchange source fetchers |
| `data/regsho_parsers.py` | **NEW** | Shared RegSHO parsing |
| `data/regsho_threshold_client.py` | **MODIFY** | Slim orchestrator |
| `data/__init__.py` | **MODIFY** | Add SecInsiderClient export |
| `tests/data/test_sec_insider_client.py` | **MODIFY** | Update internal imports if needed |
| `tests/data/test_sec_ownership_client.py` | **MODIFY** | Update internal imports if needed |
| `tests/data/test_regsho_threshold_client.py` | **MODIFY** | Update internal imports if needed |

## What's NOT Changing

- Yahoo clients (well-organized with proper inheritance)
- Finra clients (reasonably sized)
- SEC EDGAR client, SEC FTD client (appropriately scoped)
- Polygon, Morningstar, Tradier, OCC options clients
- Base client (stays as-is, gets more users)
- No client class renames — external callers unaffected
