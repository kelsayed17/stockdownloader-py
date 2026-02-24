# Round 13: Split Monoliths — data/ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split 3 data-layer monoliths (regsho_sources 684 lines, sec_insider_parsers 707 lines, sec_ownership_parsers 637 lines) into focused modules, targeting all files under 500 lines.

**Architecture:** Extract the single largest chunk from each file — NYSE query logic from regsho_sources, bulk ZIP parsing from each SEC parser — into dedicated modules. Original files re-export extracted functions for backward compatibility so no client import changes needed.

**Tech Stack:** Python, no new dependencies

---

### Task 1: Extract NYSE query logic from regsho_sources.py

**Files:**
- Create: `src/stockdownloader/data/nyse_regsho.py`
- Modify: `src/stockdownloader/data/regsho_sources.py`

**What to extract** — all NYSE-specific logic:

Move these to `nyse_regsho.py`:
- `nyse_rate_limit()` (lines 155-179) — NYSE-specific randomized delay + batch pausing
- `save_nyse_progress()` (lines 210-227) — persist queried dates to JSON
- `load_nyse_progress()` (lines 229-249) — load/migrate progress from legacy location
- `incremental_nyse_save()` (lines 251-286) — merge new records with cache
- `query_nyse()` (lines 293-461, **169 lines**) — the main NYSE query function

Also move NYSE-specific constants:
- `NYSE_DELAY_RANGE`, `NYSE_BATCH_SIZE`, `NYSE_BATCH_PAUSE_RANGE`, `NYSE_429_BACKOFF_RANGE`
- `NYSE_URL_TEMPLATE`
- `MAX_CONSECUTIVE_FAILURES` (used by NYSE and others — keep in regsho_sources too, or import)

`query_nyse()` calls `cffi_get()` and uses `BROWSER_HEADERS` from the HTTP helpers section — import them from `regsho_sources`.

**In regsho_sources.py**: Add re-exports for backward compatibility:
```python
from stockdownloader.data.nyse_regsho import (
    nyse_rate_limit,
    save_nyse_progress,
    load_nyse_progress,
    incremental_nyse_save,
    query_nyse,
    NYSE_DELAY_RANGE,
    NYSE_BATCH_SIZE,
    NYSE_BATCH_PAUSE_RANGE,
    NYSE_429_BACKOFF_RANGE,
    NYSE_URL_TEMPLATE,
)
```

This preserves all existing imports from `regsho_sources` in both `regsho_threshold_client.py` and `test_regsho_threshold_client.py`.

**Tests:**
1. Run: `python3 -m pytest tests/data/test_regsho_threshold_client.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract NYSE query logic from regsho_sources`

### Task 2: Extract bulk ZIP parser from sec_insider_parsers.py

**Files:**
- Create: `src/stockdownloader/data/bulk_insider_parser.py`
- Modify: `src/stockdownloader/data/sec_insider_parsers.py`

**What to extract:**

Move `parse_bulk_zip()` (lines 176-446, **271 lines**) to `bulk_insider_parser.py`. This is a self-contained 4-step function that:
1. Finds accessions for symbol in SUBMISSION.tsv
2. Reads owner details from REPORTING_OWNER.tsv
3. Reads transactions from NON_DERIVATIVE_TRANSACTION.tsv
4. Falls back to NON_DERIVATIVE_HOLDING.tsv if no transactions

It imports: `csv`, `io`, `logging`, `zipfile`, `Path`, `normalize_date` from `sec_parser_utils`, and model types from `regulatory_records`.

**In sec_insider_parsers.py**: Add re-export:
```python
from stockdownloader.data.bulk_insider_parser import parse_bulk_zip
```

This preserves the `parsers.parse_bulk_zip()` calls in `sec_insider_client.py` (which imports `from stockdownloader.data import sec_insider_parsers as parsers`).

**Tests:**
1. Run: `python3 -m pytest tests/data/test_sec_insider_client.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract bulk ZIP parser from sec_insider_parsers`

### Task 3: Extract bulk ZIP parser from sec_ownership_parsers.py

**Files:**
- Create: `src/stockdownloader/data/bulk_ownership_parser.py`
- Modify: `src/stockdownloader/data/sec_ownership_parsers.py`

**What to extract:**

Move `parse_bulk_zip()` (lines 165-388, **224 lines**) to `bulk_ownership_parser.py`. Same 4-step pattern as the insider version but with different TSV schemas (INFOTABLE, SUBMISSION, COVERPAGE).

It imports: `csv`, `io`, `logging`, `zipfile`, `Path`, and model types.

**In sec_ownership_parsers.py**: Add re-export:
```python
from stockdownloader.data.bulk_ownership_parser import parse_bulk_zip
```

**Tests:**
1. Run: `python3 -m pytest tests/data/test_sec_ownership_client.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract bulk ZIP parser from sec_ownership_parsers`

### Task 4: Verify line counts and full regression

**Steps:**

1. Verify line counts:
```bash
wc -l src/stockdownloader/data/regsho_sources.py \
      src/stockdownloader/data/nyse_regsho.py \
      src/stockdownloader/data/sec_insider_parsers.py \
      src/stockdownloader/data/bulk_insider_parser.py \
      src/stockdownloader/data/sec_ownership_parsers.py \
      src/stockdownloader/data/bulk_ownership_parser.py
```

**Expected results:**
| File | Before | After |
|------|--------|-------|
| `regsho_sources.py` | 684 | ~400 |
| `nyse_regsho.py` | (new) | ~300 |
| `sec_insider_parsers.py` | 707 | ~450 |
| `bulk_insider_parser.py` | (new) | ~280 |
| `sec_ownership_parsers.py` | 637 | ~430 |
| `bulk_ownership_parser.py` | (new) | ~230 |

2. Verify imports work:
```bash
python3 -c "from stockdownloader.data.regsho_sources import query_nyse, query_occ, query_nasdaq, query_cboe; print('OK')"
python3 -c "from stockdownloader.data import sec_insider_parsers as p; p.parse_bulk_zip; print('OK')"
python3 -c "from stockdownloader.data import sec_ownership_parsers as p; p.parse_bulk_zip; print('OK')"
```

3. Full regression: `python3 -m pytest tests/ -x -q`

---

## Verification

```bash
python3 -m pytest tests/ -x -q
wc -l src/stockdownloader/data/regsho_sources.py src/stockdownloader/data/nyse_regsho.py
wc -l src/stockdownloader/data/sec_insider_parsers.py src/stockdownloader/data/bulk_insider_parser.py
wc -l src/stockdownloader/data/sec_ownership_parsers.py src/stockdownloader/data/bulk_ownership_parser.py
```
