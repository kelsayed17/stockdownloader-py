# Round 10: SEC Client Deduplication — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract duplicated patterns from 3 SEC clients + 2 parsers into shared modules, eliminating copy-paste code across `sec_insider_client.py`, `sec_ownership_client.py`, `sec_ftd_client.py`, `sec_insider_parsers.py`, and `sec_ownership_parsers.py`.

**Architecture:** Move `SplitAdjustment` + `_KNOWN_SPLITS` + quarter iteration + EFTS pagination + ZIP-download-and-cache into `sec_common.py`. Migrate `SecFtdClient` to inherit `BaseDataClient`. Extract shared parser helpers into `sec_parser_utils.py`.

**Tech Stack:** Python dataclasses, requests, zipfile, xml.etree.ElementTree

---

### Task 1: Move `SplitAdjustment` and `_KNOWN_SPLITS` to `sec_common.py`

`SplitAdjustment` (a frozen dataclass) and the `_KNOWN_SPLITS` registry are
defined in `sec_ftd_client.py` (lines 142-212) but imported by 4 other files.
Moving them to `sec_common.py` removes the awkward cross-client dependency.

**Files:**
- Modify: `src/stockdownloader/data/sec_common.py` — add SplitAdjustment + _KNOWN_SPLITS
- Modify: `src/stockdownloader/data/sec_ftd_client.py` — remove definitions, import from sec_common
- Modify: `src/stockdownloader/data/sec_insider_client.py` — change import source
- Modify: `src/stockdownloader/data/sec_ownership_client.py` — change import source
- Modify: `src/stockdownloader/data/sec_insider_parsers.py` — change import source
- Modify: `src/stockdownloader/data/sec_ownership_parsers.py` — change import source
- Modify: `tests/data/test_sec_ftd_client.py` — change import source if needed

**Steps:**

1. Add to the top of `sec_common.py` (after existing imports):
   ```python
   from dataclasses import dataclass
   from datetime import date
   from decimal import Decimal
   ```
   Then add the `SplitAdjustment` dataclass and `_KNOWN_SPLITS` list (copy
   verbatim from `sec_ftd_client.py` lines 142-212). Also add a helper:
   ```python
   def splits_for_symbol(
       symbol: str,
       extra_splits: list[SplitAdjustment] | None = None,
   ) -> list[SplitAdjustment]:
       """Return known + extra splits filtered for *symbol*."""
       all_splits = _KNOWN_SPLITS + (extra_splits or [])
       return [s for s in all_splits if s.symbol == symbol.upper()]
   ```

2. In `sec_ftd_client.py`: remove the `SplitAdjustment` class, `_KNOWN_SPLITS`
   list, and `from decimal import Decimal` import (unless used elsewhere in the
   file — it IS used for FTD price parsing, so keep it). Replace with:
   ```python
   from stockdownloader.data.sec_common import SplitAdjustment, _KNOWN_SPLITS
   ```
   Keep the existing `from decimal import Decimal, InvalidOperation` import
   since `_parse_ftd_file` uses Decimal for price parsing.

3. In `sec_insider_client.py` line 48: change
   ```python
   from stockdownloader.data.sec_ftd_client import SplitAdjustment, _KNOWN_SPLITS
   ```
   to:
   ```python
   from stockdownloader.data.sec_common import SplitAdjustment, _KNOWN_SPLITS, splits_for_symbol
   ```
   Then replace lines 215-222 (the inline split filtering):
   ```python
   all_splits = _KNOWN_SPLITS + (extra_splits or [])
   symbol_splits = [s for s in all_splits if s.symbol == symbol_upper]
   ```
   with:
   ```python
   symbol_splits = splits_for_symbol(symbol_upper, extra_splits)
   ```

4. In `sec_ownership_client.py` line 34: same import change as step 3.
   Replace inline split filtering with `splits_for_symbol()`.

5. In `sec_insider_parsers.py` line 17: change
   ```python
   from stockdownloader.data.sec_ftd_client import SplitAdjustment, _KNOWN_SPLITS
   ```
   to:
   ```python
   from stockdownloader.data.sec_common import SplitAdjustment, _KNOWN_SPLITS
   ```

6. In `sec_ownership_parsers.py` line 25: same change as step 5.

7. Check `tests/data/test_sec_ftd_client.py` for imports of `SplitAdjustment`
   or `_KNOWN_SPLITS` from `sec_ftd_client`. If present, update to import from
   `sec_common` instead. Note: the test file may also import `SplitAdjustment`
   from the client — check and update.

8. Run: `python3 -m pytest tests/data/test_sec_ftd_client.py tests/data/test_sec_insider_client.py tests/data/test_sec_ownership_client.py -x -q`
   Expected: all pass.

9. Commit:
   ```bash
   git add src/stockdownloader/data/sec_common.py \
           src/stockdownloader/data/sec_ftd_client.py \
           src/stockdownloader/data/sec_insider_client.py \
           src/stockdownloader/data/sec_ownership_client.py \
           src/stockdownloader/data/sec_insider_parsers.py \
           src/stockdownloader/data/sec_ownership_parsers.py \
           tests/data/test_sec_ftd_client.py
   git commit -m "refactor: move SplitAdjustment + _KNOWN_SPLITS to sec_common"
   ```

---

### Task 2: Add quarter iteration and EFTS pagination helpers to `sec_common.py`

The quarter-iteration-with-IPO-floor pattern is nearly identical in
`sec_insider_client.py` (lines 136-176) and `sec_ownership_client.py`
(lines 157-234). EFTS pagination is 100% identical in both (insider
lines 543-558, ownership lines 349-364).

**Files:**
- Modify: `src/stockdownloader/data/sec_common.py` — add `quarter_iterator()` + `fetch_all_efts_hits()`
- Modify: `src/stockdownloader/data/sec_insider_client.py` — use new helpers
- Modify: `src/stockdownloader/data/sec_ownership_client.py` — use new helpers
- Test: `tests/data/test_sec_common.py` — new test file

**Steps:**

1. Add to `sec_common.py`:
   ```python
   from collections.abc import Iterator

   def quarter_iterator(
       num_quarters: int,
       *,
       min_year: int = 2003,
       min_quarter: int = 1,
   ) -> Iterator[tuple[int, int]]:
       """Yield *(year, quarter)* tuples backwards from current quarter.

       Stops after *num_quarters* or when reaching *min_year*/*min_quarter*
       (whichever comes first).  Skips future quarters automatically.
       """
       today = date.today()
       current_year = today.year
       current_quarter = (today.month - 1) // 3 + 1

       year, quarter = current_year, current_quarter
       yielded = 0

       while yielded < num_quarters:
           if year < min_year or (year == min_year and quarter < min_quarter):
               break
           if (year, quarter) <= (current_year, current_quarter):
               yield year, quarter
               yielded += 1
           year, quarter = prev_quarter(year, quarter)


   def ipo_quarter_floor(symbol: str) -> tuple[int, int]:
       """Return the earliest useful *(year, quarter)* for *symbol*.

       Uses the symbol registry IPO date.  Returns ``(2003, 1)`` as the
       default floor (EDGAR data starts around 2003).
       """
       from stockdownloader.model.symbol_info import get_symbol_info
       info = get_symbol_info(symbol.upper())
       if info is not None and info.ipo_date:
           y = info.ipo_date.year
           q = (info.ipo_date.month - 1) // 3 + 1
           if (y, q) > (2003, 1):
               return y, q
       return 2003, 1


   def fetch_all_efts_hits(
       session: requests.Session,
       base_url: str,
       *,
       rate_limit_fn: Callable[[], None] | None = None,
       page_size: int = 100,
       max_results: int = 5000,
   ) -> list[dict]:
       """Paginate through EFTS search results, returning all hits.

       Parameters
       ----------
       session:
           An active ``requests.Session``.
       base_url:
           The EFTS query URL *without* ``from``/``size`` params.
       rate_limit_fn:
           Optional callable invoked before each HTTP request.
       page_size:
           Results per page (default 100).
       max_results:
           Hard cap on total results (default 5000).

       Returns
       -------
       list[dict]
           All ``hits.hits`` entries concatenated.
       """
       all_hits: list[dict] = []
       offset = 0

       while offset < max_results:
           url = f"{base_url}&from={offset}&size={page_size}"
           page = fetch_efts_page(session, url, rate_limit_fn=rate_limit_fn)
           if page is None or not page:
               break
           all_hits.extend(page)
           if len(page) < page_size:
               break
           offset += page_size

       return all_hits
   ```

2. Create `tests/data/test_sec_common.py` with tests for the new helpers:
   - `test_quarter_iterator_basic` — yields correct quarters in reverse
   - `test_quarter_iterator_ipo_floor` — stops at min_year/min_quarter
   - `test_quarter_iterator_zero` — yields nothing for num_quarters=0
   - `test_ipo_quarter_floor_known_symbol` — returns IPO quarter for GME
   - `test_ipo_quarter_floor_unknown` — returns (2003, 1) default
   - `test_splits_for_symbol` — filters splits correctly
   - `test_fetch_all_efts_hits_pagination` — mock EFTS pagination

3. Run: `python3 -m pytest tests/data/test_sec_common.py -v`
   Expected: all pass.

4. Refactor `sec_insider_client.py` `fetch_insider_transactions()`:
   - Replace the manual quarter iteration (lines 135-176) with:
     ```python
     min_y, min_q = sec_common.ipo_quarter_floor(symbol_upper)
     for year, quarter in sec_common.quarter_iterator(
         num_quarters, min_year=min_y, min_quarter=min_q,
     ):
     ```
   - Replace EFTS pagination in `_fetch_13d_13g_from_efts()` (lines 543-558)
     with:
     ```python
     all_hits = sec_common.fetch_all_efts_hits(
         self._session, base_url, rate_limit_fn=self._rate_limit,
     )
     ```

5. Refactor `sec_ownership_client.py` `fetch_ownership_snapshots()`:
   - Replace the manual quarter iteration (lines 157-234) with the same
     `quarter_iterator` pattern. Note: ownership client also computes
     `quarter_end_str` inside the loop — keep that logic, just replace the
     iteration scaffolding.
   - Replace EFTS pagination in `_fetch_from_efts()` (lines 349-364) with
     `fetch_all_efts_hits()`.

6. Run: `python3 -m pytest tests/data/ -x -q`
   Expected: all pass.

7. Commit:
   ```bash
   git add src/stockdownloader/data/sec_common.py \
           src/stockdownloader/data/sec_insider_client.py \
           src/stockdownloader/data/sec_ownership_client.py \
           tests/data/test_sec_common.py
   git commit -m "refactor: extract quarter_iterator + EFTS pagination to sec_common"
   ```

---

### Task 3: Migrate `SecFtdClient` to inherit `BaseDataClient`

`SecFtdClient` (612 lines) is the only SEC client that does NOT inherit
`BaseDataClient`. It manually sets up `requests.Session`, headers, rate
limiting, and cache directories — all of which `BaseDataClient` already
provides.

**Files:**
- Modify: `src/stockdownloader/data/sec_ftd_client.py` — inherit BaseDataClient
- Test: `tests/data/test_sec_ftd_client.py` — verify no regressions

**Steps:**

1. In `sec_ftd_client.py`:
   - Add import: `from stockdownloader.data.base_client import BaseDataClient`
   - Change class declaration: `class SecFtdClient(BaseDataClient):`
   - Replace `__init__` (lines 218-231) to call `super().__init__()`:
     ```python
     def __init__(
         self,
         user_agent: str = "StockDownloader admin@example.com",
         cache_dir: str = "data",
     ) -> None:
         super().__init__(
             rate_limit_delay=_RATE_LIMIT_DELAY,
             max_retries=_MAX_RETRIES,
             data_dir=cache_dir,
             default_headers={
                 "User-Agent": user_agent,
                 "Accept-Encoding": "gzip, deflate",
             },
         )
         self._cache_dir = self._data_dir / "cache" / "ftd"
         self._cache_dir.mkdir(parents=True, exist_ok=True)
     ```
   - Remove the manual `_rate_limit()` method (lines 606-612) — inherited
     from `BaseDataClient`.
   - Replace `self._session` references — `BaseDataClient.__init__` already
     creates `self._session`. Remove manual session creation.
   - Update `_download_url` to use `self._session` from base class (it
     already does if we remove the manual `__init__` session setup).

2. Run: `python3 -m pytest tests/data/test_sec_ftd_client.py -v`
   Expected: all pass.

3. Run: `python3 -m pytest tests/data/ -x -q`
   Expected: all pass.

4. Commit:
   ```bash
   git add src/stockdownloader/data/sec_ftd_client.py
   git commit -m "refactor: migrate SecFtdClient to inherit BaseDataClient"
   ```

---

### Task 4: Extract shared parser utilities into `sec_parser_utils.py`

`sec_insider_parsers.py` (767 lines) and `sec_ownership_parsers.py` (713 lines)
share: date normalization, XML text extraction, month-name mappings, and CUSIP
constants. Extract these into a new `sec_parser_utils.py`.

**Files:**
- Create: `src/stockdownloader/data/sec_parser_utils.py`
- Modify: `src/stockdownloader/data/sec_insider_parsers.py` — use shared utils
- Modify: `src/stockdownloader/data/sec_ownership_parsers.py` — use shared utils
- Test: `tests/data/test_sec_parser_utils.py` — new test file

**Steps:**

1. Create `src/stockdownloader/data/sec_parser_utils.py` containing:
   - `GME_CUSIP = "36467W109"` (from both parsers + insider client)
   - `QUARTER_ENDS` dict (from ownership_parsers line 48)
   - `MONTH_NAMES` / `MONTH_ABBREVS` dicts (from both parsers)
   - `normalize_date(raw: str) -> str` — the general-purpose date normalizer
     (from insider_parsers lines 44-80)
   - `xml_text(content: str, tag: str) -> str` — extract text from XML tag
     (from insider_parsers lines 83-91)
   - `get_xml_element_text(element: ET.Element, tag: str) -> str` — extract
     child element text (from ownership_parsers `get_xml_text` lines 146-151)
   - `quarter_range(year: int, quarter: int) -> tuple[str, str]` — from
     ownership_parsers lines 62-72
   - `is_leap_year(year: int) -> bool` — from ownership_parsers lines 75-77
   - `filing_date_to_quarter_end(filing_date: str) -> str` — from
     ownership_parsers lines 159-187

2. Create `tests/data/test_sec_parser_utils.py` with tests:
   - `test_normalize_date_iso` — "2024-01-15" → "2024-01-15"
   - `test_normalize_date_slashes` — "01/15/2024" → "2024-01-15"
   - `test_normalize_date_month_name` — "Jan 15, 2024" → "2024-01-15"
   - `test_xml_text_found` — extracts text from XML tag
   - `test_xml_text_missing` — returns empty string
   - `test_quarter_range` — returns correct date range
   - `test_filing_date_to_quarter_end` — maps filing dates correctly

3. Run: `python3 -m pytest tests/data/test_sec_parser_utils.py -v`
   Expected: all pass.

4. Refactor `sec_insider_parsers.py`:
   - Remove `normalize_date`, `xml_text`, `MONTH_ABBREVS` definitions
   - Import from `sec_parser_utils` instead
   - Verify all call sites still work

5. Refactor `sec_ownership_parsers.py`:
   - Remove `get_xml_text`, `quarter_range`, `is_leap_year`,
     `filing_date_to_quarter_end`, `QUARTER_ENDS`, `MONTH_NAMES`, `GME_CUSIP`
   - Import from `sec_parser_utils` instead

6. Run: `python3 -m pytest tests/data/ -x -q`
   Expected: all pass.

7. Commit:
   ```bash
   git add src/stockdownloader/data/sec_parser_utils.py \
           src/stockdownloader/data/sec_insider_parsers.py \
           src/stockdownloader/data/sec_ownership_parsers.py \
           tests/data/test_sec_parser_utils.py
   git commit -m "refactor: extract shared parser utils into sec_parser_utils.py"
   ```

---

### Task 5: Update `data/__init__.py` exports and run full regression

**Files:**
- Modify: `src/stockdownloader/data/__init__.py` — export `SplitAdjustment`, `_KNOWN_SPLITS` from new location
- Run: full test suite

**Steps:**

1. Check `src/stockdownloader/data/__init__.py` for any exports of
   `SplitAdjustment` or `_KNOWN_SPLITS`. If they reference `sec_ftd_client`,
   update to reference `sec_common`.

2. Run full regression: `python3 -m pytest tests/ -x -q`
   Expected: 3226+ tests pass, 0 failures.

3. Verify line count reductions:
   ```bash
   wc -l src/stockdownloader/data/sec_ftd_client.py \
         src/stockdownloader/data/sec_insider_client.py \
         src/stockdownloader/data/sec_ownership_client.py \
         src/stockdownloader/data/sec_insider_parsers.py \
         src/stockdownloader/data/sec_ownership_parsers.py \
         src/stockdownloader/data/sec_common.py \
         src/stockdownloader/data/sec_parser_utils.py
   ```

4. Commit:
   ```bash
   git add src/stockdownloader/data/__init__.py
   git commit -m "refactor: update data/ exports for SEC client dedup"
   ```

---

## Verification

```bash
# Full suite regression
python3 -m pytest tests/ -x -q

# Specific SEC client tests
python3 -m pytest tests/data/test_sec_ftd_client.py tests/data/test_sec_insider_client.py tests/data/test_sec_ownership_client.py tests/data/test_sec_common.py tests/data/test_sec_parser_utils.py -v
```
