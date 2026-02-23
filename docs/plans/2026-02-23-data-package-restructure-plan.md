# Data Package Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Break up 3 monster files (sec_insider_client 1616 lines, sec_ownership_client 1373 lines, regsho_threshold_client 1028 lines) into focused modules, extract shared SEC helpers, and fix the SecOwnershipClient inheritance inconsistency.

**Architecture:** Extract-and-split approach — shared HTTP helpers into `sec_common.py`, parsing logic into `*_parsers.py` files, exchange-specific fetchers into `regsho_sources.py`. Client files become slim orchestrators. Atomic moves, no backward-compat shims. All callers updated in the same commit. Client class names and public import paths unchanged.

**Tech Stack:** Python 3.12, pytest, requests, curl_cffi (optional), xml.etree.ElementTree

---

### Task 1: Create `sec_common.py` — Shared SEC HTTP Helpers

**Files:**
- Create: `src/stockdownloader/data/sec_common.py`
- Test: `tests/data/test_sec_common.py`

**Context:** Both `sec_insider_client.py` and `sec_ownership_client.py` duplicate three methods: `_download_with_retry`, `_fetch_url_text`, `_fetch_efts_page`. Both also duplicate `_prev_quarter`. These must be extracted into a shared module.

**Step 1: Write the failing test**

Create `tests/data/test_sec_common.py`:

```python
"""Unit tests for sec_common — shared SEC HTTP helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.sec_common import (
    download_with_retry,
    fetch_efts_page,
    fetch_url_text,
    prev_quarter,
)


class TestPrevQuarter:
    def test_q2_to_q1(self):
        assert prev_quarter(2024, 2) == (2024, 1)

    def test_q1_wraps_to_q4_prev_year(self):
        assert prev_quarter(2024, 1) == (2023, 4)

    def test_q4_to_q3(self):
        assert prev_quarter(2024, 4) == (2024, 3)

    def test_q3_to_q2(self):
        assert prev_quarter(2024, 3) == (2024, 2)


class TestDownloadWithRetry:
    def test_success_returns_bytes(self):
        session = MagicMock()
        session.get.return_value = MagicMock(status_code=200, content=b"data")
        result = download_with_retry(session, "http://example.com/file.zip")
        assert result == b"data"

    def test_404_returns_none(self):
        session = MagicMock()
        session.get.return_value = MagicMock(status_code=404)
        result = download_with_retry(session, "http://example.com/missing.zip")
        assert result is None

    @patch("stockdownloader.data.sec_common.time.sleep")
    def test_retries_on_500(self, mock_sleep):
        session = MagicMock()
        session.get.side_effect = [
            MagicMock(status_code=500),
            MagicMock(status_code=200, content=b"ok"),
        ]
        result = download_with_retry(session, "http://example.com/file.zip", max_retries=2)
        assert result == b"ok"


class TestFetchUrlText:
    def test_success_returns_text(self):
        session = MagicMock()
        session.get.return_value = MagicMock(status_code=200, text="hello")
        result = fetch_url_text(session, "http://example.com/page")
        assert result == "hello"

    def test_404_returns_none(self):
        session = MagicMock()
        session.get.return_value = MagicMock(status_code=404)
        result = fetch_url_text(session, "http://example.com/missing")
        assert result is None


class TestFetchEftsPage:
    def test_success_returns_hits(self):
        session = MagicMock()
        hits = [{"_id": "1"}, {"_id": "2"}]
        session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"hits": {"hits": hits}},
        )
        result = fetch_efts_page(session, "http://efts.sec.gov/...")
        assert result == hits

    def test_empty_response(self):
        session = MagicMock()
        session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"hits": {"hits": []}},
        )
        result = fetch_efts_page(session, "http://efts.sec.gov/...")
        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/data/test_sec_common.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stockdownloader.data.sec_common'`

**Step 3: Write the implementation**

Create `src/stockdownloader/data/sec_common.py`:

```python
"""Shared SEC HTTP helpers used by multiple SEC data clients.

Provides rate-limited download, text fetch, and EFTS search functions
that both :class:`SecInsiderClient` and :class:`SecOwnershipClient`
share.  Extracted to eliminate code duplication.
"""

from __future__ import annotations

import json
import logging
import time

import requests

logger = logging.getLogger(__name__)

# SEC enforces 10 req/sec.  110 ms gap gives comfortable margin.
SEC_RATE_LIMIT_DELAY = 0.11

# EDGAR full-text search index
EFTS_BASE_URL = "https://efts.sec.gov/LATEST/search-index"

# Direct EDGAR archive access
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


def prev_quarter(year: int, quarter: int) -> tuple[int, int]:
    """Return ``(year, quarter)`` for the previous calendar quarter."""
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1


def download_with_retry(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = 3,
    rate_limit_fn: object | None = None,
) -> bytes | None:
    """Download binary content with retry logic.

    Parameters
    ----------
    session:
        An active ``requests.Session``.
    url:
        URL to download.
    max_retries:
        Maximum number of attempts.
    rate_limit_fn:
        Optional callable invoked before each request for pacing.
    """
    for attempt in range(max_retries):
        try:
            if rate_limit_fn is not None:
                rate_limit_fn()
            resp = session.get(url, timeout=120)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                logger.debug("Not found (404): %s", url)
                return None
            logger.warning(
                "Download returned %d: %s (attempt %d/%d)",
                resp.status_code, url, attempt + 1, max_retries,
            )
        except (requests.RequestException, OSError) as exc:
            logger.warning(
                "Download failed: %s (attempt %d/%d)",
                exc, attempt + 1, max_retries,
            )
        if attempt < max_retries - 1:
            time.sleep(2.0)
    return None


def fetch_url_text(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = 3,
    rate_limit_fn: object | None = None,
) -> str | None:
    """Download text content with retry logic.

    Parameters
    ----------
    session:
        An active ``requests.Session``.
    url:
        URL to fetch.
    max_retries:
        Maximum number of attempts.
    rate_limit_fn:
        Optional callable invoked before each request for pacing.
    """
    for attempt in range(max_retries):
        try:
            if rate_limit_fn is not None:
                rate_limit_fn()
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None
        except (requests.RequestException, OSError):
            pass
        if attempt < max_retries - 1:
            time.sleep(1.0)
    return None


def fetch_efts_page(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = 3,
    rate_limit_fn: object | None = None,
) -> list[dict] | None:
    """Fetch a single page from the EFTS search-index API.

    Parameters
    ----------
    session:
        An active ``requests.Session``.
    url:
        Full EFTS URL including query parameters.
    max_retries:
        Maximum number of attempts.
    rate_limit_fn:
        Optional callable invoked before each request for pacing.
    """
    for attempt in range(max_retries):
        try:
            if rate_limit_fn is not None:
                rate_limit_fn()
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("hits", {}).get("hits", [])
            logger.warning(
                "EFTS returned %d (attempt %d/%d): %s",
                resp.status_code, attempt + 1, max_retries, url,
            )
        except (
            requests.RequestException, json.JSONDecodeError, OSError,
        ) as exc:
            logger.warning(
                "EFTS failed: %s (attempt %d/%d)",
                exc, attempt + 1, max_retries,
            )
        if attempt < max_retries - 1:
            time.sleep(1.0)
    return None
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/data/test_sec_common.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/stockdownloader/data/sec_common.py tests/data/test_sec_common.py
git commit -m "feat: create sec_common.py with shared SEC HTTP helpers"
```

---

### Task 2: Create `sec_insider_parsers.py` — Extract Insider Parsing Logic

**Files:**
- Create: `src/stockdownloader/data/sec_insider_parsers.py`
- Modify: `src/stockdownloader/data/sec_insider_client.py`
- Modify: `tests/data/test_sec_insider_client.py`

**Context:** The module-level parsing functions (`_normalize_date`, `_xml_text`, `_prev_quarter`, `_extract_13d_13g_data`, `_apply_split_to_transaction`) and the big `_parse_bulk_zip` method need to move into a dedicated parsers module. The client-facing `_parse_form345_xml`, `_parse_form345_html`, and `_parse_13d_13g_content` methods also need to move — they use `self` only for `_fetch_url_text`, which can be passed as a parameter or called in the client before invoking the parser.

**What moves to `sec_insider_parsers.py`:**
- `_MONTH_ABBREVS` constant (line 1426)
- `_ACQUIRE_CODES`, `_DISPOSE_CODES` constants (lines 79-81)
- `normalize_date()` (was `_normalize_date`, line 1433)
- `xml_text()` (was `_xml_text`, line 1472)
- `extract_13d_13g_data()` (was `_extract_13d_13g_data`, line 1490)
- `apply_split_to_transaction()` (was `_apply_split_to_transaction`, line 1572)
- `parse_bulk_zip()` (was `SecInsiderClient._parse_bulk_zip`, line 365)
- `parse_form345_xml()` (was `SecInsiderClient._parse_form345_xml`, line 815) — takes `content: str` instead of URL
- `parse_form345_html()` (was `SecInsiderClient._parse_form345_html`, line 983) — takes `content: str` instead of URL
- `collect_form345()` (was `SecInsiderClient._collect_form345`, line 730)

**What stays in `sec_insider_client.py`:**
- `SecInsiderClient` class
- `__init__`, `fetch_insider_transactions`, `fetch_beneficial_owners`, `fetch_insider_snapshot`
- `_fetch_from_bulk` (orchestration — calls `download_with_retry` then `parse_bulk_zip`)
- `_fetch_individual_filings` (orchestration — calls `_fetch_url_text` then parsers)
- `_parse_individual_filing` (orchestration — fetches URL, delegates to parser)
- `_fetch_13d_13g_from_efts` (orchestration — calls `_fetch_efts_page` then parsers)
- `_parse_13d_13g_content` (thin wrapper — calls `fetch_url_text` then `extract_13d_13g_data`)
- `_resolve_cik` (uses session)
- All caching methods
- No `_download_with_retry` / `_fetch_url_text` / `_fetch_efts_page` (use `sec_common`)
- No `_prev_quarter` (use `sec_common.prev_quarter`)

**Step 1: Create `sec_insider_parsers.py`**

Move the functions listed above. Change private names to public (drop leading underscore) since they're now module-level exports. The `parse_form345_xml` and `parse_form345_html` functions take content strings (not URLs) — the client fetches the URL and passes the content.

**Step 2: Update `sec_insider_client.py`**

- Remove the moved functions/methods
- Remove `_download_with_retry`, `_fetch_url_text`, `_fetch_efts_page` (use `sec_common`)
- Replace `_prev_quarter` calls with `sec_common.prev_quarter`
- Import from `sec_insider_parsers` and `sec_common`
- Update `_parse_form345_xml` to call `fetch_url_text` then `parsers.parse_form345_xml(content, ...)`
- Update `_parse_form345_html` similarly
- Update `_parse_13d_13g_content` to call `fetch_url_text` then `parsers.extract_13d_13g_data(content)`

**Step 3: Update `tests/data/test_sec_insider_client.py`**

The test file imports these from `sec_insider_client`:
```python
from stockdownloader.data.sec_insider_client import (
    SecInsiderClient,
    _extract_13d_13g_data,
    _normalize_date,
    _prev_quarter,
    _xml_text,
)
```

Change to:
```python
from stockdownloader.data.sec_insider_client import SecInsiderClient
from stockdownloader.data.sec_insider_parsers import (
    extract_13d_13g_data,
    normalize_date,
    xml_text,
)
from stockdownloader.data.sec_common import prev_quarter
```

Then update all call sites in the tests (`_normalize_date(...)` → `normalize_date(...)`, etc.).

**Step 4: Run tests**

Run: `python3 -m pytest tests/data/test_sec_insider_client.py tests/data/test_sec_common.py -v`
Expected: ALL PASS

**Step 5: Run full suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All 3184 tests pass

**Step 6: Commit**

```bash
git add src/stockdownloader/data/sec_insider_parsers.py \
    src/stockdownloader/data/sec_insider_client.py \
    tests/data/test_sec_insider_client.py
git commit -m "refactor: extract sec_insider_parsers.py, wire sec_common into insider client"
```

---

### Task 3: Create `sec_ownership_parsers.py` — Extract Ownership Parsing + Refactor to BaseDataClient

**Files:**
- Create: `src/stockdownloader/data/sec_ownership_parsers.py`
- Modify: `src/stockdownloader/data/sec_ownership_client.py`
- Modify: `tests/data/test_sec_ownership_client.py`

**Context:** `SecOwnershipClient` does NOT extend `BaseDataClient` — it reimplements its own `requests.Session`, `_rate_limit()`, `_symbol_dir()`, `_progress_dir()`. This task extracts parsers AND refactors the client to extend `BaseDataClient`.

**What moves to `sec_ownership_parsers.py`:**
- `quarter_range()` (was `_quarter_range`, module-level, line 75)
- `bulk_zip_urls()` (was `_bulk_zip_urls`, module-level, line 88)
- `is_leap_year()` (was `_is_leap_year`, module-level, line 149)
- `parse_bulk_zip()` — extracted from `SecOwnershipClient._parse_bulk_zip` (line 412), takes `zip_source, symbol, cusip, quarter_end` and returns `OwnershipSnapshot | None`
- `parse_13f_xml()` (was `SecOwnershipClient._parse_13f_xml`, static method, line 889)
- `parse_13f_text()` (was `SecOwnershipClient._parse_13f_text`, static method, line 971)
- `find_infotable_url()` — extracted from `SecOwnershipClient._find_infotable_url` (line 848), takes `session, index_url, rate_limit_fn, max_retries` instead of `self`
- `get_xml_text()` (was `_get_xml_text`, module-level, line 1289)
- `filing_date_to_quarter_end()` (was `_filing_date_to_quarter_end`, module-level, line 1297)
- `apply_split_to_snapshot()` (was `_apply_split_to_snapshot`, module-level, line 1328)

Constants that move:
- `_BULK_13F_BASE`, `_NS_13F`, `_NS_13F_ALT`, `_GME_CUSIP`, `_QUARTER_ENDS`, `_MONTH_NAMES`

**What stays in `sec_ownership_client.py`:**
- `SecOwnershipClient(BaseDataClient)` — NOW EXTENDS BaseDataClient
- `__init__` — calls `super().__init__(...)`, sets up `_bulk_dir` only
- `fetch_ownership_snapshots` — orchestration
- `_fetch_from_bulk` — orchestration (calls `download_with_retry` then `parse_bulk_zip`)
- `_fetch_from_efts` — orchestration (calls `fetch_efts_page` then individual parsing)
- All caching methods (use `BaseDataClient._symbol_dir`, `_progress_dir`, `_load_csv`, `_save_csv`)
- No `_download_with_retry` / `_fetch_url_text` / `_fetch_efts_page` (use `sec_common`)
- No `_prev_quarter` (use `sec_common.prev_quarter`)
- No `_rate_limit()` (inherited from BaseDataClient)
- No `_symbol_dir()` or `_progress_dir()` (inherited from BaseDataClient)

**Key refactoring for BaseDataClient extension:**
- Remove `self._session = requests.Session()` and headers setup
- Remove `self._last_request_time = 0.0`
- Remove `self._data_dir = Path(data_dir)` and mkdir
- Remove `_rate_limit()` method
- Remove `_symbol_dir()` and `_progress_dir()` methods
- Add `super().__init__(rate_limit_delay=0.11, max_retries=3, data_dir=data_dir, default_headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})`
- Replace `self._download_with_retry(url)` with `sec_common.download_with_retry(self._session, url, rate_limit_fn=self._rate_limit)`
- Replace `self._fetch_url_text(url)` with `sec_common.fetch_url_text(self._session, url, rate_limit_fn=self._rate_limit)`
- Replace `self._fetch_efts_page(url)` with `sec_common.fetch_efts_page(self._session, url, rate_limit_fn=self._rate_limit)`

**Step 1: Create `sec_ownership_parsers.py`**

Move the functions listed above. Module-level constants (`_BULK_13F_BASE` etc.) move here too.

**Step 2: Refactor `sec_ownership_client.py`**

- Change class signature to `class SecOwnershipClient(BaseDataClient):`
- Add `from stockdownloader.data.base_client import BaseDataClient`
- Replace `__init__` body with `super().__init__()` call + `_bulk_dir` setup
- Remove duplicated methods (`_rate_limit`, `_symbol_dir`, `_progress_dir`, `_download_with_retry`, `_fetch_url_text`, `_fetch_efts_page`)
- Import from `sec_ownership_parsers` and `sec_common`
- Update all internal call sites

**Step 3: Update `tests/data/test_sec_ownership_client.py`**

The test imports:
```python
from stockdownloader.data.sec_ownership_client import (
    SecOwnershipClient,
    _filing_date_to_quarter_end,
    _get_xml_text,
)
```

Change to:
```python
from stockdownloader.data.sec_ownership_client import SecOwnershipClient
from stockdownloader.data.sec_ownership_parsers import (
    filing_date_to_quarter_end,
    get_xml_text,
)
```

Update all call sites in tests.

**Step 4: Run tests**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py tests/data/test_sec_common.py -v`
Expected: ALL PASS

**Step 5: Run full suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All 3184 tests pass

**Step 6: Commit**

```bash
git add src/stockdownloader/data/sec_ownership_parsers.py \
    src/stockdownloader/data/sec_ownership_client.py \
    tests/data/test_sec_ownership_client.py
git commit -m "refactor: extract sec_ownership_parsers.py, make SecOwnershipClient extend BaseDataClient"
```

---

### Task 4: Create `regsho_sources.py` — Extract Exchange Source Fetchers

**Files:**
- Create: `src/stockdownloader/data/regsho_sources.py`
- Modify: `src/stockdownloader/data/regsho_threshold_client.py`
- Modify: `tests/data/test_regsho_threshold_client.py`

**Context:** `regsho_threshold_client.py` has 4 exchange-specific fetcher methods (NYSE 155 lines, OCC 72 lines, Nasdaq 66 lines, CBOE 67 lines) plus their rate-limiting helpers, all crammed into one file. Extract into source-specific fetcher functions.

**What moves to `regsho_sources.py`:**
- URL template constants: `_NYSE_URL_TEMPLATE`, `_NASDAQ_URL_TEMPLATE`, `_OCC_URL_TEMPLATE`, `_CBOE_URL_TEMPLATE`
- Rate limit constants: `_NYSE_DELAY_RANGE`, `_NYSE_BATCH_SIZE`, `_NYSE_BATCH_PAUSE_RANGE`, `_NYSE_429_BACKOFF_RANGE`, `_OCC_RATE_LIMIT_DELAY`, `_CBOE_RATE_LIMIT_DELAY`
- `_BROWSER_HEADERS` constant
- `_MAX_CONSECUTIVE_FAILURES` constant
- `query_nyse()` (was `_query_nyse`) — takes `symbol, lookback_days, cached_dates, curl_session, session, progress_callbacks`
- `query_occ()` (was `_query_occ`) — takes `symbol, lookback_days, session`
- `query_nasdaq()` (was `_query_nasdaq`) — takes `symbol, lookback_days, session, rate_limit_fn`
- `query_cboe()` (was `_query_cboe`) — takes `symbol, lookback_days, session`
- `cffi_get()` (was `_cffi_get`) — takes `curl_session, url, timeout`
- `curl_get()` (was `_curl_get`, static) — standalone function
- NYSE rate limiting: `nyse_rate_limit()` — takes state dict for tracking
- OCC rate limiting: `occ_rate_limit()` — standalone
- CBOE rate limiting: `cboe_rate_limit()` — standalone
- NYSE progress helpers: `incremental_nyse_save()`, `save_nyse_progress()`, `load_nyse_progress()`

**Design decision:** The source fetchers use `ThresholdRecord` from `regsho_threshold_client.py`. To avoid circular imports, the `ThresholdRecord` dataclass stays in `regsho_threshold_client.py` and `regsho_sources.py` imports it.

**What stays in `regsho_threshold_client.py`:**
- `ThresholdRecord` dataclass
- `RegShoThresholdClient(BaseDataClient)` class
- `__init__` — sets up sessions, rate-limit state
- `fetch_threshold_status()` — orchestrator that calls source functions
- `fetch_threshold_targeted()` — targeted date fetcher
- `is_currently_on_threshold()` — convenience method
- Caching methods (`_load_cache`, `_save_cache`)
- General `_rate_limit()` (for Nasdaq, inherited)

**Step 1: Create `regsho_sources.py`**

Move the source fetcher functions. Each function takes explicit session/state parameters instead of `self`. The NYSE function is the most complex — it needs a `NyseState` dataclass (or simple dict) to track `request_count`, `last_request_time`, and callbacks for incremental save.

**Step 2: Update `regsho_threshold_client.py`**

- Remove moved methods and constants
- Import from `regsho_sources`
- Update `fetch_threshold_status()` to call source functions with explicit parameters
- Keep `RegShoThresholdClient.__init__` setting up sessions/state, but pass them to source functions
- `__init__` no longer needs per-source timing attributes — those live in the source module

**Step 3: Update `tests/data/test_regsho_threshold_client.py`**

The test imports URL template constants:
```python
from stockdownloader.data.regsho_threshold_client import (
    RegShoThresholdClient,
    ThresholdRecord,
    _CBOE_URL_TEMPLATE,
    _NASDAQ_URL_TEMPLATE,
    _NYSE_URL_TEMPLATE,
    _OCC_URL_TEMPLATE,
)
```

Change constant imports to come from `regsho_sources`:
```python
from stockdownloader.data.regsho_threshold_client import (
    RegShoThresholdClient,
    ThresholdRecord,
)
from stockdownloader.data.regsho_sources import (
    CBOE_URL_TEMPLATE,
    NASDAQ_URL_TEMPLATE,
    NYSE_URL_TEMPLATE,
    OCC_URL_TEMPLATE,
)
```

Update all test references.

**Step 4: Run tests**

Run: `python3 -m pytest tests/data/test_regsho_threshold_client.py -v`
Expected: ALL PASS

**Step 5: Run full suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All 3184 tests pass

**Step 6: Commit**

```bash
git add src/stockdownloader/data/regsho_sources.py \
    src/stockdownloader/data/regsho_threshold_client.py \
    tests/data/test_regsho_threshold_client.py
git commit -m "refactor: extract regsho_sources.py with 4 exchange fetchers"
```

---

### Task 5: Add `SecInsiderClient` to `data/__init__.py` + Final Cleanup

**Files:**
- Modify: `src/stockdownloader/data/__init__.py`

**Context:** `SecInsiderClient` is used by `scripts/download_all.py`, `scripts/archive/gme_holistic_report.py`, and pipeline code but is missing from `data/__init__.py`. Also verify no other callers need updating.

**Step 1: Update `data/__init__.py`**

Add after the `sec_ownership_client` import line:
```python
from stockdownloader.data.sec_insider_client import SecInsiderClient
```

Add `"SecInsiderClient"` to `__all__` list (alphabetically near other SEC entries).

**Step 2: Verify external callers**

These files import directly from the client modules (not from `data/`):
- `scripts/download_all.py` line 153: `from stockdownloader.data.sec_insider_client import SecInsiderClient` ← still works
- `scripts/download_all.py` line 154: `from stockdownloader.data.sec_ownership_client import SecOwnershipClient` ← still works
- `scripts/download_all.py` line 241: `from stockdownloader.data.regsho_threshold_client import RegShoThresholdClient` ← still works
- `scripts/archive/gme_holistic_report.py` line 34: `from stockdownloader.data.sec_insider_client import SecInsiderClient` ← still works
- `scripts/regsho_targeted_backfill.py` line 37: `from stockdownloader.data.regsho_threshold_client import RegShoThresholdClient` ← still works
- `src/stockdownloader/ml/pipeline/stage_data.py` line 292: `from stockdownloader.data.sec_ownership_client import SecOwnershipClient` ← still works

All external callers import client classes by their original module paths. Client class names and module paths are unchanged, so **no external callers need updating**.

**Step 3: Run full suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All 3184 tests pass

**Step 4: Commit**

```bash
git add src/stockdownloader/data/__init__.py
git commit -m "refactor: add SecInsiderClient to data/__init__.py exports"
```

---

### Task 6: Verify Line Counts and Final Regression

**Files:** None (verification only)

**Step 1: Verify file sizes reduced**

Run:
```bash
wc -l src/stockdownloader/data/sec_insider_client.py \
    src/stockdownloader/data/sec_insider_parsers.py \
    src/stockdownloader/data/sec_ownership_client.py \
    src/stockdownloader/data/sec_ownership_parsers.py \
    src/stockdownloader/data/regsho_threshold_client.py \
    src/stockdownloader/data/regsho_sources.py \
    src/stockdownloader/data/sec_common.py
```

Expected approximate line counts:
- `sec_insider_client.py`: ~600 (was 1616)
- `sec_insider_parsers.py`: ~500
- `sec_ownership_client.py`: ~500 (was 1373)
- `sec_ownership_parsers.py`: ~400
- `regsho_threshold_client.py`: ~250 (was 1028)
- `regsho_sources.py`: ~600
- `sec_common.py`: ~150

**Step 2: Full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All 3184 tests pass (no regressions)

**Step 3: Verify imports work end-to-end**

Run:
```bash
python3 -c "
from stockdownloader.data.sec_insider_client import SecInsiderClient
from stockdownloader.data.sec_ownership_client import SecOwnershipClient
from stockdownloader.data.regsho_threshold_client import RegShoThresholdClient, ThresholdRecord
from stockdownloader.data.sec_common import download_with_retry, fetch_url_text, fetch_efts_page, prev_quarter
from stockdownloader.data.sec_insider_parsers import normalize_date, xml_text, extract_13d_13g_data, parse_bulk_zip
from stockdownloader.data.sec_ownership_parsers import parse_13f_xml, parse_13f_text, get_xml_text, filing_date_to_quarter_end
from stockdownloader.data.regsho_sources import query_nyse, query_occ, query_nasdaq, query_cboe
from stockdownloader.data import SecInsiderClient as SIC2
print('All imports OK')
"
```

Expected: `All imports OK`
