# OCC Options Open Interest Client — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fetch daily per-symbol options open interest from OCC `cont-volume-download` to detect synthetic short positions in GME options.

**Architecture:** New `OccOptionsClient` extends `BaseDataClient`, downloads daily fixed-width bulk files from OCC, parses and filters for the requested symbol, caches per-symbol results with incremental progress tracking. New `OccOpenInterestRecord` frozen dataclass. Analysis script cross-references OI with FTD and short volume data.

**Tech Stack:** Python 3.12, requests, dataclasses, pytest, BaseDataClient pattern

---

### Task 1: Data Model — `OccOpenInterestRecord`

**Files:**
- Modify: `src/stockdownloader/model/regulatory_records.py` (append after line 290)
- Modify: `src/stockdownloader/model/__init__.py` (add import + export)

**Step 1: Write the failing test**

Create `tests/model/test_occ_open_interest_record.py`:

```python
"""Unit tests for OccOpenInterestRecord dataclass."""

from __future__ import annotations

import pytest

from stockdownloader.model.regulatory_records import OccOpenInterestRecord


class TestOccOpenInterestRecord:
    """Tests for OccOpenInterestRecord creation and validation."""

    def test_create_valid_record(self) -> None:
        record = OccOpenInterestRecord(
            date="2021-01-27",
            symbol="GME",
            exchange="A",
            volume=5227,
            exercised=0,
            open_interest=5121,
            product_kind="OSTK",
            expiration="2021-02-19",
        )
        assert record.date == "2021-01-27"
        assert record.symbol == "GME"
        assert record.exchange == "A"
        assert record.volume == 5227
        assert record.exercised == 0
        assert record.open_interest == 5121
        assert record.product_kind == "OSTK"
        assert record.expiration == "2021-02-19"

    def test_record_is_frozen(self) -> None:
        record = OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=100, exercised=0, open_interest=50,
            product_kind="OSTK", expiration="2021-02-19",
        )
        with pytest.raises(AttributeError):
            record.volume = 999  # type: ignore[misc]

    def test_empty_date_raises(self) -> None:
        with pytest.raises(ValueError, match="date must not be empty"):
            OccOpenInterestRecord(
                date="", symbol="GME", exchange="A",
                volume=100, exercised=0, open_interest=50,
                product_kind="OSTK", expiration="2021-02-19",
            )

    def test_empty_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol must not be empty"):
            OccOpenInterestRecord(
                date="2021-01-27", symbol="", exchange="A",
                volume=100, exercised=0, open_interest=50,
                product_kind="OSTK", expiration="2021-02-19",
            )

    def test_negative_open_interest_raises(self) -> None:
        with pytest.raises(ValueError, match="open_interest must be non-negative"):
            OccOpenInterestRecord(
                date="2021-01-27", symbol="GME", exchange="A",
                volume=100, exercised=0, open_interest=-1,
                product_kind="OSTK", expiration="2021-02-19",
            )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/model/test_occ_open_interest_record.py -v`
Expected: FAIL with `ImportError: cannot import name 'OccOpenInterestRecord'`

**Step 3: Write minimal implementation**

Append to `src/stockdownloader/model/regulatory_records.py` after line 290 (after `InsiderOwnershipSnapshot`):

```python
# ── OCC Options Open Interest ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OccOpenInterestRecord:
    """Daily per-symbol options open interest from OCC bulk download.

    Each record represents one symbol + exchange + expiration combination
    for a single trading day.  The OCC ``cont-volume-download`` endpoint
    does not distinguish puts from calls — each row is aggregated across
    option types.

    Fields map to the fixed-width columns in the OCC bulk file:
    ``symbol(6)|underlying(6)|exchange(1)|volume(9)|exercised(9)|oi(9)|kind(4)|exp(8)``
    """

    date: str              # "YYYY-MM-DD" — report date
    symbol: str            # "GME" — option/underlying symbol
    exchange: str          # Single-char exchange code ("A"=AMEX, etc.)
    volume: int            # Daily contract volume
    exercised: int         # Contracts exercised that day
    open_interest: int     # End-of-day open interest (contracts)
    product_kind: str      # "OSTK" (stock options), "OIND" (index), etc.
    expiration: str        # "YYYY-MM-DD" — option expiration date

    def __post_init__(self) -> None:
        if not self.date:
            raise ValueError("date must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.open_interest < 0:
            raise ValueError("open_interest must be non-negative")
```

**Step 4: Add export to `src/stockdownloader/model/__init__.py`**

Add to the import block (after `InsiderOwnershipSnapshot`):

```python
from stockdownloader.model.regulatory_records import (
    ...
    InsiderOwnershipSnapshot,
    OccOpenInterestRecord,
)
```

Add `"OccOpenInterestRecord"` to `__all__`.

**Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/model/test_occ_open_interest_record.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add src/stockdownloader/model/regulatory_records.py \
        src/stockdownloader/model/__init__.py \
        tests/model/test_occ_open_interest_record.py
git commit -m "feat: add OccOpenInterestRecord dataclass for OCC options data"
```

---

### Task 2: OCC Client — Fixed-Width Parsing

**Files:**
- Create: `src/stockdownloader/data/occ_options_client.py`
- Create: `tests/data/test_occ_options_client.py`

**Step 1: Write the failing test for bulk line parsing**

Create `tests/data/test_occ_options_client.py`:

```python
"""Unit tests for OccOptionsClient — OCC bulk download parsing and caching."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from stockdownloader.data.occ_options_client import OccOptionsClient
from stockdownloader.model.regulatory_records import OccOpenInterestRecord


# ------------------------------------------------------------------
# Sample data — matches OCC cont-volume-download fixed-width format
# ------------------------------------------------------------------

# Header line (always first line)
_OCC_HEADER = "        H02202026      02202026"

# GME records — 52 chars each: symbol(6)+underlying(6)+exch(1)+vol(9)+exerc(9)+oi(9)+kind(4)+exp(8)
_GME_LINE_1 = "GME   GME   A000005227000000000000005121OSTK20260220"
_GME_LINE_2 = "GME   GME   A000004015000000008000003975OSTK20260227"
_GME_LINE_3 = "GME   GME   C000012345000000000000067890OSTK20260321"

# Non-GME record
_AAPL_LINE  = "AAPL  AAPL  A000100000000000000000200000OSTK20260220"

# Full sample response (as OCC would return)
_SAMPLE_BULK_TEXT = "\r\n".join([
    _OCC_HEADER,
    _AAPL_LINE,
    _GME_LINE_1,
    _GME_LINE_2,
    _GME_LINE_3,
    "",  # trailing blank
])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(tmp_path: Path) -> OccOptionsClient:
    return OccOptionsClient(data_dir=str(tmp_path))


# ------------------------------------------------------------------
# Tests: Fixed-width line parsing
# ------------------------------------------------------------------


class TestOccBulkParsing:
    """Tests for parsing OCC fixed-width bulk download format."""

    def test_parse_single_line(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        record = client._parse_bulk_line(_GME_LINE_1, "2026-02-20")

        assert record is not None
        assert record.symbol == "GME"
        assert record.exchange == "A"
        assert record.volume == 5227
        assert record.exercised == 0
        assert record.open_interest == 5121
        assert record.product_kind == "OSTK"
        assert record.expiration == "2026-02-20"
        assert record.date == "2026-02-20"

    def test_parse_line_with_exercised(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        record = client._parse_bulk_line(_GME_LINE_2, "2026-02-20")

        assert record is not None
        assert record.volume == 4015
        assert record.exercised == 8
        assert record.open_interest == 3975
        assert record.expiration == "2026-02-27"

    def test_parse_short_line_returns_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        result = client._parse_bulk_line("too short", "2026-02-20")
        assert result is None

    def test_parse_header_line_returns_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        result = client._parse_bulk_line(_OCC_HEADER, "2026-02-20")
        assert result is None

    def test_filter_bulk_text_for_symbol(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = client._parse_bulk_text(_SAMPLE_BULK_TEXT, "GME", "2026-02-20")

        assert len(records) == 3
        assert all(r.symbol == "GME" for r in records)
        assert records[0].volume == 5227
        assert records[1].volume == 4015
        assert records[2].volume == 12345

    def test_filter_bulk_text_no_matches(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = client._parse_bulk_text(_SAMPLE_BULK_TEXT, "TSLA", "2026-02-20")
        assert len(records) == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/data/test_occ_options_client.py::TestOccBulkParsing -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stockdownloader.data.occ_options_client'`

**Step 3: Write the parsing implementation**

Create `src/stockdownloader/data/occ_options_client.py`:

```python
"""Fetches daily per-symbol options open interest from OCC bulk downloads.

The OCC (Options Clearing Corporation) publishes a daily bulk file
containing volume, exercised contracts, and open interest for every
option symbol across all exchanges and expirations.

Data source::

    https://marketdata.theocc.com/cont-volume-download?reportDate=YYYYMMDD&format=txt

The file is fixed-width text (~202K lines per day, all symbols).
Available from January 2021 to present.  Free, no authentication.

Each record is 52 characters:

    Position  Width  Field
    0-5       6      Symbol (space-padded)
    6-11      6      Underlying (space-padded)
    12        1      Exchange code (single char)
    13-21     9      Volume (zero-padded)
    22-30     9      Exercised (zero-padded)
    31-39     9      Open interest (zero-padded)
    40-43     4      Product kind (OSTK, OIND, etc.)
    44-51     8      Expiration (YYYYMMDD)

Usage::

    client = OccOptionsClient()
    records = client.fetch_open_interest("GME")
    # -> list of OccOpenInterestRecord sorted by date ascending
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from stockdownloader.data.base_client import BaseDataClient
from stockdownloader.model.regulatory_records import OccOpenInterestRecord

logger = logging.getLogger(__name__)

_OCC_URL_TEMPLATE = (
    "https://marketdata.theocc.com/cont-volume-download"
    "?reportDate={date}&format=txt"
)

_BATCH_SIZE = 25
_MAX_CONSECUTIVE_FAILURES = 15
_OCC_EARLIEST = date(2021, 1, 4)  # Earliest available OCC data


class OccOptionsClient(BaseDataClient):
    """OCC daily options open interest and volume data.

    Downloads the daily bulk file from OCC, parses fixed-width text,
    and filters for the requested symbol.  Caches per-symbol results
    and tracks download progress for incremental backfill.

    Parameters
    ----------
    data_dir:
        Root data directory.  Per-symbol data is stored under
        ``data_dir/{SYMBOL}/occ_open_interest.json``.
    """

    def __init__(self, data_dir: str = "data") -> None:
        super().__init__(
            rate_limit_delay=0.5,
            max_retries=3,
            data_dir=data_dir,
            default_headers={
                "User-Agent": "StockDownloader admin@example.com",
            },
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_bulk_line(
        self, line: str, report_date: str,
    ) -> OccOpenInterestRecord | None:
        """Parse a single fixed-width line from OCC bulk download.

        Returns ``None`` for header lines, blank lines, or lines
        shorter than the expected 52 characters.
        """
        if len(line) < 48:  # Minimum viable length
            return None

        # Header lines start with spaces + "H"
        stripped = line.lstrip()
        if stripped.startswith("H"):
            return None

        try:
            symbol = line[0:6].strip()
            if not symbol:
                return None

            exchange = line[12] if len(line) > 12 else ""
            volume = int(line[13:22]) if len(line) >= 22 else 0
            exercised = int(line[22:31]) if len(line) >= 31 else 0
            oi = int(line[31:40]) if len(line) >= 40 else 0
            product_kind = line[40:44].strip() if len(line) >= 44 else ""
            exp_raw = line[44:52].strip() if len(line) >= 52 else ""

            expiration = ""
            if len(exp_raw) == 8 and exp_raw.isdigit():
                expiration = f"{exp_raw[:4]}-{exp_raw[4:6]}-{exp_raw[6:8]}"

            return OccOpenInterestRecord(
                date=report_date,
                symbol=symbol,
                exchange=exchange,
                volume=volume,
                exercised=exercised,
                open_interest=oi,
                product_kind=product_kind,
                expiration=expiration,
            )
        except (ValueError, IndexError):
            return None

    def _parse_bulk_text(
        self, text: str, symbol: str, report_date: str,
    ) -> list[OccOpenInterestRecord]:
        """Parse full OCC bulk text and filter for *symbol*."""
        symbol_upper = symbol.upper()
        records: list[OccOpenInterestRecord] = []

        for line in text.splitlines():
            line = line.rstrip("\r\n")
            if not line:
                continue
            record = self._parse_bulk_line(line, report_date)
            if record and record.symbol == symbol_upper:
                records.append(record)

        return records

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_day(
        self, date_str: str, symbol: str,
    ) -> list[OccOpenInterestRecord]:
        """Download and parse OCC bulk file for one day.

        Parameters
        ----------
        date_str:
            Date in ``"YYYY-MM-DD"`` format.
        symbol:
            Ticker to filter for.

        Returns empty list if download fails or no data for symbol.
        """
        # OCC expects YYYYMMDD format
        occ_date = date_str.replace("-", "")
        url = _OCC_URL_TEMPLATE.format(date=occ_date)

        resp = self._fetch_with_retry("GET", url, timeout=60)
        if resp is None or not resp.ok:
            return []

        text = resp.text
        if not text or len(text) < 50:
            return []

        return self._parse_bulk_text(text, symbol, date_str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_open_interest(
        self,
        symbol: str,
        start_date: str | None = None,
    ) -> list[OccOpenInterestRecord]:
        """Fetch daily OI for *symbol* from OCC bulk files.

        Downloads one bulk file per trading day, parses and filters
        for the requested symbol.  Saves progress incrementally so
        interrupted backfills can resume.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"GME"``).
        start_date:
            Earliest date to fetch (``"YYYY-MM-DD"``).  Defaults to
            ``2021-01-04`` (earliest OCC data).

        Returns
        -------
        List of :class:`OccOpenInterestRecord` sorted by date ascending.
        """
        symbol_upper = symbol.upper()

        # Determine date range
        earliest = _OCC_EARLIEST
        if start_date:
            parsed = date.fromisoformat(start_date)
            earliest = max(parsed, _OCC_EARLIEST)

        end = date.today()

        # Generate all weekdays in range
        all_dates: list[str] = []
        current = earliest
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri
                all_dates.append(current.isoformat())
            current += timedelta(days=1)

        # Load progress and cache
        progress = self._load_progress(symbol_upper)
        cached = self._load_cache(symbol_upper) or []
        cached_dates = {r.date for r in cached}

        # Filter to dates not yet downloaded
        skip = progress | cached_dates
        remaining = [d for d in all_dates if d not in skip]

        logger.info(
            "OCC open interest for %s: %d total days, "
            "%d already done, %d remaining",
            symbol_upper, len(all_dates), len(skip), len(remaining),
        )

        if not remaining:
            return cached

        # Download remaining dates
        new_records: list[OccOpenInterestRecord] = []
        consecutive_failures = 0

        for i, date_str in enumerate(remaining):
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Too many consecutive failures (%d), stopping. "
                    "Progress saved for %s.",
                    consecutive_failures, symbol_upper,
                )
                break

            try:
                self._rate_limit()
                day_records = self._download_day(date_str, symbol_upper)

                progress.add(date_str)
                if day_records:
                    new_records.extend(day_records)
                    consecutive_failures = 0
                else:
                    # Empty response is normal for holidays
                    consecutive_failures = 0

            except Exception as exc:
                consecutive_failures += 1
                logger.debug(
                    "OCC download error for %s on %s: %s",
                    symbol_upper, date_str, exc,
                )
                continue

            # Incremental save
            if (i + 1) % _BATCH_SIZE == 0:
                merged = self._merge_records(cached, new_records)
                self._save_cache(symbol_upper, merged)
                self._save_progress(symbol_upper, progress)
                logger.info(
                    "OCC progress: %d/%d dates, %d records so far",
                    i + 1, len(remaining), len(merged),
                )

        # Final save
        merged = self._merge_records(cached, new_records)
        self._save_cache(symbol_upper, merged)
        self._save_progress(symbol_upper, progress)

        logger.info(
            "OCC open interest for %s: %d new records, %d total",
            symbol_upper, len(new_records), len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_records(
        existing: list[OccOpenInterestRecord],
        new: list[OccOpenInterestRecord],
    ) -> list[OccOpenInterestRecord]:
        """Merge existing and new records, deduplicating by composite key."""
        by_key: dict[tuple, OccOpenInterestRecord] = {}
        for r in existing:
            key = (r.date, r.symbol, r.exchange, r.expiration)
            by_key[key] = r
        for r in new:
            key = (r.date, r.symbol, r.exchange, r.expiration)
            by_key[key] = r  # New overwrites
        merged = list(by_key.values())
        merged.sort(key=lambda r: (r.date, r.exchange, r.expiration))
        return merged

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(
        self, symbol: str,
    ) -> list[OccOpenInterestRecord] | None:
        cache_file = self._symbol_dir(symbol) / "occ_open_interest.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            records = [OccOpenInterestRecord(**r) for r in data]
            records.sort(key=lambda r: (r.date, r.exchange, r.expiration))
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to load OCC cache for %s: %s", symbol, exc,
            )
            return None

    def _save_cache(
        self, symbol: str, records: list[OccOpenInterestRecord],
    ) -> None:
        cache_file = self._symbol_dir(symbol) / "occ_open_interest.json"
        try:
            cache_file.write_text(
                json.dumps([asdict(r) for r in records], indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save OCC cache for %s: %s", symbol, exc,
            )

    def _load_progress(self, symbol: str) -> set[str]:
        progress_file = self._symbol_dir(symbol) / "occ_options_progress.json"
        if not progress_file.exists():
            return set()
        try:
            return set(json.loads(progress_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return set()

    def _save_progress(self, symbol: str, dates: set[str]) -> None:
        progress_file = self._symbol_dir(symbol) / "occ_options_progress.json"
        try:
            progress_file.write_text(
                json.dumps(sorted(dates)), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Failed to save OCC progress for %s: %s", symbol, exc,
            )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/data/test_occ_options_client.py::TestOccBulkParsing -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/data/occ_options_client.py \
        tests/data/test_occ_options_client.py
git commit -m "feat: add OccOptionsClient with fixed-width parsing"
```

---

### Task 3: OCC Client — Caching & Progress

**Files:**
- Modify: `tests/data/test_occ_options_client.py` (add cache/progress tests)

**Step 1: Write the failing tests**

Append to `tests/data/test_occ_options_client.py`:

```python
# ------------------------------------------------------------------
# Tests: Cache save/load roundtrip
# ------------------------------------------------------------------


def _sample_records() -> list[OccOpenInterestRecord]:
    return [
        OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=5227, exercised=0, open_interest=5121,
            product_kind="OSTK", expiration="2021-02-19",
        ),
        OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="C",
            volume=12345, exercised=0, open_interest=67890,
            product_kind="OSTK", expiration="2021-02-19",
        ),
    ]


class TestOccCacheRoundtrip:
    """Tests for save/load cache roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        records = _sample_records()
        client._save_cache("GME", records)
        loaded = client._load_cache("GME")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].date == "2021-01-27"
        assert loaded[0].exchange == "A"
        assert loaded[1].exchange == "C"

    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._load_cache("NOSYMBOL") is None

    def test_load_corrupt_cache(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._data_dir / "BAD"
        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / "occ_open_interest.json").write_text(
            "{{invalid", encoding="utf-8",
        )
        assert client._load_cache("BAD") is None

    def test_cache_file_path(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_cache("GME", _sample_records())
        assert (client._data_dir / "GME" / "occ_open_interest.json").exists()


class TestOccProgress:
    """Tests for progress tracking (incremental backfill)."""

    def test_save_and_load_progress(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        progress = {"2021-01-27", "2021-01-28"}
        client._save_progress("GME", progress)
        loaded = client._load_progress("GME")

        assert loaded == progress

    def test_empty_progress_on_first_load(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert client._load_progress("GME") == set()

    def test_progress_file_path(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client._save_progress("GME", {"2021-01-27"})
        assert (client._data_dir / "GME" / "occ_options_progress.json").exists()


class TestOccMergeRecords:
    """Tests for record merging with deduplication."""

    def test_merge_no_overlap(self, tmp_path: Path) -> None:
        existing = [OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=100, exercised=0, open_interest=50,
            product_kind="OSTK", expiration="2021-02-19",
        )]
        new = [OccOpenInterestRecord(
            date="2021-01-28", symbol="GME", exchange="A",
            volume=200, exercised=0, open_interest=75,
            product_kind="OSTK", expiration="2021-02-19",
        )]
        merged = OccOptionsClient._merge_records(existing, new)
        assert len(merged) == 2
        assert merged[0].date == "2021-01-27"
        assert merged[1].date == "2021-01-28"

    def test_merge_deduplicates_by_key(self, tmp_path: Path) -> None:
        old = OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=100, exercised=0, open_interest=50,
            product_kind="OSTK", expiration="2021-02-19",
        )
        updated = OccOpenInterestRecord(
            date="2021-01-27", symbol="GME", exchange="A",
            volume=999, exercised=0, open_interest=888,
            product_kind="OSTK", expiration="2021-02-19",
        )
        merged = OccOptionsClient._merge_records([old], [updated])
        assert len(merged) == 1
        assert merged[0].volume == 999  # New wins
```

**Step 2: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_occ_options_client.py -v`
Expected: All tests PASS (parsing + cache + progress + merge)

**Step 3: Commit**

```bash
git add tests/data/test_occ_options_client.py
git commit -m "test: add cache, progress, and merge tests for OccOptionsClient"
```

---

### Task 4: OCC Client — `fetch_open_interest` Integration Test

**Files:**
- Modify: `tests/data/test_occ_options_client.py` (add fetch test with mocked HTTP)

**Step 1: Write the failing test**

Append to `tests/data/test_occ_options_client.py`:

```python
class TestFetchOpenInterest:
    """Tests for the full fetch_open_interest flow with mocked HTTP."""

    def test_fetch_downloads_and_caches(self, tmp_path: Path) -> None:
        """Fetch downloads bulk files, filters for symbol, saves cache."""
        client = _make_client(tmp_path)

        # Mock _download_day to return known records
        call_dates: list[str] = []

        def mock_download(date_str, symbol):
            call_dates.append(date_str)
            if date_str == "2021-01-04":
                return [OccOpenInterestRecord(
                    date="2021-01-04", symbol="GME", exchange="A",
                    volume=100, exercised=0, open_interest=50,
                    product_kind="OSTK", expiration="2021-02-19",
                )]
            return []  # No data for other days

        with patch.object(client, "_download_day", side_effect=mock_download):
            records = client.fetch_open_interest(
                "GME", start_date="2021-01-04",
            )

        # Should have called download for weekdays
        assert len(call_dates) > 0
        assert "2021-01-04" in call_dates

        # Should have cached records
        cached = client._load_cache("GME")
        assert cached is not None
        assert len(cached) >= 1

        # Progress should be saved
        progress = client._load_progress("GME")
        assert "2021-01-04" in progress

    def test_fetch_skips_already_downloaded(self, tmp_path: Path) -> None:
        """Dates in progress file are not re-downloaded."""
        client = _make_client(tmp_path)

        # Pre-populate progress
        client._save_progress("GME", {"2021-01-04", "2021-01-05"})

        call_dates: list[str] = []

        def mock_download(date_str, symbol):
            call_dates.append(date_str)
            return []

        with patch.object(client, "_download_day", side_effect=mock_download):
            client.fetch_open_interest(
                "GME", start_date="2021-01-04",
            )

        # Should NOT have downloaded pre-populated dates
        assert "2021-01-04" not in call_dates
        assert "2021-01-05" not in call_dates

    def test_fetch_returns_cached_when_all_done(self, tmp_path: Path) -> None:
        """When all dates are in progress, returns cache without downloading."""
        client = _make_client(tmp_path)

        # Pre-populate cache and mark all dates as done
        records = _sample_records()
        client._save_cache("GME", records)

        # Mark a large range as done
        from datetime import date as dt, timedelta
        all_dates = set()
        current = dt(2021, 1, 4)
        while current <= dt.today():
            if current.weekday() < 5:
                all_dates.add(current.isoformat())
            current += timedelta(days=1)
        client._save_progress("GME", all_dates)

        call_count = 0

        def mock_download(date_str, symbol):
            nonlocal call_count
            call_count += 1
            return []

        with patch.object(client, "_download_day", side_effect=mock_download):
            result = client.fetch_open_interest("GME")

        assert call_count == 0  # No downloads
        assert len(result) == 2  # Returns cached
```

**Step 2: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_occ_options_client.py::TestFetchOpenInterest -v`
Expected: All 3 tests PASS

**Step 3: Commit**

```bash
git add tests/data/test_occ_options_client.py
git commit -m "test: add fetch_open_interest integration tests"
```

---

### Task 5: Wire Up Exports

**Files:**
- Modify: `src/stockdownloader/data/__init__.py`

**Step 1: Add import and export**

Add to `src/stockdownloader/data/__init__.py`:

```python
from stockdownloader.data.occ_options_client import OccOptionsClient
```

Add `"OccOptionsClient"` to `__all__`.

**Step 2: Run full test suite to verify no regressions**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/stockdownloader/data/__init__.py
git commit -m "feat: export OccOptionsClient from data package"
```

---

### Task 6: GME Options Analysis Script

**Files:**
- Create: `scripts/gme_options_analysis.py`

**Step 1: Write the analysis script**

```python
#!/usr/bin/env python3
"""GME options open interest analysis.

Loads OCC open interest data and cross-references with FTD and short
volume data to detect synthetic short position patterns:

1. Total OI (in shares) vs float and outstanding
2. Daily OI changes and spikes
3. Expirations with anomalous OI concentration
4. Correlation with FTD and short volume spikes

Usage:
    python3 scripts/gme_options_analysis.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHARES_OUTSTANDING = 69_750_000  # SEC filing, Jan 30, 2021
FREE_FLOAT = 50_650_000  # Consistent with SEC "~140%" statement
CONTRACT_MULTIPLIER = 100  # Each options contract = 100 shares


def load_occ_data(symbol: str = "GME") -> list[dict]:
    """Load OCC open interest records."""
    oi_file = DATA_DIR / symbol / "occ_open_interest.json"
    if not oi_file.exists():
        logger.error("OCC data not found: %s", oi_file)
        logger.info(
            "Run the OCC backfill first:\n"
            "  python3 -c \"\n"
            "from stockdownloader.data.occ_options_client import OccOptionsClient\n"
            "c = OccOptionsClient()\n"
            "records = c.fetch_open_interest('GME')\n"
            "print(f'Records: {len(records)}')\n"
            "\""
        )
        return []
    return json.loads(oi_file.read_text(encoding="utf-8"))


def load_ftd_data(symbol: str = "GME") -> list[dict]:
    """Load FTD data."""
    ftd_file = DATA_DIR / symbol / "ftd_data.json"
    if not ftd_file.exists():
        return []
    return json.loads(ftd_file.read_text(encoding="utf-8"))


def load_short_volume(symbol: str = "GME") -> list[dict]:
    """Load short volume data."""
    sv_file = DATA_DIR / symbol / "short_volume.json"
    if not sv_file.exists():
        return []
    return json.loads(sv_file.read_text(encoding="utf-8"))


def analyze_daily_oi(records: list[dict]) -> dict[str, dict]:
    """Aggregate daily OI across all exchanges and expirations.

    Returns {date: {total_oi_contracts, total_oi_shares, total_volume,
                    num_expirations, num_exchanges}}.
    """
    daily: dict[str, dict] = defaultdict(lambda: {
        "total_oi_contracts": 0,
        "total_oi_shares": 0,
        "total_volume": 0,
        "total_exercised": 0,
        "num_expirations": 0,
        "num_exchanges": 0,
        "expirations": set(),
        "exchanges": set(),
    })

    for r in records:
        d = daily[r["date"]]
        d["total_oi_contracts"] += r["open_interest"]
        d["total_oi_shares"] += r["open_interest"] * CONTRACT_MULTIPLIER
        d["total_volume"] += r["volume"]
        d["total_exercised"] += r["exercised"]
        d["expirations"].add(r["expiration"])
        d["exchanges"].add(r["exchange"])

    # Finalize counts
    for d in daily.values():
        d["num_expirations"] = len(d["expirations"])
        d["num_exchanges"] = len(d["exchanges"])
        del d["expirations"]
        del d["exchanges"]

    return dict(sorted(daily.items()))


def print_summary(daily_oi: dict[str, dict]) -> None:
    """Print summary statistics."""
    if not daily_oi:
        logger.info("No OI data to analyze.")
        return

    dates = sorted(daily_oi.keys())
    logger.info("=" * 70)
    logger.info("GME OPTIONS OPEN INTEREST ANALYSIS")
    logger.info("=" * 70)
    logger.info("Date range: %s to %s (%d trading days)", dates[0], dates[-1], len(dates))
    logger.info("")

    # Overall stats
    oi_shares = [d["total_oi_shares"] for d in daily_oi.values()]
    avg_oi = sum(oi_shares) / len(oi_shares)
    max_oi = max(oi_shares)
    max_oi_date = dates[oi_shares.index(max_oi)]
    latest_oi = oi_shares[-1]

    logger.info("--- OPEN INTEREST (in equivalent shares) ---")
    logger.info("Average daily OI:    %12s shares (%5.1f%% of float)",
                f"{avg_oi:,.0f}", avg_oi / FREE_FLOAT * 100)
    logger.info("Peak daily OI:       %12s shares (%5.1f%% of float) on %s",
                f"{max_oi:,.0f}", max_oi / FREE_FLOAT * 100, max_oi_date)
    logger.info("Latest daily OI:     %12s shares (%5.1f%% of float)",
                f"{latest_oi:,.0f}", latest_oi / FREE_FLOAT * 100)
    logger.info("Shares outstanding:  %12s", f"{SHARES_OUTSTANDING:,}")
    logger.info("Free float:          %12s", f"{FREE_FLOAT:,}")
    logger.info("")

    # Top 10 peak OI days
    ranked = sorted(daily_oi.items(), key=lambda x: x[1]["total_oi_shares"], reverse=True)
    logger.info("--- TOP 10 PEAK OI DAYS ---")
    logger.info("%-12s %15s %10s %10s", "Date", "OI (shares)", "% Float", "Volume")
    for dt, d in ranked[:10]:
        logger.info(
            "%-12s %15s %9.1f%% %10s",
            dt,
            f"{d['total_oi_shares']:,}",
            d["total_oi_shares"] / FREE_FLOAT * 100,
            f"{d['total_volume']:,}",
        )
    logger.info("")

    # Detect large day-over-day spikes
    logger.info("--- LARGEST DAILY OI CHANGES ---")
    changes = []
    for i in range(1, len(dates)):
        prev_oi = daily_oi[dates[i - 1]]["total_oi_shares"]
        curr_oi = daily_oi[dates[i]]["total_oi_shares"]
        change = curr_oi - prev_oi
        pct_change = change / prev_oi * 100 if prev_oi > 0 else 0
        changes.append((dates[i], change, pct_change, curr_oi))

    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    logger.info("%-12s %15s %10s %15s", "Date", "Change", "% Change", "New OI")
    for dt, chg, pct, new_oi in changes[:10]:
        logger.info(
            "%-12s %+14s %+9.1f%% %15s",
            dt, f"{chg:,}", pct, f"{new_oi:,}",
        )


def main() -> None:
    records = load_occ_data()
    if not records:
        return

    logger.info("Loaded %d OCC open interest records for GME", len(records))

    daily_oi = analyze_daily_oi(records)
    print_summary(daily_oi)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/gme_options_analysis.py
git commit -m "feat: add GME options OI analysis script"
```

---

### Task 7: Run Backfill & Verify

**Step 1: Start OCC backfill for GME**

```bash
python3 -c "
import sys, logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s', force=True)
from stockdownloader.data.occ_options_client import OccOptionsClient
c = OccOptionsClient()
records = c.fetch_open_interest('GME')
print(f'Total records: {len(records)}')
"
```

Expected: ~1,250 requests over ~10 minutes, produces `data/GME/occ_open_interest.json`.

**Step 2: Run analysis script**

```bash
python3 scripts/gme_options_analysis.py
```

Expected: Summary with daily OI, peak dates, OI as % of float, day-over-day changes.

**Step 3: Run full test suite for regression**

```bash
python3 -m pytest tests/ -x -q
```

Expected: All tests PASS.

**Step 4: Commit data**

```bash
git add data/GME/occ_open_interest.json data/GME/occ_options_progress.json
git commit -m "data: add GME OCC open interest backfill (Jan 2021-Feb 2026)"
```

---

## Summary of Files

| File | Action | Task |
|------|--------|------|
| `src/stockdownloader/model/regulatory_records.py` | Modify | 1 |
| `src/stockdownloader/model/__init__.py` | Modify | 1 |
| `tests/model/test_occ_open_interest_record.py` | Create | 1 |
| `src/stockdownloader/data/occ_options_client.py` | Create | 2 |
| `tests/data/test_occ_options_client.py` | Create | 2, 3, 4 |
| `src/stockdownloader/data/__init__.py` | Modify | 5 |
| `scripts/gme_options_analysis.py` | Create | 6 |
| `data/GME/occ_open_interest.json` | Create | 7 |
| `data/GME/occ_options_progress.json` | Create | 7 |
