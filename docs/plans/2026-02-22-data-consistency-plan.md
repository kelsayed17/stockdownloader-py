# Data Consistency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standardize all flat data files to CSV, move progress files to `.progress/`, and add symbol variant handling.

**Architecture:** Add CSV read/write and progress-dir helpers to `BaseDataClient`. Update each client's `_save_cache`/`_load_cache` to use CSV with JSON fallback for migration. Extend `SymbolInfo` with `parent`/`security_type` fields. Update all scripts that read data files.

**Tech Stack:** Python stdlib `csv` module, `dataclasses.fields()` for introspection, existing `BaseDataClient` pattern.

---

### Task 1: BaseDataClient — CSV and progress helpers

**Files:**
- Modify: `src/stockdownloader/data/base_client.py`
- Create: `tests/data/test_base_client_csv.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_base_client_csv.py
"""Tests for BaseDataClient CSV and progress-dir helpers."""

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from stockdownloader.data.base_client import BaseDataClient


@dataclass(frozen=True, slots=True)
class _SampleRecord:
    date: str
    symbol: str
    value: int
    ratio: float
    flag: bool


@dataclass(frozen=True, slots=True)
class _DecimalRecord:
    date: str
    price: Decimal


class TestSaveCsv:
    def test_save_creates_csv_with_header(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        records = [
            _SampleRecord("2024-01-01", "GME", 100, 0.5, True),
            _SampleRecord("2024-01-02", "GME", 200, 0.75, False),
        ]
        client._save_csv("GME", "test_data.csv", records)
        path = tmp_path / "GME" / "test_data.csv"
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert lines[0] == "date,symbol,value,ratio,flag"
        assert lines[1] == "2024-01-01,GME,100,0.5,True"
        assert lines[2] == "2024-01-02,GME,200,0.75,False"

    def test_save_empty_list_creates_header_only(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        client._save_csv("GME", "empty.csv", [], _SampleRecord)
        path = tmp_path / "GME" / "empty.csv"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert lines[0] == "date,symbol,value,ratio,flag"

    def test_save_decimal_field(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        records = [_DecimalRecord("2024-01-01", Decimal("12.50"))]
        client._save_csv("GME", "dec.csv", records)
        path = tmp_path / "GME" / "dec.csv"
        lines = path.read_text().strip().splitlines()
        assert lines[1] == "2024-01-01,12.50"


class TestLoadCsv:
    def test_load_roundtrip(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        original = [
            _SampleRecord("2024-01-01", "GME", 100, 0.5, True),
            _SampleRecord("2024-01-02", "GME", 200, 0.75, False),
        ]
        client._save_csv("GME", "rt.csv", original)
        loaded = client._load_csv("GME", "rt.csv", _SampleRecord)
        assert loaded == original

    def test_load_nonexistent_returns_none(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        (tmp_path / "GME").mkdir()
        assert client._load_csv("GME", "nope.csv", _SampleRecord) is None

    def test_load_corrupt_returns_none(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        d = tmp_path / "GME"
        d.mkdir()
        (d / "bad.csv").write_text("not,a,csv\n\x00\x01\x02")
        assert client._load_csv("GME", "bad.csv", _SampleRecord) is None

    def test_load_decimal_field(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        original = [_DecimalRecord("2024-01-01", Decimal("12.50"))]
        client._save_csv("GME", "dec.csv", original)
        loaded = client._load_csv("GME", "dec.csv", _DecimalRecord)
        assert loaded is not None
        assert loaded[0].price == Decimal("12.50")

    def test_load_bool_coercion(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        original = [_SampleRecord("2024-01-01", "GME", 1, 0.0, False)]
        client._save_csv("GME", "bool.csv", original)
        loaded = client._load_csv("GME", "bool.csv", _SampleRecord)
        assert loaded is not None
        assert loaded[0].flag is False


class TestProgressDir:
    def test_progress_dir_created(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        pdir = client._progress_dir("GME")
        assert pdir == tmp_path / "GME" / ".progress"
        assert pdir.is_dir()

    def test_progress_dir_idempotent(self, tmp_path):
        client = BaseDataClient(data_dir=str(tmp_path))
        p1 = client._progress_dir("GME")
        p2 = client._progress_dir("GME")
        assert p1 == p2
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/data/test_base_client_csv.py -v`
Expected: FAIL — `_save_csv`, `_load_csv`, `_progress_dir` do not exist

**Step 3: Implement CSV helpers in BaseDataClient**

Add to `src/stockdownloader/data/base_client.py` after the existing JSON caching helpers (after line 182):

```python
import csv
from dataclasses import asdict, fields as dc_fields
from decimal import Decimal

# Add to _coerce_value as a module-level function:

def _coerce_value(raw: str, field_type: type) -> Any:
    """Convert a CSV string value to the appropriate Python type."""
    # Handle Optional / Union types
    origin = getattr(field_type, "__origin__", None)
    if origin is Union:
        args = [a for a in field_type.__args__ if a is not type(None)]
        if args:
            field_type = args[0]

    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    if field_type is bool:
        return raw == "True"
    if field_type is Decimal:
        return Decimal(raw)
    return raw

# Add to BaseDataClient class:

def _save_csv(
    self,
    symbol: str,
    filename: str,
    records: list,
    record_cls: type | None = None,
) -> None:
    """Write a list of dataclass instances to CSV.

    If *records* is empty, *record_cls* must be provided so the
    header row can still be written.
    """
    if records:
        fieldnames = [f.name for f in dc_fields(records[0])]
    elif record_cls is not None:
        fieldnames = [f.name for f in dc_fields(record_cls)]
    else:
        return  # nothing to write and no schema
    path = self._symbol_dir(symbol) / filename
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
    except OSError as exc:
        logger.warning("Failed to save CSV %s: %s", path, exc)

def _load_csv(
    self,
    symbol: str,
    filename: str,
    record_cls: type,
) -> list | None:
    """Read a CSV file into a list of dataclass instances.

    Returns ``None`` if the file does not exist or is corrupt.
    """
    path = self._symbol_dir(symbol) / filename
    if not path.exists():
        return None
    try:
        type_map = {f.name: f.type for f in dc_fields(record_cls)}
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            results = []
            for row in reader:
                coerced = {
                    k: _coerce_value(v, type_map[k])
                    for k, v in row.items()
                    if k in type_map
                }
                results.append(record_cls(**coerced))
            return results
    except Exception as exc:
        logger.warning("Failed to load CSV %s: %s", path, exc)
        return None

def _progress_dir(self, symbol: str) -> Path:
    """Return hidden ``.progress`` directory for *symbol*, creating it."""
    d = self._symbol_dir(symbol) / ".progress"
    d.mkdir(parents=True, exist_ok=True)
    return d
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_base_client_csv.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/data/base_client.py tests/data/test_base_client_csv.py
git commit -m "feat: add CSV save/load and .progress dir helpers to BaseDataClient"
```

---

### Task 2: SymbolInfo — variant fields and lookup functions

**Files:**
- Modify: `src/stockdownloader/model/symbol_info.py`
- Modify: `tests/model/test_symbol_info.py` (or create if it doesn't exist)

**Step 1: Write the failing tests**

```python
# tests/model/test_symbol_info.py
from datetime import date

import pytest

from stockdownloader.model.symbol_info import (
    SymbolInfo,
    get_symbol_info,
    get_variants,
    get_family,
    SYMBOL_REGISTRY,
)


class TestSymbolInfoVariants:
    def test_default_parent_is_none(self):
        info = get_symbol_info("GME")
        assert info is not None
        assert info.parent is None
        assert info.security_type == "common"

    def test_gmews_registered(self):
        info = get_symbol_info("GMEWS")
        assert info is not None
        assert info.parent == "GME"
        assert info.security_type == "warrant"
        assert info.cusip == "36467W117"

    def test_get_variants_returns_children(self):
        variants = get_variants("GME")
        symbols = [v.symbol for v in variants]
        assert "GMEWS" in symbols

    def test_get_variants_for_leaf_returns_empty(self):
        assert get_variants("GMEWS") == []

    def test_get_variants_unknown_returns_empty(self):
        assert get_variants("ZZZZZZ") == []

    def test_get_family_from_parent(self):
        family = get_family("GME")
        symbols = [f.symbol for f in family]
        assert "GME" in symbols
        assert "GMEWS" in symbols

    def test_get_family_from_child(self):
        family = get_family("GMEWS")
        symbols = [f.symbol for f in family]
        assert "GME" in symbols
        assert "GMEWS" in symbols

    def test_get_family_unknown_returns_empty(self):
        assert get_family("ZZZZZZ") == []

    def test_parent_field_preserved_frozen(self):
        info = SymbolInfo("TEST", "000000000", date(2020, 1, 1),
                          "Test Corp", "NYSE", parent="GME",
                          security_type="warrant")
        assert info.parent == "GME"
        assert info.security_type == "warrant"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/model/test_symbol_info.py -v`
Expected: FAIL — `parent`, `security_type`, `get_variants`, `get_family` do not exist

**Step 3: Implement**

Modify `src/stockdownloader/model/symbol_info.py`:

1. Add `parent: str | None = None` and `security_type: str = "common"` fields to `SymbolInfo` dataclass (after `exchange`).

2. Register GMEWS:
```python
_register(
    SymbolInfo(
        symbol="GMEWS",
        cusip="36467W117",
        ipo_date=date(2025, 10, 7),
        name="GameStop Corp Warrants",
        exchange="NYSE",
        parent="GME",
        security_type="warrant",
    ),
)
```

3. Add lookup functions:
```python
def get_variants(symbol: str) -> list[SymbolInfo]:
    """Return all variant symbols whose parent is *symbol*."""
    upper = symbol.upper()
    return [info for info in SYMBOL_REGISTRY.values()
            if info.parent == upper]


def get_family(symbol: str) -> list[SymbolInfo]:
    """Return *symbol* and all its variants (full family).

    Works from either the parent or a child — always returns the
    complete family rooted at the parent.
    """
    info = get_symbol_info(symbol)
    if info is None:
        return []
    root = info.parent or info.symbol
    root_info = get_symbol_info(root)
    if root_info is None:
        return [info]
    return [root_info] + get_variants(root)
```

4. Export `get_variants` and `get_family` from `model/__init__.py`.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/model/test_symbol_info.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/model/symbol_info.py src/stockdownloader/model/__init__.py tests/model/test_symbol_info.py
git commit -m "feat: add symbol variant handling (parent, security_type, get_variants, get_family)"
```

---

### Task 3: FinraShortVolumeClient — JSON → CSV

**Files:**
- Modify: `src/stockdownloader/data/finra_short_volume_client.py`
- Modify: `tests/data/test_finra_short_volume_client.py`

**Step 1: Update tests for CSV**

Find existing cache roundtrip tests and update expectations from `.json` to `.csv`. Add a JSON-fallback migration test:

```python
def test_save_creates_csv(self, tmp_path):
    client = FinraShortVolumeClient(data_dir=str(tmp_path))
    records = [ShortVolumeRecord("2024-01-01", "GME", 100, 200, 5, 0.5)]
    client._save_cache("GME", records)
    csv_path = tmp_path / "GME" / "short_volume.csv"
    json_path = tmp_path / "GME" / "short_volume.json"
    assert csv_path.exists()
    assert not json_path.exists()

def test_load_csv_roundtrip(self, tmp_path):
    client = FinraShortVolumeClient(data_dir=str(tmp_path))
    original = [ShortVolumeRecord("2024-01-01", "GME", 100, 200, 5, 0.5)]
    client._save_cache("GME", original)
    loaded = client._load_cache("GME")
    assert loaded == original

def test_load_falls_back_to_json(self, tmp_path):
    """If CSV doesn't exist but JSON does, load JSON and migrate."""
    client = FinraShortVolumeClient(data_dir=str(tmp_path))
    d = tmp_path / "GME"
    d.mkdir(parents=True)
    import json
    from dataclasses import asdict
    rec = ShortVolumeRecord("2024-01-01", "GME", 100, 200, 5, 0.5)
    (d / "short_volume.json").write_text(json.dumps([asdict(rec)]))
    loaded = client._load_cache("GME")
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].date == "2024-01-01"
    # Migration should have created the CSV
    assert (d / "short_volume.csv").exists()
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/data/test_finra_short_volume_client.py -v -k cache`
Expected: FAIL — still writing JSON

**Step 3: Update `_save_cache` and `_load_cache`**

Replace the `_save_cache` method:
```python
def _save_cache(self, symbol: str, records: list[ShortVolumeRecord]) -> None:
    self._save_csv(symbol, "short_volume.csv", records)
```

Replace the `_load_cache` method:
```python
def _load_cache(self, symbol: str) -> list[ShortVolumeRecord] | None:
    # Primary: CSV
    records = self._load_csv(symbol, "short_volume.csv", ShortVolumeRecord)
    if records is not None:
        records.sort(key=lambda r: r.date)
        return records

    # Fallback: JSON (legacy migration)
    sym_dir = self._symbol_dir(symbol)
    json_path = sym_dir / "short_volume.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        records = [ShortVolumeRecord(**r) for r in data]
        records.sort(key=lambda r: r.date)
        # Migrate: write CSV for future loads
        self._save_cache(symbol, records)
        logger.info("Migrated %s to CSV", json_path)
        return records
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to load short volume cache for %s: %s", symbol, exc)
        return None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_finra_short_volume_client.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/data/finra_short_volume_client.py tests/data/test_finra_short_volume_client.py
git commit -m "refactor: FinraShortVolumeClient writes CSV instead of JSON"
```

---

### Task 4: FinraShortInterestClient — JSON → CSV

Same pattern as Task 3. File: `short_interest.csv`.

**Files:**
- Modify: `src/stockdownloader/data/finra_short_interest_client.py`
- Modify: `tests/data/test_finra_short_interest_client.py`

Replace `_save_cache` with `self._save_csv(symbol, "short_interest.csv", records)`.

Replace `_load_cache` with CSV-first + JSON fallback (same pattern as Task 3 but for `ShortInterestRecord` and `short_interest.json`/`.csv`).

Note: The existing `_load_cache` manually constructs records field-by-field and handles `.get("short_interest_pct", 0.0)` for old data. The JSON fallback path must preserve this for backwards compatibility.

**Commit:** `refactor: FinraShortInterestClient writes CSV instead of JSON`

---

### Task 5: FinraDarkPoolClient — JSON → CSV

Same pattern. File: `dark_pool.csv`.

**Files:**
- Modify: `src/stockdownloader/data/finra_dark_pool_client.py`
- Modify: `tests/data/test_finra_dark_pool_client.py`

Note: The existing `_load_cache` manually constructs records and uses `.get("ats_pct", 0.0)`. JSON fallback path must preserve this.

**Commit:** `refactor: FinraDarkPoolClient writes CSV instead of JSON`

---

### Task 6: OccOptionsClient — JSON → CSV + progress to `.progress/`

**Files:**
- Modify: `src/stockdownloader/data/occ_options_client.py`
- Modify: `tests/data/test_occ_options_client.py`

Two changes:

**6a: Cache → CSV**

Replace `_save_cache` / `_load_cache` same pattern as Task 3.
- New file: `occ_open_interest.csv`
- JSON fallback from `occ_open_interest.json`

**6b: Progress → `.progress/`**

Replace `_save_progress` / `_load_progress`:
```python
def _load_progress(self, symbol: str) -> set[str]:
    progress_file = self._progress_dir(symbol) / "occ_options.json"
    # Also check legacy location
    legacy = self._symbol_dir(symbol) / "occ_options_progress.json"
    if not progress_file.exists() and legacy.exists():
        legacy.rename(progress_file)
        logger.info("Migrated %s → %s", legacy, progress_file)
    if not progress_file.exists():
        return set()
    try:
        return set(json.loads(progress_file.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()

def _save_progress(self, symbol: str, dates: set[str]) -> None:
    progress_file = self._progress_dir(symbol) / "occ_options.json"
    try:
        progress_file.write_text(json.dumps(sorted(dates)), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to save OCC progress for %s: %s", symbol, exc)
```

**Commit:** `refactor: OccOptionsClient writes CSV, progress to .progress/`

---

### Task 7: RegShoThresholdClient — JSON → CSV + progress to `.progress/`

**Files:**
- Modify: `src/stockdownloader/data/regsho_threshold_client.py`
- Modify: `tests/data/test_regsho_threshold_client.py`

Two changes:

**7a: Cache → CSV**
- New file: `regsho_threshold.csv`
- JSON fallback from `regsho_threshold.json`

**7b: NYSE progress → `.progress/`**

Update `_save_nyse_progress` / `_load_nyse_progress`:
- New path: `.progress/regsho_nyse.json`
- Legacy migration from `regsho_nyse_progress.json`

**Commit:** `refactor: RegShoThresholdClient writes CSV, progress to .progress/`

---

### Task 8: SecInsiderClient — JSON → CSV + quarterly cache to `.progress/`

**Files:**
- Modify: `src/stockdownloader/data/sec_insider_client.py`
- Modify: `tests/data/test_sec_insider_client.py`

Three changes:

**8a: insider_transactions.json → insider_transactions.csv**

Replace `_save_transaction_cache` / `_load_transaction_cache` with CSV pattern.

**8b: beneficial_owners.json → beneficial_owners.csv**

Replace `_save_beneficial_owners_cache` / `_load_beneficial_owners_cache` with CSV pattern.

**8c: insider/{YYYY}Q{Q}.json → .progress/insider/{YYYY}Q{Q}.json**

Update `_save_quarter_transactions` / `_load_quarter_transactions`:
- New path: `.progress/insider/{YYYY}Q{Q}.json`
- Legacy migration from `insider/{YYYY}Q{Q}.json`
- These stay JSON because they are progress/cache files (not persistent data).

**Commit:** `refactor: SecInsiderClient writes CSV, quarterly cache to .progress/`

---

### Task 9: SecOwnershipClient — quarterly cache to `.progress/`

**Files:**
- Modify: `src/stockdownloader/data/sec_ownership_client.py`
- Modify: `tests/data/test_sec_ownership_client.py`

One change — `ownership_13f.json` stays JSON (nested). Only the quarterly per-quarter cache moves:

**9a: ownership/{YYYY}Q{Q}.json → .progress/ownership/{YYYY}Q{Q}.json**

Update `_save_quarter_snapshot` / `_load_quarter_snapshot`:
- New path: `.progress/ownership/{YYYY}Q{Q}.json`
- Legacy migration from `ownership/{YYYY}Q{Q}.json`

**Commit:** `refactor: SecOwnershipClient quarterly cache to .progress/`

---

### Task 10: Update scripts — JSON reads → CSV reads

**Files:**
- Modify: `scripts/gme_options_analysis.py`
- Modify: `scripts/regsho_targeted_backfill.py`
- Modify: `scripts/archive/gme_holistic_report.py`
- Modify: `scripts/archive/gme_comprehensive_report.py`
- Modify: `scripts/archive/gme_squeeze_predictor.py`

For each script, replace `json.load(open(...json))` patterns with `csv.DictReader(open(...csv))` for the flat data files:

| Old pattern | New pattern |
|-------------|-------------|
| `json.loads(path / "ftd_data.json")` | `list(csv.DictReader(open(path / "ftd_data.csv")))` with int coercion |
| `json.loads(path / "short_volume.json")` | `list(csv.DictReader(open(path / "short_volume.csv")))` |
| `json.loads(path / "short_interest.json")` | `list(csv.DictReader(open(path / "short_interest.csv")))` |
| `json.loads(path / "dark_pool.json")` | `list(csv.DictReader(open(path / "dark_pool.csv")))` |
| `json.loads(path / "regsho_threshold.json")` | `list(csv.DictReader(open(path / "regsho_threshold.csv")))` |
| `json.loads(path / "occ_open_interest.json")` | `list(csv.DictReader(open(path / "occ_open_interest.csv")))` |
| `json.loads(path / "insider_transactions.json")` | `list(csv.DictReader(open(path / "insider_transactions.csv")))` |
| `json.loads(path / "beneficial_owners.json")` | `list(csv.DictReader(open(path / "beneficial_owners.csv")))` |

Note: CSV reads return all values as strings. Scripts need `int()` / `float()` coercion on numeric fields. Add a helper function `_load_csv(path)` to each script, or use the record classes directly.

Also update `regsho_targeted_backfill.py` progress file path from `regsho_nyse_progress.json` to `.progress/regsho_nyse.json`.

**Commit:** `refactor: update scripts to read CSV instead of JSON`

---

### Task 11: Update .gitignore + migrate existing data

**Files:**
- Modify: `.gitignore`

Add:
```
# Progress/cache tracking (rebuilable by re-running clients)
data/**/.progress/
```

**Commit:** `chore: gitignore .progress directories`

---

### Task 12: Full test suite regression

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass, no regressions

**Step 2: Run a live backfill to verify CSV output**

```bash
python3 -c "
from stockdownloader.data.occ_options_client import OccOptionsClient
c = OccOptionsClient()
records = c.fetch_open_interest('GME')
print(f'Records: {len(records)}')
import pathlib
csv_path = pathlib.Path('data/GME/occ_open_interest.csv')
print(f'CSV exists: {csv_path.exists()}')
print(f'CSV size: {csv_path.stat().st_size}')
# Show first 3 lines
lines = csv_path.read_text().splitlines()[:3]
for line in lines:
    print(line)
"
```

Expected: CSV file created with header + data rows.

**Step 3: Commit data files**

```bash
git add data/GME/*.csv data/GMEWS/*.csv
git commit -m "data: migrate GME and GMEWS data files from JSON to CSV"
```
