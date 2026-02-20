# Cache Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move per-symbol cache files into `{SYMBOL}/` subdirectories and add per-quarter EFTS snapshot caching to the ownership client so interrupted downloads don't lose work.

**Architecture:** Each client gets a `_symbol_cache_dir(symbol)` helper that returns `{cache_root}/{SYMBOL}/` (creating it if needed).  Load/save methods switch to the new paths with transparent legacy migration.  The ownership client additionally saves each EFTS quarter result as `{SYMBOL}/{YYYY}Q{Q}.json` so each quarter survives independently.

**Tech Stack:** Python 3.12, pathlib, json, pytest, `tmp_path` fixture

---

### Task 1: SecOwnershipClient — Per-Symbol Subdir + Legacy Migration

The ownership client is the most complex because it has both per-symbol JSON
results *and* shared bulk ZIPs.  Only the per-symbol JSON moves; bulk ZIPs stay
in `bulk_13f/`.

**Files:**
- Modify: `src/stockdownloader/data/sec_ownership_client.py:1038-1118`
- Test: `tests/data/test_sec_ownership_client.py`

**Step 1: Write failing tests for new cache paths**

Add a new test class `TestPerSymbolCacheDir` in `tests/data/test_sec_ownership_client.py`:

```python
class TestPerSymbolCacheDir:
    """Tests for per-symbol subdirectory cache layout."""

    def test_symbol_cache_dir_creates_subdir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._symbol_cache_dir("GME")
        assert sym_dir == client._cache_dir / "GME"
        assert sym_dir.is_dir()

    def test_symbol_cache_dir_uppercases(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._symbol_cache_dir("gme")
        assert sym_dir == client._cache_dir / "GME"

    def test_save_creates_file_in_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        snapshots = [
            OwnershipSnapshot(
                quarter_end="2024-03-31",
                symbol="AAPL",
                total_institutional_shares=1000,
                num_institutions=1,
                top_10_concentration=1.0,
                holdings=(
                    InstitutionalHolding(
                        filing_date="2024-05-15",
                        manager_name="TestFund",
                        manager_cik="123",
                        shares=1000,
                        value_usd=100,
                        share_class="COM",
                    ),
                ),
            ),
        ]
        client._save_cache("AAPL", snapshots)
        new_path = client._cache_dir / "AAPL" / "13f.json"
        old_path = client._cache_dir / "AAPL_13f.json"
        assert new_path.exists()
        assert not old_path.exists()

    def test_legacy_migration_on_load(self, tmp_path: Path) -> None:
        """Old flat file is moved into symbol subdir on first load."""
        client = _make_client(tmp_path)
        # Create a legacy-path file
        legacy = client._cache_dir / "GME_13f.json"
        legacy.write_text(json.dumps([{
            "quarter_end": "2024-03-31",
            "symbol": "GME",
            "total_institutional_shares": 1000,
            "num_institutions": 1,
            "top_10_concentration": 1.0,
            "holdings": [],
        }]), encoding="utf-8")

        loaded = client._load_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        # Legacy file should be gone, new path should exist
        assert not legacy.exists()
        new_path = client._cache_dir / "GME" / "13f.json"
        assert new_path.exists()

    def test_legacy_ownership_migration(self, tmp_path: Path) -> None:
        """Even older {SYMBOL}_ownership.json files are migrated."""
        client = _make_client(tmp_path)
        legacy = client._cache_dir / "GME_ownership.json"
        legacy.write_text(json.dumps([{
            "quarter_end": "2024-03-31",
            "symbol": "GME",
            "total_institutional_shares": 1000,
            "num_institutions": 1,
            "top_10_concentration": 1.0,
            "holdings": [],
        }]), encoding="utf-8")

        loaded = client._load_cache("GME")
        assert loaded is not None
        assert not legacy.exists()
        assert (client._cache_dir / "GME" / "13f.json").exists()

    def test_load_roundtrip_new_path(self, tmp_path: Path) -> None:
        """Save + load works with new per-symbol path."""
        client = _make_client(tmp_path)
        holdings = (
            InstitutionalHolding(
                filing_date="2024-05-15",
                manager_name="Vanguard",
                manager_cik="102909",
                shares=5_000_000,
                value_usd=150_000,
                share_class="COM",
            ),
        )
        snapshots = [
            OwnershipSnapshot(
                quarter_end="2024-03-31",
                symbol="GME",
                total_institutional_shares=5_000_000,
                num_institutions=1,
                top_10_concentration=1.0,
                holdings=holdings,
            ),
        ]
        client._save_cache("GME", snapshots)
        loaded = client._load_cache("GME")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].quarter_end == "2024-03-31"
        assert loaded[0].holdings[0].manager_name == "Vanguard"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestPerSymbolCacheDir -v`
Expected: FAIL — `_symbol_cache_dir` does not exist yet.

**Step 3: Implement `_symbol_cache_dir` and update `_load_cache` / `_save_cache`**

In `sec_ownership_client.py`, add a new method right before `_load_cache` (around line 1035):

```python
def _symbol_cache_dir(self, symbol: str) -> Path:
    """Return per-symbol cache subdirectory, creating it if needed."""
    d = self._cache_dir / symbol.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d
```

Update `_load_cache` (currently lines 1038-1081) to:

```python
def _load_cache(self, symbol: str) -> list[OwnershipSnapshot] | None:
    """Load cached ownership snapshots for *symbol*."""
    sym_dir = self._symbol_cache_dir(symbol)
    cache_file = sym_dir / "13f.json"

    # Legacy migration: move old flat files into symbol subdir
    if not cache_file.exists():
        for legacy_name in (f"{symbol}_13f.json", f"{symbol}_ownership.json"):
            legacy = self._cache_dir / legacy_name
            if legacy.exists():
                legacy.rename(cache_file)
                logger.info("Migrated %s → %s", legacy, cache_file)
                break

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        # ... rest unchanged ...
```

Update `_save_cache` (currently lines 1083-1118) to write to the new path:

```python
def _save_cache(
    self,
    symbol: str,
    snapshots: list[OwnershipSnapshot],
) -> None:
    """Persist ownership snapshots to JSON cache."""
    cache_file = self._symbol_cache_dir(symbol) / "13f.json"
    # ... rest unchanged ...
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py -v`
Expected: ALL PASS (both old and new tests).

**Step 5: Update the old `test_cache_file_naming` test**

The existing test at line 604 asserts `(client._cache_dir / "AAPL_13f.json").exists()`.  Update it to check the new path:

```python
def test_cache_file_naming(self, tmp_path: Path) -> None:
    # ... (same setup as before) ...
    client._save_cache("AAPL", snapshots)
    assert (client._cache_dir / "AAPL" / "13f.json").exists()
```

**Step 6: Run full test suite**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py -v`
Expected: ALL PASS.

**Step 7: Commit**

```bash
git add src/stockdownloader/data/sec_ownership_client.py tests/data/test_sec_ownership_client.py
git commit -m "refactor: move ownership cache to per-symbol subdirs with legacy migration"
```

---

### Task 2: SecOwnershipClient — Per-Quarter EFTS Snapshot Caching

This is the critical feature.  Each EFTS quarter result is saved as a separate
JSON file so interrupted downloads don't lose work.

**Files:**
- Modify: `src/stockdownloader/data/sec_ownership_client.py:614-738`
- Test: `tests/data/test_sec_ownership_client.py`

**Step 1: Write failing tests**

Add a new test class `TestEftsQuarterSnapshotCache`:

```python
class TestEftsQuarterSnapshotCache:
    """Tests for per-quarter EFTS snapshot caching."""

    def test_save_quarter_snapshot(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        snap = OwnershipSnapshot(
            quarter_end="2012-12-31",
            symbol="GME",
            total_institutional_shares=45_000_000,
            num_institutions=312,
            top_10_concentration=0.35,
            holdings=(
                InstitutionalHolding(
                    filing_date="2012-12-31",
                    manager_name="Fidelity",
                    manager_cik="12345",
                    shares=5_000_000,
                    value_usd=50_000,
                    share_class="COM",
                ),
            ),
        )
        client._save_quarter_snapshot("GME", 2012, 4, snap)
        path = client._cache_dir / "GME" / "2012Q4.json"
        assert path.exists()

    def test_load_quarter_snapshot(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        snap = OwnershipSnapshot(
            quarter_end="2012-12-31",
            symbol="GME",
            total_institutional_shares=45_000_000,
            num_institutions=312,
            top_10_concentration=0.35,
            holdings=(
                InstitutionalHolding(
                    filing_date="2012-12-31",
                    manager_name="Fidelity",
                    manager_cik="12345",
                    shares=5_000_000,
                    value_usd=50_000,
                    share_class="COM",
                ),
            ),
        )
        client._save_quarter_snapshot("GME", 2012, 4, snap)
        loaded = client._load_quarter_snapshot("GME", 2012, 4)
        assert loaded is not None
        assert loaded.quarter_end == "2012-12-31"
        assert loaded.total_institutional_shares == 45_000_000
        assert loaded.num_institutions == 312
        assert len(loaded.holdings) == 1
        assert loaded.holdings[0].manager_name == "Fidelity"

    def test_load_missing_quarter_returns_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        loaded = client._load_quarter_snapshot("GME", 2012, 4)
        assert loaded is None

    def test_load_corrupt_quarter_returns_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        sym_dir = client._symbol_cache_dir("GME")
        (sym_dir / "2012Q4.json").write_text("{{{bad", encoding="utf-8")
        loaded = client._load_quarter_snapshot("GME", 2012, 4)
        assert loaded is None

    def test_force_refresh_preserves_quarter_snapshots(
        self, tmp_path: Path
    ) -> None:
        """force_refresh deletes merged 13f.json but keeps quarter files."""
        client = _make_client(tmp_path)
        sym_dir = client._symbol_cache_dir("GME")
        # Create a quarter snapshot and a merged file
        (sym_dir / "2012Q4.json").write_text("{}", encoding="utf-8")
        (sym_dir / "13f.json").write_text("[]", encoding="utf-8")

        # Simulate force_refresh: delete merged file
        merged = sym_dir / "13f.json"
        if merged.exists():
            merged.unlink()

        # Quarter snapshot should survive
        assert (sym_dir / "2012Q4.json").exists()
        assert not (sym_dir / "13f.json").exists()
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestEftsQuarterSnapshotCache -v`
Expected: FAIL — `_save_quarter_snapshot` and `_load_quarter_snapshot` don't exist.

**Step 3: Implement quarter snapshot save/load**

Add two new methods after `_save_cache` in `sec_ownership_client.py`:

```python
def _save_quarter_snapshot(
    self,
    symbol: str,
    year: int,
    quarter: int,
    snapshot: OwnershipSnapshot,
) -> None:
    """Save a single quarter's EFTS result as a per-quarter cache file."""
    path = self._symbol_cache_dir(symbol) / f"{year}Q{quarter}.json"
    data = {
        "quarter_end": snapshot.quarter_end,
        "symbol": snapshot.symbol,
        "total_institutional_shares": snapshot.total_institutional_shares,
        "num_institutions": snapshot.num_institutions,
        "top_10_concentration": snapshot.top_10_concentration,
        "holdings": [
            {
                "filing_date": h.filing_date,
                "manager_name": h.manager_name,
                "manager_cik": h.manager_cik,
                "shares": h.shares,
                "value_usd": h.value_usd,
                "share_class": h.share_class,
            }
            for h in snapshot.holdings
        ],
    }
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "Failed to save quarter snapshot %s: %s", path, exc,
        )

def _load_quarter_snapshot(
    self,
    symbol: str,
    year: int,
    quarter: int,
) -> OwnershipSnapshot | None:
    """Load a cached per-quarter EFTS result."""
    path = self._symbol_cache_dir(symbol) / f"{year}Q{quarter}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        holdings = tuple(
            InstitutionalHolding(
                filing_date=h["filing_date"],
                manager_name=h["manager_name"],
                manager_cik=h["manager_cik"],
                shares=h["shares"],
                value_usd=h["value_usd"],
                share_class=h["share_class"],
            )
            for h in data.get("holdings", [])
        )
        return OwnershipSnapshot(
            quarter_end=data["quarter_end"],
            symbol=data["symbol"],
            total_institutional_shares=data["total_institutional_shares"],
            num_institutions=data["num_institutions"],
            top_10_concentration=data["top_10_concentration"],
            holdings=holdings,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning(
            "Failed to load quarter snapshot %s: %s", path, exc,
        )
        return None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestEftsQuarterSnapshotCache -v`
Expected: ALL PASS.

**Step 5: Wire quarter snapshot caching into `_fetch_from_efts`**

In `_fetch_from_efts` (starts at line 614), add a cache check at the top
and a cache save after successful parsing:

At the start of `_fetch_from_efts`, right after the docstring (line 625):

```python
# Check per-quarter cache first
cached_snap = self._load_quarter_snapshot(symbol, year, quarter)
if cached_snap is not None:
    logger.info(
        "Using cached EFTS snapshot for %s Q%d %d",
        symbol, quarter, year,
    )
    return cached_snap
```

At the end, right before the final `return` (the `try: return OwnershipSnapshot(...)` block around line 728):

After building the snapshot but before returning, save it:

```python
try:
    snap = OwnershipSnapshot(
        quarter_end=quarter_end,
        symbol=symbol,
        total_institutional_shares=total_shares,
        num_institutions=len(holdings),
        top_10_concentration=top10_conc,
        holdings=tuple(sorted_h),
    )
    self._save_quarter_snapshot(symbol, year, quarter, snap)
    return snap
except ValueError:
    return None
```

**Step 6: Wire force_refresh to delete only merged file**

In `fetch_ownership_snapshots`, update the `force_refresh` handling (around
line 242-250).  After `if not force_refresh:` block, add:

```python
if force_refresh:
    # Delete merged result but preserve per-quarter snapshots
    merged = self._symbol_cache_dir(symbol_upper) / "13f.json"
    if merged.exists():
        merged.unlink()
        logger.info("Deleted merged cache for %s (quarter snapshots preserved)", symbol_upper)
```

**Step 7: Write integration test for EFTS cache hit**

Add to the existing test file:

```python
class TestEftsSnapshotIntegration:
    """Tests that _fetch_from_efts uses per-quarter cache."""

    def test_efts_returns_cached_snapshot_without_http(
        self, tmp_path: Path
    ) -> None:
        """When a quarter snapshot exists, no HTTP requests are made."""
        client = _make_client(tmp_path)
        snap = OwnershipSnapshot(
            quarter_end="2012-12-31",
            symbol="GME",
            total_institutional_shares=45_000_000,
            num_institutions=312,
            top_10_concentration=0.35,
            holdings=(
                InstitutionalHolding(
                    filing_date="2012-12-31",
                    manager_name="Fidelity",
                    manager_cik="12345",
                    shares=5_000_000,
                    value_usd=50_000,
                    share_class="COM",
                ),
            ),
        )
        client._save_quarter_snapshot("GME", 2012, 4, snap)

        # Patch HTTP to ensure no requests are made
        with patch.object(client, "_fetch_efts_page") as mock_efts:
            result = client._fetch_from_efts("GME", "36467W109", 2012, 4, "2012-12-31")
            mock_efts.assert_not_called()

        assert result is not None
        assert result.total_institutional_shares == 45_000_000
```

**Step 8: Run full test suite**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py -v`
Expected: ALL PASS.

**Step 9: Commit**

```bash
git add src/stockdownloader/data/sec_ownership_client.py tests/data/test_sec_ownership_client.py
git commit -m "feat: add per-quarter EFTS snapshot caching for ownership data"
```

---

### Task 3: FinraShortInterestClient — Per-Symbol Subdir

**Files:**
- Modify: `src/stockdownloader/data/finra_short_interest_client.py:326-377`
- Test: `tests/data/test_finra_short_interest_client.py`

**Step 1: Write failing tests**

Add a new test class `TestPerSymbolCacheDir`:

```python
class TestSIPerSymbolCacheDir:
    """Tests for per-symbol subdirectory cache layout."""

    def test_save_creates_file_in_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_si_client(tmp_path)
        client._save_cache("AAPL", [
            ShortInterestRecord(
                settlement_date="2024-01-15",
                symbol="AAPL",
                short_interest=10000,
                avg_daily_volume=50000,
                days_to_cover=0.2,
                short_interest_pct=1.5,
            ),
        ])
        assert (client._cache_dir / "AAPL" / "si.json").exists()
        assert not (client._cache_dir / "AAPL_si.json").exists()

    def test_legacy_migration_on_load(self, tmp_path: Path) -> None:
        client = _make_si_client(tmp_path)
        legacy = client._cache_dir / "GME_si.json"
        legacy.write_text(json.dumps([{
            "settlement_date": "2024-01-15",
            "symbol": "GME",
            "short_interest": 10000,
            "avg_daily_volume": 50000,
            "days_to_cover": 0.2,
            "short_interest_pct": 1.5,
        }]), encoding="utf-8")
        loaded = client._load_cache("GME")
        assert loaded is not None
        assert not legacy.exists()
        assert (client._cache_dir / "GME" / "si.json").exists()
```

Note: You may need to add a `_make_si_client` helper or adapt the existing
test factory. Look for the existing factory function at the top of the test
file and follow its pattern.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/data/test_finra_short_interest_client.py::TestSIPerSymbolCacheDir -v`
Expected: FAIL.

**Step 3: Implement per-symbol cache in `finra_short_interest_client.py`**

Add `_symbol_cache_dir` method to the class:

```python
def _symbol_cache_dir(self, symbol: str) -> Path:
    """Return per-symbol cache subdirectory, creating it if needed."""
    d = self._cache_dir / symbol.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d
```

Update `_load_cache` (line 326):

```python
def _load_cache(self, symbol: str) -> list[ShortInterestRecord] | None:
    """Load cached short interest records for *symbol*."""
    sym_dir = self._symbol_cache_dir(symbol)
    cache_file = sym_dir / "si.json"

    # Legacy migration
    if not cache_file.exists():
        legacy = self._cache_dir / f"{symbol}_si.json"
        if legacy.exists():
            legacy.rename(cache_file)
            logger.info("Migrated %s → %s", legacy, cache_file)

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        # ... rest unchanged ...
```

Update `_save_cache` (line 354):

```python
def _save_cache(
    self,
    symbol: str,
    records: list[ShortInterestRecord],
) -> None:
    """Persist short interest records to JSON cache."""
    cache_file = self._symbol_cache_dir(symbol) / "si.json"
    # ... rest unchanged ...
```

**Step 4: Update existing `test_cache_file_naming` test**

The existing test at line 449 asserts `(client._cache_dir / "AAPL_si.json").exists()`.
Update to `(client._cache_dir / "AAPL" / "si.json").exists()`.

**Step 5: Run all tests**

Run: `python3 -m pytest tests/data/test_finra_short_interest_client.py -v`
Expected: ALL PASS.

**Step 6: Commit**

```bash
git add src/stockdownloader/data/finra_short_interest_client.py tests/data/test_finra_short_interest_client.py
git commit -m "refactor: move short interest cache to per-symbol subdirs"
```

---

### Task 4: FinraDarkPoolClient — Per-Symbol Subdir

**Files:**
- Modify: `src/stockdownloader/data/finra_dark_pool_client.py:352-407`
- Test: `tests/data/test_finra_dark_pool_client.py`

**Step 1: Write failing tests**

Add a test class `TestDPPerSymbolCacheDir` following the same pattern as Task 3:

```python
class TestDPPerSymbolCacheDir:
    """Tests for per-symbol subdirectory cache layout."""

    def test_save_creates_file_in_symbol_subdir(self, tmp_path: Path) -> None:
        client = _make_dp_client(tmp_path)
        client._save_cache("AAPL", [
            DarkPoolRecord(
                week_ending="2024-01-19",
                symbol="AAPL",
                total_weekly_volume=100000,
                ats_volume=60000,
                otc_volume=40000,
                ats_pct=60.0,
            ),
        ])
        assert (client._cache_dir / "AAPL" / "dp.json").exists()
        assert not (client._cache_dir / "AAPL_dp.json").exists()

    def test_legacy_migration_on_load(self, tmp_path: Path) -> None:
        client = _make_dp_client(tmp_path)
        legacy = client._cache_dir / "GME_dp.json"
        legacy.write_text(json.dumps([{
            "week_ending": "2024-01-19",
            "symbol": "GME",
            "total_weekly_volume": 100000,
            "ats_volume": 60000,
            "otc_volume": 40000,
            "ats_pct": 60.0,
        }]), encoding="utf-8")
        loaded = client._load_cache("GME")
        assert loaded is not None
        assert not legacy.exists()
        assert (client._cache_dir / "GME" / "dp.json").exists()
```

Note: Adapt `_make_dp_client` to match existing test factory in that file.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/data/test_finra_dark_pool_client.py::TestDPPerSymbolCacheDir -v`
Expected: FAIL.

**Step 3: Implement per-symbol cache in `finra_dark_pool_client.py`**

Same pattern as Task 3:
- Add `_symbol_cache_dir` method
- Update `_load_cache` at line 352 with legacy migration
- Update `_save_cache` at line 382 to use new path

**Step 4: Update existing `test_cache_file_naming`**

The test at line 582 asserts `(client._cache_dir / "AAPL_dp.json").exists()`.
Update to `(client._cache_dir / "AAPL" / "dp.json").exists()`.

**Step 5: Run all tests**

Run: `python3 -m pytest tests/data/test_finra_dark_pool_client.py -v`
Expected: ALL PASS.

**Step 6: Commit**

```bash
git add src/stockdownloader/data/finra_dark_pool_client.py tests/data/test_finra_dark_pool_client.py
git commit -m "refactor: move dark pool cache to per-symbol subdirs"
```

---

### Task 5: FinraShortVolumeClient — Per-Symbol Subdir

**Files:**
- Modify: `src/stockdownloader/data/finra_short_volume_client.py:369-396`
- Test: `tests/data/test_finra_short_volume_client.py` (may need to create if it doesn't exist)

**Step 1: Write failing tests**

Follow the same pattern as Task 3, adapting field names for `ShortVolumeRecord`.
The cache file suffix is `sv.json` and legacy name is `{SYMBOL}_sv.json`.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/data/test_finra_short_volume_client.py -v` (or inline if no separate test file)
Expected: FAIL.

**Step 3: Implement per-symbol cache**

Same pattern:
- Add `_symbol_cache_dir` method
- Update `_load_cache` at line 369 with legacy migration
- Update `_save_cache` at line 384 to use new path

**Step 4: Run all tests**

Run: `python3 -m pytest tests/ -x -q`
Expected: ALL PASS.

**Step 5: Commit**

```bash
git add src/stockdownloader/data/finra_short_volume_client.py
# Include test file if created/modified
git commit -m "refactor: move short volume cache to per-symbol subdirs"
```

---

### Task 6: RegShoThresholdClient — Per-Symbol Subdir

**Files:**
- Modify: `src/stockdownloader/data/regsho_threshold_client.py:356-385`
- Test: Create tests if they don't exist

**Step 1: Write failing tests**

Same pattern. Cache file suffix is `threshold.json`, legacy name is
`{SYMBOL}_threshold.json`.

**Step 2: Run tests to verify they fail**

Expected: FAIL.

**Step 3: Implement per-symbol cache**

Same pattern:
- Add `_symbol_cache_dir` method
- Update `_load_cache` at line 356 with legacy migration
- Update `_save_cache` at line 371 to use new path

**Step 4: Run all tests**

Run: `python3 -m pytest tests/ -x -q`
Expected: ALL PASS.

**Step 5: Commit**

```bash
git add src/stockdownloader/data/regsho_threshold_client.py
# Include test file if created/modified
git commit -m "refactor: move Reg SHO threshold cache to per-symbol subdirs"
```

---

### Task 7: Full Suite Verification

**Step 1: Run complete test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: ALL PASS with 0 failures.

**Step 2: Verify no old-path references remain in source**

Run: `grep -rn '_si\.json\|_dp\.json\|_sv\.json\|_threshold\.json\|_13f\.json\|_ownership\.json' src/`
Expected: No matches (all old flat-path patterns replaced).  Legacy migration
code in `_load_cache` methods will reference old filenames as string literals
— those are expected.

**Step 3: Commit any cleanup**

If any stale references found, fix and commit.
