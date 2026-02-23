# Variant Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up stale data directories, add ticker alias support so all GMEWS variant tickers are queried and stitched together, wire FINRA credentials, relocate reports out of data directories, and re-run downloads.

**Architecture:** Add an `aliases` tuple to `SymbolInfo` so the registry knows all ticker formats for a symbol. Update `download_all.py` with a `fetch_with_aliases()` helper that tries each alias when the canonical ticker returns no data, deduplicates, and saves under the canonical folder. Move report files to `reports/GME/`.

**Tech Stack:** Python dataclasses, csv, pathlib, existing FINRA/SEC/Polygon clients.

---

### Task 1: Add `aliases` field and `get_all_tickers()` to SymbolInfo

**Files:**
- Modify: `src/stockdownloader/model/symbol_info.py:26-57` (SymbolInfo dataclass)
- Modify: `src/stockdownloader/model/symbol_info.py:162-172` (GMEWS registration)
- Modify: `src/stockdownloader/model/symbol_info.py` (add `get_all_tickers` function)
- Modify: `src/stockdownloader/model/__init__.py:52-58` (add export)
- Test: `tests/model/test_symbol_info.py`

**Step 1: Write the failing tests**

Add to `tests/model/test_symbol_info.py`:

```python
from stockdownloader.model.symbol_info import get_all_tickers

class TestAliases:
    def test_default_aliases_empty(self):
        info = get_symbol_info("GME")
        assert info is not None
        assert info.aliases == ()

    def test_gmews_has_aliases(self):
        info = get_symbol_info("GMEWS")
        assert info is not None
        assert "GME WS" in info.aliases
        assert "GME-WS" in info.aliases
        assert "GME.WS" in info.aliases

    def test_get_all_tickers_parent(self):
        tickers = get_all_tickers("GME")
        assert tickers == ["GME"]

    def test_get_all_tickers_variant(self):
        tickers = get_all_tickers("GMEWS")
        assert tickers[0] == "GMEWS"
        assert "GME WS" in tickers
        assert "GME-WS" in tickers
        assert "GME.WS" in tickers
        assert "GME/WS" in tickers
        assert "GME+WS" in tickers

    def test_get_all_tickers_unknown(self):
        tickers = get_all_tickers("ZZZZZZ")
        assert tickers == ["ZZZZZZ"]

    def test_aliases_frozen(self):
        info = get_symbol_info("GMEWS")
        assert info is not None
        assert isinstance(info.aliases, tuple)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/model/test_symbol_info.py::TestAliases -v`
Expected: FAIL — `ImportError: cannot import name 'get_all_tickers'`

**Step 3: Implement aliases field and get_all_tickers**

In `src/stockdownloader/model/symbol_info.py`:

1. Add `aliases: tuple[str, ...] = ()` field to SymbolInfo after `security_type`.
2. Update GMEWS registration to include aliases:
   ```python
   SymbolInfo(
       symbol="GMEWS",
       cusip="36467W117",
       ipo_date=date(2025, 10, 7),
       name="GameStop Corp Warrants",
       exchange="NYSE",
       parent="GME",
       security_type="warrant",
       aliases=("GME WS", "GME-WS", "GME.WS", "GME/WS", "GME+WS"),
   )
   ```
3. Add function after `get_family`:
   ```python
   def get_all_tickers(symbol: str) -> list[str]:
       """Return canonical ticker plus all aliases for *symbol*.

       Unknown symbols return a single-element list with the symbol itself.
       """
       info = get_symbol_info(symbol)
       if info is None:
           return [symbol.upper()]
       return [info.symbol, *info.aliases]
   ```
4. Add `get_all_tickers` to the import in `model/__init__.py` and `__all__`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/model/test_symbol_info.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/model/symbol_info.py src/stockdownloader/model/__init__.py tests/model/test_symbol_info.py
git commit -m "feat: add ticker aliases to SymbolInfo and get_all_tickers() helper"
```

---

### Task 2: Update download_all.py with alias-fallback + FINRA credentials

**Files:**
- Modify: `scripts/download_all.py`

**Step 1: Add FINRA credentials and alias helper**

At the top of `download_all.py`, after `POLYGON_API_KEY`:
```python
FINRA_CLIENT_ID = "a876a67c64314aae9afd"
FINRA_CLIENT_SECRET = "GA&NzO0eZFdn@TB"
```

Add a generic alias-fallback helper:
```python
from stockdownloader.model.symbol_info import get_all_tickers

def _fetch_with_aliases(
    symbol: str,
    fetch_fn,
    label: str,
    dedup_key=None,
) -> list:
    """Try canonical ticker first, then aliases if zero results.

    Args:
        symbol: Canonical ticker (e.g. "GMEWS").
        fetch_fn: Callable(ticker) -> list of records.
        label: Human label for logging.
        dedup_key: Optional callable(record) -> hashable for deduplication.
            If None, no dedup is performed — results are concatenated.

    Returns:
        Merged, deduplicated list of records from all aliases.
    """
    tickers = get_all_tickers(symbol)
    all_records = []
    seen = set()
    for ticker in tickers:
        try:
            records = fetch_fn(ticker)
            if records:
                logger.info("  %s: %d records from ticker %r", label, len(records), ticker)
                for r in records:
                    if dedup_key is not None:
                        key = dedup_key(r)
                        if key in seen:
                            continue
                        seen.add(key)
                    all_records.append(r)
        except Exception as e:
            logger.warning("  %s: ticker %r failed: %s", label, ticker, e)
    return all_records
```

**Step 2: Set env vars in main()**

At the start of `main()`:
```python
os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY
os.environ["FINRA_CLIENT_ID"] = FINRA_CLIENT_ID
os.environ["FINRA_CLIENT_SECRET"] = FINRA_CLIENT_SECRET
```

**Step 3: Update run_finra to use alias fallback**

Replace the per-symbol loops in `run_finra` with `_fetch_with_aliases` calls. Each FINRA data type needs a dedup key:

- Short volume: `lambda r: (r.date, r.symbol)` — date field from ShortVolumeRecord
- Short interest: `lambda r: r.settlement_date` — settlement_date from ShortInterestRecord
- Dark pool: `lambda r: r.week_ending` — week_ending from DarkPoolRecord

After fetching with aliases, save under the canonical symbol. The FINRA clients auto-save to `_symbol_dir(ticker)` which would create dirs for each alias — so instead, call the client with the canonical symbol for saving but try fetching with each alias ticker. The simplest approach: create the clients once, then for each symbol call `_fetch_with_aliases` which calls the client's fetch method with each alias ticker. Then call `client._save_cache(canonical_symbol, merged_records)` to persist under the right dir.

Actually the FINRA clients already save internally. The cleaner approach: fetch with each alias, then move/merge into the canonical folder. But even simpler: just call each client's fetch method with each alias, collect results, save once using `_save_records_csv`. Let me check — FINRA clients save automatically and return records. We need to:
1. Fetch with each alias ticker — client saves under that alias dir
2. After all aliases tried, merge into canonical dir

Simplest: skip the client's auto-save for non-canonical tickers. Instead, create a wrapper:

```python
def run_finra(symbols: list[str]) -> None:
    """FINRA clients: short volume, short interest, dark pool."""
    from stockdownloader.data.finra_short_volume_client import FinraShortVolumeClient
    from stockdownloader.data.finra_short_interest_client import FinraShortInterestClient
    from stockdownloader.data.finra_dark_pool_client import FinraDarkPoolClient

    sv = FinraShortVolumeClient(data_dir=DATA_DIR)
    si = FinraShortInterestClient(data_dir=DATA_DIR)
    dp = FinraDarkPoolClient(data_dir=DATA_DIR)

    for sym in symbols:
        logger.info("=== FINRA Short Volume: %s ===", sym)
        records = _fetch_with_aliases(
            sym, sv.fetch_short_volume, "ShortVol",
            dedup_key=lambda r: (r.date,),
        )
        if records:
            sv._save_csv(sym, "short_volume.csv", records)
        logger.info("  -> %d short volume records", len(records))

        logger.info("=== FINRA Short Interest: %s ===", sym)
        records = _fetch_with_aliases(
            sym, si.fetch_short_interest, "ShortInt",
            dedup_key=lambda r: (r.settlement_date,),
        )
        if records:
            si._save_csv(sym, "short_interest.csv", records)
        logger.info("  -> %d short interest records", len(records))

        logger.info("=== FINRA Dark Pool: %s ===", sym)
        records = _fetch_with_aliases(
            sym, dp.fetch_dark_pool_volume, "DarkPool",
            dedup_key=lambda r: (r.week_ending,),
        )
        if records:
            dp._save_csv(sym, "dark_pool.csv", records)
        logger.info("  -> %d dark pool records", len(records))
```

Wait — the FINRA clients use `_save_csv` from BaseDataClient. Let me check what's available. Actually looking at the code, the clients auto-save during fetch. The simplest approach: just call each client with each alias ticker, then at the end consolidate by reading all CSVs and re-saving under canonical. But even simpler — for the _fetch_with_aliases approach, just re-save the final merged list using `_save_records_csv()` (the helper already in download_all.py).

**Final run_finra approach:**

```python
def run_finra(symbols: list[str]) -> None:
    from stockdownloader.data.finra_short_volume_client import FinraShortVolumeClient
    from stockdownloader.data.finra_short_interest_client import FinraShortInterestClient
    from stockdownloader.data.finra_dark_pool_client import FinraDarkPoolClient

    sv = FinraShortVolumeClient(data_dir=DATA_DIR)
    si = FinraShortInterestClient(data_dir=DATA_DIR)
    dp = FinraDarkPoolClient(data_dir=DATA_DIR)

    for sym in symbols:
        logger.info("=== FINRA Short Volume: %s ===", sym)
        records = _fetch_with_aliases(
            sym, sv.fetch_short_volume, "ShortVol",
            dedup_key=lambda r: r.date,
        )
        if records:
            _save_records_csv(sym, "short_volume.csv", records)
        logger.info("  -> %d short volume records", len(records))

        logger.info("=== FINRA Short Interest: %s ===", sym)
        records = _fetch_with_aliases(
            sym, si.fetch_short_interest, "ShortInt",
            dedup_key=lambda r: r.settlement_date,
        )
        if records:
            _save_records_csv(sym, "short_interest.csv", records)
        logger.info("  -> %d short interest records", len(records))

        logger.info("=== FINRA Dark Pool: %s ===", sym)
        records = _fetch_with_aliases(
            sym, dp.fetch_dark_pool_volume, "DarkPool",
            dedup_key=lambda r: r.week_ending,
        )
        if records:
            _save_records_csv(sym, "dark_pool.csv", records)
        logger.info("  -> %d dark pool records", len(records))
```

**Step 4: Update run_sec, run_polygon to also use alias fallback for GMEWS**

For SEC FTD:
```python
records = _fetch_with_aliases(
    sym, ftd.fetch_ftd_data, "FTD",
    dedup_key=lambda r: (r.settlement_date, r.quantity),
)
```

For Polygon daily/5m, the fetcher saves automatically, so just try aliases sequentially and stop at the first one that returns data (price data should be identical across aliases).

**Step 5: Verify script runs without errors (dry-run import check)**

Run: `python3 -c "import scripts.download_all"` (or just check syntax)
Expected: No import errors

**Step 6: Commit**

```bash
git add scripts/download_all.py
git commit -m "feat: add FINRA creds and alias-fallback to download_all.py"
```

---

### Task 3: Delete stale directories and move reports

**Files:**
- Delete: `data/GME WS/`, `data/GME-WS/`, `data/GME.WS/`
- Move: `data/GME/*.md`, `data/GME/*.txt` → `reports/GME/`
- Move: `data/GME/holistic_aligned.csv` → `reports/GME/`
- Move: `data/GME/report_data.json` → `reports/GME/`
- Modify: `scripts/archive/gme_holistic_report.py:1249-1252` (export paths)

**Step 1: Delete empty stale directories**

```bash
rm -rf "data/GME WS" "data/GME-WS" "data/GME.WS"
```

Verify: `ls data/` should show only GME, GMEWS, SPY, cache.

**Step 2: Create reports directory and move files**

```bash
mkdir -p reports/GME
mv data/GME/analysis_short_interest_theory.md reports/GME/
mv data/GME/comprehensive_report.md reports/GME/
mv data/GME/comprehensive_report.txt reports/GME/
mv data/GME/holistic_aligned.csv reports/GME/
mv data/GME/report_data.json reports/GME/
```

**Step 3: Update holistic report export paths**

In `scripts/archive/gme_holistic_report.py`, change lines ~1249-1252:

```python
    # 15. Export
    reports_dir = PROJECT_ROOT / "reports" / "GME"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "holistic_aligned.csv"
    json_path = reports_dir / "report_data.json"
    export_csv(aligned, csv_path)
    export_tradingview_json(aligned, si_by_date, dp_by_week, own_by_quarter, json_path)
```

**Step 4: Verify holistic report still loads (syntax check)**

Run: `python3 -c "from pathlib import Path; exec(open('scripts/archive/gme_holistic_report.py').read().split('if __name__')[0])"`
Expected: No syntax errors

**Step 5: Commit**

```bash
git add -A  # stages deletions + moves + modified report script
git commit -m "chore: delete stale variant dirs, move reports to reports/GME/"
```

---

### Task 4: Run full download with aliases and verify GMEWS data

**Files:**
- Run: `scripts/download_all.py`

**Step 1: Run the download script**

```bash
python3 scripts/download_all.py 2>&1 | tee download_output.log
```

This will take ~1-2 hours (RegSHO NYSE is slow). Expected improvements over previous run:
- GMEWS short volume: should find data via `GME WS` or `GME-WS` aliases if FINRA has it
- GMEWS dark pool: should find data via aliases if available
- FINRA API auth: should succeed now with credentials (faster than CDN scraping)

**Step 2: Verify GMEWS data directory**

```bash
ls -la data/GMEWS/
```

Expected: ftd_data.csv, short_interest.csv, and possibly short_volume.csv and dark_pool.csv if aliases returned data.

**Step 3: Verify no stale dirs recreated**

```bash
ls data/ | sort
```

Expected: `GME`, `GMEWS`, `SPY`, `cache` — no `GME WS`, `GME-WS`, `GME.WS`.

**Step 4: Verify reports directory**

```bash
ls reports/GME/
```

Expected: analysis_short_interest_theory.md, comprehensive_report.md, comprehensive_report.txt, holistic_aligned.csv, report_data.json.

---

### Task 5: Run tests to verify no regressions

**Step 1: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: 3192+ passed, 0 failed

**Step 2: Commit download results (if any new data files)**

Only commit if there are meaningful new CSV data files for GMEWS. Data files are gitignored so this may be a no-op.

```bash
git status
```
