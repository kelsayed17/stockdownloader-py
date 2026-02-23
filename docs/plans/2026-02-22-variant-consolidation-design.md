# Variant Consolidation, Report Relocation & Credential Wiring

**Date:** 2026-02-22
**Branch:** `claude/vigorous-easley`

## Problem

1. **Stale variant folders** — `data/GME WS/`, `data/GME-WS/`, `data/GME.WS/` are empty
   directories from earlier experiments. They should be deleted.

2. **Missing GMEWS data** — The warrant trades under several ticker aliases across
   different data sources (`GMEWS`, `GME WS`, `GME-WS`, `GME.WS`, `GME/WS`,
   `GME+WS`). The download script currently only queries the canonical `GMEWS`
   ticker, so FINRA short volume and dark pool returned zero records. Each alias
   must be tried and results stitched together under `data/GMEWS/`.

3. **Reports mixed with raw data** — Analysis files (`*.md`, `*.txt`) and derived
   outputs (`holistic_aligned.csv`, `report_data.json`) live alongside raw data
   CSVs in `data/GME/`. They belong in `reports/GME/`.

4. **FINRA credentials missing** — The previous download fell back to CDN scraping
   because FINRA OAuth creds were not configured. Now available.

## Design

### 1. SymbolInfo aliases

Add an `aliases` field to `SymbolInfo` — a tuple of alternate ticker strings that
external APIs may use for the same security.

```python
@dataclass(frozen=True, slots=True)
class SymbolInfo:
    # ... existing fields ...
    aliases: tuple[str, ...] = ()
```

GMEWS registration:
```python
SymbolInfo(
    symbol="GMEWS",
    aliases=("GME WS", "GME-WS", "GME.WS", "GME/WS", "GME+WS"),
    ...
)
```

Add helper `get_all_tickers(symbol) -> list[str]` that returns
`[canonical, *aliases]`.

### 2. download_all.py updates

- Set `FINRA_CLIENT_ID` and `FINRA_CLIENT_SECRET` environment variables.
- For each data source, try canonical ticker first. If zero records returned,
  iterate through aliases. Merge/deduplicate results from all aliases.
- All results save under the canonical folder (`data/GMEWS/`).

### 3. Report relocation

Move from `data/GME/` to `reports/GME/`:
- `analysis_short_interest_theory.md`
- `comprehensive_report.md`
- `comprehensive_report.txt`
- `holistic_aligned.csv`
- `report_data.json`

Update `gme_holistic_report.py` export paths to write to `reports/GME/`.

### 4. Cleanup

- Delete empty directories: `data/GME WS/`, `data/GME-WS/`, `data/GME.WS/`

## Files Modified

- `src/stockdownloader/model/symbol_info.py` — add `aliases` field, `get_all_tickers()`
- `scripts/download_all.py` — FINRA creds, alias-fallback logic, dedup merging
- `scripts/archive/gme_holistic_report.py` — export to `reports/GME/`
- `tests/model/test_symbol_info.py` — test aliases and `get_all_tickers()`

## Files Created

- `reports/GME/` — new directory for analysis outputs

## Files Deleted

- `data/GME WS/`, `data/GME-WS/`, `data/GME.WS/` — empty stale dirs
- `data/GME/analysis_short_interest_theory.md` (moved)
- `data/GME/comprehensive_report.md` (moved)
- `data/GME/comprehensive_report.txt` (moved)
- `data/GME/holistic_aligned.csv` (moved)
- `data/GME/report_data.json` (moved)
