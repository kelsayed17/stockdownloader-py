# Cache Reorganization Design

## Problem

Two issues identified during live data download:

1. **No intermediate EFTS caching** — The ownership client's EFTS path makes
   300-400 HTTP requests per quarter for pre-2013 data.  If the process is
   interrupted, all work for in-flight quarters is lost and must be re-fetched
   from scratch.

2. **Flat per-symbol files** — Result JSON files (e.g. `GME_13f.json`,
   `GME_si.json`) sit loose in each client's cache directory.  With multiple
   symbols this creates unorganised clutter.

## Approach

**Per-symbol subdirectories + per-quarter EFTS snapshots.**

Move every per-symbol result file into a `{SYMBOL}/` subdirectory.  Add
per-quarter JSON snapshot caching to the ownership EFTS path so each quarter's
result survives independently.

Shared resources (FTD ZIPs, bulk 13F ZIPs) stay in their current flat layout
because they contain data for all symbols — duplicating them per-symbol would
waste disk space.

## New Directory Layout

```
data/cache/
├── ftd/                           # UNCHANGED — ZIPs shared across all symbols
│   ├── cnsp_sec_fails_2004q1.zip
│   ├── cnsfails202401a.zip
│   └── ...
├── ownership/
│   ├── bulk_13f/                  # UNCHANGED — ZIPs shared across all symbols
│   │   └── 2024Q1_form13f.zip
│   ├── GME/                       # NEW per-symbol subdir
│   │   ├── 13f.json               # merged result (was GME_13f.json)
│   │   ├── 2012Q4.json            # per-quarter EFTS snapshot
│   │   └── 2003Q1.json
│   └── AAPL/
│       └── 13f.json
├── short_interest/
│   └── GME/
│       └── si.json                # was GME_si.json
├── dark_pool/
│   └── GME/
│       └── dp.json                # was GME_dp.json
├── short_volume/
│   └── GME/
│       └── sv.json                # was GME_sv.json
└── regsho/
    └── GME/
        └── threshold.json         # was GME_threshold.json
```

## Per-Quarter EFTS Snapshot Caching

### Flow

```
For each quarter (newest → oldest):
  1. Does {SYMBOL}/{YYYY}Q{Q}.json exist?
     YES → load from cache, skip HTTP requests
     NO  → query EFTS, parse filings, save snapshot, continue
```

### Snapshot format

Each file contains one serialised `OwnershipSnapshot`:

```json
{
  "quarter_end": "2012-12-31",
  "total_shares": 45000000,
  "total_value_usd": 580000000,
  "holder_count": 312,
  "top_holders": [...]
}
```

### force_refresh behaviour

`force_refresh=True` deletes the merged `13f.json` but **preserves**
per-quarter snapshots.  Historical filings are immutable — the quarterly data
never changes.  Only the merged result might need regeneration (e.g. after
adding new holders to the top-N list).

## Legacy Migration

Every client gets a `_symbol_cache_dir(symbol)` helper that:

1. Creates `{cache_dir}/{SYMBOL}/` if it doesn't exist.
2. On first load, if the old flat file exists and the new path doesn't,
   atomically moves the old file into the new directory.

```python
def _symbol_cache_dir(self, symbol: str) -> Path:
    d = self._cache_dir / symbol.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d
```

This makes the migration transparent — no manual file moves required.

## Per-Client Changes

| Client | Path change | New caching |
|--------|-------------|-------------|
| SecOwnershipClient | `{SYM}_13f.json` → `{SYM}/13f.json` | Per-quarter EFTS snapshots |
| FinraShortInterestClient | `{SYM}_si.json` → `{SYM}/si.json` | — |
| FinraDarkPoolClient | `{SYM}_dp.json` → `{SYM}/dp.json` | — |
| FinraShortVolumeClient | `{SYM}_sv.json` → `{SYM}/sv.json` | — |
| RegShoThresholdClient | `{SYM}_threshold.json` → `{SYM}/threshold.json` | — |
| SecFtdClient | No per-symbol JSON to move | — |
| YahooFinanceClient | No caching | — |

## What Does NOT Change

- Data models (OwnershipSnapshot, FtdRecord, etc.)
- ML pipeline and stage_data.py
- SymbolInfo registry
- Bulk ZIP file paths (flat, shared)
- Yahoo Finance client
- FTD ZIP caching (already incremental)
- Bulk 13F ZIP caching (already incremental)

## Deferred (Future Work)

FINRA clients (SI, DP, SV) and RegSho currently do full-refetch on every call.
Incremental fetch (check latest cached date, only request newer data) is a
natural follow-up but out of scope here — the user's primary pain point is the
ownership EFTS path losing work on interruption.
