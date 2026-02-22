# Data Consistency, Directory Cleanup & Symbol Variant Handling

## Problem

Three issues with the current data layer:

1. **Format inconsistency**: Some flat tabular data is stored as JSON (verbose, key-per-row overhead), some as CSV. No clear rule governs which format is used where.
2. **Progress files mixed with data**: Files like `occ_options_progress.json` and `regsho_nyse_progress.json` sit alongside real data files, cluttering the symbol directory and confusing what's persistent data vs download-tracking state.
3. **No symbol variant handling**: GME and GMEWS are stored as unrelated symbols. No infrastructure links warrants (GMEWS), when-issued (GME1), or other variants back to their parent.

## Design Rules

### Rule 1: CSV for flat data, JSON only for nested

- **CSV**: Any dataset where every record is a flat dict of scalars (strings, numbers, booleans). Header row defines the schema. One row per record.
- **JSON**: Only when records contain nested arrays or objects that cannot be flattened without information loss.

### Rule 2: Progress/cache tracking in hidden `.progress/` directory

- Progress files track download state (dates attempted, quarters cached). They are not analysis data.
- Move all progress files to `data/{SYMBOL}/.progress/`.
- The `.progress/` directory is `.gitignore`-able — it can be deleted and rebuilt by re-running clients.

### Rule 3: Variants linked via `SymbolInfo`

- Extend the symbol registry to express parent-child relationships between related tickers.
- Each variant keeps its own `data/{SYMBOL}/` directory — no nesting.
- A `get_variants()` function enables aggregation across related symbols.

---

## File Format Changes

### Flat JSON → CSV (10 files across 7 clients)

| Client | Current file | New file | Fields |
|--------|-------------|----------|--------|
| `FinraShortVolumeClient` | `short_volume.json` | `short_volume.csv` | date, symbol, short_volume, total_volume, short_exempt_volume, short_volume_ratio |
| `FinraShortInterestClient` | `short_interest.json` | `short_interest.csv` | settlement_date, symbol, short_interest, avg_daily_volume, days_to_cover, short_interest_pct |
| `OccOptionsClient` | `occ_open_interest.json` | `occ_open_interest.csv` | date, symbol, exchange, volume, exercised, open_interest, product_kind, expiration |
| `FinraDarkPoolClient` | `dark_pool.json` | `dark_pool.csv` | week_ending, symbol, total_weekly_volume, ats_volume, otc_volume, ats_pct |
| `RegShoThresholdClient` | `regsho_threshold.json` | `regsho_threshold.csv` | date, symbol, market, threshold_shares, consecutive_days |
| `SecInsiderClient` | `insider_transactions.json` | `insider_transactions.csv` | filing_date, transaction_date, owner_name, owner_cik, owner_title, is_director, is_officer, is_ten_pct_owner, transaction_code, shares, price_per_share, shares_owned_after, direct_or_indirect |
| `SecInsiderClient` | `beneficial_owners.json` | `beneficial_owners.csv` | filing_date, owner_name, owner_cik, form_type, shares_beneficially_owned, percent_of_class, sole_voting_power, shared_voting_power, sole_dispositive_power, shared_dispositive_power, filing_url |
| `SecFtdClient` (caller) | `ftd_data.json` | `ftd_data.csv` | settlement_date, symbol, cusip, quantity, description, price |
| `SecEdgarClient` (caller) | `sec_filings.json` | `sec_filings.csv` | accession_number, filing_date, report_date, form, primary_document, description, filing_url |
| GMEWS price files | `daily_prices.json` | `daily_prices.csv` | date, open, high, low, close, volume, vwap, transactions |

### Stay JSON (genuinely nested)

| File | Reason |
|------|--------|
| `ownership_13f.json` | `holdings[]` array per quarter (one-to-many) |
| `options_chain.json` | Nested calls[]/puts[] with greeks per strike per expiration |
| `report_data.json` | Analysis output with date-keyed nested metrics |
| `fundamentals.json` | Single object (not tabular), 688 bytes |

### Already CSV (no change needed)

| File | Notes |
|------|-------|
| `daily_bars.csv` | ✅ correct format |
| `5m_bars.csv` | ✅ correct format |
| `holistic_aligned.csv` | ✅ analysis output |

---

## Progress File Relocation

### Before

```
data/GME/
├── occ_options_progress.json      ← mixed with data
├── regsho_nyse_progress.json      ← mixed with data
├── insider/2003Q1.json ... 2026Q1.json  ← looks like data, is cache
├── ownership/2005Q3.json ... 2025Q4.json  ← looks like data, is cache
├── short_volume.json
├── ftd_data.json
└── ...
```

### After

```
data/GME/
├── .progress/
│   ├── occ_options.json           ← download tracking
│   ├── regsho_nyse.json           ← download tracking
│   ├── insider/2003Q1.json ... 2026Q1.json  ← per-quarter cache
│   └── ownership/2005Q3.json ... 2025Q4.json  ← per-quarter cache
├── short_volume.csv
├── ftd_data.csv
└── ...
```

The `.progress/` directory should be added to `.gitignore`.

---

## CSV I/O Implementation

### BaseDataClient additions

Add two helper methods to `BaseDataClient` that all clients can use:

```python
def _save_csv(self, symbol: str, filename: str, records: list, fieldnames: list[str]) -> None:
    """Write a list of dataclass instances to CSV."""
    path = self._symbol_dir(symbol) / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

def _load_csv(self, symbol: str, filename: str, record_cls: type) -> list | None:
    """Read a CSV file back into a list of dataclass instances."""
    path = self._symbol_dir(symbol) / filename
    if not path.exists():
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [record_cls(**_coerce_row(row, record_cls)) for row in reader]
    except (csv.Error, ValueError, KeyError):
        return None
```

The `_coerce_row()` helper inspects the dataclass field types and converts CSV string values to `int`, `float`, `bool` as needed (CSV reads everything as strings).

### Progress file helpers

```python
def _progress_dir(self, symbol: str) -> Path:
    """Return hidden .progress directory for a symbol."""
    d = self._symbol_dir(symbol) / ".progress"
    d.mkdir(parents=True, exist_ok=True)
    return d
```

---

## Symbol Variant Handling

### SymbolInfo extension

```python
@dataclass(frozen=True, slots=True)
class SymbolInfo:
    symbol: str
    cusip: str
    ipo_date: date
    name: str
    exchange: str
    parent: str | None = None       # parent ticker (e.g., "GME" for GMEWS)
    security_type: str = "common"   # "common", "warrant", "when_issued", "preferred"
```

### New registry functions

```python
def get_variants(symbol: str) -> list[SymbolInfo]:
    """Return all variants of a symbol (excluding the symbol itself)."""
    return [info for info in SYMBOL_REGISTRY.values()
            if info.parent == symbol.upper()]

def get_family(symbol: str) -> list[SymbolInfo]:
    """Return the symbol + all its variants (the full family)."""
    info = get_symbol_info(symbol)
    if info is None:
        return []
    # If this is a variant, get the parent's family
    root = info.parent or info.symbol
    root_info = get_symbol_info(root)
    if root_info is None:
        return [info]
    return [root_info] + get_variants(root)
```

### GMEWS registration

```python
_register(
    SymbolInfo("GMEWS", "36467W117", date(2025, 10, 7),
               "GameStop Corp Warrants", "NYSE",
               parent="GME", security_type="warrant"),
)
```

---

## Migration Strategy

Each client update follows this pattern:

1. Update `_save_cache()` to write CSV via `_save_csv()`
2. Update `_load_cache()` to read CSV via `_load_csv()`, with JSON fallback for migration
3. Move progress methods to use `_progress_dir()`
4. On first load: if `.csv` doesn't exist but `.json` does, read the JSON, write CSV, done. Old JSON can be cleaned up manually or left as-is.

This means existing data is auto-migrated on next client run — no separate migration script needed.

---

## Files Changed

| File | Action |
|------|--------|
| `src/stockdownloader/data/base_data_client.py` | Add `_save_csv()`, `_load_csv()`, `_progress_dir()` |
| `src/stockdownloader/data/finra_short_volume_client.py` | CSV save/load, progress dir |
| `src/stockdownloader/data/finra_short_interest_client.py` | CSV save/load |
| `src/stockdownloader/data/occ_options_client.py` | CSV save/load, progress to `.progress/` |
| `src/stockdownloader/data/finra_dark_pool_client.py` | CSV save/load |
| `src/stockdownloader/data/regsho_threshold_client.py` | CSV save/load, progress to `.progress/` |
| `src/stockdownloader/data/sec_insider_client.py` | CSV for txns + beneficial owners, quarterly cache to `.progress/` |
| `src/stockdownloader/data/sec_ownership_client.py` | Quarterly cache to `.progress/` (13f stays JSON — nested) |
| `src/stockdownloader/data/sec_ftd_client.py` | Caller-side persistence needs updating if any script writes ftd_data.json |
| `src/stockdownloader/model/symbol_info.py` | Add `parent`, `security_type`, `get_variants()`, `get_family()`, GMEWS entry |
| `scripts/gme_options_analysis.py` | Read CSV instead of JSON |
| `scripts/regsho_targeted_backfill.py` | Read CSV, progress paths |
| `scripts/archive/gme_holistic_report.py` | Read CSV for all flat data |
| `scripts/archive/gme_comprehensive_report.py` | Read CSV |
| `scripts/archive/gme_squeeze_predictor.py` | Read CSV |
| `.gitignore` | Add `data/**/.progress/` |
| Tests for all modified clients | Update expected file paths and formats |
