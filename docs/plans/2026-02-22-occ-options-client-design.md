# OCC Options Open Interest Client — Design

## Goal

Build an `OccOptionsClient` that downloads daily per-symbol open interest
data from the OCC `cont-volume-download` endpoint. Primary use case:
analyze GME options open interest patterns to detect synthetic short
positions (married puts, extreme OTM put accumulation, total put OI
exceeding float).

## Data Source

**OCC `cont-volume-download`** endpoint:

```
https://marketdata.theocc.com/cont-volume-download?reportDate=YYYYMMDD&format=txt
```

- Free, no authentication
- Returns fixed-width text (~202K lines per day, all symbols/exchanges)
- Available: January 2021 to present (~1,250 trading days)
- Fields per record (52 chars, fixed-width):

| Position | Width | Field          | Example        |
|----------|-------|----------------|----------------|
| 0-5      | 6     | Symbol         | `GME   `       |
| 6-11     | 6     | Underlying     | `GME   `       |
| 12       | 1     | Exchange Code  | `A`            |
| 13-21    | 9     | Volume         | `000005227`    |
| 22-30    | 9     | Exercised      | `000000000`    |
| 31-39    | 9     | Open Interest  | `000005121`    |
| 40-43    | 4     | Product Kind   | `OSTK`         |
| 44-51    | 8     | Expiration     | `20260220`     |

Header line: `        H{MMDDYYYY}      {MMDDYYYY}`

## Data Model

New frozen dataclass in `regulatory_records.py`:

```python
@dataclass(frozen=True, slots=True)
class OccOpenInterestRecord:
    """Daily per-symbol options open interest from OCC bulk download."""
    date: str              # "YYYY-MM-DD"
    symbol: str            # "GME"
    exchange: str          # Single-char: "A"=AMEX, "C"=CBOE, etc.
    volume: int            # Daily contract volume
    exercised: int         # Contracts exercised
    open_interest: int     # End-of-day OI (contracts)
    product_kind: str      # "OSTK" (stock options)
    expiration: str        # "YYYY-MM-DD"
```

Note: OCC `cont-volume-download` does not distinguish puts from calls.
Each row is aggregated by symbol + exchange + expiration. For put/call
breakdown, the `volume-query` endpoint is needed (separate method).

## Client Architecture

```python
class OccOptionsClient(BaseDataClient):
    """OCC daily options open interest and volume data.

    Downloads the daily bulk file from OCC, parses fixed-width text,
    and filters for the requested symbol. Caches per-symbol results
    and tracks download progress for incremental backfill.
    """

    def __init__(self, data_dir="data"):
        super().__init__(
            rate_limit_delay=0.5,
            data_dir=data_dir,
            default_headers={"User-Agent": "StockDownloader admin@example.com"},
        )

    def fetch_open_interest(
        self, symbol, start_date=None,
    ) -> list[OccOpenInterestRecord]:
        """Fetch daily OI for a symbol from OCC bulk files.

        1. Determine date range (Jan 4 2021 to today, or start_date)
        2. Load progress file to skip already-downloaded dates
        3. For each remaining trading day:
           - Download bulk file (~2-5MB)
           - Parse fixed-width, filter for symbol
           - Collect matching records
        4. Save incrementally every 25 dates
        5. Merge with cache, return sorted by date
        """

    def fetch_put_call_volume(
        self, symbol, date_str,
    ) -> dict:
        """Fetch put/call volume breakdown for one date.

        Uses OCC volume-query endpoint:
        https://marketdata.theocc.com/volume-query?reportDate=YYYYMMDD
            &format=csv&volumeQueryType=O&symbolType=U&symbol=GME
            &reportType=D&accountType=C&productKind=OSTK

        Returns {"calls": int, "puts": int, "ratio": float}
        """
```

### Parsing

Fixed-width parsing (52-char records):

```python
def _parse_bulk_line(self, line, report_date):
    if len(line) < 52:
        return None
    symbol = line[0:6].strip()
    exchange = line[12]
    volume = int(line[13:22])
    exercised = int(line[22:31])
    oi = int(line[31:40])
    product_kind = line[40:44].strip()
    expiration_raw = line[44:52].strip()
    expiration = f"{expiration_raw[:4]}-{expiration_raw[4:6]}-{expiration_raw[6:8]}"
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
```

### Caching & Progress

```
data/{SYMBOL}/
  occ_open_interest.json       # All OI records merged
  occ_options_progress.json    # List of dates already downloaded
```

No raw bulk file caching — each file is ~2-5MB, 1,250 files = ~3-6GB.
Parse in-memory, save only the filtered symbol records.

Progress file enables resume: if the backfill is interrupted, restart
skips already-processed dates.

### Incremental Save Pattern

Same as `RegShoThresholdClient.fetch_threshold_targeted()`:

```python
BATCH_SIZE = 25

for i, date_str in enumerate(remaining_dates):
    self._rate_limit()
    records = self._download_and_parse(date_str, symbol)
    all_records.extend(records)
    progress.add(date_str)

    if (i + 1) % BATCH_SIZE == 0:
        self._save_progress(symbol, progress)
        self._save_cache(symbol, self._merge(cached, all_records))

# Final save
self._save_progress(symbol, progress)
self._save_cache(symbol, self._merge(cached, all_records))
```

### Error Handling

- 404 / empty response: Mark date in progress (no data available), continue
- Connection error / timeout: Increment failure counter, skip date
- Consecutive failures >= 15: Stop, save progress, return partial results
- Malformed lines: Skip silently (log at DEBUG)

## Analysis Script

`scripts/gme_options_analysis.py`:

1. Load OI data from `data/GME/occ_open_interest.json`
2. Aggregate daily: total OI (contracts), total OI (shares = contracts * 100)
3. Compare total OI (shares) to float (50.65M) and outstanding (69.75M)
4. Compute daily OI change to detect sudden spikes
5. Identify expirations with anomalous OI (far OTM, high concentration)
6. Cross-reference with FTD data and short volume for correlation
7. Print summary report + key findings

## Files Changed

| File | Action |
|------|--------|
| `src/stockdownloader/model/regulatory_records.py` | Add `OccOpenInterestRecord` |
| `src/stockdownloader/model/__init__.py` | Export new type |
| `src/stockdownloader/data/occ_options_client.py` | **NEW** |
| `src/stockdownloader/data/__init__.py` | Export `OccOptionsClient` |
| `scripts/gme_options_analysis.py` | **NEW** |
| `tests/data/test_occ_options_client.py` | **NEW** |

## Estimates

- Backfill: ~1,250 requests at 0.5s = ~10 minutes
- Each bulk file: ~2-5MB, parsed in-memory
- GME records per day: ~300-400 (across exchanges/expirations)
- Total stored data: ~400K records * ~100 bytes = ~40MB JSON

## Open Questions (deferred)

- Put/call OI breakdown requires per-date `volume-query` calls (separate
  from bulk download). Can add later as `fetch_put_call_volume()`.
- Pre-2021 data requires paid provider (DeltaNeutral $500/yr or
  ThetaData $30/mo). Out of scope for now.
