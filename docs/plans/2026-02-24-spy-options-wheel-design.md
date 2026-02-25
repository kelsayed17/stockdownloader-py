# SPY ML-Guided Weekly Wheel Strategy Design

**Date**: 2026-02-24
**Status**: Approved
**Branch**: `claude/vigorous-easley`

## Overview

ML-guided weekly wheel strategy on SPY. Sell weekly cash-secured puts at 0.30
delta. If assigned, sell weekly covered calls at 0.30 delta. ML model acts as a
sell/skip filter: skip selling puts when crash signal is strong, skip selling
calls when a strong rally is detected. Backtest with real Polygon historical
option prices.

## Architecture

Standalone pipeline (`spy-options-wheel`) that reuses the existing ML
walk-forward ensemble for signal generation but has its own data client,
backtest engine, and CLI.

```
spy-options-wheel CLI
    |
    v
[1] Download SPY daily data (existing Yahoo/Polygon client)
[2] Run ML walk-forward pipeline -> daily probability predictions
[3] Download option chain data from Polygon (with caching)
[4] Run wheel backtest (ML-filtered vs mechanical)
[5] Print results comparison
```

## Data Layer

### PolygonOptionsClient

New file: `src/stockdownloader/data/market/polygon_options_client.py`

Fetches SPY option contract metadata and prices from Polygon.io REST API.
Follows the same patterns as the existing `PolygonDataClient` (rate limiting,
pagination, session-based auth via `POLYGON_API_KEY` env var).

Key methods:

- `fetch_option_contracts(symbol, from_date, to_date, expired=True)` — Lists
  all contracts (including expired) via `GET /v3/reference/options/contracts`.
  Filters to Friday expirations for weeklies.
- `fetch_option_daily_bar(option_ticker, trade_date)` — Gets OHLCV for a
  specific contract on a specific day via
  `GET /v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}`. This is the entry
  premium when selling.
- `build_weekly_chain_cache(symbol, from_date, to_date)` — Orchestrator that
  discovers contracts for each Friday expiration and saves to JSON cache.

### Cache Structure

```
data/options_cache/SPY/{expiration_date}.json
```

Each file contains all strikes, their Polygon contract tickers, and any
downloaded prices. Avoids re-downloading on subsequent runs.

### API Call Budget

- ~208 weeks x 1 contract price lookup = ~208 calls for prices
- Plus paginated contract listing (a few hundred calls)
- Manageable with unlimited API tier

### Strike Selection

Polygon does not provide historical greeks. We use the existing Black-Scholes
`delta()` function from `stockdownloader.analysis.options.pricing` to compute
delta for each available strike at the entry date, then pick the one closest to
the target delta (0.30).

### Data Needed Per Week

1. Available strikes for that week's expiration (from Polygon contracts
   reference)
2. SPY closing price on entry day (Monday) — to compute BS delta
3. Real market price of the selected option contract on entry day (from Polygon
   aggregates)
4. SPY closing price on expiry day (Friday) — for settlement (intrinsic value)

## Wheel Backtest Engine

New file: `src/stockdownloader/backtesting/engines/wheel.py`

### State Machine

```
                  assigned              called away
  PUT_PHASE  ────────────>  CALL_PHASE  ────────────>  PUT_PHASE
      |                         |                         (repeat)
      | OTM expiry              | OTM expiry
      v                         v
  keep premium              keep premium
  stay in PUT phase         stay in CALL phase
```

Four states:

- **CASH** — No stock, no option (initial state)
- **PUT_PHASE** — Cash-secured, selling weekly puts
- **HOLDING** — Own stock (just assigned), no option yet
- **CALL_PHASE** — Own stock, selling weekly covered calls

### Position Sizing

- 1 contract = 100 shares of SPY
- Cash-secured puts require `strike x 100` in collateral per contract
- With ~$100K capital and SPY at ~$500, that is ~2 contracts max
- Configurable via `--contracts` CLI flag

### Entry Timing

Sell options on **Monday close** each week. The ML model's Friday prediction
(most recent signal) informs the Monday decision. This avoids look-ahead bias.

### Premium Pricing

Use real Polygon market prices (Monday close of the selected contract). Fall
back to Black-Scholes if Polygon data is missing for a given contract/date.

### Settlement

On Friday:
- **ITM**: Assigned (put) or called away (call) at the strike price
- **OTM**: Expires worthless, keep full premium

### WheelBacktestEngine Class

```python
class WheelState(Enum):
    CASH = "CASH"
    PUT_PHASE = "PUT_PHASE"
    HOLDING = "HOLDING"
    CALL_PHASE = "CALL_PHASE"

class WheelBacktestEngine:
    def __init__(self, initial_capital, target_delta, contracts):
        ...

    def run(self, weekly_schedule, ml_signals, option_prices, spy_prices):
        # For each week:
        #   1. Check expiration of previous week's option
        #   2. Update state (assigned? called away? expired OTM?)
        #   3. Consult ML filter
        #   4. If filter says sell -> sell new option at 0.30 delta
        #   5. Record premium, update equity curve
```

### Output Metrics

- Total premium collected
- Number of assignments / calls exercised
- Total return, annualized return
- Max drawdown, Sharpe ratio
- Comparison vs buy-and-hold and vs mechanical (no ML) wheel

## ML Signal Integration

Reuses the existing walk-forward ensemble pipeline from `spy_ml_ensemble.py`.
No new ML training needed.

### Weekly Aggregation

The walk-forward pipeline produces daily probability predictions. For the
wheel, we use **Friday's prediction** for **Monday's decision** (most recent
signal without look-ahead).

### Filter Rules

```
PUT_PHASE:   if prob < 0.35 -> skip selling put (crash danger)
CALL_PHASE:  if prob > 0.65 -> skip selling call (rally expected)
Otherwise:   sell at target delta as normal
```

### Thresholds

- `--skip-put-thresh` (default 0.35) — Skip selling puts below this
- `--skip-call-thresh` (default 0.65) — Skip selling calls above this

## CLI

New file: `src/stockdownloader/app/spy_options_wheel.py`

Command: `spy-options-wheel`

### Arguments

```
--no-ml-filter           Pure mechanical wheel (no ML)
--delta 0.30             Target delta for strike selection (default: 0.30)
--skip-put-thresh 0.35   Skip puts below this prob (default: 0.35)
--skip-call-thresh 0.65  Skip calls above this prob (default: 0.65)
--initial-capital 100000 Starting capital (default: 100000)
--from-date 2022-01-01   Backtest start date
--to-date 2026-02-24     Backtest end date
--contracts 1            Number of contracts per trade (default: 1)
```

### Output

Side-by-side comparison:

```
============================================================
SPY WEEKLY WHEEL BACKTEST RESULTS
============================================================
                        ML Wheel    Mechanical    Buy & Hold
Total Return            +XX.X%      +XX.X%        +XX.X%
Annualized Return       +XX.X%      +XX.X%        +XX.X%
Premium Collected       $XX,XXX     $XX,XXX       N/A
Assignments             XX          XX            N/A
Calls Exercised         XX          XX            N/A
Weeks Skipped (ML)      XX          N/A           N/A
Trades                  XXX         XXX           0
Sharpe                  X.XX        X.XX          X.XX
Max Drawdown            -XX.X%      -XX.X%        -XX.X%
```

## Testing

### Test Files

- `tests/data/market/test_polygon_options_client.py` — Mock API responses for
  contract discovery, price fetching, pagination, cache read/write, BS delta
  strike selection
- `tests/backtesting/engines/test_wheel.py` — State transitions (all 4 states),
  OTM expiry, ITM assignment, ITM call exercise, ML filter skipping, equity
  curve and metrics
- `tests/app/test_spy_options_wheel.py` — CLI parser tests, integration smoke
  test

All Polygon API responses are mocked. No real API calls in tests.

## New Files Summary

| File | Purpose |
|------|---------|
| `src/stockdownloader/data/market/polygon_options_client.py` | Polygon options data client + cache |
| `src/stockdownloader/backtesting/engines/wheel.py` | Wheel backtest state machine |
| `src/stockdownloader/app/spy_options_wheel.py` | CLI pipeline |
| `tests/data/market/test_polygon_options_client.py` | Options client tests |
| `tests/backtesting/engines/test_wheel.py` | Wheel engine tests |
| `tests/app/test_spy_options_wheel.py` | CLI + integration tests |
