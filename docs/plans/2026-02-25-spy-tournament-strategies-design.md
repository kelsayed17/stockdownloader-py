# SPY Tournament Strategy Python Conversion — Design

**Date:** 2026-02-25
**Status:** Approved
**Branch:** claude/vigorous-easley

## Problem

Three SPY intraday strategies won the walk-forward tournament with strong
out-of-sample scores, but they only exist as Pine Script template generators
(`spy_strategies.py`).  There is no Python backtest implementation — they
cannot be run through `IntradayBacktestEngine`.

| Strategy | OOS Score | Degradation | Direction |
|----------|-----------|-------------|-----------|
| SPY MACD+OBV | +74.03 | 0.96 | Long + Short |
| SPY SMA 20/21 | +65.29 | 2.52 | Long only |
| SPY MACD 8/35/5 | +62.80 | 1.57 | Long + Short |

## Approach

**Lean Framework Integration** — create 3 new `BaseIntradayStrategy`
subclasses that plug into the existing intraday engine, CLI, and reporting.
Add a shared `TournamentRiskGuard` mixin for Pine v2 risk management
features not yet in the framework.

## Component 1: TournamentRiskGuard Mixin

Encapsulates the shared risk management logic from all three Pine v2
strategy scripts.  Designed as a mixin class so any future strategy can
opt in.

### State (reset each session)

| Field | Type | Description |
|-------|------|-------------|
| `_day_trades` | int | Trades taken today |
| `_last_bar_entry` | int | `bar_index` of most recent entry |
| `_consec_losses` | int | Consecutive losing trades |
| `_circuit_tripped` | bool | True after `max_consec_losses` losses |
| `_day_open_equity` | float | Portfolio equity at session open |
| `_day_limited` | bool | True when daily DD exceeds threshold |

### Config

| Parameter | Default | Pine v2 Source |
|-----------|---------|----------------|
| `max_day_trades` | 4 | `dayTrades < 4` |
| `min_bar_spacing` | 3 | `bar_index - lastBarEntry >= 3` |
| `max_consec_losses` | 3 | `consecLoss >= 3` |
| `max_day_dd_pct` | -3.0 | `(equity - dayEquity) / dayEquity * 100 <= -3.0` |

### Methods

```
guard_allows_entry(bar_index, current_equity) -> bool
    Checks: under trade limit, not tripped, not DD-limited, bar-spaced.

on_guard_session_start(equity)
    Resets daily counters.

on_guard_trade_closed(pnl)
    Tracks consecutive losses, trips circuit breaker if threshold hit.

record_entry(bar_index)
    Increments day_trades, records bar_index.
```

## Component 2: Three Strategy Classes

All extend `BaseIntradayStrategy` and mix in `TournamentRiskGuard`.

### 2a. MACDOBVStrategy — Tournament Winner

**File:** `src/stockdownloader/strategies/intraday/macd_obv.py`

**Signal logic:**
- Long:  `MACD(12,26,9) crosses above signal` AND `EMA(OBV, 5) rising`
- Short: `MACD(12,26,9) crosses below signal` AND `EMA(OBV, 5) falling`
- Exit:  Reverse MACD cross (crossunder exits long, crossover exits short)

**Config:** `MACDOBVConfig`

| Parameter | Default | Notes |
|-----------|---------|-------|
| macd_fast | 12 | Standard |
| macd_slow | 26 | Standard |
| macd_signal | 9 | Standard |
| obv_smooth | 5 | EMA period for OBV |
| atr_len | 14 | For SL calculation |
| sl_mult | 1.5 | ATR multiplier |
| rr_ratio | 1.5 | Reward:Risk |
| sl_cap | 2.0 | Hard dollar cap on SL |
| be_trigger | 0.5 | Breakeven at 0.5R |
| be_buffer | 0.05 | Buffer above/below entry for BE SL |

### 2b. SMACross2021Strategy — Tournament #2

**File:** `src/stockdownloader/strategies/intraday/sma_cross.py`

**Signal logic:**
- Long:  `SMA(20) crosses above SMA(21)` — tight golden cross
- Exit:  `SMA(20) crosses below SMA(21)` — death cross
- No short side.

**Config:** `SMACross2021Config`

| Parameter | Default | Notes |
|-----------|---------|-------|
| sma_short | 20 | Fast SMA |
| sma_long | 21 | Slow SMA (1-bar gap = momentum) |
| atr_len | 14 | |
| sl_mult | 1.5 | |
| rr_ratio | 1.5 | |
| sl_cap | 2.0 | |
| be_trigger | 0.5 | |
| be_buffer | 0.05 | |

### 2c. MACDOptimizedStrategy — Tournament #3

**File:** `src/stockdownloader/strategies/intraday/macd_optimized.py`

**Signal logic:**
- Long:  `MACD(8,35,5) crosses above signal` — no secondary filter
- Short: `MACD(8,35,5) crosses below signal`
- Exit:  Reverse MACD cross
- Key: wide slow (35) filters noise, fast signal (5) for quick timing

**Config:** `MACDOptimizedConfig`

| Parameter | Default | Notes |
|-----------|---------|-------|
| macd_fast | 8 | Faster than standard |
| macd_slow | 35 | Wider than standard — noise filter |
| macd_signal | 5 | Faster than standard |
| atr_len | 14 | |
| sl_mult | 1.5 | |
| rr_ratio | 1.5 | |
| sl_cap | 2.0 | |
| be_trigger | 0.5 | |
| be_buffer | 0.05 | |

## Component 3: Shared SL/TP/Breakeven

All three strategies use identical risk math:

```
sl_risk    = min(ATR(atr_len) * sl_mult, sl_cap)
long_sl    = entry - sl_risk
long_tp    = entry + sl_risk * rr_ratio
short_sl   = entry + sl_risk
short_tp   = entry - sl_risk * rr_ratio

# Breakeven: when unrealized profit reaches be_trigger * sl_risk
# Move SL to entry + be_buffer (long) or entry - be_buffer (short)
```

This integrates with `IntradayExitManager` via existing `be_trigger` config.
The $2 SL cap is applied via `clamp_sl_dist()`.

## Component 4: Strategy Registration

Register in `config/strategies/intraday_registrations.json`:

```json
{"name": "spy-macd-obv", "module": "stockdownloader.strategies.intraday.macd_obv",
 "class": "MACDOBVStrategy", "category": "intraday"}
{"name": "spy-sma-2021", "module": "stockdownloader.strategies.intraday.sma_cross",
 "class": "SMACross2021Strategy", "category": "intraday"}
{"name": "spy-macd-opt", "module": "stockdownloader.strategies.intraday.macd_optimized",
 "class": "MACDOptimizedStrategy", "category": "intraday"}
```

## Component 5: Comparison Runner

**File:** `scripts/spy_tournament_compare.py`

1. Fetches 2yr 5-min SPY data via Polygon API (with CSV cache)
2. Runs all strategies through `IntradayBacktestEngine`:
   - 3 new tournament strategies
   - 5 existing native intraday (VWAP Pullback, Reversal, ORB, ORR, PS)
   - 6 daily-adapted (RSI, MACD, SMA, BB-RSI, Breakout, Momentum)
3. Prints unified comparison table (return %, Sharpe, MaxDD, win rate, # trades)
4. Saves per-strategy results to `output/tournament_comparison/`

### Data Pipeline

```
Polygon API (5-min bars, 2 years)
    -> CSV cache: data/spy_5min_2yr.csv
    -> IntradayPriceData list
    -> IntradayBacktestEngine.run(strategy, data)
    -> BacktestResult
    -> Comparison table + per-strategy reports
```

## Files to Create

| File | Purpose |
|------|---------|
| `strategies/intraday/tournament_guard.py` | TournamentRiskGuard mixin |
| `strategies/intraday/macd_obv.py` | MACD+OBV strategy + config |
| `strategies/intraday/sma_cross.py` | SMA 20/21 strategy + config |
| `strategies/intraday/macd_optimized.py` | MACD 8/35/5 strategy + config |
| `scripts/spy_tournament_compare.py` | Comparison runner |

## Files to Modify

| File | Change |
|------|--------|
| `config/strategies/intraday_registrations.json` | Add 3 registrations |

## Test Plan

- Unit tests for each strategy's signal logic
- Unit tests for TournamentRiskGuard (daily limit, circuit breaker, etc.)
- Integration test: strategy produces valid signals on synthetic data
- Live backtest: 2yr Polygon data comparison
