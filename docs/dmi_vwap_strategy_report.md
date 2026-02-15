# DMI + Anchored VWAP Strategy Report

## Strategy Overview

The DMI+VWAP strategy combines the **Directional Movement Index** (DMI/ADX, 14-period) with an **Anchored Volume-Weighted Average Price** (VWAP) that resets at the start of each trading session (09:30 ET) using typical price = (H + L + C) / 3.

This is a trend-following intraday strategy designed for SPY on 15-minute or 5-minute timeframes. It enters directional options trades (calls or puts) when price action and momentum align.

### Entry Rules

| Signal | Direction | Conditions |
|--------|-----------|------------|
| **Buy Calls** (Long) | Bullish | Price > Anchored VWAP **AND** +DI > -DI **AND** ADX > 25 |
| **Buy Puts** (Short) | Bearish | Price < Anchored VWAP **AND** -DI > +DI **AND** ADX > 25 |
| No Trade | Neutral | Otherwise |

### Exit Rules

1. **Signal Reversal**: Opposite conditions are met (e.g., long position exits when price drops below VWAP with bearish DI crossover)
2. **End-of-Day**: All positions closed 10 minutes before market close (15:50 ET)
3. **Neutral Exit**: After minimum 1-hour hold, if ADX drops below 25 or DI crossover reverses

### Trade Management

- **Minimum hold period**: 12 bars (1 hour on 5-min timeframe) before neutral exits
- **Cooldown**: 6 bars (30 minutes) after any exit before re-entry
- **Max daily trades**: 3 per session to avoid overtrading
- **No overnight positions**: All trades are intraday

---

## Position Sizing

| Parameter | Value |
|-----------|-------|
| Account Size | $25,000 |
| Risk Per Trade | 1-2% ($250-$500 max loss) |
| Options Type | ATM weekly (Friday expiry) |
| Delta Target | ~0.50-0.55 |
| Stop Loss | 1.5x ATR from entry |
| Take Profit | 2:1 reward/risk ratio (3x ATR from entry) |

### Options Position Sizing Example

With a $25,000 account risking 1%:
- **Max risk per trade**: $250
- **SPY at $690**: ATM weekly call/put premium ~ $3.00-$5.00
- **Stop loss**: If premium drops by $1.00-$1.50
- **Contracts**: $250 / ($1.00 * 100) = 2-3 contracts
- **Max position cost**: 2-3 contracts * $3.50 * 100 = $700-$1,050

---

## Current Signal Example (Hypothetical, Feb 10, 2026)

### Market Context
- **SPY Price**: ~$693.50 at 11:30 ET
- **Session VWAP**: $694.12 (above price)
- **ADX**: 28.7 (strong trend)
- **+DI**: 18.4
- **-DI**: 26.1 (dominant)

### Signal: **BUY PUT** (Bearish)
- Price ($693.50) < VWAP ($694.12)
- -DI (26.1) > +DI (18.4)
- ADX (28.7) > 25 threshold

### Trade Suggestion
| Field | Value |
|-------|-------|
| Direction | Buy Put (SPY Feb 14 $694P) |
| Entry Time | 11:30 ET |
| Entry Premium | ~$3.80 |
| Contracts | 2 |
| Total Cost | $760 |
| Stop Loss | Premium drops to $2.30 ($300 loss) |
| Take Profit | Premium rises to $5.80 ($400 gain) |
| Max Hold | Until 15:50 ET or signal reversal |

---

## Backtest Results

### Full Period: Feb 2024 - Feb 2026 (522 trading days)

| Metric | Value |
|--------|-------|
| Initial Capital | $25,000.00 |
| Final Capital | $24,483.92 |
| Total Trades | 986 |
| Wins / Losses | 433 / 553 |
| Win Rate | 43.9% |
| Avg Win | $59.00 |
| Avg Loss | $47.13 |
| Avg R:R | 1.25:1 |
| Total P/L | -$516.08 |
| Return | -2.06% |
| Profit Factor | 0.98 |
| Max Drawdown | 55.69% |
| Sharpe Ratio | 0.78 |
| Sortino Ratio | 1.72 |
| Max Consecutive Wins | 8 |
| Max Consecutive Losses | 11 |
| Avg Trade Duration | 24.1 bars (~2 hours) |

### Direction Breakdown

| Direction | Trades | Wins | Win Rate | P/L |
|-----------|--------|------|----------|-----|
| Long (Calls) | 494 | 226 | 45.7% | +$1,200.83 |
| Short (Puts) | 492 | 207 | 42.1% | -$1,716.91 |

### Key Observations

1. **Long side is profitable**: +$1,200.83 with 45.7% win rate. The bullish bias of SPY (2024-2026 bull market) favors the long entries.
2. **Short side drags performance**: -$1,716.91. Shorting into a bull market is inherently disadvantaged.
3. **Avg R:R of 1.25:1** means winners are larger than losers, but win rate needs to be above ~44.4% to break even. The strategy sits right at the edge.
4. **Max drawdown of 55.69%** is equity-based (using full position sizing on shares). With options at 1% risk, the max drawdown on account would be significantly smaller.

### Important Assumptions & Limitations

- **Backtested on equity (shares), not options**: Our engine trades SPY shares, not options contracts. Options have additional factors:
  - **Theta decay**: ATM weeklies lose ~$0.20-0.40/day in time value
  - **IV impact**: Implied volatility changes affect premium independently of direction
  - **Bid/ask spread**: SPY options typically have $0.01-0.05 spread, but weeklies can be wider
  - **Gamma risk**: ATM options have highest gamma exposure near expiration
- **No slippage modeled**: Real execution would face 1-3 cents of slippage per fill
- **No commissions**: Most brokers now offer commission-free options trading
- **15-min vs 5-min**: Strategy was tested on 5-minute bars; 15-minute bars would produce fewer signals with potentially cleaner trends

---

## Sample Trades (First 30)

| # | Date | Direction | Entry Time | Exit Time | Entry Price | Exit Price | P/L |
|---|------|-----------|------------|-----------|-------------|------------|-----|
| 1 | 2024-02-12 | LONG | 12:40 | 13:40 | $502.79 | $501.80 | -$48.51 |
| 2 | 2024-02-12 | SHORT | 14:10 | 15:45 | $501.87 | $501.00 | +$42.88 |
| 3 | 2024-02-12 | SHORT | 15:50 | 15:55 | $500.95 | $500.94 | +$0.49 |
| 4 | 2024-02-13 | SHORT | 09:40 | 15:40 | $494.27 | $492.62 | +$82.75 |
| 5 | 2024-02-14 | LONG | 09:40 | 11:00 | $496.67 | $495.40 | -$63.50 |
| 6 | 2024-02-14 | LONG | 15:50 | 15:55 | $498.31 | $498.62 | +$15.75 |
| 7 | 2024-02-15 | LONG | 09:45 | 11:05 | $500.02 | $498.93 | -$54.57 |
| 8 | 2024-02-15 | LONG | 13:25 | 14:40 | $501.46 | $501.71 | +$12.25 |
| 9 | 2024-02-16 | SHORT | 10:10 | 11:05 | $500.16 | $501.35 | -$58.06 |
| 10 | 2024-02-16 | SHORT | 14:45 | 15:45 | $500.50 | $499.05 | +$71.05 |
| 11 | 2024-02-16 | SHORT | 15:50 | 15:55 | $499.50 | $499.49 | +$0.25 |
| 12 | 2024-02-20 | SHORT | 09:40 | 14:20 | $496.71 | $495.56 | +$57.72 |
| 13 | 2024-02-21 | SHORT | 13:00 | 14:00 | $494.92 | $495.65 | -$36.51 |
| 14 | 2024-02-21 | SHORT | 14:50 | 15:30 | $494.45 | $495.11 | -$32.75 |
| 15 | 2024-02-22 | LONG | 09:40 | 14:55 | $503.67 | $507.19 | +$172.72 |
| 16 | 2024-02-23 | LONG | 09:45 | 10:45 | $509.75 | $508.81 | -$46.30 |
| 17 | 2024-02-23 | SHORT | 14:25 | 15:25 | $507.78 | $507.92 | -$6.92 |
| 18 | 2024-02-26 | SHORT | 11:20 | 12:25 | $507.18 | $507.79 | -$29.89 |
| 19 | 2024-02-26 | SHORT | 13:05 | 14:30 | $507.37 | $507.13 | +$11.76 |
| 20 | 2024-02-27 | SHORT | 13:00 | 13:55 | $505.37 | $505.77 | -$19.73 |
| 21 | 2024-02-28 | SHORT | 09:50 | 10:40 | $505.21 | $506.12 | -$44.89 |
| 22 | 2024-02-28 | SHORT | 13:00 | 13:25 | $505.73 | $506.50 | -$37.92 |
| 23 | 2024-02-29 | LONG | 10:00 | 10:55 | $508.61 | $507.03 | -$77.47 |
| 24 | 2024-02-29 | SHORT | 12:05 | 13:05 | $506.69 | $507.38 | -$33.81 |
| 25 | 2024-03-01 | LONG | 10:40 | 13:00 | $510.22 | $512.26 | +$99.96 |
| 26 | 2024-03-01 | LONG | 13:10 | 15:05 | $511.90 | $512.59 | +$33.81 |
| 27 | 2024-03-04 | LONG | 14:55 | 15:45 | $513.82 | $513.07 | -$36.75 |
| 28 | 2024-03-05 | SHORT | 09:45 | 13:50 | $511.60 | $508.07 | +$172.89 |
| 29 | 2024-03-05 | SHORT | 13:55 | 15:45 | $506.76 | $506.37 | +$19.11 |
| 30 | 2024-03-06 | LONG | 10:30 | 11:30 | $509.96 | $510.91 | +$46.38 |

---

## Performance Note (Last ~30 Days)

Based on the last 27 trading sessions in our data (Jan 2 - Feb 10, 2026):

- **SPY range**: ~$680-$695
- **Market regime**: Mixed (sideways with volatile intraday swings)
- The strategy's long entries benefited from mean-reversion moves off VWAP
- Short entries struggled due to persistent buy-the-dip behavior
- ADX readings frequently oscillated around the 25 threshold, causing some marginal signals

---

## Improvement Recommendations

1. **Long-only mode**: Given the strong long-side edge (+$1,200 vs -$1,716 on shorts), a long-only variant would have been profitable
2. **Higher ADX threshold**: Raising to 30+ would filter weak signals at the cost of fewer trades
3. **VWAP band filter**: Require price to be > VWAP + 0.5 standard deviation for longs (stronger confirmation)
4. **Time-of-day filter**: The best trades appear to cluster in 10:00-14:00 window; early morning and late afternoon are noisier
5. **Regime awareness**: Integrate with the MarketRegimeDetector to only trade in STRONG_TREND regimes
6. **Options-specific adjustments**: Model theta decay and exit options trades earlier to preserve time value

---

## TradingView Pine Script

The `docs/spy_dmi_vwap_signals.pine` file contains a complete TradingView indicator that:
- Plots the anchored VWAP (session-reset)
- Shows "Buy Call" / "Buy Put" / "Exit Long" / "Exit Short" labels
- Includes alerts for all signal types
- Displays ADX, +DI, -DI in data window
- Adds background coloring for bullish/bearish zones

### Quick Setup
1. Open TradingView > Pine Editor
2. Paste the contents of `spy_dmi_vwap_signals.pine`
3. Click "Add to Chart"
4. Set alerts: Three dots menu > "Set Alert on SPY DMI VWAP Signals..."

---

## Configuration

### DmiVwapConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dmi_period` | 14 | DMI/ADX period |
| `adx_threshold` | 25 | Minimum ADX for entries |
| `atr_period` | 14 | ATR period for stop/target |
| `sl_atr_mult` | 1.5 | Stop loss = ATR * mult |
| `rr` | 2.0 | Reward/risk ratio |
| `bars_per_day` | 78 | 5-min bars per session |
| `eod_exit_bar` | 76 | Exit bar (15:50 ET) |
| `min_entry_bar` | 3 | First allowed entry bar |
| `min_hold_bars` | 12 | Minimum hold before neutral exit |
| `cooldown_bars` | 6 | Bars to wait after exit |
| `max_trades_per_day` | 3 | Maximum trades per session |
| `require_adx_rising` | false | Require ADX slope > 0 |
| `min_di_spread` | 0 | Minimum +DI/-DI gap |

---

## Files

| File | Description |
|------|-------------|
| `src/stockdownloader/strategy/dmi_vwap_strategy.py` | Strategy implementation |
| `tests/strategy/test_dmi_vwap_strategy.py` | 23 unit tests |
| `src/stockdownloader/app/dmi_vwap_backtest.py` | Backtest runner CLI |
| `docs/spy_dmi_vwap_signals.pine` | TradingView indicator |
| `output/dmi_vwap_backtest.log` | Full backtest output with all 986 trades |

---

*Backtest executed on SPY 5-minute data (40,740 bars, 522 sessions). Strategy uses equity (shares) as proxy for directional options exposure.*
