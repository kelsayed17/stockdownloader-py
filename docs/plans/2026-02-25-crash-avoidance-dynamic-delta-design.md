# Crash Avoidance + Dynamic Delta Design

## Problem

The ML Combined wheel strategy returns +55.1% (Sharpe 1.07, MaxDD -12.5%) over
2022-01-01 to 2026-02-25. It beats buy-and-hold (+47.9%) on return but carries a
-12.5% max drawdown. The previous attempt to reduce drawdown (collar, vol scaling,
iron condor) all degraded returns more than they improved risk metrics.

The ML ensemble pipeline's active trading mode (+40.8%) also loses to buy-and-hold
(+59.2%) because the ML signal is too weak for entry timing — it misses too many
rally days.

## Insight

The ML signal is weak for timing entries but useful for detecting danger. Instead of
buying expensive weekly insurance (collar) or varying position counts (vol scaling),
use the ML signal to go to cash when danger is high and vary strike aggressiveness
based on implied volatility.

## Two Features

### Feature 1: Crash Avoidance

**Concept:** Stay invested and sell options by default. Exit to cash only when ML
probability drops below a strong bearish threshold. Re-enter when probability
recovers. Cost: $0 in premiums (vs collar's ~$16K).

**Applies to:**
- Wheel pipeline (`spy_options_wheel.py`) — engine-level in `WheelBacktestEngine`
- ML ensemble pipeline (`spy_ml_ensemble.py`) — backtest-level in
  `_run_portfolio_backtest()`
- ML mega pipeline (`spy_ml_mega.py`) — same, wired through

**Parameters:**
- `crash_exit_thresh` (default 0.25) — exit when ML prob drops below this
- `re_entry_thresh` (default 0.50) — re-enter when ML prob recovers above this

**Wheel engine behavior:**
1. Pre-check in `process_week()` before mode dispatch
2. If invested and prob < exit thresh → sell all shares at expiry price, set
   `in_cash_mode = True`, skip option selling
3. If in cash mode and prob >= re-entry thresh → set `in_cash_mode = False`,
   let combined mode buy shares on next dispatch
4. If in cash mode → skip everything, record cash equity
5. Otherwise → fall through to normal combined mode

**ML ensemble behavior (existing plan):**
1. Start fully invested (buy max shares immediately)
2. If invested and prob < exit thresh → sell all, go to cash
3. If in cash and prob >= re-entry thresh → buy max shares
4. Otherwise → hold, mark-to-market

### Feature 2: Dynamic Delta

**Concept:** Instead of fixed 0.30 delta for every week, select delta based on IV
percentile. More aggressive (closer to ATM) when vol is high to capture rich
premium. More conservative (further OTM) when vol is low. Skip entirely when
premium is too thin.

**Applies to:** Wheel pipeline only (ML ensemble has no options).

**IV percentile to delta mapping:**

| IV Percentile | Delta | Rationale                                    |
|---------------|-------|----------------------------------------------|
| < 0.25        | Skip  | Premium too thin, not worth the risk         |
| 0.25 – 0.50   | 0.20  | Conservative — further OTM when vol is low   |
| 0.50 – 0.75   | 0.30  | Standard — current default                   |
| > 0.75        | 0.40  | Aggressive — capture rich premium in high vol |

**Implementation:** CLI-level logic in `spy_options_wheel.py`. After IV percentile
is computed but before `select_strike_by_delta`. No engine changes.

## Composition

The two features compose cleanly:
1. Dynamic delta selects strikes (or skips week if IV too low)
2. Engine receives WeekRecord with selected strikes
3. Crash avoidance pre-check runs first in `process_week()` — if in cash mode,
   skips regardless of strikes
4. If not in cash mode, combined mode processes with whatever strikes dynamic
   delta picked

No conflicts or special interaction logic needed.

## Files to Modify

### Wheel pipeline (crash avoidance + dynamic delta)

| File | Changes |
|------|---------|
| `src/stockdownloader/backtesting/engines/wheel.py` | Add crash_avoidance, crash_exit_thresh, re_entry_thresh params; pre-check in process_week(); metrics for crash exits/reentries |
| `tests/backtesting/engines/test_wheel.py` | TestCrashAvoidance (6-8 tests) |
| `src/stockdownloader/app/spy_options_wheel.py` | Add --crash-avoidance, --crash-exit-thresh, --re-entry-thresh, --dynamic-delta flags; per-week delta logic in main loop |
| `tests/app/test_spy_options_wheel.py` | Parser tests for 4 new args |

### ML ensemble pipeline (crash avoidance only)

| File | Changes |
|------|---------|
| `src/stockdownloader/app/spy_ml_ensemble.py` | Add --crash-avoidance, --crash-exit-thresh, --re-entry-thresh; new code path in _run_portfolio_backtest(); wire through orchestrator + main |
| `src/stockdownloader/app/spy_ml_mega.py` | Same 3 CLI args; wire through main |
| `tests/app/test_spy_ml_ensemble_tournament.py` | TestCrashAvoidanceBacktest (6 tests) + parser tests (6 tests) |
| `tests/app/test_spy_ml_mega.py` | Parser tests (6 tests) |

## Target Metrics

### Wheel pipeline (crash avoidance + dynamic delta)
- Return: +55-60% (maintain or improve over baseline +55.1%)
- Sharpe: 1.2-1.5 (up from 1.07)
- MaxDD: -5-7% (down from -12.5%)

### ML ensemble pipeline (crash avoidance)
- Return: ~55-58% (up from +40.8%, closer to buy-and-hold's +59.2%)
- Sharpe: >0.9 (up from 0.81)
- Fewer trades than active mode

## Verification

```bash
# Unit tests
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py tests/app/test_spy_options_wheel.py -v

# ML ensemble tests
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/app/test_spy_ml_ensemble_tournament.py tests/app/test_spy_ml_mega.py -v

# Full regression
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q

# Live: wheel with crash avoidance + dynamic delta
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --crash-avoidance --dynamic-delta --quick

# Live: ML ensemble crash avoidance
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_ml_mega --quick --no-pine --walk-forward-windows 3 --crash-avoidance
```
