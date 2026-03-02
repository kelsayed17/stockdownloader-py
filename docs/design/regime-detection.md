# Regime Detection System

## Overview

The regime detection system classifies each price bar into one of five market
regimes so that the ensemble strategy and tournament can select the best
trading strategy for current conditions.  It lives in
`src/stockdownloader/strategy/regime/` and is consumed by the ensemble
meta-strategy (Phase 5) and the grand tournament (Phase 6).

## The 5 Market Regimes

| Regime | Enum value | Description |
|---|---|---|
| **Strong Trend Up** | `STRONG_TREND_UP` | High ADX (>30) with positive slope and +DI > -DI. Trend-following strategies dominate. |
| **Strong Trend Down** | `STRONG_TREND_DOWN` | High ADX (>30) with negative slope and -DI > +DI. Short-biased or trend-following strategies. |
| **Weak Trend** | `WEAK_TREND` | ADX between 20-30 or high ADX without directional clarity. Also the default fallback regime. |
| **Mean Reverting** | `MEAN_REVERTING` | Low ADX (<20) combined with tight Bollinger Bands (below 40th percentile). Favours oscillator strategies like RSI. |
| **High Volatility** | `HIGH_VOLATILITY` | Bollinger Band width above the 80th percentile of its lookback history. Overrides all other classifications. |

Defined in `MarketRegime` enum (`regime_detector.py:33-40`).

## Detection Methodology

`MarketRegimeDetector.classify()` computes four indicators then applies a
priority-based decision tree.

### Indicators

| Indicator | Default config | Purpose |
|---|---|---|
| **ADX** (Average Directional Index) | period=14 | Measures trend strength (0-100). Also provides +DI/-DI for direction. |
| **Bollinger Band width percentile** | period=20, std=2.0, lookback=120 | Current BB width ranked against last 120 bars. Measures relative volatility. |
| **Normalised slope** | period=15, normalised by ATR(14) | Linear price change over 15 bars divided by ATR. Measures directional momentum. |
| **SMA(200) distance** | period=200 | Price distance from SMA(200) as a percentage. Captures long-term trend position. |

Configuration is held in `RegimeDetectorConfig` (`regime_detector.py:71-90`).

### Decision Tree (priority order)

1. **HIGH_VOLATILITY** -- BB width percentile > 0.80. Overrides everything.
2. **STRONG_TREND_UP / STRONG_TREND_DOWN** -- ADX > 30 AND |slope| > 0.5.
   Direction determined by slope sign and DI agreement. Confidence reduced
   by 30% when DI disagrees with slope.
3. **MEAN_REVERTING** -- ADX < 20 AND BB width percentile < 0.40.
4. **WEAK_TREND** -- ADX >= 20 (catch-all for moderate trend strength).
5. **WEAK_TREND (default)** -- Low ADX but BB width not tight enough for
   mean-reversion. Assigned confidence 0.3.

Each branch computes a confidence score in [0, 1] based on how far the
indicator exceeds its threshold.

Implementation: `_classify_decision_tree()` (`regime_detector.py:272-313`).

### Warmup Requirement

`warmup_period = max(sma_period, 120) + 1` -- defaults to 201 bars.
Bars before warmup cannot be reliably classified.

## Integration Points

### 1. Ensemble Strategy (`strategy/regime/ensemble_strategy.py`)

`EnsembleIntradayStrategy` is a meta-strategy that:
- Holds a `dict[MarketRegime, IntradayTradingStrategy]` mapping regimes to
  sub-strategies.
- On each bar, calls `regime_detector.classify()` to get the current regime.
- Delegates `evaluate()` to the sub-strategy mapped to that regime (or a
  default fallback).
- Optionally applies `DrawdownPositionScaler` to reduce position size during
  drawdowns (linear scale: full at 0%, half at 5%, zero at 10% drawdown).
- Annotates signal reasons with the detected regime, e.g. `[mean_reverting]`.

### 2. Regime-Strategy Mapper (`strategy/regime/regime_strategy_map.py`)

`RegimeStrategyMapper` tracks per-trade performance tagged by entry-bar regime:
- `record(strategy_name, regime, pnl)` accumulates P&L, trade count, wins.
- `best_strategy_for_regime(regime, min_trades=3)` returns the strategy with
  the highest average P&L in that regime.
- `performance_matrix` property returns a nested dict of strategy -> regime ->
  avg_pnl for reporting.

### 3. Tournament Analysis (`backtest/tournament_analysis.py`)

- `classify_timeframe_bars(data)` classifies every bar in a dataset into
  regimes using `MarketRegimeDetector`.
- `run_regime_analysis(key, trades, regime_at_bar)` maps each trade's entry
  date to its regime and computes `RegimeTradeStats` per regime.
- `_compute_regime_bonus()` scores strategies on four criteria:
  - **Coverage** (+1.0 per regime with >= 3 trades, max +5.0)
  - **Consistency** (low stdev of avg P&L across regimes)
  - **Collapse penalty** (worst regime with large negative avg P&L)
  - **Single-regime penalty** (-2.0 if all trades fall in one regime)

### 4. Tournament Pipeline (`app/tournament/stages_advanced.py`)

Stage 2.5 "Regime Analysis" in the tournament pipeline:
1. Classifies all bars per timeframe into regimes.
2. Runs parallel regime analysis for qualifying strategy combos.
3. Recomputes tournament scores with regime bonus applied.
4. Feeds results into `RegimeStrategyMapper` for cross-strategy comparison.
5. Prints a summary table (coverage, worst regime, consistency, bonus) and
   the best strategy per regime.

## Module Dependency Map

```
app/tournament/stages_advanced.py
    --> backtest/tournament_analysis.py
        --> strategy/regime/regime_detector.py
            --> util/indicators/hub.py (IndicatorHub)
    --> strategy/regime/regime_strategy_map.py
        --> strategy/regime/regime_detector.py (MarketRegime enum)

strategy/regime/ensemble_strategy.py
    --> strategy/regime/regime_detector.py
    --> strategy/trading_strategy.py (IntradayTradingStrategy base)
    --> model/trade.py (IntradaySignal, HOLD)
```

Key dependency: all regime modules depend on `regime_detector.py` for the
`MarketRegime` enum and `RegimeClassification` dataclass.
