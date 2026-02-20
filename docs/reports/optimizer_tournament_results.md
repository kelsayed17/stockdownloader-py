# Multi-Timeframe Optimizer Tournament — Results

## Executive Summary

Ran 1,080 parameter-optimized configurations across 7 strategies and 6 timeframes on SPY 5-minute bar data (60 sessions, Nov 2025 – Feb 2026). Total runtime: 42.9 minutes.

**Overall winner:** SMA Crossover on daily timeframe — +$35,918 (73.1% WR, 26 trades, long-only).

Key findings:
- Higher timeframes (4h, 1d) produced the best absolute returns with fewer, higher-quality trades
- RSI and Bollinger Band+RSI were the most consistent performers across timeframes
- Momentum Confluence strategy underperformed across all timeframes (too restrictive)
- Short-selling generally hurt performance except for BB+RSI on mid-timeframes

---

## Top 10 Configurations

| Rank | TF | Strategy | P&L | WR | Trades | Sharpe | PF | Longs | Shorts |
|------|-----|----------|-----|-----|--------|--------|-----|-------|--------|
| 1 | 1d | SMA Crossover | +$35,918 | 73.1% | 26 | 1.67 | 4.29 | 26 | 0 |
| 2 | 4h | RSI Strategy | +$29,500 | 84.2% | 19 | 1.17 | 10.74 | 19 | 0 |
| 3 | 15m | BB+RSI | +$17,702 | 61.2% | 325 | 0.80 | 1.16 | 152 | 173 |
| 4 | 1d | Multi-Indicator | +$15,584 | 100% | 1 | 0.60 | 999.99 | 1 | 0 |
| 5 | 30m | BB+RSI | +$15,301 | 64.1% | 167 | 0.81 | 1.20 | 78 | 89 |
| 6 | 5m | RSI Strategy | +$11,545 | 53.6% | 2036 | 1.13 | 1.06 | 1008 | 1028 |
| 7 | 5m | Multi-Indicator | +$11,461 | 32.2% | 2439 | 1.22 | 1.05 | 1247 | 1192 |
| 8 | 4h | BB+RSI | +$10,132 | 80.0% | 20 | 0.99 | 1.33 | 10 | 10 |
| 9 | 1h | RSI Strategy | +$9,902 | 49.7% | 155 | 1.18 | 1.16 | 71 | 84 |
| 10 | 15m | MACD Strategy | +$9,860 | 42.0% | 798 | 0.09 | 1.08 | 798 | 0 |

---

## Best Strategy Per Timeframe

| TF | Best by P&L | P&L | Best by WR | WR | Trades |
|----|------------|-----|-----------|-----|--------|
| 5m | RSI | +$11,545 | BB+RSI | 65.4% | 1,178 |
| 15m | BB+RSI | +$17,702 | BB+RSI | 61.2% | 325 |
| 30m | BB+RSI | +$15,301 | BB+RSI | 64.1% | 167 |
| 1h | RSI | +$9,902 | BB+RSI | 63.3% | 90 |
| 4h | RSI | +$29,500 | RSI | 84.2% | 19 |
| 1d | SMA Crossover | +$35,918 | SMA Crossover | 73.1% | 26 |

---

## Optimized Parameters

### SMA Crossover (Overall Winner)
```
short_period=20, long_period=21, allow_shorts=False
sl_atr_mult=1.5, rr=1.5, sl_cap=2.00
```

### RSI Strategy (Best on 4h)
```
period=7, oversold=35.0, overbought=65.0, allow_shorts=True
sl_atr_mult=1.5, rr=1.5, sl_cap=2.00
```

### Bollinger Band + RSI (Most Consistent)
```
rsi_period=10, rsi_oversold=25, rsi_overbought=65
bb_std_dev=1.5, adx_threshold=30, allow_shorts=True
sl_atr_mult=1.5, rr=1.5, sl_cap=2.00
```

### Multi-Indicator Confluence
```
buy_threshold=3, sell_threshold=5, allow_shorts=True
sl_atr_mult=1.5, rr=1.5, sl_cap=2.00
```

### Breakout Strategy
```
bb_period=25, squeeze_lookback=120, volume_multiplier=1.2
allow_shorts=False, sl_atr_mult=1.5, rr=1.5, sl_cap=2.00
```

### MACD Strategy
```
fast_period=8, slow_period=35, signal_period=5
allow_shorts=True, sl_atr_mult=1.5, rr=1.5, sl_cap=2.00
```

### Momentum Confluence
```
fast_ema=8, slow_ema=21, signal_period=9
ema_trend_filter=150, adx_strength_threshold=20, adx_weak_threshold=15
allow_shorts=True, sl_atr_mult=1.5, rr=1.5, sl_cap=2.00
```

---

## Key Observations

### Timeframe Effects
1. **Daily (1d):** Fewest trades, highest per-trade quality. SMA Crossover dominates with tight 20/21 period cross — effectively a momentum filter.
2. **4-Hour (4h):** Sweet spot for RSI — 84.2% WR with 10.74 profit factor. Only 19 trades but extremely high conviction.
3. **15m–30m:** BB+RSI's best range. Enough signal frequency for statistical reliability while maintaining 61–64% WR.
4. **5-Minute (5m):** Highest trade counts. RSI and Multi-Indicator are the only profitable strategies. Noise challenges most trend-following approaches.

### Strategy Analysis
- **RSI:** Universally effective. Optimized to tighter bands (35/65 vs default 30/70) and shorter period (7 vs 14). Profitable on every timeframe except 15m.
- **BB+RSI:** Most robust cross-timeframe performer. Narrower bands (1.5σ vs 2.0σ) and lower RSI thresholds (25/65).
- **SMA Crossover:** Only works on daily. The tight 20/21 cross is essentially a short-term momentum signal.
- **MACD:** Generally poor except on 15m (long-only). Wider slow period (35 vs 26) helps filter noise.
- **Momentum Confluence:** Too many required conditions. Nearly zero win rates on higher timeframes — the AND logic is too restrictive.

### Short Selling
- Short-selling hurts on higher timeframes (4h, 1d) — long-only won there
- Short-selling helps on 5m (RSI, Multi-Indicator) where mean-reversion works both ways
- BB+RSI profits from shorts on 15m–30m, contributing significantly to overall returns

### Adapter Parameters
All optimized strategies converged on identical adapter settings:
- `sl_atr_mult=1.5` — moderate stop distance
- `rr=1.5` — slight positive risk/reward
- `sl_cap=$2.00` — reasonable per-share cap for SPY

This suggests the adapter defaults are already well-calibrated for SPY.

---

## Implications for Signal Stacking

These results motivate the signal stacking system:

1. **Indicator overlap is real:** RSI appears in RSI, BB+RSI, and Multi-Indicator strategies. MACD appears in MACD, Momentum, and Multi-Indicator. Decoupling into atomic generators enables testing combinations that the monolithic strategies can't express.

2. **Cross-timeframe confluence:** RSI on 4h + BB on 15m might combine the conviction of HTF signals with the timing precision of LTF signals. The optimizer can't test this; only the combinatorial tester can.

3. **The AND problem:** Momentum Confluence's poor performance shows that requiring ALL conditions simultaneously is too restrictive. The stacking engine's multiple aggregation modes (especially MAJORITY_VOTE and WEIGHTED_AVERAGE) allow softer confluence without requiring unanimity.

4. **Parameter sensitivity:** Most strategies improved with tighter indicator thresholds (RSI 35/65, BB 1.5σ). The atomic generators preserve this tuning in their `param_space` for the combinatorial search.

---

*Generated from `multi-timeframe-optimizer` CLI output. 1,080 configurations tested in 42.9 minutes.*
