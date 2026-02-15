# Grand Tournament Results -- Institutional Strategy Evaluation

## Executive Summary

Evaluated 12 strategy candidates across 5 walk-forward validation windows on SPY 5-minute bar data (501 trading sessions, ~40,740 bars). Uses `score_v2` (Sortino, Calmar, P&L penalty, consecutive loss penalty) for institutional-grade scoring. Total runtime: 17.8 seconds.

**Overall winner:** Stack:macd+obv -- OOS +74.03, IS +76.88, degradation 0.96.

Key findings:
- The MACD+OBV signal stack is the most robust strategy, with near-perfect IS/OOS consistency (degradation 0.96)
- 3 out of 12 strategies flagged as likely overfitters (degradation < 0.5 with positive IS)
- Daily strategies with simple indicators (RSI, MACD, SMA) outperform complex multi-indicator approaches
- Momentum Confluence and Multi-Indicator Confluence consistently fail on out-of-sample data
- Walk-forward validation successfully identifies strategies that look good in-sample but fail out-of-sample

---

## Methodology

### Walk-Forward Validation

Each strategy was tested using **5 rolling walk-forward windows** with 70/30 in-sample/out-of-sample splits:

| Window | In-Sample | Out-of-Sample |
|--------|-----------|---------------|
| 0 | bars 0-5702 | bars 5702-8146 |
| 1 | bars 2444-8146 | bars 8146-10590 |
| 2 | bars 4888-10590 | bars 10590-13034 |
| 3 | bars 7332-13034 | bars 13034-15478 |
| 4 | bars 9776-15478 | bars 15478-17922 |

### Scoring (score_v2)

Institutional-grade composite fitness score:
- **Sortino ratio** (30% weight) -- penalizes downside volatility only
- **Calmar ratio** (15%) -- total return / max drawdown
- **Win rate** (20%) -- percentage of winning trades
- **Profit factor** (15%) -- gross profit / gross loss
- **Max drawdown penalty** (20%) -- penalizes large drawdowns
- **P&L floor** -- massive penalty for losing money
- **Consecutive loss penalty** -- penalizes 3+ consecutive losses

### Degradation Ratio

`degradation_ratio = OOS_score / IS_score`

| Ratio | Interpretation |
|-------|---------------|
| > 1.5 | GOOD -- strategy generalizes well |
| 0.5-1.5 | OK -- moderate generalization |
| < 0.5 (IS > 0) | OVER -- likely overfitting |
| negative | Severe overfitting or sign flip |

---

## Grand Ranking (by Out-of-Sample Score)

| Rank | OOS Score | IS Score | Degrade | Flag | Strategy |
|------|-----------|----------|---------|------|----------|
| 1 | +74.03 | +76.88 | 0.96 | OK | Stack:macd+obv |
| 2 | +65.29 | +25.94 | 2.52 | GOOD | Daily:SMA Crossover |
| 3 | +62.80 | +40.07 | 1.57 | GOOD | Daily:MACD Strategy |
| 4 | +45.91 | +76.67 | 0.60 | OK | Daily:RSI Strategy |
| 5 | +32.57 | -5.47 | -5.96 | OK | Stack:macd+obv+volume_surge |
| 6 | +22.83 | +87.43 | 0.26 | OVER | Daily:Bollinger Band + RSI |
| 7 | -21.54 | +48.51 | -0.44 | OVER | Stack:macd+obv+stochastic |
| 8 | -83.85 | +14.46 | -5.80 | OVER | Stack:obv+rsi+sma_cross |
| 9 | -108.25 | -4.45 | 24.31 | GOOD | Stack:macd+obv+ema_trend |
| 10 | -124.40 | -75.68 | 1.64 | GOOD | Daily:Breakout Strategy |
| 11 | -149.52 | -74.73 | 2.00 | GOOD | Daily:Multi-Indicator Confluence |
| 12 | -165.66 | -172.41 | 0.96 | NEG | Daily:Momentum Confluence |

---

## Strategy Candidates

### Daily Strategies (7)

Wrapped via `DailyToIntradayAdapter` with optimized parameters from the optimizer tournament (1,080 configs). All use `sl_atr_mult=1.5`, `rr=1.5`, `sl_cap=$2.00`.

| Strategy | Parameters | Shorts |
|----------|-----------|--------|
| SMA Crossover | short=20, long=21 | No |
| RSI | period=7, oversold=35, overbought=65 | Yes |
| MACD | fast=8, slow=35, signal=5 | Yes |
| Bollinger Band + RSI | rsi_period=10, rsi_os=25, rsi_ob=65, bb_std=1.5, adx_th=30 | Yes |
| Breakout | bb_period=25, squeeze_lb=120, vol_mult=1.2 | No |
| Momentum Confluence | fast_ema=8, slow_ema=21, signal=9, trend=150, adx_str=20, adx_weak=15 | Yes |
| Multi-Indicator | buy_th=3, sell_th=5 | Yes |

### Signal Stacks (5)

Top combinations from the signal stack tournament (624 configs). All use `M5` timeframe, `weighted_average` aggregation, `buy_threshold=0.2`, `sell_threshold=0.3`.

| Stack | Generators |
|-------|-----------|
| macd+obv | MACD + On-Balance Volume |
| macd+obv+stochastic | MACD + OBV + Stochastic Oscillator |
| macd+obv+volume_surge | MACD + OBV + Volume Surge |
| obv+rsi+sma_cross | OBV + RSI + SMA Crossover |
| macd+obv+ema_trend | MACD + OBV + EMA Trend |

---

## Analysis

### Top 4 Robust Strategies

1. **Stack:macd+obv (OOS +74.03, degradation 0.96)**
   - The MACD+OBV combination produces the highest out-of-sample score with near-perfect generalization
   - Degradation ratio of 0.96 means it performs almost identically on unseen data
   - This confirms the signal stack tournament finding that OBV anchors the best combos
   - Two atomic signals from different categories (momentum + volume) create complementary coverage

2. **Daily:SMA Crossover (OOS +65.29, degradation 2.52)**
   - The tight 20/21 SMA cross acts as a momentum filter on daily timeframe
   - OOS actually outperforms IS (degradation > 1), suggesting a conservative strategy that benefits from more data
   - Long-only configuration avoids the noise of short signals

3. **Daily:MACD Strategy (OOS +62.80, degradation 1.57)**
   - Wide slow period (35 vs default 26) filters noise effectively
   - Good generalization with 1.57 degradation ratio

4. **Daily:RSI Strategy (OOS +45.91, degradation 0.60)**
   - Strong IS performance (+76.67) with moderate OOS degradation
   - Tighter bands (35/65 vs 30/70) and shorter period (7 vs 14) remain effective

### Overfitters Detected

- **Daily:Bollinger Band + RSI**: IS +87.43 -> OOS +22.83 (degradation 0.26). The highest in-sample scorer collapses significantly out-of-sample. Too many parameters (rsi_period, rsi_os, rsi_ob, bb_std, adx_threshold) create overfitting risk.

- **Stack:macd+obv+stochastic**: IS +48.51 -> OOS -21.54. Adding stochastic to the winning macd+obv pair actually hurts -- the third generator adds complexity without improving robustness.

- **Stack:obv+rsi+sma_cross**: IS +14.46 -> OOS -83.85. Swapping MACD for RSI and SMA cross with OBV produces a combination that doesn't generalize.

### Strategies That Consistently Fail

- **Momentum Confluence** (OOS -165.66): Requires too many conditions simultaneously (MACD + EMA + ADX all aligning). The AND logic is too restrictive for any regime.

- **Multi-Indicator Confluence** (OOS -149.52): Similar problem -- too many required indicators reduce trade frequency to near-zero, and the few trades that pass all filters aren't high-quality enough.

### Key Insights

1. **Simplicity wins**: The top 3 strategies use 2-3 indicators max. Adding more generators or parameters tends to overfit.

2. **OBV is the anchor**: Every positive-OOS signal stack includes OBV. Volume confirmation is critical for filtering false signals.

3. **Walk-forward validation is essential**: Without it, Bollinger Band + RSI would appear to be the best strategy (IS +87.43). Walk-forward reveals it drops to +22.83 out-of-sample.

4. **Degradation ratio is the anti-overfitting metric**: Strategies with ratio 0.8-1.2 (macd+obv at 0.96) are the most trustworthy. Ratios below 0.5 signal danger.

5. **Signal stacks can beat daily strategies**: Stack:macd+obv outperforms all 7 daily strategies on out-of-sample data, validating the signal stacking architecture.

---

## Comparison: Optimizer Tournament vs Grand Tournament

| Strategy | Optimizer P&L | Grand OOS Score | Status |
|----------|--------------|-----------------|--------|
| SMA Crossover (1d) | +$35,918 | +65.29 | Confirmed robust |
| RSI (4h) | +$29,500 | +45.91 | Confirmed, moderate OOS drop |
| BB+RSI (15m) | +$17,702 | +22.83 | EXPOSED as overfitter |
| MACD (15m) | +$9,860 | +62.80 | Confirmed, stronger than expected |
| Multi-Indicator (1d) | +$15,584 | -149.52 | EXPOSED (1 trade in optimizer) |

The grand tournament overturns the optimizer ranking. BB+RSI and Multi-Indicator, which looked good in the optimizer, are exposed by walk-forward validation.

---

## Configuration

- Initial capital: $100,000
- Risk per trade: 1%
- Walk-forward windows: 5
- IS/OOS ratio: 70/30
- Scoring: score_v2 (Sortino + Calmar + WR + PF - DD - penalties)
- Data: SPY 5-minute bars, 501 trading sessions

---

*Generated from `output/grand_tournament.log`. 12 strategies evaluated with walk-forward validation in 17.8 seconds.*
