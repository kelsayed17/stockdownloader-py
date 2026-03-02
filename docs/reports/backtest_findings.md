# Strategy Backtest Findings — Parameter Optimization Report

## Test Conditions

- **Instrument:** SPY (SPDR S&P 500 ETF)
- **Data:** 40,740 five-minute bars across 501 trading sessions
- **Period:** 2024-02-12 to 2026-02-10 (~2 years)
- **Capital:** $100,000 | **Risk/trade:** 1% | **Daily loss limit:** 3%
- **Scoring:** Composite of Sharpe (40%), win rate (20%), profit factor (20%), drawdown penalty (20%), plus trade-count adjustments (see `optimizer_scoring.py`)

All numbers below are from the same SPY dataset. The greedy-sequential optimizer tests one parameter at a time, keeping the best value before moving to the next parameter. "Score" is the composite fitness from `optimizer_scoring.score()`.

---

## 1. Baseline Strategy Ranking

Ran all 8 strategies with their v11.2 / registry default parameters. Daily strategies run through `DailyToIntradayAdapter` on 5-minute bars.

| Rank | Strategy | P/L | Return | Win Rate | Profit Factor | Sharpe | Max DD | Trades | Score |
|-----:|----------|----:|-------:|---------:|--------------:|-------:|-------:|-------:|------:|
| 1 | VWAP v11.2 | +$5,128 | 5.13% | 57.39% | 1.65 | 1.21 | 1.89% | 115 | 106.8 |
| 2 | Bollinger+RSI | +$29,879 | 29.88% | 68.64% | 1.35 | 1.15 | 16.40% | 338 | 98.2 |
| 3 | Multi-Indicator | +$23,147 | 23.15% | 32.95% | 1.24 | 1.01 | 7.37% | 516 | 83.2 |
| 4 | Breakout | +$14,943 | 14.94% | 44.93% | 1.23 | 0.82 | 8.80% | 276 | 79.6 |
| 5 | MACD | +$23,277 | 23.28% | 39.12% | 1.13 | 0.96 | 12.76% | 1,710 | 76.3 |
| 6 | SMA Crossover | +$19,236 | 19.24% | 39.98% | 1.14 | 0.85 | 13.85% | 1,068 | 72.1 |
| 7 | Momentum Confluence | +$1,712 | 1.71% | 44.71% | 1.17 | 0.38 | 3.96% | 170 | 61.5 |
| 8 | RSI (alone) | -$1,939 | -1.94% | 63.40% | 0.98 | -0.02 | 20.41% | 470 | 40.8 |

**Observations:**

- 7 of 8 strategies are profitable over the 2-year window.
- VWAP ranks #1 on risk-adjusted score despite modest absolute return. It has the lowest max drawdown (1.89%), highest Sharpe (1.21), and highest profit factor (1.65).
- Bollinger+RSI has the highest absolute return and win rate but incurs 16.40% drawdown.
- RSI alone is the only loser: average loss ($510) dwarfs average win ($288) despite 63% WR.

### VWAP mode distribution

| Mode | Trades | Share |
|------|-------:|------:|
| PB (Pullback) | 58 | 50.43% |
| PS (Pattern Scalp) | 25 | 21.74% |
| ORB (Opening Range Breakout) | 17 | 14.78% |
| REV (Reversal) | 15 | 13.04% |

---

## 2. VWAP Parameter Sensitivity

Tested 88 individual parameter variations against the baseline. Each test changes one parameter and measures score delta.

### Parameters that improved score (sorted by impact)

| Parameter | Default | Best | Default Score | Best Score | Delta | Mechanism |
|-----------|---------|------|------:|-----:|------:|-----------|
| `adx_thresh` | 21 | **30** | 106.8 | **155.8** | +49.0 | Restricts PB entries to strongly trending markets |
| `pb_zone` | 0.5 | **0.3** | 106.8 | **125.0** | +18.2 | Tighter pullback zone — only enters very close to VWAP |
| `sl_cap` | $1.50 | **$1.00** | 106.8 | **120.4** | +13.6 | Hard dollar cap on stops — prevents outsized single-trade losses |
| `min_score_long` | 5 | **6** | 106.8 | **115.6** | +8.8 | Higher confluence threshold for longs — fewer but better entries |
| `be_trigger` | 0.5 | **0.3** | 106.8 | **114.0** | +7.2 | Earlier breakeven move — protects capital faster |
| `sl_atr` | 1.3 | **1.0** | 106.8 | **110.2** | +3.4 | Tighter ATR-based stop — reduces risk per trade |
| `spacing` | 3 | **5** | 106.8 | **109.3** | +2.5 | More bars between entries — avoids cluster-trading |
| `rev_shorts` | False | **True** | 106.8 | **109.7** | +2.9 | Allow short reversals at upper bands |
| `orr_enable` | False | **True** | 106.8 | **108.4** | +1.6 | Enable opening-range reversals |
| `no_friday_short` | True | **False** | 106.8 | **108.6** | +1.8 | Allow Friday shorts — they appear profitable in SPY |
| `orb_rvol` | 2.0 | **2.5** | 106.8 | **108.4** | +1.6 | Higher relative-volume bar for ORB — better quality breakouts |
| `max_day` | 2 | **1** | 106.8 | **107.7** | +0.9 | One trade per day — forces selectivity |
| `min_score` | 3 | **2** | 106.8 | **107.5** | +0.7 | Slightly lower short entry threshold |
| `orb_trail_atr` | 1.5 | **2.0** | 106.8 | **107.0** | +0.2 | Wider ORB chandelier trail — gives breakouts more room |

### Parameters with no significant impact

| Parameter | Values Tested | Observation |
|-----------|--------------|-------------|
| `rr` | 1.0, 1.2, 1.6, 1.8, 2.0, 2.5 | All scored within 3 points of default (1.4). Current value is near-optimal. |
| `trail_buf` | 0.10, 0.20, 0.30 | Identical results. Trail buffer has no measurable effect at these levels. |
| `pb_body` | 0.10, 0.20, 0.25 | Within 3 points. Default 0.15 is near-optimal. |
| `rev_body` | 0.15, 0.25 | Within 4 points. Default 0.20 is near-optimal. |
| `no_monday_long` | False | Slightly worse. Monday long filter is correctly enabled. |

### Parameters where the best value was "disable"

| Parameter | Default | Observation |
|-----------|---------|-------------|
| `rev_enable` | True | **Disabling scores 107.4 vs 106.8** — reversal trades are marginal. However, with `rev_shorts=True` enabled, reversals become net-positive. The two settings interact. |

### Greedy-validated optimal config

The greedy-sequential optimizer applies the best single-parameter values one at a time, accepting only changes that improve the running score:

```
sl_atr      = 1.0    (was 1.3)   — accepted, score 110.2
be_trigger  = 0.3    (was 0.5)   — accepted, score 112.2
min_score   = 2      (was 3)     — accepted, score 112.9
min_score_long = 6   (was 5)     — accepted, score 118.5
adx_thresh  = 30     (was 21)    — accepted, score 157.4
max_day     = 1      (was 2)     — accepted, score 158.0
```

Six parameters were rejected because they conflicted with the already-applied changes. This is expected in greedy optimization — the independently best value for `rev_enable` (False) hurts when `adx_thresh=30` is already active because the high ADX filter already eliminates weak-trend reversals.

**Greedy-optimal result:** P/L=$6,404 | WR=67.44% | PF=2.61 | Trades=86 | Score=158.0

vs **Baseline:** P/L=$5,128 | WR=57.39% | PF=1.65 | Trades=115 | Score=106.8

Delta: **+$1,276 P/L**, **+10% win rate**, **+0.96 profit factor**, **-29 trades** (more selective).

### Why `adx_thresh=30` is the biggest lever

The default `adx_thresh=21` accepts pullback entries whenever ADX is above 21, which includes weak trends and choppy ranges. Raising to 30 restricts entries to **strongly trending sessions only**. This filters out the trades that look like pullbacks but are actually range noise. Profit factor jumps from 1.65 to 2.46 on this single change because losers are eliminated more aggressively than winners.

The tradeoff: trade count drops from 115 to 90 (22% fewer trades). In a production setting, this may mean fewer trades per week but substantially better quality.

---

## 3. Daily Strategy Optimization

Optimized the five fastest daily strategies (skipped Momentum Confluence at 390s/run and Multi-Indicator at 1,368s/run). Each optimization searches strategy constructor parameters then adapter parameters (`sl_atr_mult`, `rr`, `sl_cap`).

### RSI Strategy: from -$1,939 to +$39,643

| Parameter | Default | Optimized |
|-----------|---------|-----------|
| `period` | 14 | **10** |
| `oversold` | 30.0 | **25.0** |
| `overbought` | 70.0 | **80.0** |

**Why it works:** The default RSI(14) with 30/70 thresholds generates too many signals in trending markets where mean-reversion fails. The optimized config:
- **Shorter period (10):** responds faster to price changes, fewer stale signals.
- **Lower oversold (25):** requires deeper oversold condition before buying — eliminates premature entries in downtrends.
- **Higher overbought (80):** holds winners longer, only exiting on extreme overbought — captures more of the up-move.

**Key takeaway:** RSI alone should never use the textbook 30/70 thresholds for intraday trading. The asymmetric 25/80 setup recognizes that SPY has a long bias — you want to buy deeper dips (25) and hold longer rallies (80).

### SMA Crossover: from +$19,236 to +$42,195

| Parameter | Default | Optimized |
|-----------|---------|-----------|
| `short_period` | 9 | **20** |
| `long_period` | 21 | **30** |

**Why it works:** The default 9/21 crossover generates 1,068 trades — far too many signals on 5-minute data. The 20/30 pair reduces trades to 732 (31% fewer) while increasing P/L by 119%. The longer periods filter out intra-day noise and only trigger on meaningful trend shifts.

### Bollinger+RSI: from +$29,879 to +$36,345

| Parameter | Default | Optimized |
|-----------|---------|-----------|
| `bb_period` | 20 | **25** |
| `rsi_oversold` | 30 | **35** |
| `rsi_overbought` | 70 | **75** |
| `adx_threshold` | 25 | **20** |

**Why it works:** Wider BB period (25) catches more meaningful band touches. The slightly wider RSI thresholds (35/75) enter earlier in the reversion. The lower ADX threshold (20) allows more trades in mild trends — this strategy is mean-reversion, so it works even when ADX is moderate. Trade count drops from 338 to 230 (32% fewer) with a +$6,466 P/L improvement.

### Breakout: from +$14,943 to +$27,859

| Parameter | Default | Optimized |
|-----------|---------|-----------|
| `volume_multiplier` | 1.5 | **1.2** |

**Why it works:** The default requires 1.5x average volume for a breakout signal. Lowering to 1.2x catches more early breakouts before volume fully spikes. Trade count increases modestly (276 to 318) while P/L nearly doubles. The lower threshold still provides meaningful volume confirmation but doesn't wait for the entire herd.

### MACD: from +$23,277 to +$24,566

| Parameter | Default | Optimized |
|-----------|---------|-----------|
| `fast_period` | 12 | **8** |
| `slow_period` | 26 | **30** |
| `signal_period` | 9 | **7** |

**Why it works:** Marginal improvement (+$1,289). The faster fast EMA (8) and shorter signal (7) detect crossovers earlier, while the longer slow EMA (30) provides a more stable reference. However, MACD is inherently a high-frequency signal generator on 5-minute data (2,111 trades), limiting its per-trade profitability.

### Adapter parameters (shared across daily strategies)

The adapter wraps daily strategies for intraday backtesting, adding ATR-based stops and take-profits. The default adapter parameters (`sl_atr_mult=1.5`, `rr=1.5`, `sl_cap=$2.00`) were tested alongside strategy params. No universal "best" adapter setting emerged — each strategy has its own optimal adapter values absorbed into the results above.

---

## 4. Indicator Synergy Analysis

### Which indicators appear in profitable strategies?

| Indicator | In Profitable | In Losing | Assessment |
|-----------|:------------:|:---------:|------------|
| ADX(14) | 4 | 0 | Best standalone filter. Determines trend vs range. |
| BB(20,2) | 3 | 0 | Strong volatility envelope. Mean-reversion anchor. |
| MACD | 2 | 0 | Momentum signal. Needs trend filter for reliability. |
| Volume/RVOL | 2 | 0 | Entry quality gate. Confirms conviction. |
| RSI(14) | 2 | **1** | **Context-dependent.** Loses money alone; wins with BB+ADX. |

### Effective indicator pairings (from strategy results)

1. **VWAP + ATR + ADX:** The core intraday framework. VWAP provides dynamic bias, ATR sizes stops adaptively, ADX filters for trending sessions. This trio powers the best risk-adjusted strategy (Sharpe 1.21, PF 1.65, MaxDD 1.89%).

2. **BB + RSI + Stochastic + ADX:** The mean-reversion trifecta with trend filter. Achieves the highest win rate (68.64%) by requiring three-way confirmation (BB touch, RSI extreme, Stochastic extreme) plus an ADX-based range-bound filter. The ADX filter is critical — without it, the strategy enters mean-reversion trades during strong trends, which fail.

3. **EMA(fast/slow) + MACD:** Momentum chain. Produces many signals but modest per-trade edge. Best used as a filter, not a primary trigger.

4. **BB Squeeze + Volume:** Volatility breakout with volume confirmation. The squeeze identifies compression; volume confirms the expansion is real. Lower volume threshold (1.2x) catches breakouts earlier.

### Why RSI alone fails

RSI(14) with 30/70 thresholds generates 470 trades over 501 sessions (~1 per day). The problem is structural:
- **Average win: $288** — the strategy exits at the overbought threshold, capping upside.
- **Average loss: $510** — the adapter's ATR-based stop is too wide for RSI's quick mean-reversion thesis.
- Result: even at 63% win rate, the strategy loses money because `0.63 * 288 - 0.37 * 510 = -7.5 per trade`.

**Fix:** Either combine RSI with BB+ADX (as in Bollinger+RSI) or use asymmetric thresholds (25/80) to widen the profit window. Both approaches were validated in the optimization.

---

## 5. Summary of Best Parameters

### VWAP v11.2 — Greedy-Optimized

```python
VwapStrategyConfig(
    sl_atr=Decimal("1.0"),        # was 1.3 — tighter ATR stops
    be_trigger=Decimal("0.3"),    # was 0.5 — earlier breakeven
    min_score=2,                  # was 3   — slightly more short entries
    min_score_long=6,             # was 5   — stricter long confluence
    adx_thresh=Decimal("30"),     # was 21  — only trade strong trends
    max_day=1,                    # was 2   — one trade per day
    # All other params: v11.2 defaults
)
```

### Daily Strategies — Optimized Defaults

```python
# RSI (biggest improvement: -$1,939 -> +$39,643)
RSIStrategy(period=10, oversold=25.0, overbought=80.0)

# SMA Crossover (+$19,236 -> +$42,195)
SMACrossoverStrategy(short_period=20, long_period=30)

# Bollinger+RSI (+$29,879 -> +$36,345)
BollingerBandRSIStrategy(bb_period=25, rsi_oversold=35, rsi_overbought=75, adx_threshold=20)

# Breakout (+$14,943 -> +$27,859)
BreakoutStrategy(volume_multiplier=1.2)

# MACD (+$23,277 -> +$24,566)
MACDStrategy(fast_period=8, slow_period=30, signal_period=7)
```

---

## 6. Caveats

1. **Overfitting risk.** These parameters are optimized on a single 2-year SPY dataset. The greedy optimizer finds the best-performing values on this specific data; they may not generalize to other periods, instruments, or market regimes. The optimizer does not use walk-forward validation or out-of-sample testing.

2. **SPY-specific.** SPY is the most liquid ETF with tight spreads and a well-documented long bias. These parameters may not transfer to individual stocks, lower-liquidity ETFs, or instruments with different volatility profiles.

3. **Simulation, not live trading.** The intraday backtest engine does not model slippage, partial fills, or options premium dynamics. The adapter's position sizing is based on risk-per-share without accounting for spread costs.

4. **The scoring function has opinions.** The composite score penalizes configs below 80 trades and favors ~0.5 trades/day. A different scoring function (e.g., pure Sharpe, or pure P/L) would produce different "best" parameters. The current scoring reflects a preference for statistically meaningful results with moderate trading frequency.

5. **Greedy ≠ global optimum.** The greedy-sequential optimizer does not explore the full parameter space. Some parameter combinations that are individually suboptimal may be collectively superior. The optimizer mitigates this by testing combined winners, but it cannot guarantee the global optimum.

6. **Daily strategies via adapter.** The `DailyToIntradayAdapter` evaluates daily strategies on every 5-minute bar, which generates far more signals than running them on daily bars. The optimized parameters reflect this higher-frequency context and may not be appropriate for end-of-day trading.

---

## 7. How to Reproduce

```bash
# Quick baseline (all 8 strategies, ~30 min):
python scripts/holistic_backtest.py --csv data/spy_5m_bars.csv --quick

# Full analysis with optimization (~60 min):
python scripts/holistic_backtest.py --csv data/spy_5m_bars.csv

# VWAP sensitivity + daily optimization (~30 min):
python scripts/optimize_targeted.py

# Built-in optimizer (VWAP + all daily, full search):
python -m stockdownloader.app.intraday_backtest_app --csv data/spy_5m_bars.csv --all-strategies

# Single strategy:
python -m stockdownloader.app.intraday_backtest_app --csv data/spy_5m_bars.csv --strategy rsi
```
