# Exit Mechanism Tournament — Analysis Report

## Executive Summary

Simulated 8 exit mechanisms on identical trade entries using 5-minute SPY bar data (60 sessions, Nov 13 2025 — Feb 10 2026). Two mechanisms consistently outperform the current 0.5R trail:

| Mechanism | Advantage | Tradeoff |
|-----------|-----------|----------|
| **ATR_TRAIL** (0.3×ATR from peak) | +31% total P&L, 83% median capture vs 57% | Same max loss profile |
| **HYBRID** (ATR trail + VWAP cross) | +72% Sharpe, 81% WR, -41% max loss | Clips some large winners |

The Zarattini/Maróy VWAP trailing stop — the academically-motivated hypothesis — **does not work for pullback entries**. It's structurally incompatible because pullback trades enter near VWAP, triggering premature exits on any whipsaw.

## What This Is (and Isn't)

**This is:** An out-of-sample simulation of exit alternatives on real 5-minute price paths for trades the strategy actually took. The entries come from TradingView's strategy tester; the bar-by-bar price evolution comes from Yahoo Finance. No look-ahead bias in the exit logic.

**This isn't:** A backtest replacement. Key limitations:
- 31 curated trades (v10.9.5) / 110 unfiltered trades (DIAG) — small sample
- Stop distance approximated from ATR (calibrated to actual losses, but not exact)
- Simulated WR 67.7% vs actual 75% — ~7% gap from intra-bar fill dynamics
- No modeling of spread, slippage, or options premium specifics
- 60-day window may not represent all market regimes

The direction of the findings is likely correct; the magnitude should be treated as approximate.

## Mechanism Details

### CURRENT_TRAIL (baseline)
- Hard stop at -1R
- After +1R profit: move stop to breakeven + 5% buffer
- Trail at 0.5R below peak

### ATR_TRAIL (best total P&L)
- Hard stop at -1R
- After +1R profit: trail at 0.3×ATR(14) from peak
- Key difference: trail distance adapts to current volatility rather than being fixed at 0.5R

### HYBRID (best risk-adjusted)
- Hard stop at -1R
- After +0.5R profit: activate both ATR trail AND VWAP cross exit
- Exit at whichever triggers first
- The VWAP cross rarely fires (1/31 on v10.9.5) — it acts as catastrophic insurance
- The early activation (0.5R) with ATR trail is the primary driver

### VWAP_CROSS (Zarattini/Maróy hypothesis — FAILED)
- Exit when close crosses VWAP against position
- On v10.9.5: 42% WR, positive P&L only because of a few large winners
- On DIAG: negative total P&L (-$9.10)
- Structural problem: pullback entries start near VWAP, so any small adverse move triggers exit

## Key Finding: Why ATR_TRAIL Works

The current 0.5R trail is a fixed distance from peak regardless of market conditions. On a low-volatility session, 0.5R might be too wide (leaving money on the table). On a high-volatility session, 0.5R might be too tight (getting stopped out of a continuation move by normal noise).

ATR_TRAIL at 0.3×ATR adapts: when volatility is low, the trail tightens to capture more of the move. When volatility is high, it widens to give the trade room. Median capture jumps from 57% to 83% — the trail captures roughly 25% more of the favorable excursion on winning trades.

## Honest Assessment of Limitations

1. **Small sample risk.** 31 trades is directionally useful but not statistically conclusive. At 31 trades, the standard error on win rate is ±8.4%. The difference between 67.7% and 80.6% WR (CURRENT vs HYBRID) barely clears this threshold.

2. **Extrapolation danger.** The projected +$3,175 from ATR_TRAIL assumes the 60-day window is representative of the full 6-month backtest. If this window happened to favor one exit style (e.g., strong trends → trail exits shine), the extrapolation overstates the benefit.

3. **Simulation ≠ Pine Script.** The actual Pine Script has nuances (exact stop placement, commission model, bar timing) that this Python simulation approximates. The only way to confirm is to implement ATR_TRAIL in Pine Script and run the real backtest.

4. **Selection bias in the DIAG sample.** The 110 DIAG trades include many lower-quality entries that the v10.9.5 filter stack would reject. The mechanisms may rank differently on curated vs unfiltered trades (and they do — HYBRID ranks higher on DIAG but ATR_TRAIL ranks higher on v10.9.5).

5. **The HYBRID's high WR may be an artifact.** 80.6% WR on 31 trades could include regime-specific luck. On 110 DIAG trades it drops to 64.5% — still strong but less dramatic.

## Recommendation

**Implement ATR_TRAIL (0.3×ATR from peak after 1R BE trigger) as a Pine Script A/B test against the current 0.5R trail on the v10.9.5 baseline.** This is the lowest-risk change:
- Same activation logic (1R BE trigger)
- Same stop mechanics (trail from peak)
- Only the trail distance changes (0.5R fixed → 0.3×ATR adaptive)
- Easy to validate: the backtest should show similar WR with higher avg winner size

If ATR_TRAIL confirms in Pine Script, then consider HYBRID as a second iteration — adding the VWAP cross exit as a supplementary condition would require more careful implementation and testing.

Do NOT implement pure VWAP trailing stops. The academic evidence for them is specific to breakout/momentum entries. For pullback mean-reversion entries, they are counterproductive.
