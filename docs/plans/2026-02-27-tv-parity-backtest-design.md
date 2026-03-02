# TV-Parity Config Backtest Design

**Date:** 2026-02-27
**Goal:** Run existing Python strategies with configs matched to PineScript v11.2 defaults and compare against TradingView trade export.

## Context

TradingView VWAP v11.2 produced 107 trades from Feb 2025 to Feb 2026 on SPY 5m data:
- Cumulative P&L: ~$10,892 (+10.89% on $100k)
- Mode mix: PB (~65 trades), PS (~18), ORB (~7), REV (~5), ORR (disabled)
- Heavy short bias on PB entries

Previous Python mega backtest (543 combinations) found 0 strategies beating buy-and-hold (+37.14%), but was run with default configs that are significantly more conservative than the PineScript settings.

## Root Cause Analysis

Key gaps between PineScript v11.2 defaults and Python defaults:

| Setting | PineScript v11.2 | Python Default | Impact |
|---------|-----------------|----------------|--------|
| allow_shorts (PB) | True | **False** | Misses ~70% of PB trades |
| max_day (PB) | 2 | 1 | Half the trade frequency |
| spacing (PB) | 3 | 5 | Misses close re-entries |
| rr (PB) | 1.4 | 1.8 | TP too far, gets stopped |
| be_trigger | 0.5R | 0.7R | Slower to protect |
| ar_filter | Enabled | Disabled | Enters bad vol regimes |
| va_filter | Enabled | Disabled | Enters decelerating VWAP |
| no_monday_long | True | False | Takes weak Monday longs |
| w_vol (PB) | 3 (inverted) | 2 | Scoring mismatch |
| min_score | 3 | 4 | Too selective |
| ps_atr_pct | 30% | 20% | PS fires too often |
| ps_window | 12 | 35 | PS window too wide |
| rev_band | 2σ | 1σ | REV band too tight |

## Approach

No code changes. Create a single backtest script that instantiates existing strategies with TV-parity config overrides.

### PB Config Overrides
```python
allow_shorts=True, max_day=2, spacing=3, rr=D("1.4"),
be_trigger=D("0.5"), ar_filter=True, ar_cap=D("1.15"),
va_filter=True, no_monday_long=True, min_score=3,
min_score_long=5, w_vol=3, pb_body=D("0.15"), max_vxc=6,
sl_cap=D("1.50"), sl_atr=D("1.3"), trail_buf=D("0.15")
```

### PS Config Overrides
```python
allow_shorts=True, ps_atr_pct=D("30.0"), ps_window=12,
ps_rvol=D("1.0"), ps_engulf=D("0.35"), ps_sl_cap=D("2.50"),
ps_sl_atr=D("1.5"), ps_sma_filter=False, be_trigger=D("0.5")
```

### ORB Config Overrides
```python
allow_shorts=True, orb_window=20, orb_sl_mode="OR Opposite",
orb_sl_cap=D("2.50"), orb_trail_atr=D("1.5"),
orb_body_min=D("0.2"), be_trigger=D("0.5"), adx_thresh=D("21")
```

### REV Config Overrides
```python
rev_band="2σ", rev_body=D("0.20"), rev_shorts=False,
rev_min_rr=D("0.3"), be_trigger=D("0.5"), min_score=3
```

### ORR: Disabled (matches TV v11.2 i_orrEnable=false)

## Expected Output

1. Per-strategy metrics (return, trades, win rate, Sharpe, max DD)
2. Combined metrics (summing all strategy P&Ls)
3. Comparison table vs TV benchmark
4. Results CSV at `data/SPY/tv_parity_backtest_results.csv`

## Success Criteria

- If combined Python P&L is within ±30% of TV's $10,892: configs explain the gap
- If still >30% off: cross-mode interaction (unified strategy) needed next
