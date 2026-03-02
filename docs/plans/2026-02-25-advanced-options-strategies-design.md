# Advanced Options Strategies Design

## Current State

ML Combined: +55.3% (Sharpe 1.62, MaxDD -12.3%) vs Buy & Hold +77.7%

Gap: 22 points on total return. Risk-adjusted return is strong but drawdown is high.

## Goal

Improve both total return AND risk-adjusted return simultaneously. Close the gap with buy-and-hold while reducing drawdown and improving Sharpe.

## Three Strategies

### 1. Protective Put Collar

Add a cheap OTM protective put purchase each week on all held shares. Turns the combined position into a collar: capped upside (CC), protected downside (put), plus CSP income on idle cash.

**Engine changes:**
- `collar: bool = False` and `hedge_delta: float = 0.10` parameters on `WheelBacktestEngine`
- Each week when collar enabled: buy protective put (10-delta OTM) on all held shares
- Deduct put cost from cash
- At expiry: if SPY < hedge_strike, put pays out `(hedge_strike - spy_close) x shares`
- Track `total_hedge_cost` and `total_hedge_payout` in metrics

**New WeekRecord fields:** `hedge_put_strike: float`, `hedge_put_premium: float` (fetched alongside existing contracts using 10-delta selection).

**CLI:** `--collar` flag. `--hedge-delta 0.10` controls OTM distance.

**Expected:** Premium drops ~15-20%, MaxDD drops to ~5-7%, Sharpe improves.

### 2. Volatility-Scaled Position Sizing

Dynamically scale contract count based on IV percentile.

**Scaling table:**

| IV Percentile | Multiplier |
|--------------|------------|
| 0-25th       | 0.5x (floor 1) |
| 25-50th      | 1.0x       |
| 50-75th      | 1.5x       |
| 75-100th     | 2.0x       |

**Engine changes:**
- `vol_scaling: bool = False`, `max_contracts: int = 3` parameters
- Each week: compute `effective_contracts` from IV percentile multiplier
- Cap at `max_contracts`
- Use `effective_contracts` for CC count cap and CSP count cap
- Initial share buy stays at base `contracts x 100`

**New metrics:** `avg_contracts_traded`, `max_contracts_traded`

**CLI:** `--vol-scaling` flag. `--max-contracts 3` cap.

**Expected:** ~5-10% more premium in high-IV periods, less exposure in low-IV.

### 3. Iron Condor Overlay

Separate engine that runs alongside the combined engine with its own capital allocation.

**What:** Sell put spread + call spread (4 legs) each week. Defined risk. Profits in range-bound markets.

**Architecture:** New `IronCondorEngine` class. Capital split:
- Combined engine: 70% of total capital
- IC engine: 30% of total capital

**IronCondorEngine:**
- Weekly: sell IC at 30-delta/10-delta strikes
- Max loss per IC = (wide_strike - narrow_strike) x 100 - net_credit
- Position size: `max_contracts = floor(allocated_capital / max_loss_per_IC)`
- Settlement at expiry based on SPY close vs strikes
- SPY in range: keep full credit
- SPY beyond spread: pay max loss on that side

**New files:**
- `src/stockdownloader/backtesting/engines/iron_condor.py`
- `tests/backtesting/engines/test_iron_condor.py`

**CLI:** `--iron-condor` flag. `--ic-allocation 0.30` (default 30%). `--ic-wide-delta 0.10`.

**Expected:** Extra ~5-8% from IC premium. Diversification (IC profits range-bound, combined profits trending).

## Testing

**Collar (8 tests):** buy protective put, put payout on crash, OTM expires worthless, reduces net premium, ML filter unaffected, hedge metrics tracked, commission on hedge, collar + CSP works.

**Vol Scaling (6 tests):** increases contracts high IV, floor at 1, respects max, disabled = original, dynamic CC count, metrics tracked.

**Iron Condor (10 tests):** net credit, max profit in range, max loss put side, max loss call side, partial loss, sizing by max loss, 4-leg commission, ML filter skip, equity curve, metrics dict.

## CLI Output

When all three enabled, comparison table shows up to 6 columns:
IC+ML Combined | ML Combined | Combined | Buy-Write | Wheel | Buy & Hold

## Expected Combined Impact

- Total return: +65-70% (from +55.3%)
- Sharpe: ~1.8-2.0 (from 1.62)
- Max drawdown: ~-5-7% (from -12.3%)
- Premium: higher due to vol scaling in high-IV periods
