# Combined Buy-Write + CSP Strategy Design

**Date**: 2026-02-24
**Status**: Approved
**Branch**: `claude/vigorous-easley`

## Problem

The buy-write strategy returns +27.7% vs buy-and-hold +77.7% over 3 years.
~50% of capital sits idle as cash. Premium income ($25K) is good but single-sided
(calls only). Static volatility for delta calc, no transaction costs, and no
options-derived ML features limit accuracy.

## Solution Overview

Four improvements to the wheel pipeline:

1. **Combined mode** — sell CSPs on idle cash alongside buy-write covered calls
2. **Weekly IV estimation** — back-solve implied volatility from market prices
3. **Options-derived features** — IV percentile, skew, volume, PCR as filters
4. **Transaction costs** — VWAP fill prices + commission modeling

## 1. Combined Mode (Buy-Write + CSP)

### Core Mechanic

Each week, the engine fully deploys capital across both options sides:

```
WEEK START (Monday):
  1. If no shares held -> buy initial shares (same as buy-write)
  2. Sell CCs on ALL held shares (1 CC per 100 shares)
  3. Sell CSPs on available cash (capped at `contracts` to limit risk)
  4. ML filter: skip CC when prob > 0.65, skip CSP when prob < 0.35

WEEK END (Friday expiry):
  5. CC ITM -> shares called away, immediately re-buy at market
     (maintains buy-and-hold exposure)
  6. CSP ITM -> acquire additional shares at put_strike
     (position grows, more CCs next week)
  7. Both ITM -> net out: same shares, gain (call_strike - put_strike) * 100
```

### Self-Balancing

- CC exercised -> fewer shares, more cash -> sell more CSPs next week
- CSP assigned -> more shares, less cash -> sell more CCs next week
- Capital is always fully deployed (shares + CSP collateral)

### Position Sizing

- Initial shares: `contracts * 100` (same as buy-write)
- CC count per week: `shares_held // 100` (dynamic, grows with assignments)
- CSP count per week: `min(contracts, floor(cash / (put_strike * 100)))`
  (capped at `contracts` to prevent over-leveraging)
- On CC exercise: immediately re-buy same number of shares (preserve buy-and-hold)
- On CSP assignment: position grows, no forced sell

### Example Flow ($100K, 1 contract, SPY at $500)

- Week 1: Hold 100 shares ($50K) + sell 1 CC + sell 1 CSP (using $50K cash)
- If CSP assigned: Now 200 shares, ~$0 cash -> sell 2 CCs, 0 CSPs
- If then CC exercised on 100: Back to 100 shares + $50K cash -> 1 CC + 1 CSP

### Engine Implementation

New parameter: `combined: bool = False` on `WheelBacktestEngine.__init__()`.

New method `_process_week_combined()`:

```python
def _process_week_combined(self, week, multiplier, use_ml_filter):
    # Step 1: Buy initial shares if needed
    if self._shares == 0:
        cost = week.spy_price_at_entry * multiplier
        self._cash -= cost
        self._shares = multiplier

    # Step 2: Sell covered calls on ALL held shares
    n_cc = self._shares // 100
    if n_cc > 0:
        if not (use_ml_filter and week.ml_prob > self._skip_call_thresh):
            premium = week.call_premium * n_cc * 100
            self._cash += premium
            self._total_premium += premium
            self._n_calls_sold += n_cc
            if week.spy_price_at_expiry > week.call_strike:
                # Called away + immediate re-buy
                self._cash += week.call_strike * n_cc * 100
                self._cash -= week.spy_price_at_expiry * n_cc * 100
                self._n_calls_exercised += n_cc
                self._n_rebuys += n_cc
        else:
            self._n_calls_skipped += n_cc

    # Step 3: Sell CSPs on available cash (capped)
    max_csp = min(self._contracts, int(self._cash // (week.put_strike * 100)))
    if max_csp > 0:
        if not (use_ml_filter and week.ml_prob < self._skip_put_thresh):
            premium = week.put_premium * max_csp * 100
            self._cash += premium
            self._total_premium += premium
            self._n_puts_sold += max_csp
            if week.spy_price_at_expiry < week.put_strike:
                cost = week.put_strike * max_csp * 100
                self._cash -= cost
                self._shares += max_csp * 100
                self._n_assignments += max_csp
        else:
            self._n_puts_skipped += max_csp
```

## 2. Weekly IV Estimation

### Problem

Historical volatility is computed once for the entire backtest period. This makes
0.30 delta strike selection inaccurate — low vol periods get strikes too close,
high vol periods get strikes too far.

### Solution

Back-solve for implied volatility each week from the option market prices already
fetched from Polygon.

New function in `analysis/options/pricing.py`:

```python
def implied_volatility(
    option_type: OptionType,
    market_price: Decimal,
    spot: Decimal,
    strike: Decimal,
    time_to_expiry: Decimal,
    risk_free_rate: Decimal,
) -> Decimal:
    """Bisection method to find IV from market price."""
    # Search vol in [0.01, 3.0] range
    # Converge when |BS_price(vol) - market_price| < 0.001
```

### Usage

- Each week, compute IV from fetched call and put market prices
- Average put IV and call IV for that week's delta calculation
- Store weekly IV in `WeekRecord` for options features (Section 3)
- Replaces static `hist_vol` in `select_strike_by_delta()` calls

### New WeekRecord Field

`implied_vol: float` — weekly IV estimate (average of put and call IV)

## 3. Options-Derived Features

### Problem

All 63 ML features are equity price/volume based. The ML model has no access to
options market data (IV, volume, sentiment).

### New Features

Computed weekly from cached Polygon options data and added as threshold-based
filters layered on top of the ML probability signal:

| Feature | Source | Computation | Usage |
|---------|--------|-------------|-------|
| `iv_percentile` | Weekly IV (Section 2) | Rank vs trailing 52 weeks | Sell more when IV > 70th pctile |
| `iv_skew` | Put IV / Call IV | Ratio of implied vols | > 1.2 = elevated fear, skip CSPs |
| `option_volume_ratio` | Bar `v` field | This week vol / 20-week avg | Unusual activity indicator |
| `pcr_volume` | Put vol / Call vol | From bar `v` fields | High PCR = bearish sentiment |

### Implementation

Features computed in CLI pipeline and stored in `WeekRecord` as optional fields.
Engine uses them as additional filter thresholds alongside `ml_prob`:

```python
if week.iv_percentile < 0.30:
    # IV very low -- skip selling (not enough premium to justify risk)
```

### New CLI Flags

- `--iv-filter` (store_true) — enable IV-based filtering
- `--min-iv-percentile 0.30` — minimum IV percentile to sell options

## 4. Transaction Costs

### Problem

No costs modeled. All premium received at close price. Unrealistic returns.

### Solution

Use Polygon data fields already fetched but currently ignored:

1. **Fill price:** Use `vw` (volume-weighted average price) instead of `c` (close).
   VWAP represents where most volume actually traded. Fallback to
   `c - (h - l) * 0.1` if VWAP is unavailable (10% of intraday range as slippage).

2. **Commission:** $0.65 per contract per leg (industry standard). Configurable.

### Engine Changes

- New param: `commission_per_contract: float = 0.65`
- Each option sold: `net_premium = raw_premium * multiplier - commission * n_contracts`
- New metric: `total_commissions` in output dict

### Pipeline Changes

- `put_premium` and `call_premium` in `WeekRecord` populated from `bar["vw"]`
  instead of `bar["c"]`, with close as fallback
- New CLI flag: `--commission 0.65` (default)

## CLI Changes

### New Flags

```
--combined                   Combined buy-write + CSP mode
--iv-filter                  Enable IV-based filtering
--min-iv-percentile 0.30     Min IV percentile to sell (default: 0.30)
--commission 0.65            Per-contract per-leg commission (default: 0.65)
```

### Mutual Exclusivity

`--combined` and `--buy-write` are mutually exclusive. Error if both specified.

### Output Table (Combined Mode)

When `--combined` is specified, run all 3 engines and show full comparison:

```
======================================================================
SPY WEEKLY STRATEGY COMPARISON
======================================================================
                        ML Combined   Buy-Write      Wheel  Buy & Hold
  --------------------------------------------------------------------
  Total Return              +XX.X%      +XX.X%     +XX.X%      +77.7%
  Annualized Return         +XX.X%      +XX.X%     +XX.X%         N/A
  Premium Collected         $XX,XXX     $XX,XXX    $XX,XXX        N/A
  Puts Sold                      X         N/A          X         N/A
  Calls Sold                     X           X          X         N/A
  Assignments (CSP)              X         N/A          X         N/A
  Calls Exercised / Rebuys     XX          XX          XX         N/A
  Calls Skipped (ML)             X         N/A        N/A         N/A
  Puts Skipped (ML)              X         N/A        N/A         N/A
  Sharpe                      X.XX        X.XX      X.XX         N/A
  Max Drawdown               -X.X%       -X.X%     -X.X%         N/A
======================================================================
```

## Testing

### TestCombinedMode (8 tests in test_wheel.py)

1. `test_combined_buys_shares_immediately`
2. `test_combined_sells_both_cc_and_csp`
3. `test_combined_cc_exercise_rebuys`
4. `test_combined_csp_assignment_grows_position`
5. `test_combined_both_itm_settles_correctly`
6. `test_combined_csp_capped_at_contracts`
7. `test_combined_ml_skips_cc_and_csp`
8. `test_combined_dynamic_cc_count_after_assignment`

### TestCombinedEdgeCases (5 tests in test_wheel.py)

1. `test_combined_no_cash_for_csp`
2. `test_combined_max_position_after_multiple_assignments`
3. `test_combined_cc_exercise_frees_cash_for_csp`
4. `test_combined_zero_premium_skipped`
5. `test_combined_insufficient_capital_for_initial_buy`

### TestImpliedVolatility (4 tests in test_pricing.py)

1. `test_iv_call_recovers_known_vol`
2. `test_iv_put_recovers_known_vol`
3. `test_iv_zero_price_returns_zero`
4. `test_iv_deep_otm_converges`

### TestTransactionCosts (3 tests in test_wheel.py)

1. `test_commission_deducted_from_premium`
2. `test_vwap_used_as_fill_price`
3. `test_total_commissions_tracked`

### Parser Tests (5 tests in test_spy_options_wheel.py)

1. `test_combined_default_false`
2. `test_combined_flag`
3. `test_combined_and_buy_write_exclusive`
4. `test_commission_default`
5. `test_iv_filter_flags`

### Integration Test (1 test)

1. `test_combined_full_lifecycle` — multi-week scenario with assignments,
   exercises, position growth, and equity verification

## Files Modified

| File | Changes |
|------|---------|
| `src/stockdownloader/backtesting/engines/wheel.py` | Add `combined`, `commission_per_contract` params, `_process_week_combined()`, commission tracking |
| `src/stockdownloader/analysis/options/pricing.py` | Add `implied_volatility()` function |
| `src/stockdownloader/app/spy_options_wheel.py` | Add `--combined`, `--iv-filter`, `--min-iv-percentile`, `--commission` flags. Run all 3 engines when combined. Weekly IV calc. VWAP fill prices. Full comparison table. |
| `tests/backtesting/engines/test_wheel.py` | Add TestCombinedMode (8), TestCombinedEdgeCases (5), TestTransactionCosts (3) |
| `tests/analysis/options/test_pricing.py` | Add TestImpliedVolatility (4) |
| `tests/app/test_spy_options_wheel.py` | Add parser tests (5), integration test (1) |

**Total new tests: 26**
