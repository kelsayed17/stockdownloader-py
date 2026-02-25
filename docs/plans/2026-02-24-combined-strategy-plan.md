# Combined Buy-Write + CSP Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add combined buy-write + CSP mode with weekly IV estimation, options-derived features, and transaction cost modeling to the wheel backtest pipeline.

**Architecture:** Extends the existing `WheelBacktestEngine` with a `combined` mode that sells both covered calls on held shares and cash-secured puts on idle cash. Adds `implied_volatility()` to Black-Scholes pricing for weekly IV-based delta calculation. Uses VWAP fill prices and per-contract commissions for realistic cost modeling. Options-derived features (IV percentile, skew, volume, PCR) act as additional threshold-based filters.

**Tech Stack:** Python 3.11, pytest, scipy (for BS model), Polygon.io REST API

**Working directory:** `/Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley`

**Test command prefix:** `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest`

---

### Task 1: Implied Volatility Function

Add `implied_volatility()` to the existing Black-Scholes pricing module using bisection method.

**Files:**
- Modify: `src/stockdownloader/analysis/options/pricing.py` (append after `intrinsic_value` at line 259)
- Test: `tests/analysis/options/test_pricing.py` (append new test class)

**Step 1: Write the failing tests**

Append to `tests/analysis/options/test_pricing.py`:

```python
from stockdownloader.analysis.options.pricing import implied_volatility


class TestImpliedVolatility:

    def test_iv_call_recovers_known_vol(self):
        """Given a BS price computed at vol=0.20, IV solver should recover ~0.20."""
        known_vol = Decimal("0.20")
        market_price = bs_price(OptionType.CALL, SPOT, STRIKE_ATM, TIME_30D, RATE, known_vol)
        iv = implied_volatility(OptionType.CALL, market_price, SPOT, STRIKE_ATM, TIME_30D, RATE)
        assert abs(float(iv) - 0.20) < 0.01

    def test_iv_put_recovers_known_vol(self):
        """Given a BS price computed at vol=0.30, IV solver should recover ~0.30."""
        known_vol = Decimal("0.30")
        market_price = bs_price(OptionType.PUT, SPOT, STRIKE_ATM, TIME_30D, RATE, known_vol)
        iv = implied_volatility(OptionType.PUT, market_price, SPOT, STRIKE_ATM, TIME_30D, RATE)
        assert abs(float(iv) - 0.30) < 0.01

    def test_iv_zero_price_returns_minimum(self):
        """Zero market price should return minimum IV (0.01)."""
        iv = implied_volatility(OptionType.CALL, Decimal("0"), SPOT, STRIKE_ATM, TIME_30D, RATE)
        assert float(iv) == pytest.approx(0.01, abs=0.005)

    def test_iv_deep_otm_converges(self):
        """Deep OTM option with small price should converge without error."""
        market_price = Decimal("0.05")
        iv = implied_volatility(
            OptionType.CALL, market_price, Decimal("500"), Decimal("550"),
            Decimal("0.0137"), RATE,
        )
        assert float(iv) > 0.0
        assert float(iv) < 3.0
```

**Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/analysis/options/test_pricing.py::TestImpliedVolatility -v
```

Expected: FAIL with `ImportError: cannot import name 'implied_volatility'`

**Step 3: Write the implementation**

Append to `src/stockdownloader/analysis/options/pricing.py` after `intrinsic_value()`:

```python
def implied_volatility(
    option_type: OptionType,
    market_price: Decimal,
    spot: Decimal,
    strike: Decimal,
    time_to_expiry: Decimal,
    risk_free_rate: Decimal,
    *,
    tol: float = 0.001,
    max_iter: int = 100,
) -> Decimal:
    """Find implied volatility via bisection method.

    Given a market price, find the volatility that makes the BS price
    match the market price within ``tol``.

    Returns:
        Implied volatility as a Decimal. Minimum 0.01, maximum 3.0.
    """
    mp = float(market_price)
    if mp <= 0:
        return Decimal("0.01")
    if time_to_expiry <= Decimal("0"):
        return Decimal("0.01")

    lo, hi = 0.01, 3.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        bs_mid = float(price(option_type, spot, strike, time_to_expiry, risk_free_rate, Decimal(str(mid))))
        if abs(bs_mid - mp) < tol:
            break
        if bs_mid < mp:
            lo = mid
        else:
            hi = mid

    result = (lo + hi) / 2.0
    return Decimal(str(max(result, 0.01))).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
```

**Step 4: Run tests to verify they pass**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/analysis/options/test_pricing.py::TestImpliedVolatility -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/stockdownloader/analysis/options/pricing.py tests/analysis/options/test_pricing.py
git commit -m "feat: add implied_volatility() bisection solver to BS pricing"
```

---

### Task 2: Transaction Costs in Engine

Add `commission_per_contract` parameter and `total_commissions` tracking to the engine. Also add `_n_rebuys` tracking to metrics (already exists from buy-write).

**Files:**
- Modify: `src/stockdownloader/backtesting/engines/wheel.py` (lines 62-97 init, lines 259-277 metrics)
- Test: `tests/backtesting/engines/test_wheel.py` (append new test class)

**Step 1: Write the failing tests**

Append to `tests/backtesting/engines/test_wheel.py`:

```python
class TestTransactionCosts:

    def test_commission_deducted_from_premium(self):
        """Commission reduces net premium collected."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            commission_per_contract=0.65,
        )
        week = _make_week(0, spy_open=600.0, spy_close=605.0, put_premium=3.0)
        engine.process_week(week)
        # Gross premium: 3.0 * 100 = 300. Commission: 0.65 * 1 = 0.65
        # Net premium: 299.35
        assert abs(engine.total_premium_collected - 299.35) < 0.01

    def test_zero_commission_matches_original(self):
        """With commission=0, behavior is identical to original engine."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            commission_per_contract=0.0,
        )
        week = _make_week(0, spy_open=600.0, spy_close=605.0, put_premium=3.0)
        engine.process_week(week)
        assert engine.total_premium_collected == 300.0

    def test_total_commissions_tracked_in_metrics(self):
        """Metrics dict includes total_commissions key."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            commission_per_contract=0.65,
        )
        weeks = [_make_week(i, spy_open=600.0, spy_close=605.0) for i in range(3)]
        for w in weeks:
            engine.process_week(w)
        metrics = engine.compute_metrics()
        assert "total_commissions" in metrics
        # 3 puts sold * 1 contract * 0.65 = 1.95
        assert abs(metrics["total_commissions"] - 1.95) < 0.01
```

**Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py::TestTransactionCosts -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'commission_per_contract'`

**Step 3: Write the implementation**

In `src/stockdownloader/backtesting/engines/wheel.py`:

1. Add `commission_per_contract: float = 0.0` parameter to `__init__` (after `buy_write`).
2. Store `self._commission_per_contract = commission_per_contract`.
3. Add `self._total_commissions: float = 0.0` to tracking section.
4. In `_process_week_wheel` and `_process_week_buy_write`, after computing premium, deduct commission:
   ```python
   commission = self._commission_per_contract * n_contracts
   net_premium = gross_premium - commission
   self._cash += net_premium  # instead of += gross_premium
   self._total_premium += net_premium
   self._total_commissions += commission
   ```
   For wheel mode, `n_contracts = self._contracts`. For buy-write mode, same.
5. Add `"total_commissions": self._total_commissions` to `compute_metrics()`.

Specifically, modify the docstring to add:
```
    commission_per_contract:
        Per-contract per-leg commission in dollars (default: 0.0).
```

In `__init__`, add after `self._buy_write = buy_write`:
```python
        self._commission_per_contract = commission_per_contract
```

Add to tracking section after `self._n_calls_skipped`:
```python
        self._total_commissions: float = 0.0
```

In `_process_week_wheel`, put phase, replace:
```python
                premium = week.put_premium * multiplier
                self._cash += premium
                self._total_premium += premium
```
with:
```python
                gross = week.put_premium * multiplier
                commission = self._commission_per_contract * self._contracts
                self._cash += gross - commission
                self._total_premium += gross - commission
                self._total_commissions += commission
```

In `_process_week_wheel`, call phase, replace similarly for `week.call_premium`.

In `_process_week_buy_write`, replace the premium block similarly.

In `compute_metrics()`, add `"total_commissions": self._total_commissions` to the return dict.

**Step 4: Run tests to verify they pass**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py::TestTransactionCosts -v
```

Expected: 3 PASSED

**Step 5: Run full wheel test suite to verify no regressions**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py -v
```

Expected: All existing + 3 new tests PASS (commission=0.0 default means no behavior change)

**Step 6: Commit**

```bash
git add src/stockdownloader/backtesting/engines/wheel.py tests/backtesting/engines/test_wheel.py
git commit -m "feat: add transaction cost modeling to wheel engine"
```

---

### Task 3: Combined Mode Engine Logic

Add `combined: bool = False` parameter and `_process_week_combined()` method.

**Files:**
- Modify: `src/stockdownloader/backtesting/engines/wheel.py`
- Test: `tests/backtesting/engines/test_wheel.py` (append new test classes)

**Step 1: Write the core combined mode tests**

Append to `tests/backtesting/engines/test_wheel.py`:

```python
class TestCombinedMode:

    def test_combined_buys_shares_immediately(self):
        """Combined mode buys shares on week 1."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(0, spy_open=500.0, spy_close=505.0)
        engine.process_week(week)
        assert engine.shares_held >= 100

    def test_combined_sells_both_cc_and_csp(self):
        """Combined mode collects premium from both calls and puts."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_sold"] >= 1
        assert metrics["n_puts_sold"] >= 1
        # Premium from both sides: at least 200 + 200 = 400
        assert engine.total_premium_collected >= 400.0

    def test_combined_cc_exercise_rebuys(self):
        """CC exercise in combined mode: called away + immediate re-buy."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=530.0,
            put_strike=480.0, put_premium=1.0,
            call_strike=520.0, call_premium=3.0,
        )
        engine.process_week(week)
        assert engine.shares_held >= 100  # Still holding after re-buy
        metrics = engine.compute_metrics()
        assert metrics["n_calls_exercised"] >= 1
        assert metrics["n_rebuys"] >= 1

    def test_combined_csp_assignment_grows_position(self):
        """CSP assignment in combined mode adds shares."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # SPY at 500 -> buy 100 shares ($50K). Cash left ~$50K.
        # Put strike 480 -> CSP collateral $48K. Cash supports 1 CSP.
        # SPY drops to 470 -> put ITM, assigned -> acquire 100 more shares.
        week = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        assert engine.shares_held == 200  # 100 initial + 100 from CSP assignment
        metrics = engine.compute_metrics()
        assert metrics["n_assignments"] >= 1

    def test_combined_both_itm_settles_correctly(self):
        """Both CC and CSP ITM in same week: exercises cancel out."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # SPY at 500, put strike 510 (ITM), call strike 490 (ITM)
        # This is unusual (put strike > call strike) but tests the settlement
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=510.0, put_premium=12.0,
            call_strike=490.0, call_premium=12.0,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_exercised"] >= 1
        assert metrics["n_assignments"] >= 1

    def test_combined_csp_capped_at_contracts(self):
        """CSP count doesn't exceed the contracts parameter."""
        # With $200K capital, SPY at 500: buy 100 shares ($50K).
        # Cash left = $150K, could support 3 CSPs at $480 strike.
        # But contracts=1, so max CSP = 1.
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        # Only 1 CSP should have been assigned (capped)
        metrics = engine.compute_metrics()
        assert metrics["n_puts_sold"] == 1
        assert engine.shares_held == 200  # 100 initial + 100 from 1 CSP

    def test_combined_ml_skips_cc_and_csp(self):
        """ML filter skips CC when bullish and CSP when bearish."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            skip_call_thresh=0.65, skip_put_thresh=0.35,
            combined=True,
        )
        # prob=0.70 -> skip CC (rally), do sell CSP
        week_bullish = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.70,
        )
        engine.process_week(week_bullish, use_ml_filter=True)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_skipped"] >= 1
        assert metrics["n_puts_sold"] >= 1

    def test_combined_dynamic_cc_count_after_assignment(self):
        """After CSP assignment, more CCs are sold next week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # Week 0: Buy 100 shares, sell 1 CC + 1 CSP, CSP assigned -> 200 shares
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: Should sell 2 CCs (200 shares / 100)
        week1 = _make_week(
            1, spy_open=470.0, spy_close=475.0,
            put_strike=460.0, put_premium=2.0,
            call_strike=490.0, call_premium=2.0,
        )
        engine.process_week(week1)
        metrics = engine.compute_metrics()
        # Week 0: 1 CC + week 1: 2 CCs = 3 total calls sold
        assert metrics["n_calls_sold"] == 3
```

**Step 2: Write the edge case tests**

Append to `tests/backtesting/engines/test_wheel.py`:

```python
class TestCombinedEdgeCases:

    def test_combined_no_cash_for_csp(self):
        """After CSP assignment eats all cash, no CSPs sold next week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # Week 0: CSP assigned -> 200 shares, ~$0 cash
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week0)
        puts_after_w0 = engine.compute_metrics()["n_puts_sold"]

        # Week 1: No cash for CSP. Only sell CCs.
        week1 = _make_week(
            1, spy_open=470.0, spy_close=475.0,
            put_strike=460.0, put_premium=2.0,
            call_strike=490.0, call_premium=2.0,
        )
        engine.process_week(week1)
        puts_after_w1 = engine.compute_metrics()["n_puts_sold"]
        # No new puts sold (no cash for CSP collateral)
        assert puts_after_w1 == puts_after_w0

    def test_combined_max_position_after_multiple_assignments(self):
        """Multiple CSP assignments grow position correctly."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1, combined=True,
        )
        # Week 0: Buy 100 shares at 500 ($50K). Cash=$150K. Sell 1 CSP at 480. Assigned.
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: Now 200 shares. Cash ~$102K (150K - 48K + premiums).
        # Sell 1 CSP at 460. Assigned again.
        week1 = _make_week(
            1, spy_open=470.0, spy_close=450.0,
            put_strike=460.0, put_premium=4.0,
            call_strike=490.0, call_premium=1.0,
        )
        engine.process_week(week1)
        assert engine.shares_held == 300

    def test_combined_cc_exercise_frees_cash_for_csp(self):
        """CC exercise frees cash, enabling CSP sale next week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        # Week 0: Buy 100 at 500, CSP assigned at 480 -> 200 shares, ~$0 cash
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: CC exercised (price rallies above call strike)
        # 2 CCs exercised -> sell 200 shares at 490, re-buy at 495
        # Cash increases by (490-495)*200 = -$1000 net, but now some cash freed
        week1 = _make_week(
            1, spy_open=475.0, spy_close=495.0,
            put_strike=470.0, put_premium=2.0,
            call_strike=490.0, call_premium=3.0,
        )
        engine.process_week(week1)
        # After re-buy, should still hold 200 shares
        # Premium from 2 CCs should give some cash
        metrics = engine.compute_metrics()
        assert metrics["n_calls_exercised"] >= 2

    def test_combined_zero_premium_handled(self):
        """Zero put premium is handled gracefully (still sell call)."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=0.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_sold"] >= 1

    def test_combined_insufficient_capital_for_initial_buy(self):
        """Capital too low to buy even 1 contract of shares."""
        engine = WheelBacktestEngine(
            initial_capital=1_000.0, contracts=1, combined=True,
        )
        # SPY at 500 -> 100 shares = $50K. Capital=$1K. Can't afford it.
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
        )
        engine.process_week(week)
        # Cash goes negative (the engine doesn't enforce margin).
        # Should still process without crashing.
        metrics = engine.compute_metrics()
        assert metrics["weeks"] == 1
```

**Step 3: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py::TestCombinedMode tests/backtesting/engines/test_wheel.py::TestCombinedEdgeCases -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'combined'`

**Step 4: Write the implementation**

In `src/stockdownloader/backtesting/engines/wheel.py`:

1. Add `combined: bool = False` parameter to `__init__` (after `buy_write`).
2. Store `self._combined = combined`.
3. Add module docstring line for combined mode (already present from buy-write update).
4. Update `process_week` dispatch:
   ```python
   if self._combined:
       self._process_week_combined(week, use_ml_filter)
   elif self._buy_write:
       self._process_week_buy_write(week, multiplier, use_ml_filter)
   else:
       self._process_week_wheel(week, multiplier, use_ml_filter)
   ```
5. Add `_process_week_combined()`:

```python
def _process_week_combined(
    self,
    week: WeekRecord,
    use_ml_filter: bool,
) -> None:
    """Combined mode: buy-write + CSPs on idle cash."""
    multiplier = self._contracts * 100

    # Step 1: Buy initial shares if not holding
    if self._shares == 0:
        cost = week.spy_price_at_entry * multiplier
        self._cash -= cost
        self._shares = multiplier
        self._share_cost_basis = week.spy_price_at_entry
        self._state = WheelState.CALL_PHASE

    # Step 2: Sell covered calls on ALL held shares
    n_cc = self._shares // 100
    if n_cc > 0:
        if not (use_ml_filter and week.ml_prob > self._skip_call_thresh):
            gross = week.call_premium * n_cc * 100
            commission = self._commission_per_contract * n_cc
            self._cash += gross - commission
            self._total_premium += gross - commission
            self._total_commissions += commission
            self._n_calls_sold += n_cc

            if week.spy_price_at_expiry > week.call_strike:
                # Called away on n_cc contracts, immediately re-buy
                self._cash += week.call_strike * n_cc * 100
                self._cash -= week.spy_price_at_expiry * n_cc * 100
                self._n_calls_exercised += n_cc
                self._n_rebuys += n_cc
                self._share_cost_basis = week.spy_price_at_expiry
        else:
            self._n_calls_skipped += n_cc

    # Step 3: Sell CSPs on available cash (capped at self._contracts)
    if week.put_strike > 0:
        max_csp = min(self._contracts, int(self._cash // (week.put_strike * 100)))
    else:
        max_csp = 0
    if max_csp > 0:
        if not (use_ml_filter and week.ml_prob < self._skip_put_thresh):
            gross = week.put_premium * max_csp * 100
            commission = self._commission_per_contract * max_csp
            self._cash += gross - commission
            self._total_premium += gross - commission
            self._total_commissions += commission
            self._n_puts_sold += max_csp

            if week.spy_price_at_expiry < week.put_strike:
                # Assigned: acquire more shares
                cost = week.put_strike * max_csp * 100
                self._cash -= cost
                self._shares += max_csp * 100
                self._n_assignments += max_csp
        else:
            self._n_puts_skipped += max_csp

    self._state = WheelState.CALL_PHASE
```

**Step 5: Run tests to verify they pass**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py -v
```

Expected: All tests PASS (old + new combined mode + edge cases + transaction costs)

**Step 6: Commit**

```bash
git add src/stockdownloader/backtesting/engines/wheel.py tests/backtesting/engines/test_wheel.py
git commit -m "feat: add combined buy-write + CSP mode to wheel engine"
```

---

### Task 4: CLI Changes — Parser, IV, VWAP, Multi-Engine

Add `--combined`, `--commission`, `--iv-filter`, `--min-iv-percentile` flags. Use VWAP for fill prices. Run all 4 engines when `--combined`. Weekly IV estimation.

**Files:**
- Modify: `src/stockdownloader/app/spy_options_wheel.py`
- Test: `tests/app/test_spy_options_wheel.py` (append new tests)

**Step 1: Write the failing parser tests**

Append to `tests/app/test_spy_options_wheel.py`:

```python
class TestCombinedParser:

    def test_combined_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.combined is False

    def test_combined_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--combined"])
        assert args.combined is True

    def test_commission_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.commission == 0.65

    def test_commission_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--commission", "1.00"])
        assert args.commission == 1.00

    def test_iv_filter_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.iv_filter is False

    def test_iv_filter_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--iv-filter"])
        assert args.iv_filter is True

    def test_min_iv_percentile_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.min_iv_percentile == 0.30
```

**Step 2: Write the combined integration test**

Append to `tests/app/test_spy_options_wheel.py`:

```python
    def test_combined_lifecycle(self):
        """Combined mode: buy shares + sell CC + sell CSP, with assignment."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0,
            contracts=1,
            skip_call_thresh=0.65,
            skip_put_thresh=0.35,
            commission_per_contract=0.65,
            combined=True,
        )

        weeks = [
            # Week 0: Buy 100 at 500. Sell 1 CC at 520 + 1 CSP at 480. All OTM.
            WeekRecord(0, "2025-02-07", "2025-02-03", 500.0, 505.0, 480.0, 520.0, 2.0, 2.0, 0.50),
            # Week 1: SPY drops to 470. CSP at 480 ITM -> assigned. Now 200 shares.
            WeekRecord(1, "2025-02-14", "2025-02-10", 505.0, 470.0, 480.0, 520.0, 4.0, 1.0, 0.50),
            # Week 2: SPY rallies to 495. 2 CCs sold (200 shares). CC at 490 ITM -> called away + rebuy.
            WeekRecord(2, "2025-02-21", "2025-02-17", 475.0, 495.0, 460.0, 490.0, 2.0, 3.0, 0.50),
            # Week 3: ML prob 0.30 -> skip CSP (crash danger). Still sell CC.
            WeekRecord(3, "2025-02-28", "2025-02-24", 490.0, 495.0, 475.0, 510.0, 3.0, 2.0, 0.30),
        ]

        for w in weeks:
            engine.process_week(w, use_ml_filter=True)

        metrics = engine.compute_metrics()
        assert metrics["weeks"] == 4
        assert metrics["n_assignments"] >= 1  # Week 1 CSP
        assert metrics["n_calls_exercised"] >= 1  # Week 2 CC
        assert metrics["total_premium_collected"] > 0
        assert metrics["total_commissions"] > 0
        assert engine.shares_held >= 100
```

**Step 3: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/app/test_spy_options_wheel.py::TestCombinedParser tests/app/test_spy_options_wheel.py::TestWheelIntegration::test_combined_lifecycle -v
```

Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'combined'`

**Step 4: Write the implementation**

In `src/stockdownloader/app/spy_options_wheel.py`:

1. Add new CLI args to `_build_parser()` (after `--buy-write`):
   ```python
   parser.add_argument(
       "--combined", action="store_true",
       help="Combined buy-write + CSP mode: sell CCs on shares + CSPs on idle cash.",
   )
   parser.add_argument(
       "--commission", type=float, default=0.65,
       help="Per-contract per-leg commission in dollars (default: 0.65)",
   )
   parser.add_argument(
       "--iv-filter", action="store_true",
       help="Enable IV-based filtering (skip selling when IV too low)",
   )
   parser.add_argument(
       "--min-iv-percentile", type=float, default=0.30,
       help="Minimum IV percentile to sell options (default: 0.30)",
   )
   ```

2. In `main()`, after parsing args, add mutual exclusivity check:
   ```python
   if args.combined and args.buy_write:
       parser.error("--combined and --buy-write are mutually exclusive")
   ```

3. Update `mode_label`:
   ```python
   if args.combined:
       mode_label = "COMBINED"
   elif args.buy_write:
       mode_label = "BUY-WRITE"
   else:
       mode_label = "WHEEL"
   ```

4. Update premium extraction to use VWAP (lines 432-433):
   ```python
   put_premium = put_bar.get("vw", put_bar.get("c", 0.0)) if put_bar else 0.0
   call_premium = call_bar.get("vw", call_bar.get("c", 0.0)) if call_bar else 0.0
   ```

5. Add weekly IV computation in the week loop (after getting bars):
   ```python
   from stockdownloader.analysis.options.pricing import implied_volatility
   from stockdownloader.core.models.options import OptionType
   # After fetching put/call bars and before building WeekRecord:
   weekly_vol = hist_vol  # default
   if put_bar and call_bar and spy_at_entry > 0:
       try:
           put_iv = float(implied_volatility(
               OptionType.PUT, Decimal(str(put_premium)),
               Decimal(str(spy_at_entry)), Decimal(str(put_contract["strike_price"])),
               Decimal(str(5 / 365)), Decimal("0.05"),
           ))
           call_iv = float(implied_volatility(
               OptionType.CALL, Decimal(str(call_premium)),
               Decimal(str(spy_at_entry)), Decimal(str(call_contract["strike_price"])),
               Decimal(str(5 / 365)), Decimal("0.05"),
           ))
           weekly_vol = (put_iv + call_iv) / 2
       except Exception:
           pass
   ```
   Use `weekly_vol` in `select_strike_by_delta()` calls instead of `hist_vol`.

6. Track IV history for IV percentile computation:
   ```python
   iv_history: list[float] = []
   # Inside the week loop, after computing weekly_vol:
   iv_history.append(weekly_vol)
   iv_percentile = 0.5  # default
   if len(iv_history) >= 10:
       sorted_ivs = sorted(iv_history)
       rank = sorted_ivs.index(weekly_vol) if weekly_vol in sorted_ivs else len(sorted_ivs) // 2
       iv_percentile = rank / len(sorted_ivs)
   ```

7. When `--combined` is passed, run all 4 engines in section [4/5]:
   ```python
   if args.combined:
       # ML Combined
       ml_combined_metrics = None
       if not args.no_ml_filter and ml_probs:
           ml_combined = WheelBacktestEngine(
               initial_capital=args.initial_capital, contracts=args.contracts,
               skip_put_thresh=args.skip_put_thresh, skip_call_thresh=args.skip_call_thresh,
               combined=True, commission_per_contract=args.commission,
           )
           for w in week_records:
               ml_combined.process_week(w, use_ml_filter=True)
           ml_combined_metrics = ml_combined.compute_metrics()

       # Mechanical Buy-Write
       bw_engine = WheelBacktestEngine(
           initial_capital=args.initial_capital, contracts=args.contracts,
           buy_write=True, commission_per_contract=args.commission,
       )
       for w in week_records:
           bw_engine.process_week(w, use_ml_filter=False)
       bw_metrics = bw_engine.compute_metrics()

       # Mechanical Wheel
       wheel_engine = WheelBacktestEngine(
           initial_capital=args.initial_capital, contracts=args.contracts,
           commission_per_contract=args.commission,
       )
       for w in week_records:
           wheel_engine.process_week(w, use_ml_filter=False)
       wheel_metrics = wheel_engine.compute_metrics()
   ```

8. Add `_print_comparison_table()` function for the 4-column output.

9. Call the appropriate print function based on mode.

**Step 5: Run tests to verify they pass**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/app/test_spy_options_wheel.py -v
```

Expected: All tests PASS

**Step 6: Run full test suite**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q
```

Expected: All 3500+ tests PASS

**Step 7: Commit**

```bash
git add src/stockdownloader/app/spy_options_wheel.py tests/app/test_spy_options_wheel.py
git commit -m "feat: add --combined mode with IV, VWAP, commission to wheel CLI"
```

---

### Task 5: Live Backtest Run

Run the combined backtest against real Polygon data and compare all strategies.

**Step 1: Run combined mode (no ML first)**

```bash
POLYGON_API_KEY=jtFFyq1sO7sEznPZChj2vMZEeoY9GQSu \
DYLD_LIBRARY_PATH=~/lib PYTHONUNBUFFERED=1 \
python3 -u -c "
from stockdownloader.app.spy_options_wheel import main
main(['--combined', '--no-ml-filter', '--from-date', '2023-01-01', '--contracts', '1', '--quick'])
"
```

**Step 2: Run combined mode with ML filter**

```bash
POLYGON_API_KEY=jtFFyq1sO7sEznPZChj2vMZEeoY9GQSu \
DYLD_LIBRARY_PATH=~/lib PYTHONUNBUFFERED=1 \
python3 -u -c "
from stockdownloader.app.spy_options_wheel import main
main(['--combined', '--from-date', '2023-01-01', '--contracts', '1', '--quick', '--walk-forward-windows', '3'])
"
```

**Step 3: Compare and report results**

Verify the 4-column comparison table shows all strategies and combined mode returns are higher than buy-write and wheel.

## Verification

```bash
# Unit tests
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py tests/app/test_spy_options_wheel.py tests/analysis/options/test_pricing.py -v

# Full suite regression
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q
```
