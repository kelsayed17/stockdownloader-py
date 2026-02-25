# Advanced Options Strategies Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three strategies (protective put collar, volatility-scaled position sizing, iron condor overlay) to close the return gap with buy-and-hold while improving Sharpe ratio and reducing max drawdown.

**Architecture:** Collar and vol-scaling extend the existing `WheelBacktestEngine` with new parameters and logic in `process_week()`. The iron condor is a separate `IronCondorEngine` class with its own `ICWeekRecord` dataclass. The CLI orchestrates capital splitting (70/30) and merges equity curves for the comparison table.

**Tech Stack:** Python 3.11, dataclasses, pytest. No new dependencies.

---

## Context

**Working directory:** `/Users/kelsayed/Documents/GitHub/stockdownloader-py/.claude/worktrees/vigorous-easley`

**Branch:** `claude/vigorous-easley`

**Test command prefix:** `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest`

**Current state:** ML Combined +55.3% (Sharpe 1.62, MaxDD -12.3%) vs Buy & Hold +77.7%. Goal: close the 22-point gap.

**Key files:**
- `src/stockdownloader/backtesting/engines/wheel.py` (365 lines) — engine with wheel/buy-write/combined modes
- `tests/backtesting/engines/test_wheel.py` (531 lines) — 34 tests, `_make_week()` helper at line 13
- `src/stockdownloader/app/spy_options_wheel.py` (673 lines) — CLI pipeline
- `tests/app/test_spy_options_wheel.py` (212 lines) — parser + integration tests

---

### Task 1: Extend WeekRecord with hedge and IV percentile fields

**Files:**
- Modify: `src/stockdownloader/backtesting/engines/wheel.py:40-52`
- Modify: `tests/backtesting/engines/test_wheel.py:13-34`

**Step 1: Add new fields to WeekRecord**

Add three defaulted fields to the end of the `WeekRecord` dataclass (line 52):

```python
@dataclass(frozen=True, slots=True)
class WeekRecord:
    """Data for one trading week in the wheel backtest."""
    week_num: int
    expiration_date: str
    entry_date: str
    spy_price_at_entry: float
    spy_price_at_expiry: float
    put_strike: float
    call_strike: float
    put_premium: float
    call_premium: float
    ml_prob: float
    # Collar fields (default 0 = no hedge data)
    hedge_put_strike: float = 0.0
    hedge_put_premium: float = 0.0
    # Vol scaling field
    iv_percentile: float = 0.50
```

**Step 2: Update `_make_week` helper in test_wheel.py**

Replace the helper (lines 13-34):

```python
def _make_week(
    week_num: int,
    spy_open: float,
    spy_close: float,
    put_premium: float = 2.0,
    call_premium: float = 2.0,
    put_strike: float = 580.0,
    call_strike: float = 620.0,
    ml_prob: float = 0.50,
    hedge_put_strike: float = 0.0,
    hedge_put_premium: float = 0.0,
    iv_percentile: float = 0.50,
) -> WeekRecord:
    return WeekRecord(
        week_num=week_num,
        expiration_date=f"2025-02-{7 + week_num * 7:02d}",
        entry_date=f"2025-02-{3 + week_num * 7:02d}",
        spy_price_at_entry=spy_open,
        spy_price_at_expiry=spy_close,
        put_strike=put_strike,
        call_strike=call_strike,
        put_premium=put_premium,
        call_premium=call_premium,
        ml_prob=ml_prob,
        hedge_put_strike=hedge_put_strike,
        hedge_put_premium=hedge_put_premium,
        iv_percentile=iv_percentile,
    )
```

**Step 3: Run all existing tests to verify no regression**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py tests/app/test_spy_options_wheel.py -v`

Expected: All 53 tests PASS (34 engine + 19 CLI). The new fields have defaults so existing WeekRecord construction is unaffected.

**Step 4: Commit**

```bash
git add src/stockdownloader/backtesting/engines/wheel.py tests/backtesting/engines/test_wheel.py
git commit -m "feat: extend WeekRecord with hedge put and iv_percentile fields

Add hedge_put_strike, hedge_put_premium, and iv_percentile
fields (all with defaults) to WeekRecord dataclass. Update
_make_week test helper. No behavior change.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Add collar mode to WheelBacktestEngine

**Files:**
- Modify: `src/stockdownloader/backtesting/engines/wheel.py:78-364`
- Modify: `tests/backtesting/engines/test_wheel.py` (append new test class)

**Step 1: Write 8 failing tests**

Append to `tests/backtesting/engines/test_wheel.py`:

```python
class TestCollarMode:

    def test_collar_buys_protective_put(self):
        """Collar deducts hedge put cost from cash."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week)
        # Hedge cost = 1.50 * 100 = 150. Cash should be lower than without collar.
        metrics = engine.compute_metrics()
        assert metrics["total_hedge_cost"] == 150.0

    def test_collar_put_payout_on_crash(self):
        """Protective put pays out when SPY drops below hedge strike."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Payout = (490 - 470) * 100 = 2000
        assert metrics["total_hedge_payout"] == 2000.0

    def test_collar_otm_expires_worthless(self):
        """OTM hedge put: no payout, only cost deducted."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=510.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["total_hedge_cost"] == 150.0
        assert metrics["total_hedge_payout"] == 0.0

    def test_collar_reduces_net_premium(self):
        """Net premium with collar is lower due to hedge cost."""
        # Without collar
        engine_no = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1, combined=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine_no.process_week(week)
        premium_without = engine_no.total_premium_collected

        # With collar
        engine_yes = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        engine_yes.process_week(week)
        premium_with = engine_yes.total_premium_collected

        # Collar doesn't change premium collected, but cash is lower
        assert premium_with == premium_without  # premium tracking unchanged
        metrics = engine_yes.compute_metrics()
        assert metrics["total_hedge_cost"] > 0

    def test_collar_ml_filter_unaffected(self):
        """ML filter still skips CC/CSP normally with collar enabled."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            skip_call_thresh=0.65, combined=True, collar=True,
        )
        # ML prob 0.70 -> skip CC (rally expected)
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
            ml_prob=0.70,
        )
        engine.process_week(week, use_ml_filter=True)
        metrics = engine.compute_metrics()
        assert metrics["n_calls_skipped"] >= 1
        # Collar still bought despite ML filter
        assert metrics["total_hedge_cost"] > 0

    def test_collar_hedge_metrics_tracked(self):
        """Metrics dict includes hedge cost and payout keys."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        weeks = [
            _make_week(i, spy_open=500.0, spy_close=505.0,
                       hedge_put_strike=490.0, hedge_put_premium=1.0)
            for i in range(3)
        ]
        for w in weeks:
            engine.process_week(w)
        metrics = engine.compute_metrics()
        assert "total_hedge_cost" in metrics
        assert "total_hedge_payout" in metrics
        # 3 weeks * 1.0 * 100 = 300
        assert abs(metrics["total_hedge_cost"] - 300.0) < 0.01

    def test_collar_commission_on_hedge(self):
        """Commission is charged on hedge put contracts."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            commission_per_contract=0.65,
            combined=True, collar=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Commissions: 1 CC (0.65) + 1 CSP (0.65) + 1 hedge (0.65) = 1.95
        assert abs(metrics["total_commissions"] - 1.95) < 0.01

    def test_collar_with_csp_assignment_grows_hedge(self):
        """After CSP assignment, more shares means more hedge puts next week."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, collar=True,
        )
        # Week 0: Buy 100 shares, CSP assigned -> 200 shares
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
            hedge_put_strike=490.0, hedge_put_premium=1.50,
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: Now 200 shares, should buy 2 hedge puts
        week1 = _make_week(
            1, spy_open=470.0, spy_close=475.0,
            put_strike=460.0, put_premium=2.0,
            call_strike=490.0, call_premium=2.0,
            hedge_put_strike=465.0, hedge_put_premium=1.00,
        )
        engine.process_week(week1)
        metrics = engine.compute_metrics()
        # Week 0: 1 hedge * 1.50 * 100 = 150
        # Week 1: 2 hedges * 1.00 * 100 = 200
        # Total = 350
        assert abs(metrics["total_hedge_cost"] - 350.0) < 0.01
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py::TestCollarMode -v`

Expected: All 8 FAIL with `TypeError: __init__() got an unexpected keyword argument 'collar'`

**Step 3: Implement collar mode**

In `wheel.py`, modify `WheelBacktestEngine.__init__` (add after line 86):

```python
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        contracts: int = 1,
        skip_put_thresh: float = 0.35,
        skip_call_thresh: float = 0.65,
        buy_write: bool = False,
        combined: bool = False,
        commission_per_contract: float = 0.0,
        collar: bool = False,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._contracts = contracts
        self._skip_put_thresh = skip_put_thresh
        self._skip_call_thresh = skip_call_thresh
        self._buy_write = buy_write
        self._combined = combined
        self._commission_per_contract = commission_per_contract
        self._collar = collar

        # State
        self._state = WheelState.CASH
        self._shares: int = 0
        self._share_cost_basis: float = 0.0

        # Tracking
        self._total_premium: float = 0.0
        self._total_commissions: float = 0.0
        self._total_hedge_cost: float = 0.0
        self._total_hedge_payout: float = 0.0
        self._n_assignments: int = 0
        self._n_calls_exercised: int = 0
        self._n_rebuys: int = 0
        self._n_puts_sold: int = 0
        self._n_calls_sold: int = 0
        self._n_puts_skipped: int = 0
        self._n_calls_skipped: int = 0
        self._equity_curve: list[float] = []
        self._weeks_processed: int = 0
```

Modify `process_week()` to add collar logic between mode dispatch and equity curve (lines 131-150):

```python
    def process_week(
        self,
        week: WeekRecord,
        *,
        use_ml_filter: bool = False,
    ) -> None:
        """Process one week of the wheel strategy."""
        multiplier = self._contracts * 100

        if self._combined:
            self._process_week_combined(week, use_ml_filter)
        elif self._buy_write:
            self._process_week_buy_write(week, multiplier, use_ml_filter)
        else:
            self._process_week_wheel(week, multiplier, use_ml_filter)

        # Collar: buy protective put on all held shares
        if self._collar and self._shares > 0 and week.hedge_put_premium > 0:
            n_hedge = self._shares // 100
            hedge_cost = week.hedge_put_premium * n_hedge * 100
            commission = self._commission_per_contract * n_hedge
            self._cash -= hedge_cost + commission
            self._total_hedge_cost += hedge_cost
            self._total_commissions += commission

            if week.spy_price_at_expiry < week.hedge_put_strike:
                payout = (week.hedge_put_strike - week.spy_price_at_expiry) * n_hedge * 100
                self._cash += payout
                self._total_hedge_payout += payout

        # Update equity curve
        equity = self._cash + self._shares * week.spy_price_at_expiry
        self._equity_curve.append(equity)
        self._weeks_processed += 1
```

Add hedge metrics to `compute_metrics()` (add to the return dict, after line 363):

```python
            "annualized_return_pct": annualized,
            "total_hedge_cost": self._total_hedge_cost,
            "total_hedge_payout": self._total_hedge_payout,
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py -v`

Expected: All 42 tests PASS (34 existing + 8 collar).

**Step 5: Commit**

```bash
git add src/stockdownloader/backtesting/engines/wheel.py tests/backtesting/engines/test_wheel.py
git commit -m "feat: add protective put collar mode to wheel engine

Buy OTM protective puts on all held shares each week when
collar=True. Pays out when SPY drops below hedge strike.
Tracks total_hedge_cost and total_hedge_payout in metrics.
Works with combined, buy-write, and standard wheel modes.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Add volatility-scaled position sizing

**Files:**
- Modify: `src/stockdownloader/backtesting/engines/wheel.py`
- Modify: `tests/backtesting/engines/test_wheel.py` (append new test class)

**Step 1: Write 6 failing tests**

Append to `tests/backtesting/engines/test_wheel.py`:

```python
class TestVolScaling:

    def test_vol_scaling_high_iv_increases_csp_count(self):
        """High IV (80th pctile) -> 2x multiplier -> 2 CSPs sold."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=4.0,
            call_strike=520.0, call_premium=3.0,
            iv_percentile=0.80,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # High IV: mult=2.0, eff=min(3, 1*2)=2. Should sell 2 CSPs.
        assert metrics["n_puts_sold"] == 2

    def test_vol_scaling_low_iv_floors_at_one(self):
        """Low IV (10th pctile) -> 0.5x multiplier -> floor at 1 contract."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=1.0,
            call_strike=520.0, call_premium=1.0,
            iv_percentile=0.10,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Low IV: mult=0.5, int(1*0.5)=0, floor=1. Sell 1 CSP.
        assert metrics["n_puts_sold"] == 1

    def test_vol_scaling_respects_max_contracts(self):
        """Effective contracts capped at max_contracts."""
        engine = WheelBacktestEngine(
            initial_capital=500_000.0, contracts=2,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
            iv_percentile=0.90,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # mult=2.0, 2*2=4, cap at 3. Sell 3 CSPs.
        assert metrics["n_puts_sold"] == 3

    def test_vol_scaling_disabled_original_behavior(self):
        """vol_scaling=False -> same behavior as before."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1,
            combined=True, vol_scaling=False,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            iv_percentile=0.90,
        )
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Without vol scaling, ignores iv_percentile. Sell 1 CSP.
        assert metrics["n_puts_sold"] == 1

    def test_vol_scaling_cc_cap(self):
        """Vol scaling caps CC count in combined mode."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        # Week 0: CSP assigned -> 200 shares
        week0 = _make_week(
            0, spy_open=500.0, spy_close=470.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=2.0,
            iv_percentile=0.10,  # low IV -> eff=1
        )
        engine.process_week(week0)
        assert engine.shares_held == 200

        # Week 1: 200 shares but low IV -> only 1 CC sold (not 2)
        week1 = _make_week(
            1, spy_open=470.0, spy_close=475.0,
            put_strike=460.0, put_premium=2.0,
            call_strike=490.0, call_premium=2.0,
            iv_percentile=0.10,  # low IV -> eff=1
        )
        engine.process_week(week1)
        metrics = engine.compute_metrics()
        # Week 0: 1 CC (eff=1, shares=100). Week 1: 1 CC (eff=1, capped from 2).
        assert metrics["n_calls_sold"] == 2  # not 3

    def test_vol_scaling_metrics_tracked(self):
        """Metrics include avg and max contracts traded."""
        engine = WheelBacktestEngine(
            initial_capital=200_000.0, contracts=1,
            combined=True, vol_scaling=True, max_contracts=3,
        )
        weeks = [
            _make_week(0, spy_open=500.0, spy_close=505.0, iv_percentile=0.20),
            _make_week(1, spy_open=505.0, spy_close=510.0, iv_percentile=0.60),
            _make_week(2, spy_open=510.0, spy_close=515.0, iv_percentile=0.90),
        ]
        for w in weeks:
            engine.process_week(w)
        metrics = engine.compute_metrics()
        assert "avg_contracts_traded" in metrics
        assert "max_contracts_traded" in metrics
        # Week 0: 0.20 -> 0.5x -> 1. Week 1: 0.60 -> 1.5x -> 1. Week 2: 0.90 -> 2.0x -> 2.
        # Wait: int(1*0.5)=0, floor=1. int(1*1.5)=1. int(1*2.0)=2.
        assert metrics["max_contracts_traded"] == 2.0
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py::TestVolScaling -v`

Expected: All 6 FAIL with `TypeError: __init__() got an unexpected keyword argument 'vol_scaling'`

**Step 3: Implement vol scaling**

Add `vol_scaling` and `max_contracts` params to `__init__` (add after `collar`):

```python
        collar: bool = False,
        vol_scaling: bool = False,
        max_contracts: int = 3,
```

Store in init body:
```python
        self._collar = collar
        self._vol_scaling = vol_scaling
        self._max_contracts = max_contracts
```

Add tracking list:
```python
        self._eff_contracts_history: list[int] = []
```

Add `_compute_effective_contracts` method (before `process_week`):

```python
    def _compute_effective_contracts(self, iv_percentile: float) -> int:
        """Compute effective contract count from IV percentile."""
        if not self._vol_scaling:
            return self._contracts
        if iv_percentile <= 0.25:
            mult = 0.5
        elif iv_percentile <= 0.50:
            mult = 1.0
        elif iv_percentile <= 0.75:
            mult = 1.5
        else:
            mult = 2.0
        return max(1, min(self._max_contracts, int(self._contracts * mult)))
```

Modify `process_week()` to compute effective contracts and pass to combined mode:

```python
    def process_week(
        self,
        week: WeekRecord,
        *,
        use_ml_filter: bool = False,
    ) -> None:
        """Process one week of the wheel strategy."""
        multiplier = self._contracts * 100
        eff_contracts = self._compute_effective_contracts(week.iv_percentile)

        if self._vol_scaling:
            self._eff_contracts_history.append(eff_contracts)

        if self._combined:
            self._process_week_combined(week, use_ml_filter, eff_contracts)
        elif self._buy_write:
            self._process_week_buy_write(week, multiplier, use_ml_filter)
        else:
            self._process_week_wheel(week, multiplier, use_ml_filter)

        # Collar: buy protective put on all held shares
        ...  # (unchanged from Task 2)
```

Modify `_process_week_combined` signature and body to accept `eff_contracts`:

```python
    def _process_week_combined(
        self,
        week: WeekRecord,
        use_ml_filter: bool,
        eff_contracts: int | None = None,
    ) -> None:
        """Combined mode: buy-write + CSPs on idle cash."""
        if eff_contracts is None:
            eff_contracts = self._contracts
        multiplier = self._contracts * 100

        # Step 1: Buy initial shares if not holding (uses base contracts)
        if self._shares == 0:
            cost = week.spy_price_at_entry * multiplier
            self._cash -= cost
            self._shares = multiplier
            self._share_cost_basis = week.spy_price_at_entry
            self._state = WheelState.CALL_PHASE

        # Step 2: Sell covered calls on held shares
        n_cc = self._shares // 100
        if self._vol_scaling:
            n_cc = min(n_cc, eff_contracts)
        if n_cc > 0:
            if not (use_ml_filter and week.ml_prob > self._skip_call_thresh):
                gross = week.call_premium * n_cc * 100
                commission = self._commission_per_contract * n_cc
                self._cash += gross - commission
                self._total_premium += gross - commission
                self._total_commissions += commission
                self._n_calls_sold += n_cc

                if week.spy_price_at_expiry > week.call_strike:
                    self._cash += week.call_strike * n_cc * 100
                    self._cash -= week.spy_price_at_expiry * n_cc * 100
                    self._n_calls_exercised += n_cc
                    self._n_rebuys += n_cc
                    self._share_cost_basis = week.spy_price_at_expiry
            else:
                self._n_calls_skipped += n_cc

        # Step 3: Sell CSPs on available cash (capped at eff_contracts)
        if week.put_strike > 0:
            max_csp = min(eff_contracts, int(self._cash // (week.put_strike * 100)))
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
                    cost = week.put_strike * max_csp * 100
                    self._cash -= cost
                    self._shares += max_csp * 100
                    self._n_assignments += max_csp
            else:
                self._n_puts_skipped += max_csp

        self._state = WheelState.CALL_PHASE
```

Add vol scaling metrics to `compute_metrics()`:

```python
            "total_hedge_payout": self._total_hedge_payout,
            "avg_contracts_traded": (
                statistics.mean(self._eff_contracts_history)
                if self._eff_contracts_history
                else float(self._contracts)
            ),
            "max_contracts_traded": (
                float(max(self._eff_contracts_history))
                if self._eff_contracts_history
                else float(self._contracts)
            ),
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py -v`

Expected: All 48 tests PASS (34 existing + 8 collar + 6 vol scaling).

**Step 5: Commit**

```bash
git add src/stockdownloader/backtesting/engines/wheel.py tests/backtesting/engines/test_wheel.py
git commit -m "feat: add volatility-scaled position sizing to wheel engine

Scale effective contract count by IV percentile: 0-25th -> 0.5x,
25-50th -> 1.0x, 50-75th -> 1.5x, 75-100th -> 2.0x. Caps CCs
and CSPs in combined mode. Floor at 1, max at max_contracts param.
Tracks avg_contracts_traded and max_contracts_traded.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Create IronCondorEngine

**Files:**
- Create: `src/stockdownloader/backtesting/engines/iron_condor.py`
- Create: `tests/backtesting/engines/test_iron_condor.py`

**Step 1: Create skeleton files**

Create `src/stockdownloader/backtesting/engines/iron_condor.py`:

```python
"""Iron condor overlay backtest engine.

Sells weekly iron condors (put spread + call spread) for defined-risk
premium income. Profits in range-bound markets, complements the
directional combined strategy.

Iron condor structure:
    Buy OTM put (10-delta)  = long_put_strike
    Sell put (30-delta)     = short_put_strike
    Sell call (30-delta)    = short_call_strike
    Buy OTM call (10-delta) = long_call_strike

Max profit: net credit (SPY stays between short strikes)
Max loss: wider spread width * 100 - net credit * 100
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ICWeekRecord:
    """Data for one iron condor trading week."""
    week_num: int
    expiration_date: str
    spy_price_at_entry: float
    spy_price_at_expiry: float
    short_put_strike: float
    long_put_strike: float
    short_call_strike: float
    long_call_strike: float
    net_credit_per_contract: float
    ml_prob: float


class IronCondorEngine:
    """Runs weekly iron condor backtest.

    Parameters
    ----------
    capital:
        Starting cash allocated to IC strategy.
    commission_per_contract:
        Dollar commission per contract leg.
    skip_put_thresh:
        Skip IC when ML prob < this (strong bearish = directional).
    skip_call_thresh:
        Skip IC when ML prob > this (strong bullish = directional).
    """

    def __init__(
        self,
        capital: float,
        commission_per_contract: float = 0.0,
        skip_put_thresh: float = 0.35,
        skip_call_thresh: float = 0.65,
    ) -> None:
        self._initial_capital = capital
        self._cash = capital
        self._commission = commission_per_contract
        self._skip_put_thresh = skip_put_thresh
        self._skip_call_thresh = skip_call_thresh

        # Tracking
        self._total_credit: float = 0.0
        self._total_loss: float = 0.0
        self._total_commissions: float = 0.0
        self._n_ics_sold: int = 0
        self._n_ics_skipped: int = 0
        self._n_max_loss_events: int = 0
        self._equity_curve: list[float] = []
        self._weeks_processed: int = 0

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    def process_week(
        self,
        week: ICWeekRecord,
        *,
        use_ml_filter: bool = False,
    ) -> None:
        """Process one week of iron condor strategy."""
        raise NotImplementedError

    def compute_metrics(self) -> dict[str, float]:
        """Compute summary metrics for the IC backtest."""
        raise NotImplementedError
```

**Step 2: Write 10 failing tests**

Create `tests/backtesting/engines/test_iron_condor.py`:

```python
"""Tests for the iron condor overlay backtest engine."""
from __future__ import annotations

import pytest

from stockdownloader.backtesting.engines.iron_condor import (
    ICWeekRecord,
    IronCondorEngine,
)


def _make_ic_week(
    week_num: int,
    spy_open: float,
    spy_close: float,
    short_put_strike: float = 490.0,
    long_put_strike: float = 475.0,
    short_call_strike: float = 510.0,
    long_call_strike: float = 525.0,
    net_credit: float = 3.50,
    ml_prob: float = 0.50,
) -> ICWeekRecord:
    return ICWeekRecord(
        week_num=week_num,
        expiration_date=f"2025-02-{7 + week_num * 7:02d}",
        spy_price_at_entry=spy_open,
        spy_price_at_expiry=spy_close,
        short_put_strike=short_put_strike,
        long_put_strike=long_put_strike,
        short_call_strike=short_call_strike,
        long_call_strike=long_call_strike,
        net_credit_per_contract=net_credit,
        ml_prob=ml_prob,
    )


class TestIronCondorEngine:

    def test_ic_net_credit_collected(self):
        """IC collects net credit when sold."""
        engine = IronCondorEngine(capital=10_000.0)
        # Spread width=15, credit=3.50 -> max_loss=1150 -> n=floor(10000/1150)=8
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["total_credit"] == 3.50 * 8 * 100  # 2800
        assert metrics["n_ics_sold"] == 8

    def test_ic_max_profit_in_range(self):
        """SPY between short strikes -> keep full credit."""
        engine = IronCondorEngine(capital=10_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # No losses, just credit
        assert metrics["total_loss"] == 0.0
        assert metrics["final_equity"] > metrics["initial_capital"]

    def test_ic_max_loss_put_side(self):
        """SPY below long put -> max loss on put spread."""
        engine = IronCondorEngine(capital=10_000.0)
        # long_put=475, spy_close=460 (below long put)
        week = _make_ic_week(0, spy_open=500.0, spy_close=460.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Put spread max loss = (490-475)*100*8 = 12000
        # Net PnL = 2800 credit - 12000 loss = -9200
        assert metrics["total_loss"] == 12_000.0
        assert metrics["n_max_loss_events"] >= 1

    def test_ic_max_loss_call_side(self):
        """SPY above long call -> max loss on call spread."""
        engine = IronCondorEngine(capital=10_000.0)
        # long_call=525, spy_close=540 (above long call)
        week = _make_ic_week(0, spy_open=500.0, spy_close=540.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Call spread max loss = (525-510)*100*8 = 12000
        assert metrics["total_loss"] == 12_000.0
        assert metrics["n_max_loss_events"] >= 1

    def test_ic_partial_loss_put(self):
        """SPY between long and short put -> partial put loss."""
        engine = IronCondorEngine(capital=10_000.0)
        # short_put=490, long_put=475, spy_close=485
        week = _make_ic_week(0, spy_open=500.0, spy_close=485.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Put loss = (490-485)*100*8 = 4000
        assert metrics["total_loss"] == 4_000.0
        assert metrics["n_max_loss_events"] == 0

    def test_ic_partial_loss_call(self):
        """SPY between short and long call -> partial call loss."""
        engine = IronCondorEngine(capital=10_000.0)
        # short_call=510, long_call=525, spy_close=520
        week = _make_ic_week(0, spy_open=500.0, spy_close=520.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        # Call loss = (520-510)*100*8 = 8000
        assert metrics["total_loss"] == 8_000.0

    def test_ic_sizing_by_max_loss(self):
        """Position sized by max loss per IC."""
        engine = IronCondorEngine(capital=5_000.0)
        # Spread width=15, credit=3.50 -> max_loss=1150
        # n=floor(5000/1150)=4
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_ics_sold"] == 4

    def test_ic_four_leg_commission(self):
        """Commission charged for 4 legs per IC contract."""
        engine = IronCondorEngine(capital=10_000.0, commission_per_contract=0.65)
        # max_loss = 1150, commission_per_ic = 0.65*4 = 2.60
        # n = floor(10000 / (1150 + 2.60)) = floor(10000/1152.60) = 8
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        assert metrics["n_ics_sold"] == 8
        # 8 ICs * 4 legs * 0.65 = 20.80
        assert abs(metrics["total_commissions"] - 20.80) < 0.01

    def test_ic_ml_filter_skip(self):
        """Skips IC when ML signals strong directional move."""
        engine = IronCondorEngine(
            capital=10_000.0,
            skip_put_thresh=0.35,
            skip_call_thresh=0.65,
        )
        # Bearish signal -> skip
        week_bear = _make_ic_week(0, spy_open=500.0, spy_close=500.0, ml_prob=0.25)
        engine.process_week(week_bear, use_ml_filter=True)
        metrics = engine.compute_metrics()
        assert metrics["n_ics_skipped"] >= 1
        assert metrics["n_ics_sold"] == 0

    def test_ic_metrics_dict(self):
        """Returns expected keys in metrics dict."""
        engine = IronCondorEngine(capital=10_000.0)
        week = _make_ic_week(0, spy_open=500.0, spy_close=500.0)
        engine.process_week(week)
        metrics = engine.compute_metrics()
        expected_keys = {
            "initial_capital", "final_equity", "total_return_pct",
            "total_credit", "total_loss", "total_commissions",
            "n_ics_sold", "n_ics_skipped", "n_max_loss_events",
            "weeks", "sharpe", "max_drawdown_pct", "annualized_return_pct",
        }
        assert expected_keys.issubset(set(metrics.keys()))
```

**Step 3: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_iron_condor.py -v`

Expected: All 10 FAIL with `NotImplementedError`

**Step 4: Implement IronCondorEngine.process_week**

Replace the `raise NotImplementedError` in `process_week`:

```python
    def process_week(
        self,
        week: ICWeekRecord,
        *,
        use_ml_filter: bool = False,
    ) -> None:
        """Process one week of iron condor strategy."""
        # ML filter: skip when strong directional signal
        if use_ml_filter and (
            week.ml_prob < self._skip_put_thresh
            or week.ml_prob > self._skip_call_thresh
        ):
            self._n_ics_skipped += 1
            self._equity_curve.append(self._cash)
            self._weeks_processed += 1
            return

        # Position sizing
        put_spread_width = week.short_put_strike - week.long_put_strike
        call_spread_width = week.long_call_strike - week.short_call_strike
        wider = max(put_spread_width, call_spread_width)
        max_loss_per = wider * 100 - week.net_credit_per_contract * 100

        if max_loss_per <= 0:
            max_loss_per = 1.0

        commission_per_ic = self._commission * 4
        denom = max_loss_per + commission_per_ic
        n_contracts = int(self._cash // denom) if denom > 0 else 0

        if n_contracts < 1:
            self._equity_curve.append(self._cash)
            self._weeks_processed += 1
            return

        # Collect credit
        credit = week.net_credit_per_contract * n_contracts * 100
        commission = commission_per_ic * n_contracts
        self._cash += credit - commission
        self._total_credit += credit
        self._total_commissions += commission
        self._n_ics_sold += n_contracts

        # Settlement
        spy = week.spy_price_at_expiry

        # Put spread loss
        if spy <= week.long_put_strike:
            put_loss = put_spread_width * n_contracts * 100
            self._n_max_loss_events += 1
        elif spy < week.short_put_strike:
            put_loss = (week.short_put_strike - spy) * n_contracts * 100
        else:
            put_loss = 0.0

        # Call spread loss
        if spy >= week.long_call_strike:
            call_loss = call_spread_width * n_contracts * 100
            self._n_max_loss_events += 1
        elif spy > week.short_call_strike:
            call_loss = (spy - week.short_call_strike) * n_contracts * 100
        else:
            call_loss = 0.0

        total_loss = put_loss + call_loss
        self._cash -= total_loss
        self._total_loss += total_loss

        self._equity_curve.append(self._cash)
        self._weeks_processed += 1
```

**Step 5: Implement compute_metrics**

Replace the `raise NotImplementedError` in `compute_metrics`:

```python
    def compute_metrics(self) -> dict[str, float]:
        """Compute summary metrics for the IC backtest."""
        final_equity = self._equity_curve[-1] if self._equity_curve else self._initial_capital
        total_return = final_equity - self._initial_capital
        total_return_pct = (total_return / self._initial_capital) * 100 if self._initial_capital > 0 else 0.0

        # Sharpe (weekly returns, annualized)
        sharpe = 0.0
        if len(self._equity_curve) >= 2:
            returns = []
            prev = self._initial_capital
            for eq in self._equity_curve:
                returns.append((eq - prev) / prev if prev > 0 else 0.0)
                prev = eq
            if returns:
                mean_r = statistics.mean(returns)
                std_r = statistics.pstdev(returns)
                if std_r > 0:
                    sharpe = (mean_r / std_r) * math.sqrt(52)

        # Max drawdown
        max_dd_pct = 0.0
        peak = self._initial_capital
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        # Annualized
        years = self._weeks_processed / 52.0 if self._weeks_processed > 0 else 1.0
        annualized = 0.0
        if years > 0 and final_equity > 0 and self._initial_capital > 0:
            annualized = ((final_equity / self._initial_capital) ** (1.0 / years) - 1.0) * 100

        return {
            "initial_capital": self._initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "total_return_dollar": total_return,
            "total_credit": self._total_credit,
            "total_loss": self._total_loss,
            "total_commissions": self._total_commissions,
            "n_ics_sold": float(self._n_ics_sold),
            "n_ics_skipped": float(self._n_ics_skipped),
            "n_max_loss_events": float(self._n_max_loss_events),
            "weeks": float(self._weeks_processed),
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd_pct,
            "annualized_return_pct": annualized,
        }
```

**Step 6: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_iron_condor.py -v`

Expected: All 10 PASS.

Also run full engine test suite:

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/ -v`

Expected: All 58 tests PASS (48 wheel + 10 IC).

**Step 7: Commit**

```bash
git add src/stockdownloader/backtesting/engines/iron_condor.py tests/backtesting/engines/test_iron_condor.py
git commit -m "feat: add IronCondorEngine for defined-risk IC overlay

Weekly iron condor backtest: sell 30-delta put/call spreads with
10-delta wings. Position sized by max loss. Settlement handles
in-range (full credit), partial loss, and max loss scenarios.
ML filter skips when strong directional signal detected.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: CLI integration

**Files:**
- Modify: `src/stockdownloader/app/spy_options_wheel.py`
- Modify: `tests/app/test_spy_options_wheel.py`

**Step 1: Add parser arguments**

In `_build_parser()`, add after the `--no-ml-filter` argument (after line 103):

```python
    # Advanced strategies
    parser.add_argument(
        "--collar", action="store_true",
        help="Enable protective put collar on held shares",
    )
    parser.add_argument(
        "--hedge-delta", type=float, default=0.10,
        help="Target delta for hedge put (default: 0.10)",
    )
    parser.add_argument(
        "--vol-scaling", action="store_true",
        help="Enable IV-percentile-based contract scaling",
    )
    parser.add_argument(
        "--max-contracts", type=int, default=3,
        help="Maximum contracts with vol scaling (default: 3)",
    )
    parser.add_argument(
        "--iron-condor", action="store_true",
        help="Enable iron condor overlay (30/10-delta spreads)",
    )
    parser.add_argument(
        "--ic-allocation", type=float, default=0.30,
        help="Capital fraction for iron condor overlay (default: 0.30)",
    )
```

**Step 2: Write parser tests**

Append to `tests/app/test_spy_options_wheel.py`, inside a new class:

```python
class TestAdvancedParser:

    def test_collar_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.collar is False

    def test_collar_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--collar"])
        assert args.collar is True

    def test_hedge_delta_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.hedge_delta == 0.10

    def test_hedge_delta_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--hedge-delta", "0.15"])
        assert args.hedge_delta == 0.15

    def test_vol_scaling_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vol_scaling is False

    def test_vol_scaling_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--vol-scaling"])
        assert args.vol_scaling is True

    def test_max_contracts_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.max_contracts == 3

    def test_iron_condor_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.iron_condor is False

    def test_iron_condor_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--iron-condor"])
        assert args.iron_condor is True

    def test_ic_allocation_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.ic_allocation == 0.30

    def test_ic_allocation_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--ic-allocation", "0.20"])
        assert args.ic_allocation == 0.20
```

**Step 3: Run parser tests**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/app/test_spy_options_wheel.py::TestAdvancedParser -v`

Expected: All 11 PASS (parser tests don't need engine changes).

**Step 4: Wire CLI pipeline**

In `main()`, update the header printout to show advanced strategy flags (after line 381):

```python
    if args.collar:
        print(f"  Collar:        ON (hedge delta: {args.hedge_delta})")
    if args.vol_scaling:
        print(f"  Vol Scaling:   ON (max contracts: {args.max_contracts})")
    if args.iron_condor:
        print(f"  Iron Condor:   ON (allocation: {args.ic_allocation:.0%})")
```

In the `[3/5]` section, add fetching of 10-delta contracts (after the existing `select_strike_by_delta` calls for 30-delta, around line 497):

```python
        hedge_put_contract = None
        hedge_call_contract = None  # for IC long call wing
        if args.collar or args.iron_condor:
            hedge_put_contract = select_strike_by_delta(
                week_contracts, contract_type="put", spot=spy_at_entry,
                target_delta=args.hedge_delta, days_to_expiry=5, volatility=hist_vol,
            )
        if args.iron_condor:
            hedge_call_contract = select_strike_by_delta(
                week_contracts, contract_type="call", spot=spy_at_entry,
                target_delta=args.hedge_delta, days_to_expiry=5, volatility=hist_vol,
            )
```

Fetch bars for hedge contracts (add to the cache/fetch block):

```python
        hedge_put_premium = 0.0
        hedge_put_strike = 0.0
        if hedge_put_contract:
            # Try cache first, then fetch
            hedge_put_bar = None
            if cached and hedge_put_contract["ticker"] in cached.get("bars", {}):
                hedge_put_bar = cached["bars"].get(hedge_put_contract["ticker"])
            else:
                hedge_put_bar = polygon.fetch_option_daily_bar(hedge_put_contract["ticker"], monday)
            if hedge_put_bar:
                hedge_put_premium = hedge_put_bar.get("vw", hedge_put_bar.get("c", 0.0))
                hedge_put_strike = hedge_put_contract["strike_price"]

        hedge_call_premium = 0.0
        hedge_call_strike = 0.0
        if hedge_call_contract:
            hedge_call_bar = None
            if cached and hedge_call_contract["ticker"] in cached.get("bars", {}):
                hedge_call_bar = cached["bars"].get(hedge_call_contract["ticker"])
            else:
                hedge_call_bar = polygon.fetch_option_daily_bar(hedge_call_contract["ticker"], monday)
            if hedge_call_bar:
                hedge_call_premium = hedge_call_bar.get("vw", hedge_call_bar.get("c", 0.0))
                hedge_call_strike = hedge_call_contract["strike_price"]
```

Update WeekRecord construction to include new fields (around line 562):

```python
        week_records.append(WeekRecord(
            week_num=week_num,
            expiration_date=exp_str,
            entry_date=mon_str,
            spy_price_at_entry=spy_at_entry,
            spy_price_at_expiry=spy_at_expiry,
            put_strike=put_contract["strike_price"],
            call_strike=call_contract["strike_price"],
            put_premium=put_premium,
            call_premium=call_premium,
            ml_prob=ml_prob,
            hedge_put_strike=hedge_put_strike,
            hedge_put_premium=hedge_put_premium,
            iv_percentile=iv_percentile,
        ))
```

Build IC week records if iron condor is enabled (after the WeekRecord append):

```python
        if args.iron_condor and hedge_put_strike > 0 and hedge_call_strike > 0:
            # Net credit = (sell 30d put - buy 10d put) + (sell 30d call - buy 10d call)
            put_spread_credit = put_premium - hedge_put_premium
            call_spread_credit = call_premium - hedge_call_premium
            net_credit = put_spread_credit + call_spread_credit
            if net_credit > 0:
                ic_week_records.append(ICWeekRecord(
                    week_num=week_num,
                    expiration_date=exp_str,
                    spy_price_at_entry=spy_at_entry,
                    spy_price_at_expiry=spy_at_expiry,
                    short_put_strike=put_contract["strike_price"],
                    long_put_strike=hedge_put_strike,
                    short_call_strike=call_contract["strike_price"],
                    long_call_strike=hedge_call_strike,
                    net_credit_per_contract=net_credit,
                    ml_prob=ml_prob,
                ))
```

Initialize `ic_week_records` list early (before the loop):

```python
    ic_week_records: list = []
```

Add the IC import at the top of `main()`:

```python
    if args.iron_condor:
        from stockdownloader.backtesting.engines.iron_condor import (
            ICWeekRecord,
            IronCondorEngine,
        )
```

**Step 5: Wire engine runs with capital split**

In the `[4/5]` section where engines are created, handle capital split (modify the engine creation block, around line 579):

```python
    # Capital allocation
    if args.iron_condor:
        combined_capital = args.initial_capital * (1 - args.ic_allocation)
        ic_capital = args.initial_capital * args.ic_allocation
    else:
        combined_capital = args.initial_capital
        ic_capital = 0.0
```

Update all engine creations to use `combined_capital` instead of `args.initial_capital`.

Run IC engine when enabled (after the mechanical wheel engine run):

```python
    # Run IC engine if enabled
    ic_metrics = None
    if args.iron_condor and ic_week_records:
        from stockdownloader.backtesting.engines.iron_condor import IronCondorEngine
        ic_engine = IronCondorEngine(
            capital=ic_capital,
            commission_per_contract=args.commission,
            skip_put_thresh=args.skip_put_thresh,
            skip_call_thresh=args.skip_call_thresh,
        )
        for w in ic_week_records:
            ic_engine.process_week(w, use_ml_filter=not args.no_ml_filter and bool(ml_probs))
        ic_metrics = ic_engine.compute_metrics()
        ic_equity = ic_engine.equity_curve
```

**Step 6: Add merged metrics helper**

Add a helper function (before `main()`):

```python
def _merge_metrics(
    combined_metrics: dict[str, float],
    ic_metrics: dict[str, float],
    combined_equity: list[float],
    ic_equity: list[float],
    total_initial: float,
) -> dict[str, float]:
    """Merge combined + IC metrics into a single metrics dict."""
    # Align equity curves (pad shorter with last value)
    max_len = max(len(combined_equity), len(ic_equity))
    c_eq = combined_equity + [combined_equity[-1]] * (max_len - len(combined_equity)) if combined_equity else [0.0] * max_len
    i_eq = ic_equity + [ic_equity[-1]] * (max_len - len(ic_equity)) if ic_equity else [0.0] * max_len

    merged_equity = [c + i for c, i in zip(c_eq, i_eq)]
    final_equity = merged_equity[-1] if merged_equity else total_initial

    total_return = final_equity - total_initial
    total_return_pct = (total_return / total_initial) * 100 if total_initial > 0 else 0.0

    # Sharpe from merged curve
    sharpe = 0.0
    if len(merged_equity) >= 2:
        returns = []
        prev = total_initial
        for eq in merged_equity:
            returns.append((eq - prev) / prev if prev > 0 else 0.0)
            prev = eq
        if returns:
            mean_r = statistics.mean(returns)
            std_r = statistics.pstdev(returns)
            if std_r > 0:
                sharpe = (mean_r / std_r) * math.sqrt(52)

    # Max drawdown from merged curve
    max_dd_pct = 0.0
    peak = total_initial
    for eq in merged_equity:
        if eq > peak:
            peak = eq
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    weeks = max(combined_metrics.get("weeks", 0), ic_metrics.get("weeks", 0))
    years = weeks / 52.0 if weeks > 0 else 1.0
    annualized = 0.0
    if years > 0 and final_equity > 0 and total_initial > 0:
        annualized = ((final_equity / total_initial) ** (1.0 / years) - 1.0) * 100

    return {
        "initial_capital": total_initial,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "total_return_dollar": total_return,
        "total_premium_collected": combined_metrics.get("total_premium_collected", 0) + ic_metrics.get("total_credit", 0),
        "total_commissions": combined_metrics.get("total_commissions", 0) + ic_metrics.get("total_commissions", 0),
        "n_assignments": combined_metrics.get("n_assignments", 0),
        "n_calls_exercised": combined_metrics.get("n_calls_exercised", 0),
        "n_rebuys": combined_metrics.get("n_rebuys", 0),
        "n_puts_sold": combined_metrics.get("n_puts_sold", 0),
        "n_calls_sold": combined_metrics.get("n_calls_sold", 0),
        "n_puts_skipped": combined_metrics.get("n_puts_skipped", 0),
        "n_calls_skipped": combined_metrics.get("n_calls_skipped", 0),
        "n_ics_sold": ic_metrics.get("n_ics_sold", 0),
        "weeks": weeks,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_dollar": 0.0,
        "annualized_return_pct": annualized,
        "total_hedge_cost": combined_metrics.get("total_hedge_cost", 0),
        "total_hedge_payout": combined_metrics.get("total_hedge_payout", 0),
    }
```

Add `import statistics` at the top of the file (it's already imported indirectly, but add for clarity).

**Step 7: Wire engine params and comparison table**

Update engine construction to pass collar/vol_scaling params (in the combined mode block):

```python
        ml_combined = WheelBacktestEngine(
            initial_capital=combined_capital,
            contracts=args.contracts,
            skip_put_thresh=args.skip_put_thresh,
            skip_call_thresh=args.skip_call_thresh,
            combined=True,
            commission_per_contract=args.commission,
            collar=args.collar,
            vol_scaling=args.vol_scaling,
            max_contracts=args.max_contracts,
        )
```

Same for `mech_combined` engine.

For the comparison table, build merged metrics when IC is enabled and pass as a new first column:

```python
        # Merge IC + Combined if iron condor enabled
        ic_combined_metrics = None
        if ic_metrics and ml_combined_metrics:
            ic_combined_metrics = _merge_metrics(
                ml_combined_metrics, ic_metrics,
                ml_combined.equity_curve, ic_equity,
                args.initial_capital,
            )

        _print_comparison_table(
            ic_combined_metrics,  # new: IC+ML Combined (or None)
            ml_combined_metrics, mech_combined_metrics,
            bw_metrics, wheel_metrics, bh_return_pct,
        )
```

Update `_print_comparison_table` signature to accept 6 params:

```python
def _print_comparison_table(
    ic_combined: dict[str, float] | None,
    ml_combined: dict[str, float] | None,
    mech_combined: dict[str, float],
    bw_metrics: dict[str, float],
    wheel_metrics: dict[str, float],
    bh_return_pct: float,
) -> None:
```

Update the column-building logic:

```python
    cols = []
    if ic_combined:
        cols.append(("IC+ML Comb", ic_combined))
    if ml_combined:
        cols.append(("ML Combined", ml_combined))
    cols.append(("Combined", mech_combined))
    cols.append(("Buy-Write", bw_metrics))
    cols.append(("Wheel", wheel_metrics))
```

**Step 8: Run all tests**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/ tests/app/test_spy_options_wheel.py -v`

Expected: All tests PASS. May need to update existing `_print_comparison_table` call sites to pass the new `ic_combined=None` parameter.

**Step 9: Commit**

```bash
git add src/stockdownloader/app/spy_options_wheel.py tests/app/test_spy_options_wheel.py
git commit -m "feat: wire collar, vol-scaling, and iron condor into CLI

Add --collar, --hedge-delta, --vol-scaling, --max-contracts,
--iron-condor, --ic-allocation parser flags. Fetch 10-delta
contracts for collar/IC. Capital split for IC overlay. Merge
equity curves for IC+Combined comparison column.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Live backtest

**Step 1: Run full test suite**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q`

Expected: All tests PASS.

**Step 2: Run live backtest with all strategies**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --collar --vol-scaling --iron-condor --quick`

Expected output: Comparison table with columns IC+ML Combined | ML Combined | Combined | Buy-Write | Wheel | Buy & Hold.

**Step 3: Run without IC for comparison**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --collar --vol-scaling --quick`

Expected: Table without IC column. Collar + vol scaling should show improved Sharpe and lower drawdown vs base ML Combined.

**Step 4: Commit results (if applicable)**

If any output artifacts are generated, commit them. Otherwise, just verify the numbers look reasonable:

- Target: total return +65-70% (up from +55.3%)
- Target: Sharpe ~1.8-2.0 (up from 1.62)
- Target: Max drawdown ~-5-7% (down from -12.3%)

---

## Verification Commands

```bash
# Unit tests only (fast)
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py tests/backtesting/engines/test_iron_condor.py tests/app/test_spy_options_wheel.py -v

# Full regression
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q

# Live run: all strategies
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --collar --vol-scaling --iron-condor --quick

# Live run: collar + vol scaling only
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --collar --vol-scaling --quick

# Live run: baseline comparison
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --quick
```
