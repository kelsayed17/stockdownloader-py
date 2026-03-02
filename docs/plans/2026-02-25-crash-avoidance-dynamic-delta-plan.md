# Crash Avoidance + Dynamic Delta Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add crash avoidance (exit to cash on strong bearish ML signal) and dynamic delta (IV-based strike selection) to the wheel pipeline to improve Sharpe and reduce drawdown.

**Architecture:** Crash avoidance is a pre-check in `WheelBacktestEngine.process_week()` — liquidates shares when ML prob drops below threshold, re-enters when it recovers. Dynamic delta is CLI-level logic in `spy_options_wheel.py` that varies the delta passed to `select_strike_by_delta` per-week based on IV percentile. The ML ensemble pipeline already has crash avoidance implemented — no changes needed there.

**Tech Stack:** Python 3.11, pytest, dataclasses

---

### Task 1: Add crash avoidance to WheelBacktestEngine

**Files:**
- Modify: `src/stockdownloader/backtesting/engines/wheel.py`
- Modify: `tests/backtesting/engines/test_wheel.py`

**Step 1: Add constructor params and state tracking**

In `WheelBacktestEngine.__init__()`, add three new parameters after `max_contracts`:

```python
        crash_avoidance: bool = False,
        crash_exit_thresh: float = 0.25,
        re_entry_thresh: float = 0.50,
```

Store them and add state/tracking fields in `__init__`:

```python
        self._crash_avoidance = crash_avoidance
        self._crash_exit_thresh = crash_exit_thresh
        self._re_entry_thresh = re_entry_thresh

        # Crash avoidance state
        self._in_cash_mode: bool = False
        self._n_crash_exits: int = 0
        self._n_crash_reentries: int = 0
```

**Step 2: Add crash avoidance pre-check in process_week()**

At the top of `process_week()`, after the `multiplier = self._contracts * 100` line (line 175) and before the `eff_contracts` computation (line 177), insert:

```python
        # Crash avoidance: exit to cash or skip while in cash mode
        if self._crash_avoidance and use_ml_filter:
            if not self._in_cash_mode and week.ml_prob < self._crash_exit_thresh:
                # Liquidate all shares at expiry price
                if self._shares > 0:
                    self._cash += self._shares * week.spy_price_at_expiry
                    self._shares = 0
                    self._share_cost_basis = 0.0
                    self._state = WheelState.CASH
                self._in_cash_mode = True
                self._n_crash_exits += 1
                # Record equity and return (skip all option selling)
                self._equity_curve.append(self._cash)
                self._weeks_processed += 1
                return

            if self._in_cash_mode and week.ml_prob >= self._re_entry_thresh:
                self._in_cash_mode = False
                self._n_crash_reentries += 1
                # Fall through to normal processing (combined mode will buy shares)

            if self._in_cash_mode:
                # Still in cash — skip everything
                self._equity_curve.append(self._cash)
                self._weeks_processed += 1
                return
```

**Step 3: Add crash avoidance metrics to compute_metrics()**

In the `return {` block of `compute_metrics()`, add after the `max_contracts_traded` key:

```python
            "n_crash_exits": float(self._n_crash_exits),
            "n_crash_reentries": float(self._n_crash_reentries),
```

**Step 4: Write 7 failing tests**

Append to `tests/backtesting/engines/test_wheel.py`:

```python
class TestCrashAvoidance:

    def test_crash_avoidance_starts_invested(self):
        """With neutral probs, crash avoidance stays invested normally."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
        )
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week, use_ml_filter=True)
        assert engine.shares_held >= 100  # bought shares like normal combined

    def test_crash_avoidance_exits_on_strong_bearish(self):
        """Liquidates shares when ML prob drops below exit threshold."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25, re_entry_thresh=0.50,
        )
        # Week 0: normal, buys shares
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)
        assert engine.shares_held >= 100

        # Week 1: strong bearish -> exit
        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=1.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=True)
        assert engine.shares_held == 0
        metrics = engine.compute_metrics()
        assert metrics["n_crash_exits"] == 1

    def test_crash_avoidance_re_enters_on_recovery(self):
        """Re-enters market when prob recovers above re-entry threshold."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25, re_entry_thresh=0.50,
        )
        # Week 0: invest
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)

        # Week 1: exit
        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=1.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=True)
        assert engine.shares_held == 0

        # Week 2: still bearish (0.40 < 0.50) -> stay in cash
        week2 = _make_week(
            2, spy_open=490.0, spy_close=485.0,
            put_strike=470.0, put_premium=3.0,
            call_strike=510.0, call_premium=1.0,
            ml_prob=0.40,
        )
        engine.process_week(week2, use_ml_filter=True)
        assert engine.shares_held == 0

        # Week 3: recovery (0.55 >= 0.50) -> re-enter
        week3 = _make_week(
            3, spy_open=485.0, spy_close=495.0,
            put_strike=470.0, put_premium=2.0,
            call_strike=510.0, call_premium=2.0,
            ml_prob=0.55,
        )
        engine.process_week(week3, use_ml_filter=True)
        assert engine.shares_held >= 100
        metrics = engine.compute_metrics()
        assert metrics["n_crash_reentries"] == 1

    def test_crash_avoidance_skips_options_while_in_cash(self):
        """No options sold while in cash mode."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25,
        )
        # Week 0: invest
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)
        m0 = engine.compute_metrics()

        # Week 1: exit
        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=5.0,
            call_strike=520.0, call_premium=5.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=True)
        m1 = engine.compute_metrics()

        # Premium should NOT increase in week 1 (no options sold during exit)
        assert m1["total_premium_collected"] == m0["total_premium_collected"]

    def test_crash_avoidance_equity_preserved_in_cash(self):
        """Equity in cash mode is just cash (no share exposure)."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25,
        )
        # Week 0: invest at 500
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)

        # Week 1: exit at 490
        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=1.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=True)
        cash_after_exit = engine._cash

        # Week 2: market crashes to 450, but we're in cash
        week2 = _make_week(
            2, spy_open=490.0, spy_close=450.0,
            put_strike=470.0, put_premium=5.0,
            call_strike=510.0, call_premium=0.5,
            ml_prob=0.15,
        )
        engine.process_week(week2, use_ml_filter=True)
        # Cash unchanged (no shares, no options)
        assert engine._cash == cash_after_exit
        assert engine.equity_curve[-1] == cash_after_exit

    def test_crash_avoidance_disabled_by_default(self):
        """crash_avoidance=False preserves original behavior."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=False,
        )
        # Strong bearish signal but crash avoidance disabled
        week = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.20,
        )
        engine.process_week(week, use_ml_filter=True)
        # Should still buy shares (combined mode)
        assert engine.shares_held >= 100

    def test_crash_avoidance_requires_ml_filter(self):
        """Crash avoidance only triggers when use_ml_filter=True."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            combined=True, crash_avoidance=True,
            crash_exit_thresh=0.25,
        )
        week0 = _make_week(
            0, spy_open=500.0, spy_close=505.0,
            put_strike=480.0, put_premium=2.0,
            call_strike=520.0, call_premium=2.0,
            ml_prob=0.50,
        )
        engine.process_week(week0, use_ml_filter=True)
        assert engine.shares_held >= 100

        # Strong bearish prob but use_ml_filter=False -> no exit
        week1 = _make_week(
            1, spy_open=505.0, spy_close=490.0,
            put_strike=480.0, put_premium=3.0,
            call_strike=520.0, call_premium=1.0,
            ml_prob=0.20,
        )
        engine.process_week(week1, use_ml_filter=False)
        assert engine.shares_held >= 100  # still holding
```

**Step 5: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py::TestCrashAvoidance -v`

Expected: All 7 FAIL (constructor doesn't accept crash_avoidance param yet).

**Step 6: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py -v`

Expected: All 55 tests PASS (48 existing + 7 new crash avoidance).

**Step 7: Commit**

```bash
git add src/stockdownloader/backtesting/engines/wheel.py tests/backtesting/engines/test_wheel.py
git commit -m "feat: add crash avoidance mode to wheel backtest engine

Exit to cash when ML prob drops below crash_exit_thresh (0.25),
re-enter when prob recovers above re_entry_thresh (0.50).
Zero premium cost — uses ML signal instead of buying puts.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Add dynamic delta CLI flag and per-week delta logic

**Files:**
- Modify: `src/stockdownloader/app/spy_options_wheel.py`
- Modify: `tests/app/test_spy_options_wheel.py`

**Step 1: Add parser arguments**

In `_build_parser()`, add after the `--ic-allocation` argument (after line 129):

```python
    parser.add_argument(
        "--crash-avoidance", action="store_true",
        help="Exit to cash on strong bearish ML signal, re-enter on recovery",
    )
    parser.add_argument(
        "--crash-exit-thresh", type=float, default=0.25,
        help="Exit threshold for crash avoidance (default: 0.25)",
    )
    parser.add_argument(
        "--re-entry-thresh", type=float, default=0.50,
        help="Re-entry threshold for crash avoidance (default: 0.50)",
    )
    parser.add_argument(
        "--dynamic-delta", action="store_true",
        help="Vary strike delta by IV percentile (skip <25th, 0.20/0.30/0.40)",
    )
```

**Step 2: Write 7 parser tests**

Append to `tests/app/test_spy_options_wheel.py`:

```python
class TestCrashAvoidanceDynamicDeltaParser:

    def test_crash_avoidance_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.crash_avoidance is False

    def test_crash_avoidance_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--crash-avoidance"])
        assert args.crash_avoidance is True

    def test_crash_exit_thresh_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.crash_exit_thresh == 0.25

    def test_crash_exit_thresh_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--crash-exit-thresh", "0.30"])
        assert args.crash_exit_thresh == 0.30

    def test_re_entry_thresh_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.re_entry_thresh == 0.50

    def test_dynamic_delta_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.dynamic_delta is False

    def test_dynamic_delta_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--dynamic-delta"])
        assert args.dynamic_delta is True
```

**Step 3: Run parser tests**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/app/test_spy_options_wheel.py::TestCrashAvoidanceDynamicDeltaParser -v`

Expected: All 7 PASS.

**Step 4: Wire crash avoidance into header printout**

In `main()`, after the iron condor header line (after line 487), add:

```python
    if args.crash_avoidance:
        print(f"  Crash Avoid:   ON (exit: {args.crash_exit_thresh}, re-enter: {args.re_entry_thresh})")
    if args.dynamic_delta:
        print(f"  Dynamic Delta: ON (skip <25th, 0.20/0.30/0.40 by IV)")
```

**Step 5: Add dynamic delta logic to the weekly loop**

In `main()`, inside the weekly loop, after the `iv_percentile` computation (after line 693) and before the `if put_premium <= 0 and call_premium <= 0:` check (line 695), add:

```python
        # Dynamic delta: choose delta based on IV percentile
        if args.dynamic_delta:
            if iv_percentile < 0.25:
                continue  # Skip week — premium too thin
            elif iv_percentile < 0.50:
                week_delta = 0.20
            elif iv_percentile < 0.75:
                week_delta = 0.30
            else:
                week_delta = 0.40

            # Re-select strikes with dynamic delta
            put_contract = select_strike_by_delta(
                week_contracts, contract_type="put", spot=spy_at_entry,
                target_delta=week_delta, days_to_expiry=5, volatility=hist_vol,
            )
            call_contract = select_strike_by_delta(
                week_contracts, contract_type="call", spot=spy_at_entry,
                target_delta=week_delta, days_to_expiry=5, volatility=hist_vol,
            )
            if put_contract is None or call_contract is None:
                continue

            # Re-fetch premiums for new strikes
            put_bar_dd = None
            call_bar_dd = None
            if cached and put_contract["ticker"] in cached.get("bars", {}):
                put_bar_dd = cached["bars"].get(put_contract["ticker"])
            else:
                put_bar_dd = polygon.fetch_option_daily_bar(put_contract["ticker"], monday)
            if cached and call_contract["ticker"] in cached.get("bars", {}):
                call_bar_dd = cached["bars"].get(call_contract["ticker"])
            else:
                call_bar_dd = polygon.fetch_option_daily_bar(call_contract["ticker"], monday)

            put_premium = put_bar_dd.get("vw", put_bar_dd.get("c", 0.0)) if put_bar_dd else 0.0
            call_premium = call_bar_dd.get("vw", call_bar_dd.get("c", 0.0)) if call_bar_dd else 0.0
```

**Step 6: Wire crash avoidance into engine construction**

In `main()`, update ALL `WheelBacktestEngine` constructors to pass crash avoidance params. For each engine creation (mech_engine ~line 753, ml_engine ~line 769, ml_combined ~line 814, mech_combined ~line 830), add:

```python
            crash_avoidance=args.crash_avoidance,
            crash_exit_thresh=args.crash_exit_thresh,
            re_entry_thresh=args.re_entry_thresh,
```

Note: Only add to the ML-filtered engines (ml_engine, ml_combined) and the mech engines that could use it. Since crash avoidance only activates when `use_ml_filter=True`, it's safe to pass to all engines.

**Step 7: Run all tests**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py tests/app/test_spy_options_wheel.py -v`

Expected: All tests PASS (55 wheel + 37 CLI).

**Step 8: Commit**

```bash
git add src/stockdownloader/app/spy_options_wheel.py tests/app/test_spy_options_wheel.py
git commit -m "feat: add crash avoidance + dynamic delta to wheel CLI

Add --crash-avoidance, --crash-exit-thresh, --re-entry-thresh,
--dynamic-delta parser flags. Dynamic delta varies strike selection
by IV percentile (skip <25th, 0.20/0.30/0.40). Wire crash avoidance
params through to all engine constructors.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Live backtest

**Step 1: Run full test suite**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q`

Expected: All tests PASS.

**Step 2: Run with crash avoidance + dynamic delta**

Run: `POLYGON_API_KEY=<key> DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --crash-avoidance --dynamic-delta --quick`

Expected: Comparison table with columns: ML Combined | Combined | Buy-Write | Wheel | Buy & Hold. Look for:
- ML Combined: Return ~+55-60%, Sharpe ~1.2-1.5, MaxDD ~-5-7%
- The n_crash_exits / n_crash_reentries should be small (2-5 over the period)

**Step 3: Run crash avoidance only (no dynamic delta)**

Run: `POLYGON_API_KEY=<key> DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --crash-avoidance --quick`

Expected: Isolate the impact of crash avoidance vs baseline.

**Step 4: Run dynamic delta only (no crash avoidance)**

Run: `POLYGON_API_KEY=<key> DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --dynamic-delta --quick`

Expected: Isolate the impact of dynamic delta vs baseline.

**Step 5: Run baseline for comparison**

Run: `POLYGON_API_KEY=<key> DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --quick`

Expected: Same ML Combined +55.1% baseline from earlier.

**Step 6: Run ML ensemble crash avoidance (verify existing implementation)**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_ml_mega --quick --no-pine --walk-forward-windows 3 --crash-avoidance`

Expected: Return closer to buy-and-hold (~55-58%) with better drawdown than active trading mode.

---

## Verification Commands

```bash
# Unit tests (fast)
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/backtesting/engines/test_wheel.py tests/app/test_spy_options_wheel.py -v

# Full regression
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q

# Live: crash avoidance + dynamic delta
POLYGON_API_KEY=<key> DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --crash-avoidance --dynamic-delta --quick

# Live: crash avoidance only
POLYGON_API_KEY=<key> DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --crash-avoidance --quick

# Live: dynamic delta only
POLYGON_API_KEY=<key> DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --dynamic-delta --quick

# Live: baseline
POLYGON_API_KEY=<key> DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_options_wheel --combined --quick

# ML ensemble crash avoidance (existing)
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m stockdownloader.app.spy_ml_mega --quick --no-pine --walk-forward-windows 3 --crash-avoidance
```
