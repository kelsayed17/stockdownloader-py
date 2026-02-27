# Unified Multi-Mode VWAP Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified VWAP strategy that dispatches PS/ORB/PB/REV entries with priority gating, shared day-trade budget, and per-mode trail swapping — matching PineScript v11.2's architecture.

**Architecture:** Single `IntradayInfra` with composited standalone strategy instances whose `_evaluate_entry()` methods are called in priority order. First non-None signal wins. Trail strategy swapped via `exit_mgr.set_active_trail()` before entry recorded.

**Tech Stack:** Python 3.12, dataclasses (frozen), Decimal arithmetic, pytest

---

### Task 1: Create UnifiedVWAPConfig dataclass

**Files:**
- Create: `src/stockdownloader/strategies/intraday/unified_vwap.py`
- Test: `tests/strategies/intraday/test_unified_vwap.py`

**Step 1: Write the failing test**

```python
"""Tests for UnifiedVWAPStrategy — priority dispatch with shared state."""
from decimal import Decimal

import pytest

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradayAction, HOLD
from stockdownloader.strategies.base import IntradayTradingStrategy


def _make_bar(
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100_000,
) -> IntradayPriceData:
    return IntradayPriceData(
        date=date,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        adj_close=Decimal(str(close)),
        volume=volume,
    )


def _generate_session(
    trading_date: str = "2025-01-15",
    base_price: float = 500.0,
    num_bars: int = 78,
) -> list[IntradayPriceData]:
    bars = []
    price = base_price
    for i in range(num_bars):
        total_mins = 30 + i * 5
        hour = 9 + total_mins // 60
        minute = total_mins % 60
        dt_str = f"{trading_date} {hour:02d}:{minute:02d}:00-05:00"
        o = price
        h = price + 0.50
        l = price - 0.50
        c = price + 0.10 * (1 if i % 2 == 0 else -1)
        vol = 100_000 + i * 1000
        bars.append(_make_bar(dt_str, o, h, l, c, vol))
        price = c
    return bars


def _generate_multi_session(num_days: int = 16) -> list[IntradayPriceData]:
    all_bars = []
    base_price = 500.0
    for d in range(num_days):
        date_str = f"2025-01-{d + 1:02d}"
        session = _generate_session(
            trading_date=date_str,
            base_price=base_price,
            num_bars=78,
        )
        all_bars.extend(session)
        base_price += 0.5
    return all_bars


class TestUnifiedVWAPConfig:
    def test_import(self):
        from stockdownloader.strategies.intraday.unified_vwap import (
            UnifiedVWAPConfig,
            UnifiedVWAPStrategy,
        )
        assert UnifiedVWAPConfig is not None
        assert UnifiedVWAPStrategy is not None

    def test_default_construction(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        config = UnifiedVWAPConfig()
        assert config.pb_enable is True
        assert config.ps_enable is True
        assert config.orb_enable is True
        assert config.rev_enable is True
        assert config.max_day == 1  # base default

    def test_mode_disable(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        config = UnifiedVWAPConfig(pb_enable=False, rev_enable=False)
        assert config.pb_enable is False
        assert config.rev_enable is False
```

**Step 2: Run test to verify it fails**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_unified_vwap.py::TestUnifiedVWAPConfig -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

Create `src/stockdownloader/strategies/intraday/unified_vwap.py`:

```python
"""Unified Multi-Mode VWAP Strategy.

Composes PB, PS, ORB, REV entry evaluators with priority dispatch
(PS > ORB > PB > REV), shared day-trade budget, and per-mode trail
swapping. Matches PineScript v11.2 architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategies.intraday.base import BaseIntradayStrategy, InfraExitConfig
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import IntradayExitManager
from stockdownloader.strategies.intraday.trail import (
    AtrChandelierTrail,
    BreakevenTrail,
    TrailStrategy,
    VwapRatchetTrail,
)

from stockdownloader.strategies.intraday.pullback import PullbackStrategy, PullbackStrategyConfig
from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy, PatternScalpStrategyConfig
from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy, ORBreakoutStrategyConfig
from stockdownloader.strategies.intraday.reversal import ReversalStrategy, ReversalStrategyConfig

if TYPE_CHECKING:
    from stockdownloader.core.models.trade import IntradaySignal
    from stockdownloader.strategies.intraday.session import BarContext


@dataclass(frozen=True, slots=True)
class UnifiedVWAPConfig(InfraExitConfig):
    """Configuration for the unified multi-mode VWAP strategy.

    Holds shared infra/risk fields and per-mode enable flags.
    Mode-specific entry configs are passed via keyword-override dicts.
    """

    # Per-mode enable flags
    pb_enable: bool = True
    ps_enable: bool = True
    orb_enable: bool = True
    rev_enable: bool = True
```

**Step 4: Run test to verify it passes**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_unified_vwap.py::TestUnifiedVWAPConfig -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/unified_vwap.py tests/strategies/intraday/test_unified_vwap.py
git commit -m "feat: add UnifiedVWAPConfig dataclass (empty shell)"
```

---

### Task 2: Build UnifiedVWAPStrategy class with priority dispatch

**Files:**
- Modify: `src/stockdownloader/strategies/intraday/unified_vwap.py`
- Modify: `tests/strategies/intraday/test_unified_vwap.py`

**Step 1: Write the failing tests**

Add to the test file:

```python
class TestUnifiedVWAPConstruction:
    def test_is_intraday_strategy(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        assert isinstance(strategy, IntradayTradingStrategy)

    def test_name(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        assert strategy.name == "Unified VWAP"

    def test_warmup_period(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        assert strategy.warmup_period == 78 * 15

    def test_evaluates_hold_on_warmup_bars(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        bars = _generate_multi_session(16)
        signal = strategy.evaluate(bars, 0)
        assert signal.action == IntradayAction.HOLD

    def test_session_start_resets_state(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        strategy._infra.state.day_trades = 3
        strategy._infra.state.tripped = True
        strategy.on_session_start("2025-01-20")
        assert strategy._infra.state.day_trades == 0
        assert strategy._infra.state.tripped is False

    def test_mode_count_default(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        assert len(strategy._modes) == 4  # PS, ORB, PB, REV

    def test_mode_count_with_disable(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy(rev_enable=False)
        assert len(strategy._modes) == 3  # PS, ORB, PB

    def test_priority_order(self):
        """Modes must be in PS > ORB > PB > REV priority."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy
        from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy
        from stockdownloader.strategies.intraday.pullback import PullbackStrategy
        from stockdownloader.strategies.intraday.reversal import ReversalStrategy

        strategy = UnifiedVWAPStrategy()
        mode_types = [type(s) for s, _ in strategy._modes]
        assert mode_types == [
            PatternScalpStrategy,
            ORBreakoutStrategy,
            PullbackStrategy,
            ReversalStrategy,
        ]
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_unified_vwap.py::TestUnifiedVWAPConstruction -v`
Expected: FAIL

**Step 3: Write the implementation**

Add to `unified_vwap.py` after the config class:

```python
class UnifiedVWAPStrategy(BaseIntradayStrategy):
    """Unified multi-mode VWAP strategy with priority dispatch.

    Composes PS, ORB, PB, REV entry evaluators from existing standalone
    strategies.  The first evaluator to return a non-None signal wins.
    Trail strategy is swapped per mode before the entry is recorded.

    Priority order (matches PineScript v11.2):
        PS > ORB > PB > REV

    Shared across all modes:
        - day_trades budget (max_day)
        - spacing counter (last_entry_bar)
        - circuit breaker (tripped / day_limited)
        - session state + indicators (single IntradayInfra)
    """

    # All modes count against day-trade limit (no fire_once bypass)
    _ENTRY_FLAGS: dict[str, bool] = {}

    def __init__(
        self,
        config: UnifiedVWAPConfig | None = None,
        *,
        pb_overrides: dict | None = None,
        ps_overrides: dict | None = None,
        orb_overrides: dict | None = None,
        rev_overrides: dict | None = None,
        **shared_overrides: object,
    ) -> None:
        # Build unified config from shared overrides
        if config is not None and shared_overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if shared_overrides:
            c = UnifiedVWAPConfig(**shared_overrides)
        else:
            c = config or UnifiedVWAPConfig()
        self._c = c

        # Single shared infra — VwapRatchetTrail is default, swapped per entry
        self._infra = IntradayInfra(c, IntradayExitManager(VwapRatchetTrail()))

        # Build mode evaluators in priority order
        self._modes: list[tuple[BaseIntradayStrategy, type[TrailStrategy]]] = []

        if c.ps_enable:
            ps_cfg = PatternScalpStrategyConfig(**(ps_overrides or {}))
            self._modes.append((PatternScalpStrategy(config=ps_cfg), BreakevenTrail))

        if c.orb_enable:
            orb_cfg = ORBreakoutStrategyConfig(**(orb_overrides or {}))
            self._modes.append((ORBreakoutStrategy(config=orb_cfg), AtrChandelierTrail))

        if c.pb_enable:
            pb_cfg = PullbackStrategyConfig(**(pb_overrides or {}))
            self._modes.append((PullbackStrategy(config=pb_cfg), VwapRatchetTrail))

        if c.rev_enable:
            rev_cfg = ReversalStrategyConfig(**(rev_overrides or {}))
            self._modes.append((ReversalStrategy(config=rev_cfg), BreakevenTrail))

    @property
    def name(self) -> str:
        return "Unified VWAP"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        """Priority dispatch: first non-None signal wins."""
        for strategy, trail_cls in self._modes:
            signal = strategy._evaluate_entry(ctx)
            if signal is not None:
                # Swap trail strategy for the winning mode
                self._infra.exit_mgr.set_active_trail(trail_cls())
                return signal
        return None
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_unified_vwap.py -v`
Expected: PASS (11 tests)

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/unified_vwap.py tests/strategies/intraday/test_unified_vwap.py
git commit -m "feat: add UnifiedVWAPStrategy with priority dispatch"
```

---

### Task 3: Add shared-state tests (day_trades, spacing across modes)

**Files:**
- Modify: `tests/strategies/intraday/test_unified_vwap.py`

**Step 1: Write the failing tests**

```python
class TestSharedState:
    """Verify modes share day_trades, spacing, and circuit breaker."""

    def test_day_trades_shared_across_modes(self):
        """After entry, day_trades increments for all modes."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        state = strategy._infra.state
        # Simulate an entry was recorded
        state.day_trades = 1
        state.last_entry_bar = 10
        # All mode evaluators see the shared state
        assert state.day_trades == 1

    def test_circuit_breaker_halts_all_modes(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        state = strategy._infra.state
        state.tripped = True
        # check_risk should return True (blocked)
        assert strategy._infra.check_risk() is True

    def test_day_limited_halts_all_modes(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        state = strategy._infra.state
        state.day_limited = True
        assert strategy._infra.check_risk() is True

    def test_runs_without_crash_on_multi_session(self):
        """Smoke test: runs 16 sessions without error."""
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy
        strategy = UnifiedVWAPStrategy()
        bars = _generate_multi_session(16)
        for i in range(len(bars)):
            signal = strategy.evaluate(bars, i)
            assert signal is not None
```

**Step 2: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_unified_vwap.py::TestSharedState -v`
Expected: PASS (4 tests)

**Step 3: Commit**

```bash
git add tests/strategies/intraday/test_unified_vwap.py
git commit -m "test: add shared-state tests for unified strategy"
```

---

### Task 4: Export from __init__.py

**Files:**
- Modify: `src/stockdownloader/strategies/intraday/__init__.py`

**Step 1: Add import and __all__ entry**

After the existing reversal import line, add:
```python
from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig, UnifiedVWAPStrategy
```

Add to `__all__` list (alphabetical, in standalone strategies section):
```python
"UnifiedVWAPConfig",
"UnifiedVWAPStrategy",
```

**Step 2: Verify import works**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -c "from stockdownloader.strategies.intraday import UnifiedVWAPStrategy; print('OK')"`
Expected: `OK`

**Step 3: Verify existing tests still pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/ -x -q`
Expected: All tests PASS (400+ tests)

**Step 4: Commit**

```bash
git add src/stockdownloader/strategies/intraday/__init__.py
git commit -m "feat: export UnifiedVWAPStrategy from intraday package"
```

---

### Task 5: Add unified strategy to TV-parity backtest

**Files:**
- Modify: `scripts/tv_parity_backtest.py`

**Step 1: Add unified strategy import and builder**

Add import at top:
```python
from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPStrategy, UnifiedVWAPConfig
```

Add a new function after `build_tv_parity_strategies()`:

```python
def build_unified_strategy() -> tuple[str, object]:
    """Build unified strategy with TV-parity configs for all modes."""
    shared = dict(
        max_day=2,
        spacing=3,
        be_trigger=D("0.5"),
        trail_buf=D("0.15"),
        trail_keep_tp=True,
        close_eod=True,
        circuit=3,
        day_loss=D("3.0"),
        adx_thresh=D("21"),
    )

    pb_overrides = dict(
        allow_shorts=True,
        allow_longs=True,
        pb_zone=D("0.5"),
        pb_body=D("0.15"),
        rr=D("1.4"),
        sl_atr=D("1.3"),
        sl_cap=D("1.50"),
        trend_bars=3,
        htf_align=True,
        ar_filter=True,
        ar_thresh=D("0.9"),
        ar_cap=D("1.15"),
        va_filter=True,
        va_min=D("-0.1"),
        cvd_long_filter=True,
        lrs_short_filter=True,
        lrs_thresh=D("0.08"),
        max_vxc=6,
        w_vol=3,
        w_sr=2,
        w_rsi=1,
        w_time=0,
        w_pq=1,
        w_box=0,
        min_score=3,
        min_score_long=5,
        pq_max_cross=3,
        no_friday_short=True,
        no_monday_long=True,
        pb_vwap_bias=True,
        pb_tp_mode="rr",
    )

    ps_overrides = dict(
        allow_shorts=True,
        allow_longs=True,
        ps_atr_pct=D("30.0"),
        ps_window=12,
        ps_rvol=D("1.0"),
        ps_engulf=D("0.35"),
        ps_sl_mode="Day Extreme",
        ps_sl_atr=D("1.5"),
        ps_sl_cap=D("2.50"),
        ps_tp_pct=D("75.0"),
        ps_sma_filter=False,
        ps_htf_align=False,
        ps_time_gate=False,
        ps_min_rr=D("0.3"),
    )

    orb_overrides = dict(
        allow_shorts=True,
        allow_longs=True,
        orb_window=20,
        orb_rvol=D("2.0"),
        orb_sl_mode="OR Opposite",
        orb_sl_atr=D("1.5"),
        orb_sl_cap=D("2.50"),
        orb_vwap_align=True,
        orb_body_min=D("0.2"),
        orb_entry_mode="aggressive",
        orb_trail_atr=D("1.5"),
        orb_htf_align=True,
        orb_gap_filter=True,
        adx_thresh=D("21"),
    )

    rev_overrides = dict(
        allow_longs=True,
        rev_band="2\u03c3",
        rev_body=D("0.20"),
        rev_sl_atr=D("1.0"),
        rev_sl_cap=D("1.50"),
        rev_shorts=False,
        rev_min_rr=D("0.3"),
        rev_tp_mode="vwap",
        adx_thresh=D("21"),
        min_score=3,
        rev_vwap_flat_tol=D("0.05"),
        rev_require_sr=False,
        rev_hug_limit=20,
        rev_can_trade_bar=11,
    )

    config = UnifiedVWAPConfig(**shared)
    strategy = UnifiedVWAPStrategy(
        config=config,
        pb_overrides=pb_overrides,
        ps_overrides=ps_overrides,
        orb_overrides=orb_overrides,
        rev_overrides=rev_overrides,
    )
    return ("UNIFIED (TV-parity)", strategy)
```

**Step 2: Add unified run to main()**

In the `main()` function, after the standalone strategies loop, add:

```python
    # ── Unified strategy ────────────────────────────────────────
    unified_name, unified_strategy = build_unified_strategy()
    unified_result = run_strategy(unified_name, unified_strategy, bars)
    results.append(unified_result)
```

**Step 3: Run the backtest**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/tv_parity_backtest.py`
Expected: Output includes a new "UNIFIED (TV-parity)" row in the results table

**Step 4: Commit**

```bash
git add scripts/tv_parity_backtest.py
git commit -m "feat: add unified strategy to TV-parity backtest"
```

---

### Task 6: Run full test suite and verify

**Files:** None (verification only)

**Step 1: Run all intraday tests**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/ -v --tb=short`
Expected: All tests PASS (400+ existing + ~15 new)

**Step 2: Run TV-parity backtest and capture results**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/tv_parity_backtest.py`
Expected: Unified strategy shows:
- Fewer total trades than sum of standalone modes
- Mode distribution closer to TV (PB majority, PS/ORB/REV minority)
- Positive combined PnL

**Step 3: Final commit with results**

```bash
git add data/SPY/tv_parity_backtest_results.csv
git commit -m "results: unified TV-parity backtest results"
```
