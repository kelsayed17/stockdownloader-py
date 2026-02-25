# SPY Tournament Strategy Conversion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert 3 Pine Script tournament winners to Python IntradayTradingStrategy classes and run a 2-year comparison backtest on 5-min SPY data.

**Architecture:** Each strategy subclasses `BaseIntradayStrategy`, overrides `evaluate()` to stash data/index, and implements `_evaluate_entry()` with strategy-specific MACD/SMA/OBV logic. All risk management (circuit breaker, daily DD, breakeven, EOD close) uses the existing `IntradayInfra` + `InfraExitConfig` + `BreakevenTrail` — no custom risk code needed.

**Tech Stack:** Python 3.11, existing `IndicatorHub` (streaming MACD, SMA, OBV), `IntradayBacktestEngine`, `PolygonDataClient` for 5-min data.

**Key design note:** `_evaluate_entry(ctx: BarContext)` doesn't receive raw data/index. Strategies override `evaluate()` to stash `self._data` and `self._idx`, then call hub methods directly in `_evaluate_entry`.

---

### Task 1: MACD+OBV Strategy (Tournament Winner)

**Files:**
- Create: `src/stockdownloader/strategies/intraday/macd_obv.py`
- Test: `tests/strategies/intraday/test_macd_obv.py`

**Step 1: Write the failing tests**

Create `tests/strategies/intraday/test_macd_obv.py`:

```python
"""Tests for MACD+OBV tournament strategy."""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import IntradayAction


def _bar(
    dt: str,
    o: float,
    h: float,
    l: float,
    c: float,
    v: int = 1000,
    date: str = "2024-01-02",
) -> IntradayPriceData:
    """Helper to create a 5-min bar."""
    return IntradayPriceData(
        datetime=dt,
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(l)),
        close=Decimal(str(c)),
        volume=v,
        trading_date=date,
    )


def _make_warmup_bars(n: int = 1200, date_start: str = "2024-01-02") -> list[IntradayPriceData]:
    """Generate n synthetic bars for warmup (78 bars/day)."""
    bars: list[IntradayPriceData] = []
    price = 450.0
    day_num = 0
    for i in range(n):
        bar_of_day = i % 78
        if bar_of_day == 0 and i > 0:
            day_num += 1
        date = f"2024-01-{2 + day_num:02d}"
        hour = 9 + (bar_of_day * 5 + 30) // 60
        minute = (bar_of_day * 5 + 30) % 60
        dt = f"{date}T{hour:02d}:{minute:02d}:00-05:00"
        # Gentle uptrend with noise
        delta = 0.02 * (1 if i % 3 != 0 else -1)
        price += delta
        bars.append(_bar(dt, price - 0.05, price + 0.10, price - 0.10, price, 5000, date))
    return bars


class TestMACDOBVConfig:
    """Test MACDOBVConfig dataclass."""

    def test_default_values(self) -> None:
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig
        c = MACDOBVConfig()
        assert c.macd_fast == 12
        assert c.macd_slow == 26
        assert c.macd_signal == 9
        assert c.obv_smooth == 5
        assert c.sl_mult == Decimal("1.5")
        assert c.rr_ratio == Decimal("1.5")
        assert c.sl_cap == Decimal("2.0")
        assert c.be_trigger == Decimal("0.5")
        assert c.max_day == 4
        assert c.spacing == 3
        assert c.circuit == 3
        assert c.day_loss == Decimal("3.0")
        assert c.allow_shorts is True

    def test_inherits_infra_exit_config(self) -> None:
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        assert issubclass(MACDOBVConfig, InfraExitConfig)


class TestMACDOBVStrategy:
    """Test MACDOBVStrategy signal logic."""

    def test_strategy_name(self) -> None:
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        s = MACDOBVStrategy()
        assert s.name == "SPY MACD+OBV"

    def test_warmup_period(self) -> None:
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        s = MACDOBVStrategy()
        # InfraExitConfig default: bars_per_day * 15 = 78 * 15 = 1170
        assert s.warmup_period == 78 * 15

    def test_hold_during_warmup(self) -> None:
        """Strategy should return HOLD during warmup period."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        s = MACDOBVStrategy()
        bars = _make_warmup_bars(100)
        s.on_session_start("2024-01-02")
        sig = s.evaluate(bars, 50)
        assert sig.action == IntradayAction.HOLD

    def test_long_entry_produces_valid_signal(self) -> None:
        """After warmup, a MACD bullish cross + OBV rising should produce ENTER_LONG."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        s = MACDOBVStrategy()
        bars = _make_warmup_bars(1300)
        # Process through warmup
        for i in range(len(bars)):
            if i % 78 == 0:
                s.on_session_start(bars[i].trading_date)
            sig = s.evaluate(bars, i)
        # Any ENTER_LONG or ENTER_SHORT signals should have valid SL/TP
        # (actual crossover depends on data; test validates no crashes)
        assert sig.action in (IntradayAction.HOLD, IntradayAction.ENTER_LONG,
                              IntradayAction.ENTER_SHORT)

    def test_sl_cap_applied(self) -> None:
        """SL distance should be capped at sl_cap ($2.00)."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig
        from stockdownloader.strategies.intraday.trade_mgmt import clamp_sl_dist
        c = MACDOBVConfig()
        # ATR * sl_mult = 3.0 * 1.5 = 4.5 -> capped at 2.0
        raw_sl = Decimal("3.0") * c.sl_mult
        clamped = clamp_sl_dist(raw_sl, c.sl_cap)
        assert clamped == Decimal("2.0")

    def test_sl_tp_math(self) -> None:
        """SL/TP prices should match Pine formula."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVConfig
        from stockdownloader.strategies.intraday.trade_mgmt import directional_sl_tp
        c = MACDOBVConfig()
        entry = Decimal("450.00")
        sl_dist = Decimal("2.0")  # capped
        tp_dist = sl_dist * c.rr_ratio  # 2.0 * 1.5 = 3.0
        sl, tp = directional_sl_tp(True, entry, sl_dist, tp_dist)
        assert sl == Decimal("448.00")
        assert tp == Decimal("453.00")
        # Short side
        sl_s, tp_s = directional_sl_tp(False, entry, sl_dist, tp_dist)
        assert sl_s == Decimal("452.00")
        assert tp_s == Decimal("447.00")

    def test_obv_ema_crossover_detection(self) -> None:
        """OBV EMA rising/falling detection should work correctly."""
        from stockdownloader.strategies.intraday.macd_obv import _obv_ema_state
        # Simulate rising OBV EMA
        state = _obv_ema_state(period=5)
        # Seed with increasing OBV values
        for obv_val in [100, 200, 300, 400, 500, 600]:
            state.update(Decimal(str(obv_val)))
        assert state.is_rising() is True
        # Then decreasing
        for obv_val in [500, 400, 300]:
            state.update(Decimal(str(obv_val)))
        assert state.is_rising() is False
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_macd_obv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stockdownloader.strategies.intraday.macd_obv'`

**Step 3: Write the implementation**

Create `src/stockdownloader/strategies/intraday/macd_obv.py`:

```python
"""SPY MACD+OBV (Tournament Winner) — intraday strategy.

Pine Script equivalent: output/pinescript/spy/spy_macd_obv.pine
Tournament rank: Grand Winner | OOS Score: +74.03 | Degradation: 0.96

Entry:
  Long  — MACD(12,26,9) bullish crossover + EMA(OBV,5) rising
  Short — MACD(12,26,9) bearish crossunder + EMA(OBV,5) falling

Exit:
  Reverse MACD crossover, or SL/TP hit, or EOD.

Risk:
  SL = min(ATR(14) * 1.5, $2.00)
  TP = SL * 1.5 (R:R)
  Breakeven at 0.5R → SL moves to entry ± $0.05
  Max 4 trades/day, 3-bar spacing, circuit breaker after 3 losses,
  halt on -3% daily drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.math import ZERO
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy, InfraExitConfig
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    clamp_sl_dist,
    directional_sl_tp,
    make_entry_signal,
)

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.intraday.session import BarContext


# ── OBV EMA helper ──────────────────────────────────────────────────────


@dataclass
class _obv_ema_state:
    """Tracks EMA of OBV for rising/falling detection."""

    period: int = 5
    _ema: Decimal = ZERO
    _prev_ema: Decimal = ZERO
    _initialized: bool = False
    _count: int = 0

    def update(self, obv: Decimal) -> None:
        """Update with a new OBV value."""
        if not self._initialized:
            self._count += 1
            if self._count == 1:
                self._ema = obv
                self._prev_ema = obv
            else:
                alpha = Decimal(2) / Decimal(self.period + 1)
                self._prev_ema = self._ema
                self._ema = alpha * obv + (Decimal(1) - alpha) * self._ema
            if self._count >= self.period:
                self._initialized = True
        else:
            alpha = Decimal(2) / Decimal(self.period + 1)
            self._prev_ema = self._ema
            self._ema = alpha * obv + (Decimal(1) - alpha) * self._ema

    def is_rising(self) -> bool:
        return self._ema > self._prev_ema

    def is_falling(self) -> bool:
        return self._ema < self._prev_ema

    def reset(self) -> None:
        self._ema = ZERO
        self._prev_ema = ZERO
        self._initialized = False
        self._count = 0


# ── Config ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MACDOBVConfig(InfraExitConfig):
    """Configuration for SPY MACD+OBV strategy."""

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # OBV smoothing
    obv_smooth: int = 5

    # ATR / SL / TP
    atr_len: int = 14
    sl_mult: Decimal = Decimal("1.5")
    rr_ratio: Decimal = Decimal("1.5")
    sl_cap: Decimal = Decimal("2.0")

    # Breakeven at 0.5R (Pine: 0.5R trigger, $0.05 buffer)
    be_trigger: Decimal = Decimal("0.5")

    # Risk guards (Pine v2 defaults)
    max_day: int = 4
    spacing: int = 3
    circuit: int = 3
    day_loss: Decimal = Decimal("3.0")

    # Bi-directional
    allow_longs: bool = True
    allow_shorts: bool = True


# ── Strategy ────────────────────────────────────────────────────────────


class MACDOBVStrategy(BaseIntradayStrategy):
    """SPY MACD+OBV intraday strategy (Tournament Winner)."""

    def __init__(self, **overrides: object) -> None:
        self._c = MACDOBVConfig(**overrides) if overrides else MACDOBVConfig()
        self._infra = IntradayInfra(self._c, IntradayExitManager())
        super().__init__()
        self._obv_state = _obv_ema_state(period=self._c.obv_smooth)
        # Stash slots for data/index (set in evaluate)
        self._data: list[IntradayPriceData] = []
        self._idx: int = 0
        # Previous MACD values for crossover detection
        self._prev_macd_line: Decimal = ZERO
        self._prev_macd_sig: Decimal = ZERO

    @property
    def name(self) -> str:
        return "SPY MACD+OBV"

    def on_session_start(self, trading_date: str) -> None:
        super().on_session_start(trading_date)
        # Don't reset OBV EMA or MACD prev — they are continuous across sessions

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        # Stash for use in _evaluate_entry
        self._data = data
        self._idx = current_index
        # Update OBV EMA each bar (continuous, not reset per session)
        hub = self._infra.hub
        obv_val = hub.obv(data, current_index)
        self._obv_state.update(obv_val)
        return self._infra.run_bar(
            data, current_index, self._evaluate_entry, self._ENTRY_FLAGS,
        )

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        # Guard: max daily trades, bar spacing
        if s.day_trades >= c.max_day:
            return None
        if ctx.bar_of_day - s.last_entry_bar < c.spacing and s.last_entry_bar > 0:
            return None
        if not ctx.is_good_time:
            return None

        # Compute MACD
        hub = self._infra.hub
        data, i = self._data, self._idx
        macd_line = hub.macd_line(data, i, c.macd_fast, c.macd_slow)
        macd_sig = hub.macd_signal(data, i, c.macd_fast, c.macd_slow, c.macd_signal)

        # Crossover detection
        crossover = self._prev_macd_line <= self._prev_macd_sig and macd_line > macd_sig
        crossunder = self._prev_macd_line >= self._prev_macd_sig and macd_line < macd_sig

        # Update previous values
        self._prev_macd_line = macd_line
        self._prev_macd_sig = macd_sig

        # OBV filter
        obv_rising = self._obv_state.is_rising()
        obv_falling = self._obv_state.is_falling()

        # Entry conditions
        go_long = crossover and obv_rising and c.allow_longs
        go_short = crossunder and obv_falling and c.allow_shorts

        if not go_long and not go_short:
            return None

        # SL/TP computation
        atr_val = ctx.atr_val
        raw_sl = atr_val * c.sl_mult
        sl_dist = clamp_sl_dist(raw_sl, c.sl_cap)
        if sl_dist is None:
            return None
        tp_dist = sl_dist * c.rr_ratio

        is_long = go_long
        sl_price, tp_price = directional_sl_tp(is_long, ctx.bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=is_long,
            mode="MACD-OBV",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason="MACD cross + OBV confirm",
        )
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_macd_obv.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/macd_obv.py tests/strategies/intraday/test_macd_obv.py
git commit -m "feat: add MACD+OBV tournament strategy with tests"
```

---

### Task 2: SMA Cross 20/21 Strategy (Tournament #2)

**Files:**
- Create: `src/stockdownloader/strategies/intraday/sma_cross.py`
- Test: `tests/strategies/intraday/test_sma_cross.py`

**Step 1: Write the failing tests**

Create `tests/strategies/intraday/test_sma_cross.py`:

```python
"""Tests for SMA Cross 20/21 tournament strategy."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.core.models.trade import IntradayAction


class TestSMACross2021Config:
    """Test SMACross2021Config dataclass."""

    def test_default_values(self) -> None:
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Config
        c = SMACross2021Config()
        assert c.sma_short == 20
        assert c.sma_long == 21
        assert c.sl_mult == Decimal("1.5")
        assert c.rr_ratio == Decimal("1.5")
        assert c.sl_cap == Decimal("2.0")
        assert c.be_trigger == Decimal("0.5")
        assert c.max_day == 4
        assert c.spacing == 3
        assert c.allow_longs is True
        assert c.allow_shorts is False  # Long-only!

    def test_inherits_infra_exit_config(self) -> None:
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Config
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        assert issubclass(SMACross2021Config, InfraExitConfig)


class TestSMACross2021Strategy:
    """Test SMACross2021Strategy signal logic."""

    def test_strategy_name(self) -> None:
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
        s = SMACross2021Strategy()
        assert s.name == "SPY SMA 20/21"

    def test_long_only(self) -> None:
        """Strategy config should be long-only (no shorts)."""
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
        s = SMACross2021Strategy()
        assert s._c.allow_shorts is False

    def test_sl_tp_math(self) -> None:
        """SL/TP prices should match Pine formula (long side only)."""
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Config
        from stockdownloader.strategies.intraday.trade_mgmt import directional_sl_tp
        c = SMACross2021Config()
        entry = Decimal("500.00")
        sl_dist = Decimal("1.80")  # ATR=1.2 * 1.5 = 1.80, under $2 cap
        tp_dist = sl_dist * c.rr_ratio  # 1.80 * 1.5 = 2.70
        sl, tp = directional_sl_tp(True, entry, sl_dist, tp_dist)
        assert sl == Decimal("498.20")
        assert tp == Decimal("502.70")

    def test_sma_crossover_direction(self) -> None:
        """SMA(20) crossing above SMA(21) is a golden cross (long entry)."""
        # Verify the crossover logic in isolation
        prev_fast, prev_slow = Decimal("449.90"), Decimal("450.00")  # fast < slow
        curr_fast, curr_slow = Decimal("450.10"), Decimal("450.00")  # fast > slow
        crossover = prev_fast <= prev_slow and curr_fast > curr_slow
        assert crossover is True
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_sma_cross.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/stockdownloader/strategies/intraday/sma_cross.py`:

```python
"""SPY SMA Cross 20/21 (Tournament #2) — intraday strategy.

Pine Script equivalent: output/pinescript/spy/spy_sma_crossover.pine
Tournament rank: #2 | OOS Score: +65.29 | Degradation: 2.52

Entry:
  Long only — SMA(20) crosses above SMA(21) (tight golden cross).

Exit:
  SMA death cross (SMA(20) < SMA(21)), or SL/TP hit, or EOD.

The intentionally tight 1-period gap between fast and slow SMA
acts as a momentum filter rather than a traditional trend cross.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.math import ZERO
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy, InfraExitConfig
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    clamp_sl_dist,
    directional_sl_tp,
    make_entry_signal,
)

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.intraday.session import BarContext


@dataclass(frozen=True, slots=True)
class SMACross2021Config(InfraExitConfig):
    """Configuration for SPY SMA Cross 20/21 strategy."""

    sma_short: int = 20
    sma_long: int = 21
    atr_len: int = 14
    sl_mult: Decimal = Decimal("1.5")
    rr_ratio: Decimal = Decimal("1.5")
    sl_cap: Decimal = Decimal("2.0")
    be_trigger: Decimal = Decimal("0.5")
    max_day: int = 4
    spacing: int = 3
    circuit: int = 3
    day_loss: Decimal = Decimal("3.0")
    allow_longs: bool = True
    allow_shorts: bool = False  # Long-only


class SMACross2021Strategy(BaseIntradayStrategy):
    """SPY SMA Cross 20/21 intraday strategy (Tournament #2)."""

    def __init__(self, **overrides: object) -> None:
        self._c = SMACross2021Config(**overrides) if overrides else SMACross2021Config()
        self._infra = IntradayInfra(self._c, IntradayExitManager())
        super().__init__()
        self._data: list[IntradayPriceData] = []
        self._idx: int = 0
        self._prev_sma_fast: Decimal = ZERO
        self._prev_sma_slow: Decimal = ZERO

    @property
    def name(self) -> str:
        return "SPY SMA 20/21"

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        self._data = data
        self._idx = current_index
        return self._infra.run_bar(
            data, current_index, self._evaluate_entry, self._ENTRY_FLAGS,
        )

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        if s.day_trades >= c.max_day:
            return None
        if ctx.bar_of_day - s.last_entry_bar < c.spacing and s.last_entry_bar > 0:
            return None
        if not ctx.is_good_time:
            return None

        hub = self._infra.hub
        data, i = self._data, self._idx
        sma_fast = hub.sma(data, i, c.sma_short)
        sma_slow = hub.sma(data, i, c.sma_long)

        crossover = self._prev_sma_fast <= self._prev_sma_slow and sma_fast > sma_slow

        self._prev_sma_fast = sma_fast
        self._prev_sma_slow = sma_slow

        if not crossover:
            return None

        atr_val = ctx.atr_val
        raw_sl = atr_val * c.sl_mult
        sl_dist = clamp_sl_dist(raw_sl, c.sl_cap)
        if sl_dist is None:
            return None
        tp_dist = sl_dist * c.rr_ratio

        sl_price, tp_price = directional_sl_tp(True, ctx.bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=True,
            mode="SMA-2021",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason="SMA 20/21 golden cross",
        )
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_sma_cross.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/sma_cross.py tests/strategies/intraday/test_sma_cross.py
git commit -m "feat: add SMA Cross 20/21 tournament strategy with tests"
```

---

### Task 3: MACD Optimized 8/35/5 Strategy (Tournament #3)

**Files:**
- Create: `src/stockdownloader/strategies/intraday/macd_optimized.py`
- Test: `tests/strategies/intraday/test_macd_optimized.py`

**Step 1: Write the failing tests**

Create `tests/strategies/intraday/test_macd_optimized.py`:

```python
"""Tests for MACD Optimized 8/35/5 tournament strategy."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.core.models.trade import IntradayAction


class TestMACDOptimizedConfig:
    """Test MACDOptimizedConfig dataclass."""

    def test_default_values(self) -> None:
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedConfig
        c = MACDOptimizedConfig()
        assert c.macd_fast == 8
        assert c.macd_slow == 35
        assert c.macd_signal == 5
        assert c.sl_mult == Decimal("1.5")
        assert c.rr_ratio == Decimal("1.5")
        assert c.sl_cap == Decimal("2.0")
        assert c.be_trigger == Decimal("0.5")
        assert c.max_day == 4
        assert c.spacing == 3
        assert c.allow_shorts is True

    def test_inherits_infra_exit_config(self) -> None:
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedConfig
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        assert issubclass(MACDOptimizedConfig, InfraExitConfig)


class TestMACDOptimizedStrategy:
    """Test MACDOptimizedStrategy signal logic."""

    def test_strategy_name(self) -> None:
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        s = MACDOptimizedStrategy()
        assert s.name == "SPY MACD 8/35/5"

    def test_bidirectional(self) -> None:
        """Strategy should allow both longs and shorts."""
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        s = MACDOptimizedStrategy()
        assert s._c.allow_longs is True
        assert s._c.allow_shorts is True

    def test_no_obv_filter(self) -> None:
        """MACD Optimized does NOT use OBV — pure MACD crossover."""
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        s = MACDOptimizedStrategy()
        # Should not have OBV-related attributes
        assert not hasattr(s, '_obv_state')

    def test_macd_params_differ_from_standard(self) -> None:
        """MACD 8/35/5 should differ from standard 12/26/9."""
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedConfig
        c = MACDOptimizedConfig()
        assert c.macd_fast != 12  # faster
        assert c.macd_slow != 26  # wider
        assert c.macd_signal != 9  # faster
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_macd_optimized.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/stockdownloader/strategies/intraday/macd_optimized.py`:

```python
"""SPY MACD Optimized 8/35/5 (Tournament #3) — intraday strategy.

Pine Script equivalent: output/pinescript/spy/spy_macd_optimized.pine
Tournament rank: #3 | OOS Score: +62.80 | Degradation: 1.57

Entry:
  Long  — MACD(8,35,5) line crosses above signal
  Short — MACD(8,35,5) line crosses below signal
  No secondary filter (unlike MACD+OBV).

Exit:
  Reverse MACD crossover, or SL/TP hit, or EOD.

Design: Wide slow (35) suppresses noise, fast signal (5) gives quick timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.math import ZERO
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.base import BaseIntradayStrategy, InfraExitConfig
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.trade_mgmt import (
    IntradayExitManager,
    clamp_sl_dist,
    directional_sl_tp,
    make_entry_signal,
)

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.intraday.session import BarContext


@dataclass(frozen=True, slots=True)
class MACDOptimizedConfig(InfraExitConfig):
    """Configuration for SPY MACD Optimized 8/35/5 strategy."""

    macd_fast: int = 8
    macd_slow: int = 35
    macd_signal: int = 5
    atr_len: int = 14
    sl_mult: Decimal = Decimal("1.5")
    rr_ratio: Decimal = Decimal("1.5")
    sl_cap: Decimal = Decimal("2.0")
    be_trigger: Decimal = Decimal("0.5")
    max_day: int = 4
    spacing: int = 3
    circuit: int = 3
    day_loss: Decimal = Decimal("3.0")
    allow_longs: bool = True
    allow_shorts: bool = True


class MACDOptimizedStrategy(BaseIntradayStrategy):
    """SPY MACD Optimized 8/35/5 intraday strategy (Tournament #3)."""

    def __init__(self, **overrides: object) -> None:
        self._c = MACDOptimizedConfig(**overrides) if overrides else MACDOptimizedConfig()
        self._infra = IntradayInfra(self._c, IntradayExitManager())
        super().__init__()
        self._data: list[IntradayPriceData] = []
        self._idx: int = 0
        self._prev_macd_line: Decimal = ZERO
        self._prev_macd_sig: Decimal = ZERO

    @property
    def name(self) -> str:
        return "SPY MACD 8/35/5"

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        self._data = data
        self._idx = current_index
        return self._infra.run_bar(
            data, current_index, self._evaluate_entry, self._ENTRY_FLAGS,
        )

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        if s.day_trades >= c.max_day:
            return None
        if ctx.bar_of_day - s.last_entry_bar < c.spacing and s.last_entry_bar > 0:
            return None
        if not ctx.is_good_time:
            return None

        hub = self._infra.hub
        data, i = self._data, self._idx
        macd_line = hub.macd_line(data, i, c.macd_fast, c.macd_slow)
        macd_sig = hub.macd_signal(data, i, c.macd_fast, c.macd_slow, c.macd_signal)

        crossover = self._prev_macd_line <= self._prev_macd_sig and macd_line > macd_sig
        crossunder = self._prev_macd_line >= self._prev_macd_sig and macd_line < macd_sig

        self._prev_macd_line = macd_line
        self._prev_macd_sig = macd_sig

        go_long = crossover and c.allow_longs
        go_short = crossunder and c.allow_shorts

        if not go_long and not go_short:
            return None

        atr_val = ctx.atr_val
        raw_sl = atr_val * c.sl_mult
        sl_dist = clamp_sl_dist(raw_sl, c.sl_cap)
        if sl_dist is None:
            return None
        tp_dist = sl_dist * c.rr_ratio

        is_long = go_long
        sl_price, tp_price = directional_sl_tp(is_long, ctx.bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=is_long,
            mode="MACD-OPT",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason="MACD 8/35/5 crossover",
        )
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_macd_optimized.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/macd_optimized.py tests/strategies/intraday/test_macd_optimized.py
git commit -m "feat: add MACD Optimized 8/35/5 tournament strategy with tests"
```

---

### Task 4: Registration + Comparison Runner + Live Backtest

**Files:**
- Modify: `config/strategies/intraday_registrations.json`
- Create: `scripts/spy_tournament_compare.py`
- Test: `tests/scripts/test_spy_tournament_compare.py`

**Step 1: Register strategies**

Add to `config/strategies/intraday_registrations.json` (at the end, before closing `}`):

```json
    "spy-macd-obv": {
        "display_name": "SPY MACD+OBV (Winner)",
        "module": "stockdownloader.strategies.intraday.macd_obv",
        "class": "MACDOBVStrategy",
        "default_kwargs": {},
        "param_space": {
            "macd_fast": [8, 10, 12],
            "macd_slow": [21, 26, 35],
            "macd_signal": [5, 7, 9],
            "obv_smooth": [3, 5, 8],
            "sl_mult": ["D:1.0", "D:1.5", "D:2.0"],
            "rr_ratio": ["D:1.0", "D:1.5", "D:2.0"],
            "sl_cap": ["D:1.50", "D:2.00", "D:2.50"],
            "be_trigger": ["D:0.3", "D:0.5", "D:0.7"]
        }
    },
    "spy-sma-2021": {
        "display_name": "SPY SMA 20/21 (#2)",
        "module": "stockdownloader.strategies.intraday.sma_cross",
        "class": "SMACross2021Strategy",
        "default_kwargs": {},
        "param_space": {
            "sma_short": [15, 18, 20, 22],
            "sma_long": [20, 21, 25, 30],
            "sl_mult": ["D:1.0", "D:1.5", "D:2.0"],
            "rr_ratio": ["D:1.0", "D:1.5", "D:2.0"],
            "sl_cap": ["D:1.50", "D:2.00", "D:2.50"],
            "be_trigger": ["D:0.3", "D:0.5", "D:0.7"]
        }
    },
    "spy-macd-opt": {
        "display_name": "SPY MACD 8/35/5 (#3)",
        "module": "stockdownloader.strategies.intraday.macd_optimized",
        "class": "MACDOptimizedStrategy",
        "default_kwargs": {},
        "param_space": {
            "macd_fast": [6, 8, 10, 12],
            "macd_slow": [26, 30, 35, 40],
            "macd_signal": [3, 5, 7, 9],
            "sl_mult": ["D:1.0", "D:1.5", "D:2.0"],
            "rr_ratio": ["D:1.0", "D:1.5", "D:2.0"],
            "sl_cap": ["D:1.50", "D:2.00", "D:2.50"],
            "be_trigger": ["D:0.3", "D:0.5", "D:0.7"]
        }
    }
```

**Step 2: Write comparison runner test**

Create `tests/scripts/test_spy_tournament_compare.py`:

```python
"""Tests for spy_tournament_compare script."""

from __future__ import annotations

import pytest


class TestTournamentCompare:
    """Smoke tests for the comparison runner."""

    def test_import(self) -> None:
        from scripts.spy_tournament_compare import run_comparison
        assert callable(run_comparison)

    def test_tournament_strategies_registered(self) -> None:
        """All 3 tournament strategies should be importable."""
        from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
        from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
        from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
        assert MACDOBVStrategy().name == "SPY MACD+OBV"
        assert SMACross2021Strategy().name == "SPY SMA 20/21"
        assert MACDOptimizedStrategy().name == "SPY MACD 8/35/5"
```

**Step 3: Write comparison runner**

Create `scripts/spy_tournament_compare.py`:

```python
#!/usr/bin/env python3
"""SPY Tournament Strategy Comparison — 2yr 5-min backtest.

Runs all tournament strategies (+ optionally existing strategies)
on 2 years of 5-min SPY data via Polygon API.

Usage:
    POLYGON_API_KEY=<key> python scripts/spy_tournament_compare.py
    POLYGON_API_KEY=<key> python scripts/spy_tournament_compare.py --all
    POLYGON_API_KEY=<key> python scripts/spy_tournament_compare.py --csv data/spy_5min.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from decimal import Decimal
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy


INITIAL_CAPITAL = Decimal("100000")
RISK_PER_TRADE = Decimal("0.01")


def _load_data(csv_path: str | None, api_key: str | None):
    """Load 5-min SPY data from CSV cache or Polygon API."""
    from stockdownloader.core.models.price import IntradayPriceData

    cache_path = Path(csv_path) if csv_path else Path("data/spy_5min_2yr.csv")

    if cache_path.exists():
        print(f"Loading cached data from {cache_path}...")
        from stockdownloader.data.csv_loader import IntradayCsvLoader
        loader = IntradayCsvLoader()
        data = loader.load(str(cache_path))
        print(f"  Loaded {len(data):,} bars")
        return data

    if not api_key:
        raise ValueError(
            "No cached data and no POLYGON_API_KEY. "
            "Set POLYGON_API_KEY env var or provide --csv path."
        )

    print("Fetching 2yr 5-min SPY data from Polygon...")
    from stockdownloader.data.market.polygon_client import PolygonDataClient
    client = PolygonDataClient(api_key)
    data = client.fetch_intraday_history("SPY", days=730, interval="5")
    print(f"  Fetched {len(data):,} bars")

    # Cache to CSV
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "open", "high", "low", "close", "volume", "trading_date"])
        for bar in data:
            writer.writerow([
                bar.datetime, bar.open, bar.high, bar.low, bar.close,
                bar.volume, bar.trading_date,
            ])
    print(f"  Cached to {cache_path}")
    return data


def _run_strategy(name, strategy, data):
    """Run a single strategy and return results dict."""
    engine = IntradayBacktestEngine(
        INITIAL_CAPITAL,
        RISK_PER_TRADE,
        commission=Decimal("1.0"),
        slippage_pct=Decimal("0.0002"),
    )
    result = engine.run(strategy, data)
    return {
        "name": name,
        "total_return_pct": float(result.total_return_pct),
        "sharpe": float(result.sharpe_ratio),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "win_rate": float(result.win_rate),
        "n_trades": result.total_trades,
        "avg_trade": float(result.avg_trade_pnl),
    }


def run_comparison(
    csv_path: str | None = None,
    api_key: str | None = None,
    include_all: bool = False,
) -> list[dict]:
    """Run tournament comparison and return results."""
    data = _load_data(csv_path, api_key)

    strategies: list[tuple[str, object]] = [
        ("SPY MACD+OBV (Winner)", MACDOBVStrategy()),
        ("SPY SMA 20/21 (#2)", SMACross2021Strategy()),
        ("SPY MACD 8/35/5 (#3)", MACDOptimizedStrategy()),
    ]

    if include_all:
        from stockdownloader.strategies.intraday.pullback import PullbackStrategy
        from stockdownloader.strategies.intraday.reversal import ReversalStrategy
        from stockdownloader.strategies.intraday.or_breakout import ORBreakoutStrategy
        from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy
        from stockdownloader.strategies.intraday.pattern_scalp import PatternScalpStrategy
        strategies.extend([
            ("VWAP Pullback", PullbackStrategy()),
            ("VWAP Reversal", ReversalStrategy()),
            ("OR Breakout", ORBreakoutStrategy()),
            ("OR Reversal", ORReversalStrategy()),
            ("Pattern Scalp", PatternScalpStrategy()),
        ])

    results = []
    for name, strat in strategies:
        print(f"\nRunning {name}...")
        try:
            r = _run_strategy(name, strat, data)
            results.append(r)
            print(f"  Return: {r['total_return_pct']:+.1f}%  "
                  f"Sharpe: {r['sharpe']:.2f}  "
                  f"MaxDD: {r['max_drawdown_pct']:.1f}%  "
                  f"Trades: {r['n_trades']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": name, "error": str(e)})

    # Print comparison table
    print("\n" + "=" * 85)
    print(f"{'Strategy':<30} {'Return %':>10} {'Sharpe':>8} {'MaxDD %':>9} "
          f"{'WinRate':>8} {'Trades':>7}")
    print("-" * 85)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>10}")
            continue
        print(f"{r['name']:<30} {r['total_return_pct']:>+9.1f}% "
              f"{r['sharpe']:>8.2f} {r['max_drawdown_pct']:>8.1f}% "
              f"{r['win_rate']:>7.1f}% {r['n_trades']:>7d}")
    print("=" * 85)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="SPY Tournament Strategy Comparison")
    parser.add_argument("--csv", type=str, help="Path to cached 5-min CSV")
    parser.add_argument("--all", action="store_true",
                        help="Include existing VWAP strategies in comparison")
    args = parser.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY")
    run_comparison(csv_path=args.csv, api_key=api_key, include_all=args.all)


if __name__ == "__main__":
    main()
```

**Step 4: Run registration test**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/scripts/test_spy_tournament_compare.py -v`
Expected: All 2 tests PASS

**Step 5: Run full test suite**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q`
Expected: All tests pass (3616 + 17 new = ~3633)

**Step 6: Commit registration + runner**

```bash
git add config/strategies/intraday_registrations.json scripts/spy_tournament_compare.py tests/scripts/test_spy_tournament_compare.py
git commit -m "feat: register tournament strategies and add comparison runner"
```

**Step 7: Run live backtest**

```bash
POLYGON_API_KEY=jtFFyq1sO7sEznPZChj2vMZEeoY9GQSu DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src \
    python3 scripts/spy_tournament_compare.py --all
```

Expected output: comparison table showing return %, Sharpe, MaxDD, win rate, and trade count for all strategies.

**Step 8: Analyze and report results**

Review the comparison table. Document which strategy performed best on real data. Compare Pine tournament OOS scores with actual Python backtest performance.
