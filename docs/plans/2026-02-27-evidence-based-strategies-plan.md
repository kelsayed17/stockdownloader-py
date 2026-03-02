# Evidence-Based SPY Intraday Strategies Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 5 academically-validated intraday strategies plus a MarketContext infrastructure layer for VIX/FOMC filtering.

**Architecture:** Add a `MarketContext` dataclass and `MarketContextProvider` protocol to the intraday infrastructure. Inject market context into `BarContext` at session boundaries via `IntradayInfra`. Build 3 standalone strategies (Gao momentum, noise boundary, Connors RSI(2)), one wrapper (VIX filter), and one event-driven strategy (FOMC drift). All use the existing `BaseIntradayStrategy` pattern.

**Tech Stack:** Python dataclasses, Decimal arithmetic, existing IntradayInfra/BarContext/make_entry_signal infrastructure.

---

## Task 1: MarketContext dataclass + provider protocol

**Files:**
- Create: `src/stockdownloader/strategies/intraday/market_context.py`
- Create: `tests/strategies/intraday/test_market_context.py`

**Step 1: Write test**

```python
# tests/strategies/intraday/test_market_context.py
"""Tests for MarketContext and FileMarketContextProvider."""
from __future__ import annotations

import csv
from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.market_context import (
    MarketContext,
    FileMarketContextProvider,
)


class TestMarketContext:
    def test_vix_regime_low(self):
        ctx = MarketContext(
            vix_close=Decimal("12.5"),
            vix_sma20=Decimal("14.0"),
            vix_regime="low",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("45"),
            daily_close_above_sma200=True,
        )
        assert ctx.vix_regime == "low"

    def test_vix_regime_from_level(self):
        from stockdownloader.strategies.intraday.market_context import vix_regime_from_level
        assert vix_regime_from_level(Decimal("12")) == "low"
        assert vix_regime_from_level(Decimal("18")) == "mid"
        assert vix_regime_from_level(Decimal("28")) == "high"
        assert vix_regime_from_level(Decimal("40")) == "extreme"


class TestFileMarketContextProvider:
    def test_load_and_get_context(self, tmp_path):
        csv_path = tmp_path / "market_ctx.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "vix_close", "vix_sma20", "is_fomc",
                         "is_fomc_pc", "is_opex", "rsi2", "above_sma200"])
            w.writerow(["2024-03-15", "14.5", "15.2", "0", "0", "1", "42.3", "1"])
            w.writerow(["2024-03-18", "22.1", "16.0", "1", "1", "0", "3.2", "1"])
        provider = FileMarketContextProvider(str(csv_path))
        ctx = provider.get_context("2024-03-15")
        assert ctx is not None
        assert ctx.vix_close == Decimal("14.5")
        assert ctx.is_opex is True
        assert ctx.is_fomc_day is False

    def test_get_context_missing_date_returns_none(self, tmp_path):
        csv_path = tmp_path / "market_ctx.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "vix_close", "vix_sma20", "is_fomc",
                         "is_fomc_pc", "is_opex", "rsi2", "above_sma200"])
            w.writerow(["2024-03-15", "14.5", "15.2", "0", "0", "0", "42.3", "1"])
        provider = FileMarketContextProvider(str(csv_path))
        assert provider.get_context("2024-03-20") is None

    def test_vix_regime_computed_on_load(self, tmp_path):
        csv_path = tmp_path / "market_ctx.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "vix_close", "vix_sma20", "is_fomc",
                         "is_fomc_pc", "is_opex", "rsi2", "above_sma200"])
            w.writerow(["2024-03-18", "28.0", "20.0", "0", "0", "0", "50", "1"])
        provider = FileMarketContextProvider(str(csv_path))
        ctx = provider.get_context("2024-03-18")
        assert ctx is not None
        assert ctx.vix_regime == "high"
```

**Step 2: Run test to verify it fails**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_market_context.py -v
```

Expected: FAIL (module not found)

**Step 3: Write implementation**

```python
# src/stockdownloader/strategies/intraday/market_context.py
"""MarketContext for daily regime data injected into intraday strategies.

Provides VIX levels, FOMC/OPEX calendar flags, and daily RSI(2)/SMA(200)
trend filters. Loaded from a pre-computed CSV and accessed per-session
via a provider protocol.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol


def vix_regime_from_level(vix: Decimal) -> str:
    """Classify VIX into regime buckets."""
    if vix < 15:
        return "low"
    if vix < 25:
        return "mid"
    if vix < 35:
        return "high"
    return "extreme"


@dataclass(slots=True)
class MarketContext:
    """Daily market regime data, computed externally and injected per-session."""

    vix_close: Decimal
    vix_sma20: Decimal
    vix_regime: str  # "low", "mid", "high", "extreme"
    is_fomc_day: bool
    is_fomc_press_conf: bool
    is_opex: bool
    daily_rsi2: Decimal
    daily_close_above_sma200: bool


class MarketContextProvider(Protocol):
    """Protocol for objects that supply MarketContext per trading date."""

    def get_context(self, trading_date: str) -> MarketContext | None: ...


class FileMarketContextProvider:
    """Load MarketContext from a pre-computed CSV file.

    CSV columns: date, vix_close, vix_sma20, is_fomc, is_fomc_pc,
                 is_opex, rsi2, above_sma200
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._data: dict[str, MarketContext] = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vix = Decimal(row["vix_close"])
                self._data[row["date"]] = MarketContext(
                    vix_close=vix,
                    vix_sma20=Decimal(row["vix_sma20"]),
                    vix_regime=vix_regime_from_level(vix),
                    is_fomc_day=row["is_fomc"] == "1",
                    is_fomc_press_conf=row["is_fomc_pc"] == "1",
                    is_opex=row["is_opex"] == "1",
                    daily_rsi2=Decimal(row["rsi2"]),
                    daily_close_above_sma200=row["above_sma200"] == "1",
                )

    def get_context(self, trading_date: str) -> MarketContext | None:
        return self._data.get(trading_date)
```

**Step 4: Run test to verify it passes**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_market_context.py -v
```

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/market_context.py tests/strategies/intraday/test_market_context.py
git commit -m "feat: add MarketContext dataclass and FileMarketContextProvider"
```

---

## Task 2: Wire MarketContext into BarContext and IntradayInfra

**Files:**
- Modify: `src/stockdownloader/strategies/intraday/session.py` (line 208-242, BarContext)
- Modify: `src/stockdownloader/strategies/intraday/infra.py` (lines 50-60 __init__, lines 74-234 on_new_bar)

**Step 1: Add `market_ctx` field to BarContext**

In `session.py`, add to the end of the BarContext dataclass (after line 241, before closing):

```python
    # Market context (optional — None when no provider configured)
    market_ctx: MarketContext | None = None
```

Also add the import at the top of session.py (TYPE_CHECKING block):

```python
if TYPE_CHECKING:
    from stockdownloader.strategies.intraday.market_context import MarketContext
```

**Step 2: Add provider support to IntradayInfra**

In `infra.py`, modify `__init__` (lines 50-60) to accept optional provider:

```python
def __init__(
    self,
    config: InfraExitConfig,
    exit_manager: IntradayExitManager,
    hub: IndicatorHub | None = None,
    market_ctx_provider: MarketContextProvider | None = None,
) -> None:
    self.state = SessionState()
    self.hub = hub or IndicatorHub()
    self.exit_mgr = exit_manager
    self._c = config
    self._day = DayTracker()
    self._market_ctx_provider = market_ctx_provider
    self._market_ctx: MarketContext | None = None
```

Add import at top of infra.py (TYPE_CHECKING block):

```python
if TYPE_CHECKING:
    from stockdownloader.strategies.intraday.market_context import MarketContext, MarketContextProvider
```

In `on_new_bar()`, after the session boundary detection (after line 91 where `self._day.on_new_day(...)` is called), add:

```python
    # Fetch market context for new session
    if self._market_ctx_provider is not None:
        self._market_ctx = self._market_ctx_provider.get_context(
            bar.trading_date
        )
    else:
        self._market_ctx = None
```

In the BarContext construction (lines 206-233), add `market_ctx=self._market_ctx` as the last field.

**Step 3: Run all intraday tests**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/ -x -q
```

Expected: All 341+ existing tests still pass (market_ctx defaults to None).

**Step 4: Commit**

```bash
git add src/stockdownloader/strategies/intraday/session.py src/stockdownloader/strategies/intraday/infra.py
git commit -m "feat: wire MarketContext into BarContext and IntradayInfra"
```

---

## Task 3: Gao Intraday Momentum strategy

**Files:**
- Create: `src/stockdownloader/strategies/intraday/gao_momentum.py`
- Create: `tests/strategies/intraday/test_gao_momentum.py`

**Step 1: Write tests**

```python
# tests/strategies/intraday/test_gao_momentum.py
"""Tests for GaoMomentumStrategy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.gao_momentum import (
    GaoMomentumConfig,
    GaoMomentumStrategy,
)
from tests.strategies.intraday.conftest import make_session_bars, ZERO


class TestGaoMomentumConfig:
    def test_defaults(self):
        c = GaoMomentumConfig()
        assert c.entry_start_bar == 73
        assert c.first_hh_bars == 6
        assert c.require_dual_signal is True

    def test_json_round_trip(self):
        c = GaoMomentumConfig(min_r1_magnitude=Decimal("0.001"))
        c2 = GaoMomentumConfig.from_json(c.to_json())
        assert c == c2

    def test_overrides(self):
        s = GaoMomentumStrategy(entry_start_bar=70)
        assert s._c.entry_start_bar == 70


class TestGaoMomentumEntry:
    def test_no_signal_before_entry_bar(self):
        """Strategy should not fire in bars 1-72."""
        strat = GaoMomentumStrategy(require_dual_signal=False)
        # Build 78 bars with positive first half-hour (bars 0-5 rising)
        bars = make_session_bars(
            n=78, date="2024-06-15",
            open_price=Decimal("500"), trend=Decimal("0.10"),
        )
        for i in range(72):
            sig = strat.evaluate(bars, i)
            assert sig.action.name == "HOLD", f"Unexpected signal at bar {i}"

    def test_long_signal_on_positive_first_hh(self):
        """Positive first half-hour → long in last 30 min."""
        strat = GaoMomentumStrategy(
            require_dual_signal=False,
            allow_longs=True,
            allow_shorts=True,
            can_trade_bar=1,
        )
        # Build bars: first 6 bars trending up, rest flat until bar 73+
        bars = make_session_bars(
            n=78, date="2024-06-15",
            open_price=Decimal("500"), trend=Decimal("0.10"),
        )
        # Evaluate through all bars — signal should fire at bar 73+
        signal_fired = False
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            if sig.action.name == "ENTER_LONG":
                signal_fired = True
                assert i >= 72  # 0-indexed bar 72 = bar_of_day 73
                break
        assert signal_fired, "Expected long signal in last 30 min"

    def test_no_signal_on_flat_first_hh(self):
        """Flat first half-hour (below min magnitude) → no signal."""
        strat = GaoMomentumStrategy(
            require_dual_signal=False,
            min_r1_magnitude=Decimal("0.005"),  # High threshold
            can_trade_bar=1,
        )
        bars = make_session_bars(
            n=78, date="2024-06-15",
            open_price=Decimal("500"), trend=Decimal("0.001"),  # Tiny move
        )
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            assert sig.action.name == "HOLD"

    def test_fire_once_per_session(self):
        """Only one entry per session."""
        strat = GaoMomentumStrategy(
            require_dual_signal=False,
            can_trade_bar=1,
        )
        bars = make_session_bars(
            n=78, date="2024-06-15",
            open_price=Decimal("500"), trend=Decimal("0.10"),
        )
        entries = 0
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            if sig.action.name in ("ENTER_LONG", "ENTER_SHORT"):
                entries += 1
                strat.on_position_opened(True)
                strat.on_position_closed()
        assert entries == 1
```

**Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_gao_momentum.py -v
```

**Step 3: Write implementation**

```python
# src/stockdownloader/strategies/intraday/gao_momentum.py
"""Gao Intraday Momentum strategy.

Based on Gao, Han, Li & Zhou (2018), "Market Intraday Momentum,"
Journal of Financial Economics 129(2): 394-414.

The first half-hour return (9:30-10:00) predicts the last half-hour
return (3:30-4:00).  Enter long in the last 30 min if r1 > 0, short
if r1 < 0.  Enhanced mode requires both first and penultimate
half-hour returns to agree (win rate: 54% → 77%).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategies.intraday.base import (
    BaseIntradayStrategy,
    InfraExitConfig,
)
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.exit import IntradayExitManager
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.trade_mgmt import (
    make_entry_signal,
    directional_sl_tp,
)

if TYPE_CHECKING:
    from stockdownloader.strategies.intraday.session import BarContext
    from stockdownloader.strategies.intraday.market_context import MarketContextProvider

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class GaoMomentumConfig(InfraExitConfig):
    """Config for Gao Intraday Momentum strategy."""

    entry_start_bar: int = 73
    first_hh_bars: int = 6
    penult_hh_start: int = 67
    penult_hh_end: int = 72
    require_dual_signal: bool = True
    min_r1_magnitude: Decimal = Decimal("0.0005")
    sl_atr_mult: Decimal = Decimal("1.5")
    allow_longs: bool = True
    allow_shorts: bool = True
    close_eod: bool = True
    max_day: int = 1
    vix_filter: bool = True  # Skip low-VIX days when context available


class GaoMomentumStrategy(BaseIntradayStrategy):
    """Intraday momentum: first half-hour predicts last half-hour."""

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(
        self,
        config: GaoMomentumConfig | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            self._c = GaoMomentumConfig(**overrides)
        else:
            self._c = config or GaoMomentumConfig()
        self._infra = IntradayInfra(
            self._c, IntradayExitManager(), market_ctx_provider=market_ctx_provider,
        )
        super().__init__()
        # Session state for first half-hour tracking
        self._session_open: Decimal = ZERO
        self._first_hh_close: Decimal = ZERO
        self._r1: Decimal = ZERO
        self._r1_valid: bool = False
        self._penult_close: Decimal = ZERO
        self._r12: Decimal = ZERO
        self._r12_valid: bool = False
        self._last_session_date: str = ""

    @property
    def name(self) -> str:
        mode = "dual" if self._c.require_dual_signal else "single"
        return f"Gao Momentum ({mode})"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        # Track session open and first half-hour close
        if ctx.bar.trading_date != self._last_session_date:
            self._last_session_date = ctx.bar.trading_date
            self._session_open = ctx.bar.open
            self._r1_valid = False
            self._r12_valid = False

        # Record first half-hour close
        if ctx.bar_of_day == c.first_hh_bars and self._session_open > ZERO:
            self._first_hh_close = ctx.bar.close
            self._r1 = (self._first_hh_close - self._session_open) / self._session_open
            self._r1_valid = True

        # Record penultimate half-hour close
        if ctx.bar_of_day == c.penult_hh_end:
            self._penult_close = ctx.bar.close
            self._r12 = (self._penult_close - ctx.bar.open) / ctx.bar.open if ctx.bar.open > ZERO else ZERO
            self._r12_valid = True

        # Only trade in the last 30 minutes
        if ctx.bar_of_day < c.entry_start_bar:
            return None

        # Need valid first half-hour return
        if not self._r1_valid:
            return None

        # Filter by magnitude
        if abs(self._r1) < c.min_r1_magnitude:
            return None

        # VIX filter (skip low-VIX when context available)
        if c.vix_filter and ctx.market_ctx is not None:
            if ctx.market_ctx.vix_regime == "low":
                return None

        # Determine direction
        go_long = self._r1 > ZERO
        go_short = self._r1 < ZERO

        # Dual signal check
        if c.require_dual_signal:
            if not self._r12_valid:
                return None
            r12_long = self._r12 > ZERO
            r12_short = self._r12 < ZERO
            if go_long and not r12_long:
                return None
            if go_short and not r12_short:
                return None

        # Direction filter
        if go_long and not c.allow_longs:
            return None
        if go_short and not c.allow_shorts:
            return None

        is_long = go_long

        # SL/TP
        sl_dist = ctx.atr_val * c.sl_atr_mult
        if sl_dist <= ZERO:
            return None
        # No TP — ride to close (EOD exit handles it)
        tp_dist = sl_dist * Decimal("10")  # Very wide TP, EOD exit dominates

        sl_price, tp_price = directional_sl_tp(is_long, ctx.bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=is_long,
            mode="Gao-Mom",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason=f"r1={'+'if self._r1>0 else ''}{self._r1:.4f}"
                   + (f" r12={'+'if self._r12>0 else ''}{self._r12:.4f}" if c.require_dual_signal else ""),
        )
```

**Step 4: Run tests**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_gao_momentum.py -v
```

**Step 5: Run all intraday tests for regression**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/ -x -q
```

**Step 6: Commit**

```bash
git add src/stockdownloader/strategies/intraday/gao_momentum.py tests/strategies/intraday/test_gao_momentum.py
git commit -m "feat: add Gao Intraday Momentum strategy (JFE 2018, Sharpe 1.08)"
```

---

## Task 4: Noise Boundary Breakout strategy

**Files:**
- Create: `src/stockdownloader/strategies/intraday/noise_boundary.py`
- Create: `tests/strategies/intraday/test_noise_boundary.py`

**Step 1: Write tests**

```python
# tests/strategies/intraday/test_noise_boundary.py
"""Tests for NoiseBoundaryStrategy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.noise_boundary import (
    NoiseBoundaryConfig,
    NoiseBoundaryStrategy,
)
from tests.strategies.intraday.conftest import make_session_bars


class TestNoiseBoundaryConfig:
    def test_defaults(self):
        c = NoiseBoundaryConfig()
        assert c.lookback_days == 14
        assert c.vol_multiplier == Decimal("1.0")
        assert c.checkpoint_interval == 6

    def test_json_round_trip(self):
        c = NoiseBoundaryConfig(lookback_days=10, vol_multiplier=Decimal("1.5"))
        c2 = NoiseBoundaryConfig.from_json(c.to_json())
        assert c == c2


class TestNoiseBoundaryEntry:
    def test_no_signal_before_warmup(self):
        """Needs lookback_days of daily data before generating signals."""
        strat = NoiseBoundaryStrategy(lookback_days=14, can_trade_bar=1)
        # Build only 1 day of bars — not enough history
        bars = make_session_bars(
            n=78, date="2024-06-15",
            open_price=Decimal("500"), trend=Decimal("0.10"),
        )
        for i in range(len(bars)):
            sig = strat.evaluate(bars, i)
            assert sig.action.name == "HOLD"

    def test_long_on_upper_breakout(self):
        """Price above upper noise boundary → long entry."""
        strat = NoiseBoundaryStrategy(
            lookback_days=3,  # Short lookback for testing
            can_trade_bar=1,
            allow_longs=True,
            allow_shorts=True,
        )
        # Build 4 days of bars (3 warmup + 1 trading)
        # Days 1-3: tight range around 500 → small noise boundaries
        # Day 4: strong breakout above noise boundary
        all_bars = []
        for d in range(3):
            date = f"2024-06-{15+d:02d}"
            bars = make_session_bars(
                n=78, date=date,
                open_price=Decimal("500"), trend=Decimal("0.01"),
            )
            all_bars.extend(bars)
        # Day 4: big gap up and trend → should break noise boundary
        breakout_bars = make_session_bars(
            n=78, date="2024-06-18",
            open_price=Decimal("510"), trend=Decimal("0.50"),
        )
        all_bars.extend(breakout_bars)

        entry_found = False
        for i in range(len(all_bars)):
            sig = strat.evaluate(all_bars, i)
            if sig.action.name == "ENTER_LONG":
                entry_found = True
                break
        assert entry_found, "Expected long entry on upper noise breakout"

    def test_fire_once_per_session(self):
        """Only one entry per session."""
        strat = NoiseBoundaryStrategy(lookback_days=3, can_trade_bar=1)
        all_bars = []
        for d in range(4):
            date = f"2024-06-{15+d:02d}"
            bars = make_session_bars(
                n=78, date=date,
                open_price=Decimal("500"), trend=Decimal("0.01") if d < 3 else Decimal("0.50"),
            )
            all_bars.extend(bars)
        entries = 0
        for i in range(len(all_bars)):
            sig = strat.evaluate(all_bars, i)
            if sig.action.name in ("ENTER_LONG", "ENTER_SHORT"):
                entries += 1
                strat.on_position_opened(True)
                strat.on_position_closed()
        assert entries <= 1
```

**Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_noise_boundary.py -v
```

**Step 3: Write implementation**

```python
# src/stockdownloader/strategies/intraday/noise_boundary.py
"""Noise Boundary Breakout strategy.

Based on Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective
Intraday Momentum Strategy for S&P500 ETF (SPY)," Swiss Finance Institute
Research Paper No. 24-97.

Defines a "noise area" around the open based on historical intraday volatility
at half-hourly checkpoints. Price beyond the noise band signals genuine
momentum; price within is noise.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategies.intraday.base import (
    BaseIntradayStrategy,
    InfraExitConfig,
)
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.exit import IntradayExitManager
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.trade_mgmt import (
    make_entry_signal,
    directional_sl_tp,
)

if TYPE_CHECKING:
    from stockdownloader.strategies.intraday.session import BarContext
    from stockdownloader.strategies.intraday.market_context import MarketContextProvider

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(frozen=True, slots=True)
class NoiseBoundaryConfig(InfraExitConfig):
    """Config for Noise Boundary Breakout strategy."""

    lookback_days: int = 14
    vol_multiplier: Decimal = Decimal("1.0")
    checkpoint_interval: int = 6  # Every 30 min
    sl_atr_mult: Decimal = Decimal("2.0")
    allow_longs: bool = True
    allow_shorts: bool = True
    close_eod: bool = True
    max_day: int = 1


class NoiseBoundaryStrategy(BaseIntradayStrategy):
    """Noise boundary breakout: trade genuine moves, filter noise."""

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(
        self,
        config: NoiseBoundaryConfig | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            self._c = NoiseBoundaryConfig(**overrides)
        else:
            self._c = config or NoiseBoundaryConfig()
        self._infra = IntradayInfra(
            self._c, IntradayExitManager(), market_ctx_provider=market_ctx_provider,
        )
        super().__init__()
        # Rolling history: checkpoint_bar → list of |close/open - 1| values
        self._checkpoint_vols: dict[int, list[Decimal]] = defaultdict(list)
        self._daily_opens: list[Decimal] = []
        self._prev_closes: list[Decimal] = []
        self._session_open: Decimal = ZERO
        self._prev_close: Decimal = ZERO
        self._last_session_date: str = ""
        self._warmup_days: int = 0

    @property
    def name(self) -> str:
        return f"Noise Boundary (L{self._c.lookback_days})"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        bar = ctx.bar

        # Track session open
        if bar.trading_date != self._last_session_date:
            # Record previous session's close
            if self._last_session_date:
                self._prev_close = ctx.prev_bar.close if ctx.prev_bar else ZERO
                self._prev_closes.append(self._prev_close)
            self._last_session_date = bar.trading_date
            self._session_open = bar.open
            self._daily_opens.append(bar.open)
            self._warmup_days += 1

        # Record checkpoint volatility
        if ctx.bar_of_day % c.checkpoint_interval == 0 and self._session_open > ZERO:
            cp = ctx.bar_of_day
            vol = abs(bar.close / self._session_open - ONE)
            self._checkpoint_vols[cp].append(vol)
            # Trim to lookback
            if len(self._checkpoint_vols[cp]) > c.lookback_days:
                self._checkpoint_vols[cp] = self._checkpoint_vols[cp][-c.lookback_days:]

        # Need enough warmup days
        if self._warmup_days <= c.lookback_days:
            return None

        # Only evaluate at checkpoints
        if ctx.bar_of_day % c.checkpoint_interval != 0:
            return None

        # Compute noise boundaries for this checkpoint
        cp = ctx.bar_of_day
        vols = self._checkpoint_vols.get(cp, [])
        if len(vols) < c.lookback_days:
            return None

        sigma = sum(vols[-c.lookback_days:]) / c.lookback_days
        prev_c = self._prev_close if self._prev_close > ZERO else self._session_open
        anchor_high = max(self._session_open, prev_c)
        anchor_low = min(self._session_open, prev_c)
        upper = anchor_high * (ONE + c.vol_multiplier * sigma)
        lower = anchor_low * (ONE - c.vol_multiplier * sigma)

        # Check breakout
        go_long = bar.close > upper and c.allow_longs
        go_short = bar.close < lower and c.allow_shorts

        if not go_long and not go_short:
            return None

        is_long = go_long

        # SL/TP
        sl_dist = ctx.atr_val * c.sl_atr_mult
        if sl_dist <= ZERO:
            return None
        tp_dist = sl_dist * Decimal("2.0")

        sl_price, tp_price = directional_sl_tp(is_long, bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=is_long,
            mode="NoiseBnd",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason=f"{'Upper' if is_long else 'Lower'} noise break cp={cp}",
        )
```

**Step 4: Run tests, then regression**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_noise_boundary.py -v
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/ -x -q
```

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/noise_boundary.py tests/strategies/intraday/test_noise_boundary.py
git commit -m "feat: add Noise Boundary Breakout strategy (Zarattini 2024, Sharpe 1.33)"
```

---

## Task 5: RSI(2) Connors Mean Reversion strategy

**Files:**
- Create: `src/stockdownloader/strategies/intraday/connors_rsi2.py`
- Create: `tests/strategies/intraday/test_connors_rsi2.py`

**Step 1: Write tests**

```python
# tests/strategies/intraday/test_connors_rsi2.py
"""Tests for ConnorsRSI2Strategy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.connors_rsi2 import (
    ConnorsRSI2Config,
    ConnorsRSI2Strategy,
)
from stockdownloader.strategies.intraday.market_context import MarketContext


class TestConnorsRSI2Config:
    def test_defaults(self):
        c = ConnorsRSI2Config()
        assert c.rsi_period == 2
        assert c.rsi_threshold == Decimal("5")
        assert c.allow_shorts is False

    def test_json_round_trip(self):
        c = ConnorsRSI2Config(rsi_threshold=Decimal("10"))
        c2 = ConnorsRSI2Config.from_json(c.to_json())
        assert c == c2


class TestConnorsRSI2Entry:
    def test_long_on_oversold_rsi2(self):
        """Entry when daily RSI(2) < 5 and above SMA(200)."""
        ctx = MarketContext(
            vix_close=Decimal("18"),
            vix_sma20=Decimal("17"),
            vix_regime="mid",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("3.5"),
            daily_close_above_sma200=True,
        )
        strat = ConnorsRSI2Strategy()
        assert strat._should_enter_today(ctx) is True

    def test_no_entry_when_above_threshold(self):
        """No entry when RSI(2) > threshold."""
        ctx = MarketContext(
            vix_close=Decimal("18"),
            vix_sma20=Decimal("17"),
            vix_regime="mid",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("35"),
            daily_close_above_sma200=True,
        )
        strat = ConnorsRSI2Strategy()
        assert strat._should_enter_today(ctx) is False

    def test_no_entry_below_sma200(self):
        """No entry when daily close below SMA(200) — bearish regime."""
        ctx = MarketContext(
            vix_close=Decimal("18"),
            vix_sma20=Decimal("17"),
            vix_regime="mid",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("3.5"),
            daily_close_above_sma200=False,
        )
        strat = ConnorsRSI2Strategy()
        assert strat._should_enter_today(ctx) is False
```

**Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_connors_rsi2.py -v
```

**Step 3: Write implementation**

```python
# src/stockdownloader/strategies/intraday/connors_rsi2.py
"""RSI(2) Connors Mean Reversion strategy.

Based on Larry Connors' research — 75% win rate, profit factor 2.3,
backtested on SPY 1993-present.

Buy when RSI(2) < 5 and daily close > SMA(200).
Exit when close > 5-day SMA of closes.

Implemented as intraday: entry at session open based on prior day's
daily RSI(2) from MarketContext, close at EOD.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategies.intraday.base import (
    BaseIntradayStrategy,
    InfraExitConfig,
)
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.exit import IntradayExitManager
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.trade_mgmt import (
    make_entry_signal,
    directional_sl_tp,
)

if TYPE_CHECKING:
    from stockdownloader.strategies.intraday.session import BarContext
    from stockdownloader.strategies.intraday.market_context import (
        MarketContext,
        MarketContextProvider,
    )

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class ConnorsRSI2Config(InfraExitConfig):
    """Config for Connors RSI(2) mean reversion strategy."""

    rsi_period: int = 2
    rsi_threshold: Decimal = Decimal("5")
    sma_trend_period: int = 200
    exit_sma_period: int = 5
    entry_bar: int = 1  # Enter at market open
    sl_atr_mult: Decimal = Decimal("2.0")
    allow_longs: bool = True
    allow_shorts: bool = False  # Long-only per Connors
    close_eod: bool = True
    max_day: int = 1


class ConnorsRSI2Strategy(BaseIntradayStrategy):
    """Connors RSI(2) mean reversion: buy extreme oversold, ride the bounce."""

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(
        self,
        config: ConnorsRSI2Config | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            self._c = ConnorsRSI2Config(**overrides)
        else:
            self._c = config or ConnorsRSI2Config()
        self._infra = IntradayInfra(
            self._c, IntradayExitManager(), market_ctx_provider=market_ctx_provider,
        )
        super().__init__()
        self._enter_today: bool = False
        self._last_session_date: str = ""

    @property
    def name(self) -> str:
        return f"Connors RSI({self._c.rsi_period})"

    def _should_enter_today(self, ctx: MarketContext) -> bool:
        """Check daily-level entry conditions from MarketContext."""
        return (
            ctx.daily_rsi2 < self._c.rsi_threshold
            and ctx.daily_close_above_sma200
        )

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c

        # Determine entry signal at session start
        if ctx.bar.trading_date != self._last_session_date:
            self._last_session_date = ctx.bar.trading_date
            self._enter_today = False
            if ctx.market_ctx is not None:
                self._enter_today = self._should_enter_today(ctx.market_ctx)

        # Only enter at the configured entry bar
        if ctx.bar_of_day != c.entry_bar:
            return None

        if not self._enter_today:
            return None

        if not c.allow_longs:
            return None

        # SL/TP
        sl_dist = ctx.atr_val * c.sl_atr_mult
        if sl_dist <= ZERO:
            return None
        tp_dist = sl_dist * Decimal("3.0")  # Wide TP, EOD exit dominates

        sl_price, tp_price = directional_sl_tp(True, ctx.bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=True,
            mode="Connors-RSI2",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason=f"RSI(2)={ctx.market_ctx.daily_rsi2:.1f} < {c.rsi_threshold}",
        )
```

**Step 4: Run tests, then regression**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_connors_rsi2.py -v
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/ -x -q
```

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/connors_rsi2.py tests/strategies/intraday/test_connors_rsi2.py
git commit -m "feat: add Connors RSI(2) mean reversion strategy (75% win rate)"
```

---

## Task 6: VIX Regime Filter wrapper

**Files:**
- Create: `src/stockdownloader/strategies/intraday/vix_regime_filter.py`
- Create: `tests/strategies/intraday/test_vix_regime_filter.py`

**Step 1: Write tests**

```python
# tests/strategies/intraday/test_vix_regime_filter.py
"""Tests for VixFilteredStrategy."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from stockdownloader.core.models.trade import IntradayAction, IntradaySignal
from stockdownloader.strategies.intraday.vix_regime_filter import VixFilteredStrategy
from stockdownloader.strategies.intraday.market_context import (
    MarketContext,
    FileMarketContextProvider,
)


def _mock_market_ctx(vix_regime: str) -> MarketContext:
    return MarketContext(
        vix_close=Decimal("20"),
        vix_sma20=Decimal("18"),
        vix_regime=vix_regime,
        is_fomc_day=False,
        is_fomc_press_conf=False,
        is_opex=False,
        daily_rsi2=Decimal("50"),
        daily_close_above_sma200=True,
    )


class TestVixFilteredStrategy:
    def test_blocks_in_disallowed_regime(self):
        """Wrapping a strategy that would fire — blocked by VIX filter."""
        inner = MagicMock()
        entry_sig = IntradaySignal(action=IntradayAction.ENTER_LONG)
        inner.evaluate.return_value = entry_sig
        inner.name = "Inner"
        inner.warmup_period = 100

        wrapped = VixFilteredStrategy(
            inner=inner,
            allowed_regimes=["high", "extreme"],
        )
        # We need the wrapper to know the current VIX regime
        # This happens via MarketContext injection — test the regime check method
        assert wrapped._is_regime_allowed("low") is False
        assert wrapped._is_regime_allowed("mid") is False
        assert wrapped._is_regime_allowed("high") is True

    def test_allows_in_permitted_regime(self):
        wrapped = VixFilteredStrategy(
            inner=MagicMock(),
            allowed_regimes=["mid", "high"],
        )
        assert wrapped._is_regime_allowed("mid") is True
        assert wrapped._is_regime_allowed("high") is True

    def test_name_includes_vix_label(self):
        inner = MagicMock()
        inner.name = "OR Reversal"
        wrapped = VixFilteredStrategy(inner=inner, allowed_regimes=["mid", "high"])
        assert "VIX" in wrapped.name
        assert "OR Reversal" in wrapped.name
```

**Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_vix_regime_filter.py -v
```

**Step 3: Write implementation**

```python
# src/stockdownloader/strategies/intraday/vix_regime_filter.py
"""VIX Regime Filter — wraps any intraday strategy with VIX conditioning.

Cross-cutting finding from academic research: VIX filtering improves
risk-adjusted returns by 8-15 percentage points across all strategy
categories. This wrapper lets you add VIX regime filtering to any
existing strategy without modifying it.

Usage:
    from stockdownloader.strategies.intraday.or_reversal import ORReversalStrategy
    from stockdownloader.strategies.intraday.vix_regime_filter import VixFilteredStrategy

    filtered = VixFilteredStrategy(
        inner=ORReversalStrategy(),
        allowed_regimes=["mid", "high"],
        market_ctx_provider=provider,
    )
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from stockdownloader.core.models.trade import IntradaySignal, IntradayAction
from stockdownloader.strategies.base import IntradayTradingStrategy

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.intraday.market_context import (
        MarketContext,
        MarketContextProvider,
    )

HOLD = IntradaySignal(action=IntradayAction.HOLD)


class VixFilteredStrategy(IntradayTradingStrategy):
    """Wraps any intraday strategy with VIX regime filtering.

    Delegates all calls to the inner strategy, but blocks entry signals
    when the VIX regime is not in the allowed list.
    """

    def __init__(
        self,
        inner: IntradayTradingStrategy,
        allowed_regimes: list[str] | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
    ) -> None:
        self._inner = inner
        self._allowed = allowed_regimes or ["mid", "high"]
        self._provider = market_ctx_provider
        self._current_ctx: MarketContext | None = None
        self._last_session_date: str = ""

    @property
    def name(self) -> str:
        regimes = "/".join(self._allowed)
        return f"VIX[{regimes}] {self._inner.name}"

    @property
    def warmup_period(self) -> int:
        return self._inner.warmup_period

    def _is_regime_allowed(self, regime: str) -> bool:
        return regime in self._allowed

    def on_session_start(self, trading_date: str) -> None:
        self._inner.on_session_start(trading_date)
        if self._provider is not None:
            self._current_ctx = self._provider.get_context(trading_date)

    def on_position_opened(self, is_long: bool) -> None:
        self._inner.on_position_opened(is_long)

    def on_position_closed(self) -> None:
        self._inner.on_position_closed()

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        # Detect session change and update context
        bar = data[current_index]
        if bar.trading_date != self._last_session_date:
            self._last_session_date = bar.trading_date
            if self._provider is not None:
                self._current_ctx = self._provider.get_context(bar.trading_date)

        sig = self._inner.evaluate(data, current_index)

        # Only filter entry signals — let exits through
        if sig.action in (IntradayAction.ENTER_LONG, IntradayAction.ENTER_SHORT):
            if self._current_ctx is None:
                return HOLD  # No context → block entries
            if not self._is_regime_allowed(self._current_ctx.vix_regime):
                return HOLD
        return sig
```

**Step 4: Run tests, then regression**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_vix_regime_filter.py -v
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/ -x -q
```

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/vix_regime_filter.py tests/strategies/intraday/test_vix_regime_filter.py
git commit -m "feat: add VIX Regime Filter wrapper for any intraday strategy"
```

---

## Task 7: FOMC Drift strategy

**Files:**
- Create: `src/stockdownloader/strategies/intraday/fomc_drift.py`
- Create: `tests/strategies/intraday/test_fomc_drift.py`

**Step 1: Write tests**

```python
# tests/strategies/intraday/test_fomc_drift.py
"""Tests for FOMCDriftStrategy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.fomc_drift import (
    FOMCDriftConfig,
    FOMCDriftStrategy,
)
from stockdownloader.strategies.intraday.market_context import MarketContext


class TestFOMCDriftConfig:
    def test_defaults(self):
        c = FOMCDriftConfig()
        assert c.entry_bar == 1
        assert c.sl_atr_mult == Decimal("2.0")
        assert c.allow_shorts is False

    def test_json_round_trip(self):
        c = FOMCDriftConfig(vix_threshold=Decimal("25"))
        c2 = FOMCDriftConfig.from_json(c.to_json())
        assert c == c2


class TestFOMCDriftEntry:
    def test_enter_on_fomc_press_conf_day(self):
        ctx = MarketContext(
            vix_close=Decimal("22"),
            vix_sma20=Decimal("20"),
            vix_regime="mid",
            is_fomc_day=True,
            is_fomc_press_conf=True,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy()
        assert strat._should_enter_today(ctx) is True

    def test_no_entry_on_non_fomc_day(self):
        ctx = MarketContext(
            vix_close=Decimal("22"),
            vix_sma20=Decimal("20"),
            vix_regime="mid",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy()
        assert strat._should_enter_today(ctx) is False

    def test_no_entry_fomc_but_low_vix(self):
        """FOMC press conf + low VIX → skip (require_high_vix=True)."""
        ctx = MarketContext(
            vix_close=Decimal("12"),
            vix_sma20=Decimal("13"),
            vix_regime="low",
            is_fomc_day=True,
            is_fomc_press_conf=True,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy(require_high_vix=True)
        assert strat._should_enter_today(ctx) is False

    def test_entry_fomc_low_vix_when_filter_off(self):
        """FOMC press conf + low VIX → enter when filter disabled."""
        ctx = MarketContext(
            vix_close=Decimal("12"),
            vix_sma20=Decimal("13"),
            vix_regime="low",
            is_fomc_day=True,
            is_fomc_press_conf=True,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy(require_high_vix=False)
        assert strat._should_enter_today(ctx) is True
```

**Step 2: Run tests to verify they fail**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_fomc_drift.py -v
```

**Step 3: Write implementation**

```python
# src/stockdownloader/strategies/intraday/fomc_drift.py
"""FOMC Drift strategy.

Based on Lucca & Moench (2015), "The Pre-FOMC Announcement Drift,"
Journal of Finance.  Updated by Ignatieva & Ohashi (2024) — now
concentrated on press conference days only (Sharpe ~1.8).

Enter long at market open on FOMC press conference days.  The drift
is strongest during elevated VIX periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.strategies.intraday.base import (
    BaseIntradayStrategy,
    InfraExitConfig,
)
from stockdownloader.strategies.intraday.infra import IntradayInfra
from stockdownloader.strategies.intraday.exit import IntradayExitManager
from stockdownloader.core.models.trade import IntradaySignal
from stockdownloader.strategies.intraday.trade_mgmt import (
    make_entry_signal,
    directional_sl_tp,
)

if TYPE_CHECKING:
    from stockdownloader.strategies.intraday.session import BarContext
    from stockdownloader.strategies.intraday.market_context import (
        MarketContext,
        MarketContextProvider,
    )

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class FOMCDriftConfig(InfraExitConfig):
    """Config for FOMC Drift strategy."""

    entry_bar: int = 1
    sl_atr_mult: Decimal = Decimal("2.0")
    require_high_vix: bool = True
    vix_threshold: Decimal = Decimal("20")
    allow_longs: bool = True
    allow_shorts: bool = False
    close_eod: bool = True
    max_day: int = 1


class FOMCDriftStrategy(BaseIntradayStrategy):
    """FOMC press conference day drift: long at open, close at EOD."""

    _ENTRY_FLAGS = {"fire_once": True}

    def __init__(
        self,
        config: FOMCDriftConfig | None = None,
        market_ctx_provider: MarketContextProvider | None = None,
        **overrides: object,
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Cannot pass both 'config' and keyword overrides")
        if overrides:
            self._c = FOMCDriftConfig(**overrides)
        else:
            self._c = config or FOMCDriftConfig()
        self._infra = IntradayInfra(
            self._c, IntradayExitManager(), market_ctx_provider=market_ctx_provider,
        )
        super().__init__()
        self._enter_today: bool = False
        self._last_session_date: str = ""

    @property
    def name(self) -> str:
        vix_label = f" VIX>{self._c.vix_threshold}" if self._c.require_high_vix else ""
        return f"FOMC Drift{vix_label}"

    def _should_enter_today(self, ctx: MarketContext) -> bool:
        """Check if today is a tradeable FOMC day."""
        if not ctx.is_fomc_press_conf:
            return False
        if self._c.require_high_vix and ctx.vix_close < self._c.vix_threshold:
            return False
        return True

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c

        # Determine entry at session start
        if ctx.bar.trading_date != self._last_session_date:
            self._last_session_date = ctx.bar.trading_date
            self._enter_today = False
            if ctx.market_ctx is not None:
                self._enter_today = self._should_enter_today(ctx.market_ctx)

        # Only enter at configured bar
        if ctx.bar_of_day != c.entry_bar:
            return None

        if not self._enter_today:
            return None

        if not c.allow_longs:
            return None

        # SL/TP — wide stop for event-driven trade
        sl_dist = ctx.atr_val * c.sl_atr_mult
        if sl_dist <= ZERO:
            return None
        tp_dist = sl_dist * Decimal("5.0")  # Wide TP, EOD exit dominates

        sl_price, tp_price = directional_sl_tp(True, ctx.bar.close, sl_dist, tp_dist)

        return make_entry_signal(
            go_long=True,
            mode="FOMC-Drift",
            sl_price=sl_price,
            tp_price=tp_price,
            score=1,
            max_score=1,
            risk_per_share=sl_dist,
            reason="FOMC press conf day drift",
        )
```

**Step 4: Run tests, then regression**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_fomc_drift.py -v
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/ -x -q
```

**Step 5: Commit**

```bash
git add src/stockdownloader/strategies/intraday/fomc_drift.py tests/strategies/intraday/test_fomc_drift.py
git commit -m "feat: add FOMC Drift strategy (Lucca & Moench 2015, Sharpe 1.14-1.8)"
```

---

## Task 8: Data pipeline, registry, and integration tests

**Files:**
- Create: `scripts/prepare_market_context.py`
- Modify: `config/strategies/intraday_registrations.json`
- Create: `tests/strategies/intraday/test_evidence_integration.py`

**Step 1: Create the market context data pipeline script**

```python
# scripts/prepare_market_context.py
"""Prepare MarketContext CSV from Polygon VIX + SPY daily data.

Usage:
    PYTHONPATH=src python3 scripts/prepare_market_context.py

Outputs:
    data/SPY/market_context.csv
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockdownloader.strategies.intraday.market_context import vix_regime_from_level

# ── FOMC dates (press conference days) ────────────────────────────────
# Source: Federal Reserve website
# Format: YYYY-MM-DD for press conference days (every other meeting)
FOMC_PRESS_CONF_DATES = {
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
}

FOMC_ALL_DATES = FOMC_PRESS_CONF_DATES | {
    # Non-press-conference FOMC meetings
    # 2024
    "2024-03-19", "2024-04-30", "2024-06-11", "2024-07-30",
    "2024-09-17", "2024-11-06", "2024-12-17",
    # 2025
    "2025-01-28", "2025-03-18", "2025-05-06", "2025-06-17",
    "2025-07-29", "2025-09-16", "2025-10-28", "2025-12-16",
}


def _is_opex(date_str: str) -> bool:
    """Third Friday of the month = monthly options expiration."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() != 4:  # Not Friday
        return False
    day = dt.day
    return 15 <= day <= 21


def _compute_rsi(closes: list[Decimal], period: int = 2) -> Decimal:
    """Compute RSI from a list of daily closes."""
    if len(closes) < period + 1:
        return Decimal("50")  # Neutral default
    gains = []
    losses = []
    for i in range(-period, 0):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return Decimal("100")
    rs = avg_gain / avg_loss
    return Decimal("100") - Decimal("100") / (1 + rs)


def _compute_sma(values: list[Decimal], period: int) -> Decimal:
    """Compute simple moving average."""
    if len(values) < period:
        return values[-1] if values else Decimal("0")
    return sum(values[-period:]) / period


def main():
    """Build market_context.csv from local data files."""
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # Load SPY daily bars (aggregated from 5m or fetched separately)
    spy_5m_path = data_dir / "SPY" / "5m_bars.csv"
    if not spy_5m_path.exists():
        print(f"SPY 5m data not found at {spy_5m_path}")
        sys.exit(1)

    # Aggregate 5m bars to daily
    print("Aggregating SPY 5m bars to daily...")
    daily_bars: dict[str, dict] = {}
    with open(spy_5m_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["Datetime"][:10]
            o = Decimal(row["Open"])
            h = Decimal(row["High"])
            l = Decimal(row["Low"])
            c = Decimal(row["Close"])
            v = int(row["Volume"])
            if date not in daily_bars:
                daily_bars[date] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
            else:
                d = daily_bars[date]
                d["high"] = max(d["high"], h)
                d["low"] = min(d["low"], l)
                d["close"] = c
                d["volume"] += v

    dates = sorted(daily_bars.keys())
    closes = [daily_bars[d]["close"] for d in dates]

    # Try loading VIX data — fall back to synthetic if not available
    vix_path = data_dir / "VIX" / "daily.csv"
    vix_data: dict[str, Decimal] = {}
    if vix_path.exists():
        with open(vix_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vix_data[row["Date"][:10]] = Decimal(row["Close"])
        print(f"Loaded {len(vix_data)} VIX daily records")
    else:
        print("VIX daily data not found — using synthetic VIX proxy (ATR-based)")
        # Synthetic VIX proxy from 20-day realized volatility
        for i, date in enumerate(dates):
            if i < 20:
                vix_data[date] = Decimal("18")  # Default
            else:
                returns = []
                for j in range(i - 20, i):
                    if closes[j - 1] > 0:
                        ret = abs((closes[j] - closes[j - 1]) / closes[j - 1])
                        returns.append(ret)
                avg_ret = sum(returns) / len(returns) if returns else Decimal("0.01")
                vix_data[date] = avg_ret * Decimal("1590")  # Annualized approx

    # Build context CSV
    out_path = data_dir / "SPY" / "market_context.csv"
    vix_closes: list[Decimal] = []
    print(f"Writing market context for {len(dates)} dates...")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "vix_close", "vix_sma20", "is_fomc",
                         "is_fomc_pc", "is_opex", "rsi2", "above_sma200"])
        for i, date in enumerate(dates):
            vix = vix_data.get(date, Decimal("18"))
            vix_closes.append(vix)
            vix_sma20 = _compute_sma(vix_closes, 20)
            rsi2 = _compute_rsi(closes[:i + 1], period=2)
            sma200 = _compute_sma(closes[:i + 1], 200)
            above_sma200 = closes[i] > sma200

            writer.writerow([
                date,
                f"{vix:.2f}",
                f"{vix_sma20:.2f}",
                "1" if date in FOMC_ALL_DATES else "0",
                "1" if date in FOMC_PRESS_CONF_DATES else "0",
                "1" if _is_opex(date) else "0",
                f"{rsi2:.2f}",
                "1" if above_sma200 else "0",
            ])

    print(f"Wrote {out_path} ({len(dates)} rows)")


if __name__ == "__main__":
    main()
```

**Step 2: Add strategy registrations to intraday_registrations.json**

Add the following entries to the existing JSON file:

```json
{
    "spy-gao-momentum": {
        "display_name": "Gao Intraday Momentum",
        "module": "stockdownloader.strategies.intraday.gao_momentum",
        "class": "GaoMomentumStrategy",
        "default_kwargs": {},
        "param_space": {
            "require_dual_signal": [true, false],
            "min_r1_magnitude": ["D:0.0003", "D:0.0005", "D:0.001"],
            "sl_atr_mult": ["D:1.0", "D:1.5", "D:2.0"],
            "vix_filter": [true, false]
        }
    },
    "spy-noise-boundary": {
        "display_name": "Noise Boundary Breakout",
        "module": "stockdownloader.strategies.intraday.noise_boundary",
        "class": "NoiseBoundaryStrategy",
        "default_kwargs": {},
        "param_space": {
            "lookback_days": [7, 14, 21],
            "vol_multiplier": ["D:0.8", "D:1.0", "D:1.2"],
            "sl_atr_mult": ["D:1.5", "D:2.0", "D:2.5"]
        }
    },
    "spy-connors-rsi2": {
        "display_name": "Connors RSI(2) Mean Reversion",
        "module": "stockdownloader.strategies.intraday.connors_rsi2",
        "class": "ConnorsRSI2Strategy",
        "default_kwargs": {},
        "param_space": {
            "rsi_threshold": ["D:3", "D:5", "D:10"],
            "sl_atr_mult": ["D:1.5", "D:2.0", "D:3.0"]
        }
    },
    "spy-fomc-drift": {
        "display_name": "FOMC Drift",
        "module": "stockdownloader.strategies.intraday.fomc_drift",
        "class": "FOMCDriftStrategy",
        "default_kwargs": {},
        "param_space": {
            "require_high_vix": [true, false],
            "vix_threshold": ["D:15", "D:20", "D:25"],
            "sl_atr_mult": ["D:1.5", "D:2.0", "D:3.0"]
        }
    }
}
```

**Step 3: Write integration test**

```python
# tests/strategies/intraday/test_evidence_integration.py
"""Integration tests for evidence-based strategies."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.gao_momentum import GaoMomentumStrategy
from stockdownloader.strategies.intraday.noise_boundary import NoiseBoundaryStrategy
from stockdownloader.strategies.intraday.connors_rsi2 import ConnorsRSI2Strategy
from stockdownloader.strategies.intraday.fomc_drift import FOMCDriftStrategy
from stockdownloader.strategies.intraday.vix_regime_filter import VixFilteredStrategy


class TestAllStrategiesConstructible:
    """Every new strategy must be constructible with defaults."""

    def test_gao_momentum_default(self):
        s = GaoMomentumStrategy()
        assert s.name
        assert s.warmup_period > 0

    def test_noise_boundary_default(self):
        s = NoiseBoundaryStrategy()
        assert s.name
        assert s.warmup_period > 0

    def test_connors_rsi2_default(self):
        s = ConnorsRSI2Strategy()
        assert s.name
        assert s.warmup_period > 0

    def test_fomc_drift_default(self):
        s = FOMCDriftStrategy()
        assert s.name
        assert s.warmup_period > 0

    def test_vix_filtered_wrapper(self):
        inner = GaoMomentumStrategy()
        wrapped = VixFilteredStrategy(inner=inner, allowed_regimes=["mid", "high"])
        assert "VIX" in wrapped.name
        assert wrapped.warmup_period > 0


class TestOverridesWork:
    """New strategies must accept **overrides via unified constructor."""

    def test_gao_overrides(self):
        s = GaoMomentumStrategy(entry_start_bar=70, require_dual_signal=False)
        assert s._c.entry_start_bar == 70
        assert s._c.require_dual_signal is False

    def test_noise_overrides(self):
        s = NoiseBoundaryStrategy(lookback_days=10, vol_multiplier=Decimal("1.5"))
        assert s._c.lookback_days == 10

    def test_connors_overrides(self):
        s = ConnorsRSI2Strategy(rsi_threshold=Decimal("10"))
        assert s._c.rsi_threshold == Decimal("10")

    def test_fomc_overrides(self):
        s = FOMCDriftStrategy(require_high_vix=False)
        assert s._c.require_high_vix is False


class TestConfigSerialization:
    """Config round-trip via StrategyConfigMixin."""

    def test_gao_config_round_trip(self):
        from stockdownloader.strategies.intraday.gao_momentum import GaoMomentumConfig
        c = GaoMomentumConfig(min_r1_magnitude=Decimal("0.001"))
        c2 = GaoMomentumConfig.from_json(c.to_json())
        assert c == c2

    def test_noise_config_round_trip(self):
        from stockdownloader.strategies.intraday.noise_boundary import NoiseBoundaryConfig
        c = NoiseBoundaryConfig(vol_multiplier=Decimal("1.5"))
        c2 = NoiseBoundaryConfig.from_json(c.to_json())
        assert c == c2

    def test_connors_config_round_trip(self):
        from stockdownloader.strategies.intraday.connors_rsi2 import ConnorsRSI2Config
        c = ConnorsRSI2Config(rsi_threshold=Decimal("3"))
        c2 = ConnorsRSI2Config.from_json(c.to_json())
        assert c == c2

    def test_fomc_config_round_trip(self):
        from stockdownloader.strategies.intraday.fomc_drift import FOMCDriftConfig
        c = FOMCDriftConfig(vix_threshold=Decimal("25"))
        c2 = FOMCDriftConfig.from_json(c.to_json())
        assert c == c2
```

**Step 4: Run all tests**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_evidence_integration.py -v
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q
```

**Step 5: Run the data pipeline**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 scripts/prepare_market_context.py
```

**Step 6: Commit**

```bash
git add scripts/prepare_market_context.py config/strategies/intraday_registrations.json tests/strategies/intraday/test_evidence_integration.py
git commit -m "feat: add data pipeline, registry entries, and integration tests for evidence-based strategies"
```

---

## Verification

```bash
# Unit + integration tests
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/strategies/intraday/test_market_context.py tests/strategies/intraday/test_gao_momentum.py tests/strategies/intraday/test_noise_boundary.py tests/strategies/intraday/test_connors_rsi2.py tests/strategies/intraday/test_vix_regime_filter.py tests/strategies/intraday/test_fomc_drift.py tests/strategies/intraday/test_evidence_integration.py -v

# Full regression
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q

# Verify all new strategies create via registry
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -c "
from stockdownloader.strategies.loader import ensure_registered
from stockdownloader.strategies.registry import StrategyRegistry
ensure_registered()
for name in ['spy-gao-momentum', 'spy-noise-boundary', 'spy-connors-rsi2', 'spy-fomc-drift']:
    s = StrategyRegistry.create(name)
    print(f'  OK: {name:<25} → {s.name}')
print('All evidence-based strategies registered')
"

# Verify config serialization
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -c "
from stockdownloader.strategies.intraday.gao_momentum import GaoMomentumConfig
from decimal import Decimal
c = GaoMomentumConfig(min_r1_magnitude=Decimal('0.001'))
j = c.to_json()
c2 = GaoMomentumConfig.from_json(j)
assert c == c2
print('Config serialization round-trip: OK')
print(j)
"
```
