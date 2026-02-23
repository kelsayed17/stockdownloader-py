# Backtest Package Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `tournament_engine.py` (932 lines) into 3 focused modules by responsibility, fix report formatter duplication, complete `__init__.py` exports.

**Architecture:** Extract data models, worker functions, and analysis logic into separate modules. The slim `tournament_engine.py` keeps bracket logic and re-exports everything for backward compatibility (zero caller changes). Shared rounding utilities extracted from duplicate report formatters.

**Tech Stack:** Python 3.12, dataclasses, pytest, multiprocessing-safe functions

---

### Task 1: Create `report_helpers.py` — Shared Rounding Utilities

**Files:**
- Create: `src/stockdownloader/backtest/report_helpers.py`
- Create: `tests/backtest/test_report_helpers.py`
- Modify: `src/stockdownloader/backtest/report_formatter.py:50-52`
- Modify: `src/stockdownloader/backtest/exit_tournament_report_formatter.py:15-22`

**Step 1: Write the tests**

Create `tests/backtest/test_report_helpers.py`:

```python
"""Tests for shared report rounding utilities."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.backtest.report_helpers import scale2, scale3


class TestScale2:
    def test_rounds_down(self):
        assert scale2(Decimal("1.234")) == Decimal("1.23")

    def test_rounds_up(self):
        assert scale2(Decimal("1.235")) == Decimal("1.24")

    def test_no_change(self):
        assert scale2(Decimal("1.23")) == Decimal("1.23")

    def test_whole_number(self):
        assert scale2(Decimal("5")) == Decimal("5.00")

    def test_negative(self):
        assert scale2(Decimal("-3.456")) == Decimal("-3.46")

    def test_zero(self):
        assert scale2(Decimal("0")) == Decimal("0.00")


class TestScale3:
    def test_rounds_down(self):
        assert scale3(Decimal("1.2344")) == Decimal("1.234")

    def test_rounds_up(self):
        assert scale3(Decimal("1.2345")) == Decimal("1.235")

    def test_no_change(self):
        assert scale3(Decimal("1.234")) == Decimal("1.234")

    def test_whole_number(self):
        assert scale3(Decimal("5")) == Decimal("5.000")

    def test_negative(self):
        assert scale3(Decimal("-3.4567")) == Decimal("-3.457")
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/backtest/test_report_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockdownloader.backtest.report_helpers'`

**Step 3: Create `report_helpers.py`**

Create `src/stockdownloader/backtest/report_helpers.py`:

```python
"""Shared rounding utilities for report formatters.

Provides :func:`scale2` and :func:`scale3` — previously duplicated
as ``scale2()`` in :mod:`report_formatter` and ``_s2()`` / ``_s3()``
in :mod:`exit_tournament_report_formatter`.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def scale2(value: Decimal) -> Decimal:
    """Round a Decimal to 2 decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def scale3(value: Decimal) -> Decimal:
    """Round a Decimal to 3 decimal places."""
    return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/backtest/test_report_helpers.py -v`
Expected: 11 PASSED

**Step 5: Wire `report_formatter.py` to import from `report_helpers`**

In `src/stockdownloader/backtest/report_formatter.py`:
- Remove lines 50–52 (the inline `scale2` function)
- Add import after the existing imports (around line 39): `from stockdownloader.backtest.report_helpers import scale2`

**Step 6: Wire `exit_tournament_report_formatter.py` to import from `report_helpers`**

In `src/stockdownloader/backtest/exit_tournament_report_formatter.py`:
- Remove lines 15–22 (inline `_s2` and `_s3` functions)
- Add import after line 9: `from stockdownloader.backtest.report_helpers import scale2 as _s2, scale3 as _s3`

This preserves the private naming convention — all internal call sites using `_s2()` and `_s3()` continue to work without changes.

**Step 7: Run full test suite to verify nothing broke**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3209+)

**Step 8: Commit**

```bash
git add src/stockdownloader/backtest/report_helpers.py \
        src/stockdownloader/backtest/report_formatter.py \
        src/stockdownloader/backtest/exit_tournament_report_formatter.py \
        tests/backtest/test_report_helpers.py
git commit -m "refactor: extract report_helpers.py with shared scale2/scale3"
```

---

### Task 2: Create `tournament_models.py` — Data Models

**Files:**
- Create: `src/stockdownloader/backtest/tournament_models.py`
- Modify: `src/stockdownloader/backtest/tournament_engine.py`

This is a pure extraction — the 8 dataclasses have comprehensive existing tests in `test_tournament_engine.py`, `test_tournament_regime.py`, and `test_tournament_monte_carlo.py` that will validate the move.

**Step 1: Create `tournament_models.py`**

Create `src/stockdownloader/backtest/tournament_models.py` with the 8 dataclasses extracted verbatim from `tournament_engine.py` lines 44–187:

```python
"""Data models for the tournament engine.

Contains all dataclasses used by the tournament pipeline:
strategy × timeframe combo identifiers, backtest results,
regime analysis, Monte Carlo results, match results, and
the final tournament output container.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from stockdownloader.backtest.backtest_result import BacktestResult, BaseBacktestResult
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.walk_forward import WalkForwardResult
from stockdownloader.strategy.regime.regime_detector import MarketRegime


@dataclass(frozen=True, slots=True)
class RegimeTradeStats:
    """Performance of one combo in a single market regime."""

    regime: MarketRegime
    trade_count: int
    total_pnl: float
    win_count: int
    avg_pnl: float
    win_rate: float  # 0.0 to 1.0


@dataclass(frozen=True, slots=True)
class RegimeAnalysis:
    """Regime-aware analysis for one combo."""

    per_regime: dict[MarketRegime, RegimeTradeStats]
    regime_coverage: int  # how many of 5 regimes had >=3 trades
    worst_regime_pnl: float  # avg_pnl of worst active regime
    regime_consistency: float  # stdev of avg_pnl across active regimes
    regime_bonus: float  # computed bonus/penalty for tournament_score


@dataclass(frozen=True, slots=True)
class MonteCarloPercentiles:
    """Percentile values for a single metric across MC simulations."""

    p5: float
    p25: float
    p50: float  # median
    p75: float
    p95: float


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Monte Carlo robustness test results for one combo."""

    n_simulations: int
    n_trades: int
    # Trade-order shuffling results
    max_drawdown: MonteCarloPercentiles
    final_equity: MonteCarloPercentiles
    total_return: MonteCarloPercentiles
    # Bootstrap resampling results
    bootstrap_return: MonteCarloPercentiles
    bootstrap_sharpe: MonteCarloPercentiles
    # Robustness assessment
    is_robust: bool  # bootstrap_return.p5 > 0
    mc_penalty: float  # 0 if robust, proportional to fragility otherwise


@dataclass(frozen=True, slots=True)
class ComboKey:
    """Unique identifier for a strategy × timeframe combination."""

    strategy_name: str
    timeframe: str  # "5m", "15m", "30m", "1h", "4h", "1d"
    data_source: str = "resampled"  # CSV filename or "resampled"

    @property
    def label(self) -> str:
        return f"{self.strategy_name} @ {self.timeframe}"


@dataclass(slots=True)
class ComboResult:
    """Result of running one strategy × timeframe combination."""

    key: ComboKey
    display_name: str = ""
    category: str = ""
    baseline: BacktestResult | None = None
    optimized_kwargs: dict[str, Any] | None = None
    optimized: BacktestResult | None = None
    wf_result: WalkForwardResult | None = None
    regime_analysis: RegimeAnalysis | None = None
    monte_carlo: MonteCarloResult | None = None
    tournament_score: float = -999.0
    elapsed: float = 0.0
    error: str | None = None

    @property
    def best_result(self) -> BacktestResult | None:
        return self.optimized or self.baseline

    def compute_tournament_score(self, trading_days: int) -> float:
        """Compute composite tournament score (score_v2 + WF degradation)."""
        result = self.best_result
        if result is None:
            self.tournament_score = -999.0
            return self.tournament_score

        base = score_v2(result, trading_days)

        # Apply walk-forward quality factor
        if self.wf_result is not None:
            deg = self.wf_result.degradation_ratio
            if deg >= 0.8:
                factor = 1.0  # ROBUST
            elif deg >= 0.5:
                factor = 0.8  # ACCEPTABLE
            else:
                factor = 0.5  # OVERFIT

            if base >= 0:
                base *= factor
            else:
                base /= factor  # makes negative more negative

        # Regime bonus (additive)
        if self.regime_analysis is not None:
            base += self.regime_analysis.regime_bonus

        # Monte Carlo penalty (subtractive)
        if self.monte_carlo is not None:
            base -= self.monte_carlo.mc_penalty

        self.tournament_score = base
        return self.tournament_score


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of one head-to-head match in elimination bracket."""

    winner: ComboKey
    loser: ComboKey
    winner_score: float
    loser_score: float
    round_num: int
    round_name: str = ""


@dataclass(slots=True)
class TournamentResult:
    """Full tournament results across all stages."""

    combos: list[ComboResult] = field(default_factory=list)
    matches: list[MatchResult] = field(default_factory=list)
    bracket_champion: ComboKey | None = None
    portfolio: list[ComboResult] = field(default_factory=list)
    correlation_matrix: dict[tuple[str, str], float] = field(default_factory=dict)
```

**Step 2: Update `tournament_engine.py` — replace model definitions with imports**

In `tournament_engine.py`:
- Remove lines 39–187 (all 8 dataclass definitions and the `# Data models` section header)
- Remove now-unused imports: `field` from `dataclasses`, `Decimal`, `Any` (check if still used by workers)
- Remove `from stockdownloader.backtest.backtest_result import BacktestResult, BaseBacktestResult` (keep only if still used by workers — yes, workers use `BacktestResult`)
- Remove `from stockdownloader.backtest.optimizer_scoring import score_v2` (moved to models)
- Remove `from stockdownloader.backtest.walk_forward import WalkForwardResult, WalkForwardValidator` — keep `WalkForwardValidator` (used by workers), move `WalkForwardResult` to models import
- Add at top (after existing imports):

```python
from stockdownloader.backtest.tournament_models import (
    ComboKey,
    ComboResult,
    MatchResult,
    MonteCarloPercentiles,
    MonteCarloResult,
    RegimeAnalysis,
    RegimeTradeStats,
    TournamentResult,
)
```

Keep all remaining imports needed by workers/analysis/bracket.

**Step 3: Run existing tests to verify models still work**

Run: `python3 -m pytest tests/backtest/test_tournament_engine.py tests/backtest/test_tournament_regime.py tests/backtest/test_tournament_monte_carlo.py -v`
Expected: All PASS (tests import from `tournament_engine` which re-exports from `tournament_models`)

**Step 4: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3209+)

**Step 5: Commit**

```bash
git add src/stockdownloader/backtest/tournament_models.py \
        src/stockdownloader/backtest/tournament_engine.py
git commit -m "refactor: extract tournament_models.py with 8 dataclasses"
```

---

### Task 3: Create `tournament_workers.py` — Process-Safe Worker Functions

**Files:**
- Create: `src/stockdownloader/backtest/tournament_workers.py`
- Modify: `src/stockdownloader/backtest/tournament_engine.py`

Workers are process-safe standalone functions that run in multiprocessing pools. No new tests needed — existing callers (`stages.py`) import from `tournament_engine` which will re-export.

**Step 1: Create `tournament_workers.py`**

Create `src/stockdownloader/backtest/tournament_workers.py` with the 4 worker functions extracted verbatim from `tournament_engine.py` lines 189–449:

```python
"""Process-safe worker functions for tournament backtest execution.

Each function is designed to run safely in a multiprocessing pool:
no shared state, deferred imports for non-picklable objects, self-contained
error handling returning result tuples.

Functions
---------
run_combo_backtest
    Baseline backtest for one strategy × timeframe combo.
run_combo_optimize
    Walk-forward optimization for one combo.
run_combo_rebacktest
    Re-backtest with optimized params on full data.
run_combo_walkforward
    Walk-forward validation for one combo.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any

from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.tournament_models import ComboKey, ComboResult
from stockdownloader.backtest.walk_forward import WalkForwardResult, WalkForwardValidator
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.strategy.base_registry import StrategyRegistry
from stockdownloader.util.config import INITIAL_CAPITAL, RISK_PER_TRADE


def run_combo_backtest(
    key: ComboKey,
    data: list[IntradayPriceData],
) -> ComboResult:
    """Run baseline backtest for one combo (process-safe).

    Parameters
    ----------
    key:
        The strategy × timeframe combo to test.
    data:
        The price data for this timeframe (already resampled).
    """
    ensure_registered()
    t0 = time.time()

    try:
        entry = StrategyRegistry.get(key.strategy_name)
    except KeyError:
        return ComboResult(
            key=key,
            error=f"Strategy {key.strategy_name!r} not found in registry",
            elapsed=time.time() - t0,
        )

    engine = IntradayBacktestEngine(
        INITIAL_CAPITAL, RISK_PER_TRADE,
        vol_scale=False, dd_throttle=True,
    )

    try:
        if entry.category == "intraday":
            strategy = entry.factory(**entry.default_kwargs)
            result = engine.run(strategy, data)
        elif entry.category == "daily":
            daily_strategy = entry.factory(**entry.default_kwargs)
            adapter = DailyToIntradayAdapter(daily_strategy)
            result = engine.run(adapter, data)
        else:
            return ComboResult(
                key=key,
                display_name=entry.display_name,
                category=entry.category,
                error=f"Unsupported category: {entry.category}",
                elapsed=time.time() - t0,
            )

        trading_days = len({d.date[:10] for d in data})
        combo = ComboResult(
            key=key,
            display_name=entry.display_name,
            category=entry.category,
            baseline=result,
            elapsed=time.time() - t0,
        )
        combo.compute_tournament_score(trading_days)
        return combo

    except Exception as e:
        return ComboResult(
            key=key,
            display_name=entry.display_name,
            category=entry.category,
            error=str(e),
            elapsed=time.time() - t0,
        )


def run_combo_optimize(
    key: ComboKey,
    category: str,
    data: list[IntradayPriceData],
) -> tuple[ComboKey, Any | None, float, str | None]:
    """Run walk-forward optimization for one combo (process-safe).

    Creates a fresh WalkForwardOptimizer instance per process, searches
    IS data only, accepts if OOS improves.

    Parameters
    ----------
    key:
        The strategy × timeframe combo to optimize.
    category:
        Strategy category (``"intraday"`` or ``"daily"``).
    data:
        The price data for this timeframe.

    Returns
    -------
    tuple of (key, WFOptResult | None, elapsed, error)
    """
    from stockdownloader.backtest.walk_forward_optimizer import (
        WalkForwardOptimizer,
    )

    ensure_registered()
    t0 = time.time()

    try:
        wf_opt = WalkForwardOptimizer(
            data,
            split_ratio=0.7,
            initial_capital=INITIAL_CAPITAL,
            risk_per_trade=RISK_PER_TRADE,
            verbose=False,
        )
        results = wf_opt.optimize_all(
            categories=[category],
            strategy_filter=key.strategy_name,
        )
        wf_result = results[0] if results else None
        return key, wf_result, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)


def run_combo_rebacktest(
    key: ComboKey,
    category: str,
    optimized_kwargs: dict[str, Any],
    data: list[IntradayPriceData],
) -> tuple[ComboKey, BacktestResult | None, float, str | None]:
    """Re-backtest one combo with optimized params on full data (process-safe).

    Parameters
    ----------
    key:
        The strategy × timeframe combo.
    category:
        Strategy category (``"intraday"`` or ``"daily"``).
    optimized_kwargs:
        Optimized parameter overrides from walk-forward optimization.
    data:
        The *full* price data for this timeframe.

    Returns
    -------
    tuple of (key, result, elapsed, error)
    """
    ensure_registered()
    t0 = time.time()

    engine = IntradayBacktestEngine(
        INITIAL_CAPITAL, RISK_PER_TRADE,
        vol_scale=False, dd_throttle=True,
    )

    try:
        entry = StrategyRegistry.get(key.strategy_name)

        if category == "intraday":
            baseline_strategy = entry.factory(**entry.default_kwargs)
            default_config = baseline_strategy._infra._c
            opt_config = dataclasses.replace(default_config, **optimized_kwargs)
            strategy = entry.factory(config=opt_config)
            result = engine.run(strategy, data)
        elif category == "daily":
            adapter_keys = {"sl_atr_mult", "rr", "sl_cap", "allow_shorts"}
            strat_kwargs = {
                k: v for k, v in optimized_kwargs.items()
                if k not in adapter_keys
            }
            adapt_kwargs = {
                k: v for k, v in optimized_kwargs.items()
                if k in adapter_keys
            }
            daily_strategy = entry.factory(**strat_kwargs)
            adapter = DailyToIntradayAdapter(daily_strategy, **adapt_kwargs)
            result = engine.run(adapter, data)
        else:
            return key, None, time.time() - t0, f"Unsupported category: {category}"

        return key, result, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)


def run_combo_walkforward(
    key: ComboKey,
    data: list[IntradayPriceData],
    optimized_kwargs: dict[str, Any] | None = None,
) -> tuple[ComboKey, WalkForwardResult | None, float, str | None]:
    """Run walk-forward validation for one combo (process-safe).

    Parameters
    ----------
    key:
        The strategy × timeframe combo.
    data:
        The price data for this timeframe.
    optimized_kwargs:
        If provided, validate using these optimized params instead
        of default kwargs.

    Returns
    -------
    tuple of (key, wf_result, elapsed, error)
    """
    ensure_registered()
    t0 = time.time()

    try:
        entry = StrategyRegistry.get(key.strategy_name)
    except KeyError:
        return key, None, time.time() - t0, f"Strategy not found: {key.strategy_name}"

    try:
        validator = WalkForwardValidator(data, n_windows=5, is_ratio=0.7)
        engine = IntradayBacktestEngine(
            INITIAL_CAPITAL, RISK_PER_TRADE,
            vol_scale=False, dd_throttle=True,
        )

        if entry.category == "intraday":
            if optimized_kwargs:
                def _factory(e=entry, kw=optimized_kwargs):
                    baseline = e.factory(**e.default_kwargs)
                    default_config = baseline._infra._c
                    opt_config = dataclasses.replace(default_config, **kw)
                    return e.factory(config=opt_config)
            else:
                def _factory(e=entry):
                    return e.factory(**e.default_kwargs)
        elif entry.category == "daily":
            if optimized_kwargs:
                adapter_keys = {"sl_atr_mult", "rr", "sl_cap", "allow_shorts"}
                strat_kw = {
                    k: v for k, v in optimized_kwargs.items()
                    if k not in adapter_keys
                }
                adapt_kw = {
                    k: v for k, v in optimized_kwargs.items()
                    if k in adapter_keys
                }

                def _factory(e=entry, sk=strat_kw, ak=adapt_kw):
                    daily = e.factory(**sk)
                    return DailyToIntradayAdapter(daily, **ak)
            else:
                def _factory(e=entry):
                    daily = e.factory(**e.default_kwargs)
                    return DailyToIntradayAdapter(daily)
        else:
            return key, None, time.time() - t0, "unsupported category"

        wf = validator.validate(
            strategy_factory=_factory,
            engine=engine,
            strategy_name=entry.display_name,
        )
        return key, wf, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)
```

**Step 2: Update `tournament_engine.py` — replace worker functions with imports**

In `tournament_engine.py`:
- Remove the `# Process-safe worker functions` section header and all 4 functions (lines 189–449)
- Remove now-unused imports that were only needed by workers:
  - `dataclasses` (used by workers only)
  - `from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine`
  - `from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter`
  - `from stockdownloader.strategy.registration_loader import ensure_registered`
  - `from stockdownloader.strategy.base_registry import StrategyRegistry`
  - Check each — some may still be used by analysis functions. `IntradayBacktestEngine`, `DailyToIntradayAdapter`, `ensure_registered`, `StrategyRegistry` are workers-only. `dataclasses` is workers-only. Keep `time` (used by analysis).
- Add re-export:

```python
from stockdownloader.backtest.tournament_workers import (
    run_combo_backtest,
    run_combo_optimize,
    run_combo_rebacktest,
    run_combo_walkforward,
)
```

Also update the `INITIAL_CAPITAL`, `RISK_PER_TRADE` import — keep it in `tournament_engine.py` since external callers import it from there.

**Step 3: Run existing tests**

Run: `python3 -m pytest tests/backtest/test_tournament_engine.py tests/backtest/test_tournament_regime.py tests/backtest/test_tournament_monte_carlo.py -v`
Expected: All PASS

**Step 4: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3209+)

**Step 5: Commit**

```bash
git add src/stockdownloader/backtest/tournament_workers.py \
        src/stockdownloader/backtest/tournament_engine.py
git commit -m "refactor: extract tournament_workers.py with 4 process-safe functions"
```

---

### Task 4: Create `tournament_analysis.py` — Regime + Monte Carlo + Cross-TF

**Files:**
- Create: `src/stockdownloader/backtest/tournament_analysis.py`
- Modify: `src/stockdownloader/backtest/tournament_engine.py`

All 7 analysis functions (4 public, 3 private) move out. Existing tests in `test_tournament_regime.py` and `test_tournament_monte_carlo.py` validate correctness.

**Step 1: Create `tournament_analysis.py`**

Create `src/stockdownloader/backtest/tournament_analysis.py` with the 7 functions extracted verbatim from `tournament_engine.py` lines 452–845:

```python
"""Regime analysis, Monte Carlo robustness testing, and cross-timeframe
consistency bonus for the tournament pipeline.

Functions
---------
classify_timeframe_bars
    Classify every bar's market regime for one timeframe.
run_regime_analysis
    Compute regime-aware stats for one combo (process-safe).
run_monte_carlo
    Run Monte Carlo robustness tests for one combo (process-safe).
apply_cross_timeframe_bonus
    Add consistency bonus for strategies profitable across TFs.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from collections import defaultdict
from typing import Any

from stockdownloader.backtest.tournament_models import (
    ComboKey,
    ComboResult,
    MonteCarloPercentiles,
    MonteCarloResult,
    RegimeAnalysis,
    RegimeTradeStats,
)
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.strategy.regime.regime_detector import MarketRegime


# ======================================================================
# Regime analysis
# ======================================================================


def classify_timeframe_bars(
    data: list[IntradayPriceData],
) -> dict[str, MarketRegime]:
    """Classify every bar's regime for one timeframe.

    Runs in the main process (IndicatorHub is not process-safe).
    Creates a fresh detector and classifies every bar, mapping
    ``data[i].date -> MarketRegime``.  Bars within the warmup
    period are mapped to ``WEAK_TREND``.

    Parameters
    ----------
    data:
        Price data for one timeframe.

    Returns
    -------
    Dict mapping bar date string to MarketRegime.
    """
    from stockdownloader.util.indicators.hub import IndicatorHub
    from stockdownloader.strategy.regime.regime_detector import MarketRegimeDetector

    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub)
    warmup = detector.warmup_period

    regime_map: dict[str, MarketRegime] = {}

    for i, bar in enumerate(data):
        if i < warmup:
            regime_map[bar.date] = MarketRegime.WEAK_TREND
        else:
            try:
                rc = detector.classify(data, i)
                regime_map[bar.date] = rc.regime
            except (ValueError, KeyError, IndexError):
                regime_map[bar.date] = MarketRegime.WEAK_TREND

    return regime_map


def _compute_regime_bonus(
    per_regime: dict[MarketRegime, RegimeTradeStats],
) -> float:
    """Compute regime-aware scoring bonus/penalty.

    Rewards coverage (trades in many regimes), consistency (similar
    avg_pnl across regimes), and penalises regime collapse (one regime
    with large negative avg_pnl) and single-regime dependency.
    """
    # Filter regimes with meaningful trade count
    active = {r: s for r, s in per_regime.items() if s.trade_count >= 3}

    if not active:
        return 0.0

    # 1. Coverage bonus: +1.0 per regime with >= 3 trades (max +5.0)
    coverage_bonus = len(active) * 1.0

    # 2. Consistency bonus: low stdev of avg_pnl across regimes
    avg_pnls = [s.avg_pnl for s in active.values()]
    if len(avg_pnls) >= 2:
        stdev = statistics.stdev(avg_pnls)
        consistency_bonus = 3.0 / (1.0 + stdev / 50.0)
    else:
        consistency_bonus = 0.0

    # 3. Regime collapse penalty: worst regime drags score down
    worst_avg = min(avg_pnls)
    collapse_penalty = 0.0
    if worst_avg < -50.0:
        collapse_penalty = abs(worst_avg + 50.0) * 0.1

    # 4. Single-regime penalty
    single_regime_penalty = 0.0
    if len(active) == 1:
        single_regime_penalty = 2.0

    return coverage_bonus + consistency_bonus - collapse_penalty - single_regime_penalty


def run_regime_analysis(
    key: ComboKey,
    trades: list[tuple[str, float, bool]],
    regime_at_bar: dict[str, MarketRegime],
) -> tuple[ComboKey, RegimeAnalysis | None, float, str | None]:
    """Compute regime-aware stats for one combo (process-safe).

    Parameters
    ----------
    key:
        The strategy x timeframe combo.
    trades:
        List of ``(entry_date_str, pnl_float, is_win)`` tuples.
    regime_at_bar:
        Pre-computed mapping from bar date string to MarketRegime.

    Returns
    -------
    tuple of (key, RegimeAnalysis | None, elapsed, error)
    """
    t0 = time.time()

    if not trades:
        return key, None, time.time() - t0, None

    try:
        # Accumulate stats per regime
        regime_data: dict[MarketRegime, dict] = defaultdict(
            lambda: {"count": 0, "pnl": 0.0, "wins": 0},
        )

        for entry_date, pnl, is_win in trades:
            # Look up regime — try exact match, then prefix match
            regime = regime_at_bar.get(entry_date)
            if regime is None:
                # Fallback: match on first 19 chars (YYYY-MM-DD HH:MM:SS)
                prefix = entry_date[:19]
                for bar_date, bar_regime in regime_at_bar.items():
                    if bar_date[:19] == prefix:
                        regime = bar_regime
                        break
            if regime is None:
                regime = MarketRegime.WEAK_TREND

            d = regime_data[regime]
            d["count"] += 1
            d["pnl"] += pnl
            if is_win:
                d["wins"] += 1

        # Build RegimeTradeStats per regime
        per_regime: dict[MarketRegime, RegimeTradeStats] = {}
        for regime, d in regime_data.items():
            count = d["count"]
            per_regime[regime] = RegimeTradeStats(
                regime=regime,
                trade_count=count,
                total_pnl=d["pnl"],
                win_count=d["wins"],
                avg_pnl=d["pnl"] / count if count else 0.0,
                win_rate=d["wins"] / count if count else 0.0,
            )

        # Compute aggregate stats
        active = {r: s for r, s in per_regime.items() if s.trade_count >= 3}
        regime_coverage = len(active)

        active_avg_pnls = [s.avg_pnl for s in active.values()]
        worst_regime_pnl = min(active_avg_pnls) if active_avg_pnls else 0.0

        if len(active_avg_pnls) >= 2:
            regime_consistency = statistics.stdev(active_avg_pnls)
        else:
            regime_consistency = 0.0

        regime_bonus = _compute_regime_bonus(per_regime)

        analysis = RegimeAnalysis(
            per_regime=per_regime,
            regime_coverage=regime_coverage,
            worst_regime_pnl=worst_regime_pnl,
            regime_consistency=regime_consistency,
            regime_bonus=regime_bonus,
        )
        return key, analysis, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)


# ======================================================================
# Monte Carlo robustness testing
# ======================================================================


def _compute_percentiles(
    values: list[float],
) -> MonteCarloPercentiles:
    """Compute 5th/25th/50th/75th/95th percentiles from a list of values."""
    values.sort()
    n = len(values)
    if n == 0:
        return MonteCarloPercentiles(0.0, 0.0, 0.0, 0.0, 0.0)

    def _pctl(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    return MonteCarloPercentiles(
        p5=_pctl(0.05),
        p25=_pctl(0.25),
        p50=_pctl(0.50),
        p75=_pctl(0.75),
        p95=_pctl(0.95),
    )


def _equity_max_drawdown(equity: list[float]) -> float:
    """Compute max drawdown percentage from an equity curve."""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for eq in equity:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def run_monte_carlo(
    key: ComboKey,
    trade_pnls: list[float],
    initial_capital: float,
    n_simulations: int = 1000,
) -> tuple[ComboKey, MonteCarloResult | None, float, str | None]:
    """Run Monte Carlo robustness tests for one combo (process-safe).

    Two independent simulation types:

    1. **Trade shuffling** — randomly permute trade order, rebuild equity
       curve, measure drawdown and return distribution.
    2. **Bootstrap resampling** — sample trades WITH replacement, build
       equity curve, measure return and Sharpe distributions.

    Parameters
    ----------
    key:
        The strategy x timeframe combo.
    trade_pnls:
        List of float P&L values from closed trades.
    initial_capital:
        Starting capital for equity curve construction.
    n_simulations:
        Number of Monte Carlo iterations (default 1000).

    Returns
    -------
    tuple of (key, MonteCarloResult | None, elapsed, error)
    """
    t0 = time.time()

    if not trade_pnls or len(trade_pnls) < 2:
        return key, None, time.time() - t0, None

    try:
        rng = random.Random(42 + hash(key))
        n_trades = len(trade_pnls)

        # ---- Trade shuffling ----
        shuffle_dds: list[float] = []
        shuffle_finals: list[float] = []
        shuffle_returns: list[float] = []

        for _ in range(n_simulations):
            pnls = trade_pnls[:]
            rng.shuffle(pnls)

            equity = [initial_capital]
            for pnl in pnls:
                equity.append(equity[-1] + pnl)

            final = equity[-1]
            shuffle_dds.append(_equity_max_drawdown(equity))
            shuffle_finals.append(final)
            shuffle_returns.append(
                (final - initial_capital) / initial_capital * 100.0
            )

        # ---- Bootstrap resampling ----
        boot_returns: list[float] = []
        boot_sharpes: list[float] = []

        for _ in range(n_simulations):
            sampled = rng.choices(trade_pnls, k=n_trades)

            equity = [initial_capital]
            for pnl in sampled:
                equity.append(equity[-1] + pnl)

            final = equity[-1]
            ret = (final - initial_capital) / initial_capital * 100.0
            boot_returns.append(ret)

            # Per-trade returns for Sharpe approximation
            if len(sampled) >= 2:
                mean_pnl = sum(sampled) / len(sampled)
                std_pnl = (
                    sum((p - mean_pnl) ** 2 for p in sampled) / (len(sampled) - 1)
                ) ** 0.5
                if std_pnl > 0:
                    boot_sharpes.append(
                        mean_pnl / std_pnl * math.sqrt(252)
                    )
                else:
                    boot_sharpes.append(0.0)
            else:
                boot_sharpes.append(0.0)

        # ---- Compute percentiles ----
        dd_pctl = _compute_percentiles(shuffle_dds)
        final_pctl = _compute_percentiles(shuffle_finals)
        return_pctl = _compute_percentiles(shuffle_returns)
        boot_ret_pctl = _compute_percentiles(boot_returns)
        boot_sharpe_pctl = _compute_percentiles(boot_sharpes)

        # ---- Robustness assessment ----
        is_robust = boot_ret_pctl.p5 > 0.0

        penalty = 0.0
        if boot_ret_pctl.p5 < 0:
            penalty += abs(boot_ret_pctl.p5) * 2.0
        if dd_pctl.p95 > 20:
            penalty += (dd_pctl.p95 - 20) * 0.5

        mc_result = MonteCarloResult(
            n_simulations=n_simulations,
            n_trades=n_trades,
            max_drawdown=dd_pctl,
            final_equity=final_pctl,
            total_return=return_pctl,
            bootstrap_return=boot_ret_pctl,
            bootstrap_sharpe=boot_sharpe_pctl,
            is_robust=is_robust,
            mc_penalty=penalty,
        )
        return key, mc_result, time.time() - t0, None

    except Exception as e:
        return key, None, time.time() - t0, str(e)


# ======================================================================
# Cross-timeframe consistency
# ======================================================================


def apply_cross_timeframe_bonus(
    combos: list[ComboResult],
    min_profitable_tfs: int = 3,
) -> None:
    """Add a consistency bonus for strategies profitable across multiple TFs.

    Strategies that perform well in many timeframes are more likely to be
    genuinely robust rather than curve-fit to a single TF.  The bonus is
    larger when the score variance across timeframes is lower (more
    consistent).

    Parameters
    ----------
    combos:
        All combo results (mutated in-place).
    min_profitable_tfs:
        Minimum number of profitable timeframes to qualify for bonus.
    """
    # Group combos by strategy name
    by_strategy: dict[str, list[ComboResult]] = defaultdict(list)
    for c in combos:
        by_strategy[c.key.strategy_name].append(c)

    for _strat_name, group in by_strategy.items():
        # Count profitable timeframes
        profitable_tfs = [
            c for c in group
            if c.best_result is not None and c.best_result.total_pnl > 0
        ]

        if len(profitable_tfs) < min_profitable_tfs:
            continue

        # Compute score variance across profitable TFs
        scores = [c.tournament_score for c in profitable_tfs]
        if len(scores) < 2:
            continue

        variance = statistics.variance(scores)
        # Lower variance = more consistent = bigger bonus
        # Bonus capped at 5.0, decays with increasing variance
        bonus = 5.0 / (1.0 + variance / 100.0)

        for c in group:
            c.tournament_score += bonus
```

**Step 2: Update `tournament_engine.py` — replace analysis functions with imports**

In `tournament_engine.py`:
- Remove the 3 section headers and all 7 functions (Regime analysis, Monte Carlo, Cross-timeframe)
- Remove now-unused imports: `math`, `random`, `statistics`, `defaultdict` (from collections), `IntradayPriceData`, `MarketRegime`
- Keep: `time` (if bracket uses it — actually bracket doesn't use `time`, so remove it too)
- Add re-export:

```python
from stockdownloader.backtest.tournament_analysis import (
    _compute_percentiles,
    _compute_regime_bonus,
    _equity_max_drawdown,
    apply_cross_timeframe_bonus,
    classify_timeframe_bars,
    run_monte_carlo,
    run_regime_analysis,
)
```

Note: Private functions `_compute_percentiles`, `_compute_regime_bonus`, and `_equity_max_drawdown` must be re-exported because existing tests import them from `tournament_engine`.

**Step 3: Run existing tests**

Run: `python3 -m pytest tests/backtest/test_tournament_engine.py tests/backtest/test_tournament_regime.py tests/backtest/test_tournament_monte_carlo.py -v`
Expected: All PASS

**Step 4: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3209+)

**Step 5: Commit**

```bash
git add src/stockdownloader/backtest/tournament_analysis.py \
        src/stockdownloader/backtest/tournament_engine.py
git commit -m "refactor: extract tournament_analysis.py with regime/MC/cross-TF functions"
```

---

### Task 5: Slim `tournament_engine.py` — Verify Final State

**Files:**
- Verify: `src/stockdownloader/backtest/tournament_engine.py`

After Tasks 2–4, `tournament_engine.py` should contain ONLY:
1. Module docstring
2. Imports from the 3 new modules (re-exports)
3. `INITIAL_CAPITAL`, `RISK_PER_TRADE` re-exports from `util.config`
4. `_round_name()` function
5. `run_elimination_bracket()` function

**Step 1: Verify the final state of `tournament_engine.py`**

The file should look approximately like this (~90 lines):

```python
"""Multi-timeframe strategy tournament engine.

Core data models and process-safe worker functions for running
strategy × timeframe combinatorial backtests with walk-forward
optimization, validation, elimination brackets, and portfolio analysis.

Usage::

    from stockdownloader.backtest.tournament_engine import (
        ComboKey, ComboResult, run_combo_backtest, run_combo_optimize,
        run_combo_rebacktest, run_combo_walkforward,
        run_elimination_bracket, apply_cross_timeframe_bonus,
    )
"""
from __future__ import annotations

import math

from stockdownloader.util.config import INITIAL_CAPITAL, RISK_PER_TRADE

# Re-export models
from stockdownloader.backtest.tournament_models import (  # noqa: F401
    ComboKey,
    ComboResult,
    MatchResult,
    MonteCarloPercentiles,
    MonteCarloResult,
    RegimeAnalysis,
    RegimeTradeStats,
    TournamentResult,
)

# Re-export workers
from stockdownloader.backtest.tournament_workers import (  # noqa: F401
    run_combo_backtest,
    run_combo_optimize,
    run_combo_rebacktest,
    run_combo_walkforward,
)

# Re-export analysis (including private functions used by tests)
from stockdownloader.backtest.tournament_analysis import (  # noqa: F401
    _compute_percentiles,
    _compute_regime_bonus,
    _equity_max_drawdown,
    apply_cross_timeframe_bonus,
    classify_timeframe_bars,
    run_monte_carlo,
    run_regime_analysis,
)


# ======================================================================
# Elimination bracket
# ======================================================================


def _round_name(round_num: int, total_rounds: int) -> str:
    """Human-readable round name."""
    remaining = total_rounds - round_num
    if remaining == 1:
        return "FINAL"
    if remaining == 2:
        return "SEMIFINAL"
    if remaining == 3:
        return "QUARTERFINAL"
    return f"Round {round_num + 1}"


def run_elimination_bracket(
    combos: list[ComboResult],
    bracket_size: int = 16,
) -> tuple[list[MatchResult], ComboKey | None]:
    """Run single-elimination bracket on top combos.

    Parameters
    ----------
    combos:
        All combo results, sorted by tournament_score descending.
    bracket_size:
        Number of combos to include (must be power of 2).

    Returns
    -------
    (matches, champion_key)
    """
    # Ensure bracket_size is a power of 2
    if not combos:
        return [], None

    actual = min(bracket_size, len(combos))
    if actual < 2:
        return [], combos[0].key

    bracket_exp = max(1, int(math.log2(actual)))
    actual = 2 ** bracket_exp

    # Seed: #1 vs #N, #2 vs #N-1, etc.
    seeded = combos[:actual]
    current_round: list[ComboResult] = []
    for i in range(actual // 2):
        current_round.append(seeded[i])
        current_round.append(seeded[actual - 1 - i])

    matches: list[MatchResult] = []
    total_rounds = bracket_exp
    round_num = 0

    while len(current_round) > 1:
        next_round: list[ComboResult] = []
        rname = _round_name(round_num, total_rounds)

        for i in range(0, len(current_round), 2):
            a = current_round[i]
            b = current_round[i + 1]

            if a.tournament_score >= b.tournament_score:
                winner, loser = a, b
            else:
                winner, loser = b, a

            matches.append(MatchResult(
                winner=winner.key,
                loser=loser.key,
                winner_score=winner.tournament_score,
                loser_score=loser.tournament_score,
                round_num=round_num,
                round_name=rname,
            ))
            next_round.append(winner)

        current_round = next_round
        round_num += 1

    champion = current_round[0].key if current_round else None
    return matches, champion
```

**Step 2: Verify line count**

Run: `wc -l src/stockdownloader/backtest/tournament_engine.py`
Expected: ~90-120 lines (was 932)

**Step 3: Run ALL tests one final time**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3209+)

**Step 4: Verify line counts for all new files**

Run:
```bash
wc -l src/stockdownloader/backtest/tournament_models.py \
      src/stockdownloader/backtest/tournament_workers.py \
      src/stockdownloader/backtest/tournament_analysis.py \
      src/stockdownloader/backtest/tournament_engine.py \
      src/stockdownloader/backtest/report_helpers.py
```

Expected approximate:
- `tournament_models.py`: ~155 lines
- `tournament_workers.py`: ~260 lines
- `tournament_analysis.py`: ~380 lines
- `tournament_engine.py`: ~115 lines (was 932)
- `report_helpers.py`: ~20 lines

---

### Task 6: Update `backtest/__init__.py` — Complete Exports

**Files:**
- Modify: `src/stockdownloader/backtest/__init__.py`

**Step 1: Update `__init__.py` with all missing exports**

Replace the entire contents of `src/stockdownloader/backtest/__init__.py`:

```python
"""Backtesting engines and reporting."""

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_result import (
    BaseBacktestResult,
    BacktestResult,
    OptionsBacktestResult,
)
from stockdownloader.backtest import report_formatter
from stockdownloader.backtest.options_backtest_engine import OptionsBacktestEngine
from stockdownloader.backtest.exit_tournament_engine import ExitTournamentEngine
from stockdownloader.backtest.exit_tournament_result import ExitTournamentResult
from stockdownloader.backtest import exit_tournament_report_formatter
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_base import OptimizerBase
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.strategy_optimizer import StrategyOptimizer
from stockdownloader.backtest.daily_strategy_optimizer import DailyStrategyOptimizer
from stockdownloader.backtest.walk_forward import WalkForwardValidator, WalkForwardResult
from stockdownloader.backtest.walk_forward_optimizer import WalkForwardOptimizer, WFOptResult
from stockdownloader.backtest.combinatorial_tester import CombinatorialTester, CombinatorialConfig
from stockdownloader.backtest.tournament_models import (
    ComboKey,
    ComboResult,
    MatchResult,
    TournamentResult,
    RegimeTradeStats,
    RegimeAnalysis,
    MonteCarloPercentiles,
    MonteCarloResult,
)
from stockdownloader.backtest import tournament_engine

__all__ = [
    "BacktestEngine",
    "BaseBacktestResult",
    "BacktestResult",
    "report_formatter",
    "OptionsBacktestEngine",
    "OptionsBacktestResult",
    "ExitTournamentEngine",
    "ExitTournamentResult",
    "exit_tournament_report_formatter",
    "IntradayBacktestEngine",
    "OptimizerBase",
    "score_v2",
    "StrategyOptimizer",
    "DailyStrategyOptimizer",
    "WalkForwardValidator",
    "WalkForwardResult",
    "WalkForwardOptimizer",
    "WFOptResult",
    "CombinatorialTester",
    "CombinatorialConfig",
    "ComboKey",
    "ComboResult",
    "MatchResult",
    "TournamentResult",
    "RegimeTradeStats",
    "RegimeAnalysis",
    "MonteCarloPercentiles",
    "MonteCarloResult",
    "tournament_engine",
]
```

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3209+)

**Step 3: Commit**

```bash
git add src/stockdownloader/backtest/__init__.py
git commit -m "refactor: complete backtest/__init__.py with all missing exports"
```
