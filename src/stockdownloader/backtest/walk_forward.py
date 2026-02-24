"""Walk-forward validation framework for anti-overfitting analysis.

Splits data into rolling in-sample (IS) and out-of-sample (OOS) windows,
runs strategies on each window, and computes degradation ratios to detect
overfitting.

Usage::

    from stockdownloader.backtest.walk_forward import (
        WalkForwardValidator, WalkForwardResult,
    )

    validator = WalkForwardValidator(data, n_windows=5)
    result = validator.validate(strategy_factory, engine)
    print(f"Degradation ratio: {result.degradation_ratio:.2f}")
    # ratio ~1.0 = robust, ratio < 0.5 = likely overfit
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.backtest.optimizer_scoring import score_v2

if TYPE_CHECKING:
    from stockdownloader.backtest.backtest_result import BacktestResult
    from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategy.trading_strategy import IntradayTradingStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """Defines one IS/OOS split within the walk-forward framework.

    Attributes
    ----------
    in_sample_start:
        First bar index (inclusive) of the in-sample region.
    in_sample_end:
        Last bar index (exclusive) of the in-sample region.
    out_of_sample_start:
        First bar index (inclusive) of the OOS region.
    out_of_sample_end:
        Last bar index (exclusive) of the OOS region.
    window_id:
        Sequential identifier (0-based).
    """

    in_sample_start: int
    in_sample_end: int
    out_of_sample_start: int
    out_of_sample_end: int
    window_id: int


@dataclass(slots=True)
class WalkForwardResult:
    """Aggregated results from walk-forward validation.

    Attributes
    ----------
    strategy_name:
        Name of the strategy tested.
    windows:
        The window definitions used.
    in_sample_results:
        Backtest result for each IS window.
    out_of_sample_results:
        Backtest result for each OOS window.
    in_sample_score:
        Average ``score_v2`` across IS windows.
    out_of_sample_score:
        Average ``score_v2`` across OOS windows.
    degradation_ratio:
        ``OOS / IS`` score ratio.  Closer to 1.0 means robust (not overfit).
        Values < 0.5 suggest significant overfitting.
    """

    strategy_name: str
    windows: list[WalkForwardWindow]
    in_sample_results: list[BacktestResult]
    out_of_sample_results: list[BacktestResult]
    in_sample_score: float
    out_of_sample_score: float
    degradation_ratio: float


class WalkForwardValidator:
    """Rolling walk-forward validation framework.

    Splits data into *n_windows* rolling IS/OOS pairs and evaluates
    a strategy factory across all of them.

    Parameters
    ----------
    data:
        Full 5-minute bar dataset.
    n_windows:
        Number of walk-forward windows (default 5).
    is_ratio:
        Fraction of each window's total span devoted to in-sample
        (default 0.7 = 70/30 split).
    """

    def __init__(
        self,
        data: list[IntradayPriceData],
        n_windows: int = 5,
        is_ratio: float = 0.7,
    ) -> None:
        if not data:
            raise ValueError("data must not be empty")
        if n_windows < 1:
            raise ValueError("n_windows must be >= 1")
        if not 0.1 <= is_ratio <= 0.95:
            raise ValueError("is_ratio must be between 0.1 and 0.95")

        self._data = data
        self._n_windows = n_windows
        self._is_ratio = is_ratio
        self._windows = self._build_windows()

    @property
    def windows(self) -> list[WalkForwardWindow]:
        """The computed walk-forward windows."""
        return list(self._windows)

    def _build_windows(self) -> list[WalkForwardWindow]:
        """Construct rolling IS/OOS windows.

        Strategy: divide total data into *n_windows + 1* equal segments.
        Each window uses *is_ratio* fraction for IS and the rest for OOS.
        Windows overlap in their IS portions, rolling forward.
        """
        total_bars = len(self._data)
        n = self._n_windows

        # Each window spans (total / n) bars of forward movement
        # But the full window (IS + OOS) is larger
        # IS fraction determines how much is training vs testing

        # Total window span = total_bars / n_windows * (1 / (1 - is_ratio))
        # but capped at total_bars
        oos_frac = 1.0 - self._is_ratio
        step = int(total_bars * oos_frac / n)
        if step < 1:
            step = 1

        # Window size: IS + OOS
        window_size = int(step / oos_frac)
        is_size = int(window_size * self._is_ratio)
        oos_size = window_size - is_size

        if oos_size < 1:
            oos_size = 1
        if is_size < 1:
            is_size = 1

        windows: list[WalkForwardWindow] = []
        for i in range(n):
            is_start = i * step
            is_end = is_start + is_size
            oos_start = is_end
            oos_end = oos_start + oos_size

            # Don't create windows where OOS would start beyond data
            if oos_start >= total_bars:
                break

            # Clamp IS end to data bounds so the IS slice doesn't
            # reference out-of-range indices.
            if is_end > total_bars:
                is_end = total_bars
                # If IS was clamped, OOS starts at the clamp boundary
                oos_start = is_end

            # Skip if OOS region is now invalid after clamping
            if oos_start >= total_bars:
                break

            if oos_end > total_bars:
                oos_end = total_bars

            windows.append(WalkForwardWindow(
                in_sample_start=is_start,
                in_sample_end=is_end,
                out_of_sample_start=oos_start,
                out_of_sample_end=oos_end,
                window_id=i,
            ))

        return windows

    def validate(
        self,
        strategy_factory: Callable[[], IntradayTradingStrategy],
        engine: IntradayBacktestEngine,
        strategy_name: str = "unnamed",
    ) -> WalkForwardResult:
        """Run walk-forward validation across all windows.

        Parameters
        ----------
        strategy_factory:
            Callable that creates a fresh strategy instance (needed
            because strategies maintain internal state).
        engine:
            The backtest engine to use.
        strategy_name:
            Label for reporting.

        Returns
        -------
        :class:`WalkForwardResult` with IS/OOS scores and degradation ratio.
        """
        is_results: list[BacktestResult] = []
        oos_results: list[BacktestResult] = []
        # Track which windows succeeded (both IS and OOS) so we can
        # pair results with the correct window for scoring.
        succeeded_windows: list[WalkForwardWindow] = []

        for w in self._windows:
            # Get IS and OOS data slices
            is_data = self._data[w.in_sample_start:w.in_sample_end]
            oos_data = self._data[w.out_of_sample_start:w.out_of_sample_end]

            if not is_data or not oos_data:
                logger.warning(
                    "Window %d has empty data (IS=%d, OOS=%d), skipping",
                    w.window_id, len(is_data), len(oos_data),
                )
                continue

            # Run IS backtest
            try:
                is_strategy = strategy_factory()
                is_result = engine.run(is_strategy, is_data)
            except (ValueError, ZeroDivisionError, ArithmeticError) as e:
                logger.warning("IS backtest failed for window %d: %s", w.window_id, e)
                continue

            # Run OOS backtest
            try:
                oos_strategy = strategy_factory()
                oos_result = engine.run(oos_strategy, oos_data)
            except (ValueError, ZeroDivisionError, ArithmeticError) as e:
                logger.warning("OOS backtest failed for window %d: %s", w.window_id, e)
                continue

            # Both succeeded — record together to keep lists aligned
            is_results.append(is_result)
            oos_results.append(oos_result)
            succeeded_windows.append(w)

        # Compute average scores — use each window's own trading days,
        # not the full dataset's, so the trades-per-day bonus is accurate
        # for each window's time span.
        is_scores: list[float] = []
        oos_scores: list[float] = []
        for w, is_r, oos_r in zip(succeeded_windows, is_results, oos_results, strict=True):
            is_data_slice = self._data[w.in_sample_start:w.in_sample_end]
            oos_data_slice = self._data[w.out_of_sample_start:w.out_of_sample_end]
            is_scores.append(score_v2(is_r, self._unique_days(is_data_slice)))
            oos_scores.append(score_v2(oos_r, self._unique_days(oos_data_slice)))

        avg_is = sum(is_scores) / len(is_scores) if is_scores else 0.0
        avg_oos = sum(oos_scores) / len(oos_scores) if oos_scores else 0.0

        # Degradation ratio: OOS/IS (handle IS=0 or negative)
        if avg_is > 0:
            degradation = avg_oos / avg_is
        elif avg_is == 0:
            degradation = 0.0
        else:
            # Both negative: ratio > 1 means OOS is "less bad"
            degradation = avg_oos / avg_is if avg_is != 0 else 0.0

        return WalkForwardResult(
            strategy_name=strategy_name,
            windows=self._windows,
            in_sample_results=is_results,
            out_of_sample_results=oos_results,
            in_sample_score=avg_is,
            out_of_sample_score=avg_oos,
            degradation_ratio=degradation,
        )

    @staticmethod
    def _unique_days(data: list[IntradayPriceData]) -> int:
        """Count unique trading days in data."""
        return len({d.date[:10] for d in data})
