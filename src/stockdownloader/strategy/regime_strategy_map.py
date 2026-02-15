"""Regime-to-strategy performance mapping.

Records per-trade performance tagged by the market regime at entry,
then aggregates to determine which strategy works best in each regime.

Usage::

    from stockdownloader.strategy.regime_strategy_map import (
        RegimeStrategyMapper,
    )

    mapper = RegimeStrategyMapper()
    mapper.record("RSI", MarketRegime.MEAN_REVERTING, pnl=150.0)
    mapper.record("SMA", MarketRegime.STRONG_TREND_UP, pnl=300.0)
    best = mapper.best_strategy_for_regime(MarketRegime.MEAN_REVERTING)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from operator import attrgetter

from stockdownloader.strategy.regime_detector import MarketRegime


@dataclass(slots=True)
class RegimePerformance:
    """Aggregated performance of a strategy within a specific regime.

    Attributes
    ----------
    regime:
        The market regime.
    strategy_name:
        The strategy being evaluated.
    total_pnl:
        Sum of per-trade P&L in this regime.
    trade_count:
        Number of trades in this regime.
    wins:
        Number of winning trades in this regime.
    """

    regime: MarketRegime
    strategy_name: str
    total_pnl: float = 0.0
    trade_count: int = 0
    wins: int = 0

    @property
    def avg_pnl(self) -> float:
        """Average P&L per trade."""
        return self.total_pnl / self.trade_count if self.trade_count else 0.0

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction [0, 1]."""
        return self.wins / self.trade_count if self.trade_count else 0.0


class RegimeStrategyMapper:
    """Maps (strategy, regime) pairs to their historical performance.

    Built incrementally by calling :meth:`record` for each trade with its
    entry-bar regime.  Then query :meth:`best_strategy_for_regime` to find
    the highest-performing strategy for any given regime.
    """

    def __init__(self) -> None:
        self._perfs: dict[
            tuple[str, MarketRegime], RegimePerformance
        ] = {}

    def record(
        self,
        strategy_name: str,
        regime: MarketRegime,
        pnl: float,
        is_win: bool | None = None,
    ) -> None:
        """Record a single trade's outcome in a specific regime.

        Parameters
        ----------
        strategy_name:
            Name of the strategy that generated the trade.
        regime:
            Market regime at the time of trade entry.
        pnl:
            Dollar P&L of the trade.
        is_win:
            Whether the trade was profitable.  If ``None``, inferred
            from ``pnl > 0``.
        """
        key = (strategy_name, regime)
        if key not in self._perfs:
            self._perfs[key] = RegimePerformance(
                regime=regime,
                strategy_name=strategy_name,
            )

        perf = self._perfs[key]
        perf.total_pnl += pnl
        perf.trade_count += 1
        if is_win is None:
            is_win = pnl > 0
        if is_win:
            perf.wins += 1

    def best_strategy_for_regime(
        self,
        regime: MarketRegime,
        min_trades: int = 3,
    ) -> str | None:
        """Return the strategy with the highest avg P&L in *regime*.

        Parameters
        ----------
        regime:
            The target regime.
        min_trades:
            Minimum number of trades required to be considered.

        Returns ``None`` if no strategy has enough trades in this regime.
        """
        candidates = [
            perf for (_, r), perf in self._perfs.items()
            if r == regime and perf.trade_count >= min_trades
        ]
        if not candidates:
            return None

        best = max(candidates, key=attrgetter("avg_pnl"))
        return best.strategy_name

    def get_performance(
        self,
        strategy_name: str,
        regime: MarketRegime,
    ) -> RegimePerformance | None:
        """Get a specific strategy's performance in a specific regime."""
        return self._perfs.get((strategy_name, regime))

    def all_performances(self) -> list[RegimePerformance]:
        """Get all recorded performances."""
        return list(self._perfs.values())

    def strategy_names(self) -> list[str]:
        """Get unique strategy names that have been recorded."""
        return sorted({name for name, _ in self._perfs.keys()})

    def regime_coverage(self, strategy_name: str) -> dict[MarketRegime, int]:
        """Get trade counts per regime for a given strategy."""
        coverage: dict[MarketRegime, int] = {}
        for (name, regime), perf in self._perfs.items():
            if name == strategy_name:
                coverage[regime] = perf.trade_count
        return coverage

    @property
    def performance_matrix(self) -> dict[str, dict[str, float]]:
        """Get a nested dict of strategy -> regime -> avg_pnl.

        Returns a dict keyed by strategy name, each containing
        a dict of regime.value -> avg_pnl.
        """
        matrix: dict[str, dict[str, float]] = defaultdict(dict)
        for (name, regime), perf in self._perfs.items():
            matrix[name][regime.value] = perf.avg_pnl
        return dict(matrix)
