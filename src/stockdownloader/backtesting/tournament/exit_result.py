"""Holds exit tournament results and provides aggregate analysis.

Collects :class:`ExitMechanismTradeResult` objects for every mechanism / trade
combination and provides aggregate summaries, breakdowns, and head-to-head
comparisons.
"""
from __future__ import annotations

from decimal import Decimal
from operator import attrgetter, itemgetter

from stockdownloader.core.models.exit_result import (
    ExitMechanismSummary,
    ExitMechanismTradeResult,
)
from stockdownloader.core.models.trade import Direction


class ExitTournamentResult:
    """Aggregated results from running trades through multiple exit mechanisms."""

    def __init__(self) -> None:
        self._all_results: list[ExitMechanismTradeResult] = []
        self._summaries: dict[str, ExitMechanismSummary] = {}
        self._trades_simulated: int = 0
        self._trades_skipped: int = 0

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_result(self, result: ExitMechanismTradeResult) -> None:
        """Record a single trade-mechanism result."""
        self._all_results.append(result)
        if result.mechanism_name not in self._summaries:
            self._summaries[result.mechanism_name] = ExitMechanismSummary(
                result.mechanism_name
            )
        self._summaries[result.mechanism_name].add_result(result)

    @property
    def trades_simulated(self) -> int:
        return self._trades_simulated

    @trades_simulated.setter
    def trades_simulated(self, value: int) -> None:
        self._trades_simulated = value

    @property
    def trades_skipped(self) -> int:
        return self._trades_skipped

    @trades_skipped.setter
    def trades_skipped(self, value: int) -> None:
        self._trades_skipped = value

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def mechanism_names(self) -> list[str]:
        return list(self._summaries.keys())

    @property
    def mechanism_summaries(self) -> dict[str, ExitMechanismSummary]:
        return dict(self._summaries)

    def get_summary(self, mechanism_name: str) -> ExitMechanismSummary | None:
        return self._summaries.get(mechanism_name)

    @property
    def all_results(self) -> list[ExitMechanismTradeResult]:
        return list(self._all_results)

    def get_best_mechanism(self, metric: str = "total_pnl") -> str:
        """Return the mechanism name with the best value for *metric*.

        Supported metrics: ``total_pnl``, ``win_rate``, ``profit_factor``,
        ``sharpe_approx``, ``avg_capture_pct``.
        """
        best_name = ""
        best_val = Decimal("-999999")
        for name, summary in self._summaries.items():
            val = getattr(summary, metric, Decimal("0"))
            if val > best_val:
                best_val = val
                best_name = name
        return best_name

    # ------------------------------------------------------------------
    # Breakdown helpers
    # ------------------------------------------------------------------

    def get_results_for_mechanism(
        self, mechanism_name: str
    ) -> list[ExitMechanismTradeResult]:
        """Return all trade results for a single mechanism."""
        return [r for r in self._all_results if r.mechanism_name == mechanism_name]

    def get_breakdown_by_direction(
        self, mechanism_name: str
    ) -> dict[str, ExitMechanismSummary]:
        """Breakdown results for *mechanism_name* by trade direction."""
        results = self.get_results_for_mechanism(mechanism_name)
        summaries: dict[str, ExitMechanismSummary] = {}
        for r in results:
            key = r.direction.value
            if key not in summaries:
                summaries[key] = ExitMechanismSummary(f"{mechanism_name}_{key}")
            summaries[key].add_result(r)
        return summaries

    def get_breakdown_by_signal_type(
        self, mechanism_name: str
    ) -> dict[str, ExitMechanismSummary]:
        """Breakdown results for *mechanism_name* by signal type."""
        results = self.get_results_for_mechanism(mechanism_name)
        summaries: dict[str, ExitMechanismSummary] = {}
        for r in results:
            key = r.signal_type
            if key not in summaries:
                summaries[key] = ExitMechanismSummary(f"{mechanism_name}_{key}")
            summaries[key].add_result(r)
        return summaries

    @property
    def head_to_head(self) -> dict[str, int]:
        """For each trade, which mechanism produced the best P&L?

        Returns a dict mapping mechanism name to the number of trades it
        won (i.e. produced the highest P&L).
        """
        # Group results by trade_id
        trades: dict[int, list[ExitMechanismTradeResult]] = {}
        for r in self._all_results:
            trades.setdefault(r.trade_id, []).append(r)

        counts: dict[str, int] = {name: 0 for name in self._summaries}
        for trade_id, results in trades.items():
            best = max(results, key=attrgetter("pnl"))
            counts[best.mechanism_name] = counts.get(best.mechanism_name, 0) + 1

        return dict(sorted(counts.items(), key=itemgetter(1), reverse=True))
