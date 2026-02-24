"""Data model for the strategy pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stockdownloader.backtesting.results.result import (
    BacktestResult,
    BaseBacktestResult,
)
from stockdownloader.backtesting.optimization.scoring import score_v2
from stockdownloader.backtesting.optimization.walk_forward import WalkForwardResult


@dataclass
class SlotResult:
    """One strategy's results across all pipeline stages."""

    name: str
    display_name: str
    category: str
    baseline: BaseBacktestResult | None = None
    optimized_kwargs: dict[str, Any] | None = None
    optimized: BaseBacktestResult | None = None
    wf_result: WalkForwardResult | None = None

    @property
    def best_result(self) -> BaseBacktestResult | None:
        """Return optimized result if available, else baseline."""
        return self.optimized or self.baseline

    def ranking_score(self, trading_days: int) -> float:
        """Unified score for cross-category ranking.

        Combines score_v2 on the best result with walk-forward
        degradation as a multiplier.  This prevents negative-return
        strategies from ranking above genuinely profitable ones just
        because they have a high degradation ratio.
        """
        result = self.best_result
        if result is None:
            return -999.0

        if isinstance(result, BacktestResult):
            base = score_v2(result, trading_days)
        else:
            # OptionsBacktestResult — use basic P&L-based ranking
            base = float(result.total_return)

        # Apply walk-forward quality factor: ROBUST strategies
        # keep their full score; OVERFIT ones get penalised.
        if self.wf_result is not None:
            deg = self.wf_result.degradation_ratio
            if deg >= 0.8:
                factor = 1.0  # ROBUST — no change
            elif deg >= 0.5:
                factor = 0.8  # ACCEPTABLE — mild penalty
            else:
                factor = 0.5  # OVERFIT — significant penalty

            # For positive scores, multiply down by factor.
            # For negative scores, multiply the *absolute* value up
            # so overfit negative strategies rank lower.
            if base >= 0:
                return base * factor
            else:
                return base / factor  # makes negative more negative
        return base
