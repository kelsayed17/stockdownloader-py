"""Data models for the tournament engine.

Frozen / slotted dataclasses used across the multi-timeframe tournament
pipeline: regime analysis, Monte Carlo robustness, bracket elimination,
and portfolio construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stockdownloader.backtest.backtest_result import BacktestResult
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
    """Unique identifier for a strategy x timeframe combination."""

    strategy_name: str
    timeframe: str  # "5m", "15m", "30m", "1h", "4h", "1d"
    data_source: str = "resampled"  # CSV filename or "resampled"

    @property
    def label(self) -> str:
        return f"{self.strategy_name} @ {self.timeframe}"


@dataclass(slots=True)
class ComboResult:
    """Result of running one strategy x timeframe combination."""

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
