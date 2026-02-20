"""Adaptive ensemble meta-strategy with regime-aware strategy selection.

Selects the best sub-strategy based on the current market regime,
with optional drawdown-based position scaling.

Usage::

    from stockdownloader.strategy.regime import (
        EnsembleIntradayStrategy,
        DrawdownPositionScaler,
    )

    ensemble = EnsembleIntradayStrategy(
        sub_strategies={
            MarketRegime.MEAN_REVERTING: rsi_strategy,
            MarketRegime.STRONG_TREND_UP: sma_strategy,
        },
        regime_detector=detector,
        default_strategy=rsi_strategy,
    )
    signal = ensemble.evaluate(data, index)
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.intraday_signal import HOLD, IntradaySignal
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.strategy.regime.regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
)
from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from stockdownloader.model.price_data import IntradayPriceData

class DrawdownPositionScaler:
    """Scale position size based on recent equity drawdown.

    Full size at 0% drawdown, half size at ``half_dd`` drawdown,
    zero size at ``zero_dd`` drawdown.  Linear interpolation between
    thresholds.

    Parameters
    ----------
    full_dd:
        Drawdown percentage at which full position is taken (default 0.0).
    half_dd:
        Drawdown percentage at which position is halved (default 5.0).
    zero_dd:
        Drawdown percentage at which position is zero (default 10.0).
    """

    def __init__(
        self,
        full_dd: float = 0.0,
        half_dd: float = 5.0,
        zero_dd: float = 10.0,
    ) -> None:
        if not full_dd <= half_dd <= zero_dd:
            raise ValueError(
                f"Thresholds must be full_dd <= half_dd <= zero_dd, "
                f"got {full_dd}, {half_dd}, {zero_dd}"
            )
        self._full_dd = full_dd
        self._half_dd = half_dd
        self._zero_dd = zero_dd

    def scale_factor(self, current_dd_pct: float) -> float:
        """Compute position scale factor for the given drawdown.

        Parameters
        ----------
        current_dd_pct:
            Current drawdown as a positive percentage (e.g. 3.0 = 3%).

        Returns
        -------
        Scale factor [0.0, 1.0].
        """
        if current_dd_pct <= self._full_dd:
            return 1.0
        if current_dd_pct >= self._zero_dd:
            return 0.0
        if current_dd_pct <= self._half_dd:
            # Linear from 1.0 to 0.5
            denom = self._half_dd - self._full_dd
            if denom == 0.0:
                return 0.5  # at boundary — return midpoint
            frac = (current_dd_pct - self._full_dd) / denom
            return 1.0 - 0.5 * frac
        # Linear from 0.5 to 0.0
        denom = self._zero_dd - self._half_dd
        if denom == 0.0:
            return 0.0  # at boundary — return lower bound
        frac = (current_dd_pct - self._half_dd) / denom
        return 0.5 * (1.0 - frac)

class EnsembleIntradayStrategy(IntradayTradingStrategy):
    """Meta-strategy that adapts sub-strategy based on detected regime.

    1. Classifies current market regime using :class:`MarketRegimeDetector`.
    2. Selects the best sub-strategy for that regime from the mapping.
    3. Delegates signal generation to the selected sub-strategy.
    4. Optionally scales risk_per_share via :class:`DrawdownPositionScaler`.

    Parameters
    ----------
    sub_strategies:
        Mapping of regime → strategy to use in that regime.
    regime_detector:
        Detector instance for classifying market conditions.
    default_strategy:
        Fallback strategy when no mapping exists for the current regime.
    drawdown_scaler:
        Optional position scaler based on equity drawdown.
    name:
        Display name for reporting.
    """

    def __init__(
        self,
        sub_strategies: dict[MarketRegime, IntradayTradingStrategy],
        regime_detector: MarketRegimeDetector,
        default_strategy: IntradayTradingStrategy,
        drawdown_scaler: DrawdownPositionScaler | None = None,
        name: str = "Ensemble",
    ) -> None:
        self._subs = sub_strategies
        self._detector = regime_detector
        self._default = default_strategy
        self._scaler = drawdown_scaler
        self._name = name
        self._current_session: str | None = None
        self._peak_equity: Decimal = ZERO
        self._current_equity: Decimal = ZERO
        self._current_dd_pct: float = 0.0
        # Track which sub-strategy has the active position for callback routing
        self._active_strategy: IntradayTradingStrategy | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def warmup_period(self) -> int:
        """Max warmup across all sub-strategies + regime detector."""
        sub_warmups = [
            s.warmup_period for s in self._subs.values()
        ]
        sub_warmups.append(self._default.warmup_period)
        return max(sub_warmups + [self._detector.warmup_period])

    def on_session_start(self, trading_date: str) -> None:
        """Propagate session start to all sub-strategies."""
        for s in self._subs.values():
            s.on_session_start(trading_date)
        self._default.on_session_start(trading_date)
        self._current_session = trading_date
        self._active_strategy = None

    # ------------------------------------------------------------------
    # Engine callbacks — route to the sub-strategy that owns the position
    # ------------------------------------------------------------------

    def on_position_opened(self, is_long: bool) -> None:
        """Propagate to the sub-strategy that generated the entry signal."""
        if self._active_strategy is not None:
            self._active_strategy.on_position_opened(is_long)

    def on_position_closed(self) -> None:
        """Propagate to the sub-strategy that owns the position."""
        if self._active_strategy is not None:
            self._active_strategy.on_position_closed()
            self._active_strategy = None

    def update_equity(self, equity: Decimal) -> None:
        """Update equity tracking for drawdown scaling.

        Called externally (by the tournament runner or backtest harness)
        to track current equity for position sizing.
        """
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > ZERO:
            self._current_dd_pct = float(
                (self._peak_equity - equity) / self._peak_equity * 100
            )
        else:
            self._current_dd_pct = 0.0

    def evaluate(
        self,
        data: list[IntradayPriceData],
        current_index: int,
    ) -> IntradaySignal:
        """Evaluate the ensemble at *current_index*.

        1. Detect session boundary and reset if needed.
        2. Classify market regime.
        3. Select appropriate sub-strategy.
        4. Delegate signal generation.
        5. Apply drawdown scaling to risk_per_share if applicable.
        """
        # Session boundary detection
        if current_index < len(data):
            today = data[current_index].date[:10]
            if self._current_session is None or today != self._current_session:
                self.on_session_start(today)

        # Check warmup
        if current_index < self.warmup_period:
            return HOLD

        # Classify regime
        rc = self._detector.classify(data, current_index)

        # Select sub-strategy
        strategy = self._subs.get(rc.regime, self._default)

        # Get signal from selected strategy
        signal = strategy.evaluate(data, current_index)

        # Track which sub-strategy owns the entry for callback routing
        if signal.action.value.startswith("ENTER"):
            self._active_strategy = strategy

        # Apply drawdown scaling if configured
        if (
            self._scaler is not None
            and signal.action.value.startswith("ENTER")
            and signal.risk_per_share > ZERO
        ):
            scale = self._scaler.scale_factor(self._current_dd_pct)
            if scale <= 0.0:
                return HOLD  # Don't enter during heavy drawdown
            if scale < 1.0:
                # Scale down the risk_per_share
                scaled_risk = Decimal(str(
                    float(signal.risk_per_share) * scale
                )).quantize(Decimal("0.01"))
                signal = IntradaySignal(
                    action=signal.action,
                    mode=signal.mode,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confluence_score=signal.confluence_score,
                    max_score=signal.max_score,
                    risk_per_share=scaled_risk,
                    reason=f"[{rc.regime.value}] {signal.reason}",
                )
            else:
                # Full size, just annotate with regime
                signal = IntradaySignal(
                    action=signal.action,
                    mode=signal.mode,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confluence_score=signal.confluence_score,
                    max_score=signal.max_score,
                    risk_per_share=signal.risk_per_share,
                    reason=f"[{rc.regime.value}] {signal.reason}",
                )

        return signal
