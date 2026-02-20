"""ML-driven mean-reversion strategy — model decides entries.

No hardcoded indicator thresholds.  A trained gradient boosting model
evaluates 63 normalised features (RSI, MFI, BB%B, Williams %R, CCI,
Stochastic, ADX, volume ratios, regime signals, cross-feature
interactions) and returns ``P(profitable)``.  If the probability
exceeds the configured threshold the strategy enters long.

SL/TP are ATR-based risk management, the only non-ML component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.ml.predictor import MLPredictor
from stockdownloader.model.intraday_signal import IntradaySignal
from stockdownloader.strategy.intraday.base_config import InfraExitConfig
from stockdownloader.strategy.intraday.entry_helpers import (
    clamp_sl_dist,
    directional_sl_tp,
    make_entry_signal,
)
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.bar_context import BarContext
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.trail_strategy import BreakevenTrail
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.pinescript_models import ModeDefinition
from stockdownloader.util.pinescript_modes import ml_oversold_mode

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData


@dataclass(frozen=True, slots=True)
class MLOversoldConfig(InfraExitConfig):
    """Config for :class:`MLOversoldStrategy`.

    The ML model decides *when* to trade; these fields control
    *how much* to risk and basic session limits.
    """

    # ── ML model ──────────────────────────────────────────────────────
    model_path: str = "output/models/spy/gradient_boosting_latest.joblib"

    # Probability threshold — model must be at least this confident.
    ml_threshold: Decimal = Decimal("0.60")

    # ── SL / TP (standard risk management) ────────────────────────────
    ml_sl_atr: Decimal = Decimal("1.5")
    ml_sl_cap: Decimal = Decimal("2.50")
    ml_tp_mode: str = "vwap"          # "vwap" or "rr"
    ml_rr: Decimal = Decimal("1.5")   # R:R multiplier (rr mode)
    ml_min_rr: Decimal = Decimal("0.5")

    # ── Session limits ────────────────────────────────────────────────
    # Long-only: SPY mean-reversion bias.
    allow_shorts: bool = False
    max_day: int = 2
    spacing: int = 8

logger = logging.getLogger(__name__)


class MLOversoldStrategy(BaseIntradayStrategy):
    """ML-driven mean-reversion strategy.

    The trained model does ALL the decision-making.  The strategy
    is a thin wrapper that:

    1. Captures ``data`` / ``current_index`` in :meth:`evaluate`.
    2. Runs ``MLPredictor.predict_proba`` inside :meth:`_evaluate_entry`.
    3. Applies SL/TP risk management if the model is confident.
    """

    def __init__(
        self,
        config: MLOversoldConfig | None = None,
        model_path: str | None = None,
    ) -> None:
        c = config or MLOversoldConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(BreakevenTrail()))

        # Load ML model via MLPredictor (gracefully handle missing model)
        path = model_path or c.model_path
        try:
            self._predictor = MLPredictor.from_path(path, hub=self._infra.hub)
        except FileNotFoundError:
            logger.warning(
                "ML model not found at %s — strategy will not generate signals. "
                "Run `ml-train` CLI command first.",
                path,
            )
            self._predictor = None  # type: ignore[assignment]

        # Set during evaluate() so _evaluate_entry can access data.
        self._data: list[IntradayPriceData] = []
        self._idx: int = 0

    _ENTRY_FLAGS: dict[str, bool] = {}

    @staticmethod
    def pinescript_mode() -> ModeDefinition:
        """Return the PineScript mode definition for ML Oversold."""
        return ml_oversold_mode()

    @property
    def name(self) -> str:
        return "ML Oversold"

    # ------------------------------------------------------------------
    # Override evaluate to capture data/index for ML prediction
    # (same pattern as PatternDiscoveryStrategy)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Entry logic — the model decides
    # ------------------------------------------------------------------

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state

        # ── Standard guards ───────────────────────────────────────────
        if s.day_trades >= c.max_day:
            return None
        if s.last_entry_bar > 0 and (ctx.bar_of_day - s.last_entry_bar) < c.spacing:
            return None
        if ctx.bar_of_day < c.can_trade_bar:
            return None

        # Need enough bars for feature extraction (warmup = 201)
        if self._idx < 201:
            return None

        # ── THE MODEL DECIDES ─────────────────────────────────────────
        if self._predictor is None:
            return None  # No model loaded — cannot make predictions

        prob = self._predictor.predict_proba(self._data, self._idx)

        if prob < float(c.ml_threshold):
            return None  # Model not confident enough

        # Model says go → long entry (SPY mean-reversion bias)
        go_long = True

        # ── SL / TP (standard risk management) ────────────────────────
        sl_dist = clamp_sl_dist(ctx.atr_val * c.ml_sl_atr, c.ml_sl_cap)
        if sl_dist is None:
            return None

        if c.ml_tp_mode == "vwap":
            tp_price = ctx.vwap_bands.vwap
            sl_price = ctx.bar.close - sl_dist
            # If VWAP is below close (already there), fall back to R:R
            if tp_price <= ctx.bar.close:
                tp_dist = sl_dist * c.ml_rr
                sl_price, tp_price = directional_sl_tp(
                    go_long, ctx.bar.close, sl_dist, tp_dist,
                )
        else:  # "rr"
            tp_dist = sl_dist * c.ml_rr
            sl_price, tp_price = directional_sl_tp(
                go_long, ctx.bar.close, sl_dist, tp_dist,
            )

        # ── R:R filter ────────────────────────────────────────────────
        reward = abs(tp_price - ctx.bar.close)
        rr = reward / sl_dist if sl_dist > ZERO else ZERO
        if rr < c.ml_min_rr:
            return None

        # Score = probability × 100 (for reporting / confluence UI)
        score = int(prob * 100)

        return make_entry_signal(
            go_long=True,
            mode="ML",
            sl_price=sl_price,
            tp_price=tp_price,
            score=score,
            max_score=100,
            risk_per_share=sl_dist,
            reason=f"ML prob={prob:.1%}",
        )
