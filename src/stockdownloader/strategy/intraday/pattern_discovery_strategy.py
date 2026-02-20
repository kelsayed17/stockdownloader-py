"""Strategy that trades data-mined price action patterns.

On each bar, encodes the last N bars into :class:`BarFeatures`, looks
up the pattern in a pre-computed :class:`PatternCatalog`, and enters
if a pattern with a validated statistical edge is found.

SL/TP are derived from the pattern's historical MAE/MFE, not from
fixed ATR multiples.

Usage::

    from stockdownloader.analysis.pattern_discovery import PatternCatalog
    from stockdownloader.strategy.intraday.pattern_discovery_strategy import (
        PatternDiscoveryStrategy,
    )

    catalog = PatternCatalog.load("output/patterns/catalog.json")
    strategy = PatternDiscoveryStrategy(catalog)
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.analysis.pattern_encoder import BarEncoder, BarFeatures
from stockdownloader.model.trade import IntradaySignal
from stockdownloader.strategy.intraday.base_config import InfraExitConfig
from stockdownloader.strategy.intraday.entry_helpers import (
    directional_sl_tp,
    make_entry_signal,
)
from stockdownloader.strategy.intraday.exit_manager import IntradayExitManager
from stockdownloader.strategy.intraday.session_state import BarContext
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.trail_strategy import VwapRatchetTrail
from stockdownloader.strategy.intraday_trading_strategy import IntradayTradingStrategy
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.indicator_hub import IndicatorHub

if TYPE_CHECKING:
    from stockdownloader.analysis.pattern_discovery import (
        DiscoveredPattern,
        PatternCatalog,
    )
    from stockdownloader.model.price_data import IntradayPriceData


@dataclass(frozen=True, slots=True)
class PatternDiscoveryConfig(InfraExitConfig):
    """Configuration for the pattern discovery strategy.

    Inherits infrastructure / exit / trail fields from
    :class:`InfraExitConfig`.  Pattern-specific fields control SL/TP
    sizing from historical MAE/MFE.
    """

    # ── SL/TP from pattern statistics ─────────────────────────────────
    sl_mae_multiplier: Decimal = Decimal("1.2")
    """SL = avg_mae × multiplier (buffer above historical worst case)."""

    tp_mfe_multiplier: Decimal = Decimal("0.8")
    """TP = avg_mfe × multiplier (conservative — take profit early)."""

    sl_cap: Decimal = Decimal("2.50")
    """Absolute stop-loss cap in dollars."""

    min_rr: Decimal = Decimal("1.0")
    """Minimum risk:reward ratio to enter."""

    # ── Entry filters ─────────────────────────────────────────────────
    max_day: int = 2
    """Maximum trades per day."""

    spacing: int = 3
    """Minimum bars between entries."""

    use_confirmation: bool = True
    """Check indicator confirmation conditions before entry."""

    require_htf_alignment: bool = False
    """When ``True``, reject long patterns when HTF trend is down and
    short patterns when HTF trend is up."""


class PatternDiscoveryStrategy(IntradayTradingStrategy):
    """Trades discovered price action patterns from a pre-mined catalog.

    Composes :class:`IntradayInfra` for session management, exit
    evaluation, and trailing stops — identical to other strategies
    in the codebase.

    The only custom logic is in :meth:`_evaluate_entry`, which:
    1. Encodes the current bar into :class:`BarFeatures`.
    2. Maintains a rolling buffer of recent features.
    3. Looks up the buffer in the :class:`PatternCatalog`.
    4. If a pattern matches, computes SL/TP from historical MAE/MFE.
    """

    def __init__(
        self,
        catalog: PatternCatalog,
        config: PatternDiscoveryConfig | None = None,
        hub: IndicatorHub | None = None,
    ) -> None:
        c = config or PatternDiscoveryConfig()
        self._c = c
        self._catalog = catalog
        self._hub = hub or IndicatorHub()
        self._encoder = BarEncoder(self._hub)
        self._infra = IntradayInfra(c, IntradayExitManager(VwapRatchetTrail()))
        self._max_pattern_len = (
            max(len(p.key) for p in catalog.patterns) if catalog.patterns else 5
        )
        self._feature_buffer: list[BarFeatures] = []
        self._last_trading_date = ""
        # Set during evaluate() so _check_confirmation can access hub
        self._data: list[IntradayPriceData] = []
        self._idx: int = 0

    _ENTRY_FLAGS: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "Pattern Discovery"

    @property
    def warmup_period(self) -> int:
        return self._infra.warmup_period

    def on_session_start(self, trading_date: str) -> None:
        self._infra.on_session_start(trading_date)
        if trading_date != self._last_trading_date:
            self._feature_buffer.clear()
            self._last_trading_date = trading_date

    def on_position_opened(self, is_long: bool) -> None:
        self._infra.confirm_position_opened(is_long)

    def on_position_closed(self) -> None:
        self._infra.confirm_position_closed()

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

        # ── Rate limiting ─────────────────────────────────────────────
        if s.day_trades >= c.max_day:
            return None
        if s.last_entry_bar > 0 and (ctx.bar_of_day - s.last_entry_bar) < c.spacing:
            return None

        # ── Encode current bar ────────────────────────────────────────
        features = self._encoder.encode_bar(
            ctx.bar, ctx.atr_val,
            # We don't pass data/index here since atr_val is pre-computed
            # but we need rel_vol. Use "normal" as fallback.
        )
        self._feature_buffer.append(features)

        # Keep buffer at max pattern length
        if len(self._feature_buffer) > self._max_pattern_len:
            self._feature_buffer = self._feature_buffer[-self._max_pattern_len:]

        # ── Lookup pattern ────────────────────────────────────────────
        recent = tuple(self._feature_buffer)
        pattern = self._catalog.lookup(recent)
        if pattern is None:
            return None

        # ── Indicator confirmation ───────────────────────────────────
        if not self._check_confirmation(pattern, ctx):
            return None

        # ── Direction ─────────────────────────────────────────────────
        go_long = pattern.direction == "long"
        go_short = pattern.direction == "short"

        # ── HTF alignment gate ────────────────────────────────────────
        if c.require_htf_alignment:
            try:
                htf_trend = self._hub.htf_ema_trend(self._data, self._idx)
                if go_long and htf_trend < 0:
                    return None
                if go_short and htf_trend > 0:
                    return None
            except (ValueError, KeyError, IndexError):
                pass  # skip gate if HTF trend unavailable
        if not go_long and not go_short:
            return None

        # ── SL/TP from historical MAE/MFE ─────────────────────────────
        close_price = ctx.bar.close
        atr_val = ctx.atr_val

        # MAE is negative (adverse), MFE is positive (favorable)
        mae_pct = abs(pattern.avg_mae)
        mfe_pct = abs(pattern.avg_mfe)

        # Convert percentages to dollar distances
        mae_sl = Decimal(str(mae_pct)) * c.sl_mae_multiplier * close_price / Decimal("100")
        mfe_tp = Decimal(str(mfe_pct)) * c.tp_mfe_multiplier * close_price / Decimal("100")

        # Clamp SL
        sl_dist = min(mae_sl, c.sl_cap)
        if sl_dist <= ZERO:
            sl_dist = atr_val  # fallback to ATR

        # Check minimum R:R
        if mfe_tp <= ZERO or sl_dist <= ZERO:
            return None
        rr = mfe_tp / sl_dist
        if rr < c.min_rr:
            return None

        sl_price, tp_price = directional_sl_tp(go_long, close_price, sl_dist, mfe_tp)

        return make_entry_signal(
            go_long=go_long,
            mode="PAT",
            sl_price=sl_price,
            tp_price=tp_price,
            score=min(pattern.occurrences, 100),
            max_score=100,
            risk_per_share=sl_dist,
            reason=pattern.human_label,
        )

    def _check_confirmation(
        self, pattern: DiscoveredPattern, ctx: BarContext,
    ) -> bool:
        """Verify indicator confirmation conditions before entry.

        Returns ``True`` if all confirmation conditions are met or
        if confirmation checking is disabled / pattern has no conditions.
        """
        if not self._c.use_confirmation or not pattern.confirmation:
            return True

        for field, required_value in pattern.confirmation.items():
            actual: str | None = None

            if field == "trend_dir":
                if ctx.ema_fast > ctx.ema_slow:
                    actual = "1"
                elif ctx.ema_fast < ctx.ema_slow:
                    actual = "-1"
                else:
                    actual = "0"

            elif field == "rsi_zone":
                if ctx.rsi_val < Decimal("30"):
                    actual = "oversold"
                elif ctx.rsi_val > Decimal("70"):
                    actual = "overbought"
                else:
                    actual = "neutral"

            elif field == "vwap_position":
                vwap = ctx.vwap_bands.vwap
                if vwap > ZERO:
                    delta_pct = abs(ctx.bar.close - vwap) / vwap
                    if delta_pct < Decimal("0.001"):
                        actual = "at"
                    elif ctx.bar.close > vwap:
                        actual = "above"
                    else:
                        actual = "below"

            elif field == "adx_level":
                if ctx.adx_val < Decimal("20"):
                    actual = "weak"
                elif ctx.adx_val > Decimal("40"):
                    actual = "strong"
                else:
                    actual = "moderate"

            elif field == "obv_trend":
                try:
                    obv_rising = self._hub.is_obv_rising(
                        self._data, self._idx, 5,
                    )
                    actual = "rising" if obv_rising else "falling"
                except (ValueError, KeyError, IndexError):
                    continue  # skip if hub can't compute

            elif field == "macd_signal":
                try:
                    macd_hist = self._hub.macd_histogram(
                        self._data, self._idx,
                    )
                    actual = "bullish" if macd_hist > ZERO else "bearish"
                except (ValueError, KeyError, IndexError):
                    continue

            elif field == "htf_trend":
                try:
                    htf = self._hub.htf_ema_trend(
                        self._data, self._idx,
                    )
                    actual = str(htf)
                except (ValueError, KeyError, IndexError):
                    continue

            elif field == "cvd_direction":
                try:
                    cvd_val = self._hub.cvd_normalized(
                        self._data, self._idx,
                    )
                    if cvd_val > Decimal("0.3"):
                        actual = "buying"
                    elif cvd_val < Decimal("-0.3"):
                        actual = "selling"
                    else:
                        actual = "neutral"
                except (ValueError, KeyError, IndexError):
                    continue

            else:
                continue  # unknown field — skip

            if actual is not None and actual != required_value:
                return False

        return True
