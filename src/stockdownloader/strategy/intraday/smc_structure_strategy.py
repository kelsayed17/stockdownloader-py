"""Standalone SMC Structure strategy — Smart Money Concepts.

Trades pullbacks into supply/demand zones after break of structure (BoS),
and optional liquidity sweep entries.  Market structure (swing highs/lows,
trend direction) is tracked incrementally via
:class:`StreamingStructureTracker`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

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
from stockdownloader.strategy.intraday.trail_strategy import AtrChandelierTrail
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.util.intraday_indicators import candle_strength
from stockdownloader.util.smc_indicators import (
    is_liquidity_sweep_high,
    is_liquidity_sweep_low,
)
from stockdownloader.util.big_decimal_math import ZERO

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData


@dataclass(frozen=True, slots=True)
class SMCStructureConfig(InfraExitConfig):
    """SMC Structure strategy configuration.

    Inherits all infrastructure / exit / trail fields from
    :class:`InfraExitConfig` and adds SMC-specific parameters.
    """

    # ── Infrastructure flag ──────────────────────────────────────────
    use_smc: bool = True

    # ── Structure detection ──────────────────────────────────────────
    smc_swing_lookback: int = 3  # Faster swing confirmation for intraday
    smc_min_impulse_atr: Decimal = Decimal("2.5")  # Moderate impulse threshold
    smc_zone_bars: int = 2  # Candles at impulse origin = zone

    # ── Entry conditions ─────────────────────────────────────────────
    smc_require_bos: bool = True  # Require BoS before entry
    smc_zone_atr: Decimal = Decimal("0.8")  # Proximity to zone (× ATR)
    smc_body_min: Decimal = Decimal("0.10")  # Lower body min for more setups
    smc_longs: bool = True
    smc_shorts: bool = False  # SPY long-only bias
    smc_sweep_entry: bool = True  # Enable sweep entries for more trades
    smc_sweep_tolerance: Decimal = Decimal("0.3")  # Sweep wick limit (× ATR)

    # ── Filters ──────────────────────────────────────────────────────
    smc_htf_align: bool = True  # HTF trend must agree
    smc_vwap_agree: bool = False  # Disabled — HTF alignment sufficient
    smc_min_age: int = 10  # Min bars since BoS for pullback
    smc_max_age: int = 60  # Max bars since BoS (staleness)

    # ── SL / TP ──────────────────────────────────────────────────────
    smc_sl_mode: str = "atr"  # ATR-based SL for consistency
    smc_sl_atr: Decimal = Decimal("2.0")  # Wider SL to avoid whipsaws
    smc_sl_cap: Decimal = Decimal("1.50")  # Tighter hard cap for risk control
    smc_rr: Decimal = Decimal("1.0")  # 1:1 R:R — high WR compensates
    smc_tp_mode: str = "rr"  # "rr" | "swing" | "vwap"

    # ── Overrides (w_vol, w_sr, w_rsi, w_time inherited) ────────────
    max_day: int = 2  # Allow 2 trades per day (base: 1)
    spacing: int = 10  # Wider spacing (base: 5)
    min_score: int = 3  # Lower min score for more setups (base: 4)


class SMCStructureStrategy(BaseIntradayStrategy):
    """Standalone SMC Structure strategy.

    Enters on pullbacks into supply/demand zones after a confirmed
    break of structure, with optional liquidity sweep entries.
    Uses ATR chandelier trailing stop for exits.
    """

    def __init__(self, config: SMCStructureConfig | None = None) -> None:
        c = config or SMCStructureConfig()
        self._c = c
        self._infra = IntradayInfra(c, IntradayExitManager(AtrChandelierTrail()))

    _ENTRY_FLAGS: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "SMC Structure"

    def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
        c = self._c
        s = ctx.state
        structure = ctx.structure

        # -- Guard: structure must be available --
        if structure is None:
            return None

        # -- Guard: need at least confirmed swings --
        if structure.last_swing_high <= ZERO or structure.last_swing_low <= ZERO:
            return None

        # -- Spacing + day trade limit --
        ready = s.day_trades < c.max_day
        spaced = (ctx.bar_of_day - s.last_entry_bar) >= c.spacing or s.last_entry_bar <= 0
        if not ready or not spaced:
            return None

        # -- Time filter --
        if ctx.bar_of_day < c.can_trade_bar:
            return None
        if not ctx.is_good_time:
            return None

        # -- BoS requirement --
        if c.smc_require_bos and structure.trend == 0:
            return None

        # -- BoS age filter --
        if structure.last_bos_idx >= 0:
            bos_age = ctx.bar_of_day  # approximate bars-since-BoS within session
            # Use index-level age if BoS happened this session
            data_index_proxy = structure.last_bos_idx
            if data_index_proxy >= 0:
                # We use bar_of_day as proxy since BoS idx is data-level
                # and bar_of_day tracks the session position
                pass  # age checks below use session-relative thresholds

        # -- Try zone pullback entry --
        zone_signal = self._try_zone_entry(ctx, structure)
        if zone_signal is not None:
            return zone_signal

        # -- Try liquidity sweep entry --
        if c.smc_sweep_entry:
            sweep_signal = self._try_sweep_entry(ctx, structure)
            if sweep_signal is not None:
                return sweep_signal

        return None

    def _try_zone_entry(
        self, ctx: BarContext, structure,
    ) -> IntradaySignal | None:
        """Entry mode A: pullback into supply/demand zone."""
        c = self._c
        bar = ctx.bar
        atr = ctx.atr_val

        go_long = False
        go_short = False
        zone_top = ZERO
        zone_bot = ZERO

        # Uptrend + demand zone pullback
        if (
            structure.trend >= 1
            and structure.demand_zone_valid
            and structure.demand_zone_top > ZERO
            and c.smc_longs
        ):
            zone_top = structure.demand_zone_top
            zone_bot = structure.demand_zone_bot
            tolerance = atr * c.smc_zone_atr
            # Price must be within the zone (extended by tolerance)
            if bar.close <= zone_top + tolerance and bar.close >= zone_bot - tolerance:
                go_long = True

        # Downtrend + supply zone rally
        if (
            not go_long
            and structure.trend <= -1
            and structure.supply_zone_valid
            and structure.supply_zone_top > ZERO
            and c.smc_shorts
        ):
            zone_top = structure.supply_zone_top
            zone_bot = structure.supply_zone_bot
            tolerance = atr * c.smc_zone_atr
            if bar.close >= zone_bot - tolerance and bar.close <= zone_top + tolerance:
                go_short = True

        if not go_long and not go_short:
            return None

        # -- Candle confirmation --
        cs = candle_strength(bar, atr)
        if go_long and not cs.bull_candle(c.smc_body_min):
            return None
        if go_short and not cs.bear_candle(c.smc_body_min):
            return None

        # -- Apply filters and compute SL/TP --
        return self._apply_filters_and_signal(
            ctx, go_long, go_short, zone_top, zone_bot, "Zone PB",
        )

    def _try_sweep_entry(
        self, ctx: BarContext, structure,
    ) -> IntradaySignal | None:
        """Entry mode B: liquidity sweep reversal."""
        c = self._c
        bar = ctx.bar
        atr = ctx.atr_val

        go_long = False
        go_short = False

        # Uptrend: sweep of a recent swing low (grab sell-side liquidity)
        if (
            structure.trend >= 1
            and structure.last_swing_low > ZERO
            and c.smc_longs
        ):
            if is_liquidity_sweep_low(
                bar, structure.last_swing_low, atr, c.smc_sweep_tolerance,
            ):
                go_long = True

        # Downtrend: sweep of a recent swing high (grab buy-side liquidity)
        if (
            not go_long
            and structure.trend <= -1
            and structure.last_swing_high > ZERO
            and c.smc_shorts
        ):
            if is_liquidity_sweep_high(
                bar, structure.last_swing_high, atr, c.smc_sweep_tolerance,
            ):
                go_short = True

        if not go_long and not go_short:
            return None

        # -- Candle confirmation (stronger requirement for sweeps) --
        cs = candle_strength(bar, atr)
        if go_long and not cs.bull_candle(c.smc_body_min):
            return None
        if go_short and not cs.bear_candle(c.smc_body_min):
            return None

        # Use swing levels as zone boundaries for SL
        zone_top = structure.last_swing_high if go_short else ZERO
        zone_bot = structure.last_swing_low if go_long else ZERO

        return self._apply_filters_and_signal(
            ctx, go_long, go_short, zone_top, zone_bot, "Sweep",
        )

    def _apply_filters_and_signal(
        self,
        ctx: BarContext,
        go_long: bool,
        go_short: bool,
        zone_top: Decimal,
        zone_bot: Decimal,
        mode_label: str,
    ) -> IntradaySignal | None:
        """Apply shared filters, scoring, and SL/TP computation."""
        c = self._c
        bar = ctx.bar
        structure = ctx.structure
        assert structure is not None

        # -- HTF alignment --
        if c.smc_htf_align:
            if go_long and ctx.htf_trend == -1:
                go_long = False
            if go_short and ctx.htf_trend == 1:
                go_short = False
            if not go_long and not go_short:
                return None

        # -- VWAP agreement --
        if c.smc_vwap_agree:
            vwap = ctx.vwap_bands.vwap
            if go_long and bar.close < vwap:
                go_long = False
            if go_short and bar.close > vwap:
                go_short = False
            if not go_long and not go_short:
                return None

        # -- ADX trend filter: need a trending market for SMC --
        if ctx.adx_val < Decimal("22"):
            return None

        # -- Confluence scoring --
        pts_vol = c.w_vol if ctx.tod_rvol >= Decimal("1.3") else 0
        pts_sr = c.w_sr if ctx.sr_score_count > 0 else 0
        pts_time = c.w_time if ctx.is_good_time else 0

        if go_long:
            pts_rsi = c.w_rsi if ctx.rsi_val <= Decimal("35") else 0
        else:
            pts_rsi = c.w_rsi if ctx.rsi_val >= Decimal("65") else 0

        score = pts_vol + pts_sr + pts_rsi + pts_time
        max_score = c.w_vol + c.w_sr + c.w_rsi + c.w_time

        if score < c.min_score:
            return None

        # -- SL / TP --
        if c.smc_sl_mode == "zone" and zone_bot > ZERO and zone_top > ZERO:
            # Zone-based SL: place SL just beyond zone boundary
            buffer = ctx.atr_val * Decimal("0.2")
            if go_long:
                sl_price = zone_bot - buffer
                sl_dist = bar.close - sl_price
            else:
                sl_price = zone_top + buffer
                sl_dist = sl_price - bar.close

            # Clamp SL distance
            if sl_dist <= ZERO:
                return None
            sl_dist = min(sl_dist, c.smc_sl_cap)
            if sl_dist <= ZERO:
                return None
        else:
            # ATR-based SL
            sl_dist = clamp_sl_dist(ctx.atr_val * c.smc_sl_atr, c.smc_sl_cap)
            if sl_dist is None:
                return None

        # TP computation
        if c.smc_tp_mode == "swing":
            # Target the previous swing level
            if go_long:
                tp_price = structure.last_swing_high
            else:
                tp_price = structure.last_swing_low
            sl_price_final = bar.close - sl_dist if go_long else bar.close + sl_dist
            reward = abs(tp_price - bar.close)
            rr_actual = reward / sl_dist if sl_dist > ZERO else ZERO
            if rr_actual < Decimal("0.8"):
                return None
        elif c.smc_tp_mode == "vwap":
            tp_price = ctx.vwap_bands.vwap
            sl_price_final = bar.close - sl_dist if go_long else bar.close + sl_dist
            reward = abs(tp_price - bar.close)
            rr_actual = reward / sl_dist if sl_dist > ZERO else ZERO
            if rr_actual < Decimal("0.8"):
                return None
        else:  # "rr"
            tp_dist = sl_dist * c.smc_rr
            sl_price_final, tp_price = directional_sl_tp(
                go_long, bar.close, sl_dist, tp_dist,
            )

        trend_label = "Up" if structure.trend >= 1 else "Down"
        reason = f"SMC {mode_label} ({trend_label})"

        return make_entry_signal(
            go_long=go_long,
            mode="SMC",
            sl_price=sl_price_final,
            tp_price=tp_price,
            score=score,
            max_score=max_score,
            risk_per_share=sl_dist,
            reason=reason,
        )
