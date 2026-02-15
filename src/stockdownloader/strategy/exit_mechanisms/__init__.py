"""Concrete exit mechanism implementations for the exit tournament.

All six exit mechanisms extend :class:`TrailingExitBase` and implement
``evaluate_bar`` to decide when to close a position.
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.model.trade import Direction
from stockdownloader.strategy.exit_mechanism import TrailingExitBase
from stockdownloader.util.big_decimal_math import ZERO
from stockdownloader.util.indicator_hub import IndicatorHub

if TYPE_CHECKING:
    from stockdownloader.model.intraday_price_data import IntradayPriceData
    from stockdownloader.model.tournament_trade import TournamentTrade


# =====================================================================
# TrailingStopExit — fixed R-based trailing stop
# =====================================================================


class TrailingStopExit(TrailingExitBase):
    """0.5R trailing stop from peak after 1R breakeven trigger.

    Args:
        activation_r: R-multiple at which the breakeven stop activates.
        trail_r: R-multiple width of the trailing stop from peak.
        be_buffer_pct: Percentage of 1R to add as a breakeven buffer.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("1"),
        trail_r: Decimal = Decimal("0.5"),
        be_buffer_pct: Decimal = Decimal("0.05"),
    ) -> None:
        super().__init__()
        self._activation_r = activation_r
        self._trail_r = trail_r
        self._be_buffer_pct = be_buffer_pct

    @property
    def name(self) -> str:
        return "CURRENT_TRAIL"

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        stop_dist = trade.stop_distance
        return self._evaluate_trail(
            bar=bar,
            direction=trade.direction,
            entry=trade.entry_price,
            stop_dist=stop_dist,
            trail_buffer=stop_dist * self._trail_r,
            activation_r=self._activation_r,
            be_buffer=stop_dist * self._be_buffer_pct,
        )


# =====================================================================
# VwapCrossExit — exit on session VWAP cross
# =====================================================================


class VwapCrossExit(TrailingExitBase):
    """Exit on VWAP cross after activation threshold.

    Args:
        activation_r: R-multiple threshold before VWAP trailing begins.
        name: Display name override (defaults to ``'VWAP_CROSS'``).
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("0.5"),
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._activation_r = activation_r
        self._name = name or "VWAP_CROSS"
        self._hub = IndicatorHub()
        self._vwap_active: bool = False

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        super().reset()
        self._vwap_active = False

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        direction = trade.direction
        entry = trade.entry_price
        stop_dist = trade.stop_distance
        activation = stop_dist * self._activation_r

        self._init_stop(entry, stop_dist, direction)
        data_index = self._data_offset + bar_index

        # Hard stop check
        if self._check_hard_stop(bar, direction):
            return self._trigger_exit(self._stop, "hard_stop")

        # Track peak
        self._update_peak(bar, direction)

        # Activation
        if not self._vwap_active:
            if direction == Direction.LONG:
                if self._peak >= entry + activation:
                    self._vwap_active = True
            else:
                if self._peak <= entry - activation:
                    self._vwap_active = True

        # VWAP cross exit
        if self._vwap_active and self._data:
            vwap_val = self._hub.session_vwap(self._data, data_index)
            if direction == Direction.LONG and bar.close < vwap_val:
                return self._trigger_exit(bar.close, "vwap_cross")
            if direction == Direction.SHORT and bar.close > vwap_val:
                return self._trigger_exit(bar.close, "vwap_cross")

        return False


# =====================================================================
# AtrTrailExit — adaptive ATR-based trailing stop
# =====================================================================


class AtrTrailExit(TrailingExitBase):
    """0.3x ATR(14) trailing stop from peak after activation.

    Args:
        atr_multiplier: Fraction of ATR to use as trail distance.
        activation_r: R-multiple threshold before trailing begins.
        atr_period: ATR lookback period.
    """

    def __init__(
        self,
        atr_multiplier: Decimal = Decimal("0.3"),
        activation_r: Decimal = Decimal("1"),
        atr_period: int = 14,
    ) -> None:
        super().__init__()
        self._atr_multiplier = atr_multiplier
        self._activation_r = activation_r
        self._atr_period = atr_period
        self._hub = IndicatorHub()

    @property
    def name(self) -> str:
        return "ATR_TRAIL"

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        stop_dist = trade.stop_distance
        data_index = self._data_offset + bar_index

        # Compute current ATR for adaptive trail distance
        current_atr = ZERO
        if self._data and data_index >= self._atr_period:
            current_atr = self._hub.atr(self._data, data_index, self._atr_period)
        if current_atr <= ZERO:
            current_atr = stop_dist  # fallback

        trail_dist = current_atr * self._atr_multiplier

        return self._evaluate_trail(
            bar=bar,
            direction=trade.direction,
            entry=trade.entry_price,
            stop_dist=stop_dist,
            trail_buffer=trail_dist,
            activation_r=self._activation_r,
            be_buffer=None,  # ATR trail doesn't set breakeven stop
        )


# =====================================================================
# HybridExit — ATR trail + VWAP cross combination
# =====================================================================


class HybridExit(TrailingExitBase):
    """ATR trail + VWAP cross, whichever triggers first.

    Args:
        activation_r: R-multiple threshold before both trails activate.
        atr_multiplier: Fraction of ATR to use as trail distance.
        atr_period: ATR lookback period.
        name: Display name override.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("0.5"),
        atr_multiplier: Decimal = Decimal("0.3"),
        atr_period: int = 14,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._activation_r = activation_r
        self._atr_multiplier = atr_multiplier
        self._atr_period = atr_period
        self._name = name or "HYBRID"
        self._hub = IndicatorHub()
        self._trailing_active: bool = False
        self._be_buffer: Decimal = ZERO

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        super().reset()
        self._trailing_active = False
        self._be_buffer = ZERO

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        direction = trade.direction
        entry = trade.entry_price
        stop_dist = trade.stop_distance
        activation = stop_dist * self._activation_r

        self._init_stop(entry, stop_dist, direction)

        if not self._be_buffer:
            self._be_buffer = stop_dist * Decimal("0.05")

        data_index = self._data_offset + bar_index

        # Compute current ATR
        current_atr = ZERO
        if self._data and data_index >= self._atr_period:
            current_atr = self._hub.atr(self._data, data_index, self._atr_period)
        if current_atr <= ZERO:
            current_atr = stop_dist

        trail_dist = current_atr * self._atr_multiplier

        # Hard stop check
        if self._check_hard_stop(bar, direction):
            reason = "hard_stop" if not self._trailing_active else "trail_stop"
            return self._trigger_exit(self._stop, reason)

        # Track peak
        self._update_peak(bar, direction)

        # Activation
        if not self._trailing_active:
            if direction == Direction.LONG:
                if self._peak >= entry + activation:
                    self._trailing_active = True
                    self._stop = entry + self._be_buffer
            else:
                if self._peak <= entry - activation:
                    self._trailing_active = True
                    self._stop = entry - self._be_buffer

        # Trailing: ATR stop + VWAP cross
        if self._trailing_active:
            if direction == Direction.LONG:
                atr_stop = self._peak - trail_dist
                if atr_stop > self._stop:
                    self._stop = atr_stop
            else:
                atr_stop = self._peak + trail_dist
                if atr_stop < self._stop:
                    self._stop = atr_stop

            if self._data:
                vwap_val = self._hub.session_vwap(self._data, data_index)
                if direction == Direction.LONG and bar.close < vwap_val:
                    return self._trigger_exit(bar.close, "vwap_cross")
                if direction == Direction.SHORT and bar.close > vwap_val:
                    return self._trigger_exit(bar.close, "vwap_cross")

        return False


# =====================================================================
# VwapBandExit — exit at sigma VWAP band
# =====================================================================


class VwapBandExit(TrailingExitBase):
    """Exit at VWAP standard-deviation band.

    Args:
        band_sigma: Sigma multiplier for the band (0.5 = half-sigma).
        activation_r: R-multiple threshold before band tracking begins.
    """

    def __init__(
        self,
        band_sigma: Decimal = Decimal("0.5"),
        activation_r: Decimal = Decimal("0.5"),
    ) -> None:
        super().__init__()
        self._band_sigma = band_sigma
        self._activation_r = activation_r
        self._hub = IndicatorHub()
        self._vwap_active: bool = False

    @property
    def name(self) -> str:
        return "VWAP_BAND"

    def reset(self) -> None:
        super().reset()
        self._vwap_active = False

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        direction = trade.direction
        entry = trade.entry_price
        stop_dist = trade.stop_distance
        activation = stop_dist * self._activation_r

        self._init_stop(entry, stop_dist, direction)
        data_index = self._data_offset + bar_index

        # Hard stop check
        if self._check_hard_stop(bar, direction):
            return self._trigger_exit(self._stop, "hard_stop")

        # Track peak
        self._update_peak(bar, direction)

        # Activation
        if not self._vwap_active:
            if direction == Direction.LONG:
                if self._peak >= entry + activation:
                    self._vwap_active = True
            else:
                if self._peak <= entry - activation:
                    self._vwap_active = True

        # VWAP band exit
        if self._vwap_active and self._data:
            bands = self._hub.session_vwap_bands(self._data, data_index)
            if direction == Direction.LONG:
                band_level = bands.vwap - self._band_sigma * bands.std_dev
                if bar.close < band_level:
                    return self._trigger_exit(bar.close, "band_cross")
            else:
                band_level = bands.vwap + self._band_sigma * bands.std_dev
                if bar.close > band_level:
                    return self._trigger_exit(bar.close, "band_cross")

        return False


# =====================================================================
# TimeDecayExit — trail tightens as session progresses
# =====================================================================


class TimeDecayExit(TrailingExitBase):
    """Trailing stop that tightens by time-of-day.

    Args:
        activation_r: R-multiple threshold for breakeven trigger.
        early_trail_r: Trail width before 11:00 (default 0.6R).
        mid_trail_r: Trail width 11:00-13:00 (default 0.4R).
        late_trail_r: Trail width after 13:00 (default 0.2R).
        be_buffer_pct: Percentage of 1R added as breakeven buffer.
    """

    def __init__(
        self,
        activation_r: Decimal = Decimal("1"),
        early_trail_r: Decimal = Decimal("0.6"),
        mid_trail_r: Decimal = Decimal("0.4"),
        late_trail_r: Decimal = Decimal("0.2"),
        be_buffer_pct: Decimal = Decimal("0.05"),
    ) -> None:
        super().__init__()
        self._activation_r = activation_r
        self._early_trail_r = early_trail_r
        self._mid_trail_r = mid_trail_r
        self._late_trail_r = late_trail_r
        self._be_buffer_pct = be_buffer_pct

    @property
    def name(self) -> str:
        return "TIME_DECAY"

    def _trail_multiplier(self, time_str: str) -> Decimal:
        """Choose trail width based on time of day.

        *time_str* is expected in ``HH:MM:SS`` format.
        """
        hour_str = time_str[:2]
        try:
            hour = int(hour_str)
        except ValueError:
            return self._mid_trail_r

        if hour < 11:
            return self._early_trail_r
        if hour < 13:
            return self._mid_trail_r
        return self._late_trail_r

    def evaluate_bar(
        self,
        bar: IntradayPriceData,
        trade: TournamentTrade,
        bar_index: int,
    ) -> bool:
        stop_dist = trade.stop_distance
        trail_mult = self._trail_multiplier(bar.time_str)
        return self._evaluate_trail(
            bar=bar,
            direction=trade.direction,
            entry=trade.entry_price,
            stop_dist=stop_dist,
            trail_buffer=stop_dist * trail_mult,
            activation_r=self._activation_r,
            be_buffer=stop_dist * self._be_buffer_pct,
        )


__all__ = [
    "TrailingStopExit",
    "VwapCrossExit",
    "AtrTrailExit",
    "HybridExit",
    "VwapBandExit",
    "TimeDecayExit",
]
