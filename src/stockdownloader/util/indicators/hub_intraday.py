"""Intraday, session-scoped, and anchored-VWAP indicators for IndicatorHub.

Extracted from :mod:`hub` to keep the main hub file focused on core
indicators (moving averages, momentum, volatility, trend, volume basics).

This mixin class is inherited by :class:`IndicatorHub` and provides
session-scoped VWAP, anchored VWAP, intraday composites (CVD, LRS,
VWAP slope/acceleration), higher-timeframe EMA, relative volume, and
market structure (SMC) indicators.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from stockdownloader.util.math import ZERO, quantize as _quantize
from stockdownloader.util.indicators._core import ema as _ema
from stockdownloader.util.indicators import volume, intraday as ii
from stockdownloader.util.indicators.volume import (
    StreamingAnchoredVWAP,
    StreamingCVD,
    StreamingSessionVWAP,
)
from stockdownloader.util.indicators.htf import StreamingHTFResample
from stockdownloader.util.indicators.smc import StreamingStructureTracker

if TYPE_CHECKING:
    from collections.abc import Sequence
    from stockdownloader.model.price_data import PriceData


class IntraDayHubMixin:
    """Mixin providing intraday and session-scoped indicator methods.

    Assumes the host class (IndicatorHub) provides:

    - ``self._cache: dict[tuple, Any]``
    - ``self._ensure_bound(data)``
    - ``self._get(key, data, fn, *args, **kwargs)``
    - ``self._s_vwap``, ``self._s_avwap``, ``self._s_cvd``,
      ``self._s_htf``, ``self._s_structure`` streaming accumulators
    - ``self.atr(data, index, period)`` for ATR lookups
    """

    # ==================================================================
    # Session-scoped indicators (intraday)
    # ==================================================================

    def _vwap_core(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> tuple[Decimal, Decimal]:
        """Streaming session VWAP core — returns ``(vwap, std_dev)``."""
        self._ensure_bound(data)
        if self._s_vwap is None:
            self._s_vwap = StreamingSessionVWAP()
        return self._s_vwap.update(data, index)

    def session_vwap(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> Decimal:
        """Intraday session VWAP (streaming)."""
        key = ("session_vwap", index)
        self._ensure_bound(data)
        if key not in self._cache:
            vwap, _ = self._vwap_core(data, index)
            self._cache[key] = vwap
        return self._cache[key]

    def session_vwap_bands(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> volume.SessionVWAP:
        """Intraday session VWAP with standard deviation bands (streaming)."""
        key = ("session_vwap_bands", index)
        self._ensure_bound(data)
        if key not in self._cache:
            vwap_val, std_val = self._vwap_core(data, index)
            if vwap_val == ZERO and std_val == ZERO:
                self._cache[key] = volume.SessionVWAP(ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)
            else:
                half_std = _quantize(std_val * Decimal("0.5"))
                self._cache[key] = volume.SessionVWAP(
                    vwap=vwap_val,
                    std_dev=std_val,
                    upper_1=_quantize(vwap_val + std_val),
                    lower_1=_quantize(vwap_val - std_val),
                    upper_05=_quantize(vwap_val + half_std),
                    lower_05=_quantize(vwap_val - half_std),
                )
        return self._cache[key]

    def extended_session_vwap_bands(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> volume.ExtendedSessionVWAP:
        """Extended session VWAP with multi-sigma bands (streaming)."""
        key = ("extended_session_vwap_bands", index)
        self._ensure_bound(data)
        if key not in self._cache:
            vwap_val, std_val = self._vwap_core(data, index)
            if vwap_val == ZERO and std_val == ZERO:
                self._cache[key] = volume._EMPTY_VWAP
            else:
                s05 = _quantize(std_val * Decimal("0.5"))
                s15 = _quantize(std_val * Decimal("1.5"))
                s2 = _quantize(std_val * Decimal("2"))
                s3 = _quantize(std_val * Decimal("3"))
                self._cache[key] = volume.ExtendedSessionVWAP(
                    vwap=vwap_val,
                    std_dev=std_val,
                    upper_05=_quantize(vwap_val + s05),
                    lower_05=_quantize(vwap_val - s05),
                    upper_1=_quantize(vwap_val + std_val),
                    lower_1=_quantize(vwap_val - std_val),
                    upper_15=_quantize(vwap_val + s15),
                    lower_15=_quantize(vwap_val - s15),
                    upper_2=_quantize(vwap_val + s2),
                    lower_2=_quantize(vwap_val - s2),
                    upper_3=_quantize(vwap_val + s3),
                    lower_3=_quantize(vwap_val - s3),
                )
        return self._cache[key]

    # ==================================================================
    # Anchored VWAP (event-anchored, persists across sessions)
    # ==================================================================

    def _avwap_core(
        self,
        data: Sequence[PriceData],
        index: int,
        anchor_type: str = "fomc",
    ) -> tuple[Decimal, Decimal]:
        """Streaming anchored VWAP core — returns ``(avwap, std_dev)``."""
        self._ensure_bound(data)
        if anchor_type not in self._s_avwap:
            self._s_avwap[anchor_type] = StreamingAnchoredVWAP(anchor_type)
        return self._s_avwap[anchor_type].update(data, index)

    def anchored_vwap_bands(
        self,
        data: Sequence[PriceData],
        index: int,
        anchor_type: str = "fomc",
    ) -> volume.AnchoredVWAPBands:
        """Anchored VWAP with +-1sigma and +-2sigma bands (streaming).

        Returns :data:`~volume._EMPTY_AVWAP` when no anchor
        has been reached yet.
        """
        key = ("anchored_vwap_bands", index, anchor_type)
        self._ensure_bound(data)
        if key not in self._cache:
            avwap_val, std_val = self._avwap_core(data, index, anchor_type)

            # Get metadata from the streaming accumulator
            acc = self._s_avwap[anchor_type]
            if not acc.valid or avwap_val == ZERO:
                self._cache[key] = volume._EMPTY_AVWAP
            else:
                from stockdownloader.util.config import days_since_anchor

                anchor_date = acc.current_anchor
                trading_date = data[index].date[:10]
                days = days_since_anchor(trading_date, anchor_type)
                if days is None:
                    days = 0

                s1 = std_val
                s2 = _quantize(std_val * Decimal("2"))

                self._cache[key] = volume.AnchoredVWAPBands(
                    avwap=avwap_val,
                    std_dev=std_val,
                    upper_1=_quantize(avwap_val + s1),
                    lower_1=_quantize(avwap_val - s1),
                    upper_2=_quantize(avwap_val + s2),
                    lower_2=_quantize(avwap_val - s2),
                    anchor_date=anchor_date,
                    days_since_anchor=days,
                    valid=True,
                )
        return self._cache[key]

    # ==================================================================
    # Intraday indicators
    # ==================================================================

    def tod_rvol(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback_days: int = 10,
        bars_per_day: int = 78,
    ) -> Decimal:
        """Time-of-day relative volume."""
        return self._get(
            ("tod_rvol", index, lookback_days, bars_per_day),
            data, ii.tod_rvol, data, index, lookback_days, bars_per_day,
        )

    def cvd_session(
        self,
        data: Sequence[PriceData],
        index: int,
    ) -> Decimal:
        """Cumulative Volume Delta for current session (streaming)."""
        key = ("cvd_session", index)
        self._ensure_bound(data)
        if key not in self._cache:
            if self._s_cvd is None:
                self._s_cvd = StreamingCVD()
            self._cache[key] = self._s_cvd.update(data, index)
        return self._cache[key]

    def cvd_normalized(
        self,
        data: Sequence[PriceData],
        index: int,
        vol_sma_period: int = 20,
    ) -> Decimal:
        """Normalized CVD (routes through streaming CVD)."""
        key = ("cvd_normalized", index, vol_sma_period)
        self._ensure_bound(data)
        if key not in self._cache:
            cvd = self.cvd_session(data, index)
            if index < vol_sma_period:
                self._cache[key] = ZERO
            else:
                total = ZERO
                for i in range(index - vol_sma_period + 1, index + 1):
                    total += Decimal(str(data[i].volume))
                vol_sma = total / Decimal(str(vol_sma_period))
                denom = vol_sma * Decimal("20")
                if denom <= ZERO:
                    self._cache[key] = ZERO
                else:
                    self._cache[key] = _quantize(cvd / denom)
        return self._cache[key]

    def linear_regression_slope(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 15,
    ) -> Decimal:
        """Linear regression slope of close prices."""
        return self._get(
            ("linear_regression_slope", index, period),
            data, ii.linear_regression_slope, data, index, period,
        )

    def lrs_normalized(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 15,
        atr_period: int = 14,
    ) -> Decimal:
        """LRS normalized by ATR (routes ATR through cache)."""
        key = ("lrs_normalized", index, period, atr_period)
        self._ensure_bound(data)
        if key not in self._cache:
            self._cache[key] = ii.lrs_normalized(
                data, index, period, atr_period,
                _atr_fn=lambda d, i, p: self.atr(d, i, p),
            )
        return self._cache[key]

    def vwap_slope(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback: int = 5,
    ) -> Decimal:
        """VWAP slope (routes session VWAP lookups through cache)."""
        key = ("vwap_slope", index, lookback)
        self._ensure_bound(data)
        if key not in self._cache:
            self._cache[key] = ii.vwap_slope(
                data, index, lookback,
                _vwap_fn=lambda d, i: self.session_vwap(d, i),
            )
        return self._cache[key]

    def vwap_acceleration(
        self,
        data: Sequence[PriceData],
        index: int,
        lookback: int = 5,
        atr_period: int = 14,
    ) -> Decimal:
        """VWAP acceleration (routes ATR and slope through cache)."""
        key = ("vwap_acceleration", index, lookback, atr_period)
        self._ensure_bound(data)
        if key not in self._cache:
            self._cache[key] = ii.vwap_acceleration(
                data, index, lookback, atr_period,
                _atr_fn=lambda d, i, p: self.atr(d, i, p),
                _slope_fn=lambda d, i, lb: self.vwap_slope(d, i, lb),
            )
        return self._cache[key]

    # ==================================================================
    # Higher-timeframe EMA trend
    # ==================================================================

    def htf_ema_trend(
        self,
        data: Sequence[PriceData],
        index: int,
        fast_period: int = 9,
        slow_period: int = 21,
        htf_factor: int = 3,
    ) -> int:
        """Higher-timeframe EMA trend (+1 bullish, -1 bearish, 0 neutral).

        Uses streaming HTF resampling so the resample is O(1) per bar
        instead of rebuilding all session candles from scratch.
        """
        key = ("htf_ema_trend", index, fast_period, slow_period, htf_factor)
        self._ensure_bound(data)
        if key not in self._cache:
            if htf_factor not in self._s_htf:
                self._s_htf[htf_factor] = StreamingHTFResample(htf_factor)
            htf_bars = self._s_htf[htf_factor].update(data, index)
            if len(htf_bars) < slow_period + 1:
                self._cache[key] = 0
            else:
                htf_idx = len(htf_bars) - 1
                fast_val = _ema(htf_bars, htf_idx, fast_period)
                slow_val = _ema(htf_bars, htf_idx, slow_period)
                if fast_val > slow_val:
                    self._cache[key] = 1
                elif fast_val < slow_val:
                    self._cache[key] = -1
                else:
                    self._cache[key] = 0
        return self._cache[key]

    # ==================================================================
    # Relative volume
    # ==================================================================

    def rel_vol(
        self,
        data: Sequence[PriceData],
        index: int,
        period: int = 20,
    ) -> Decimal:
        """Relative volume (current / SMA of volume)."""
        return self._get(
            ("rel_vol", index, period),
            data, ii.rel_vol, data, index, period,
        )

    # ==================================================================
    # Market Structure (SMC)
    # ==================================================================

    def structure_state(
        self,
        data: Sequence[PriceData],
        index: int,
        atr_val: Decimal,
        lookback: int = 5,
        min_impulse: float = 2.0,
        zone_bars: int = 2,
    ):
        """Market structure state (streaming).

        Returns a :class:`~indicators.smc.StructureState` with swing levels,
        BoS flags, and supply/demand zones.
        """
        from stockdownloader.util.indicators.smc import StructureState

        key = ("structure_state", index, lookback, min_impulse, zone_bars)
        self._ensure_bound(data)
        if key not in self._cache:
            params = (lookback, min_impulse, zone_bars)
            if params not in self._s_structure:
                self._s_structure[params] = StreamingStructureTracker(
                    lookback=lookback,
                    min_impulse_atr=min_impulse,
                    zone_bars=zone_bars,
                )
            self._cache[key] = self._s_structure[params].update(
                data, index, atr_val,
            )
        return self._cache[key]
