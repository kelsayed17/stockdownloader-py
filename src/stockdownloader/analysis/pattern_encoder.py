"""Bar-to-feature encoding for price action pattern discovery.

Converts each OHLCV bar into a compact categorical representation
(``BarFeatures``) that captures body type, body strength, wick
dominance, relative size, and volume profile.  Also encodes market
context (time of day, regime, trend) as ``PatternContext``.

The encoder delegates to :class:`IndicatorHub` for ATR, RSI, EMA, and
relative volume, getting O(1) streaming performance per bar.

Usage::

    from stockdownloader.analysis.pattern_encoder import BarEncoder, BarFeatures
    from stockdownloader.indicators.hub import IndicatorHub

    hub = IndicatorHub()
    encoder = BarEncoder(hub)
    features = encoder.encode(data, index)
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from stockdownloader.core.math import ZERO

if TYPE_CHECKING:
    from stockdownloader.core.models.price import IntradayPriceData
    from stockdownloader.strategies.regime.detector import MarketRegime
    from stockdownloader.indicators.hub import IndicatorHub


# =========================================================================
# Data structures
# =========================================================================


@dataclass(frozen=True, slots=True)
class BarFeatures:
    """Categorical encoding of a single bar's price action.

    Five features with 3-5 bins each produce ~720 theoretical bar types.
    In practice most bars cluster into a much smaller effective set.
    """

    body_type: str
    """``"bull"`` | ``"bear"`` | ``"doji"`` — direction of the bar body."""

    body_strength: str
    """``"strong"`` | ``"moderate"`` | ``"weak"`` — body size relative to ATR."""

    wick_signal: str
    """``"upper_wick"`` | ``"lower_wick"`` | ``"both_wick"`` | ``"no_wick"``."""

    relative_size: str
    """``"tiny"`` | ``"small"`` | ``"normal"`` | ``"large"`` | ``"huge"``
    — bar range relative to ATR."""

    volume_profile: str
    """``"low"`` | ``"normal"`` | ``"high"`` | ``"extreme"``
    — volume relative to 20-bar average."""


@dataclass(frozen=True, slots=True)
class PatternContext:
    """Market context at the time a pattern was observed.

    Stored alongside each pattern occurrence to enable context-aware
    filtering (e.g. patterns that only work in the morning, or only
    in trending markets).

    The indicator snapshot fields (``rsi_zone`` … ``macd_signal``) are
    optional — ``None`` means the indicator was not computed (e.g. when
    mining with insufficient warmup or when loading old catalogs).
    """

    time_bucket: str
    """``"open"`` | ``"morning"`` | ``"midday"`` | ``"afternoon"``."""

    day_of_week: int
    """0 = Monday … 4 = Friday."""

    regime: str
    """Market regime value from :class:`MarketRegime` enum."""

    trend_dir: int
    """EMA fast vs slow: +1 (uptrend), 0 (neutral), -1 (downtrend)."""

    # ── Indicator snapshots (for confirmation analysis) ──────────

    rsi_zone: str | None = None
    """``"oversold"`` (<30) | ``"neutral"`` (30-70) | ``"overbought"`` (>70)."""

    vwap_position: str | None = None
    """``"above"`` | ``"at"`` | ``"below"`` — close vs session VWAP."""

    obv_trend: str | None = None
    """``"rising"`` | ``"falling"`` — OBV direction over last 5 bars."""

    adx_level: str | None = None
    """``"weak"`` (<20) | ``"moderate"`` (20-40) | ``"strong"`` (>40)."""

    macd_signal: str | None = None
    """``"bullish"`` (histogram > 0) | ``"bearish"`` (histogram < 0)."""

    htf_trend: int | None = None
    """Higher-timeframe EMA trend: +1 (up), -1 (down), 0 (neutral)."""

    cvd_direction: str | None = None
    """``"buying"`` | ``"selling"`` | ``"neutral"`` — CVD normalized."""


# =========================================================================
# Encoding thresholds
# =========================================================================

# Body type: body_ratio = abs(close - open) / (high - low)
_DOJI_THRESHOLD = Decimal("0.15")

# Body strength: body / ATR(14)
_STRONG_BODY = Decimal("0.6")
_MODERATE_BODY = Decimal("0.3")

# Wick ratios: wick / (high - low)
_WICK_DOMINANT = Decimal("0.5")
_WICK_PRESENT = Decimal("0.3")

# Relative size: (high - low) / ATR(14)
_SIZE_TINY = Decimal("0.3")
_SIZE_SMALL = Decimal("0.7")
_SIZE_NORMAL_MAX = Decimal("1.3")
_SIZE_LARGE = Decimal("2.0")

# Volume profile: current_vol / SMA(vol, 20)
_VOL_LOW = Decimal("0.6")
_VOL_HIGH = Decimal("1.4")
_VOL_EXTREME = Decimal("2.5")

# Time bucket thresholds (bar-of-day, 1-indexed, 78 bars/day)
_OPEN_END = 12        # first hour (bars 1-12)
_MORNING_END = 30     # 10:30-12:00 (bars 13-30)
_MIDDAY_END = 54      # 12:00-2:30 (bars 31-54)
# Afternoon = bars 55-78

# Indicator classification thresholds
_RSI_OVERSOLD = Decimal("30")
_RSI_OVERBOUGHT = Decimal("70")
_ADX_WEAK = Decimal("20")
_ADX_STRONG = Decimal("40")
_VWAP_AT_TOLERANCE = Decimal("0.001")  # 0.1% tolerance for "at VWAP"
_CVD_THRESHOLD = Decimal("0.3")  # |CVD normalized| > 0.3 = directional


# =========================================================================
# Encoder
# =========================================================================


class BarEncoder:
    """Encodes :class:`IntradayPriceData` bars into categorical
    :class:`BarFeatures`.

    Stateless — all indicator state lives in the :class:`IndicatorHub`.
    Call :meth:`encode` sequentially for streaming performance.

    Parameters
    ----------
    hub:
        Shared indicator hub providing ATR and relative volume.
    atr_period:
        ATR lookback period (default 14).
    vol_period:
        Volume SMA lookback period for relative volume (default 20).
    """

    def __init__(
        self,
        hub: IndicatorHub,
        *,
        atr_period: int = 14,
        vol_period: int = 20,
    ) -> None:
        self._hub = hub
        self._atr_period = atr_period
        self._vol_period = vol_period

    def encode(
        self,
        data: list[IntradayPriceData],
        index: int,
    ) -> BarFeatures:
        """Encode the bar at *index* into categorical features.

        Parameters
        ----------
        data:
            Full price data (passed to hub for streaming indicators).
        index:
            Current bar index to encode.

        Returns
        -------
        Frozen :class:`BarFeatures` suitable for use as a hashable key.
        """
        bar = data[index]
        atr_val = self._hub.atr(data, index, self._atr_period)

        return self.encode_bar(bar, atr_val, data, index)

    def encode_bar(
        self,
        bar: IntradayPriceData,
        atr_val: Decimal,
        data: list[IntradayPriceData] | None = None,
        index: int = 0,
    ) -> BarFeatures:
        """Encode a single bar given pre-computed ATR.

        Parameters
        ----------
        bar:
            The OHLCV bar to encode.
        atr_val:
            Pre-computed ATR value at this bar.
        data:
            Optional full data for relative volume computation.
            If ``None``, volume_profile defaults to ``"normal"``.
        index:
            Bar index for relative volume lookup.

        Returns
        -------
        Frozen :class:`BarFeatures`.
        """
        bar_range = bar.high - bar.low

        # ── Body type ─────────────────────────────────────────────────
        if bar_range <= ZERO:
            body_type = "doji"
            body_strength = "weak"
            wick_signal = "no_wick"
        else:
            body = abs(bar.close - bar.open)
            body_ratio = body / bar_range

            if body_ratio < _DOJI_THRESHOLD:
                body_type = "doji"
            elif bar.close > bar.open:
                body_type = "bull"
            else:
                body_type = "bear"

            # ── Body strength ─────────────────────────────────────────
            if atr_val > ZERO:
                body_atr = body / atr_val
            else:
                body_atr = ZERO

            if body_atr >= _STRONG_BODY:
                body_strength = "strong"
            elif body_atr >= _MODERATE_BODY:
                body_strength = "moderate"
            else:
                body_strength = "weak"

            # ── Wick signal ───────────────────────────────────────────
            upper_wick = bar.high - max(bar.close, bar.open)
            lower_wick = min(bar.close, bar.open) - bar.low
            upper_ratio = upper_wick / bar_range
            lower_ratio = lower_wick / bar_range

            if upper_ratio >= _WICK_DOMINANT and lower_ratio < _WICK_PRESENT:
                wick_signal = "upper_wick"
            elif lower_ratio >= _WICK_DOMINANT and upper_ratio < _WICK_PRESENT:
                wick_signal = "lower_wick"
            elif upper_ratio >= _WICK_PRESENT and lower_ratio >= _WICK_PRESENT:
                wick_signal = "both_wick"
            else:
                wick_signal = "no_wick"

        # ── Relative size ─────────────────────────────────────────────
        if atr_val > ZERO:
            size_ratio = bar_range / atr_val
        else:
            size_ratio = Decimal("1.0")

        if size_ratio < _SIZE_TINY:
            relative_size = "tiny"
        elif size_ratio < _SIZE_SMALL:
            relative_size = "small"
        elif size_ratio < _SIZE_NORMAL_MAX:
            relative_size = "normal"
        elif size_ratio < _SIZE_LARGE:
            relative_size = "large"
        else:
            relative_size = "huge"

        # ── Volume profile ────────────────────────────────────────────
        if data is not None:
            rel_vol = self._hub.rel_vol(data, index, self._vol_period)
        else:
            rel_vol = Decimal("1.0")

        if rel_vol < _VOL_LOW:
            volume_profile = "low"
        elif rel_vol < _VOL_HIGH:
            volume_profile = "normal"
        elif rel_vol < _VOL_EXTREME:
            volume_profile = "high"
        else:
            volume_profile = "extreme"

        return BarFeatures(
            body_type=body_type,
            body_strength=body_strength,
            wick_signal=wick_signal,
            relative_size=relative_size,
            volume_profile=volume_profile,
        )

    def encode_context(
        self,
        data: list[IntradayPriceData],
        index: int,
        bar_of_day: int,
        *,
        regime: MarketRegime | None = None,
    ) -> PatternContext:
        """Encode the market context around bar *index*.

        Parameters
        ----------
        data:
            Full price data.
        index:
            Current bar index.
        bar_of_day:
            1-indexed position within the trading day.
        regime:
            Pre-classified market regime, or ``None`` (defaults to
            ``"unknown"``).

        Returns
        -------
        Frozen :class:`PatternContext`.
        """
        # ── Time bucket ───────────────────────────────────────────────
        if bar_of_day <= _OPEN_END:
            time_bucket = "open"
        elif bar_of_day <= _MORNING_END:
            time_bucket = "morning"
        elif bar_of_day <= _MIDDAY_END:
            time_bucket = "midday"
        else:
            time_bucket = "afternoon"

        # ── Day of week ───────────────────────────────────────────────
        bar = data[index]
        try:
            dt = bar.datetime_parsed
            dow = dt.weekday()
        except (ValueError, AttributeError):
            dow = 0

        # ── Trend direction ───────────────────────────────────────────
        ema_fast = self._hub.ema(data, index, 9)
        ema_slow = self._hub.ema(data, index, 21)
        if ema_fast > ema_slow:
            trend_dir = 1
        elif ema_fast < ema_slow:
            trend_dir = -1
        else:
            trend_dir = 0

        # ── Regime ────────────────────────────────────────────────────
        regime_str = regime.value if regime is not None else "unknown"

        # ── Indicator snapshots ──────────────────────────────────────
        rsi_zone: str | None = None
        vwap_position: str | None = None
        obv_trend: str | None = None
        adx_level: str | None = None
        macd_signal: str | None = None

        try:
            rsi_val = self._hub.rsi(data, index, 14)
            if rsi_val < _RSI_OVERSOLD:
                rsi_zone = "oversold"
            elif rsi_val > _RSI_OVERBOUGHT:
                rsi_zone = "overbought"
            else:
                rsi_zone = "neutral"
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
            pass

        try:
            vwap_val = self._hub.session_vwap(data, index)
            if vwap_val > ZERO:
                delta_pct = abs(bar.close - vwap_val) / vwap_val
                if delta_pct < _VWAP_AT_TOLERANCE:
                    vwap_position = "at"
                elif bar.close > vwap_val:
                    vwap_position = "above"
                else:
                    vwap_position = "below"
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
            pass

        try:
            obv_rising = self._hub.is_obv_rising(data, index, 5)
            obv_trend = "rising" if obv_rising else "falling"
        except (ValueError, TypeError, KeyError, IndexError):
            pass

        try:
            adx_result = self._hub.adx(data, index, 14)
            adx_val = adx_result.adx
            if adx_val < _ADX_WEAK:
                adx_level = "weak"
            elif adx_val > _ADX_STRONG:
                adx_level = "strong"
            else:
                adx_level = "moderate"
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
            pass

        try:
            macd_hist = self._hub.macd_histogram(data, index)
            macd_signal = "bullish" if macd_hist > ZERO else "bearish"
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
            pass

        # ── HTF trend ────────────────────────────────────────────────
        htf_trend: int | None = None
        try:
            htf_trend = self._hub.htf_ema_trend(data, index)
        except (ValueError, TypeError, KeyError, IndexError):
            pass

        # ── CVD direction ────────────────────────────────────────────
        cvd_direction: str | None = None
        try:
            cvd_val = self._hub.cvd_normalized(data, index)
            if cvd_val > _CVD_THRESHOLD:
                cvd_direction = "buying"
            elif cvd_val < -_CVD_THRESHOLD:
                cvd_direction = "selling"
            else:
                cvd_direction = "neutral"
        except (ValueError, TypeError, KeyError, IndexError):
            pass

        return PatternContext(
            time_bucket=time_bucket,
            day_of_week=dow,
            regime=regime_str,
            trend_dir=trend_dir,
            rsi_zone=rsi_zone,
            vwap_position=vwap_position,
            obv_trend=obv_trend,
            adx_level=adx_level,
            macd_signal=macd_signal,
            htf_trend=htf_trend,
            cvd_direction=cvd_direction,
        )
