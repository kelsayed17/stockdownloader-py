"""Tests for pattern_encoder — bar-to-feature encoding."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from stockdownloader.analysis.pattern_encoder import (
    BarEncoder,
    BarFeatures,
    PatternContext,
)


def _make_bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 1000,
    date: str = "2025-01-15 10:30:00-05:00",
) -> MagicMock:
    """Create a mock IntradayPriceData bar."""
    bar = MagicMock()
    bar.open = Decimal(str(open_))
    bar.high = Decimal(str(high))
    bar.low = Decimal(str(low))
    bar.close = Decimal(str(close))
    bar.volume = volume
    bar.date = date
    bar.datetime_parsed = MagicMock()
    bar.datetime_parsed.weekday.return_value = 2  # Wednesday
    return bar


def _make_hub(atr: float = 1.0, rel_vol: float = 1.0, ema_fast: float = 500.0, ema_slow: float = 500.0):
    """Create a mock IndicatorHub."""
    hub = MagicMock()
    hub.atr.return_value = Decimal(str(atr))
    hub.rel_vol.return_value = Decimal(str(rel_vol))
    hub.ema.side_effect = lambda data, idx, period: (
        Decimal(str(ema_fast)) if period == 9 else Decimal(str(ema_slow))
    )
    hub.session_vwap.return_value = Decimal("500.0")
    hub.rsi.return_value = Decimal("50.0")
    return hub


# ======================================================================
# BarFeatures construction
# ======================================================================


class TestBarFeatures:
    def test_frozen(self):
        bf = BarFeatures("bull", "strong", "no_wick", "normal", "normal")
        with pytest.raises(AttributeError):
            bf.body_type = "bear"  # type: ignore[misc]

    def test_hashable(self):
        bf1 = BarFeatures("bull", "strong", "no_wick", "normal", "normal")
        bf2 = BarFeatures("bull", "strong", "no_wick", "normal", "normal")
        assert bf1 == bf2
        assert hash(bf1) == hash(bf2)

    def test_usable_as_dict_key(self):
        bf = BarFeatures("bull", "strong", "no_wick", "normal", "normal")
        d = {bf: 42}
        assert d[bf] == 42

    def test_tuple_of_features_hashable(self):
        bf1 = BarFeatures("bull", "strong", "no_wick", "normal", "normal")
        bf2 = BarFeatures("bear", "weak", "upper_wick", "large", "high")
        key = (bf1, bf2)
        d = {key: "pattern"}
        assert d[key] == "pattern"


# ======================================================================
# Body type encoding
# ======================================================================


class TestBodyType:
    def test_bullish(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # close > open, body_ratio = 0.5 (> 0.15)
        bar = _make_bar(100.0, 101.0, 99.0, 101.0)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.body_type == "bull"

    def test_bearish(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # close < open, body_ratio = 0.5
        bar = _make_bar(101.0, 101.0, 99.0, 99.0)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.body_type == "bear"

    def test_doji(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # body_ratio = 0.1 (< 0.15)
        bar = _make_bar(100.0, 101.0, 99.0, 100.2)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.body_type == "doji"

    def test_zero_range(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 100.0, 100.0, 100.0)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.body_type == "doji"


# ======================================================================
# Body strength encoding
# ======================================================================


class TestBodyStrength:
    def test_strong(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # body = 0.8, ATR = 1.0 → body_atr = 0.8 (>= 0.6)
        bar = _make_bar(100.0, 101.0, 99.2, 100.8)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.body_strength == "strong"

    def test_moderate(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # body = 0.4, ATR = 1.0 → body_atr = 0.4 (0.3-0.6)
        bar = _make_bar(100.0, 101.0, 99.0, 100.4)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.body_strength == "moderate"

    def test_weak(self):
        hub = _make_hub(atr=2.0)
        encoder = BarEncoder(hub)
        # body = 0.4, ATR = 2.0 → body_atr = 0.2 (< 0.3)
        bar = _make_bar(100.0, 101.0, 99.0, 100.4)
        features = encoder.encode_bar(bar, Decimal("2.0"))
        assert features.body_strength == "weak"


# ======================================================================
# Wick signal encoding
# ======================================================================


class TestWickSignal:
    def test_lower_wick(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # Long lower wick: open=close near high, low far below
        bar = _make_bar(100.8, 101.0, 99.0, 100.9)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.wick_signal == "lower_wick"

    def test_upper_wick(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # Long upper wick: open=close near low, high far above
        bar = _make_bar(99.1, 101.0, 99.0, 99.2)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.wick_signal == "upper_wick"

    def test_both_wick(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # Both wicks large, small body in middle
        bar = _make_bar(100.0, 101.0, 99.0, 100.1)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.wick_signal == "both_wick"

    def test_no_wick(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # Strong body, little wick
        bar = _make_bar(99.1, 101.0, 99.0, 100.9)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.wick_signal == "no_wick"


# ======================================================================
# Relative size encoding
# ======================================================================


class TestRelativeSize:
    def test_tiny(self):
        hub = _make_hub(atr=5.0)
        encoder = BarEncoder(hub)
        # range = 1.0, ATR = 5.0 → 0.2 (< 0.3)
        bar = _make_bar(100.0, 100.5, 99.5, 100.3)
        features = encoder.encode_bar(bar, Decimal("5.0"))
        assert features.relative_size == "tiny"

    def test_small(self):
        hub = _make_hub(atr=2.0)
        encoder = BarEncoder(hub)
        # range = 1.0, ATR = 2.0 → 0.5 (0.3-0.7)
        bar = _make_bar(100.0, 100.5, 99.5, 100.3)
        features = encoder.encode_bar(bar, Decimal("2.0"))
        assert features.relative_size == "small"

    def test_normal(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        # range = 1.0, ATR = 1.0 → 1.0 (0.7-1.3)
        bar = _make_bar(100.0, 100.5, 99.5, 100.3)
        features = encoder.encode_bar(bar, Decimal("1.0"))
        assert features.relative_size == "normal"

    def test_large(self):
        hub = _make_hub(atr=0.6)
        encoder = BarEncoder(hub)
        # range = 1.0, ATR = 0.6 → 1.67 (1.3-2.0)
        bar = _make_bar(100.0, 100.5, 99.5, 100.3)
        features = encoder.encode_bar(bar, Decimal("0.6"))
        assert features.relative_size == "large"

    def test_huge(self):
        hub = _make_hub(atr=0.4)
        encoder = BarEncoder(hub)
        # range = 1.0, ATR = 0.4 → 2.5 (>= 2.0)
        bar = _make_bar(100.0, 100.5, 99.5, 100.3)
        features = encoder.encode_bar(bar, Decimal("0.4"))
        assert features.relative_size == "huge"


# ======================================================================
# Volume profile encoding
# ======================================================================


class TestVolumeProfile:
    def test_low_volume(self):
        hub = _make_hub(atr=1.0, rel_vol=0.4)
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        features = encoder.encode([bar], 0)
        assert features.volume_profile == "low"

    def test_normal_volume(self):
        hub = _make_hub(atr=1.0, rel_vol=1.0)
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        features = encoder.encode([bar], 0)
        assert features.volume_profile == "normal"

    def test_high_volume(self):
        hub = _make_hub(atr=1.0, rel_vol=2.0)
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        features = encoder.encode([bar], 0)
        assert features.volume_profile == "high"

    def test_extreme_volume(self):
        hub = _make_hub(atr=1.0, rel_vol=3.0)
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        features = encoder.encode([bar], 0)
        assert features.volume_profile == "extreme"

    def test_no_data_defaults_normal(self):
        hub = _make_hub(atr=1.0)
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        # Call encode_bar without data → defaults to normal
        features = encoder.encode_bar(bar, Decimal("1.0"), data=None)
        assert features.volume_profile == "normal"


# ======================================================================
# PatternContext encoding
# ======================================================================


class TestPatternContext:
    def test_frozen(self):
        ctx = PatternContext("open", 0, "weak_trend", 1)
        with pytest.raises(AttributeError):
            ctx.time_bucket = "morning"  # type: ignore[misc]

    def test_time_bucket_open(self):
        hub = _make_hub()
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.time_bucket == "open"

    def test_time_bucket_morning(self):
        hub = _make_hub()
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=20)
        assert ctx.time_bucket == "morning"

    def test_time_bucket_midday(self):
        hub = _make_hub()
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=40)
        assert ctx.time_bucket == "midday"

    def test_time_bucket_afternoon(self):
        hub = _make_hub()
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=60)
        assert ctx.time_bucket == "afternoon"

    def test_day_of_week(self):
        hub = _make_hub()
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=1)
        assert ctx.day_of_week == 2  # Wednesday (mock)

    def test_trend_up(self):
        hub = _make_hub(ema_fast=510.0, ema_slow=500.0)
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=1)
        assert ctx.trend_dir == 1

    def test_trend_down(self):
        hub = _make_hub(ema_fast=490.0, ema_slow=500.0)
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=1)
        assert ctx.trend_dir == -1

    def test_regime_passed(self):
        from stockdownloader.strategy.regime.regime_detector import MarketRegime
        hub = _make_hub()
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context(
            [bar], 0, bar_of_day=1,
            regime=MarketRegime.HIGH_VOLATILITY,
        )
        assert ctx.regime == "high_volatility"

    def test_regime_none(self):
        hub = _make_hub()
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=1)
        assert ctx.regime == "unknown"


# ======================================================================
# Indicator snapshot fields in PatternContext
# ======================================================================


class TestPatternContextIndicators:
    """Tests for the 5 new indicator snapshot fields on PatternContext."""

    def test_backward_compat_defaults_none(self):
        """Old-style 4-arg construction should leave indicator fields None."""
        ctx = PatternContext("morning", 2, "unknown", 1)
        assert ctx.rsi_zone is None
        assert ctx.vwap_position is None
        assert ctx.obv_trend is None
        assert ctx.adx_level is None
        assert ctx.macd_signal is None

    def test_explicit_indicator_fields(self):
        """All 5 indicator fields can be set explicitly."""
        ctx = PatternContext(
            "open", 0, "unknown", 1,
            rsi_zone="oversold",
            vwap_position="below",
            obv_trend="falling",
            adx_level="weak",
            macd_signal="bearish",
        )
        assert ctx.rsi_zone == "oversold"
        assert ctx.vwap_position == "below"
        assert ctx.obv_trend == "falling"
        assert ctx.adx_level == "weak"
        assert ctx.macd_signal == "bearish"

    def test_rsi_zone_oversold(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("25.0")  # < 30
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.rsi_zone == "oversold"

    def test_rsi_zone_overbought(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("75.0")  # > 70
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.rsi_zone == "overbought"

    def test_rsi_zone_neutral(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("50.0")  # 30-70
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.rsi_zone == "neutral"

    def test_vwap_position_above(self):
        hub = _make_hub()
        hub.session_vwap.return_value = Decimal("99.0")  # close=100.5 > vwap=99.0
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.vwap_position == "above"

    def test_vwap_position_below(self):
        hub = _make_hub()
        hub.session_vwap.return_value = Decimal("102.0")  # close=100.5 < vwap=102.0
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.vwap_position == "below"

    def test_vwap_position_at(self):
        hub = _make_hub()
        # close=100.5, vwap=100.5 → delta_pct = 0 < 0.001 → "at"
        hub.session_vwap.return_value = Decimal("100.5")
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.vwap_position == "at"

    def test_obv_trend_rising(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.obv_trend == "rising"

    def test_obv_trend_falling(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = False
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.obv_trend == "falling"

    def test_adx_level_weak(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("15.0"))  # < 20
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.adx_level == "weak"

    def test_adx_level_moderate(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("30.0"))  # 20-40
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.adx_level == "moderate"

    def test_adx_level_strong(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("45.0"))  # > 40
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.adx_level == "strong"

    def test_macd_signal_bullish(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")  # > 0
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.macd_signal == "bullish"

    def test_macd_signal_bearish(self):
        hub = _make_hub()
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("-0.5")  # < 0
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.macd_signal == "bearish"

    def test_indicator_error_leaves_none(self):
        """If an indicator raises, its field should remain None."""
        hub = _make_hub()
        hub.rsi.side_effect = ValueError("No data")
        hub.adx.side_effect = ValueError("No data")
        hub.is_obv_rising.side_effect = ValueError("No data")
        hub.macd_histogram.side_effect = ValueError("No data")
        hub.session_vwap.side_effect = ValueError("No data")
        hub.htf_ema_trend.side_effect = ValueError("No data")
        hub.cvd_normalized.side_effect = ValueError("No data")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.rsi_zone is None
        assert ctx.vwap_position is None
        assert ctx.obv_trend is None
        assert ctx.adx_level is None
        assert ctx.macd_signal is None
        assert ctx.htf_trend is None
        assert ctx.cvd_direction is None


# ======================================================================
# HTF trend + CVD direction context fields
# ======================================================================


class TestPatternContextHTFAndCVD:
    """Tests for htf_trend and cvd_direction fields on PatternContext."""

    def test_backward_compat_defaults_none(self):
        """Old-style construction without HTF/CVD fields defaults to None."""
        ctx = PatternContext("morning", 2, "unknown", 1)
        assert ctx.htf_trend is None
        assert ctx.cvd_direction is None

    def test_htf_trend_bullish(self):
        hub = _make_hub()
        hub.htf_ema_trend.return_value = 1
        hub.cvd_normalized.return_value = Decimal("0.0")
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.htf_trend == 1

    def test_htf_trend_bearish(self):
        hub = _make_hub()
        hub.htf_ema_trend.return_value = -1
        hub.cvd_normalized.return_value = Decimal("0.0")
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.htf_trend == -1

    def test_htf_trend_error_leaves_none(self):
        hub = _make_hub()
        hub.htf_ema_trend.side_effect = ValueError("Not enough data")
        hub.cvd_normalized.return_value = Decimal("0.0")
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.htf_trend is None

    def test_cvd_direction_buying(self):
        hub = _make_hub()
        hub.htf_ema_trend.return_value = 0
        hub.cvd_normalized.return_value = Decimal("0.5")  # > 0.3
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.cvd_direction == "buying"

    def test_cvd_direction_selling(self):
        hub = _make_hub()
        hub.htf_ema_trend.return_value = 0
        hub.cvd_normalized.return_value = Decimal("-0.5")  # < -0.3
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.cvd_direction == "selling"

    def test_cvd_direction_neutral(self):
        hub = _make_hub()
        hub.htf_ema_trend.return_value = 0
        hub.cvd_normalized.return_value = Decimal("0.1")  # between -0.3 and 0.3
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.cvd_direction == "neutral"

    def test_cvd_direction_error_leaves_none(self):
        hub = _make_hub()
        hub.htf_ema_trend.return_value = 0
        hub.cvd_normalized.side_effect = ValueError("Not enough data")
        hub.rsi.return_value = Decimal("50.0")
        hub.adx.return_value = MagicMock(adx=Decimal("25.0"))
        hub.is_obv_rising.return_value = True
        hub.macd_histogram.return_value = Decimal("0.5")
        encoder = BarEncoder(hub)
        bar = _make_bar(100.0, 101.0, 99.0, 100.5)
        ctx = encoder.encode_context([bar], 0, bar_of_day=5)
        assert ctx.cvd_direction is None
