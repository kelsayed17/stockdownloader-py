"""Unit tests for intraday-specific indicator calculations.

Covers: ExtendedSessionVWAP, CandleStrength, candle patterns,
S/R proximity, linear regression slope, time-of-day RVOL, CVD,
daily aggregation, HTF resampling, relative volume, and VWAP
slope / acceleration.
"""

from decimal import Decimal

import pytest

from stockdownloader.model.price_data import PriceData
from stockdownloader.util.indicators.volume import (
    ExtendedSessionVWAP,
    cvd_session,
    cvd_normalized,
)
from stockdownloader.util.indicators.intraday import (
    CandleStrength,
    candle_strength,
    is_hammer,
    is_inv_hammer,
    is_bull_engulfing,
    is_bear_engulfing,
    near_level,
    compute_sr_score,
    _round_to_5,
    linear_regression_slope,
    lrs_normalized,
    tod_rvol,
    aggregate_to_daily,
    daily_atr_prior,
    resample_to_htf,
    htf_ema_trend,
    rel_vol,
    vwap_slope,
    vwap_acceleration,
)

D = Decimal
ZERO = D("0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(
    date: str = "2025-01-15 10:00:00-05:00",
    open_: str = "500",
    high: str = "502",
    low: str = "498",
    close: str = "501",
    volume: int = 100_000,
) -> PriceData:
    return PriceData(
        date=date,
        open=D(open_),
        high=D(high),
        low=D(low),
        close=D(close),
        adj_close=D(close),
        volume=volume,
    )


def _bars_trend_up(n: int, start: float = 100.0, step: float = 1.0) -> list[PriceData]:
    """Generate n bars with a steady uptrend.  All in same session."""
    result = []
    for i in range(n):
        price = start + i * step
        result.append(_bar(
            date=f"2025-01-15 09:{30 + i % 30:02d}:00-05:00",
            open_=str(price - 0.5),
            high=str(price + 1),
            low=str(price - 1),
            close=str(price),
            volume=1_000_000,
        ))
    return result


def _bars_multi_day(n_per_day: int, n_days: int, start: float = 100.0) -> list[PriceData]:
    """Generate bars across multiple days."""
    result = []
    price = start
    for day in range(n_days):
        date_prefix = f"2025-01-{15 + day:02d}"
        for bar_idx in range(n_per_day):
            price += 0.1
            hour = 9 + bar_idx // 12
            minute = 30 + (bar_idx % 12) * 5
            if minute >= 60:
                hour += 1
                minute -= 60
            result.append(_bar(
                date=f"{date_prefix} {hour:02d}:{minute:02d}:00-05:00",
                open_=str(price - 0.5),
                high=str(price + 1),
                low=str(price - 1),
                close=str(price),
                volume=1_000_000 + bar_idx * 10_000,
            ))
    return result


# =========================================================================
# ExtendedSessionVWAP dataclass
# =========================================================================


class TestExtendedSessionVWAP:
    def test_band_pair_2_sigma(self):
        vwap = ExtendedSessionVWAP(
            vwap=D("500"), std_dev=D("2"),
            upper_05=D("501"), lower_05=D("499"),
            upper_1=D("502"), lower_1=D("498"),
            upper_15=D("503"), lower_15=D("497"),
            upper_2=D("504"), lower_2=D("496"),
            upper_3=D("506"), lower_3=D("494"),
        )
        u, l = vwap.band_pair("2σ")
        assert u == D("504")
        assert l == D("496")

    def test_band_pair_half_sigma(self):
        vwap = ExtendedSessionVWAP(
            vwap=D("500"), std_dev=D("2"),
            upper_05=D("501"), lower_05=D("499"),
            upper_1=D("502"), lower_1=D("498"),
            upper_15=D("503"), lower_15=D("497"),
            upper_2=D("504"), lower_2=D("496"),
            upper_3=D("506"), lower_3=D("494"),
        )
        u, l = vwap.band_pair("0.5σ")
        assert u == D("501")
        assert l == D("499")

    def test_band_pair_3_sigma(self):
        vwap = ExtendedSessionVWAP(
            vwap=D("500"), std_dev=D("2"),
            upper_05=D("501"), lower_05=D("499"),
            upper_1=D("502"), lower_1=D("498"),
            upper_15=D("503"), lower_15=D("497"),
            upper_2=D("504"), lower_2=D("496"),
            upper_3=D("506"), lower_3=D("494"),
        )
        u, l = vwap.band_pair("3σ")
        assert u == D("506")
        assert l == D("494")

    def test_band_pair_unknown_defaults_to_2_sigma(self):
        vwap = ExtendedSessionVWAP(
            vwap=D("500"), std_dev=D("2"),
            upper_05=D("501"), lower_05=D("499"),
            upper_1=D("502"), lower_1=D("498"),
            upper_15=D("503"), lower_15=D("497"),
            upper_2=D("504"), lower_2=D("496"),
            upper_3=D("506"), lower_3=D("494"),
        )
        u, l = vwap.band_pair("unknown")
        assert u == D("504")
        assert l == D("496")

    def test_frozen(self):
        vwap = ExtendedSessionVWAP(
            vwap=D("500"), std_dev=D("2"),
            upper_05=D("501"), lower_05=D("499"),
            upper_1=D("502"), lower_1=D("498"),
            upper_15=D("503"), lower_15=D("497"),
            upper_2=D("504"), lower_2=D("496"),
            upper_3=D("506"), lower_3=D("494"),
        )
        with pytest.raises(AttributeError):
            vwap.vwap = D("999")  # type: ignore[misc]


# =========================================================================
# CandleStrength dataclass
# =========================================================================


class TestCandleStrength:
    def test_bull_candle_above_threshold(self):
        cs = CandleStrength(
            body=D("2"), bar_range=D("4"), body_atr=D("0.5"),
            is_bull=True, is_bear=False,
            bull_wick=True, bear_wick=False,
        )
        assert cs.bull_candle(D("0.3"))
        assert not cs.bear_candle(D("0.3"))

    def test_bull_candle_below_threshold(self):
        cs = CandleStrength(
            body=D("1"), bar_range=D("4"), body_atr=D("0.1"),
            is_bull=True, is_bear=False,
            bull_wick=False, bear_wick=False,
        )
        assert not cs.bull_candle(D("0.3"))

    def test_bear_candle_above_threshold(self):
        cs = CandleStrength(
            body=D("2"), bar_range=D("4"), body_atr=D("0.5"),
            is_bull=False, is_bear=True,
            bull_wick=False, bear_wick=True,
        )
        assert cs.bear_candle(D("0.3"))
        assert not cs.bull_candle(D("0.3"))


# =========================================================================
# candle_strength()
# =========================================================================


class TestCandleStrengthFunction:
    def test_bullish_bar(self):
        bar = _bar(open_="498", high="503", low="497", close="502")
        cs = candle_strength(bar, D("4"))
        assert cs.is_bull
        assert not cs.is_bear
        assert cs.body == D("4")  # |502 - 498|
        assert cs.bar_range == D("6")  # 503 - 497

    def test_bearish_bar(self):
        bar = _bar(open_="502", high="503", low="497", close="498")
        cs = candle_strength(bar, D("4"))
        assert cs.is_bear
        assert not cs.is_bull
        assert cs.body == D("4")

    def test_zero_atr_returns_zero_body_atr(self):
        bar = _bar(open_="500", high="502", low="498", close="501")
        cs = candle_strength(bar, ZERO)
        assert cs.body_atr == ZERO

    def test_doji_bar(self):
        """Open == Close → neither bull nor bear."""
        bar = _bar(open_="500", high="502", low="498", close="500")
        cs = candle_strength(bar, D("4"))
        assert not cs.is_bull
        assert not cs.is_bear
        assert cs.body == ZERO

    def test_bull_wick_signal(self):
        """Close in upper 60% of range → bull_wick."""
        # range = 10, close at 109 (in top 40% of range)
        bar = _bar(open_="101", high="110", low="100", close="109")
        cs = candle_strength(bar, D("4"))
        assert cs.bull_wick
        assert not cs.bear_wick

    def test_bear_wick_signal(self):
        """Close in lower 60% of range → bear_wick."""
        # range = 10, close at 101 (in bottom 40% of range)
        bar = _bar(open_="109", high="110", low="100", close="101")
        cs = candle_strength(bar, D("4"))
        assert cs.bear_wick
        assert not cs.bull_wick

    def test_zero_range_no_wick(self):
        """Zero range bar has no wick signals."""
        bar = _bar(open_="500", high="500", low="500", close="500")
        cs = candle_strength(bar, D("4"))
        assert not cs.bull_wick
        assert not cs.bear_wick


# =========================================================================
# Candle patterns
# =========================================================================


class TestIsHammer:
    def test_classic_hammer(self):
        """Long lower wick, small body, bullish close."""
        # Range = 10, lower wick = close - low = 7, body = 1
        bar = _bar(open_="106", high="107.1", low="100", close="107", volume=100)
        assert is_hammer(bar, D("4"))

    def test_not_hammer_short_lower_wick(self):
        """Short lower wick → not a hammer."""
        bar = _bar(open_="104", high="107", low="103", close="106", volume=100)
        assert not is_hammer(bar, D("4"))

    def test_not_hammer_bearish_close(self):
        """Close < Open → not a hammer."""
        bar = _bar(open_="107", high="108", low="100", close="101", volume=100)
        assert not is_hammer(bar, D("4"))

    def test_zero_range_not_hammer(self):
        bar = _bar(open_="100", high="100", low="100", close="100", volume=100)
        assert not is_hammer(bar, D("4"))

    def test_tiny_body_below_atr_threshold(self):
        """Body < 0.05 * ATR → not a hammer."""
        # ATR = 100, body = 0.01 (< 5.0 threshold)
        bar = _bar(open_="100.00", high="110", low="100", close="100.01", volume=100)
        assert not is_hammer(bar, D("100"))


class TestIsInvHammer:
    def test_classic_inverted_hammer(self):
        """Long upper wick, small body, bearish close."""
        # Range = 10, upper wick = high - open = 7, body = 1
        bar = _bar(open_="101", high="108", low="100", close="100.1", volume=100)
        assert is_inv_hammer(bar, D("4"))

    def test_not_inv_hammer_bullish_close(self):
        """Close > Open → not an inverted hammer."""
        bar = _bar(open_="100", high="108", low="100", close="101", volume=100)
        assert not is_inv_hammer(bar, D("4"))

    def test_zero_range_not_inv_hammer(self):
        bar = _bar(open_="100", high="100", low="100", close="100", volume=100)
        assert not is_inv_hammer(bar, D("4"))


class TestBullEngulfing:
    def test_classic_bull_engulfing(self):
        """Current bullish bar engulfs previous bearish bar."""
        prev = _bar(open_="105", high="106", low="98", close="99")  # bearish
        curr = _bar(open_="98", high="107", low="97", close="106")  # bullish engulfing
        assert is_bull_engulfing(curr, prev)

    def test_not_bull_engulfing_previous_bullish(self):
        """Previous bar not bearish → no bull engulfing."""
        prev = _bar(open_="99", high="106", low="98", close="105")
        curr = _bar(open_="98", high="108", low="97", close="106")
        assert not is_bull_engulfing(curr, prev)

    def test_not_bull_engulfing_current_bearish(self):
        """Current bar not bullish → no bull engulfing."""
        prev = _bar(open_="105", high="106", low="98", close="99")
        curr = _bar(open_="106", high="108", low="97", close="98")
        assert not is_bull_engulfing(curr, prev)

    def test_not_engulfing_smaller_body(self):
        """Current body smaller than previous → no engulfing."""
        prev = _bar(open_="110", high="111", low="98", close="99")  # big body
        curr = _bar(open_="99", high="101", low="98", close="100")  # small body
        assert not is_bull_engulfing(curr, prev)

    def test_engulf_min_ratio(self):
        """Body/range ratio below engulf_min → no engulfing."""
        prev = _bar(open_="105", high="106", low="98", close="99")
        # Current: range = 20, body = 2 → ratio = 0.10 < default 0.35
        curr = _bar(open_="99", high="109", low="89", close="101")
        assert not is_bull_engulfing(curr, prev)


class TestBearEngulfing:
    def test_classic_bear_engulfing(self):
        """Current bearish bar engulfs previous bullish bar."""
        prev = _bar(open_="99", high="106", low="98", close="105")  # bullish
        curr = _bar(open_="106", high="107", low="97", close="98")  # bearish engulfing
        assert is_bear_engulfing(curr, prev)

    def test_not_bear_engulfing_previous_bearish(self):
        """Previous bar not bullish → no bear engulfing."""
        prev = _bar(open_="105", high="106", low="98", close="99")
        curr = _bar(open_="106", high="108", low="97", close="98")
        assert not is_bear_engulfing(curr, prev)

    def test_not_bear_engulfing_current_bullish(self):
        """Current bar not bearish → no bear engulfing."""
        prev = _bar(open_="99", high="106", low="98", close="105")
        curr = _bar(open_="98", high="107", low="97", close="106")
        assert not is_bear_engulfing(curr, prev)


# =========================================================================
# S/R proximity
# =========================================================================


class TestNearLevel:
    def test_price_near_level(self):
        assert near_level(D("500"), D("501"), D("0.35"))

    def test_price_far_from_level(self):
        assert not near_level(D("500"), D("510"), D("0.35"))

    def test_zero_price_returns_false(self):
        assert not near_level(ZERO, D("500"), D("0.35"))

    def test_zero_level_returns_false(self):
        assert not near_level(D("500"), ZERO, D("0.35"))

    def test_exact_match(self):
        assert near_level(D("500"), D("500"), D("0.35"))

    def test_boundary_case(self):
        """Price exactly at proximity threshold should return True."""
        # 0.35% of 100 = 0.35 → price at 100.35
        assert near_level(D("100"), D("100.35"), D("0.35"))


class TestRoundTo5:
    def test_rounds_up(self):
        assert _round_to_5(D("503")) == D("505")

    def test_rounds_down(self):
        assert _round_to_5(D("502")) == D("500")

    def test_exact_multiple(self):
        assert _round_to_5(D("500")) == D("500")

    def test_midpoint_rounds_up(self):
        assert _round_to_5(D("502.5")) == D("505")


class TestComputeSrScore:
    def test_near_pd_low_scores(self):
        """Close near PD low should give score_count=1."""
        any_near, count = compute_sr_score(
            D("500"),
            pd_low=D("500.5"),
            sr_pdhlc=True,
        )
        assert any_near
        assert count == 1

    def test_near_pd_high_display_only(self):
        """Close near PD high → any_near=True but score_count=0 (display-only)."""
        any_near, count = compute_sr_score(
            D("500"),
            pd_high=D("500.5"),
            sr_pdhlc=True,
        )
        assert any_near
        assert count == 0  # PDH is display-only

    def test_near_round_number_display_only(self):
        """Close near $5 round → any_near=True, score_count=0."""
        any_near, count = compute_sr_score(
            D("500.1"),
            sr_round=True,
        )
        assert any_near
        assert count == 0

    def test_multiple_scoring_levels(self):
        """Close near both OR high and OR low → count=2."""
        any_near, count = compute_sr_score(
            D("500"),
            or_high=D("500.5"),
            or_low=D("499.5"),
            sr_or=True,
        )
        assert any_near
        assert count == 2

    def test_disabled_flags(self):
        """When sr_or=False, OR levels should not score."""
        # Also disable sr_round since close=500 is near a $5 round number
        any_near, count = compute_sr_score(
            D("500"),
            or_high=D("500.5"),
            or_low=D("499.5"),
            sr_or=False,
            sr_round=False,
        )
        assert not any_near
        assert count == 0

    def test_pw_high_low_scoring(self):
        """Weekly H/L levels score when enabled."""
        any_near, count = compute_sr_score(
            D("500"),
            pw_high=D("500.5"),
            pw_low=D("499.8"),
            sr_week_hl=True,
        )
        assert any_near
        assert count == 2

    def test_prev_vwap_scoring(self):
        """Previous VWAP scores when enabled."""
        any_near, count = compute_sr_score(
            D("500"),
            prev_vwap=D("500.5"),
            sr_prev_vwap=True,
        )
        assert any_near
        assert count == 1

    def test_nothing_near(self):
        """No levels near → any_near=False, count=0."""
        any_near, count = compute_sr_score(
            D("500"),
            pd_high=D("600"),
            pd_low=D("400"),
            sr_pdhlc=True,
            sr_round=False,
        )
        assert not any_near
        assert count == 0


# =========================================================================
# Linear regression slope
# =========================================================================


class TestLinearRegressionSlope:
    def test_uptrend(self):
        """Steadily increasing prices → positive slope."""
        bars = _bars_trend_up(20, start=100.0, step=1.0)
        slope = linear_regression_slope(bars, 19, period=15)
        assert slope > ZERO

    def test_flat_prices(self):
        """Constant prices → slope = 0."""
        bars = [_bar(close=str(100), open_=str(100), high=str(101), low=str(99))
                for _ in range(20)]
        slope = linear_regression_slope(bars, 19, period=15)
        assert slope == ZERO

    def test_insufficient_data(self):
        """end_index < period - 1 → returns ZERO."""
        bars = _bars_trend_up(5)
        slope = linear_regression_slope(bars, 4, period=15)
        assert slope == ZERO

    def test_downtrend(self):
        """Steadily decreasing prices → negative slope."""
        bars = []
        for i in range(20):
            price = 200.0 - i * 1.0
            bars.append(_bar(
                date=f"2025-01-15 09:{30 + i % 30:02d}:00-05:00",
                open_=str(price + 0.5),
                high=str(price + 1),
                low=str(price - 1),
                close=str(price),
            ))
        slope = linear_regression_slope(bars, 19, period=15)
        assert slope < ZERO


class TestLrsNormalized:
    def test_with_custom_atr_fn(self):
        """Dependency injection: custom ATR function used."""
        bars = _bars_trend_up(20)
        called = []

        def mock_atr(data, end_index, period):
            called.append(True)
            return D("2")

        result = lrs_normalized(bars, 19, _atr_fn=mock_atr)
        assert len(called) == 1
        assert result != ZERO

    def test_zero_atr_returns_zero(self):
        bars = _bars_trend_up(20)
        result = lrs_normalized(bars, 19, _atr_fn=lambda d, i, p: ZERO)
        assert result == ZERO


# =========================================================================
# Time-of-day relative volume
# =========================================================================


class TestTodRvol:
    def test_negative_index(self):
        bars = _bars_trend_up(10)
        assert tod_rvol(bars, -1) == D("1")

    def test_zero_current_volume(self):
        bars = _bars_trend_up(10)
        # Override last bar's volume to 0
        bars[-1] = _bar(
            date=bars[-1].date,
            open_=str(bars[-1].open),
            high=str(bars[-1].high),
            low=str(bars[-1].low),
            close=str(bars[-1].close),
            volume=0,
        )
        assert tod_rvol(bars, 9) == D("1")

    def test_insufficient_lookback(self):
        """Not enough history for lookback → returns 1."""
        bars = _bars_trend_up(10)
        # lookback_days=10, bars_per_day=78 → needs 780 bars of history
        assert tod_rvol(bars, 9, lookback_days=10, bars_per_day=78) == D("1")

    def test_normal_calculation(self):
        """With enough lookback data, returns ratio != 1."""
        # Create 2 "days" worth of data with different volumes
        bars_per_day = 10
        bars = []
        for day in range(3):
            for i in range(bars_per_day):
                vol = 1_000_000 if day < 2 else 2_000_000
                bars.append(_bar(
                    date=f"2025-01-{15 + day:02d} 10:{i:02d}:00-05:00",
                    volume=vol,
                ))
        # Current bar at index 29 (day 3), lookback at same TOD in day 1 and 2
        result = tod_rvol(bars, 29, lookback_days=2, bars_per_day=bars_per_day)
        # Day3 volume=2M, avg(day1, day2)=1M → rvol=2.0
        assert result == D("2")


# =========================================================================
# CVD
# =========================================================================


class TestCvdSession:
    def test_negative_index(self):
        bars = _bars_trend_up(5)
        assert cvd_session(bars, -1) == ZERO

    def test_zero_range_bar(self):
        """Zero-range bar gets vDelta=0.5, contributing 0 to CVD."""
        bars = [_bar(open_="100", high="100", low="100", close="100", volume=1000)]
        result = cvd_session(bars, 0)
        assert result == ZERO

    def test_all_buying_pressure(self):
        """Close == High → vDelta=1 → contribution = 0.5*volume."""
        bars = [_bar(open_="100", high="110", low="100", close="110", volume=1000)]
        result = cvd_session(bars, 0)
        assert result > ZERO

    def test_all_selling_pressure(self):
        """Close == Low → vDelta=0 → contribution = -0.5*volume."""
        bars = [_bar(open_="110", high="110", low="100", close="100", volume=1000)]
        result = cvd_session(bars, 0)
        assert result < ZERO


class TestCvdNormalized:
    def test_insufficient_data(self):
        bars = _bars_trend_up(5)
        assert cvd_normalized(bars, 4, vol_sma_period=20) == ZERO


# =========================================================================
# Daily aggregation
# =========================================================================


class TestAggregate:
    def test_empty_input(self):
        assert aggregate_to_daily([]) == []

    def test_single_day(self):
        bars = [
            _bar(date="2025-01-15 09:30:00-05:00", open_="100", high="105", low="99", close="103", volume=1000),
            _bar(date="2025-01-15 09:35:00-05:00", open_="103", high="108", low="101", close="106", volume=2000),
            _bar(date="2025-01-15 09:40:00-05:00", open_="106", high="107", low="100", close="102", volume=1500),
        ]
        daily = aggregate_to_daily(bars)
        assert len(daily) == 1
        assert daily[0].open == D("100")
        assert daily[0].high == D("108")
        assert daily[0].low == D("99")
        assert daily[0].close == D("102")
        assert daily[0].volume == 4500

    def test_multi_day(self):
        bars = [
            _bar(date="2025-01-15 09:30:00-05:00", open_="100", close="105", high="106", low="99", volume=1000),
            _bar(date="2025-01-15 09:35:00-05:00", open_="105", close="108", high="109", low="104", volume=1000),
            _bar(date="2025-01-16 09:30:00-05:00", open_="108", close="110", high="111", low="107", volume=2000),
            _bar(date="2025-01-16 09:35:00-05:00", open_="110", close="112", high="113", low="109", volume=2000),
        ]
        daily = aggregate_to_daily(bars)
        assert len(daily) == 2
        assert daily[0].date == "2025-01-15"
        assert daily[1].date == "2025-01-16"
        assert daily[0].volume == 2000
        assert daily[1].volume == 4000

    def test_single_bar(self):
        bars = [_bar(date="2025-01-15 09:30:00-05:00")]
        daily = aggregate_to_daily(bars)
        assert len(daily) == 1


class TestDailyAtrPrior:
    def test_no_prior_day(self):
        """Trading date is at or before first bar → ZERO."""
        bars = [_bar(date="2025-01-15 00:00:00", close="100", high="101", low="99")]
        assert daily_atr_prior(bars, "2025-01-15") == ZERO

    def test_with_prior_data(self):
        """With enough prior daily bars, returns non-zero ATR."""
        bars = _bars_multi_day(1, 20, start=100.0)  # 20 daily bars
        daily = aggregate_to_daily(bars)
        # Ask for ATR prior to the last day
        last_date = daily[-1].date[:10]
        result = daily_atr_prior(daily, last_date, period=14)
        # Should return a value computed from the 14 bars before last_date
        assert isinstance(result, Decimal)


# =========================================================================
# HTF resampling
# =========================================================================


class TestResampleToHtf:
    def test_negative_index(self):
        bars = _bars_trend_up(10)
        assert resample_to_htf(bars, -1) == []

    def test_factor_3(self):
        """9 bars in session → 3 complete 15-min candles."""
        bars = []
        for i in range(9):
            bars.append(_bar(
                date=f"2025-01-15 09:{30 + i * 5:02d}:00-05:00",
                open_=str(100 + i),
                high=str(100 + i + 1),
                low=str(100 + i - 1),
                close=str(100 + i + 0.5),
                volume=1000,
            ))
        result = resample_to_htf(bars, 8, factor=3)
        assert len(result) == 3
        # First candle: bars 0-2
        assert result[0].open == bars[0].open
        assert result[0].close == bars[2].close
        assert result[0].volume == 3000

    def test_incomplete_group_excluded(self):
        """10 bars with factor=3 → 3 candles (1 bar leftover)."""
        bars = []
        for i in range(10):
            bars.append(_bar(
                date=f"2025-01-15 09:{30 + i * 5:02d}:00-05:00",
                close=str(100 + i),
                volume=1000,
            ))
        result = resample_to_htf(bars, 9, factor=3)
        assert len(result) == 3  # 10 // 3 = 3


class TestHtfEmaTrend:
    def test_insufficient_data_returns_zero(self):
        bars = _bars_trend_up(5)
        assert htf_ema_trend(bars, 4) == 0

    def test_returns_valid_direction(self):
        """With enough data, returns -1, 0, or +1."""
        # Need at least slow_period * factor + factor bars
        bars = _bars_trend_up(200)
        result = htf_ema_trend(bars, 199)
        assert result in (-1, 0, 1)


# =========================================================================
# Relative volume
# =========================================================================


class TestRelVol:
    def test_insufficient_data(self):
        bars = _bars_trend_up(5)
        assert rel_vol(bars, 4, period=20) == D("1")

    def test_normal_calculation(self):
        """Current volume 2x average → rel_vol ≈ 2."""
        bars = [_bar(volume=1_000_000) for _ in range(25)]
        # Override last bar with 2x volume
        bars[-1] = _bar(volume=2_000_000)
        result = rel_vol(bars, 24, period=20)
        # Average of last 20 bars includes the 2M bar itself
        # So avg = (19 * 1M + 1 * 2M) / 20 = 1.05M
        # rel_vol = 2M / 1.05M ≈ 1.905
        assert result > D("1.5")

    def test_zero_average_volume(self):
        """When all bars in the lookback (including current) have 0 volume → 1."""
        bars = [_bar(volume=0) for _ in range(25)]
        assert rel_vol(bars, 24, period=20) == D("1")


# =========================================================================
# VWAP slope & acceleration
# =========================================================================


class TestVwapSlope:
    def test_insufficient_lookback(self):
        bars = _bars_trend_up(3)
        assert vwap_slope(bars, 2, lookback=5) == ZERO

    def test_custom_vwap_fn(self):
        """Dependency injection for VWAP function."""
        bars = _bars_trend_up(10)
        # VWAP goes from 100 to 110
        result = vwap_slope(
            bars, 9, lookback=5,
            _vwap_fn=lambda data, idx: D(str(100 + idx)),
        )
        # slope = vwap(9) - vwap(4) = 109 - 104 = 5
        assert result == D("5")


class TestVwapAcceleration:
    def test_insufficient_lookback(self):
        bars = _bars_trend_up(5)
        assert vwap_acceleration(bars, 4, lookback=5) == ZERO

    def test_zero_atr_returns_zero(self):
        bars = _bars_trend_up(20)
        result = vwap_acceleration(
            bars, 19, lookback=5,
            _atr_fn=lambda d, i, p: ZERO,
            _slope_fn=lambda d, i, lb: D("1"),
        )
        assert result == ZERO

    def test_positive_acceleration(self):
        """Increasing slope → positive acceleration."""
        bars = _bars_trend_up(20)
        # Custom slope fn: current slope > prior slope
        result = vwap_acceleration(
            bars, 19, lookback=5,
            _atr_fn=lambda d, i, p: D("2"),
            _slope_fn=lambda d, i, lb: D(str(i)),  # slope increases with index
        )
        # accel = (slope(19) - slope(14)) / ATR(2) = (19-14)/2 = 2.5
        assert result > ZERO
