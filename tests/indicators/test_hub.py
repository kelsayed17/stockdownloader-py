"""Tests for the IndicatorHub caching layer.

Verifies that:
- Cache hits return the same object (identity check).
- Different indices produce separate cache entries.
- Different parameters produce separate cache entries.
- Data reference change clears the cache.
- ``clear()`` manually empties the cache.
- Every hub method matches the raw utility function output.
"""

import random
from decimal import Decimal
from unittest.mock import patch

from stockdownloader.core.models.price import PriceData
from stockdownloader.indicators.hub import IndicatorHub
from stockdownloader import indicators as ti
from stockdownloader.indicators import sma as _sma, ema as _ema
from stockdownloader.indicators.momentum import StreamingMACD, StreamingRSI
from stockdownloader.indicators.volatility import StreamingATR, StreamingEMA
from stockdownloader.indicators.trend import StreamingADX


def _generate_test_data(days: int) -> list[PriceData]:
    """Generate synthetic price data with a mild uptrend."""
    data: list[PriceData] = []
    price = 100.0

    for i in range(days):
        change = (random.random() - 0.48) * 3
        price = max(50, price + change)

        open_ = price + (random.random() - 0.5) * 2
        high = max(open_, price) + random.random() * 2
        low = min(open_, price) - random.random() * 2
        close = price
        volume = int(1_000_000 + random.random() * 5_000_000)

        data.append(
            PriceData(
                date=f"2020-01-{min(i + 1, 28):02d}",
                open=Decimal(str(round(open_, 4))),
                high=Decimal(str(round(high, 4))),
                low=Decimal(str(round(low, 4))),
                close=Decimal(str(round(close, 4))),
                adj_close=Decimal(str(round(close, 4))),
                volume=volume,
            )
        )
    return data


# Fixed seed for reproducibility
random.seed(99)
DATA = _generate_test_data(300)


# =========================================================================
# Cache behaviour
# =========================================================================


def test_cache_hit_returns_same_object():
    # Use a cache-based indicator for identity check
    hub = IndicatorHub()
    result1 = hub.bollinger_bands(DATA, 50, period=20)
    result2 = hub.bollinger_bands(DATA, 50, period=20)
    assert result1 is result2


def test_streaming_rsi_returns_same_value():
    """Streaming RSI returns equal values for repeated calls at same index."""
    hub = IndicatorHub()
    r1 = hub.rsi(DATA, 50, period=14)
    r2 = hub.rsi(DATA, 50, period=14)
    assert r1 == r2


def test_different_index_creates_separate_entry():
    hub = IndicatorHub()
    # Use SMA (cache-based, not streaming) to test cache entry counting
    r1 = hub.sma(DATA, 50, period=20)
    r2 = hub.sma(DATA, 51, period=20)
    assert r1 is not r2
    assert hub.cache_size == 2


def test_different_params_create_separate_entries():
    hub = IndicatorHub()
    # Use SMA (cache-based) to test cache entry counting
    r1 = hub.sma(DATA, 50, period=10)
    r2 = hub.sma(DATA, 50, period=20)
    assert r1 is not r2
    assert hub.cache_size == 2


def test_data_reference_change_clears_cache():
    hub = IndicatorHub()
    hub.sma(DATA, 50, period=20)
    assert hub.cache_size == 1

    # Same content, different list object
    data2 = list(DATA)
    hub.sma(data2, 50, period=20)
    # Old entry was cleared when data reference changed
    assert hub.cache_size == 1


def test_clear_empties_cache():
    hub = IndicatorHub()
    hub.sma(DATA, 50, period=10)
    hub.sma(DATA, 50, period=20)
    assert hub.cache_size == 2
    hub.clear()
    assert hub.cache_size == 0


def test_underlying_function_called_only_once():
    # Use a cache-based indicator to test function call counting
    # Patch the momentum module object that hub.py imported, since hub
    # calls `momentum.cci(...)`.
    from stockdownloader.indicators import momentum as _mom
    hub = IndicatorHub()
    with patch.object(_mom, "cci", wraps=_mom.cci) as mock_cci:
        hub.cci(DATA, 50, period=20)
        hub.cci(DATA, 50, period=20)
        hub.cci(DATA, 50, period=20)
        assert mock_cci.call_count == 1


def test_cache_size_grows_correctly():
    hub = IndicatorHub()
    # Use only cache-based indicators for this test
    # (streaming indicators like EMA, RSI don't add to cache_size)
    hub.sma(DATA, 50, period=10)
    hub.sma(DATA, 50, period=20)
    hub.cci(DATA, 50, period=20)
    hub.roc(DATA, 50, period=12)
    assert hub.cache_size == 4


# =========================================================================
# Moving Averages — match raw functions
# =========================================================================


def test_sma_matches_raw():
    hub = IndicatorHub()
    expected = _sma(DATA, 50, 20)
    actual = hub.sma(DATA, 50, period=20)
    assert expected == actual


def test_ema_matches_streaming():
    """Hub EMA uses streaming accumulator (processes from bar 0)."""
    hub = IndicatorHub()
    s = StreamingEMA(10)
    expected = s.update(DATA, 50)
    actual = hub.ema(DATA, 50, period=10)
    assert expected == actual


# =========================================================================
# Momentum — match raw functions
# =========================================================================


def test_rsi_matches_streaming():
    """Hub RSI uses streaming accumulator (Wilder smoothing from bar 0)."""
    hub = IndicatorHub()
    s = StreamingRSI(14)
    expected = s.update(DATA, 50)
    actual = hub.rsi(DATA, 50, period=14)
    assert expected == actual


def test_macd_line_matches_streaming():
    """Hub MACD line uses streaming accumulator."""
    hub = IndicatorHub()
    s = StreamingMACD(12, 26, 9)
    line, _, _ = s.update(DATA, 50)
    actual = hub.macd_line(DATA, 50, fast=12, slow=26)
    assert line == actual


def test_macd_signal_matches_streaming():
    """Hub MACD signal uses streaming accumulator."""
    hub = IndicatorHub()
    s = StreamingMACD(12, 26, 9)
    _, sig, _ = s.update(DATA, 50)
    actual = hub.macd_signal(DATA, 50, fast=12, slow=26, signal=9)
    assert sig == actual


def test_macd_histogram_matches_streaming():
    """Hub MACD histogram uses streaming accumulator."""
    hub = IndicatorHub()
    s = StreamingMACD(12, 26, 9)
    _, _, hist = s.update(DATA, 50)
    actual = hub.macd_histogram(DATA, 50, fast=12, slow=26, signal=9)
    assert hist == actual


def test_roc_matches_raw():
    hub = IndicatorHub()
    expected = ti.roc(DATA, 50, 12)
    actual = hub.roc(DATA, 50, period=12)
    assert expected == actual


def test_williams_r_matches_raw():
    hub = IndicatorHub()
    expected = ti.williams_r(DATA, 50, 14)
    actual = hub.williams_r(DATA, 50, period=14)
    assert expected == actual


# =========================================================================
# Bollinger Bands — match raw functions
# =========================================================================


def test_bollinger_bands_matches_raw():
    hub = IndicatorHub()
    expected = ti.bollinger_bands(DATA, 50, 20, 2.0)
    actual = hub.bollinger_bands(DATA, 50, period=20, num_std_dev=2.0)
    assert expected == actual


def test_bollinger_percent_b_matches_raw():
    hub = IndicatorHub()
    expected = ti.bollinger_percent_b(DATA, 50, 20)
    actual = hub.bollinger_percent_b(DATA, 50, period=20)
    assert expected == actual


# =========================================================================
# Stochastic — match raw functions
# =========================================================================


def test_stochastic_matches_raw():
    hub = IndicatorHub()
    expected = ti.stochastic(DATA, 50, 14, 3)
    actual = hub.stochastic(DATA, 50, k_period=14, d_period=3)
    assert expected == actual


# =========================================================================
# Volatility — match raw functions
# =========================================================================


def test_atr_matches_streaming():
    """Hub ATR uses streaming accumulator (Wilder smoothing from bar 0)."""
    hub = IndicatorHub()
    s = StreamingATR(14)
    expected = s.update(DATA, 50)
    actual = hub.atr(DATA, 50, period=14)
    assert expected == actual


def test_true_range_matches_raw():
    hub = IndicatorHub()
    expected = ti.true_range(DATA, 50)
    actual = hub.true_range(DATA, 50)
    assert expected == actual


def test_standard_deviation_matches_raw():
    hub = IndicatorHub()
    expected = ti.standard_deviation(DATA, 50, 20)
    actual = hub.standard_deviation(DATA, 50, period=20)
    assert expected == actual


# =========================================================================
# Volume — match raw functions
# =========================================================================


def test_obv_matches_raw():
    hub = IndicatorHub()
    expected = ti.obv(DATA, 50)
    actual = hub.obv(DATA, 50)
    assert expected == actual


def test_is_obv_rising_matches_raw():
    hub = IndicatorHub()
    expected = ti.is_obv_rising(DATA, 50, 5)
    actual = hub.is_obv_rising(DATA, 50, lookback=5)
    assert expected == actual


def test_average_volume_matches_raw():
    hub = IndicatorHub()
    expected = ti.average_volume(DATA, 50, 20)
    actual = hub.average_volume(DATA, 50, period=20)
    assert expected == actual


def test_mfi_matches_raw():
    hub = IndicatorHub()
    expected = ti.mfi(DATA, 50, 14)
    actual = hub.mfi(DATA, 50, period=14)
    assert expected == actual


# =========================================================================
# Trend — match raw functions
# =========================================================================


def test_adx_matches_streaming():
    """Hub ADX uses streaming accumulator (Wilder smoothing from bar 0)."""
    hub = IndicatorHub()
    s = StreamingADX(14)
    vals = s.update(DATA, 50)
    actual = hub.adx(DATA, 50, period=14)
    assert actual.adx == vals[0]
    assert actual.plus_di == vals[1]
    assert actual.minus_di == vals[2]


def test_parabolic_sar_matches_raw():
    hub = IndicatorHub()
    expected = ti.parabolic_sar(DATA, 50)
    actual = hub.parabolic_sar(DATA, 50)
    assert expected == actual


def test_is_sar_bullish_matches_raw():
    hub = IndicatorHub()
    expected = ti.is_sar_bullish(DATA, 50)
    actual = hub.is_sar_bullish(DATA, 50)
    assert expected == actual


def test_ichimoku_matches_raw():
    hub = IndicatorHub()
    expected = ti.ichimoku(DATA, 60)
    actual = hub.ichimoku(DATA, 60)
    assert expected == actual


# =========================================================================
# CCI — match raw functions
# =========================================================================


def test_cci_matches_raw():
    hub = IndicatorHub()
    expected = ti.cci(DATA, 50, 20)
    actual = hub.cci(DATA, 50, period=20)
    assert expected == actual


# =========================================================================
# VWAP — match raw functions
# =========================================================================


def test_vwap_matches_raw():
    hub = IndicatorHub()
    expected = ti.vwap(DATA, 50, 20)
    actual = hub.vwap(DATA, 50, lookback=20)
    assert expected == actual


# =========================================================================
# Fibonacci — match raw functions
# =========================================================================


def test_fibonacci_retracement_matches_raw():
    hub = IndicatorHub()
    expected = ti.fibonacci_retracement(DATA, 80, 50)
    actual = hub.fibonacci_retracement(DATA, 80, lookback=50)
    assert expected == actual


# =========================================================================
# Support & Resistance — match raw functions
# =========================================================================


def test_support_resistance_matches_raw():
    hub = IndicatorHub()
    expected = ti.support_resistance(DATA, 80, 50, 5)
    actual = hub.support_resistance(DATA, 80, lookback=50, window=5)
    assert expected == actual


# =========================================================================
# Session-scoped (daily data — session_vwap uses date[:10])
# =========================================================================


def test_session_vwap_matches_raw():
    hub = IndicatorHub()
    expected = ti.session_vwap(DATA, 10)
    actual = hub.session_vwap(DATA, 10)
    assert expected == actual


def test_session_vwap_bands_matches_raw():
    hub = IndicatorHub()
    expected = ti.session_vwap_bands(DATA, 10)
    actual = hub.session_vwap_bands(DATA, 10)
    assert expected == actual
