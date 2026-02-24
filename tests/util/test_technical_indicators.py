"""Tests for technical_indicators utility functions."""

import random
from decimal import Decimal

from stockdownloader.core.models.price import PriceData
from stockdownloader.util.indicators import (
    bollinger_bands,
    bollinger_percent_b,
    stochastic,
    atr,
    true_range,
    obv,
    is_obv_rising,
    adx,
    parabolic_sar,
    is_sar_bullish,
    williams_r,
    cci,
    vwap,
    fibonacci_retracement,
    roc,
    mfi,
    ichimoku,
    rsi,
    macd_line,
    macd_signal,
    macd_histogram,
    support_resistance,
    average_volume,
    standard_deviation,
)


def _generate_test_data(days):
    data = []
    price = 100.0

    for i in range(days):
        # Simulate a mild uptrend with noise
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
                open=Decimal(str(open_)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close)),
                adj_close=Decimal(str(close)),
                volume=volume,
            )
        )
    return data


# Use a fixed seed for reproducibility
random.seed(42)
DATA = _generate_test_data(300)


# =========================================================================
# Bollinger Bands
# =========================================================================

def test_bollinger_bands_returns_valid_bands():
    bb = bollinger_bands(DATA, 50, 20, 2.0)
    assert bb.upper > bb.middle, "Upper band should be above middle"
    assert bb.middle > bb.lower, "Middle band should be above lower"
    assert bb.width > Decimal("0"), "Width should be positive"


def test_bollinger_bands_returns_zero_for_insufficient_data():
    bb = bollinger_bands(DATA, 5, 20, 2.0)
    assert bb.upper.compare(Decimal("0")) == 0


def test_bollinger_percent_b_within_expected_range():
    pct_b = bollinger_percent_b(DATA, 50, 20)
    # %B should be between -2 and 3 for most cases
    assert -2 <= float(pct_b) <= 3


# =========================================================================
# Stochastic
# =========================================================================

def test_stochastic_returns_valid_values():
    stoch = stochastic(DATA, 30, 14, 3)
    assert 0 <= float(stoch.percent_k) <= 100, (
        f"%K should be between 0 and 100, got {stoch.percent_k}"
    )
    assert 0 <= float(stoch.percent_d) <= 100, (
        f"%D should be between 0 and 100, got {stoch.percent_d}"
    )


def test_stochastic_returns_zero_for_insufficient_data():
    stoch = stochastic(DATA, 5, 14, 3)
    assert stoch.percent_k.compare(Decimal("0")) == 0


# =========================================================================
# ATR
# =========================================================================

def test_atr_returns_positive_value():
    result = atr(DATA, 30, 14)
    assert result > Decimal("0"), "ATR should be positive"


def test_true_range_computes_correctly():
    tr = true_range(DATA, 10)
    assert tr >= Decimal("0"), "True range should be non-negative"


# =========================================================================
# OBV
# =========================================================================

def test_obv_computes_without_error():
    obv_value = obv(DATA, 50)
    assert obv_value is not None


def test_is_obv_rising_returns_boolean():
    result = is_obv_rising(DATA, 50, 5)
    assert isinstance(result, bool)


# =========================================================================
# ADX
# =========================================================================

def test_adx_returns_valid_result():
    result = adx(DATA, 60, 14)
    assert result.adx >= Decimal("0"), "ADX should be non-negative"
    assert result.plus_di >= Decimal("0"), "+DI should be non-negative"
    assert result.minus_di >= Decimal("0"), "-DI should be non-negative"


# =========================================================================
# Parabolic SAR
# =========================================================================

def test_parabolic_sar_returns_positive_value():
    sar_val = parabolic_sar(DATA, 50)
    assert sar_val > Decimal("0"), "SAR should be positive"


def test_is_sar_bullish_returns_boolean():
    result = is_sar_bullish(DATA, 50)
    assert isinstance(result, bool)


# =========================================================================
# Williams %R
# =========================================================================

def test_williams_r_within_expected_range():
    will_r = williams_r(DATA, 30, 14)
    assert -100 <= float(will_r) <= 0, (
        f"Williams %R should be between -100 and 0, got {will_r}"
    )


# =========================================================================
# CCI
# =========================================================================

def test_cci_computes_without_error():
    cci_val = cci(DATA, 40, 20)
    assert cci_val is not None


# =========================================================================
# VWAP
# =========================================================================

def test_vwap_returns_positive_value():
    vwap_val = vwap(DATA, 30, 20)
    assert vwap_val > Decimal("0"), "VWAP should be positive"


# =========================================================================
# Fibonacci
# =========================================================================

def test_fibonacci_returns_ordered_levels():
    fib = fibonacci_retracement(DATA, 100, 50)
    assert fib.high >= fib.level_236, "High should be >= 23.6% level"
    assert fib.level_236 >= fib.level_382, "23.6% should be >= 38.2%"
    assert fib.level_382 >= fib.level_500, "38.2% should be >= 50%"
    assert fib.level_500 >= fib.level_618, "50% should be >= 61.8%"
    assert fib.level_618 >= fib.level_786, "61.8% should be >= 78.6%"
    assert fib.level_786 >= fib.low, "78.6% should be >= low"


# =========================================================================
# ROC
# =========================================================================

def test_roc_computes_without_error():
    roc_val = roc(DATA, 30, 12)
    assert roc_val is not None


def test_roc_returns_zero_for_insufficient_data():
    roc_val = roc(DATA, 5, 12)
    assert roc_val.compare(Decimal("0")) == 0


# =========================================================================
# MFI
# =========================================================================

def test_mfi_within_expected_range():
    mfi_val = mfi(DATA, 30, 14)
    assert 0 <= float(mfi_val) <= 100, (
        f"MFI should be between 0 and 100, got {mfi_val}"
    )


# =========================================================================
# Ichimoku
# =========================================================================

def test_ichimoku_returns_valid_result():
    ichi = ichimoku(DATA, 100)
    assert ichi.tenkan_sen > Decimal("0"), "Tenkan should be positive"
    assert ichi.kijun_sen > Decimal("0"), "Kijun should be positive"
    assert ichi.senkou_span_a > Decimal("0"), "Senkou A should be positive"
    assert ichi.senkou_span_b > Decimal("0"), "Senkou B should be positive"


def test_ichimoku_returns_zero_for_insufficient_data():
    ichi = ichimoku(DATA, 30)
    assert ichi.tenkan_sen.compare(Decimal("0")) == 0


# =========================================================================
# RSI
# =========================================================================

def test_rsi_within_expected_range():
    rsi_val = rsi(DATA, 30, 14)
    assert 0 <= float(rsi_val) <= 100, (
        f"RSI should be between 0 and 100, got {rsi_val}"
    )


# =========================================================================
# MACD
# =========================================================================

def test_macd_computes_without_error():
    macd_l = macd_line(DATA, 50, 12, 26)
    macd_s = macd_signal(DATA, 50, 12, 26, 9)
    histogram = macd_histogram(DATA, 50, 12, 26, 9)
    assert macd_l is not None
    assert macd_s is not None
    assert histogram is not None
    expected = (macd_l - macd_s).quantize(Decimal("0.000001"))
    actual = histogram.quantize(Decimal("0.000001"))
    assert expected == actual


# =========================================================================
# Support & Resistance
# =========================================================================

def test_support_resistance_returns_valid_levels():
    sr = support_resistance(DATA, 200, 100, 5)
    assert sr.support_levels is not None
    assert sr.resistance_levels is not None


# =========================================================================
# Average Volume
# =========================================================================

def test_average_volume_returns_positive_value():
    avg_vol = average_volume(DATA, 30, 20)
    assert avg_vol > Decimal("0"), "Average volume should be positive"


# =========================================================================
# Standard Deviation
# =========================================================================

def test_standard_deviation_returns_positive_value():
    std_dev = standard_deviation(DATA, 30, 20)
    assert std_dev > Decimal("0"), "Std dev should be positive"
