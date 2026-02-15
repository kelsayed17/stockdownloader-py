"""Tests for signal_generator module."""

import random
from decimal import Decimal

from stockdownloader.analysis.signal_generator import generate_alert
from stockdownloader.model.alert_result import AlertDirection
from stockdownloader.model.price_data import PriceData


def _generate_test_data(days):
    """Generate random test price data with mild uptrend."""
    data = []
    price = 100.0
    for i in range(days):
        price += (random.random() - 0.48) * 3
        price = max(50, price)
        high = price + random.random() * 3
        low = price - random.random() * 3
        data.append(
            PriceData(
                date="2020-01-01",
                open=Decimal(str(price - 1)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(price)),
                adj_close=Decimal(str(price)),
                volume=int(1_000_000 + random.random() * 5_000_000),
            )
        )
    return data


# Use fixed seed for reproducibility
random.seed(42)
DATA_300 = _generate_test_data(300)
DATA_50 = _generate_test_data(50)


def test_generate_alert_with_sufficient_data():
    alert = generate_alert("TEST", DATA_300)

    assert alert is not None
    assert alert.symbol == "TEST"
    assert alert.direction is not None
    assert alert.call_recommendation is not None
    assert alert.put_recommendation is not None
    assert alert.current_price > Decimal("0")


def test_generate_alert_with_insufficient_data():
    alert = generate_alert("TEST", DATA_50)

    assert alert is not None
    assert alert.direction == AlertDirection.NEUTRAL


def test_generate_alert_has_indicators():
    alert = generate_alert("TEST", DATA_300)

    assert alert.bullish_indicators is not None
    assert alert.bearish_indicators is not None
    assert len(alert.bullish_indicators) + len(alert.bearish_indicators) > 0, (
        "Should detect at least one indicator"
    )


def test_generate_alert_has_support_resistance():
    alert = generate_alert("TEST", DATA_300)

    assert alert.support_levels is not None
    assert alert.resistance_levels is not None


def test_generate_alert_options_recommendation():
    alert = generate_alert("TEST", DATA_300)

    assert alert.call_recommendation.type is not None
    assert alert.put_recommendation.type is not None
    assert alert.call_recommendation.action is not None
    assert alert.put_recommendation.action is not None


def test_generate_alert_to_string_does_not_throw():
    alert = generate_alert("TEST", DATA_300)

    output = str(alert)
    assert output is not None
    assert "TRADING ALERT" in output
    assert "TEST" in output


def test_generate_alert_signal_strength():
    alert = generate_alert("TEST", DATA_300)

    strength = alert.signal_strength
    assert strength is not None
    assert "%" in strength


def test_generate_alert_at_specific_index():
    alert = generate_alert("TEST", DATA_300, 250)

    assert alert is not None
    assert alert.symbol == "TEST"
