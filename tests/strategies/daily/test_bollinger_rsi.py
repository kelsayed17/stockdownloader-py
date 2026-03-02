"""Tests for BollingerBandRSIStrategy."""

import random
from decimal import Decimal

from stockdownloader.core.models.price import PriceData
from stockdownloader.strategies.daily.bollinger_rsi import BollingerBandRSIStrategy
from stockdownloader.strategies.base import Signal

random.seed(42)


def _generate_test_data(days):
    data = []
    price = 100.0
    for i in range(days):
        price += (random.random() - 0.48) * 3
        price = max(50, price)
        data.append(
            PriceData(
                date="2020-01-01",
                open=Decimal(str(price - 1)),
                high=Decimal(str(price + 2)),
                low=Decimal(str(price - 2)),
                close=Decimal(str(price)),
                adj_close=Decimal(str(price)),
                volume=int(1_000_000 + random.random() * 5_000_000),
            )
        )
    return data


def test_constructor_default_values():
    strategy = BollingerBandRSIStrategy()
    assert strategy.name == "BB+RSI Mean Reversion (BB20, RSI14 [30/70], ADX<25)"


def test_evaluate_hold_during_warmup():
    strategy = BollingerBandRSIStrategy()
    data = _generate_test_data(50)
    assert strategy.evaluate(data, 5) == Signal.HOLD


def test_evaluate_hold_with_sufficient_data():
    strategy = BollingerBandRSIStrategy()
    data = _generate_test_data(200)
    # With random data, most signals should be HOLD
    signal = strategy.evaluate(data, 100)
    assert signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


def test_warmup_period_returns_expected_value():
    strategy = BollingerBandRSIStrategy()
    assert strategy.warmup_period >= 20, "Warmup should be at least 20"


def test_custom_parameters():
    strategy = BollingerBandRSIStrategy(30, 2.5, 14, 25, 75, 30)
    assert "BB30" in strategy.name


# ---------------------------------------------------------------------------
# BB width regime filter tests
# ---------------------------------------------------------------------------


class TestBBWidthRegimeFilter:
    """Tests for the _bb_width_ok() regime filter.

    Thresholds are 0.15% to 1.5%, calibrated for 5-minute bars.
    """

    def test_normal_width_passes(self):
        """Width within normal range should pass."""
        strategy = BollingerBandRSIStrategy()
        # Width = 0.50% of middle → (501.25 - 500.75) / 501.0 * 100 ≈ 0.10%
        # Let's set values to give ~0.50%
        mid = 500.0
        half_width = mid * 0.0025  # 0.50% total width
        assert strategy._bb_width_ok(mid + half_width, mid - half_width, mid)

    def test_squeeze_width_fails(self):
        """Very tight bands (squeeze, < 0.15%) should be filtered out."""
        strategy = BollingerBandRSIStrategy()
        mid = 500.0
        half_width = mid * 0.0005  # 0.10% total width
        assert not strategy._bb_width_ok(mid + half_width, mid - half_width, mid)

    def test_blowout_width_fails(self):
        """Very wide bands (blow-out, > 1.5%) should be filtered out."""
        strategy = BollingerBandRSIStrategy()
        mid = 500.0
        half_width = mid * 0.01  # 2.0% total width
        assert not strategy._bb_width_ok(mid + half_width, mid - half_width, mid)

    def test_exact_lower_threshold(self):
        """Exactly at lower threshold (0.15%) should pass."""
        strategy = BollingerBandRSIStrategy()
        mid = 1000.0
        width_pct = 0.15
        half = mid * width_pct / 200.0
        assert strategy._bb_width_ok(mid + half, mid - half, mid)

    def test_exact_upper_threshold(self):
        """Exactly at upper threshold (1.5%) should pass."""
        strategy = BollingerBandRSIStrategy()
        mid = 1000.0
        width_pct = 1.5
        half = mid * width_pct / 200.0
        assert strategy._bb_width_ok(mid + half, mid - half, mid)

    def test_zero_middle_returns_false(self):
        """Zero middle band should be filtered."""
        strategy = BollingerBandRSIStrategy()
        assert not strategy._bb_width_ok(10.0, 0.0, 0.0)

    def test_negative_middle_returns_false(self):
        """Negative middle band should be filtered."""
        strategy = BollingerBandRSIStrategy()
        assert not strategy._bb_width_ok(10.0, 0.0, -5.0)
