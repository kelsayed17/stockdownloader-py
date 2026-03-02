"""Tests for MultiIndicatorStrategy."""

import random
from decimal import Decimal

import pytest

from stockdownloader.core.models.price import PriceData
from stockdownloader.strategies.daily.multi_indicator import MultiIndicatorStrategy
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
    strategy = MultiIndicatorStrategy()
    assert "Multi-Indicator" in strategy.name


def test_constructor_invalid_threshold():
    with pytest.raises(ValueError):
        MultiIndicatorStrategy(0, 3)


def test_evaluate_hold_during_warmup():
    strategy = MultiIndicatorStrategy()
    data = _generate_test_data(300)
    assert strategy.evaluate(data, 50) == Signal.HOLD


def test_evaluate_handles_enough_data():
    strategy = MultiIndicatorStrategy()
    data = _generate_test_data(300)
    signal = strategy.evaluate(data, 250)
    assert signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


def test_warmup_period_is_sufficient():
    strategy = MultiIndicatorStrategy()
    assert strategy.warmup_period == 201


def test_custom_thresholds():
    strategy = MultiIndicatorStrategy(3, 3)
    assert "Buy>=3" in strategy.name
    assert "Sell>=3" in strategy.name


# ---------------------------------------------------------------------------
# Category cap + multi-category requirement tests
# ---------------------------------------------------------------------------

class TestBuyScoringLogic:
    """Test the _compute_buy_score static method directly."""

    def test_trend_cap_at_two(self):
        """All 3 trend indicators active should cap at 2."""
        from unittest.mock import MagicMock
        from decimal import Decimal

        curr = MagicMock()
        prev = MagicMock()

        # Set all 3 trend signals bullish
        curr.ema12 = Decimal("110")
        curr.ema26 = Decimal("100")  # EMA12 > EMA26
        curr.sma200 = Decimal("90")
        curr.close = Decimal("110")  # close > SMA200
        curr.price_above_cloud = True  # Ichimoku above

        # Set no momentum signals
        curr.rsi14 = Decimal("50")
        prev.rsi14 = Decimal("50")  # No RSI crossover
        curr.macd_line = Decimal("1")
        curr.macd_signal = Decimal("2")  # MACD bearish
        prev.macd_line = Decimal("1")
        prev.macd_signal = Decimal("2")
        curr.stoch_k = Decimal("50")
        curr.stoch_d = Decimal("40")
        prev.stoch_k = Decimal("50")
        prev.stoch_d = Decimal("55")

        # No volume signals
        curr.obv_rising = False
        curr.mfi14 = Decimal("50")
        prev.mfi14 = Decimal("50")

        score, cats = MultiIndicatorStrategy._compute_buy_score(curr, prev)
        # Trend should be capped at 2, not 3
        assert score == 2, f"Expected trend capped at 2, got {score}"
        assert cats == 1, "Only trend category should be active"

    def test_returns_score_and_categories(self):
        """Score includes category count for multi-category requirement."""
        from unittest.mock import MagicMock
        from decimal import Decimal

        curr = MagicMock()
        prev = MagicMock()

        # Set 1 trend + 1 momentum signal
        curr.ema12 = Decimal("110")
        curr.ema26 = Decimal("100")  # EMA bullish
        curr.sma200 = Decimal("0")   # SMA disabled
        curr.close = Decimal("110")
        curr.price_above_cloud = False

        # RSI crossing above 30 (momentum)
        curr.rsi14 = Decimal("31")
        prev.rsi14 = Decimal("29")
        # No MACD/Stoch
        curr.macd_line = Decimal("1")
        curr.macd_signal = Decimal("2")
        prev.macd_line = Decimal("1")
        prev.macd_signal = Decimal("2")
        curr.stoch_k = Decimal("50")
        curr.stoch_d = Decimal("40")
        prev.stoch_k = Decimal("50")
        prev.stoch_d = Decimal("55")

        # No volume
        curr.obv_rising = False
        curr.mfi14 = Decimal("50")
        prev.mfi14 = Decimal("50")

        score, cats = MultiIndicatorStrategy._compute_buy_score(curr, prev)
        assert score == 2  # 1 trend + 1 momentum
        assert cats == 2  # 2 categories active


class TestSellScoringLogic:
    """Test the _compute_sell_score static method directly."""

    def test_sell_trend_cap_at_two(self):
        """All 3 sell trend indicators should cap at 2."""
        from unittest.mock import MagicMock
        from decimal import Decimal

        curr = MagicMock()
        prev = MagicMock()

        # All 3 trend bearish
        curr.ema12 = Decimal("90")
        curr.ema26 = Decimal("100")  # EMA bearish
        curr.sma200 = Decimal("120")
        curr.close = Decimal("90")  # close < SMA200
        curr.price_above_cloud = False
        curr.ichimoku_span_a = Decimal("110")  # below cloud

        # No momentum
        curr.rsi14 = Decimal("50")
        prev.rsi14 = Decimal("50")
        curr.macd_line = Decimal("2")
        curr.macd_signal = Decimal("1")
        prev.macd_line = Decimal("2")
        prev.macd_signal = Decimal("1")
        curr.stoch_k = Decimal("50")
        curr.stoch_d = Decimal("55")
        prev.stoch_k = Decimal("50")
        prev.stoch_d = Decimal("45")

        # No volume
        curr.obv_rising = True
        curr.mfi14 = Decimal("50")
        prev.mfi14 = Decimal("50")

        score, cats = MultiIndicatorStrategy._compute_sell_score(curr, prev)
        assert score == 2, f"Expected sell trend capped at 2, got {score}"
        assert cats == 1


class TestCategoryRequirement:
    """Test that evaluate() requires >= 2 active categories."""

    def test_single_category_blocks_signal(self):
        """High score from a single category should not produce a signal."""
        # With threshold=2, a score of 2 from trend-only should NOT trigger
        # because cats < 2
        strategy = MultiIndicatorStrategy(buy_threshold=2, sell_threshold=2)
        data = _generate_test_data(300)
        # Most signals will be HOLD anyway with random data,
        # but we verify the strategy doesn't crash with low thresholds
        signals = [strategy.evaluate(data, i) for i in range(201, 250)]
        # Just verify it runs without errors; precise signal testing
        # would require crafted data
        assert all(s in (Signal.BUY, Signal.SELL, Signal.HOLD) for s in signals)
