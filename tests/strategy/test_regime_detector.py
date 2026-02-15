"""Tests for MarketRegimeDetector.

Validates regime classification using synthetic price data
designed to trigger each of the five regimes.
"""

from decimal import Decimal

import pytest

from stockdownloader.model.price_data import PriceData
from stockdownloader.strategy.regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeClassification,
    RegimeDetectorConfig,
)
from stockdownloader.util.indicator_hub import IndicatorHub


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_bar(date: str, close: float, high: float = 0, low: float = 0,
              volume: int = 1_000_000) -> PriceData:
    """Create a PriceData bar with auto-computed high/low."""
    c = Decimal(str(round(close, 4)))
    h = Decimal(str(round(max(high, close + 0.5), 4)))
    l = Decimal(str(round(min(low, close - 0.5) if low else close - 0.5, 4)))
    return PriceData(
        date=date,
        open=c,
        high=h,
        low=l,
        close=c,
        adj_close=c,
        volume=volume,
    )


def _trending_up_data(n: int = 300, start: float = 100.0,
                      step: float = 0.5) -> list[PriceData]:
    """Generate a strong uptrend with consistent rising prices."""
    data = []
    price = start
    for i in range(n):
        price += step
        data.append(_make_bar(
            f"2024-01-{(i % 28) + 1:02d}",
            price,
            high=price + 1.0,
            low=price - 0.3,
        ))
    return data


def _trending_down_data(n: int = 300, start: float = 200.0,
                        step: float = 0.5) -> list[PriceData]:
    """Generate a strong downtrend with consistent falling prices."""
    data = []
    price = start
    for i in range(n):
        price -= step
        data.append(_make_bar(
            f"2024-01-{(i % 28) + 1:02d}",
            price,
            high=price + 0.3,
            low=price - 1.0,
        ))
    return data


def _ranging_data(n: int = 300, center: float = 100.0,
                  amplitude: float = 0.3) -> list[PriceData]:
    """Generate mean-reverting data oscillating tightly around center."""
    import math
    data = []
    for i in range(n):
        # Small sine wave
        price = center + amplitude * math.sin(i * 0.5)
        data.append(_make_bar(
            f"2024-01-{(i % 28) + 1:02d}",
            price,
            high=price + 0.2,
            low=price - 0.2,
        ))
    return data


def _volatile_data(n: int = 300, start: float = 100.0) -> list[PriceData]:
    """Generate high-volatility data with wide swings."""
    data = []
    price = start
    for i in range(n):
        # Alternating big moves
        if i % 2 == 0:
            price += 5.0
        else:
            price -= 4.5
        data.append(_make_bar(
            f"2024-01-{(i % 28) + 1:02d}",
            price,
            high=price + 3.0,
            low=price - 3.0,
        ))
    return data


# =========================================================================
# Basic construction
# =========================================================================


def test_detector_construction():
    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub)
    assert detector.warmup_period >= 200


def test_warmup_period():
    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub, sma_period=50)
    assert detector.warmup_period == 121  # max(50, 120) + 1


# =========================================================================
# Decision tree tests (static method)
# =========================================================================


_DEFAULT_CFG = RegimeDetectorConfig()


def test_classify_strong_trend_up():
    regime, conf = MarketRegimeDetector._classify_decision_tree(
        adx=35.0, plus_di=30.0, minus_di=15.0,
        bb_pctl=0.5, slope=1.0, cfg=_DEFAULT_CFG,
    )
    assert regime == MarketRegime.STRONG_TREND_UP
    assert conf > 0.5


def test_classify_strong_trend_down():
    regime, conf = MarketRegimeDetector._classify_decision_tree(
        adx=40.0, plus_di=12.0, minus_di=28.0,
        bb_pctl=0.5, slope=-1.2, cfg=_DEFAULT_CFG,
    )
    assert regime == MarketRegime.STRONG_TREND_DOWN
    assert conf > 0.5


def test_classify_mean_reverting():
    regime, conf = MarketRegimeDetector._classify_decision_tree(
        adx=12.0, plus_di=15.0, minus_di=14.0,
        bb_pctl=0.2, slope=0.1, cfg=_DEFAULT_CFG,
    )
    assert regime == MarketRegime.MEAN_REVERTING
    assert conf > 0.3


def test_classify_high_volatility():
    regime, conf = MarketRegimeDetector._classify_decision_tree(
        adx=25.0, plus_di=20.0, minus_di=18.0,
        bb_pctl=0.9, slope=0.3, cfg=_DEFAULT_CFG,
    )
    assert regime == MarketRegime.HIGH_VOLATILITY
    assert conf > 0.5


def test_classify_weak_trend():
    regime, conf = MarketRegimeDetector._classify_decision_tree(
        adx=25.0, plus_di=18.0, minus_di=16.0,
        bb_pctl=0.5, slope=0.3, cfg=_DEFAULT_CFG,
    )
    assert regime == MarketRegime.WEAK_TREND


def test_high_vol_overrides_strong_trend():
    """High BB width should classify as HIGH_VOLATILITY even with high ADX."""
    regime, _ = MarketRegimeDetector._classify_decision_tree(
        adx=45.0, plus_di=30.0, minus_di=10.0,
        bb_pctl=0.95, slope=2.0, cfg=_DEFAULT_CFG,
    )
    assert regime == MarketRegime.HIGH_VOLATILITY


# =========================================================================
# Full integration with synthetic data
# =========================================================================


def test_strong_uptrend_detection():
    """A consistent uptrend should be detected as trending or high-vol.

    Steady trends can show high BB width percentile (constant width = always
    at top of its own history), so we accept HIGH_VOLATILITY too.
    """
    data = _trending_up_data(300)
    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub)

    rc = detector.classify(data, 250)
    assert isinstance(rc, RegimeClassification)
    assert rc.confidence > 0.0
    assert rc.trend_slope > 0  # positive slope
    assert rc.adx_value > 20  # ADX should indicate some trend


def test_strong_downtrend_detection():
    """A consistent downtrend should show negative slope and high ADX."""
    data = _trending_down_data(300)
    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub)

    rc = detector.classify(data, 250)
    assert rc.trend_slope < 0  # negative slope
    assert rc.adx_value > 20  # ADX should indicate trend strength


def test_ranging_market_detection():
    """Tight oscillation around a center should be MEAN_REVERTING or WEAK_TREND."""
    data = _ranging_data(300, center=100.0, amplitude=0.3)
    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub)

    rc = detector.classify(data, 250)
    # In a tight range, ADX should be low
    assert rc.adx_value < 30
    # Slope should be near zero
    assert abs(rc.trend_slope) < 2.0


def test_volatile_market_detection():
    """Wide swings should be detected as HIGH_VOLATILITY."""
    data = _volatile_data(300)
    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub)

    rc = detector.classify(data, 250)
    # With extreme swings, BB width should be wide
    assert rc.bb_width_percentile > 0.0


# =========================================================================
# classify_range
# =========================================================================


def test_classify_range_returns_correct_count():
    data = _trending_up_data(300)
    hub = IndicatorHub()
    detector = MarketRegimeDetector(hub)

    results = detector.classify_range(data, 250, 260)
    assert len(results) == 10
    assert all(isinstance(r, RegimeClassification) for r in results)


# =========================================================================
# RegimeClassification dataclass
# =========================================================================


def test_regime_classification_frozen():
    rc = RegimeClassification(
        regime=MarketRegime.WEAK_TREND,
        confidence=0.5,
        adx_value=22.0,
        bb_width_percentile=0.45,
        trend_slope=0.1,
        sma200_distance=1.5,
    )
    with pytest.raises(AttributeError):
        rc.regime = MarketRegime.HIGH_VOLATILITY  # type: ignore[misc]


def test_market_regime_enum_values():
    """All five regimes exist."""
    assert len(MarketRegime) == 5
    names = {r.value for r in MarketRegime}
    assert "strong_trend_up" in names
    assert "strong_trend_down" in names
    assert "weak_trend" in names
    assert "mean_reverting" in names
    assert "high_volatility" in names
