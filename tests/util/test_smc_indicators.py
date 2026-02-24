"""Tests for SMC indicator primitives."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import PriceData
from stockdownloader.util.indicators.smc import (
    StructureState,
    SwingPoint,
    _EMPTY_STRUCTURE,
    impulse_strength,
    is_liquidity_sweep_high,
    is_liquidity_sweep_low,
    is_swing_high,
    is_swing_low,
    zone_from_impulse_origin,
)

_D = Decimal


def _bar(high: str, low: str, close: str, open_: str = "0") -> PriceData:
    """Create a minimal PriceData for testing."""
    h, l, c = _D(high), _D(low), _D(close)
    o = _D(open_) if open_ != "0" else c
    return PriceData(date="2024-01-01", open=o, high=h, low=l, close=c, adj_close=c, volume=1000)


# Synthetic data with a clear swing high at index 5 and swing low at index 10
_SWING_DATA = [
    _bar("100", "98", "99"),    # 0
    _bar("101", "99", "100"),   # 1
    _bar("102", "100", "101"),  # 2
    _bar("103", "101", "102"),  # 3
    _bar("104", "102", "103"),  # 4
    _bar("108", "105", "107"),  # 5 — swing high (highest in ±5)
    _bar("106", "104", "105"),  # 6
    _bar("105", "103", "104"),  # 7
    _bar("104", "102", "103"),  # 8
    _bar("103", "101", "102"),  # 9
    _bar("100", "96", "97"),    # 10 — swing low (lowest in ±5)
    _bar("101", "98", "100"),   # 11
    _bar("102", "99", "101"),   # 12
    _bar("103", "100", "102"),  # 13
    _bar("104", "101", "103"),  # 14
    _bar("105", "102", "104"),  # 15
]


class TestSwingDetection:
    """Tests for is_swing_high and is_swing_low."""

    def test_swing_high_detected(self) -> None:
        assert is_swing_high(_SWING_DATA, 5, lookback=5) is True

    def test_swing_low_detected(self) -> None:
        assert is_swing_low(_SWING_DATA, 10, lookback=5) is True

    def test_no_swing_high_at_non_pivot(self) -> None:
        assert is_swing_high(_SWING_DATA, 3, lookback=3) is False

    def test_no_swing_low_at_non_pivot(self) -> None:
        assert is_swing_low(_SWING_DATA, 3, lookback=3) is False

    def test_insufficient_left_bars(self) -> None:
        assert is_swing_high(_SWING_DATA, 2, lookback=5) is False

    def test_insufficient_right_bars(self) -> None:
        assert is_swing_high(_SWING_DATA, 14, lookback=5) is False

    def test_small_lookback(self) -> None:
        # With lookback=2, bar 5 should still be highest in ±2
        assert is_swing_high(_SWING_DATA, 5, lookback=2) is True

    def test_swing_low_small_lookback(self) -> None:
        assert is_swing_low(_SWING_DATA, 10, lookback=2) is True


class TestLiquiditySweep:
    """Tests for liquidity sweep detection."""

    def test_sweep_high_detected(self) -> None:
        # Bar wicks above 108 but closes at 106 (below 108)
        bar = _bar("109", "104", "106", "105")
        assert is_liquidity_sweep_high(bar, _D("108"), _D("1.0"), _D("1.5")) is True

    def test_sweep_high_body_above(self) -> None:
        # Bar closes above the level — not a sweep
        bar = _bar("110", "107", "109", "108")
        assert is_liquidity_sweep_high(bar, _D("108"), _D("1.0")) is False

    def test_sweep_high_no_wick(self) -> None:
        # Bar doesn't reach the level
        bar = _bar("107", "104", "106", "105")
        assert is_liquidity_sweep_high(bar, _D("108"), _D("1.0")) is False

    def test_sweep_high_overshoot_too_large(self) -> None:
        # Bar overshoots way past the level
        bar = _bar("115", "104", "106", "105")
        assert is_liquidity_sweep_high(bar, _D("108"), _D("1.0"), _D("0.3")) is False

    def test_sweep_low_detected(self) -> None:
        # Bar wicks below 96 but closes at 98 (above 96)
        bar = _bar("100", "95", "98", "99")
        assert is_liquidity_sweep_low(bar, _D("96"), _D("1.0"), _D("1.5")) is True

    def test_sweep_low_body_below(self) -> None:
        # Bar closes below the level — not a sweep
        bar = _bar("100", "94", "95", "99")
        assert is_liquidity_sweep_low(bar, _D("96"), _D("1.0")) is False

    def test_sweep_low_no_wick(self) -> None:
        bar = _bar("100", "97", "98", "99")
        assert is_liquidity_sweep_low(bar, _D("96"), _D("1.0")) is False

    def test_zero_level_returns_false(self) -> None:
        bar = _bar("100", "95", "98")
        assert is_liquidity_sweep_high(bar, _D("0"), _D("1.0")) is False
        assert is_liquidity_sweep_low(bar, _D("0"), _D("1.0")) is False

    def test_zero_atr_returns_false(self) -> None:
        bar = _bar("100", "95", "98")
        assert is_liquidity_sweep_high(bar, _D("99"), _D("0")) is False
        assert is_liquidity_sweep_low(bar, _D("96"), _D("0")) is False


class TestImpulseStrength:
    """Tests for impulse_strength."""

    def test_basic_calculation(self) -> None:
        # Move from 99 to 107: |107-99|/2 = 4.0 ATR
        result = impulse_strength(_SWING_DATA, 0, 5, _D("2.0"))
        assert result == _D("4.0")

    def test_zero_atr(self) -> None:
        assert impulse_strength(_SWING_DATA, 0, 5, _D("0")) == _D("0")

    def test_negative_start(self) -> None:
        assert impulse_strength(_SWING_DATA, -1, 5, _D("2.0")) == _D("0")


class TestZoneFromImpulseOrigin:
    """Tests for zone_from_impulse_origin."""

    def test_basic_zone(self) -> None:
        top, bot = zone_from_impulse_origin(_SWING_DATA, 0, zone_bars=2)
        # Bars 0-1: highs=100,101; lows=98,99
        assert top == _D("101")
        assert bot == _D("98")

    def test_single_bar_zone(self) -> None:
        top, bot = zone_from_impulse_origin(_SWING_DATA, 5, zone_bars=1)
        assert top == _D("108")
        assert bot == _D("105")

    def test_negative_index(self) -> None:
        top, bot = zone_from_impulse_origin(_SWING_DATA, -1)
        assert top == _D("0")
        assert bot == _D("0")


class TestEmptyStructure:
    """Tests for _EMPTY_STRUCTURE sentinel."""

    def test_trend_zero(self) -> None:
        assert _EMPTY_STRUCTURE.trend == 0

    def test_no_bos(self) -> None:
        assert _EMPTY_STRUCTURE.bos_up is False
        assert _EMPTY_STRUCTURE.bos_down is False

    def test_zones_invalid(self) -> None:
        assert _EMPTY_STRUCTURE.demand_zone_valid is False
        assert _EMPTY_STRUCTURE.supply_zone_valid is False


class TestSwingPoint:
    """Tests for SwingPoint dataclass."""

    def test_frozen(self) -> None:
        sp = SwingPoint(index=5, price=_D("108"), kind="high")
        with pytest.raises(AttributeError):
            sp.price = _D("0")  # type: ignore[misc]

    def test_fields(self) -> None:
        sp = SwingPoint(index=10, price=_D("96"), kind="low")
        assert sp.index == 10
        assert sp.price == _D("96")
        assert sp.kind == "low"
