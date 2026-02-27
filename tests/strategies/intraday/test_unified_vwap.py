"""Tests for the UnifiedVWAPStrategy — single-infra multi-mode dispatcher.

The unified strategy holds a SINGLE IntradayInfra and delegates to
existing standalone strategies' ``_evaluate_entry()`` methods with
priority dispatch (PS > ORB > PB > REV).
"""

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import HOLD


# -- Helpers (matching test_standalone_strategies.py patterns) --


def _make_bar(
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100_000,
) -> IntradayPriceData:
    return IntradayPriceData(
        date=date,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        adj_close=Decimal(str(close)),
        volume=volume,
    )


def _generate_session(
    trading_date: str = "2025-01-15",
    base_price: float = 500.0,
    num_bars: int = 78,
) -> list[IntradayPriceData]:
    """Generate a realistic intraday session of 5-minute bars."""
    bars = []
    price = base_price
    for i in range(num_bars):
        total_mins = 30 + i * 5
        hour = 9 + total_mins // 60
        minute = total_mins % 60
        dt_str = f"{trading_date} {hour:02d}:{minute:02d}:00-05:00"

        o = price
        h = price + 0.50
        l = price - 0.50
        c = price + 0.10 * (1 if i % 2 == 0 else -1)
        vol = 100_000 + i * 1000
        bars.append(_make_bar(dt_str, o, h, l, c, vol))
        price = c
    return bars


def _generate_multi_session(num_days: int = 16) -> list[IntradayPriceData]:
    """Generate multiple sessions (enough for warmup)."""
    all_bars = []
    base_price = 500.0
    for d in range(num_days):
        date_str = f"2025-01-{d + 1:02d}"
        session = _generate_session(
            trading_date=date_str,
            base_price=base_price,
            num_bars=78,
        )
        all_bars.extend(session)
        base_price += 0.5
    return all_bars


# ======================================================================
# Task 1: UnifiedVWAPConfig
# ======================================================================


class TestUnifiedVWAPConfig:
    """Verify UnifiedVWAPConfig imports and has correct defaults."""

    def test_import(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        assert UnifiedVWAPConfig is not None

    def test_defaults(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        cfg = UnifiedVWAPConfig()
        assert cfg.pb_enable is True
        assert cfg.ps_enable is True
        assert cfg.orb_enable is True
        assert cfg.rev_enable is True

    def test_mode_disable(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        cfg = UnifiedVWAPConfig(pb_enable=False, rev_enable=False)
        assert cfg.pb_enable is False
        assert cfg.ps_enable is True
        assert cfg.orb_enable is True
        assert cfg.rev_enable is False

    def test_inherits_infra_exit_config(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        from stockdownloader.strategies.intraday.base import InfraExitConfig
        cfg = UnifiedVWAPConfig()
        assert isinstance(cfg, InfraExitConfig)

    def test_is_frozen(self):
        from stockdownloader.strategies.intraday.unified_vwap import UnifiedVWAPConfig
        cfg = UnifiedVWAPConfig()
        with pytest.raises(AttributeError):
            cfg.pb_enable = False  # type: ignore[misc]
