"""Configuration for the DMI + VWAP intraday strategy."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(slots=True)
class DmiVwapConfig:
    """Tunable parameters for the DMI+VWAP strategy."""

    # DMI / ADX
    dmi_period: int = 14
    adx_threshold: Decimal = Decimal("25")

    # Risk management
    atr_period: int = 14
    sl_atr_mult: Decimal = Decimal("1.5")
    rr: Decimal = Decimal("2.0")

    # Session management
    bars_per_day: int = 78  # 5-min bars 09:30-16:00
    eod_exit_bar: int = 76  # exit ~2 bars before close (15:50)
    min_entry_bar: int = 3  # skip first 15 min for indicator warmup

    # Trade management
    min_hold_bars: int = 12  # hold at least 1 hr before neutral exit
    cooldown_bars: int = 6  # wait 30 min after exit before re-entry
    max_trades_per_day: int = 3  # cap daily trades to avoid overtrading

    # Filters
    require_adx_rising: bool = False
    min_di_spread: Decimal = Decimal("0")  # minimum +DI - -DI gap
