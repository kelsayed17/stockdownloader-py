"""MarketContext for daily regime data injected into intraday strategies.

Provides VIX levels, FOMC/OPEX calendar flags, and daily RSI(2)/SMA(200)
trend filters. Loaded from a pre-computed CSV and accessed per-session
via a provider protocol.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol


def vix_regime_from_level(vix: Decimal) -> str:
    """Classify VIX into regime buckets."""
    if vix < 15:
        return "low"
    if vix < 25:
        return "mid"
    if vix < 35:
        return "high"
    return "extreme"


@dataclass(slots=True)
class MarketContext:
    """Daily market regime data, computed externally and injected per-session."""

    vix_close: Decimal
    vix_sma20: Decimal
    vix_regime: str  # "low", "mid", "high", "extreme"
    is_fomc_day: bool
    is_fomc_press_conf: bool
    is_opex: bool
    daily_rsi2: Decimal
    daily_close_above_sma200: bool


class MarketContextProvider(Protocol):
    """Protocol for objects that supply MarketContext per trading date."""

    def get_context(self, trading_date: str) -> MarketContext | None: ...


class FileMarketContextProvider:
    """Load MarketContext from a pre-computed CSV file.

    CSV columns: date, vix_close, vix_sma20, is_fomc, is_fomc_pc,
                 is_opex, rsi2, above_sma200
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._data: dict[str, MarketContext] = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vix = Decimal(row["vix_close"])
                self._data[row["date"]] = MarketContext(
                    vix_close=vix,
                    vix_sma20=Decimal(row["vix_sma20"]),
                    vix_regime=vix_regime_from_level(vix),
                    is_fomc_day=row["is_fomc"] == "1",
                    is_fomc_press_conf=row["is_fomc_pc"] == "1",
                    is_opex=row["is_opex"] == "1",
                    daily_rsi2=Decimal(row["rsi2"]),
                    daily_close_above_sma200=row["above_sma200"] == "1",
                )

    def get_context(self, trading_date: str) -> MarketContext | None:
        return self._data.get(trading_date)
