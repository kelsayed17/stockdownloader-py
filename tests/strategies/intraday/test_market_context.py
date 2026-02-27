"""Tests for MarketContext and FileMarketContextProvider."""
from __future__ import annotations

import csv
from decimal import Decimal

import pytest

from stockdownloader.strategies.intraday.market_context import (
    MarketContext,
    FileMarketContextProvider,
    vix_regime_from_level,
)


class TestMarketContext:
    def test_vix_regime_low(self):
        ctx = MarketContext(
            vix_close=Decimal("12.5"),
            vix_sma20=Decimal("14.0"),
            vix_regime="low",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("45"),
            daily_close_above_sma200=True,
        )
        assert ctx.vix_regime == "low"

    def test_vix_regime_from_level(self):
        assert vix_regime_from_level(Decimal("12")) == "low"
        assert vix_regime_from_level(Decimal("18")) == "mid"
        assert vix_regime_from_level(Decimal("28")) == "high"
        assert vix_regime_from_level(Decimal("40")) == "extreme"

    def test_vix_regime_boundaries(self):
        assert vix_regime_from_level(Decimal("14.99")) == "low"
        assert vix_regime_from_level(Decimal("15")) == "mid"
        assert vix_regime_from_level(Decimal("24.99")) == "mid"
        assert vix_regime_from_level(Decimal("25")) == "high"
        assert vix_regime_from_level(Decimal("34.99")) == "high"
        assert vix_regime_from_level(Decimal("35")) == "extreme"


class TestFileMarketContextProvider:
    def test_load_and_get_context(self, tmp_path):
        csv_path = tmp_path / "market_ctx.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "vix_close", "vix_sma20", "is_fomc",
                         "is_fomc_pc", "is_opex", "rsi2", "above_sma200"])
            w.writerow(["2024-03-15", "14.5", "15.2", "0", "0", "1", "42.3", "1"])
            w.writerow(["2024-03-18", "22.1", "16.0", "1", "1", "0", "3.2", "1"])
        provider = FileMarketContextProvider(str(csv_path))
        ctx = provider.get_context("2024-03-15")
        assert ctx is not None
        assert ctx.vix_close == Decimal("14.5")
        assert ctx.is_opex is True
        assert ctx.is_fomc_day is False

    def test_get_context_missing_date_returns_none(self, tmp_path):
        csv_path = tmp_path / "market_ctx.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "vix_close", "vix_sma20", "is_fomc",
                         "is_fomc_pc", "is_opex", "rsi2", "above_sma200"])
            w.writerow(["2024-03-15", "14.5", "15.2", "0", "0", "0", "42.3", "1"])
        provider = FileMarketContextProvider(str(csv_path))
        assert provider.get_context("2024-03-20") is None

    def test_vix_regime_computed_on_load(self, tmp_path):
        csv_path = tmp_path / "market_ctx.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "vix_close", "vix_sma20", "is_fomc",
                         "is_fomc_pc", "is_opex", "rsi2", "above_sma200"])
            w.writerow(["2024-03-18", "28.0", "20.0", "0", "0", "0", "50", "1"])
        provider = FileMarketContextProvider(str(csv_path))
        ctx = provider.get_context("2024-03-18")
        assert ctx is not None
        assert ctx.vix_regime == "high"

    def test_fomc_flags_parsed(self, tmp_path):
        csv_path = tmp_path / "market_ctx.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "vix_close", "vix_sma20", "is_fomc",
                         "is_fomc_pc", "is_opex", "rsi2", "above_sma200"])
            w.writerow(["2024-03-18", "22.1", "16.0", "1", "1", "0", "3.2", "0"])
        provider = FileMarketContextProvider(str(csv_path))
        ctx = provider.get_context("2024-03-18")
        assert ctx is not None
        assert ctx.is_fomc_day is True
        assert ctx.is_fomc_press_conf is True
        assert ctx.daily_close_above_sma200 is False
