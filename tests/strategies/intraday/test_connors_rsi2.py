"""Tests for ConnorsRSI2Strategy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.core.models.trade import IntradayAction
from stockdownloader.strategies.intraday.connors_rsi2 import (
    ConnorsRSI2Config,
    ConnorsRSI2Strategy,
)
from stockdownloader.strategies.intraday.market_context import MarketContext


class TestConnorsRSI2Config:
    def test_defaults(self):
        c = ConnorsRSI2Config()
        assert c.rsi_period == 2
        assert c.rsi_threshold == Decimal("5")
        assert c.allow_shorts is False

    def test_json_round_trip(self):
        c = ConnorsRSI2Config(rsi_threshold=Decimal("10"))
        c2 = ConnorsRSI2Config.from_json(c.to_json())
        assert c == c2

    def test_overrides(self):
        s = ConnorsRSI2Strategy(rsi_threshold=Decimal("10"))
        assert s._c.rsi_threshold == Decimal("10")


class TestConnorsRSI2Entry:
    def test_long_on_oversold_rsi2(self):
        """Entry when daily RSI(2) < 5 and above SMA(200)."""
        ctx = MarketContext(
            vix_close=Decimal("18"),
            vix_sma20=Decimal("17"),
            vix_regime="mid",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("3.5"),
            daily_close_above_sma200=True,
        )
        strat = ConnorsRSI2Strategy()
        assert strat._should_enter_today(ctx) is True

    def test_no_entry_when_above_threshold(self):
        """No entry when RSI(2) > threshold."""
        ctx = MarketContext(
            vix_close=Decimal("18"),
            vix_sma20=Decimal("17"),
            vix_regime="mid",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("35"),
            daily_close_above_sma200=True,
        )
        strat = ConnorsRSI2Strategy()
        assert strat._should_enter_today(ctx) is False

    def test_no_entry_below_sma200(self):
        """No entry when daily close below SMA(200) -- bearish regime."""
        ctx = MarketContext(
            vix_close=Decimal("18"),
            vix_sma20=Decimal("17"),
            vix_regime="mid",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("3.5"),
            daily_close_above_sma200=False,
        )
        strat = ConnorsRSI2Strategy()
        assert strat._should_enter_today(ctx) is False

    def test_warmup_period(self):
        s = ConnorsRSI2Strategy()
        assert s.warmup_period == 78 * 15
