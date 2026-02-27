"""Tests for FOMCDriftStrategy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.core.models.trade import IntradayAction
from stockdownloader.strategies.intraday.fomc_drift import (
    FOMCDriftConfig,
    FOMCDriftStrategy,
)
from stockdownloader.strategies.intraday.market_context import MarketContext


class TestFOMCDriftConfig:
    def test_defaults(self):
        c = FOMCDriftConfig()
        assert c.entry_bar == 1
        assert c.sl_atr_mult == Decimal("2.0")
        assert c.allow_shorts is False

    def test_json_round_trip(self):
        c = FOMCDriftConfig(vix_threshold=Decimal("25"))
        c2 = FOMCDriftConfig.from_json(c.to_json())
        assert c == c2

    def test_overrides(self):
        s = FOMCDriftStrategy(require_high_vix=False)
        assert s._c.require_high_vix is False


class TestFOMCDriftEntry:
    def test_enter_on_fomc_press_conf_day(self):
        ctx = MarketContext(
            vix_close=Decimal("22"),
            vix_sma20=Decimal("20"),
            vix_regime="mid",
            is_fomc_day=True,
            is_fomc_press_conf=True,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy()
        assert strat._should_enter_today(ctx) is True

    def test_no_entry_on_non_fomc_day(self):
        ctx = MarketContext(
            vix_close=Decimal("22"),
            vix_sma20=Decimal("20"),
            vix_regime="mid",
            is_fomc_day=False,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy()
        assert strat._should_enter_today(ctx) is False

    def test_no_entry_fomc_but_low_vix(self):
        """FOMC press conf + low VIX -> skip (require_high_vix=True)."""
        ctx = MarketContext(
            vix_close=Decimal("12"),
            vix_sma20=Decimal("13"),
            vix_regime="low",
            is_fomc_day=True,
            is_fomc_press_conf=True,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy(require_high_vix=True)
        assert strat._should_enter_today(ctx) is False

    def test_entry_fomc_low_vix_when_filter_off(self):
        """FOMC press conf + low VIX -> enter when filter disabled."""
        ctx = MarketContext(
            vix_close=Decimal("12"),
            vix_sma20=Decimal("13"),
            vix_regime="low",
            is_fomc_day=True,
            is_fomc_press_conf=True,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy(require_high_vix=False)
        assert strat._should_enter_today(ctx) is True

    def test_no_entry_on_fomc_non_press_conf(self):
        """FOMC day but not press conference -> skip."""
        ctx = MarketContext(
            vix_close=Decimal("22"),
            vix_sma20=Decimal("20"),
            vix_regime="mid",
            is_fomc_day=True,
            is_fomc_press_conf=False,
            is_opex=False,
            daily_rsi2=Decimal("50"),
            daily_close_above_sma200=True,
        )
        strat = FOMCDriftStrategy()
        assert strat._should_enter_today(ctx) is False

    def test_warmup_period(self):
        s = FOMCDriftStrategy()
        assert s.warmup_period == 78 * 15
