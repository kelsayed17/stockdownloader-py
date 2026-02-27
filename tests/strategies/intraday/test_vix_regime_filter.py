"""Tests for VixFilteredStrategy."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from stockdownloader.core.models.trade import IntradayAction, IntradaySignal
from stockdownloader.strategies.intraday.vix_regime_filter import VixFilteredStrategy
from stockdownloader.strategies.intraday.market_context import MarketContext


def _mock_market_ctx(vix_regime: str) -> MarketContext:
    return MarketContext(
        vix_close=Decimal("20"),
        vix_sma20=Decimal("18"),
        vix_regime=vix_regime,
        is_fomc_day=False,
        is_fomc_press_conf=False,
        is_opex=False,
        daily_rsi2=Decimal("50"),
        daily_close_above_sma200=True,
    )


class TestVixFilteredStrategy:
    def test_blocks_in_disallowed_regime(self):
        """Regime check: 'low' and 'mid' are blocked when only high/extreme allowed."""
        inner = MagicMock()
        inner.name = "Inner"
        inner.warmup_period = 100

        wrapped = VixFilteredStrategy(
            inner=inner,
            allowed_regimes=["high", "extreme"],
        )
        assert wrapped._is_regime_allowed("low") is False
        assert wrapped._is_regime_allowed("mid") is False
        assert wrapped._is_regime_allowed("high") is True

    def test_allows_in_permitted_regime(self):
        wrapped = VixFilteredStrategy(
            inner=MagicMock(),
            allowed_regimes=["mid", "high"],
        )
        assert wrapped._is_regime_allowed("mid") is True
        assert wrapped._is_regime_allowed("high") is True

    def test_name_includes_vix_label(self):
        inner = MagicMock()
        inner.name = "OR Reversal"
        wrapped = VixFilteredStrategy(inner=inner, allowed_regimes=["mid", "high"])
        assert "VIX" in wrapped.name
        assert "OR Reversal" in wrapped.name

    def test_warmup_delegates_to_inner(self):
        inner = MagicMock()
        inner.warmup_period = 1170
        wrapped = VixFilteredStrategy(inner=inner)
        assert wrapped.warmup_period == 1170

    def test_default_regimes_are_mid_high(self):
        wrapped = VixFilteredStrategy(inner=MagicMock())
        assert wrapped._is_regime_allowed("mid") is True
        assert wrapped._is_regime_allowed("high") is True
        assert wrapped._is_regime_allowed("low") is False
