"""Tests for the SignalAdvisory model and related types."""

from __future__ import annotations

import json

import pytest

from stockdownloader.model.signal_advisory import (
    AdvisoryAction,
    AdvisoryReasoning,
    OptionsAdvisory,
    SignalAdvisory,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_reasoning(**overrides: object) -> AdvisoryReasoning:
    defaults = dict(
        signal_confluence="10/15 indicators bullish (67%)",
        regime_alignment="Aligned with STRONG_TREND_UP",
        walk_forward_validated="Degradation ratio 0.87",
        key_bullish=["RSI oversold bounce", "MACD crossover"],
        key_bearish=["BB squeeze"],
        support_levels=[470.0, 468.5],
        resistance_levels=[478.0, 480.0],
    )
    defaults.update(overrides)
    return AdvisoryReasoning(**defaults)  # type: ignore[arg-type]


def _make_advisory(**overrides: object) -> SignalAdvisory:
    defaults: dict = dict(
        symbol="SPY",
        timestamp="2024-01-15T16:00:00-05:00",
        action=AdvisoryAction.BUY,
        confidence=0.82,
        regime="STRONG_TREND_UP",
        regime_confidence=0.91,
        entry_price=475.50,
        stop_loss=471.20,
        take_profit=483.90,
        risk_reward=1.95,
        position_size_pct=1.0,
        reasoning=_make_reasoning(),
    )
    defaults.update(overrides)
    return SignalAdvisory(**defaults)


# ------------------------------------------------------------------
# AdvisoryAction enum
# ------------------------------------------------------------------


class TestAdvisoryAction:
    def test_values(self) -> None:
        assert AdvisoryAction.STRONG_BUY.value == "STRONG_BUY"
        assert AdvisoryAction.HOLD.value == "HOLD"
        assert AdvisoryAction.STRONG_SELL.value == "STRONG_SELL"

    def test_all_members(self) -> None:
        names = {m.name for m in AdvisoryAction}
        assert names == {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}


# ------------------------------------------------------------------
# AdvisoryReasoning
# ------------------------------------------------------------------


class TestAdvisoryReasoning:
    def test_construction(self) -> None:
        r = _make_reasoning()
        assert r.signal_confluence == "10/15 indicators bullish (67%)"
        assert len(r.key_bullish) == 2
        assert len(r.support_levels) == 2

    def test_frozen(self) -> None:
        r = _make_reasoning()
        with pytest.raises(AttributeError):
            r.signal_confluence = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        r = AdvisoryReasoning(
            signal_confluence="x",
            regime_alignment="y",
            walk_forward_validated="z",
        )
        assert r.key_bullish == []
        assert r.key_bearish == []
        assert r.support_levels == []
        assert r.resistance_levels == []


# ------------------------------------------------------------------
# OptionsAdvisory
# ------------------------------------------------------------------


class TestOptionsAdvisory:
    def test_construction(self) -> None:
        oa = OptionsAdvisory(
            action="BUY", strike=476.0, dte=30, delta=0.50, est_premium=5.45,
        )
        assert oa.action == "BUY"
        assert oa.strike == 476.0
        assert oa.dte == 30

    def test_frozen(self) -> None:
        oa = OptionsAdvisory(
            action="SELL", strike=470.0, dte=30, delta=-0.30, est_premium=3.20,
        )
        with pytest.raises(AttributeError):
            oa.action = "HOLD"  # type: ignore[misc]


# ------------------------------------------------------------------
# SignalAdvisory
# ------------------------------------------------------------------


class TestSignalAdvisory:
    def test_construction(self) -> None:
        sa = _make_advisory()
        assert sa.symbol == "SPY"
        assert sa.action == AdvisoryAction.BUY
        assert sa.confidence == 0.82
        assert sa.call_advisory is None
        assert sa.put_advisory is None

    def test_frozen(self) -> None:
        sa = _make_advisory()
        with pytest.raises(AttributeError):
            sa.confidence = 0.99  # type: ignore[misc]

    def test_with_options(self) -> None:
        call = OptionsAdvisory(action="BUY", strike=476.0, dte=30, delta=0.50, est_premium=5.45)
        put = OptionsAdvisory(action="SELL", strike=470.0, dte=30, delta=-0.30, est_premium=3.20)
        sa = _make_advisory(call_advisory=call, put_advisory=put)
        assert sa.call_advisory is not None
        assert sa.call_advisory.strike == 476.0
        assert sa.put_advisory is not None
        assert sa.put_advisory.action == "SELL"

    # -- dedup_key --

    def test_dedup_key_basic(self) -> None:
        sa = _make_advisory()
        assert sa.dedup_key == "SPY:BUY:2024-01-15"

    def test_dedup_key_same_day_same_action_matches(self) -> None:
        sa1 = _make_advisory(timestamp="2024-01-15T10:00:00-05:00")
        sa2 = _make_advisory(timestamp="2024-01-15T16:00:00-05:00")
        assert sa1.dedup_key == sa2.dedup_key

    def test_dedup_key_different_action_differs(self) -> None:
        sa_buy = _make_advisory(action=AdvisoryAction.BUY)
        sa_sell = _make_advisory(action=AdvisoryAction.SELL)
        assert sa_buy.dedup_key != sa_sell.dedup_key

    def test_dedup_key_different_day_differs(self) -> None:
        sa1 = _make_advisory(timestamp="2024-01-15T16:00:00-05:00")
        sa2 = _make_advisory(timestamp="2024-01-16T16:00:00-05:00")
        assert sa1.dedup_key != sa2.dedup_key

    def test_dedup_key_different_symbol_differs(self) -> None:
        sa_spy = _make_advisory(symbol="SPY")
        sa_aapl = _make_advisory(symbol="AAPL")
        assert sa_spy.dedup_key != sa_aapl.dedup_key

    def test_dedup_key_short_timestamp(self) -> None:
        sa = _make_advisory(timestamp="2024-01")
        assert sa.dedup_key == "SPY:BUY:2024-01"

    # -- to_dict --

    def test_to_dict_roundtrip(self) -> None:
        sa = _make_advisory()
        d = sa.to_dict()

        assert d["symbol"] == "SPY"
        assert d["action"] == "BUY"  # string, not enum
        assert d["confidence"] == 0.82
        assert isinstance(d["reasoning"], dict)
        assert d["reasoning"]["key_bullish"] == ["RSI oversold bounce", "MACD crossover"]
        assert d["call_advisory"] is None

    def test_to_dict_with_options(self) -> None:
        call = OptionsAdvisory(action="BUY", strike=476.0, dte=30, delta=0.50, est_premium=5.45)
        sa = _make_advisory(call_advisory=call)
        d = sa.to_dict()
        assert d["call_advisory"]["strike"] == 476.0

    # -- to_json --

    def test_to_json_valid(self) -> None:
        sa = _make_advisory()
        j = sa.to_json()
        parsed = json.loads(j)
        assert parsed["symbol"] == "SPY"
        assert parsed["action"] == "BUY"
        assert parsed["reasoning"]["regime_alignment"] == "Aligned with STRONG_TREND_UP"

    def test_to_json_with_options(self) -> None:
        call = OptionsAdvisory(action="BUY", strike=476.0, dte=30, delta=0.50, est_premium=5.45)
        put = OptionsAdvisory(action="SELL", strike=470.0, dte=30, delta=-0.30, est_premium=3.20)
        sa = _make_advisory(call_advisory=call, put_advisory=put)
        parsed = json.loads(sa.to_json())
        assert parsed["call_advisory"]["action"] == "BUY"
        assert parsed["put_advisory"]["est_premium"] == 3.20
