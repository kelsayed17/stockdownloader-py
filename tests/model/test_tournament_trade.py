"""Tests for TournamentTrade."""

from decimal import Decimal

import pytest

from stockdownloader.model.tournament_trade import TournamentTrade
from stockdownloader.model.trade import Direction


def _make_trade(**overrides):
    defaults = dict(
        trade_id=1,
        direction=Direction.LONG,
        signal_type="PB",
        entry_datetime="2025-12-01 10:00:00-05:00",
        exit_datetime="2025-12-01 11:30:00-05:00",
        entry_price=Decimal("680.00"),
        original_exit_price=Decimal("681.50"),
        original_pnl=Decimal("1.50"),
        stop_distance=Decimal("1.00"),
        shares=1,
    )
    defaults.update(overrides)
    return TournamentTrade(**defaults)


class TestTournamentTrade:

    def test_create_tournament_trade(self):
        t = _make_trade()
        assert t.trade_id == 1
        assert t.direction == Direction.LONG
        assert t.signal_type == "PB"
        assert t.entry_price == Decimal("680.00")

    def test_stop_price_long(self):
        t = _make_trade(direction=Direction.LONG, entry_price=Decimal("680"), stop_distance=Decimal("1"))
        assert t.stop_price == Decimal("679")

    def test_stop_price_short(self):
        t = _make_trade(direction=Direction.SHORT, entry_price=Decimal("680"), stop_distance=Decimal("1"))
        assert t.stop_price == Decimal("681")

    def test_risk_per_share(self):
        t = _make_trade(stop_distance=Decimal("0.96"))
        assert t.stop_distance == Decimal("0.96")

    def test_original_r_multiple(self):
        t = _make_trade(original_pnl=Decimal("2.00"), stop_distance=Decimal("1.00"))
        assert t.original_r_multiple == Decimal("2.0000")

    def test_original_r_multiple_negative(self):
        t = _make_trade(original_pnl=Decimal("-0.50"), stop_distance=Decimal("1.00"))
        assert t.original_r_multiple == Decimal("-0.5000")

    def test_entry_trading_date(self):
        t = _make_trade(entry_datetime="2025-12-01 10:00:00-05:00")
        assert t.entry_trading_date == "2025-12-01"

    def test_str_representation(self):
        t = _make_trade()
        s = str(t)
        assert "TournamentTrade" in s
        assert "#1" in s

    def test_validation_no_direction(self):
        with pytest.raises(ValueError, match="direction"):
            _make_trade(direction=None)

    def test_validation_negative_entry_price(self):
        with pytest.raises(ValueError, match="entry_price"):
            _make_trade(entry_price=Decimal("-1"))

    def test_validation_zero_stop_distance(self):
        with pytest.raises(ValueError, match="stop_distance"):
            _make_trade(stop_distance=Decimal("0"))

    def test_validation_zero_shares(self):
        with pytest.raises(ValueError, match="shares"):
            _make_trade(shares=0)
