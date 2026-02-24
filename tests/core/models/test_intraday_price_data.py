"""Tests for IntradayPriceData."""

from decimal import Decimal

import pytest

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.price import PriceData


def _make_intraday(dt_str="2025-11-13 09:30:00-05:00", close="680.06"):
    p = Decimal(close)
    return IntradayPriceData(
        date=dt_str,
        open=Decimal("680.50"),
        high=Decimal("680.71"),
        low=Decimal("679.78"),
        close=p,
        adj_close=p,
        volume=3099060,
    )


class TestIntradayPriceData:

    def test_create_intraday_price_data(self):
        ipd = _make_intraday()
        assert ipd.close == Decimal("680.06")
        assert ipd.volume == 3099060

    def test_datetime_parsed(self):
        ipd = _make_intraday()
        dt = ipd.datetime_parsed
        assert dt.hour == 9
        assert dt.minute == 30
        assert dt.year == 2025
        assert dt.month == 11
        assert dt.day == 13

    def test_trading_date(self):
        ipd = _make_intraday()
        assert ipd.trading_date == "2025-11-13"

    def test_time_str(self):
        ipd = _make_intraday()
        assert ipd.time_str == "09:30:00"

    def test_different_datetime(self):
        ipd = _make_intraday("2026-01-15 14:45:00-05:00")
        assert ipd.trading_date == "2026-01-15"
        assert ipd.time_str == "14:45:00"
        dt = ipd.datetime_parsed
        assert dt.hour == 14
        assert dt.minute == 45

    def test_is_subclass_of_price_data(self):
        ipd = _make_intraday()
        assert isinstance(ipd, PriceData)

    def test_frozen(self):
        ipd = _make_intraday()
        with pytest.raises(AttributeError):
            ipd.close = Decimal("999")

    def test_compatible_with_sequence_of_price_data(self):
        """IntradayPriceData should work in lists typed as list[PriceData]."""
        data: list[PriceData] = [_make_intraday(), _make_intraday("2025-11-13 09:35:00-05:00")]
        assert len(data) == 2
        assert data[0].close == Decimal("680.06")
