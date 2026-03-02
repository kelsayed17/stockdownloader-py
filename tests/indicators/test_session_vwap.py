"""Tests for session_vwap and session_vwap_bands."""

from decimal import Decimal

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.indicators import (
    SessionVWAP,
    session_vwap,
    session_vwap_bands,
)


def _make_bar(dt_str, high, low, close, volume=1000):
    o = Decimal(str(close))
    h = Decimal(str(high))
    l_ = Decimal(str(low))
    c = Decimal(str(close))
    return IntradayPriceData(
        date=dt_str, open=o, high=h, low=l_, close=c,
        adj_close=c, volume=volume,
    )


class TestSessionVwap:

    def test_single_bar(self):
        data = [_make_bar("2025-12-01 09:30:00-05:00", 101, 99, 100)]
        result = session_vwap(data, 0)
        # TP = (101 + 99 + 100) / 3 = 100
        assert result == Decimal("100")

    def test_two_bars_same_session(self):
        data = [
            _make_bar("2025-12-01 09:30:00-05:00", 102, 98, 100, volume=1000),
            _make_bar("2025-12-01 09:35:00-05:00", 104, 100, 102, volume=1000),
        ]
        result = session_vwap(data, 1)
        # TP1 = (102+98+100)/3 = 100, TP2 = (104+100+102)/3 = 102
        # VWAP = (100*1000 + 102*1000) / 2000 = 101
        assert abs(float(result) - 101.0) < 0.01

    def test_resets_at_day_boundary(self):
        data = [
            _make_bar("2025-12-01 15:55:00-05:00", 102, 98, 100, volume=1000),
            _make_bar("2025-12-02 09:30:00-05:00", 110, 108, 109, volume=1000),
        ]
        result = session_vwap(data, 1)
        # Should only use bar from 12/02, not 12/01
        tp2 = (110 + 108 + 109) / 3.0
        assert abs(float(result) - tp2) < 0.01

    def test_zero_volume(self):
        data = [_make_bar("2025-12-01 09:30:00-05:00", 101, 99, 100, volume=0)]
        result = session_vwap(data, 0)
        assert result == Decimal("0")


class TestSessionVwapBands:

    def test_returns_session_vwap_dataclass(self):
        data = [
            _make_bar("2025-12-01 09:30:00-05:00", 102, 98, 100, volume=1000),
            _make_bar("2025-12-01 09:35:00-05:00", 104, 100, 102, volume=1000),
        ]
        bands = session_vwap_bands(data, 1)
        assert isinstance(bands, SessionVWAP)
        assert bands.vwap > Decimal("0")
        assert bands.upper_1 >= bands.vwap
        assert bands.lower_1 <= bands.vwap

    def test_bands_symmetry(self):
        data = [
            _make_bar("2025-12-01 09:30:00-05:00", 102, 98, 100, volume=1000),
            _make_bar("2025-12-01 09:35:00-05:00", 104, 100, 102, volume=1000),
        ]
        bands = session_vwap_bands(data, 1)
        # Upper and lower should be equidistant from VWAP
        upper_dist = bands.upper_1 - bands.vwap
        lower_dist = bands.vwap - bands.lower_1
        assert abs(float(upper_dist - lower_dist)) < 0.001

    def test_half_sigma_bands(self):
        data = [
            _make_bar("2025-12-01 09:30:00-05:00", 102, 98, 100, volume=1000),
            _make_bar("2025-12-01 09:35:00-05:00", 104, 100, 102, volume=1000),
        ]
        bands = session_vwap_bands(data, 1)
        # Half-sigma bands should be closer to VWAP than 1-sigma bands
        assert abs(float(bands.upper_05 - bands.vwap)) < abs(float(bands.upper_1 - bands.vwap))
        assert abs(float(bands.vwap - bands.lower_05)) < abs(float(bands.vwap - bands.lower_1))

    def test_zero_volume_returns_zero_bands(self):
        data = [_make_bar("2025-12-01 09:30:00-05:00", 100, 100, 100, volume=0)]
        bands = session_vwap_bands(data, 0)
        assert bands.vwap == Decimal("0")
        assert bands.std_dev == Decimal("0")
