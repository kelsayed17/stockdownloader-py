"""Tests for StreamingAnchoredVWAP accumulator."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stockdownloader.model.price_data import PriceData
from stockdownloader.util.indicators.volume import (
    StreamingAnchoredVWAP,
    StreamingSessionVWAP,
)


def _bar(date: str, high: str, low: str, close: str, volume: int) -> PriceData:
    """Create a PriceData bar for testing."""
    o = Decimal(close)  # open = close for simplicity
    h = Decimal(high)
    l = Decimal(low)
    c = Decimal(close)
    return PriceData(date=date, open=o, high=h, low=l, close=c, adj_close=c, volume=volume)


# ── Synthetic data spanning two FOMC anchors ──
# FOMC: 2024-01-31, 2024-03-20
_DATA = [
    # Day 1: 2024-01-30 (before first FOMC in data range — invalid)
    _bar("2024-01-30 09:30:00", "100", "99", "99.5", 1000),
    _bar("2024-01-30 09:35:00", "100.5", "99.5", "100", 1200),
    # Day 2: 2024-01-31 (FOMC day — accumulator resets and becomes valid)
    _bar("2024-01-31 09:30:00", "101", "100", "100.5", 1500),
    _bar("2024-01-31 09:35:00", "101.5", "100.5", "101", 1300),
    # Day 3: 2024-02-01 (continues accumulating — no reset)
    _bar("2024-02-01 09:30:00", "102", "101", "101.5", 1400),
    _bar("2024-02-01 09:35:00", "102.5", "101.5", "102", 1100),
    # Day 4: 2024-03-20 (new FOMC — accumulator resets)
    _bar("2024-03-20 09:30:00", "105", "104", "104.5", 2000),
    _bar("2024-03-20 09:35:00", "105.5", "104.5", "105", 1800),
]


class TestStreamingAnchoredVWAP:
    """Core accumulator tests."""

    def test_invalid_before_first_anchor(self) -> None:
        acc = StreamingAnchoredVWAP("fomc")
        # Bar 0 is 2024-01-30, before first known FOMC (2024-01-31)
        vwap, std = acc.update(_DATA, 0)
        assert vwap == Decimal("0")
        assert std == Decimal("0")
        assert acc.valid is False

    def test_becomes_valid_on_anchor(self) -> None:
        acc = StreamingAnchoredVWAP("fomc")
        # Process through first FOMC day
        acc.update(_DATA, 1)  # Still before anchor
        assert acc.valid is False
        vwap, std = acc.update(_DATA, 2)  # 2024-01-31 09:30 = FOMC day
        assert acc.valid is True
        assert vwap > Decimal("0")

    def test_no_reset_on_session_boundary(self) -> None:
        """AVWAP should NOT reset on new trading day (unlike session VWAP)."""
        acc = StreamingAnchoredVWAP("fomc")
        # Process through 2024-01-31 and 2024-02-01
        acc.update(_DATA, 3)  # End of FOMC day
        vwap_fomc_end, _ = acc.update(_DATA, 3)

        # Next day — should continue accumulating, not reset
        vwap_next, _ = acc.update(_DATA, 4)
        # AVWAP should be valid and non-zero (it doesn't reset)
        assert vwap_next > Decimal("0")
        assert acc.current_anchor == "2024-01-31"

    def test_session_vwap_resets_but_avwap_does_not(self) -> None:
        """Compare: session VWAP resets on day boundary, AVWAP does not."""
        avwap = StreamingAnchoredVWAP("fomc")
        svwap = StreamingSessionVWAP()

        # End of 2024-01-31 session
        avwap.update(_DATA, 3)
        svwap.update(_DATA, 3)

        # Start of 2024-02-01 — session VWAP resets, AVWAP continues
        avwap_val, _ = avwap.update(_DATA, 4)
        svwap_val, _ = svwap.update(_DATA, 4)

        # Session VWAP is calculated from just bar 4 (2024-02-01 09:30)
        # AVWAP is calculated from bars 2-4 (2024-01-31 + 2024-02-01)
        # They should differ
        assert avwap_val != svwap_val

    def test_resets_on_new_anchor(self) -> None:
        acc = StreamingAnchoredVWAP("fomc")
        # Process up to end of 2024-02-01
        acc.update(_DATA, 5)
        assert acc.current_anchor == "2024-01-31"

        # Process 2024-03-20 (new FOMC)
        acc.update(_DATA, 6)
        assert acc.current_anchor == "2024-03-20"

    def test_history_lookup(self) -> None:
        acc = StreamingAnchoredVWAP("fomc")
        acc.update(_DATA, 7)  # Process all bars

        # Can look up any previous bar
        v0, _ = acc.update(_DATA, 0)
        assert v0 == Decimal("0")  # Before anchor

        v2, _ = acc.update(_DATA, 2)
        assert v2 > Decimal("0")  # On FOMC day

    def test_reset_clears_state(self) -> None:
        acc = StreamingAnchoredVWAP("fomc")
        acc.update(_DATA, 5)
        assert acc.valid is True

        acc.reset()
        assert acc.valid is False
        assert acc.current_anchor == ""

    def test_anchor_type_stored(self) -> None:
        acc = StreamingAnchoredVWAP("fomc")
        assert acc._anchor_type == "fomc"

    def test_nonzero_std_dev(self) -> None:
        """Std dev should be non-zero after multiple bars."""
        acc = StreamingAnchoredVWAP("fomc")
        # Process enough bars to have variance
        _, std = acc.update(_DATA, 5)
        assert std > Decimal("0")
