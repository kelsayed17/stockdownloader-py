"""Tests for shared report rounding utilities."""
from __future__ import annotations

from decimal import Decimal

from stockdownloader.backtest.report_helpers import scale2, scale3


class TestScale2:
    def test_rounds_down(self):
        assert scale2(Decimal("1.234")) == Decimal("1.23")

    def test_rounds_up(self):
        assert scale2(Decimal("1.235")) == Decimal("1.24")

    def test_no_change(self):
        assert scale2(Decimal("1.23")) == Decimal("1.23")

    def test_whole_number(self):
        assert scale2(Decimal("5")) == Decimal("5.00")

    def test_negative(self):
        assert scale2(Decimal("-3.456")) == Decimal("-3.46")

    def test_zero(self):
        assert scale2(Decimal("0")) == Decimal("0.00")


class TestScale3:
    def test_rounds_down(self):
        assert scale3(Decimal("1.2344")) == Decimal("1.234")

    def test_rounds_up(self):
        assert scale3(Decimal("1.2345")) == Decimal("1.235")

    def test_no_change(self):
        assert scale3(Decimal("1.234")) == Decimal("1.234")

    def test_whole_number(self):
        assert scale3(Decimal("5")) == Decimal("5.000")

    def test_negative(self):
        assert scale3(Decimal("-3.4567")) == Decimal("-3.457")
