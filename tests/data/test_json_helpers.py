"""Tests for json_helpers utility functions."""
from decimal import Decimal

from stockdownloader.data.json_helpers import (
    get_decimal,
    get_long,
    get_raw_decimal,
    get_raw_long,
)


class TestGetRawDecimal:
    """Tests for the get_raw_decimal helper."""

    def test_wrapped_format(self) -> None:
        obj = {"profitMargins": {"raw": 0.1576, "fmt": "15.76%"}}
        result = get_raw_decimal(obj, "profitMargins")
        assert result == Decimal("0.1576")

    def test_direct_value(self) -> None:
        obj = {"someField": 42.5}
        result = get_raw_decimal(obj, "someField")
        assert result == Decimal("42.5")

    def test_missing_field(self) -> None:
        obj = {}
        result = get_raw_decimal(obj, "missing")
        assert result == Decimal("0")

    def test_none_value(self) -> None:
        obj = {"field": None}
        result = get_raw_decimal(obj, "field")
        assert result == Decimal("0")

    def test_wrapped_with_none_raw(self) -> None:
        obj = {"field": {"raw": None, "fmt": "N/A"}}
        result = get_raw_decimal(obj, "field")
        assert result == Decimal("0")

    def test_invalid_value(self) -> None:
        obj = {"field": "not_a_number"}
        result = get_raw_decimal(obj, "field")
        assert result == Decimal("0")

    def test_negative_wrapped(self) -> None:
        obj = {"earningsGrowth": {"raw": -0.052, "fmt": "-5.20%"}}
        result = get_raw_decimal(obj, "earningsGrowth")
        assert result == Decimal("-0.052")

    def test_zero_wrapped(self) -> None:
        obj = {"field": {"raw": 0, "fmt": "0"}}
        result = get_raw_decimal(obj, "field")
        assert result == Decimal("0")
