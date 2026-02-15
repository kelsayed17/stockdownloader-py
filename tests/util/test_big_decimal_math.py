"""Tests for big_decimal_math utility functions."""

from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.util.big_decimal_math import (
    divide,
    scale2,
    average,
    percent_change,
)


def test_divide_normal():
    result = divide(Decimal("10"), Decimal("3"))
    assert result > Decimal("3.33")
    assert result < Decimal("3.34")


def test_divide_by_zero_returns_zero():
    result = divide(Decimal("10"), Decimal("0"))
    assert Decimal("0").compare(result) == 0


def test_divide_with_scale():
    result = divide(Decimal("10"), Decimal("3"), 2)
    assert Decimal("3.33").compare(result) == 0


def test_divide_by_zero_with_scale_returns_zero():
    result = divide(Decimal("10"), Decimal("0"), 2)
    assert Decimal("0").compare(result) == 0



def test_scale2():
    result = scale2(Decimal("3.14159"))
    assert Decimal("3.14").compare(result) == 0


def test_scale2_rounds_up():
    result = scale2(Decimal("3.145"))
    assert Decimal("3.15").compare(result) == 0


def test_average_of_values():
    result = average(Decimal("10"), Decimal("20"), Decimal("30"))
    assert Decimal("20").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_average_ignores_zeros():
    result = average(Decimal("10"), Decimal("0"), Decimal("30"))
    assert Decimal("20").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_average_of_all_zeros():
    result = average(Decimal("0"), Decimal("0"))
    assert Decimal("0").compare(result) == 0


def test_average_of_empty_returns_zero():
    assert Decimal("0").compare(average()) == 0


def test_average_of_none_returns_zero():
    assert Decimal("0").compare(average(None)) == 0


def test_percent_change_positive():
    result = percent_change(Decimal("100"), Decimal("120"))
    assert Decimal("20").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_percent_change_negative():
    result = percent_change(Decimal("100"), Decimal("80"))
    assert Decimal("-20").compare(
        result.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ) == 0


def test_percent_change_from_zero():
    result = percent_change(Decimal("0"), Decimal("10"))
    assert Decimal("0").compare(result) == 0
