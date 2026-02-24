"""End-to-end test that exercises all utility classes working in concert:
  DateHelper + FileHelper + BigDecimalMath + CsvParser
  + MovingAverageCalculator

This test validates that the utility layer correctly supports the
application's data processing, file I/O, date handling, mathematical
computations that are used across the entire system.
"""
from __future__ import annotations

import io
import re
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.data.parsers import CsvPriceDataLoader
from stockdownloader.core.models.price import PriceData
from stockdownloader.core.math import (
    average,
    divide,
    percent_change,
    scale2,
)
from stockdownloader.core.io import CsvParser
from stockdownloader.core.io import (
    MORNINGSTAR_FORMAT,
    YAHOO_EARNINGS_FORMAT,
    YAHOO_FORMAT,
    DateHelper,
    adjust_to_market_day,
)
from stockdownloader.core.io import (
    append_line,
    delete_file,
    read_csv_lines,
    read_lines,
    write_content,
    write_lines,
)
from stockdownloader.util.indicators import ema, sma


@pytest.fixture(scope="module")
def spy_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    assert len(data) > 0
    return data


# ========== DateHelper ==========


def test_date_helper_default_constructor_uses_today():
    helper = DateHelper()
    today = date.today()

    # Today market day should be today or a recent weekday
    today_market = helper.today_market
    assert today_market is not None
    assert today_market.weekday() != 5  # Not Saturday
    assert today_market.weekday() != 6  # Not Sunday


def test_date_helper_market_day_adjustment():
    # Saturday -> Friday
    saturday = date(2024, 1, 6)  # Saturday
    assert adjust_to_market_day(saturday) == date(2024, 1, 5)

    # Sunday -> Friday
    sunday = date(2024, 1, 7)
    assert adjust_to_market_day(sunday) == date(2024, 1, 5)

    # Monday stays Monday
    monday = date(2024, 1, 8)
    assert adjust_to_market_day(monday) == monday


def test_date_helper_tomorrow_is_after_today():
    helper = DateHelper(date(2024, 1, 10))  # Wednesday
    today = helper.today_market
    tomorrow = helper.tomorrow_market
    yesterday = helper.yesterday_market

    assert tomorrow > today, "Tomorrow should be after today"
    assert yesterday < today, "Yesterday should be before today"


def test_date_helper_formats():
    helper = DateHelper(date(2024, 6, 15))  # Saturday
    # Should adjust to Friday June 14
    assert helper.today == "06/14/2024"

    # Verify multiple format accessors
    assert helper.current_month is not None
    assert helper.current_day is not None
    assert helper.current_year is not None
    assert helper.six_months_ago is not None
    assert helper.from_month is not None
    assert helper.from_day is not None
    assert helper.from_year is not None


def test_date_helper_six_months_ago():
    helper = DateHelper(date(2024, 7, 15))
    six_months_ago = helper.six_months_ago
    # Should be around January 2024
    assert "01/15/2024" in six_months_ago


def test_date_helper_weekday_stress_test():
    # Test every day in a year to ensure no crashes
    start = date(2024, 1, 1)
    for i in range(366):
        d = date.fromordinal(start.toordinal() + i)
        helper = DateHelper(d)

        # These should never throw
        assert helper.today_market is not None
        assert helper.tomorrow_market is not None
        assert helper.yesterday_market is not None

        # Market days should never be on weekends
        assert helper.today_market.weekday() != 5
        assert helper.today_market.weekday() != 6
        assert helper.tomorrow_market.weekday() != 5
        assert helper.tomorrow_market.weekday() != 6
        assert helper.yesterday_market.weekday() != 5
        assert helper.yesterday_market.weekday() != 6


# ========== FileHelper ==========


def test_file_helper_write_and_read_lines(tmp_path):
    filepath = str(tmp_path / "test-lines.txt")
    lines = sorted({"AAPL", "MSFT", "GOOGL", "AMZN"})

    write_lines(lines, filepath)
    read_back = read_lines(filepath)

    assert len(read_back) == 4
    assert "AAPL" in read_back
    assert "MSFT" in read_back
    assert "GOOGL" in read_back
    assert "AMZN" in read_back


def test_file_helper_append_line(tmp_path):
    filepath = str(tmp_path / "test-append.txt")

    append_line("TICK1", filepath)
    append_line("TICK2", filepath)
    append_line("TICK3", filepath)

    lines = read_lines(filepath)
    assert len(lines) == 3
    assert "TICK1" in lines
    assert "TICK2" in lines
    assert "TICK3" in lines


def test_file_helper_write_content(tmp_path):
    filepath = str(tmp_path / "test-content.txt")
    content = "[AAPL, MSFT, GOOGL]"

    write_content(content, filepath)
    lines = read_lines(filepath)

    # writeContent removes [ and ] characters
    assert len(lines) > 0
    joined = "".join(lines)
    assert "[" not in joined
    assert "]" not in joined


def test_file_helper_read_csv_lines(tmp_path):
    filepath = str(tmp_path / "test-csv.txt")
    write_lines(["AAPL,MSFT,GOOGL"], filepath)

    tickers = read_csv_lines(filepath)
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "GOOGL" in tickers


def test_file_helper_delete_file(tmp_path):
    filepath = str(tmp_path / "test-delete.txt")
    write_lines(["data"], filepath)

    assert delete_file(filepath)
    # Reading deleted file should return empty
    lines = read_lines(filepath)
    assert len(lines) == 0


def test_file_helper_read_nonexistent_file_returns_empty(tmp_path):
    lines = read_lines(str(tmp_path / "nonexistent.txt"))
    assert len(lines) == 0


def test_file_helper_delete_nonexistent_returns_false(tmp_path):
    assert not delete_file(str(tmp_path / "nonexistent.txt"))


# ========== FileHelper + DateHelper Integration ==========


def test_file_helper_and_date_helper_integration(tmp_path):
    # Simulate the incomplete file workflow from TrendAnalysisApp
    incomplete_file = str(tmp_path / "incomplete.txt")

    date_helper = DateHelper(date(2024, 3, 15))

    # Write some tickers with dates
    append_line(f"AAPL_{date_helper.today}", incomplete_file)
    append_line(f"MSFT_{date_helper.yesterday}", incomplete_file)

    incomplete = read_lines(incomplete_file)
    assert len(incomplete) == 2

    # Clean up
    delete_file(incomplete_file)


# ========== BigDecimalMath ==========


def test_big_decimal_math_division_safe_on_zero():
    result = divide(Decimal("10"), Decimal("0"))
    assert result == Decimal("0"), "Division by zero should return ZERO"


def test_big_decimal_math_division_with_custom_scale():
    result = divide(Decimal("1"), Decimal("3"), 2)
    assert result == Decimal("0.33")


def test_big_decimal_math_percent_change():
    from_val = Decimal("100")
    to_val = Decimal("110")
    pct_change = percent_change(from_val, to_val)
    assert Decimal("10.000000") == pct_change

    # Negative change
    neg_change = percent_change(Decimal("200"), Decimal("180"))
    assert neg_change < Decimal("0")

    # Zero from
    assert percent_change(Decimal("0"), Decimal("10")) == Decimal("0")


def test_big_decimal_math_average():
    avg = average(Decimal("10"), Decimal("20"), Decimal("30"))
    # (10 + 20 + 30) / 3 = 20
    assert Decimal("20") == avg.quantize(Decimal("1"), rounding=ROUND_HALF_UP)

    # Average skips zeros
    avg_with_zero = average(Decimal("10"), Decimal("0"), Decimal("20"))
    # (10 + 20) / 2 = 15
    assert Decimal("15") == avg_with_zero.quantize(Decimal("1"), rounding=ROUND_HALF_UP)

    # All zeros
    assert average(Decimal("0"), Decimal("0")) == Decimal("0")

    # Null/empty
    assert average() == Decimal("0")


def test_big_decimal_math_scale2():
    val = Decimal("3.14159265")
    scaled = scale2(val)
    assert scaled == Decimal("3.14")


# ========== BigDecimalMath + Real Data ==========


def test_big_decimal_math_with_real_spy_returns(spy_data):
    # Calculate daily returns using BigDecimalMath on real data
    for i in range(1, min(50, len(spy_data))):
        prev_close = spy_data[i - 1].close
        curr_close = spy_data[i].close

        daily_return = percent_change(prev_close, curr_close)
        assert daily_return is not None

        # Daily SPY returns should be within reasonable bounds (-10% to +10%)
        assert daily_return > Decimal("-10"), \
            f"Daily return should be > -10% on {spy_data[i].date}"
        assert daily_return < Decimal("10"), \
            f"Daily return should be < 10% on {spy_data[i].date}"


# ========== MovingAverageCalculator + Real Data ==========


def test_sma_computation_on_real_data(spy_data):
    period = 20
    for i in range(period, len(spy_data)):
        sma_val = sma(spy_data, i, period)

        # SMA should be between the min and max close in the window
        min_close = min(spy_data[j].close for j in range(i - period + 1, i + 1))
        max_close = max(spy_data[j].close for j in range(i - period + 1, i + 1))

        assert sma_val >= min_close, \
            f"SMA should be >= min close in window at index {i}"
        assert sma_val <= max_close, \
            f"SMA should be <= max close in window at index {i}"


def test_ema_computation_smoothness(spy_data):
    period = 12
    prev_ema = None
    direction_changes = 0
    prev_up = False

    for i in range(period, len(spy_data)):
        ema_val = ema(spy_data, i, period)
        assert ema_val > Decimal("0"), "EMA should always be positive"

        if prev_ema is not None:
            is_up = ema_val > prev_ema
            if i > period + 1 and is_up != prev_up:
                direction_changes += 1
            prev_up = is_up
        prev_ema = ema_val

    # EMA should be smoother than raw price data
    price_direction_changes = 0
    for i in range(1, len(spy_data)):
        price_up = spy_data[i].close > spy_data[i - 1].close
        prev_price_up = (
            i > 1 and spy_data[i - 1].close > spy_data[i - 2].close
        )
        if i > 1 and price_up != prev_price_up:
            price_direction_changes += 1

    assert direction_changes < price_direction_changes, \
        "EMA should have fewer direction changes than raw price"


def test_sma_and_ema_converge_on_flat_data():
    # For flat data, SMA and EMA should be the same
    flat_price = Decimal("100.00")
    flat_data = [
        PriceData(
            f"2024-01-{(i % 28) + 1:02d}",
            flat_price, flat_price, flat_price, flat_price, flat_price, 1000,
        )
        for i in range(50)
    ]

    sma_val = sma(flat_data, 49, 20)
    ema_val = ema(flat_data, 49, 20)

    assert flat_price == sma_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), \
        "SMA should equal the flat price"
    assert flat_price == ema_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), \
        "EMA should equal the flat price for flat data"


# ========== Full Utility Chain Integration ==========


def test_full_utility_chain_from_csv_to_analysis(tmp_path):
    import math

    # Step 1: Parse CSV data
    csv_lines = "Date,Open,High,Low,Close,Adj Close,Volume\n"
    for i in range(60):
        price = 100 + math.sin(i * 0.1) * 10
        csv_lines += (
            f"2024-01-{(i % 28) + 1:02d},"
            f"{price:.6f},{price + 1:.6f},{price - 1:.6f},"
            f"{price:.6f},{price:.6f},1000000\n"
        )

    with CsvParser(io.StringIO(csv_lines)) as parser:
        rows = parser.read_all()
        assert len(rows) == 61  # header + 60 data rows

    # Step 2: Load via CsvPriceDataLoader
    stream = io.BytesIO(csv_lines.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)
    assert len(data) == 60

    # Step 3: Compute moving averages
    sma20 = sma(data, 30, 20)
    ema12 = ema(data, 30, 12)
    assert sma20 is not None
    assert ema12 is not None

    # Step 4: Use BigDecimalMath to compute percent change
    pct_change = percent_change(data[0].close, data[-1].close)
    assert pct_change is not None

    # Step 5: Store and retrieve results via FileHelper
    result_file = str(tmp_path / "analysis-result.txt")
    append_line(f"SMA20: {sma20}", result_file)
    append_line(f"EMA12: {ema12}", result_file)
    append_line(f"PctChange: {pct_change}", result_file)

    results = read_lines(result_file)
    assert len(results) == 3

    delete_file(result_file)


def test_date_helper_formats_used_in_data_pipeline():
    # Verify DateHelper formats are compatible with PriceData date format
    helper = DateHelper(date(2024, 3, 15))
    yahoo_date = helper.today_market.strftime(YAHOO_FORMAT)

    # Should be in yyyy-MM-dd format, matching CSV data
    assert re.match(r"\d{4}-\d{2}-\d{2}", yahoo_date), \
        "Yahoo format should be yyyy-MM-dd"

    # Standard format for display
    std_date = helper.today
    assert re.match(r"\d{2}/\d{2}/\d{4}", std_date), \
        "Standard format should be MM/dd/yyyy"

    # Morningstar format for quarterly data
    ms_date = helper.today_market.strftime(MORNINGSTAR_FORMAT)
    assert re.match(r"\d{4}-\d{2}", ms_date), \
        "Morningstar format should be yyyy-MM"

    # Yahoo earnings format
    earnings_date = helper.today_market.strftime(YAHOO_EARNINGS_FORMAT)
    assert re.match(r"\d{8}", earnings_date), \
        "Yahoo earnings format should be yyyyMMdd"
