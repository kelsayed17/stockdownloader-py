"""End-to-end test for the complete data loading pipeline and trade lifecycle.

Pipeline 1 (Data Loading):
  Raw CSV string -> CsvParser -> CsvPriceDataLoader -> list[PriceData]
  -> MovingAverageCalculator (SMA/EMA) -> strategy signals

Pipeline 2 (Trade Lifecycle):
  Strategy signal -> Trade creation (OPEN) -> position tracking
  -> Trade close (CLOSED) -> P/L calculation -> BacktestResult aggregation
  -> metric computation (return, win rate, Sharpe, drawdown, profit factor)

Exercises: CsvParser, CsvPriceDataLoader, PriceData, Trade, BacktestEngine,
BacktestResult, all strategy implementations, MovingAverageCalculator.
"""
from __future__ import annotations

import io
from decimal import Decimal, ROUND_HALF_UP

import pytest

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.data.data_parsers import CsvPriceDataLoader
from stockdownloader.model.price_data import PriceData
from stockdownloader.model.trade import Trade, Direction, TradeStatus
from stockdownloader.strategy.daily.simple_strategies import MACDStrategy
from stockdownloader.strategy.daily.simple_strategies import RSIStrategy
from stockdownloader.strategy.daily.simple_strategies import SMACrossoverStrategy
from stockdownloader.strategy.trading_strategy import Signal
from stockdownloader.util.parsers import CsvParser
from stockdownloader.util.technical import sma, ema


@pytest.fixture(scope="module")
def real_data(test_data_path):
    data = CsvPriceDataLoader.load_from_file(str(test_data_path))
    assert len(data) > 0
    return data


# ========== CsvParser -> CsvPriceDataLoader Pipeline ==========


def test_csv_parser_to_loader_pipeline():
    csv = (
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2024-01-02,100.00,105.00,99.00,103.50,103.50,5000000\n"
        "2024-01-03,103.50,108.00,102.00,107.25,107.25,6000000\n"
        "2024-01-04,107.25,110.00,106.00,108.75,108.75,4500000\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 3
    assert data[0].date == "2024-01-02"
    assert data[0].open == Decimal("100.00")
    assert data[0].high == Decimal("105.00")
    assert data[0].low == Decimal("99.00")
    assert data[0].close == Decimal("103.50")
    assert data[0].adj_close == Decimal("103.50")
    assert data[0].volume == 5000000


def test_csv_parser_handles_real_data_format(test_data_path):
    with open(test_data_path) as f:
        with CsvParser(f) as parser:
            header = parser.read_next()
            assert header is not None
            assert header[0] == "Date"
            assert header[1] == "Open"
            assert header[2] == "High"
            assert header[3] == "Low"
            assert header[4] == "Close"
            assert header[5] == "Adj Close"
            assert header[6] == "Volume"

            # Read first data row
            first_row = parser.read_next()
            assert first_row is not None
            assert first_row[0] == "2023-01-03"
            # Verify values are parseable as numbers
            Decimal(first_row[1])  # Should not throw
            Decimal(first_row[4])  # Should not throw
            int(first_row[6])  # Should not throw


def test_csv_parser_with_tab_separator():
    tsv = "Col1\tCol2\tCol3\nA\tB\tC\nD\tE\tF"
    with CsvParser(io.StringIO(tsv), separator='\t') as parser:
        header = parser.read_next()
        assert len(header) == 3
        assert header[0] == "Col1"
        assert header[2] == "Col3"

        row1 = parser.read_next()
        assert row1[0] == "A"
        assert row1[2] == "C"


def test_csv_parser_handles_quoted_fields():
    csv = 'Name,Value\n"Smith, John",100\n"O\'Brien",200'
    with CsvParser(io.StringIO(csv)) as parser:
        parser.read_next()  # skip header
        row1 = parser.read_next()
        assert row1[0] == "Smith, John"
        assert row1[1] == "100"


def test_csv_parser_read_all_returns_all_rows():
    csv = "A,B\n1,2\n3,4\n5,6"
    with CsvParser(io.StringIO(csv)) as parser:
        rows = parser.read_all()
        assert len(rows) == 4  # header + 3 data rows


def test_csv_parser_skip_lines():
    csv = "Header1\nHeader2\nData1,A\nData2,B"
    with CsvParser(io.StringIO(csv)) as parser:
        parser.skip_lines(2)
        data_row = parser.read_next()
        assert data_row[0] == "Data1"


def test_loader_skips_invalid_numeric_rows():
    csv = (
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2024-01-02,100.00,105.00,99.00,103.50,103.50,5000000\n"
        "2024-01-03,null,null,null,null,null,0\n"
        "2024-01-04,107.25,110.00,106.00,108.75,108.75,4500000\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 2, "Should skip the row with null values"
    assert data[0].date == "2024-01-02"
    assert data[1].date == "2024-01-04"


def test_loader_handles_missing_adj_close_and_volume():
    csv = (
        "Date,Open,High,Low,Close\n"
        "2024-01-02,100.00,105.00,99.00,103.50\n"
    )

    stream = io.BytesIO(csv.encode("utf-8"))
    data = CsvPriceDataLoader.load_from_stream(stream)

    assert len(data) == 1
    # adjClose should default to close, volume should default to 0
    assert data[0].close == data[0].adj_close
    assert data[0].volume == 0


# ========== Data -> Strategy Signal Pipeline ==========


def test_data_to_signal_pipeline_sma(real_data):
    strategy = SMACrossoverStrategy(20, 50)

    # Before warmup, all signals should be HOLD
    for i in range(strategy.warmup_period):
        assert strategy.evaluate(real_data, i) == Signal.HOLD, \
            f"Before warmup, signal should be HOLD at index {i}"

    # After warmup, signals depend on SMA crossover
    buy_count = 0
    sell_count = 0
    hold_count = 0
    for i in range(strategy.warmup_period, len(real_data)):
        signal = strategy.evaluate(real_data, i)
        if signal == Signal.BUY:
            buy_count += 1
        elif signal == Signal.SELL:
            sell_count += 1
        elif signal == Signal.HOLD:
            hold_count += 1

    # Should have at least some signals with real data
    assert buy_count + sell_count + hold_count > 0, \
        "Should evaluate signals after warmup"
    assert hold_count > buy_count, "HOLD should be more common than BUY"
    assert hold_count > sell_count, "HOLD should be more common than SELL"


def test_data_to_signal_pipeline_rsi(real_data):
    strategy = RSIStrategy(14, 30.0, 70.0)

    # Warmup period check
    assert strategy.warmup_period == 15  # period + 1

    buy_count = 0
    sell_count = 0
    for i in range(strategy.warmup_period, len(real_data)):
        signal = strategy.evaluate(real_data, i)
        if signal == Signal.BUY:
            buy_count += 1
        if signal == Signal.SELL:
            sell_count += 1

    # RSI signals should be relatively rare (extreme readings)
    total_bars = len(real_data) - strategy.warmup_period
    assert buy_count < total_bars // 2, "RSI BUY signals should be infrequent"
    assert sell_count < total_bars // 2, "RSI SELL signals should be infrequent"


def test_data_to_signal_pipeline_macd(real_data):
    strategy = MACDStrategy(12, 26, 9)

    assert strategy.warmup_period == 35  # slowPeriod + signalPeriod

    buy_count = 0
    sell_count = 0
    for i in range(strategy.warmup_period, len(real_data)):
        signal = strategy.evaluate(real_data, i)
        if signal == Signal.BUY:
            buy_count += 1
        if signal == Signal.SELL:
            sell_count += 1

    # MACD should generate both buy and sell signals with real data
    assert buy_count > 0 or sell_count > 0, \
        "MACD should generate at least one signal with real SPY data"


# ========== Trade Lifecycle E2E ==========


def test_trade_creation_and_closure():
    trade = Trade(Direction.LONG, "2024-01-02", Decimal("100.00"), 50)

    assert trade.status == TradeStatus.OPEN
    assert trade.direction == Direction.LONG
    assert trade.shares == 50
    assert trade.exit_date is None
    assert trade.exit_price is None

    # Close with profit
    trade.close("2024-01-10", Decimal("110.00"))

    assert trade.status == TradeStatus.CLOSED
    assert trade.exit_date == "2024-01-10"
    assert trade.exit_price == Decimal("110.00")

    # P/L = (110 - 100) * 50 = 500
    assert Decimal("500.00") == trade.profit_loss
    assert trade.is_win()

    # Return % = (110 - 100) / 100 * 100 = 10%
    assert Decimal("10.000000") == trade.return_pct


def test_trade_losing_position():
    trade = Trade(Direction.LONG, "2024-01-02", Decimal("100.00"), 100)

    trade.close("2024-01-10", Decimal("95.00"))

    # P/L = (95 - 100) * 100 = -500
    assert trade.profit_loss < Decimal("0")
    assert not trade.is_win()


def test_trade_cannot_be_closed_twice():
    trade = Trade(Direction.LONG, "2024-01-02", Decimal("100.00"), 10)
    trade.close("2024-01-10", Decimal("110.00"))

    with pytest.raises(RuntimeError):
        trade.close("2024-01-15", Decimal("120.00"))


def test_trade_short_direction():
    trade = Trade(Direction.SHORT, "2024-01-02", Decimal("100.00"), 50)

    trade.close("2024-01-10", Decimal("90.00"))

    # Short P/L = (entry - exit) * shares = (100 - 90) * 50 = 500
    assert Decimal("500.00") == trade.profit_loss
    assert trade.is_win()


def test_trade_to_string_format():
    trade = Trade(Direction.LONG, "2024-01-02", Decimal("100.00"), 10)
    trade.close("2024-01-10", Decimal("110.00"))

    s = str(trade)
    assert "LONG" in s
    assert "CLOSED" in s
    assert "2024-01-02" in s
    assert "2024-01-10" in s
    assert "100.00" in s
    assert "110.00" in s


# ========== Engine -> Trade -> Result Full Lifecycle ==========


def test_engine_produces_closed_trades_with_valid_lifecycle(real_data):
    engine = BacktestEngine(Decimal("100000"), Decimal("0"))
    strategy = SMACrossoverStrategy(20, 50)

    result = engine.run(strategy, real_data)

    for trade in result.closed_trades:
        # Every trade should go through complete lifecycle
        assert trade.status == TradeStatus.CLOSED
        assert trade.direction == Direction.LONG
        assert trade.entry_date is not None
        assert trade.exit_date is not None
        assert trade.entry_price > Decimal("0")
        assert trade.exit_price > Decimal("0")
        assert trade.shares > 0

        # Entry date should come before exit date
        assert trade.entry_date < trade.exit_date, \
            "Entry date should precede exit date"

        # Verify P/L calculation
        expected_pl = (trade.exit_price - trade.entry_price) * Decimal(str(trade.shares))
        assert expected_pl == trade.profit_loss, \
            "P/L should match (exit - entry) * shares"

        # Verify return percentage (code quantizes ratio first, then multiplies by 100)
        expected_return = (
            (trade.exit_price - trade.entry_price)
            / trade.entry_price
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) * Decimal("100")
        assert expected_return == trade.return_pct, \
            "Return % should match price change percentage"


def test_engine_trade_entry_dates_match_data_dates(real_data):
    engine = BacktestEngine(Decimal("100000"), Decimal("0"))
    strategy = MACDStrategy(12, 26, 9)

    result = engine.run(strategy, real_data)

    # Collect all valid dates from the data
    valid_dates = [bar.date for bar in real_data]

    for trade in result.closed_trades:
        assert trade.entry_date in valid_dates, \
            f"Entry date {trade.entry_date} should be a valid trading date"
        assert trade.exit_date in valid_dates, \
            f"Exit date {trade.exit_date} should be a valid trading date"


def test_engine_trade_entry_prices_match_close_prices(real_data):
    engine = BacktestEngine(Decimal("100000"), Decimal("0"))
    strategy = SMACrossoverStrategy(20, 50)

    result = engine.run(strategy, real_data)

    for trade in result.closed_trades:
        # Find the entry bar in real data
        for bar in real_data:
            if bar.date == trade.entry_date:
                assert bar.close == trade.entry_price, \
                    "Entry price should match the closing price of the entry date"
                break
        # Find the exit bar
        for bar in real_data:
            if bar.date == trade.exit_date:
                assert bar.close == trade.exit_price, \
                    "Exit price should match the closing price of the exit date"
                break


def test_engine_shares_calculated_from_capital(real_data):
    capital = Decimal("50000.00")
    engine = BacktestEngine(capital, Decimal("0"))
    strategy = SMACrossoverStrategy(20, 50)

    result = engine.run(strategy, real_data)

    if result.closed_trades:
        first_trade = result.closed_trades[0]
        # shares = floor(capital / price)
        expected_shares = int(capital / first_trade.entry_price)
        assert expected_shares == first_trade.shares, \
            "Shares should be floor(capital / entry price)"


# ========== Result Metrics from Trade Lifecycle ==========


def test_result_metrics_reflect_trade_outcomes(real_data):
    engine = BacktestEngine(Decimal("100000"), Decimal("0"))
    strategy = MACDStrategy(12, 26, 9)
    result = engine.run(strategy, real_data)

    if result.total_trades > 0:
        # Average win should be positive
        if result.winning_trades > 0:
            assert result.average_win > Decimal("0"), \
                "Average win should be positive"

        # Average loss should be negative or zero
        if result.losing_trades > 0:
            assert result.average_loss <= Decimal("0"), \
                "Average loss should be negative or zero"

        # Sharpe ratio should be computable
        sharpe = result.sharpe_ratio(252)
        assert sharpe is not None


def test_result_sum_of_trade_pl_matches_total_pl(real_data):
    engine = BacktestEngine(Decimal("100000"), Decimal("0"))
    strategy = RSIStrategy(14, 30.0, 70.0)
    result = engine.run(strategy, real_data)

    # The total P/L from result should be close to sum of individual trade P/L
    trade_pl_sum = sum(t.profit_loss for t in result.closed_trades)

    # Total P/L from result
    total_pl = result.total_pnl

    # The difference should be small (just the cash that couldn't buy full shares)
    diff = abs(total_pl - trade_pl_sum)
    # Difference should be less than one share price (max rounding error)
    if result.closed_trades:
        max_share_price = max(bar.close for bar in real_data)
        assert diff < max_share_price, \
            "P/L difference should be within one share price rounding error"


# ========== Edge Case: Very Small Capital ==========


def test_very_small_capital_still_works(real_data):
    # Capital too small to buy even one share
    tiny_cap = Decimal("1.00")
    engine = BacktestEngine(tiny_cap, Decimal("0"))
    strategy = SMACrossoverStrategy(20, 50)

    result = engine.run(strategy, real_data)

    # Should complete without errors
    assert tiny_cap == result.final_capital, \
        "Capital should remain unchanged when unable to buy shares"
    assert result.total_trades == 0, \
        "No trades should occur with insufficient capital"


# ========== PriceData Record Validation ==========


def test_price_data_record_equality():
    p1 = PriceData("2024-01-02",
                    Decimal("100"), Decimal("105"),
                    Decimal("99"), Decimal("103"),
                    Decimal("103"), 5000000)

    p2 = PriceData("2024-01-02",
                    Decimal("100"), Decimal("105"),
                    Decimal("99"), Decimal("103"),
                    Decimal("103"), 5000000)

    assert p1 == p2, "PriceData records with same values should be equal"
    assert hash(p1) == hash(p2)


def test_price_data_null_validation():
    with pytest.raises(ValueError):
        PriceData(None, Decimal("1"), Decimal("1"),
                  Decimal("1"), Decimal("1"), Decimal("1"), 0)
    with pytest.raises(ValueError):
        PriceData("2024-01-02", Decimal("1"), Decimal("1"),
                  Decimal("1"), Decimal("1"), Decimal("1"), -1)


def test_price_data_to_string_format(real_data):
    pd = real_data[0]
    s = str(pd)
    assert pd.date in s
    assert "O:" in s
    assert "H:" in s
    assert "L:" in s
    assert "C:" in s
    assert "V:" in s
