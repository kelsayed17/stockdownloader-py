"""Main entry point for backtesting multiple trading strategies on any symbol.

Supports dynamic data fetching from Yahoo Finance or loading from CSV.

Usage:
    python -m stockdownloader.app.spy_backtest_app                  # Fetches SPY data
    python -m stockdownloader.app.spy_backtest_app AAPL             # Fetches AAPL data
    python -m stockdownloader.app.spy_backtest_app --csv data.csv   # Loads from CSV file
"""
from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest.backtest_report_formatter import BacktestReportFormatter
from stockdownloader.data.csv_price_data_loader import CsvPriceDataLoader
from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.strategy.sma_crossover_strategy import SMACrossoverStrategy
from stockdownloader.strategy.rsi_strategy import RSIStrategy
from stockdownloader.strategy.macd_strategy import MACDStrategy
from stockdownloader.strategy.bollinger_band_rsi_strategy import BollingerBandRSIStrategy
from stockdownloader.strategy.momentum_confluence_strategy import MomentumConfluenceStrategy
from stockdownloader.strategy.breakout_strategy import BreakoutStrategy
from stockdownloader.strategy.multi_indicator_strategy import MultiIndicatorStrategy

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = Decimal("100000.00")
COMMISSION = Decimal("0")


def _fetch_data(symbol: str) -> list:
    """Fetch price data from Yahoo Finance.

    Args:
        symbol: Ticker symbol.

    Returns:
        List of PriceData bars, or empty list on failure.
    """
    print(f"Fetching {symbol} data from Yahoo Finance...")
    try:
        client = YahooDataClient()
        data = client.fetch_price_data(symbol, "5y", "1d")
        if data:
            print(f"Fetched {len(data)} days of {symbol} data")
            return data
    except Exception as exc:
        logger.warning("Could not fetch %s data: %s", symbol, exc)
    return []


def main() -> None:
    """Entry point for the equity backtesting application."""
    parser = argparse.ArgumentParser(
        description="Trading Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  spy_backtest_app                  # Fetch SPY from Yahoo Finance\n"
            "  spy_backtest_app AAPL             # Fetch AAPL from Yahoo Finance\n"
            "  spy_backtest_app --csv data.csv   # Load from CSV file\n"
        ),
    )
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--csv", dest="csv_file", default=None, help="Load data from CSV file")
    args = parser.parse_args()

    print("========================================")
    print("  Trading Strategy Backtester")
    print("========================================")
    print()

    symbol = args.symbol.upper()

    if args.csv_file:
        print(f"Loading data from file: {args.csv_file}")
        data = CsvPriceDataLoader.load_from_file(args.csv_file)
    else:
        data = _fetch_data(symbol)

    if not data:
        print("ERROR: No price data loaded.")
        print("Usage:")
        print("  spy_backtest_app                  # Fetch SPY from Yahoo Finance")
        print("  spy_backtest_app AAPL             # Fetch AAPL from Yahoo Finance")
        print("  spy_backtest_app --csv data.csv   # Load from CSV file")
        return

    print(f"Loaded {len(data)} trading days for {symbol}")
    print(f"Date range: {data[0].date} to {data[-1].date}")
    print(f"Starting capital: ${INITIAL_CAPITAL.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
    print()

    strategies = [
        SMACrossoverStrategy(50, 200),
        SMACrossoverStrategy(20, 50),
        RSIStrategy(14, 30, 70),
        RSIStrategy(14, 25, 75),
        MACDStrategy(12, 26, 9),
        BollingerBandRSIStrategy(),
        MomentumConfluenceStrategy(),
        BreakoutStrategy(),
        MultiIndicatorStrategy(),
    ]

    engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION)
    results = []

    for strategy in strategies:
        print(f"Running backtest: {strategy.get_name()}...")
        result = engine.run(strategy, data)
        results.append(result)
        BacktestReportFormatter.print_report(result, data)

    BacktestReportFormatter.print_comparison(results, data)


if __name__ == "__main__":
    main()
