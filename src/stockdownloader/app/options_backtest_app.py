"""Main entry point for backtesting options strategies on historical price data.

Supports dynamic data fetching from Yahoo Finance or loading from CSV.

Usage:
    python -m stockdownloader.app.options_backtest_app                  # Fetches SPY data
    python -m stockdownloader.app.options_backtest_app AAPL             # Fetches AAPL data
    python -m stockdownloader.app.options_backtest_app --csv data.csv   # Loads from CSV file
"""
from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.backtest.options_backtest_engine import OptionsBacktestEngine
from stockdownloader.backtest.options_backtest_report_formatter import OptionsBacktestReportFormatter
from stockdownloader.data.csv_price_data_loader import CsvPriceDataLoader
from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.strategy.covered_call_strategy import CoveredCallStrategy
from stockdownloader.strategy.protective_put_strategy import ProtectivePutStrategy

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = Decimal("100000.00")
COMMISSION = Decimal("0.65")


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
    """Entry point for the options backtesting application."""
    parser = argparse.ArgumentParser(
        description="Options Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  options_backtest_app                  # Fetch SPY from Yahoo Finance\n"
            "  options_backtest_app AAPL             # Fetch AAPL from Yahoo Finance\n"
            "  options_backtest_app --csv data.csv   # Load from CSV file\n"
        ),
    )
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--csv", dest="csv_file", default=None, help="Load data from CSV file")
    args = parser.parse_args()

    print("================================================")
    print("  Options Strategy Backtester")
    print("================================================")
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
        print("  options_backtest_app                  # Fetch SPY from Yahoo Finance")
        print("  options_backtest_app AAPL             # Fetch AAPL from Yahoo Finance")
        print("  options_backtest_app --csv data.csv   # Load from CSV file")
        return

    print(f"Loaded {len(data)} trading days for {symbol}")
    print(f"Date range: {data[0].date} to {data[-1].date}")
    print(f"Starting capital: ${INITIAL_CAPITAL.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
    print(f"Commission: ${COMMISSION} per contract")
    print()

    strategies = [
        # Covered calls: varying OTM% and DTE
        CoveredCallStrategy(20, Decimal("0.03"), 30, Decimal("0.03")),
        CoveredCallStrategy(20, Decimal("0.05"), 30, Decimal("0.03")),
        CoveredCallStrategy(50, Decimal("0.05"), 45, Decimal("0.04")),
        # Protective puts: varying OTM% and DTE
        ProtectivePutStrategy(20, Decimal("0.05"), 30, 5),
        ProtectivePutStrategy(20, Decimal("0.03"), 45, 10),
        ProtectivePutStrategy(50, Decimal("0.05"), 60, 10),
    ]

    engine = OptionsBacktestEngine(INITIAL_CAPITAL, COMMISSION)
    results = []

    for strategy in strategies:
        print(f"Running options backtest: {strategy.get_name()}...")
        result = engine.run(strategy, data)
        results.append(result)
        OptionsBacktestReportFormatter.print_report(result)

    OptionsBacktestReportFormatter.print_comparison(results)


if __name__ == "__main__":
    main()
