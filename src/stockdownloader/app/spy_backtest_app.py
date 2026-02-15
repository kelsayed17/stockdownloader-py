"""Main entry point for backtesting multiple trading strategies on any symbol.

Supports dynamic data fetching from Yahoo Finance or loading from CSV.

Usage:
    spy-backtest                          # Fetch SPY, run all strategies
    spy-backtest AAPL                     # Fetch AAPL, run all strategies
    spy-backtest --csv data.csv           # Load from CSV file
    spy-backtest --strategy rsi           # Run only RSI with defaults
    spy-backtest --list-strategies        # Show available strategies
"""
from __future__ import annotations

import argparse
import logging
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.backtest.backtest_engine import BacktestEngine
from stockdownloader.backtest import report_formatter
from stockdownloader.data.csv_price_data_loader import CsvPriceDataLoader
from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.strategy.daily.sma_crossover_strategy import SMACrossoverStrategy
from stockdownloader.strategy.daily.rsi_strategy import RSIStrategy
from stockdownloader.strategy.daily.macd_strategy import MACDStrategy
from stockdownloader.strategy.daily.bollinger_band_rsi_strategy import BollingerBandRSIStrategy
from stockdownloader.strategy.daily.momentum_confluence_strategy import MomentumConfluenceStrategy
from stockdownloader.strategy.daily.breakout_strategy import BreakoutStrategy
from stockdownloader.strategy.daily.multi_indicator_strategy import MultiIndicatorStrategy

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = Decimal("100000.00")
COMMISSION = Decimal("0")


def _fetch_data(symbol: str) -> list:
    """Fetch price data from Yahoo Finance."""
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
            "  spy-backtest                          # Fetch SPY, run all\n"
            "  spy-backtest AAPL                     # Fetch AAPL data\n"
            "  spy-backtest --csv data.csv           # Load from CSV file\n"
            "  spy-backtest --strategy rsi           # Run only RSI\n"
            "  spy-backtest --list-strategies        # Show available\n"
        ),
    )
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--csv", dest="csv_file", default=None, help="Load data from CSV file")
    parser.add_argument(
        "--strategy",
        default=None,
        help="Run a single strategy by name (e.g., rsi, macd, sma). Case-insensitive.",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        dest="list_strategies",
        help="List all available daily strategies and exit.",
    )
    args = parser.parse_args()

    if args.list_strategies:
        from stockdownloader.strategy.registrations import ensure_registered
        from stockdownloader.strategy.registry import StrategyRegistry
        ensure_registered()
        print("Available daily strategies:")
        for entry in StrategyRegistry.all_entries(category="daily"):
            params = ", ".join(
                f"{k}={v}" for k, v in entry.default_kwargs.items()
            ) if entry.default_kwargs else "(defaults)"
            print(f"  {entry.name:<15s}  {entry.display_name:<30s}  {params}")
        return

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
        return

    print(f"Loaded {len(data)} trading days for {symbol}")
    print(f"Date range: {data[0].date} to {data[-1].date}")
    print(f"Starting capital: ${INITIAL_CAPITAL.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
    print()

    if args.strategy:
        from stockdownloader.strategy.registrations import ensure_registered
        from stockdownloader.strategy.registry import StrategyRegistry
        ensure_registered()
        strategies = [StrategyRegistry.create(args.strategy)]
    else:
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
        print(f"Running backtest: {strategy.name}...")
        result = engine.run(strategy, data)
        results.append(result)
        report_formatter.print_daily_report(result, data)

    if len(results) > 1:
        report_formatter.print_daily_comparison(results, data)


if __name__ == "__main__":
    main()
