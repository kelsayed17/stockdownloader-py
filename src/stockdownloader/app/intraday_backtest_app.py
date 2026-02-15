"""Entry point for backtesting intraday strategies on 5-minute bar data.

Supports fetching live data from Yahoo Finance or Polygon.io, loading
from a CSV file, or accumulating data over time.

Usage:
    python -m stockdownloader.app.intraday_backtest_app                            # Fetch SPY 5m (~60 days)
    python -m stockdownloader.app.intraday_backtest_app --csv data/spy_5m_bars.csv  # Load from CSV
    python -m stockdownloader.app.intraday_backtest_app --accumulate               # Fetch, merge with CSV, backtest
    python -m stockdownloader.app.intraday_backtest_app --source polygon --days 730 # Polygon, 2 years
"""
from __future__ import annotations

import argparse
import logging
import os
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest import report_formatter
from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader
from stockdownloader.data.intraday_data_accumulator import (
    IntradayDataAccumulator,
    default_csv_path,
)
from stockdownloader.strategy.intraday.or_breakout_strategy import ORBreakoutStrategy
from stockdownloader.strategy.intraday.or_reversal_strategy import ORReversalStrategy
from stockdownloader.strategy.intraday.pattern_scalp_strategy import PatternScalpStrategy
from stockdownloader.strategy.intraday.pullback_strategy import PullbackStrategy
from stockdownloader.strategy.intraday.reversal_strategy import ReversalStrategy

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = Decimal("100000.00")
RISK_PER_TRADE = Decimal("0.01")  # 1% risk per trade


def _build_client(source: str, api_key: str | None = None):
    """Create a data client for the given source."""
    if source == "polygon":
        from stockdownloader.data.polygon_data_client import PolygonDataClient
        key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not key:
            print("ERROR: Polygon API key required.")
            print("  Set POLYGON_API_KEY env var or pass --api-key")
            print("  Get a free key at https://polygon.io/")
            raise SystemExit(1)
        return PolygonDataClient(api_key=key)
    else:
        from stockdownloader.data.yahoo_data_client import YahooDataClient
        return YahooDataClient()


def _fetch_intraday(symbol: str, days: int, source: str = "yahoo", api_key: str | None = None) -> list:
    """Fetch intraday 5-minute data from the specified source."""
    source_label = "Polygon.io" if source == "polygon" else "Yahoo Finance"
    print(f"Fetching {symbol} 5-minute data ({days} days back) from {source_label}...")
    try:
        client = _build_client(source, api_key)
        data = client.fetch_intraday_history(symbol, total_days=days)
        if data:
            trading_days = len({bar.date[:10] for bar in data})
            print(f"Fetched {len(data):,} bars across {trading_days} trading days")
            return data
    except Exception as exc:
        logger.warning("Could not fetch %s intraday data: %s", symbol, exc)
    return []


def main() -> None:
    """Entry point for the intraday backtesting application."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Intraday Strategy Backtester (5-minute bars)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  intraday_backtest_app                            # Fetch SPY 5m (~60 days)\n"
            "  intraday_backtest_app --csv data/spy_5m_bars.csv # Load from CSV\n"
            "  intraday_backtest_app --accumulate               # Fetch, merge, backtest\n"
            "  intraday_backtest_app --source polygon --days 730 # Polygon, 2 years\n"
        ),
    )
    parser.add_argument(
        "symbol", nargs="?", default="SPY",
        help="Ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Calendar days to fetch (default: 60 for Yahoo, 730 for Polygon)",
    )
    parser.add_argument(
        "--csv", dest="csv_file", default=None,
        help="Load data from an intraday CSV file",
    )
    parser.add_argument(
        "--accumulate", action="store_true",
        help="Fetch latest data, merge with CSV on disk, then backtest the full history",
    )
    parser.add_argument(
        "--source",
        choices=["yahoo", "polygon"],
        default="yahoo",
        help="Data source (default: yahoo)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key for Polygon.io (or set POLYGON_API_KEY env var)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Run a single strategy by name (e.g., rsi, vwap). "
             "Daily strategies are auto-adapted for intraday.",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        dest="all_strategies",
        help="Run all 8 strategies (VWAP + 7 adapted daily) and compare",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        dest="list_strategies",
        help="List all available strategies and exit.",
    )
    args = parser.parse_args()

    if args.list_strategies:
        from stockdownloader.strategy.registrations import ensure_registered
        from stockdownloader.strategy.registry import StrategyRegistry
        ensure_registered()
        print("Available strategies for intraday backtest:")
        for entry in StrategyRegistry.all_entries(category="intraday"):
            print(f"  {entry.name:<15s}  {entry.display_name:<30s}  (native)")
        for entry in StrategyRegistry.all_entries(category="daily"):
            print(f"  {entry.name:<15s}  {entry.display_name:<30s}  (adapted)")
        return

    print("========================================")
    print("  Intraday Strategy Backtester")
    print("========================================")
    print()

    symbol = args.symbol.upper()

    # Default days based on source
    if args.days is not None:
        days = args.days
    elif args.source == "polygon":
        days = 730
    else:
        days = 60

    if args.accumulate:
        csv_path = args.csv_file or default_csv_path(symbol)
        print(f"Accumulating {symbol} data into {csv_path}...")
        client = _build_client(args.source, args.api_key)
        accumulator = IntradayDataAccumulator(client=client, fetch_days=days)
        data = accumulator.accumulate(symbol, csv_path)
    elif args.csv_file:
        print(f"Loading intraday data from file: {args.csv_file}")
        data = IntradayCsvLoader.load_from_file(args.csv_file)
    else:
        data = _fetch_intraday(symbol, days, args.source, args.api_key)

    if not data:
        print("ERROR: No intraday data loaded.")
        return

    trading_days = len({bar.date[:10] for bar in data})
    print(f"Loaded {len(data):,} bars across {trading_days} trading days for {symbol}")
    print(f"Date range: {data[0].date} to {data[-1].date}")
    print(f"Starting capital: ${INITIAL_CAPITAL.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
    print(f"Risk per trade: {RISK_PER_TRADE * 100}%")
    print()

    if args.strategy:
        from stockdownloader.strategy.registrations import ensure_registered
        from stockdownloader.strategy.registry import StrategyRegistry
        from stockdownloader.strategy.daily_to_intraday_adapter import DailyToIntradayAdapter

        ensure_registered()
        entry = StrategyRegistry.get(args.strategy)
        raw = StrategyRegistry.create(args.strategy)
        if entry.category == "daily":
            strategies = [DailyToIntradayAdapter(raw)]
        elif entry.category == "intraday":
            strategies = [raw]
        else:
            print(f"ERROR: Strategy '{args.strategy}' is category '{entry.category}', "
                  "not usable for intraday backtest.")
            return
    elif args.all_strategies:
        from stockdownloader.strategy.registrations import ensure_registered
        from stockdownloader.strategy.registry import StrategyRegistry
        from stockdownloader.strategy.daily_to_intraday_adapter import DailyToIntradayAdapter

        ensure_registered()
        strategies = [
            PullbackStrategy(),
            ReversalStrategy(),
            ORBreakoutStrategy(),
            ORReversalStrategy(),
            PatternScalpStrategy(),
        ]
        for entry in StrategyRegistry.all_entries(category="daily"):
            strategies.append(DailyToIntradayAdapter(StrategyRegistry.create(entry.name)))
    else:
        strategies = [
            PullbackStrategy(),
            ReversalStrategy(),
            ORBreakoutStrategy(),
            ORReversalStrategy(),
            PatternScalpStrategy(),
        ]

    engine = IntradayBacktestEngine(INITIAL_CAPITAL, RISK_PER_TRADE)
    results = []

    for strategy in strategies:
        print(f"Running intraday backtest: {strategy.name}...")
        result = engine.run(strategy, data)
        results.append(result)
        report_formatter.print_intraday_report(result, data)

    if len(results) > 1:
        report_formatter.print_intraday_comparison(results, data)


if __name__ == "__main__":
    main()
