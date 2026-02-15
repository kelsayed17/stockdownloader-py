"""Standalone CLI for accumulating intraday 5-minute bar data over time.

Supports Yahoo Finance (free, ~60-day limit) and Polygon.io (free tier,
2 years of history).

Usage::

    intraday-accumulate                                      # Yahoo, SPY, 60 days
    intraday-accumulate --source polygon --days 730          # Polygon, SPY, 2 years
    intraday-accumulate AAPL --source polygon --days 365     # Polygon, AAPL, 1 year
    intraday-accumulate SPY --file my_spy_data.csv           # custom path
"""
from __future__ import annotations

import argparse
import logging
import os

from stockdownloader.data.intraday_data_accumulator import (
    IntradayDataAccumulator,
    default_csv_path,
)


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


def main() -> None:
    """Entry point for the intraday data accumulation CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Accumulate intraday 5-minute bar data. "
            "Merges fresh data with an existing CSV file, deduplicating "
            "overlapping bars.  Supports Yahoo Finance and Polygon.io."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  intraday-accumulate                                  # Yahoo, SPY, 60 days\n"
            "  intraday-accumulate --source polygon --days 730      # Polygon, SPY, 2 years\n"
            "  intraday-accumulate AAPL --source polygon --days 365 # Polygon, AAPL, 1 year\n"
            "  intraday-accumulate SPY --file my.csv                # custom output path\n"
        ),
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="SPY",
        help="Ticker symbol (default: SPY)",
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
        "--file",
        dest="csv_file",
        default=None,
        help="CSV file path (default: data/<symbol>_5m_bars.csv)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Calendar days to fetch (default: 60 for Yahoo, 730 for Polygon)",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    csv_path = args.csv_file or default_csv_path(symbol)

    # Default days based on source
    if args.days is not None:
        days = args.days
    elif args.source == "polygon":
        days = 730
    else:
        days = 60

    client = _build_client(args.source, args.api_key)
    source_label = "Polygon.io" if args.source == "polygon" else "Yahoo Finance"

    print("=" * 60)
    print("  Intraday Data Accumulator")
    print("=" * 60)
    print()
    print(f"  Symbol:      {symbol}")
    print(f"  Source:       {source_label}")
    print(f"  CSV file:    {csv_path}")
    print(f"  Fetch days:  {days}")
    print()

    accumulator = IntradayDataAccumulator(client=client, fetch_days=days)
    bars = accumulator.accumulate(symbol, csv_path)

    trading_days = len({bar.date[:10] for bar in bars})
    print()
    print("-" * 60)
    print(f"  Total bars:      {len(bars):,}")
    print(f"  Trading days:    {trading_days}")
    if bars:
        print(f"  Date range:      {bars[0].date[:10]} to {bars[-1].date[:10]}")
    print("-" * 60)
    print()
    print("  Done. Run again later to accumulate more data.")
    print()


if __name__ == "__main__":
    main()
