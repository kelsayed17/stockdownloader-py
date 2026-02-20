"""Shared helpers for CLI application entry points.

Extracts common patterns used across multiple app runners:

- ``fetch_daily_data`` — Yahoo Finance daily price fetch
- ``load_or_fetch_daily`` — CSV-or-fetch branch + guard + summary
- ``load_intraday_data`` — load CSV + guard + summary (replaces boilerplate)
- ``setup_output_dir`` — create output dir + return log path
- ``add_symbol_arg`` / ``add_csv_arg`` / ``add_strategy_args`` — argparse
- ``print_banner`` — consistent header formatting
- ``list_strategies`` — ``--list-strategies`` handler

Constants (``INITIAL_CAPITAL``, ``RISK_PER_TRADE``, etc.) live in
:mod:`stockdownloader.util.constants` — re-exported here for convenience.
"""

from __future__ import annotations

import argparse
import logging
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Callable

from stockdownloader.data.csv_price_data_loader import CsvPriceDataLoader
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.data.yahoo_data_client import YahooDataClient
from stockdownloader.model.intraday_price_data import IntradayPriceData
from stockdownloader.model.price_data import PriceData
from stockdownloader.util.constants import (
    DEFAULT_DATA_FILE,
    DEFAULT_OUTPUT_DIR,
    INITIAL_CAPITAL,
    OPTIONS_COMMISSION,
    RISK_PER_TRADE,
)
from stockdownloader.util.timeframe_aggregator import Timeframe

logger = logging.getLogger(__name__)

# ======================================================================
# Re-exports & supplementary constants
# ======================================================================

#: Standard 6-element timeframe list used by multi-TF tournaments.
STANDARD_TIMEFRAMES: list[tuple[Timeframe, str]] = [
    (Timeframe.M5, "5m"),
    (Timeframe.M15, "15m"),
    (Timeframe.M30, "30m"),
    (Timeframe.H1, "1h"),
    (Timeframe.H4, "4h"),
    (Timeframe.DAILY, "1d"),
]

# ======================================================================
# Data loading — intraday
# ======================================================================


def load_intraday_data(
    csv_file: str | Path = DEFAULT_DATA_FILE,
    print_fn: Callable[..., object] = print,
) -> list[IntradayPriceData]:
    """Load intraday 5-minute bars from *csv_file*, print summary, guard empty.

    This consolidates the CSV-load → empty-guard → summary pattern that
    was duplicated across 8+ app entry points.

    Parameters
    ----------
    csv_file:
        Path to the intraday CSV file.
    print_fn:
        Callable used for output (defaults to :func:`print`).  Pass a
        :class:`~stockdownloader.util.file_helper.TeeWriter`
        ``write`` method to log simultaneously.

    Returns
    -------
    list[IntradayPriceData]
        Loaded bars.  Returns an empty list if the file cannot be read.
    """
    csv_file = Path(csv_file)
    print_fn(f"Loading 5m data from {csv_file}...")
    data = IntradayCsvLoader.load_from_file(csv_file)
    if not data:
        print_fn(f"ERROR: Could not load data from {csv_file}")
        return []

    print_fn(f"Loaded {len(data):,} 5-minute bars")
    print_fn(f"Date range: {data[0].date} to {data[-1].date}")
    trading_days = len({d.date[:10] for d in data})
    print_fn(f"Trading days: {trading_days}")
    return data


def setup_output_dir(log_name: str) -> Path:
    """Ensure :data:`DEFAULT_OUTPUT_DIR` exists and return the log path.

    Parameters
    ----------
    log_name:
        Filename for the log file (e.g. ``"tournament_results.log"``).

    Returns
    -------
    Path
        Full path to ``DEFAULT_OUTPUT_DIR / log_name``.
    """
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / log_name


# ======================================================================
# Data loading — daily
# ======================================================================


def fetch_daily_data(
    symbol: str,
    period: str = "5y",
    interval: str = "1d",
) -> list[PriceData]:
    """Fetch daily price data from Yahoo Finance.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    period:
        Look-back period (default ``"5y"``).
    interval:
        Bar interval (default ``"1d"``).

    Returns
    -------
    list[PriceData]
        Price bars, or an empty list on failure.
    """
    print(f"Fetching {symbol} data from Yahoo Finance...")
    try:
        client = YahooDataClient()
        data = client.fetch_price_data(symbol, period, interval)
        if data:
            print(f"Fetched {len(data)} days of {symbol} data")
            return data
    except Exception as exc:
        logger.warning("Could not fetch %s data: %s", symbol, exc)
    return []


def load_or_fetch_daily(
    symbol: str,
    csv_file: str | None,
    *,
    period: str = "5y",
) -> list[PriceData]:
    """Load daily data from CSV or fetch from Yahoo Finance.

    Parameters
    ----------
    symbol:
        Ticker symbol (used for fetch and summary printing).
    csv_file:
        Path to CSV file, or ``None`` to fetch from Yahoo.
    period:
        Yahoo look-back period (default ``"5y"``).

    Returns
    -------
    list[PriceData]
        Price bars (may be empty on failure).
    """
    if csv_file:
        print(f"Loading data from file: {csv_file}")
        return CsvPriceDataLoader.load_from_file(csv_file)
    return fetch_daily_data(symbol, period)


def print_data_summary(
    data: list,
    symbol: str,
    capital: Decimal = INITIAL_CAPITAL,
    *,
    extra_lines: list[str] | None = None,
) -> None:
    """Print a standard data-loaded summary block.

    Parameters
    ----------
    data:
        Loaded price bars (daily or intraday).
    symbol:
        Ticker symbol.
    capital:
        Starting capital to display.
    extra_lines:
        Additional summary lines (e.g. risk-per-trade, commission).
    """
    print(f"Loaded {len(data)} trading days for {symbol}")
    print(f"Date range: {data[0].date} to {data[-1].date}")
    cap_str = capital.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    print(f"Starting capital: ${cap_str}")
    if extra_lines:
        for line in extra_lines:
            print(line)
    print()


# ======================================================================
# Argparse helpers
# ======================================================================


def add_symbol_arg(parser: argparse.ArgumentParser, default: str = "SPY") -> None:
    """Add a positional ``symbol`` argument."""
    parser.add_argument(
        "symbol", nargs="?", default=default,
        help=f"Ticker symbol (default: {default})",
    )


def add_csv_arg(parser: argparse.ArgumentParser) -> None:
    """Add a ``--csv`` argument for loading data from file."""
    parser.add_argument(
        "--csv", dest="csv_file", default=None,
        help="Load data from CSV file",
    )


def add_strategy_args(
    parser: argparse.ArgumentParser,
    *,
    list_help: str = "List all available strategies and exit.",
) -> None:
    """Add ``--strategy`` and ``--list-strategies`` arguments."""
    parser.add_argument(
        "--strategy", default=None,
        help="Run a single strategy by name (case-insensitive).",
    )
    parser.add_argument(
        "--list-strategies", action="store_true", dest="list_strategies",
        help=list_help,
    )


# ======================================================================
# Display helpers
# ======================================================================


def print_banner(title: str, width: int = 40) -> None:
    """Print a centered banner header."""
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def list_strategies_and_exit(category: str) -> None:
    """Print all registered strategies for *category* and return.

    Call this when ``args.list_strategies`` is ``True``.
    """
    from stockdownloader.strategy.registrations import ensure_registered
    from stockdownloader.strategy.registry import StrategyRegistry

    ensure_registered()
    print(f"Available {category} strategies:")
    for entry in StrategyRegistry.all_entries(category=category):
        params = ", ".join(
            f"{k}={v}" for k, v in entry.default_kwargs.items()
        ) if entry.default_kwargs else "(defaults)"
        print(f"  {entry.name:<15s}  {entry.display_name:<30s}  {params}")
