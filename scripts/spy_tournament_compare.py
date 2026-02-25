#!/usr/bin/env python3
"""SPY Tournament Strategy Comparison Runner.

Fetches 2-year 5-minute SPY data from Polygon, runs each tournament
strategy through IntradayBacktestEngine, and prints a comparison table.

Usage::

    # Tournament strategies only (default)
    POLYGON_API_KEY=... python scripts/spy_tournament_compare.py

    # Include all registered VWAP strategies
    POLYGON_API_KEY=... python scripts/spy_tournament_compare.py --all

    # Use cached CSV data
    python scripts/spy_tournament_compare.py --csv data/SPY/5m_bars.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

# Ensure src/ is on the path when running as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from stockdownloader.backtesting.engines.intraday import IntradayBacktestEngine
from stockdownloader.backtesting.results.result import BacktestResult
from stockdownloader.data.intraday_csv import IntradayCsvLoader, write_to_file
from stockdownloader.strategies.intraday.macd_obv import MACDOBVStrategy
from stockdownloader.strategies.intraday.sma_cross import SMACross2021Strategy
from stockdownloader.strategies.intraday.macd_optimized import MACDOptimizedStrategy
from stockdownloader.strategies.loader import ensure_registered
from stockdownloader.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

# Default cache path
_CACHE_DIR = _ROOT / "data" / "SPY"
_CACHE_FILE = _CACHE_DIR / "5m_bars.csv"

# Tournament strategy keys (from intraday_registrations.json)
_TOURNAMENT_KEYS = ["spy-macd-obv", "spy-sma-2021", "spy-macd-opt"]

# VWAP strategy keys to include with --all
_VWAP_KEYS = [
    "vwap-pullback",
    "vwap-reversal",
    "vwap-orb",
    "vwap-orr",
    "vwap-ps",
    "avwap-pullback",
    "ml-oversold",
    "smc-structure",
]


def _fetch_data(csv_path: str | None = None) -> list:
    """Fetch or load 2-year 5-minute SPY data.

    If *csv_path* is provided and exists, loads from that file.
    Otherwise fetches from Polygon and caches to the default location.

    Returns
    -------
    list[IntradayPriceData]
    """
    # Try loading from explicit CSV path first
    if csv_path:
        p = Path(csv_path)
        if p.exists():
            print(f"Loading cached data from {p} ...")
            data = IntradayCsvLoader.load_from_file(str(p))
            print(f"  Loaded {len(data)} bars")
            return data
        else:
            print(f"CSV file not found: {p}")
            sys.exit(1)

    # Try loading from default cache
    if _CACHE_FILE.exists():
        print(f"Loading cached data from {_CACHE_FILE} ...")
        data = IntradayCsvLoader.load_from_file(str(_CACHE_FILE))
        if data:
            print(f"  Loaded {len(data)} bars from cache")
            return data

    # Fetch from Polygon
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY env var required for data fetch.")
        print("Set it or provide --csv with a cached data file.")
        sys.exit(1)

    from stockdownloader.data.market.polygon_client import PolygonDataClient

    print("Fetching 2-year 5-minute SPY data from Polygon ...")
    print("(This will take ~5 minutes due to rate limiting)")
    client = PolygonDataClient(api_key=api_key)
    start = time.time()
    data = client.fetch_intraday_history("SPY", total_days=730)
    elapsed = time.time() - start
    print(f"  Fetched {len(data)} bars in {elapsed:.0f}s")

    # Cache to CSV
    if data:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        write_to_file(data, _CACHE_FILE)
        print(f"  Cached to {_CACHE_FILE}")

    return data


def _run_strategy(
    name: str,
    strategy,
    engine: IntradayBacktestEngine,
    data: list,
) -> BacktestResult:
    """Run a single strategy and return the result."""
    print(f"  Running {name} ...", end="", flush=True)
    start = time.time()
    result = engine.run(strategy, data)
    elapsed = time.time() - start
    print(f" done ({elapsed:.1f}s, {result.total_trades} trades)")
    return result


def _print_table(results: dict[str, BacktestResult]) -> None:
    """Print a formatted comparison table."""
    if not results:
        print("No results to display.")
        return

    # Header
    print()
    print("=" * 110)
    print(f"{'Strategy':<28} {'Return%':>9} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7} {'AvgPnL':>10} {'PF':>6}")
    print("-" * 110)

    for name, r in results.items():
        avg_pnl = Decimal("0")
        if r.total_trades > 0:
            avg_pnl = r.total_pnl / Decimal(str(r.total_trades))

        print(
            f"{name:<28} "
            f"{float(r.total_return):>9.2f} "
            f"{float(r.sharpe_ratio()):>8.2f} "
            f"{float(r.sortino_ratio()):>8.2f} "
            f"{float(r.max_drawdown):>8.2f} "
            f"{float(r.win_rate):>9.2f} "
            f"{r.total_trades:>7d} "
            f"${float(avg_pnl):>9.2f} "
            f"{float(r.profit_factor):>6.2f}"
        )

    print("=" * 110)


def run_comparison(
    csv_path: str | None = None,
    include_all: bool = False,
) -> dict[str, BacktestResult]:
    """Run the strategy comparison and return results.

    Parameters
    ----------
    csv_path:
        Optional path to cached 5-minute bar CSV.
    include_all:
        If True, include existing VWAP strategies alongside tournament ones.

    Returns
    -------
    dict[str, BacktestResult]
        Mapping of strategy display name to backtest result.
    """
    # Make sure all strategies are registered
    ensure_registered()

    # Build strategy list
    strategy_keys = list(_TOURNAMENT_KEYS)
    if include_all:
        strategy_keys = _VWAP_KEYS + strategy_keys

    # Load data
    data = _fetch_data(csv_path)
    if not data:
        print("ERROR: No data loaded.")
        return {}

    # Create engine
    engine = IntradayBacktestEngine(
        initial_capital=Decimal("100000"),
        risk_per_trade=Decimal("0.01"),
    )

    # Run strategies
    results: dict[str, BacktestResult] = {}
    print(f"\nRunning {len(strategy_keys)} strategies ...")

    for key in strategy_keys:
        try:
            entry = StrategyRegistry.get(key)
        except ValueError:
            print(f"  WARNING: Strategy '{key}' not registered, skipping.")
            continue

        strategy = entry.factory(**entry.default_kwargs)
        display = entry.display_name
        result = _run_strategy(display, strategy, engine, data)
        results[display] = result

    # Print results
    _print_table(results)

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SPY Tournament Strategy Comparison Runner"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to cached 5-minute bar CSV file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Include all registered VWAP strategies (not just tournament)",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    run_comparison(csv_path=args.csv, include_all=args.all)


if __name__ == "__main__":
    main()
