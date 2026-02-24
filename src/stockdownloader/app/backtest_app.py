"""Unified backtesting entry point for daily, intraday, and options strategies.

Consolidates the former ``spy_backtest_app``, ``gme_backtest_runner``,
``options_backtest_app``, ``intraday_backtest_app``, and
``dmi_vwap_backtest`` into a single parameterised CLI.

Usage::

    spy-backtest                       # Daily equity strategies on SPY
    spy-backtest AAPL --strategy rsi   # Single strategy on AAPL
    intraday-backtest --csv bars.csv   # Intraday VWAP strategies from CSV
    options-backtest AAPL              # Options strategies on AAPL
    dmi-vwap-backtest                  # DMI+VWAP intraday strategy
"""

from __future__ import annotations

import argparse
import logging
import os
from decimal import Decimal, ROUND_HALF_UP

from stockdownloader.app.app_helpers import (
    INITIAL_CAPITAL,
    OPTIONS_COMMISSION,
    RISK_PER_TRADE,
    add_csv_arg,
    add_strategy_args,
    add_symbol_arg,
    list_strategies_and_exit,
    load_or_fetch_daily,
    print_banner,
    print_data_summary,
)
from stockdownloader.backtest import report_formatter

logger = logging.getLogger(__name__)


# ======================================================================
# Data helpers (intraday)
# ======================================================================


def _build_client(source: str, api_key: str | None = None):
    """Create a data client for the given source."""
    if source == "polygon":
        from stockdownloader.data.market.polygon_client import PolygonDataClient

        key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not key:
            print("ERROR: Polygon API key required.")
            print("  Set POLYGON_API_KEY env var or pass --api-key")
            print("  Get a free key at https://polygon.io/")
            raise SystemExit(1)
        return PolygonDataClient(api_key=key)
    from stockdownloader.data.market.yahoo_data_client import YahooDataClient

    return YahooDataClient()


def _fetch_intraday(
    symbol: str,
    days: int,
    source: str = "yahoo",
    api_key: str | None = None,
) -> list:
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


def _load_intraday_data(args, symbol: str) -> list:
    """Load intraday data from CSV, fetch, or accumulate."""
    from stockdownloader.data.intraday_csv import IntradayCsvLoader

    days = args.days
    if days is None:
        days = 730 if getattr(args, "source", "yahoo") == "polygon" else 60

    if getattr(args, "accumulate", False):
        from stockdownloader.data.accumulator import (
            IntradayDataAccumulator,
            default_csv_path,
        )

        csv_path = args.csv_file or default_csv_path(symbol)
        print(f"Accumulating {symbol} data into {csv_path}...")
        client = _build_client(
            getattr(args, "source", "yahoo"),
            getattr(args, "api_key", None),
        )
        accumulator = IntradayDataAccumulator(client=client, fetch_days=days)
        return accumulator.accumulate(symbol, csv_path)

    if args.csv_file:
        print(f"Loading intraday data from file: {args.csv_file}")
        return IntradayCsvLoader.load_from_file(args.csv_file)

    return _fetch_intraday(
        symbol,
        days,
        getattr(args, "source", "yahoo"),
        getattr(args, "api_key", None),
    )


# ======================================================================
# Strategy builders
# ======================================================================


def _build_daily_strategies(args):
    """Build the list of daily equity strategies to backtest."""
    if getattr(args, "strategy", None):
        from stockdownloader.strategies.loader import ensure_registered
        from stockdownloader.strategies.registry import StrategyRegistry

        ensure_registered()
        return [StrategyRegistry.create(args.strategy)]

    from stockdownloader.strategies.daily.bollinger_rsi import (
        BollingerBandRSIStrategy,
    )
    from stockdownloader.strategies.daily.breakout import BreakoutStrategy
    from stockdownloader.strategies.daily.simple import MACDStrategy
    from stockdownloader.strategies.daily.momentum import (
        MomentumConfluenceStrategy,
    )
    from stockdownloader.strategies.daily.multi_indicator import (
        MultiIndicatorStrategy,
    )
    from stockdownloader.strategies.daily.simple import RSIStrategy
    from stockdownloader.strategies.daily.simple import SMACrossoverStrategy

    return [
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


def _build_intraday_strategies(args):
    """Build the list of intraday strategies to backtest."""
    from stockdownloader.strategies.intraday.or_breakout import (
        ORBreakoutStrategy,
    )
    from stockdownloader.strategies.intraday.or_reversal import (
        ORReversalStrategy,
    )
    from stockdownloader.strategies.intraday.pattern_scalp import (
        PatternScalpStrategy,
    )
    from stockdownloader.strategies.intraday.pullback import PullbackStrategy
    from stockdownloader.strategies.intraday.reversal import ReversalStrategy

    if getattr(args, "strategy", None):
        from stockdownloader.strategies.intraday.daily_adapter import (
            DailyToIntradayAdapter,
        )
        from stockdownloader.strategies.loader import ensure_registered
        from stockdownloader.strategies.registry import StrategyRegistry

        ensure_registered()
        entry = StrategyRegistry.get(args.strategy)
        raw = StrategyRegistry.create(args.strategy)
        if entry.category == "daily":
            return [DailyToIntradayAdapter(raw)]
        if entry.category == "intraday":
            return [raw]
        print(
            f"ERROR: Strategy '{args.strategy}' is category '{entry.category}', "
            "not usable for intraday backtest."
        )
        raise SystemExit(1)

    if getattr(args, "all_strategies", False):
        from stockdownloader.strategies.intraday.daily_adapter import (
            DailyToIntradayAdapter,
        )
        from stockdownloader.strategies.loader import ensure_registered
        from stockdownloader.strategies.registry import StrategyRegistry

        ensure_registered()
        strategies = [
            PullbackStrategy(),
            ReversalStrategy(),
            ORBreakoutStrategy(),
            ORReversalStrategy(),
            PatternScalpStrategy(),
        ]
        for entry in StrategyRegistry.all_entries(category="daily"):
            strategies.append(
                DailyToIntradayAdapter(StrategyRegistry.create(entry.name))
            )
        return strategies

    return [
        PullbackStrategy(),
        ReversalStrategy(),
        ORBreakoutStrategy(),
        ORReversalStrategy(),
        PatternScalpStrategy(),
    ]


def _build_options_strategies(args):
    """Build the list of options strategies to backtest."""
    if getattr(args, "strategy", None):
        from stockdownloader.strategies.loader import ensure_registered
        from stockdownloader.strategies.registry import StrategyRegistry

        ensure_registered()
        entry = StrategyRegistry.get(args.strategy)
        if entry.category != "options":
            print(
                f"ERROR: Strategy '{args.strategy}' is category '{entry.category}', "
                "not an options strategy."
            )
            raise SystemExit(1)
        return [StrategyRegistry.create(args.strategy)]

    from stockdownloader.strategies.options.strategies import (
        CoveredCallStrategy,
        ProtectivePutStrategy,
    )

    return [
        CoveredCallStrategy(20, Decimal("0.03"), 30, Decimal("0.03")),
        CoveredCallStrategy(20, Decimal("0.05"), 30, Decimal("0.03")),
        CoveredCallStrategy(50, Decimal("0.05"), 45, Decimal("0.04")),
        ProtectivePutStrategy(20, Decimal("0.05"), 30, 5),
        ProtectivePutStrategy(20, Decimal("0.03"), 45, 10),
        ProtectivePutStrategy(50, Decimal("0.05"), 60, 10),
    ]


# ======================================================================
# Unified runner
# ======================================================================


def _add_intraday_args(parser: argparse.ArgumentParser) -> None:
    """Add intraday-specific arguments to the parser."""
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Calendar days to fetch (default: 60 for Yahoo, 730 for Polygon)",
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="Fetch latest data, merge with CSV on disk, then backtest",
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
        "--all-strategies",
        action="store_true",
        dest="all_strategies",
        help="Run all strategies (VWAP + adapted daily) and compare",
    )


def _run_backtest(
    backtest_type: str = "daily",
    argv: list[str] | None = None,
) -> None:
    """Core backtest runner for all types."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    type_labels = {
        "daily": "Trading Strategy Backtester",
        "intraday": "Intraday Strategy Backtester (5-minute bars)",
        "options": "Options Strategy Backtester",
        "dmi_vwap": "DMI + VWAP Strategy Backtest",
    }

    parser = argparse.ArgumentParser(
        description=type_labels.get(backtest_type, "Strategy Backtester"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_symbol_arg(parser)
    add_csv_arg(parser)

    if backtest_type != "dmi_vwap":
        add_strategy_args(parser)

    if backtest_type in ("intraday", "dmi_vwap"):
        _add_intraday_args(parser)

    args = parser.parse_args(argv)

    # --list-strategies
    if getattr(args, "list_strategies", False):
        category = "options" if backtest_type == "options" else (
            "intraday" if backtest_type == "intraday" else "daily"
        )
        list_strategies_and_exit(category)
        return

    print_banner(type_labels.get(backtest_type, "Backtester"))

    symbol = args.symbol.upper()

    # ── Load data ────────────────────────────────────────────────────
    if backtest_type in ("intraday", "dmi_vwap"):
        data = _load_intraday_data(args, symbol)
        if not data:
            print("ERROR: No intraday data loaded.")
            return

        trading_days = len({bar.date[:10] for bar in data})
        print(f"Loaded {len(data):,} bars across {trading_days} trading days for {symbol}")
        print(f"Date range: {data[0].date} to {data[-1].date}")
        cap_str = INITIAL_CAPITAL.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        print(f"Starting capital: ${cap_str}")
        print(f"Risk per trade: {RISK_PER_TRADE * 100}%")
        print()
    else:
        data = load_or_fetch_daily(symbol, args.csv_file)
        if not data:
            print("ERROR: No price data loaded.")
            return
        extra = [f"Commission: ${OPTIONS_COMMISSION} per contract"] if backtest_type == "options" else None
        print_data_summary(data, symbol, extra_lines=extra)

    # ── Build strategies ─────────────────────────────────────────────
    if backtest_type == "daily":
        strategies = _build_daily_strategies(args)
    elif backtest_type == "intraday":
        strategies = _build_intraday_strategies(args)
    elif backtest_type == "options":
        strategies = _build_options_strategies(args)
    elif backtest_type == "dmi_vwap":
        from stockdownloader.strategies.intraday.dmi_vwap import DmiVwapStrategy

        strategies = [DmiVwapStrategy()]
    else:
        print(f"ERROR: Unknown backtest type '{backtest_type}'")
        return

    # ── Run ──────────────────────────────────────────────────────────
    if backtest_type in ("intraday", "dmi_vwap"):
        from stockdownloader.backtest.intraday_backtest_engine import (
            IntradayBacktestEngine,
        )

        engine = IntradayBacktestEngine(INITIAL_CAPITAL, RISK_PER_TRADE)
    elif backtest_type == "options":
        from stockdownloader.backtest.options_backtest_engine import (
            OptionsBacktestEngine,
        )

        engine = OptionsBacktestEngine(INITIAL_CAPITAL, OPTIONS_COMMISSION)
    else:
        from stockdownloader.backtest.backtest_engine import BacktestEngine

        engine = BacktestEngine(INITIAL_CAPITAL, Decimal("0"))

    results = []
    for strategy in strategies:
        label = backtest_type.replace("_", " ")
        print(f"Running {label} backtest: {strategy.name}...")
        try:
            result = engine.run(strategy, data)
            results.append(result)
            # Use the appropriate report formatter
            if backtest_type in ("intraday", "dmi_vwap"):
                report_formatter.print_intraday_report(result, data)
            elif backtest_type == "options":
                report_formatter.print_options_report(result)
            else:
                report_formatter.print_daily_report(result, data)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback

            traceback.print_exc()

    # ── Comparison ───────────────────────────────────────────────────
    if len(results) > 1:
        if backtest_type in ("intraday", "dmi_vwap"):
            report_formatter.print_intraday_comparison(results, data)
        elif backtest_type == "options":
            report_formatter.print_options_comparison(results)
        else:
            report_formatter.print_daily_comparison(results, data)


# ======================================================================
# CLI entry points (one per pyproject.toml script name)
# ======================================================================


def main() -> None:
    """``spy-backtest`` — daily equity strategy backtester."""
    _run_backtest("daily")


def main_intraday() -> None:
    """``intraday-backtest`` — intraday 5-minute bar backtester."""
    _run_backtest("intraday")


def main_options() -> None:
    """``options-backtest`` — options strategy backtester."""
    _run_backtest("options")


def main_dmi_vwap() -> None:
    """``dmi-vwap-backtest`` — DMI+VWAP intraday strategy."""
    _run_backtest("dmi_vwap")


def main_accumulate() -> None:
    """``intraday-accumulate`` — accumulate intraday 5-minute bar data over time.

    Supports Yahoo Finance (free, ~60-day limit) and Polygon.io (free tier,
    2 years of history).

    Usage::

        intraday-accumulate                                      # Yahoo, SPY, 60 days
        intraday-accumulate --source polygon --days 730          # Polygon, SPY, 2 years
        intraday-accumulate AAPL --source polygon --days 365     # Polygon, AAPL, 1 year
        intraday-accumulate SPY --file my_spy_data.csv           # custom path
    """
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
        help="CSV file path (default: data/{SYMBOL}/5m_bars.csv)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Calendar days to fetch (default: 60 for Yahoo, 730 for Polygon)",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()

    from stockdownloader.data.accumulator import (
        IntradayDataAccumulator,
        default_csv_path,
    )

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
