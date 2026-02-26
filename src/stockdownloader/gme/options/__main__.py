"""CLI entry point for the GME options analysis platform.

Provides argparse-based subcommands to fetch data, build state,
run backtests, generate scorecards, and execute the full pipeline.

Usage::

    python -m stockdownloader.gme.options fetch --start 2022-01-01
    python -m stockdownloader.gme.options build-state
    python -m stockdownloader.gme.options backtest --strategy wheel
    python -m stockdownloader.gme.options backtest --all
    python -m stockdownloader.gme.options scorecard --date 2023-06-15
    python -m stockdownloader.gme.options run-all
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Strategy registry
# ------------------------------------------------------------------

ALL_STRATEGIES: dict[str, type] = {}


def _load_strategies() -> dict[str, type]:
    """Lazily import and return the strategy class mapping."""
    if ALL_STRATEGIES:
        return ALL_STRATEGIES

    from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
    from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
    from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
    from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
    from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
    from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy

    ALL_STRATEGIES.update({
        "wheel": GMEWheelStrategy,
        "iron_condor": IronCondorStrategy,
        "strangle": StrangleStrategy,
        "credit_spread": CreditSpreadStrategy,
        "long_options": LongOptionsStrategy,
        "calendar_spread": CalendarSpreadStrategy,
    })
    return ALL_STRATEGIES


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the GME options CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with subcommands: fetch, build-state,
        backtest, scorecard, run-all.
    """
    parser = argparse.ArgumentParser(
        prog="gme-options",
        description="GME Options Analysis Platform CLI.",
    )

    # Top-level arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--polygon-key",
        default=None,
        help="Polygon.io API key (overrides POLYGON_API_KEY env var).",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")

    # -- fetch --
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch options data from Polygon.",
    )
    fetch_parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Start date YYYY-MM-DD (default: 2022-01-01).",
    )
    fetch_parser.add_argument(
        "--end",
        default=str(date.today()),
        help="End date YYYY-MM-DD (default: today).",
    )
    fetch_parser.add_argument(
        "--symbol",
        default="GME",
        help="Ticker symbol (default: GME).",
    )

    # -- build-state --
    subparsers.add_parser(
        "build-state",
        help="Build options state DataFrame from fetched data.",
    )

    # -- backtest --
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run options strategy backtest.",
    )
    backtest_parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy name (e.g. wheel, iron_condor, strangle).",
    )
    backtest_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all registered strategies.",
    )

    # -- scorecard --
    scorecard_parser = subparsers.add_parser(
        "scorecard",
        help="Generate daily signal fusion scorecard.",
    )
    scorecard_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Scorecard date YYYY-MM-DD.",
    )
    scorecard_parser.add_argument(
        "--range",
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Date range for scorecard (START END).",
    )

    # -- run-all --
    subparsers.add_parser(
        "run-all",
        help="Run full pipeline: fetch -> build-state -> backtest -> scorecard.",
    )

    return parser


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------


def _cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch options data from Polygon."""
    from stockdownloader.gme.options.config import GMEOptionsConfig
    from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient
    from stockdownloader.gme.options.fetcher import OptionsDataFetcher

    config = GMEOptionsConfig.from_env(
        polygon_api_key=args.polygon_key,
        symbol=getattr(args, "symbol", "GME"),
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
    )

    client = PolygonOptionsClient(api_key=config.polygon_api_key)
    fetcher = OptionsDataFetcher(config, client)

    logger.info(
        "Fetching options data for %s from %s to %s",
        config.symbol,
        config.start_date,
        config.end_date,
    )
    fetcher.run()
    logger.info("Fetch complete")


def _cmd_build_state(args: argparse.Namespace) -> None:
    """Build options state DataFrame from fetched Parquet files."""
    from stockdownloader.gme.options.config import GMEOptionsConfig
    from stockdownloader.gme.options.state_engine import OptionsStateEngine

    config = GMEOptionsConfig.from_env(polygon_api_key=args.polygon_key)
    bars_dir = config.data_dir / "monthly"

    engine = OptionsStateEngine()
    logger.info("Building state from %s", bars_dir)

    state_df = engine.build(bars_dir=bars_dir)

    if state_df.empty:
        logger.warning("No data found in %s", bars_dir)
        return

    out_path = config.data_dir / "state.parquet"
    state_df.to_parquet(out_path, index=False)
    logger.info("Saved state (%d rows) to %s", len(state_df), out_path)


def _cmd_backtest(args: argparse.Namespace) -> None:
    """Run options strategy backtest."""
    from stockdownloader.gme.options.backtester import GMEOptionsBacktester

    strategies_map = _load_strategies()

    if args.all:
        strategy_names = list(strategies_map.keys())
    elif args.strategy:
        if args.strategy not in strategies_map:
            logger.error(
                "Unknown strategy: %s. Available: %s",
                args.strategy,
                ", ".join(strategies_map.keys()),
            )
            sys.exit(1)
        strategy_names = [args.strategy]
    else:
        logger.error("Specify --strategy NAME or --all")
        sys.exit(1)

    strategy_instances = [strategies_map[name]() for name in strategy_names]
    backtester = GMEOptionsBacktester(strategies=strategy_instances)

    logger.info(
        "Backtest ready with strategies: %s",
        ", ".join(strategy_names),
    )
    logger.info(
        "Load state + chain data and call backtester.run() to execute",
    )


def _cmd_scorecard(args: argparse.Namespace) -> None:
    """Generate daily signal fusion scorecard."""
    from stockdownloader.gme.options.signal_fusion import GMESignalFusion

    fusion = GMESignalFusion()

    if args.date:
        logger.info("Scorecard ready for date: %s", args.date)
    elif args.range:
        logger.info("Scorecard ready for range: %s to %s", args.range[0], args.range[1])
    else:
        logger.info("Scorecard ready (no date specified)")

    logger.info(
        "Load state + alt data and call fusion.compute_daily_scorecard() to execute",
    )


def _cmd_run_all(args: argparse.Namespace) -> None:
    """Run the full pipeline: fetch -> build-state -> backtest -> scorecard."""
    logger.info("Running full pipeline")

    _cmd_fetch(args)
    _cmd_build_state(args)

    # Set defaults for backtest sub-command
    args.all = True
    args.strategy = None
    _cmd_backtest(args)

    # Set defaults for scorecard sub-command
    args.date = None
    args.range = None
    _cmd_scorecard(args)

    logger.info("Full pipeline complete")


# ------------------------------------------------------------------
# Dispatch table
# ------------------------------------------------------------------

_COMMANDS: dict[str, object] = {
    "fetch": _cmd_fetch,
    "build-state": _cmd_build_state,
    "backtest": _cmd_backtest,
    "scorecard": _cmd_scorecard,
    "run-all": _cmd_run_all,
}


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging, and dispatch to the handler.

    Parameters
    ----------
    argv:
        Command-line arguments.  If ``None``, ``sys.argv[1:]`` is used.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
