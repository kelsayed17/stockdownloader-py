"""CLI argument builder helpers for argparse-based app entry points."""

from __future__ import annotations

import argparse

from stockdownloader.util.constants import DEFAULT_DATA_FILE


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


def add_intraday_csv_arg(parser: argparse.ArgumentParser) -> None:
    """Add a ``--csv`` argument defaulting to the standard intraday CSV.

    This is the intraday-specific variant of :func:`add_csv_arg` — it
    defaults to :data:`DEFAULT_DATA_FILE` (``data/spy/5m_bars.csv``)
    instead of ``None``.
    """
    parser.add_argument(
        "--csv", dest="csv_file", default=str(DEFAULT_DATA_FILE),
        help="Intraday CSV file path (default: data/spy/5m_bars.csv)",
    )


def add_log_arg(
    parser: argparse.ArgumentParser,
    *,
    help: str = "Write output to a log file",
    metavar: str | None = None,
) -> None:
    """Add a ``--log`` argument for optional log-file output.

    All callers share ``dest="log_file"`` and ``default=None``.
    """
    kwargs: dict = dict(dest="log_file", default=None, help=help)
    if metavar is not None:
        kwargs["metavar"] = metavar
    parser.add_argument("--log", **kwargs)


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
