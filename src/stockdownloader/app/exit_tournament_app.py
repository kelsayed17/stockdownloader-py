"""Main entry point for the exit mechanism tournament.

Compares multiple exit mechanisms against historical trades using 5-minute
intraday bar data.

Usage:
    python -m stockdownloader.app.exit_tournament_app \\
        --bars data/spy/5m_bars.csv \\
        --trades trades.csv

    python -m stockdownloader.app.exit_tournament_app \\
        --bars data/spy/5m_bars.csv \\
        --trades trades.csv \\
        --stop-distance 0.96
"""
from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal

from stockdownloader.backtest.exit_tournament_engine import ExitTournamentEngine
from stockdownloader.backtest import exit_tournament_report_formatter
from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.data.tradingview_trade_loader import TradingViewTradeLoader
from stockdownloader.strategy.exit_mechanisms import (
    AtrTrailExit,
    HybridExit,
    TimeDecayExit,
    TrailingStopExit,
    VwapBandExit,
    VwapCrossExit,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the exit tournament application."""
    parser = argparse.ArgumentParser(
        description="Exit Mechanism Tournament",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  exit_tournament_app --bars data/spy/5m_bars.csv --trades trades.csv\n"
            "  exit_tournament_app --bars data/spy/5m_bars.csv --trades trades.csv "
            "--stop-distance 0.96\n"
        ),
    )
    parser.add_argument(
        "--bars", required=True, help="Path to 5-minute OHLCV CSV file"
    )
    parser.add_argument(
        "--trades", required=True, help="Path to TradingView trade export CSV"
    )
    parser.add_argument(
        "--stop-distance",
        type=float,
        default=None,
        help="Default stop distance (1R) in dollars if not inferred from signals",
    )
    parser.add_argument(
        "--tz-offset",
        type=int,
        default=0,
        help="Hours to add to TradingView timestamps (e.g. 3 for PST->EST)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Label for the report header",
    )
    args = parser.parse_args()

    print("========================================")
    print("  Exit Mechanism Tournament")
    print("========================================")
    print()

    # ── Load intraday bars ───────────────────────────────────────────

    print(f"Loading 5-minute bars from: {args.bars}")
    bars = IntradayCsvLoader.load_from_file(args.bars)
    if not bars:
        print("ERROR: No bar data loaded.")
        sys.exit(1)

    # Determine date range from bar data
    dates = sorted({b.date[:10] for b in bars})
    print(f"  {len(bars)} bars, {len(dates)} sessions")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print()

    # ── Load trades ──────────────────────────────────────────────────

    print(f"Loading trades from: {args.trades}")
    default_sd = (
        Decimal(str(args.stop_distance))
        if args.stop_distance is not None
        else None
    )
    trades = TradingViewTradeLoader.load_from_file(
        args.trades,
        default_stop_distance=default_sd,
        timezone_offset_hours=args.tz_offset,
    )
    if not trades:
        print("ERROR: No trades loaded.")
        sys.exit(1)

    print(f"  {len(trades)} trades loaded")

    # Filter trades to those within the bar data window
    bar_date_set = set(dates)
    trades = [t for t in trades if t.entry_trading_date in bar_date_set]
    print(f"  {len(trades)} trades within bar data window")
    print()

    if not trades:
        print("ERROR: No trades overlap with bar data date range.")
        sys.exit(1)

    # ── Create exit mechanisms ───────────────────────────────────────

    mechanisms = [
        TrailingStopExit(),                                             # CURRENT_TRAIL
        VwapCrossExit(activation_r=Decimal("0.5"), name="VWAP_CROSS"),  # VWAP_CROSS
        VwapCrossExit(activation_r=Decimal("1.0"), name="VWAP_CROSS_LATE"),
        AtrTrailExit(),                                                 # ATR_TRAIL
        HybridExit(activation_r=Decimal("0.5"), name="HYBRID"),         # HYBRID
        HybridExit(activation_r=Decimal("1.0"), name="HYBRID_LATE"),
        VwapBandExit(),                                                 # VWAP_BAND
        TimeDecayExit(),                                                # TIME_DECAY
    ]

    # ── Run tournament ───────────────────────────────────────────────

    print("Running tournament...")
    engine = ExitTournamentEngine()
    result = engine.run(trades, bars, mechanisms)

    # ── Print report ─────────────────────────────────────────────────

    exit_tournament_report_formatter.print_report(result, label=args.label)
    exit_tournament_report_formatter.print_comparison(result)


if __name__ == "__main__":
    main()
