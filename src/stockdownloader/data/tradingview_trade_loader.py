"""Loads trades from TradingView strategy tester CSV exports.

TradingView exports two rows per trade (entry + exit) with columns like:

    Trade #, Type, Signal, Date and time, Price USD, Contracts/Shares,
    Profit USD, Cumulative Profit USD, Run-up USD, Drawdown USD, ...

This loader pairs entry/exit rows into :class:`TournamentTrade` objects
suitable for the exit tournament engine.
"""
from __future__ import annotations

import contextlib
import csv
import logging
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO

from stockdownloader.model.tournament_trade import TournamentTrade
from stockdownloader.model.trade import Direction
from stockdownloader.util.big_decimal_math import ZERO

logger = logging.getLogger(__name__)

class TradingViewTradeLoader:
    """Loads :class:`TournamentTrade` objects from TradingView CSV exports.

    All methods are static; the class is not intended to be instantiated.
    """

    def __init__(self) -> None:  # pragma: no cover
        raise TypeError("TradingViewTradeLoader should not be instantiated")

    @staticmethod
    def load_from_file(
        filename: str | Path,
        default_stop_distance: Decimal | None = None,
        timezone_offset_hours: int = 0,
    ) -> list[TournamentTrade]:
        """Load trades from a TradingView CSV export file.

        Args:
            filename: Path to the CSV file.
            default_stop_distance: Default stop distance (1R) in dollars
                if the signal string does not encode one.
            timezone_offset_hours: Hours to add to TradingView timestamps
                to align with the bar data timezone (e.g. 3 for PST -> EST).

        Returns:
            List of parsed :class:`TournamentTrade` instances.
        """
        try:
            with open(filename, newline="", encoding="utf-8") as fh:
                return _parse_trades(fh, default_stop_distance, timezone_offset_hours)
        except (OSError, csv.Error, ValueError) as exc:
            logger.warning("Error loading TradingView CSV %s: %s", filename, exc)
            return []

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _parse_trades(
    text_io,
    default_stop_distance: Decimal | None,
    tz_offset_hours: int,
) -> list[TournamentTrade]:
    reader = csv.DictReader(text_io)

    # Group rows by Trade #
    trade_rows: dict[str, list[dict]] = {}
    for row in reader:
        tn = row.get("Trade #", "").strip()
        if not tn:
            continue
        trade_rows.setdefault(tn, []).append(row)

    trades: list[TournamentTrade] = []
    for tn, rows in trade_rows.items():
        if len(rows) < 2:
            continue

        # Sort by datetime
        rows.sort(key=lambda r: r.get("Date and time", ""))
        entry_row = rows[0]
        exit_row = rows[1]

        try:
            trade = _build_tournament_trade(
                tn, entry_row, exit_row, default_stop_distance, tz_offset_hours
            )
            if trade is not None:
                trades.append(trade)
        except (ValueError, KeyError, InvalidOperation, TypeError) as exc:
            logger.debug("Skipping trade #%s: %s", tn, exc)
            continue

    return trades

def _build_tournament_trade(
    trade_num_str: str,
    entry_row: dict,
    exit_row: dict,
    default_stop_distance: Decimal | None,
    tz_offset_hours: int,
) -> TournamentTrade | None:
    """Build a single TournamentTrade from entry/exit row pair."""

    # Determine direction
    direction = _parse_direction(entry_row)
    if direction is None:
        return None

    # Parse signal type
    signal = entry_row.get("Signal", "")
    signal_type = _parse_signal_type(signal)

    # Parse prices and P&L
    entry_price = _parse_decimal(entry_row, "Price USD")
    if entry_price is None:
        entry_price = _parse_decimal(entry_row, "Price USDT")
    if entry_price is None:
        return None

    exit_price = _parse_decimal(exit_row, "Price USD")
    if exit_price is None:
        exit_price = _parse_decimal(exit_row, "Price USDT")
    if exit_price is None:
        return None

    pnl = _parse_decimal(exit_row, "Profit USD")
    if pnl is None:
        pnl = _parse_decimal(exit_row, "Profit USDT")
    if pnl is None:
        pnl = ZERO

    # Parse datetimes
    entry_dt = entry_row.get("Date and time", "").strip()
    exit_dt = exit_row.get("Date and time", "").strip()

    # Parse stop distance from signal or use default
    stop_distance = _parse_stop_distance(signal, entry_price)
    if stop_distance is None or stop_distance <= ZERO:
        if default_stop_distance is not None and default_stop_distance > ZERO:
            stop_distance = default_stop_distance
        else:
            # Fallback: use adverse excursion if available, else 1% of entry
            ae = _parse_decimal(exit_row, "Drawdown USD")
            if ae is not None and ae > ZERO:
                stop_distance = ae
            else:
                stop_distance = (entry_price * Decimal("0.01")).quantize(
                    Decimal("0.01")
                )

    # Parse trade number
    try:
        trade_id = int(trade_num_str)
    except ValueError:
        trade_id = 0

    return TournamentTrade(
        trade_id=trade_id,
        direction=direction,
        signal_type=signal_type,
        entry_datetime=entry_dt,
        exit_datetime=exit_dt,
        entry_price=entry_price,
        original_exit_price=exit_price,
        original_pnl=pnl,
        stop_distance=stop_distance,
    )

def _parse_direction(row: dict) -> Direction | None:
    """Determine trade direction from the row."""
    signal = str(row.get("Signal", ""))
    type_col = str(row.get("Type", ""))

    # Check signal string for direction markers
    if "|L|" in signal:
        return Direction.LONG
    if "|S|" in signal:
        return Direction.SHORT

    # Fall back to Type column
    type_lower = type_col.lower()
    if "long" in type_lower or "buy" in type_lower:
        return Direction.LONG
    if "short" in type_lower or "sell" in type_lower:
        return Direction.SHORT

    return None

def _parse_signal_type(signal: str) -> str:
    """Extract strategy type from signal string (e.g. 'PB|L|S:3')."""
    if not signal:
        return "UNK"
    parts = signal.split("|")
    if parts:
        tag = parts[0].strip()
        if tag in ("PB", "REV", "ORB", "PS"):
            return tag
    return "UNK"

def _parse_stop_distance(signal: str, entry_price: Decimal) -> Decimal | None:
    """Try to extract stop distance from signal string."""
    # Look for patterns like |SL:675.50| or |RR:1.4|
    # If we find RR and a score, we can estimate the stop
    if not signal:
        return None

    # Try to find ATR-based stop in signal
    atr_match = re.search(r"\|ATR:([0-9.]+)", signal)
    if atr_match:
        with contextlib.suppress(InvalidOperation):
            atr_val = Decimal(atr_match.group(1))
            return atr_val * Decimal("1.5")  # typical ATR multiplier

    return None

def _parse_decimal(row: dict, key: str) -> Decimal | None:
    """Parse a Decimal from the named column, returning None on failure."""
    raw = row.get(key, "").strip()
    if not raw:
        return None
    # Remove currency symbols, commas, percentage signs
    cleaned = raw.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None
