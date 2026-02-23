"""Exit tournament engine: replays trades through multiple exit mechanisms.

For each trade, the engine:
1. Finds the entry bar in the intraday data.
2. Replays forward bar-by-bar through the session.
3. Asks each :class:`ExitMechanism` whether it would exit on each bar.
4. Records the result for each mechanism.
5. Aggregates into :class:`ExitTournamentResult`.
"""
from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING

from stockdownloader.backtest.exit_tournament_result import ExitTournamentResult
from stockdownloader.model.exit_mechanism_result import ExitMechanismTradeResult
from stockdownloader.model.trade import Direction
from stockdownloader.util.math import HUNDRED, ZERO

if TYPE_CHECKING:
    from stockdownloader.model.price_data import IntradayPriceData
    from stockdownloader.model.trade import TournamentTrade
    from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import ExitMechanism

logger = logging.getLogger(__name__)

class ExitTournamentEngine:
    """Runs trades through multiple exit mechanisms and collects results.

    Modelled after :class:`BacktestEngine`, but instead of executing a
    forward-pass strategy the engine *replays* known trade entries through
    alternative exit mechanisms on intraday bar data.
    """

    def run(
        self,
        trades: list[TournamentTrade],
        data: list[IntradayPriceData],
        mechanisms: list[ExitMechanism],
    ) -> ExitTournamentResult:
        """Execute the tournament.

        Args:
            trades: Trade entries to evaluate (from TradingView export).
            data: 5-minute intraday bar data.
            mechanisms: Exit mechanisms to compare.

        Returns:
            :class:`ExitTournamentResult` with all per-trade, per-mechanism
            results and aggregate summaries.
        """
        if not trades:
            raise ValueError("trades must not be empty")
        if not data:
            raise ValueError("data must not be empty")
        if not mechanisms:
            raise ValueError("mechanisms must not be empty")

        result = ExitTournamentResult()

        # Pre-index: build date -> bar index ranges for fast session lookup
        date_ranges = _build_date_ranges(data)

        simulated = 0
        skipped = 0

        for trade in trades:
            entry_idx = _find_entry_bar(trade, data, date_ranges)
            if entry_idx is None:
                skipped += 1
                continue

            # Determine session end (last bar of the same trading day)
            trading_date = data[entry_idx].date[:10]
            session_end_idx = date_ranges.get(trading_date, (entry_idx, entry_idx))[1]

            # Need at least 2 bars from entry to session end
            if session_end_idx - entry_idx < 1:
                skipped += 1
                continue

            for mechanism in mechanisms:
                trade_result = _replay_trade(
                    trade, data, entry_idx, session_end_idx, mechanism
                )
                result.add_result(trade_result)

            simulated += 1

        result.trades_simulated = simulated
        result.trades_skipped = skipped

        return result

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_date_ranges(
    data: list[IntradayPriceData],
) -> dict[str, tuple[int, int]]:
    """Build a mapping from trading date to (first_idx, last_idx) in data."""
    ranges: dict[str, tuple[int, int]] = {}
    for i, bar in enumerate(data):
        day = bar.date[:10]
        if day not in ranges:
            ranges[day] = (i, i)
        else:
            ranges[day] = (ranges[day][0], i)
    return ranges

def _find_entry_bar(
    trade: TournamentTrade,
    data: list[IntradayPriceData],
    date_ranges: dict[str, tuple[int, int]],
) -> int | None:
    """Find the bar index closest to the trade entry datetime.

    Searches within the entry's trading day for the bar whose datetime
    is closest to (but not after) the entry datetime.
    """
    entry_day = trade.entry_trading_date
    if entry_day not in date_ranges:
        return None

    start_idx, end_idx = date_ranges[entry_day]
    entry_dt_str = trade.entry_datetime

    best_idx = start_idx
    best_diff = float("inf")

    for i in range(start_idx, end_idx + 1):
        bar_dt_str = data[i].date
        # Compare as strings -- works for ISO-format datetimes
        if bar_dt_str <= entry_dt_str:
            # This bar is at or before entry -- prefer it
            best_idx = i
            best_diff = 0  # exact or earlier is always best candidate
        elif best_diff > 0:
            # First bar after entry -- use it if nothing else found
            best_idx = i
            break

    return best_idx

def _replay_trade(
    trade: TournamentTrade,
    data: list[IntradayPriceData],
    entry_idx: int,
    session_end_idx: int,
    mechanism: ExitMechanism,
) -> ExitMechanismTradeResult:
    """Replay a single trade through one exit mechanism.

    Iterates bar-by-bar from entry to session end, calling
    ``mechanism.evaluate_bar()`` on each bar.  If no exit is triggered
    before session end, the trade exits at the session close.
    """
    mechanism.reset()

    # Set data context for mechanisms that need full data (VWAP, ATR)
    if hasattr(mechanism, "set_data_context"):
        mechanism.set_data_context(data, entry_idx)

    entry = trade.entry_price
    stop_dist = trade.stop_distance
    direction = trade.direction

    exit_price = ZERO
    exit_datetime = ""
    exit_reason = "session_end"
    holding_bars = 0

    # Track peak favorable
    peak = entry

    for bar_offset in range(session_end_idx - entry_idx + 1):
        data_idx = entry_idx + bar_offset
        bar = data[data_idx]

        # Track peak for our own capture calculation
        if direction == Direction.LONG:
            if bar.high > peak:
                peak = bar.high
        else:
            if bar.low < peak:
                peak = bar.low

        exited = mechanism.evaluate_bar(bar, trade, bar_offset)
        if exited:
            exit_price = mechanism.exit_price
            exit_datetime = bar.date
            exit_reason = mechanism.exit_reason
            holding_bars = bar_offset
            break
    else:
        # Session end -- exit at last bar's close
        last_bar = data[session_end_idx]
        exit_price = last_bar.close
        exit_datetime = last_bar.date
        exit_reason = "session_end"
        holding_bars = session_end_idx - entry_idx

    # Compute P&L and metrics
    if direction == Direction.LONG:
        pnl = exit_price - entry
        max_fav = peak - entry
    else:
        pnl = entry - exit_price
        max_fav = entry - peak

    r_multiple = ZERO
    if stop_dist > ZERO:
        r_multiple = (pnl / stop_dist).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    capture = ZERO
    if max_fav > ZERO:
        capture = (pnl / max_fav * HUNDRED).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    return ExitMechanismTradeResult(
        mechanism_name=mechanism.name,
        trade_id=trade.trade_id,
        direction=direction,
        signal_type=trade.signal_type,
        entry_price=entry,
        exit_price=exit_price,
        exit_datetime=exit_datetime,
        pnl=pnl,
        r_multiple=r_multiple,
        holding_bars=holding_bars,
        peak_favorable=max_fav,
        capture_pct=capture,
        exit_reason=exit_reason,
    )
