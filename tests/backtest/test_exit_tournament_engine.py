"""Tests for ExitTournamentEngine."""

from decimal import Decimal

import pytest

from stockdownloader.backtest.exit_tournament_engine import ExitTournamentEngine
from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import Direction, TournamentTrade
from stockdownloader.strategy.exit_mechanisms import TrailingStopExit


def _make_bar(dt_str, open_, high, low, close, volume=100000):
    return IntradayPriceData(
        date=dt_str,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        adj_close=Decimal(str(close)),
        volume=volume,
    )


def _make_trade(trade_id=1, entry_time="2025-12-01 10:00:00-05:00",
                entry_price=100, stop_dist=1, direction=Direction.LONG):
    return TournamentTrade(
        trade_id=trade_id,
        direction=direction,
        signal_type="PB",
        entry_datetime=entry_time,
        exit_datetime="2025-12-01 12:00:00-05:00",
        entry_price=Decimal(str(entry_price)),
        original_exit_price=Decimal(str(entry_price + 1)),
        original_pnl=Decimal("1"),
        stop_distance=Decimal(str(stop_dist)),
    )


def _session_bars():
    """Full session of bars on 2025-12-01, rising then flat."""
    base = "2025-12-01"
    return [
        _make_bar(f"{base} 09:30:00-05:00", 99, 100, 98.5, 99.5),
        _make_bar(f"{base} 09:35:00-05:00", 99.5, 100.2, 99, 100),
        _make_bar(f"{base} 09:40:00-05:00", 100, 100.5, 99.8, 100.3),
        _make_bar(f"{base} 09:45:00-05:00", 100.3, 100.8, 100, 100.5),
        _make_bar(f"{base} 09:50:00-05:00", 100.5, 101, 100.3, 100.8),
        _make_bar(f"{base} 09:55:00-05:00", 100.8, 101.2, 100.5, 101),
        _make_bar(f"{base} 10:00:00-05:00", 101, 101.5, 100.8, 101.2),
        _make_bar(f"{base} 10:05:00-05:00", 101.2, 101.8, 101, 101.5),
        _make_bar(f"{base} 10:10:00-05:00", 101.5, 102, 101.2, 101.8),
        _make_bar(f"{base} 10:15:00-05:00", 101.8, 102, 101.5, 101.7),
        _make_bar(f"{base} 10:20:00-05:00", 101.7, 101.8, 101, 101.2),
        _make_bar(f"{base} 10:25:00-05:00", 101.2, 101.5, 100.8, 101),
        _make_bar(f"{base} 10:30:00-05:00", 101, 101.2, 100.5, 100.8),
        _make_bar(f"{base} 10:35:00-05:00", 100.8, 101, 100.5, 100.7),
        _make_bar(f"{base} 10:40:00-05:00", 100.7, 100.8, 100, 100.2),
        _make_bar(f"{base} 10:45:00-05:00", 100.2, 100.5, 100, 100.3),
    ]


class TestExitTournamentEngine:

    def test_run_single_trade_single_mechanism(self):
        engine = ExitTournamentEngine()
        bars = _session_bars()
        trade = _make_trade(entry_time="2025-12-01 10:00:00-05:00", entry_price=101.2)
        mechanisms = [TrailingStopExit()]

        result = engine.run([trade], bars, mechanisms)
        assert result.trades_simulated == 1
        assert result.trades_skipped == 0
        assert len(result.all_results) == 1

    def test_run_multiple_mechanisms(self):
        engine = ExitTournamentEngine()
        bars = _session_bars()
        trade = _make_trade(entry_time="2025-12-01 10:00:00-05:00", entry_price=101.2)
        mechanisms = [TrailingStopExit(), TrailingStopExit()]

        result = engine.run([trade], bars, mechanisms)
        assert len(result.all_results) == 2

    def test_skips_trades_not_in_data(self):
        engine = ExitTournamentEngine()
        bars = _session_bars()
        # Trade on a different day
        trade = _make_trade(entry_time="2025-12-05 10:00:00-05:00")
        mechanisms = [TrailingStopExit()]

        result = engine.run([trade], bars, mechanisms)
        assert result.trades_simulated == 0
        assert result.trades_skipped == 1

    def test_result_has_mechanism_summaries(self):
        engine = ExitTournamentEngine()
        bars = _session_bars()
        trade = _make_trade(entry_time="2025-12-01 10:00:00-05:00", entry_price=101.2)
        mechanisms = [TrailingStopExit()]

        result = engine.run([trade], bars, mechanisms)
        summaries = result.mechanism_summaries
        assert "CURRENT_TRAIL" in summaries
        assert summaries["CURRENT_TRAIL"].total_trades == 1

    def test_empty_trades_raises(self):
        engine = ExitTournamentEngine()
        bars = _session_bars()
        with pytest.raises(ValueError):
            engine.run([], bars, [TrailingStopExit()])

    def test_empty_data_raises(self):
        engine = ExitTournamentEngine()
        trade = _make_trade()
        with pytest.raises(ValueError):
            engine.run([trade], [], [TrailingStopExit()])

    def test_empty_mechanisms_raises(self):
        engine = ExitTournamentEngine()
        bars = _session_bars()
        trade = _make_trade()
        with pytest.raises(ValueError):
            engine.run([trade], bars, [])

    def test_pnl_positive_for_winning_trade(self):
        engine = ExitTournamentEngine()
        bars = _session_bars()
        # Enter at 100 with stop at 97 (1R = 3.0)
        # First bar low=98.5 is above stop=97, so no immediate stop-out
        # Price rises to 102, should capture profit or exit at session end
        trade = _make_trade(
            entry_time="2025-12-01 09:30:00-05:00",
            entry_price=100,
            stop_dist=3,
        )
        mechanisms = [TrailingStopExit()]
        result = engine.run([trade], bars, mechanisms)

        trade_result = result.all_results[0]
        # The trade should exit with positive P&L (price rose from 100)
        assert trade_result.pnl > Decimal("0")
