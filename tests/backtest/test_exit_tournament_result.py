"""Tests for ExitTournamentResult."""

from decimal import Decimal

from stockdownloader.backtest.exit_tournament_result import ExitTournamentResult
from stockdownloader.model.exit_mechanism_result import ExitMechanismTradeResult
from stockdownloader.model.trade import Direction


def _make_result(mechanism, trade_id, pnl, direction=Direction.LONG, signal_type="PB"):
    return ExitMechanismTradeResult(
        mechanism_name=mechanism,
        trade_id=trade_id,
        direction=direction,
        signal_type=signal_type,
        entry_price=Decimal("680"),
        exit_price=Decimal("680") + Decimal(str(pnl)),
        exit_datetime="2025-12-01 11:00:00-05:00",
        pnl=Decimal(str(pnl)),
        r_multiple=Decimal(str(pnl)),
        holding_bars=10,
        peak_favorable=Decimal("2"),
        capture_pct=Decimal("50"),
        exit_reason="trail_stop",
    )


class TestExitTournamentResult:

    def test_add_results(self):
        result = ExitTournamentResult()
        result.add_result(_make_result("MECH_A", 1, 1.5))
        result.add_result(_make_result("MECH_B", 1, 2.0))
        assert len(result.all_results) == 2
        assert "MECH_A" in result.mechanism_names
        assert "MECH_B" in result.mechanism_names

    def test_mechanism_summaries(self):
        result = ExitTournamentResult()
        result.add_result(_make_result("MECH_A", 1, 1.5))
        result.add_result(_make_result("MECH_A", 2, -0.5))
        summaries = result.mechanism_summaries
        assert summaries["MECH_A"].total_trades == 2

    def test_get_best_mechanism(self):
        result = ExitTournamentResult()
        result.add_result(_make_result("MECH_A", 1, 1.0))
        result.add_result(_make_result("MECH_B", 1, 3.0))
        assert result.get_best_mechanism("total_pnl") == "MECH_B"

    def test_breakdown_by_direction(self):
        result = ExitTournamentResult()
        result.add_result(_make_result("MECH_A", 1, 1.0, Direction.LONG))
        result.add_result(_make_result("MECH_A", 2, -0.5, Direction.SHORT))
        breakdown = result.get_breakdown_by_direction("MECH_A")
        assert "LONG" in breakdown
        assert "SHORT" in breakdown
        assert breakdown["LONG"].total_trades == 1
        assert breakdown["SHORT"].total_trades == 1

    def test_breakdown_by_signal_type(self):
        result = ExitTournamentResult()
        result.add_result(_make_result("MECH_A", 1, 1.0, signal_type="PB"))
        result.add_result(_make_result("MECH_A", 2, 2.0, signal_type="REV"))
        result.add_result(_make_result("MECH_A", 3, -1.0, signal_type="PB"))
        breakdown = result.get_breakdown_by_signal_type("MECH_A")
        assert "PB" in breakdown
        assert "REV" in breakdown
        assert breakdown["PB"].total_trades == 2

    def test_head_to_head(self):
        result = ExitTournamentResult()
        # Trade 1: MECH_A wins
        result.add_result(_make_result("MECH_A", 1, 2.0))
        result.add_result(_make_result("MECH_B", 1, 1.0))
        # Trade 2: MECH_B wins
        result.add_result(_make_result("MECH_A", 2, 0.5))
        result.add_result(_make_result("MECH_B", 2, 1.5))
        # Trade 3: MECH_A wins
        result.add_result(_make_result("MECH_A", 3, 3.0))
        result.add_result(_make_result("MECH_B", 3, 0.5))

        h2h = result.head_to_head
        assert h2h["MECH_A"] == 2
        assert h2h["MECH_B"] == 1

    def test_trades_simulated_and_skipped(self):
        result = ExitTournamentResult()
        result.trades_simulated = 10
        result.trades_skipped = 3
        assert result.trades_simulated == 10
        assert result.trades_skipped == 3

    def test_get_summary_returns_none_for_unknown(self):
        result = ExitTournamentResult()
        assert result.get_summary("NONEXISTENT") is None
