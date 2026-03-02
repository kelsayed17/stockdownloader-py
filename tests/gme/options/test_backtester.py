"""Tests for GMEOptionsBacktester."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import date
from dataclasses import dataclass

from stockdownloader.gme.options.backtester import (
    GMEOptionsStrategy,
    GMEOptionsBacktester,
    Trade,
    Position,
    BacktestConfig,
    BacktestResult,
)


class DummyStrategy(GMEOptionsStrategy):
    """Always opens a long call on first day."""

    @property
    def name(self) -> str:
        return "dummy"

    def evaluate(self, trade_date, state, chain):
        if chain is not None and not chain.empty:
            row = chain.iloc[0]
            return [Trade(
                option_ticker=row["option_ticker"],
                direction="buy",
                option_type="call",
                strike=row["strike"],
                expiration=row["expiration"],
                premium=row["close"],
                contracts=1,
            )]
        return []

    def on_expiry(self, trade_date, positions):
        return []


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission_per_contract == 0.65


class TestGMEOptionsBacktester:
    def test_runs_with_dummy_strategy(self):
        cfg = BacktestConfig(initial_capital=50_000.0)
        bt = GMEOptionsBacktester(strategies=[DummyStrategy()], config=cfg)
        chain = pd.DataFrame([{
            "option_ticker": "O:GME230721C00025000",
            "strike": 25.0, "option_type": "call",
            "expiration": "2023-07-21", "close": 2.50,
            "volume": 100, "open_interest": 500, "date": "2023-06-15",
        }])
        state = pd.DataFrame([{"date": date(2023, 6, 15), "atm_iv_30d": 0.6, "net_gex": 1000.0}])
        result = bt.run_single_day(date(2023, 6, 15), state.iloc[0], chain)
        assert isinstance(result, list)

    def test_tracks_positions(self):
        cfg = BacktestConfig()
        bt = GMEOptionsBacktester(strategies=[DummyStrategy()], config=cfg)
        chain = pd.DataFrame([{
            "option_ticker": "O:GME230721C00025000",
            "strike": 25.0, "option_type": "call",
            "expiration": "2023-07-21", "close": 2.50,
            "volume": 100, "open_interest": 500, "date": "2023-06-15",
        }])
        state_row = pd.Series({"date": date(2023, 6, 15), "atm_iv_30d": 0.6})
        bt.run_single_day(date(2023, 6, 15), state_row, chain)
        assert len(bt.positions) > 0

    def test_compute_metrics(self):
        cfg = BacktestConfig(initial_capital=50_000.0)
        bt = GMEOptionsBacktester(strategies=[DummyStrategy()], config=cfg)
        bt.equity_curve = [50_000.0, 50_250.0, 50_100.0, 50_500.0]
        metrics = bt.compute_metrics()
        assert "total_return_pct" in metrics
        assert "max_drawdown_pct" in metrics
        assert "sharpe" in metrics
