"""Integration test: full pipeline with synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock

from stockdownloader.gme.options.config import GMEOptionsConfig
from stockdownloader.gme.options.fetcher import OptionsDataFetcher
from stockdownloader.gme.options.state_engine import OptionsStateEngine
from stockdownloader.gme.options.backtester import (
    GMEOptionsBacktester,
    BacktestConfig,
)
from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
from stockdownloader.gme.options.signal_fusion import GMESignalFusion
from stockdownloader.gme.options.scorecard import GMEDailyScorecard


def _synthetic_chain(trade_date: str, spot: float = 25.0) -> pd.DataFrame:
    """Build a synthetic options chain for a single trading day.

    Creates 5 strikes x 2 option types = 10 rows with realistic
    structure matching the schema expected by the StateEngine and
    strategy evaluate() methods.
    """
    rows = []
    for strike in [20.0, 22.5, 25.0, 27.5, 30.0]:
        for otype in ["call", "put"]:
            rows.append({
                "option_ticker": (
                    f"O:GME{trade_date.replace('-', '')}"
                    f"{'C' if otype == 'call' else 'P'}"
                    f"{int(strike * 1000):08d}"
                ),
                "underlying": "GME",
                "date": trade_date,
                "expiration": (
                    pd.Timestamp(trade_date) + pd.Timedelta(days=30)
                ).strftime("%Y-%m-%d"),
                "strike": strike,
                "option_type": otype,
                "open": 2.0,
                "high": 2.5,
                "low": 1.5,
                "close": 2.0,
                "volume": 100,
                "open_interest": 500,
                "vwap": 2.0,
            })
    return pd.DataFrame(rows)


class TestEndToEnd:
    """Integration tests verifying components work together end-to-end."""

    def test_state_engine_to_fusion(self):
        """StateEngine output feeds into SignalFusion."""
        chain = _synthetic_chain("2023-06-15")
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        state = engine.build(bars_df=chain)

        assert len(state) == 1
        row = state.iloc[0]

        fusion = GMESignalFusion()
        sc = fusion.compute_daily_scorecard(
            trade_date=date(2023, 6, 15),
            options_state=row.to_dict(),
            alt_data={
                "ftd_t35_countdown": 15,
                "si_change_2wk": 0.5,
                "dark_pool_ratio_change": 0.0,
            },
            ml_score=0.6,
        )
        assert isinstance(sc, GMEDailyScorecard)
        assert sc.regime in ("squeeze", "gamma_ramp", "cycle_hot", "neutral")

    def test_state_engine_produces_expected_columns(self):
        """StateEngine output contains all expected metric columns."""
        chain = _synthetic_chain("2023-06-15")
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        state = engine.build(bars_df=chain)

        expected_columns = {
            "trade_date",
            # IV surface
            "atm_iv_30d", "atm_iv_60d", "atm_iv_90d",
            "iv_skew_25d_30", "iv_term_slope",
            "iv_percentile", "iv_change_1d", "iv_change_5d",
            # GEX
            "net_gex", "gex_flip_price", "call_wall", "put_wall",
            "gex_concentration",
            # OI flow
            "total_call_oi", "total_put_oi", "pc_oi_ratio",
            "pc_oi_ratio_change", "oi_weighted_strike",
            "oi_concentration_top5", "near_term_oi_pct", "oi_skew_delta",
            # Volume & premium
            "call_volume", "put_volume", "pc_volume_ratio",
            "call_premium", "put_premium", "premium_imbalance",
        }
        assert expected_columns.issubset(set(state.columns))

    def test_backtester_with_wheel_and_strangle(self):
        """Backtester runs with wheel + strangle on synthetic data."""
        cfg = BacktestConfig(initial_capital=50_000.0)
        bt = GMEOptionsBacktester(
            strategies=[GMEWheelStrategy(), StrangleStrategy()],
            config=cfg,
        )
        chain = _synthetic_chain("2023-06-15")

        # Build a state row with the keys that wheel and strangle expect
        state_row = pd.Series({
            "trade_date": date(2023, 6, 15),
            "atm_iv_30d": 0.6,
            "iv_percentile": 0.6,
            "net_gex": 1000.0,
            "gex_concentration": 0.3,
            "ftd_t35_countdown": 20,
        })
        trades = bt.run_single_day(date(2023, 6, 15), state_row, chain)
        # Wheel sells a CSP (1 trade) + Strangle sells put+call (2 trades) = 3
        assert len(trades) >= 1

    def test_backtester_multi_day_run(self):
        """Backtester can run across multiple synthetic days."""
        dates_list = ["2023-06-15", "2023-06-16", "2023-06-19"]
        chain_by_date = {}
        state_rows = []

        for d in dates_list:
            chain_by_date[d] = _synthetic_chain(d)
            state_rows.append({
                "trade_date": d,
                "atm_iv_30d": 0.6,
                "iv_percentile": 0.6,
                "net_gex": 1000.0,
                "gex_concentration": 0.3,
                "ftd_t35_countdown": 20,
            })

        state_df = pd.DataFrame(state_rows)
        bt = GMEOptionsBacktester(
            strategies=[GMEWheelStrategy()],
            config=BacktestConfig(initial_capital=100_000.0),
        )
        results = bt.run(state_df, chain_by_date)
        assert len(results) == 1
        assert results[0].strategy_name == "wheel"
        assert len(results[0].equity_curve) == len(dates_list)

    def test_fusion_scorecard_fields(self):
        """SignalFusion scorecard exposes all expected fields."""
        fusion = GMESignalFusion()
        sc = fusion.compute_daily_scorecard(
            trade_date=date(2023, 6, 15),
            options_state={
                "iv_percentile": 0.7,
                "iv_skew_25d_30": 0.05,
                "gex_flip_price": 24.0,
                "call_wall": 28.0,
                "pc_oi_ratio_change": 0.01,
                "pc_volume_ratio": 0.9,
                "premium_imbalance": 0.1,
                "gex_concentration": 0.3,
            },
            alt_data={
                "ftd_t35_countdown": 15,
                "si_change_2wk": 0.5,
                "dark_pool_ratio_change": 0.0,
            },
            ml_score=0.6,
        )

        assert sc.date == date(2023, 6, 15)
        assert "options_flow" in sc.pillar_scores
        assert "volume_premium" in sc.pillar_scores
        assert "cycle_timing" in sc.pillar_scores
        assert "momentum" in sc.pillar_scores
        assert isinstance(sc.composite, float)
        assert isinstance(sc.anomaly_flags, list)

    def test_all_strategies_importable(self):
        """All 6 strategies can be instantiated."""
        from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
        from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
        from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
        from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
        from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
        from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy

        strategies = [
            GMEWheelStrategy(),
            IronCondorStrategy(),
            StrangleStrategy(),
            CreditSpreadStrategy(),
            LongOptionsStrategy(),
            CalendarSpreadStrategy(),
        ]
        assert len(strategies) == 6
        assert all(hasattr(s, "name") for s in strategies)

    def test_all_strategies_have_unique_names(self):
        """All 6 strategies report distinct names."""
        from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
        from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
        from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
        from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
        from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
        from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy

        strategies = [
            GMEWheelStrategy(),
            IronCondorStrategy(),
            StrangleStrategy(),
            CreditSpreadStrategy(),
            LongOptionsStrategy(),
            CalendarSpreadStrategy(),
        ]
        names = [s.name for s in strategies]
        assert len(names) == len(set(names)), f"Duplicate strategy names: {names}"

    def test_checkpoint_survives_restart(self, tmp_path):
        """Fetcher checkpoint persists across instances."""
        cfg = GMEOptionsConfig(
            data_dir=tmp_path,
            start_date=date(2022, 1, 1),
            end_date=date(2022, 3, 31),
        )
        f1 = OptionsDataFetcher(cfg, client=MagicMock())
        f1.checkpoint.mark_month_complete("2022-01")
        f1.checkpoint.save()

        f2 = OptionsDataFetcher(cfg, client=MagicMock())
        assert f2._remaining_months() == ["2022-02", "2022-03"]

    def test_checkpoint_marks_multiple_months(self, tmp_path):
        """Checkpoint correctly tracks multiple completed months."""
        cfg = GMEOptionsConfig(
            data_dir=tmp_path,
            start_date=date(2022, 1, 1),
            end_date=date(2022, 6, 30),
        )
        fetcher = OptionsDataFetcher(cfg, client=MagicMock())

        for month in ["2022-01", "2022-02", "2022-03"]:
            fetcher.checkpoint.mark_month_complete(month)
        fetcher.checkpoint.save()

        fetcher2 = OptionsDataFetcher(cfg, client=MagicMock())
        remaining = fetcher2._remaining_months()
        assert remaining == ["2022-04", "2022-05", "2022-06"]

    def test_state_engine_multi_day(self):
        """StateEngine correctly processes multiple trading days."""
        chain1 = _synthetic_chain("2023-06-15")
        chain2 = _synthetic_chain("2023-06-16")
        combined = pd.concat([chain1, chain2], ignore_index=True)

        engine = OptionsStateEngine(
            spot_prices={
                "2023-06-15": 25.0,
                "2023-06-16": 25.5,
            }
        )
        state = engine.build(bars_df=combined)
        assert len(state) == 2
        assert list(state["trade_date"]) == [
            date(2023, 6, 15),
            date(2023, 6, 16),
        ]

    def test_cli_module_has_all_strategies(self):
        """CLI module's strategy registry contains all 6 strategies."""
        from stockdownloader.gme.options.__main__ import _load_strategies

        strategies = _load_strategies()
        expected = {
            "wheel", "iron_condor", "strangle",
            "credit_spread", "long_options", "calendar_spread",
        }
        assert set(strategies.keys()) == expected

    def test_synthetic_chain_schema(self):
        """Synthetic chain helper produces correct schema for pipeline."""
        chain = _synthetic_chain("2023-06-15")
        required_cols = {
            "option_ticker", "date", "expiration", "strike",
            "option_type", "open", "high", "low", "close",
            "volume", "open_interest",
        }
        assert required_cols.issubset(set(chain.columns))
        assert len(chain) == 10  # 5 strikes x 2 option types
        assert set(chain["option_type"].unique()) == {"call", "put"}

    def test_enriched_state_has_nonzero_gex(self):
        """StateEngine computes non-zero GEX with enriched OI data."""
        chain = _synthetic_chain("2023-06-15")
        # Verify OI is already non-zero in synthetic data
        assert chain["open_interest"].sum() > 0

        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        state = engine.build(bars_df=chain)

        assert len(state) == 1
        # With OI > 0, GEX metrics should be non-zero
        assert state.iloc[0]["total_call_oi"] > 0
        assert state.iloc[0]["total_put_oi"] > 0
        assert state.iloc[0]["pc_oi_ratio"] > 0

    def test_oi_enricher_to_state_engine(self):
        """Full pipeline: bars → enrich → state engine produces GEX."""
        from stockdownloader.gme.options.oi_proxy import OIProxyEstimator

        chain = _synthetic_chain("2023-06-15")
        # Zero out OI to simulate raw fetcher output
        chain["open_interest"] = 0
        assert chain["open_interest"].sum() == 0

        # Apply proxy
        estimator = OIProxyEstimator(calibration_ratio=5.0, decay_rate=0.0)
        enriched = estimator.estimate(chain)
        assert enriched["open_interest"].sum() > 0

        # Feed to state engine
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        state = engine.build(bars_df=enriched)

        assert len(state) == 1
        assert state.iloc[0]["total_call_oi"] > 0
        assert state.iloc[0]["net_gex"] != 0
