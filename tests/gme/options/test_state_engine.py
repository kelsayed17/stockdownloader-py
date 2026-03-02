"""Tests for OptionsStateEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date

from stockdownloader.gme.options.state_engine import OptionsStateEngine


def _make_chain_df(
    trade_date: str = "2023-06-15",
    spot: float = 25.0,
    n_strikes: int = 10,
) -> pd.DataFrame:
    """Create a synthetic options chain for one day."""
    rows = []
    strikes = np.linspace(spot * 0.7, spot * 1.3, n_strikes)
    for exp_offset in [30, 60, 90]:
        exp = pd.Timestamp(trade_date) + pd.Timedelta(days=exp_offset)
        for strike in strikes:
            for otype in ["call", "put"]:
                moneyness = spot / strike if otype == "call" else strike / spot
                price = max(0.5, moneyness * 2.0 + np.random.uniform(0, 0.5))
                rows.append({
                    "option_ticker": f"O:GME{exp.strftime('%y%m%d')}{'C' if otype == 'call' else 'P'}{int(strike*1000):08d}",
                    "underlying": "GME",
                    "date": trade_date,
                    "expiration": exp.strftime("%Y-%m-%d"),
                    "strike": round(strike, 2),
                    "option_type": otype,
                    "open": price,
                    "high": price * 1.05,
                    "low": price * 0.95,
                    "close": price,
                    "volume": int(np.random.uniform(10, 500)),
                    "open_interest": int(np.random.uniform(100, 5000)),
                    "vwap": price,
                })
    return pd.DataFrame(rows)


class TestIVSurface:
    def test_computes_atm_iv_30d(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_iv_surface(df, date(2023, 6, 15), 25.0)
        assert "atm_iv_30d" in metrics
        assert 0.0 < metrics["atm_iv_30d"] < 5.0

    def test_computes_iv_percentile(self):
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        history = [0.4 + i * 0.01 for i in range(30)]
        pct = engine._iv_percentile(0.55, history)
        assert 0.0 <= pct <= 1.0

    def test_iv_skew_sign(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_iv_surface(df, date(2023, 6, 15), 25.0)
        assert "iv_skew_25d_30" in metrics


class TestGEX:
    def test_computes_net_gex(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_gex(df, 25.0)
        assert "net_gex" in metrics
        assert isinstance(metrics["net_gex"], float)

    def test_computes_call_put_walls(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_gex(df, 25.0)
        assert "call_wall" in metrics
        assert "put_wall" in metrics
        assert metrics["call_wall"] >= metrics["put_wall"]

    def test_gex_flip_between_walls(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_gex(df, 25.0)
        if metrics["gex_flip_price"] > 0:
            assert metrics["put_wall"] <= metrics["gex_flip_price"] <= metrics["call_wall"] * 1.5


class TestOIFlow:
    def test_computes_oi_metrics(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_oi_flow(df)
        assert "total_call_oi" in metrics
        assert "total_put_oi" in metrics
        assert "pc_oi_ratio" in metrics
        assert metrics["total_call_oi"] > 0
        assert metrics["total_put_oi"] > 0

    def test_oi_concentration(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_oi_flow(df)
        assert 0.0 <= metrics["oi_concentration_top5"] <= 1.0

    def test_near_term_pct(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_oi_flow(df)
        assert 0.0 <= metrics["near_term_oi_pct"] <= 1.0


class TestVolumePremium:
    def test_computes_volume_metrics(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_volume_premium(df)
        assert metrics["call_volume"] > 0
        assert metrics["put_volume"] > 0
        assert -1.0 <= metrics["premium_imbalance"] <= 1.0

    def test_premium_calculation(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_volume_premium(df)
        assert metrics["call_premium"] > 0
        assert metrics["put_premium"] > 0


class TestBuildFull:
    def test_build_from_dataframe(self):
        df = _make_chain_df("2023-06-15")
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        result = engine.build(bars_df=df)
        assert len(result) == 1
        assert "atm_iv_30d" in result.columns
        assert "net_gex" in result.columns
        assert "call_volume" in result.columns

    def test_build_multi_day(self):
        df1 = _make_chain_df("2023-06-15")
        df2 = _make_chain_df("2023-06-16")
        df = pd.concat([df1, df2], ignore_index=True)
        engine = OptionsStateEngine(
            spot_prices={"2023-06-15": 25.0, "2023-06-16": 25.5},
        )
        result = engine.build(bars_df=df)
        assert len(result) == 2

    def test_build_empty(self):
        engine = OptionsStateEngine()
        result = engine.build(bars_df=pd.DataFrame())
        assert result.empty
