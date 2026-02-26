"""Tests for GMESignalFusion and GMEDailyScorecard."""
from __future__ import annotations

import pytest
from datetime import date

from stockdownloader.gme.options.scorecard import GMEDailyScorecard
from stockdownloader.gme.options.signal_fusion import GMESignalFusion


class TestGMEDailyScorecard:
    def test_fields(self):
        sc = GMEDailyScorecard(
            date=date(2023, 6, 15),
            pillar_scores={"options_flow": 1.2, "volume_premium": 0.5,
                           "cycle_timing": -0.3, "momentum": 0.8},
            composite=0.55,
            regime="neutral",
            anomaly_flags=[],
        )
        assert sc.composite == 0.55
        assert sc.regime == "neutral"

    def test_has_anomalies(self):
        sc = GMEDailyScorecard(
            date=date(2023, 6, 15),
            pillar_scores={"options_flow": 2.5, "volume_premium": 0.1,
                           "cycle_timing": 0.0, "momentum": 0.0},
            composite=0.65,
            regime="neutral",
            anomaly_flags=["IV skew at 98th percentile"],
        )
        assert len(sc.anomaly_flags) == 1


class TestGMESignalFusion:
    def test_compute_pillar_scores(self):
        fusion = GMESignalFusion()
        options_state = {
            "iv_percentile": 0.8, "iv_skew_25d_30": 0.15,
            "gex_flip_price": 25.0, "call_wall": 30.0,
            "pc_oi_ratio_change": 0.05,
            "pc_volume_ratio": 1.2, "premium_imbalance": -0.3,
        }
        pillars = fusion._score_options(options_state)
        assert isinstance(pillars, float)

    def test_detect_squeeze_regime(self):
        fusion = GMESignalFusion()
        assert fusion._detect_regime(
            options={"iv_percentile": 0.15, "gex_concentration": 0.8},
            alt={"dark_pool_ratio_change": 0.0, "ftd_t35_countdown": 20,
                 "si_change_2wk": 0.0},
            composite=0.1,
        ) == "squeeze"

    def test_detect_cycle_hot_regime(self):
        fusion = GMESignalFusion()
        assert fusion._detect_regime(
            options={"iv_percentile": 0.5, "gex_concentration": 0.3},
            alt={"dark_pool_ratio_change": 0.15,
                 "ftd_t35_countdown": 3, "si_change_2wk": 5.0},
            composite=0.5,
        ) == "cycle_hot"

    def test_detect_neutral_regime(self):
        fusion = GMESignalFusion()
        assert fusion._detect_regime(
            options={"iv_percentile": 0.5, "gex_concentration": 0.3},
            alt={"dark_pool_ratio_change": 0.0,
                 "ftd_t35_countdown": 20, "si_change_2wk": 0.5},
            composite=0.0,
        ) == "neutral"

    def test_flag_anomalies(self):
        fusion = GMESignalFusion()
        pillars = {
            "options_flow": 2.5,
            "volume_premium": 0.5,
            "cycle_timing": 0.0,
            "momentum": 0.0,
        }
        flags = fusion._flag_anomalies(pillars)
        assert len(flags) >= 1
        assert "options_flow" in flags[0].lower()

    def test_compute_daily_scorecard(self):
        fusion = GMESignalFusion()
        sc = fusion.compute_daily_scorecard(
            trade_date=date(2023, 6, 15),
            options_state={
                "iv_percentile": 0.6, "iv_skew_25d_30": 0.1,
                "gex_flip_price": 25.0, "call_wall": 30.0,
                "pc_oi_ratio_change": 0.02,
                "pc_volume_ratio": 1.1, "premium_imbalance": 0.1,
                "gex_concentration": 0.3, "dark_pool_ratio_change": 0.0,
            },
            alt_data={
                "ftd_t35_countdown": 15, "si_change_2wk": 0.5,
                "dark_pool_ratio": 0.4, "dark_pool_ratio_change": 0.0,
                "regsho_proximity": 5.0,
            },
            ml_score=0.6,
        )
        assert isinstance(sc, GMEDailyScorecard)
        assert sc.regime in ("squeeze", "gamma_ramp", "cycle_hot", "neutral")
        assert -10.0 < sc.composite < 10.0
