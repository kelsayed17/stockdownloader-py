"""Tests for OI proxy estimation and enrichment."""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import pytest
from datetime import date
from pathlib import Path

from stockdownloader.gme.options.oi_proxy import OIProxyEstimator, OIEnricher
from stockdownloader.gme.options.config import GMEOptionsConfig


class TestOIProxyEstimator:
    def test_calibrate_computes_ratio(self):
        """calibrate() learns OI/cumulative-volume ratio from snapshot."""
        snapshot_df = pd.DataFrame([
            {"option_ticker": "O:GME260227C00025000", "open_interest": 100, "volume": 50},
            {"option_ticker": "O:GME260227P00025000", "open_interest": 200, "volume": 80},
        ])
        # Simulate bars with cumulative volume
        bars_df = pd.DataFrame([
            {"option_ticker": "O:GME260227C00025000", "volume": 200, "date": "2026-02-26"},
            {"option_ticker": "O:GME260227P00025000", "volume": 400, "date": "2026-02-26"},
        ])

        estimator = OIProxyEstimator()
        estimator.calibrate(snapshot_df, bars_df)

        # ratio = mean(100/200, 200/400) = mean(0.5, 0.5) = 0.5
        assert abs(estimator.calibration_ratio - 0.5) < 0.01

    def test_calibrate_uses_default_when_no_data(self):
        """calibrate() uses default ratio when no matching data."""
        estimator = OIProxyEstimator()
        estimator.calibrate(pd.DataFrame(), pd.DataFrame())
        assert estimator.calibration_ratio == estimator.DEFAULT_RATIO

    def test_estimate_produces_nonzero_oi(self):
        """estimate() fills open_interest with proxy values."""
        bars_df = pd.DataFrame([
            {"option_ticker": "A", "date": "2023-01-03", "volume": 100, "open_interest": 0},
            {"option_ticker": "A", "date": "2023-01-04", "volume": 150, "open_interest": 0},
            {"option_ticker": "A", "date": "2023-01-05", "volume": 200, "open_interest": 0},
        ])
        estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.0)
        result = estimator.estimate(bars_df)

        # With decay=0, OI = cumulative_volume * ratio
        # Day 1: cumvol=100, oi=50
        # Day 2: cumvol=250, oi=125
        # Day 3: cumvol=450, oi=225
        assert result.iloc[0]["open_interest"] == 50
        assert result.iloc[1]["open_interest"] == 125
        assert result.iloc[2]["open_interest"] == 225

    def test_estimate_applies_decay(self):
        """Exponential decay reduces OI for older volume."""
        bars_df = pd.DataFrame([
            {"option_ticker": "A", "date": "2023-01-03", "volume": 1000, "open_interest": 0},
            {"option_ticker": "A", "date": "2023-01-04", "volume": 0, "open_interest": 0},
            {"option_ticker": "A", "date": "2023-01-05", "volume": 0, "open_interest": 0},
        ])
        estimator = OIProxyEstimator(calibration_ratio=1.0, decay_rate=0.1)
        result = estimator.estimate(bars_df)

        oi_day1 = result.iloc[0]["open_interest"]
        oi_day3 = result.iloc[2]["open_interest"]

        # OI should decrease over time due to decay
        assert oi_day3 < oi_day1
        assert oi_day3 > 0

    def test_estimate_handles_multiple_contracts(self):
        """estimate() computes OI independently per contract."""
        bars_df = pd.DataFrame([
            {"option_ticker": "A", "date": "2023-01-03", "volume": 100, "open_interest": 0},
            {"option_ticker": "B", "date": "2023-01-03", "volume": 500, "open_interest": 0},
        ])
        estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.0)
        result = estimator.estimate(bars_df)

        a_oi = result[result["option_ticker"] == "A"].iloc[0]["open_interest"]
        b_oi = result[result["option_ticker"] == "B"].iloc[0]["open_interest"]
        assert a_oi == 50
        assert b_oi == 250

    def test_save_and_load_calibration(self, tmp_path):
        """Calibration persists to JSON and reloads."""
        estimator = OIProxyEstimator(calibration_ratio=0.42)
        estimator.save_calibration(tmp_path / "cal.json")

        loaded = OIProxyEstimator.load_calibration(tmp_path / "cal.json")
        assert abs(loaded.calibration_ratio - 0.42) < 0.001
