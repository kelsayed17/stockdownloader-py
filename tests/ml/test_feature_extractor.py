"""Tests for FeatureExtractor — ML feature vector extraction."""

from __future__ import annotations

import math
import random
from decimal import Decimal

import pytest

from stockdownloader.ml.feature_extractor import (
    FeatureExtractor,
    FeatureVector,
    _ALL_FEATURE_NAMES,
    _ENHANCED_NAMES,
    _GENERATOR_NAMES,
    _INDICATOR_NAMES,
    _REGIME_NAMES,
    _SIGNAL_NAMES,
    _safe_div,
)
from stockdownloader.model.price_data import PriceData


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_data(n: int = 300, seed: int = 42) -> list[PriceData]:
    """Generate synthetic price data with mild uptrend."""
    rng = random.Random(seed)
    data: list[PriceData] = []
    price = 100.0
    for i in range(n):
        price += (rng.random() - 0.48) * 2
        price = max(50, price)
        h = price + rng.random() * 2
        l = price - rng.random() * 2
        data.append(PriceData(
            date=f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(h, 2))),
            low=Decimal(str(round(l, 2))),
            close=Decimal(str(round(price, 2))),
            adj_close=Decimal(str(round(price, 2))),
            volume=int(1_000_000 + rng.random() * 5_000_000),
        ))
    return data


# Module-level data for reuse
DATA_300 = _make_data(300)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestFeatureExtractorNames:
    def test_feature_count(self) -> None:
        assert len(_ALL_FEATURE_NAMES) == 63

    def test_generator_count(self) -> None:
        assert len(_GENERATOR_NAMES) == 17

    def test_feature_names_unique(self) -> None:
        assert len(set(_ALL_FEATURE_NAMES)) == len(_ALL_FEATURE_NAMES)


class TestFeatureExtractorExtract:
    def test_extract_returns_feature_vector(self) -> None:
        ext = FeatureExtractor()
        fv = ext.extract(DATA_300, 250)
        assert isinstance(fv, FeatureVector)
        assert len(fv.values) == 63
        assert len(fv.names) == 63
        assert fv.bar_date != ""

    def test_all_values_finite(self) -> None:
        ext = FeatureExtractor()
        fv = ext.extract(DATA_300, 250)
        for i, val in enumerate(fv.values):
            assert isinstance(val, float), f"Feature {fv.names[i]} is not float"
            assert math.isfinite(val), f"Feature {fv.names[i]} = {val} is not finite"

    def test_feature_names_match(self) -> None:
        ext = FeatureExtractor()
        assert ext.FEATURE_NAMES == _ALL_FEATURE_NAMES
        fv = ext.extract(DATA_300, 250)
        assert fv.names == ext.FEATURE_NAMES

    def test_feature_count_property(self) -> None:
        ext = FeatureExtractor()
        assert ext.feature_count == 63

    def test_raises_on_insufficient_index(self) -> None:
        ext = FeatureExtractor()
        with pytest.raises(ValueError, match="Need index >= 201"):
            ext.extract(DATA_300, 100)

    def test_boundary_index(self) -> None:
        ext = FeatureExtractor()
        # Exactly at min index should work
        fv = ext.extract(DATA_300, 201)
        assert len(fv.values) == 63


class TestFeatureExtractorBatch:
    def test_batch_matches_individual(self) -> None:
        ext = FeatureExtractor()
        batch = ext.extract_batch(DATA_300, 250, 253)
        assert len(batch) == 3
        for i, offset in enumerate(range(250, 253)):
            individual = ext.extract(DATA_300, offset)
            assert batch[i].values == individual.values

    def test_batch_empty_range(self) -> None:
        ext = FeatureExtractor()
        batch = ext.extract_batch(DATA_300, 250, 250)
        assert batch == []


class TestEnhancedFeatures:
    def test_enhanced_feature_count(self) -> None:
        assert len(_ENHANCED_NAMES) == 15

    def test_total_feature_groups(self) -> None:
        total = len(_INDICATOR_NAMES) + len(_SIGNAL_NAMES) + len(_REGIME_NAMES) + len(_ENHANCED_NAMES)
        assert total == 63
        assert total == len(_ALL_FEATURE_NAMES)

    def test_enhanced_names_in_vector(self) -> None:
        ext = FeatureExtractor()
        fv = ext.extract(DATA_300, 250)
        for name in _ENHANCED_NAMES:
            assert name in fv.names, f"Enhanced feature '{name}' missing from vector"

    def test_lag_features_reasonable(self) -> None:
        ext = FeatureExtractor()
        fv = ext.extract(DATA_300, 250)
        idx_map = {n: i for i, n in enumerate(fv.names)}
        for name in ("return_1d", "return_3d", "return_5d", "return_10d"):
            val = fv.values[idx_map[name]]
            assert -0.5 < val < 0.5, f"{name} = {val} out of range"

    def test_calendar_features_bounded(self) -> None:
        ext = FeatureExtractor()
        fv = ext.extract(DATA_300, 250)
        idx_map = {n: i for i, n in enumerate(fv.names)}
        dow = fv.values[idx_map["day_of_week_norm"]]
        moy = fv.values[idx_map["month_of_year_norm"]]
        assert 0.0 <= dow <= 1.0, f"day_of_week_norm = {dow}"
        assert 0.0 <= moy <= 1.0, f"month_of_year_norm = {moy}"

    def test_volatility_features_nonnegative(self) -> None:
        ext = FeatureExtractor()
        fv = ext.extract(DATA_300, 250)
        idx_map = {n: i for i, n in enumerate(fv.names)}
        assert fv.values[idx_map["realized_vol_5d"]] >= 0.0
        assert fv.values[idx_map["vol_ratio_5d_20d"]] >= 0.0
        assert fv.values[idx_map["intraday_range_pct"]] >= 0.0

    def test_interaction_features_exist(self) -> None:
        ext = FeatureExtractor()
        fv = ext.extract(DATA_300, 250)
        idx_map = {n: i for i, n in enumerate(fv.names)}
        for name in ("rsi_x_adx", "bb_width_x_vol_ratio", "regime_conf_x_slope"):
            val = fv.values[idx_map[name]]
            assert math.isfinite(val), f"{name} = {val} not finite"


class TestSafeDiv:
    def test_normal(self) -> None:
        assert _safe_div(10.0, 2.0) == 5.0

    def test_zero_denominator(self) -> None:
        assert _safe_div(10.0, 0.0) == 0.0

    def test_inf_result(self) -> None:
        assert _safe_div(float("inf"), 1.0) == 0.0
